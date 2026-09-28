# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bounded, opt-in HtoD staging for selected FSDP1 CPU-offloaded teacher handles.

FSDP's ordinary ``pre_unshard`` copies the CPU local shard to CUDA. The wrapper
below substitutes the staged CUDA shard at that exact point. FSDP still owns
the subsequent all-gather, forward, and reshard lifecycle.
"""

from contextlib import contextmanager

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._runtime_utils import _lazy_init


class TeacherHandlePrefetch:
    def __init__(self, model: FSDP, max_mb: int, nvtx_name="openmopd::io::h2d::teacher_prefetch", num_handles: int = 1):
        if max_mb <= 0:
            raise ValueError("teacher_param_prefetch_max_mb must be positive")
        if not isinstance(model, FSDP):
            raise TypeError("teacher parameter prefetch currently requires FSDP1")
        _lazy_init(model, model)
        if not model._all_handles:
            raise ValueError("teacher FSDP model has no handles to prefetch")
        if not isinstance(num_handles, int) or not 1 <= num_handles <= len(model._all_handles):
            raise ValueError("prefetch num_handles must select a nonempty available prefix")
        self.num_handles = num_handles
        self.handles = list(model._all_handles[:num_handles])
        self.handle = self.handles[0]
        self.nvtx_name = nvtx_name
        self.max_bytes = max_mb * 1024 * 1024
        self.bytes = sum(h.flat_param._local_shard.numel() * h.flat_param._local_shard.element_size()
                         for h in self.handles)
        if self.bytes > self.max_bytes:
            raise ValueError(f"teacher staging requires {self.bytes} bytes, above the {self.max_bytes} byte cap")
        for handle in self.handles:
            if not handle._offload_params or handle._uses_param_mixed_precision:
                raise ValueError("teacher prefetch requires CPU offload without mixed-precision shards")
            if not handle.flat_param._local_shard.is_pinned():
                raise ValueError("teacher FSDP CPU shard must be pinned for asynchronous HtoD")
        self.stream = torch.cuda.Stream(device=model.compute_device)
        # Allocate everything before installing hooks, so allocation failures leave FSDP untouched.
        self.buffers = [torch.empty_like(h.flat_param._local_shard, device=h.device) for h in self.handles]
        self.staged = self.buffers[0]
        self._pending = {}
        self.last_use_done = None
        self.copy_count = 0
        self.reuse_count = 0
        self.original_pre_unshards = [h.pre_unshard for h in self.handles]
        self.original_pre_unshard = self.original_pre_unshards[0]
        for index, handle in enumerate(self.handles):
            handle.pre_unshard = lambda index=index: self.pre_unshard(index)

    @property
    def pending(self):
        return self._pending or None

    def start(self):
        if self.pending is not None:
            raise RuntimeError("previous teacher parameter prefetch has not been consumed")
        if any(h.flat_param.device.type != "cpu" for h in self.handles):
            raise RuntimeError("teacher FSDP handle is not CPU-resident before prefetch")
        with torch.cuda.stream(self.stream), torch.cuda.nvtx.range(self.nvtx_name):
            if self.last_use_done is not None:
                self.stream.wait_event(self.last_use_done)
            for index, (handle, staged) in enumerate(zip(self.handles, self.buffers, strict=True)):
                # Register before enqueue so exception cleanup also drains a partial copy.
                ready = torch.cuda.Event()
                self._pending[index] = ready
                staged.copy_(handle.flat_param._local_shard, non_blocking=True)
                staged.record_stream(self.stream)
                ready.record(self.stream)
                self.copy_count += 1

    def pre_unshard(self, index=0):
        if index not in self._pending:
            return self.original_pre_unshards[index]()
        staged = self.buffers[index]
        torch.cuda.current_stream(device=staged.device).wait_event(self._pending[index])
        self.handles[index].flat_param.data = staged
        del self._pending[index]
        self.reuse_count += 1
        return True

    def record_use_done(self):
        """Order the next staging write after this forward's GPU reads, across streams."""
        if self.pending is not None:
            raise RuntimeError("cannot finish a teacher use with an unconsumed prefetch")
        self.last_use_done = torch.cuda.Event()
        self.last_use_done.record(torch.cuda.current_stream(self.staged.device))

    def discard(self):
        """Drain unconsumed copies when the enclosing scoring operation fails."""
        if self.pending is not None:
            self.stream.synchronize()
            self._pending.clear()

    def assert_consumed(self, previous_reuse_count: int):
        if self.pending is not None or self.reuse_count != previous_reuse_count + len(self.handles):
            raise RuntimeError("teacher forward did not reuse every prefetched FSDP shard")


@contextmanager
def after_teacher_parameter_copies(model, callback=None):
    """Trigger once after all FSDP handles enqueue their CPU-offloaded shards.

    The callback runs on the common pre-unshard stream, so its recorded event
    covers every parameter copy (and waits for any separately staged shard).
    Wrappers are scoped to this forward and restored even on failure.
    """
    if callback is None:
        yield
        return
    if not isinstance(model, FSDP):
        raise TypeError("Math HtoD completion trigger requires FSDP1")
    _lazy_init(model, model)
    handles = list(model._all_handles)
    if not handles or any(not h._offload_params for h in handles):
        raise ValueError("Math HtoD completion trigger requires CPU-offloaded handles")
    originals = [h.pre_unshard for h in handles]
    visited = set()
    copy_stream = None

    def pre_unshard(index):
        nonlocal copy_stream
        copied = originals[index]()
        if not copied or index in visited:
            raise RuntimeError("Math HtoD trigger requires one parameter copy per handle")
        stream = torch.cuda.current_stream(model.compute_device)
        if copy_stream is None:
            copy_stream = stream
        elif stream != copy_stream:
            raise RuntimeError("Math parameter copies must share a pre-unshard stream")
        visited.add(index)
        if len(visited) == len(handles):
            callback()
        return copied

    try:
        for index, handle in enumerate(handles):
            handle.pre_unshard = lambda index=index: pre_unshard(index)
        yield
        if len(visited) != len(handles):
            raise RuntimeError("Math forward did not visit every parameter handle")
    finally:
        for handle, original in zip(handles, originals, strict=True):
            handle.pre_unshard = original
