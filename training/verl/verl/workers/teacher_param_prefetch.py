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
"""Bounded, opt-in HtoD staging for one FSDP1 CPU-offloaded teacher handle.

FSDP's ordinary ``pre_unshard`` copies the CPU local shard to CUDA. The wrapper
below substitutes the staged CUDA shard at that exact point. FSDP still owns
the subsequent all-gather, forward, and reshard lifecycle.
"""

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._runtime_utils import _lazy_init


class TeacherHandlePrefetch:
    def __init__(self, model: FSDP, max_mb: int):
        if max_mb <= 0:
            raise ValueError("teacher_param_prefetch_max_mb must be positive")
        if not isinstance(model, FSDP):
            raise TypeError("teacher parameter prefetch currently requires FSDP1")
        _lazy_init(model, model)
        if not model._all_handles:
            raise ValueError("teacher FSDP model has no handles to prefetch")

        self.handle = model._all_handles[0]
        if not self.handle._offload_params:
            raise ValueError("teacher parameter prefetch requires FSDP1 CPU offload")
        if self.handle._uses_param_mixed_precision:
            raise ValueError("teacher parameter prefetch does not support mixed-precision shards")
        self.max_bytes = max_mb * 1024 * 1024
        self.bytes = self.handle.flat_param._local_shard.numel() * self.handle.flat_param._local_shard.element_size()
        if self.bytes > self.max_bytes:
            raise ValueError(f"first teacher FSDP shard is {self.bytes} bytes, above the {self.max_bytes} byte cap")
        if not self.handle.flat_param._local_shard.is_pinned():
            raise ValueError("teacher FSDP CPU shard must be pinned for asynchronous HtoD")

        self.stream = torch.cuda.Stream(device=model.compute_device)
        self.staged = torch.empty_like(self.handle.flat_param._local_shard, device=self.handle.device)
        self.pending = None
        self.copy_count = 0
        self.reuse_count = 0
        self.original_pre_unshard = self.handle.pre_unshard
        self.handle.pre_unshard = self.pre_unshard

    def start(self):
        if self.pending is not None:
            raise RuntimeError("previous teacher parameter prefetch has not been consumed")
        source = self.handle.flat_param._local_shard
        if self.handle.flat_param.device.type != "cpu":
            raise RuntimeError("teacher FSDP handle is not CPU-resident before prefetch")
        with torch.cuda.stream(self.stream), torch.cuda.nvtx.range("openmopd::io::h2d::teacher_prefetch"):
            staged = self.staged
            staged.copy_(source, non_blocking=True)
            staged.record_stream(self.stream)
            ready = torch.cuda.Event()
            ready.record(self.stream)
        self.pending = (staged, ready)
        self.copy_count += 1

    def pre_unshard(self):
        if self.pending is None:
            return self.original_pre_unshard()
        staged, ready = self.pending
        # FSDP calls this on its pre-unshard stream. Returning True makes its
        # all-gather stream wait for this stream, preserving the original order.
        torch.cuda.current_stream(device=staged.device).wait_event(ready)
        self.handle.flat_param.data = staged
        self.pending = None
        self.reuse_count += 1
        return True

    def assert_consumed(self, previous_reuse_count: int):
        if self.pending is not None or self.reuse_count != previous_reuse_count + 1:
            raise RuntimeError("teacher forward did not reuse the prefetched FSDP shard")
