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
"""One teacher's worth of CUDA slots, handed to the next teacher layer by layer.

Sequential mode copies the active teacher from shared CPU mappings before its
forward, with no prefetch or private pinned replica. The asynchronous single-GPU
mode is retained below.

Frozen FSDP1 teachers execute in a fixed ring. After each FSDP wrapper returns,
its compute stream records the last use of that slot; a copy stream waits for
that event before overwriting it with the next teacher's pinned CPU shard.
FSDP's pre-unshard stream waits for the copy before exposing the slot as views.
The root slot (including tied embeddings/head) is released only at model exit.
"""

import threading
from dataclasses import dataclass
from functools import partial

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._flat_param import HandleTrainingState
from torch.distributed.fsdp._runtime_utils import _lazy_init


@dataclass
class _Slot:
    tensor: torch.Tensor
    owner: int | None = None
    ready: torch.cuda.Event | None = None
    in_use: bool = False


class TeacherLayerPipeline:
    """Inference-only NO_SHARD with identical rank-local FSDP wrapping.

    sequential=True uses blocking copies directly from shared CPU mappings.
    The default retains the single-GPU pinned-memory prefetch ring.

    Each wrapper must execute exactly once per forward, under no_grad. Parameters
    must remain immutable and may only be used inside their owning wrapper (root
    parameters can be used throughout the model). Models must not return parameter
    aliases or launch parameter-consuming work on unsynchronized side streams.
    """

    def __init__(self, models: list[FSDP], names: list[str], max_mb: int, slot_tensors=None, *, sequential=False):
        if len(models) < 2 or len(models) != len(names) or len(set(names)) != len(names):
            raise ValueError("teacher layer pipeline requires at least two distinct named teachers")
        if max_mb <= 0:
            raise ValueError("teacher_layer_pipeline_max_mb must be positive")
        self.sequential = sequential
        self.modules = []
        signature = None
        for model in models:
            if not isinstance(model, FSDP):
                raise ValueError("teacher layer pipeline requires FSDP1")
            _lazy_init(model, model)
            modules = {
                name: module
                for name, module in model.named_modules()
                if isinstance(module, FSDP) and module._handle is not None
            }
            if not modules or model.training:
                raise ValueError("teacher layer pipeline requires eval models with FSDP parameters")
            layout = []
            for name, module in modules.items():
                handle = module._handle
                source = handle.flat_param._local_shard
                if (
                    (handle.world_size != 1 and not sequential)
                    or handle.uses_sharded_strategy
                    or not handle._offload_params
                    or handle._use_orig_params
                    or handle._uses_param_mixed_precision
                    or module.forward_prefetch
                ):
                    raise ValueError(
                        "teacher layer pipeline requires CPU-offloaded NO_SHARD, orig_params=false, "
                        "no parameter mixed precision, and forward_prefetch=false"
                    )
                if (
                    source.device.type != "cpu"
                    or handle.flat_param.requires_grad
                    or (not sequential and not source.is_pinned())
                    or handle.flat_param.device.type != "cpu"
                    or handle._training_state != HandleTrainingState.IDLE
                ):
                    raise ValueError("teacher layer pipeline requires idle CPU shards (pinned for asynchronous mode)")
                shared = getattr(handle.flat_param, "_openmopd_shared_cpu", None)
                if shared is not None and shared.data_ptr() != source.data_ptr():
                    raise ValueError("teacher slot source must retain its shared CPU mapping")
                if "pre_unshard" in handle.__dict__:
                    raise ValueError("teacher layer pipeline cannot share a handle with another prefetcher")
                layout.append(
                    (
                        name,
                        handle.device,
                        source.dtype,
                        source.shape,
                        tuple(handle.flat_param._fqns),
                        tuple(handle.flat_param._shapes),
                    )
                )
            if signature is not None and layout != signature:
                raise ValueError("teacher layer pipeline requires identical FSDP parameter layouts and devices")
            signature = layout
            self.modules.append(modules)

        self.bytes = sum(m._handle.flat_param._local_shard.nbytes for m in self.modules[0].values())
        if self.bytes > max_mb * 1024 * 1024:
            raise ValueError(f"teacher slot pool is {self.bytes} bytes, above the {max_mb} MiB cap")
        self.names = names
        self.device = models[0].compute_device
        self.stream = None if sequential else torch.cuda.Stream(device=self.device)
        self.slots = {
            name: _Slot(
                slot_tensors[name] if slot_tensors is not None
                else torch.empty_like(module._handle.flat_param._local_shard, device=self.device)
            )
            for name, module in self.modules[0].items()
        }
        # Respect the allocator history of storage allocated on the caller's stream.
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
        self.copy_counts = [0] * len(models)
        self.use_counts = [0] * len(models)
        self.expected = 0
        self.active = None
        self.seen = set()
        self.released = set()
        self.broken = False
        self.closed = False
        self.lock = threading.Lock()
        self.hooks = []
        self.originals = []
        for index, (model, modules) in enumerate(zip(models, self.modules, strict=True)):
            self.hooks.append(model.register_forward_pre_hook(partial(self._begin, index), prepend=True))
            for name, module in modules.items():
                handle = module._handle
                self.originals.append((handle, handle.pre_unshard))
                handle.pre_unshard = partial(self._acquire, index, name)
                self.hooks.append(module.register_forward_hook(partial(self._release, index, name)))
            # Register after the root slot's release hook. Runs even if forward failed.
            self.hooks.append(model.register_forward_hook(partial(self._finish, index), always_call=True))

    def _check_live(self):
        if self.closed or self.broken:
            raise RuntimeError("teacher layer pipeline is closed or failed; discard the failed models")

    def _stage(self, index, name, last_use=None):
        slot = self.slots[name]
        if slot.in_use:
            raise RuntimeError(f"attempt to overwrite an active teacher slot: {name}")
        flat = self.modules[index][name]._handle.flat_param
        source = flat._local_shard
        if hasattr(flat, "_openmopd_shared_cpu"):
            from verl.workers.multi_gpu.teacher_store import SharedTeacherStore

            source = SharedTeacherStore.h2d_source(flat)
        if self.sequential:
            if last_use is not None:
                last_use.synchronize()
            # Pageable mmap stays the actual H2D source; no private pinned replica.
            slot.tensor.copy_(source, non_blocking=False)
            slot.ready = torch.cuda.Event()
            slot.ready.record(torch.cuda.current_stream(self.device))
            slot.owner = index
            self.copy_counts[index] += 1
            return
        label = name or "root"
        with (
            torch.cuda.stream(self.stream),
            torch.cuda.nvtx.range(f"openmopd::io::h2d::teacher_layer_prefetch::{self.names[index]}::{label}"),
        ):
            if last_use is not None:
                self.stream.wait_event(last_use)
            slot.tensor.copy_(source, non_blocking=True)
            slot.tensor.record_stream(self.stream)
            slot.ready = torch.cuda.Event()
            slot.ready.record(self.stream)
        slot.owner = index
        self.copy_counts[index] += 1

    def prefetch_first(self):
        """Prime the cold ring during student scoring; later cycles are already primed."""
        self._check_live()
        if self.sequential:
            return  # No copying during student scoring.
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("cannot prime teacher slots during a teacher forward")
        try:
            if self.expected != 0:
                raise RuntimeError("previous teacher scoring cycle has not completed")
            self._prime()
        finally:
            self.lock.release()

    def _prime(self):
        for name, slot in self.slots.items():
            if slot.owner is None:
                self._stage(0, name)

    def _begin(self, index, module, args):
        self._check_live()
        if torch.is_grad_enabled() or module.training:
            raise RuntimeError("teacher layer pipeline only supports eval forwards under no_grad")
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("teacher layer pipeline does not support concurrent or reentrant teachers")
        try:
            if index != self.expected:
                raise RuntimeError(f"expected teacher {self.names[self.expected]}, got {self.names[index]}")
            if self.sequential:
                for name in self.slots:
                    self._stage(index, name, self.slots[name].ready)
            elif index == 0:
                self._prime()
            self.active = (index, threading.get_ident())
            self.seen = set()
            self.released = set()
            torch.cuda.nvtx.range_push(f"openmopd::compute::teacher_layer_model_forward::{self.names[index]}")
        except BaseException:
            self.lock.release()
            raise

    def _acquire(self, index, name):
        self._check_live()
        if self.active != (index, threading.get_ident()) or name in self.seen:
            raise RuntimeError("teacher layers must execute exactly once within their model forward")
        slot = self.slots[name]
        if slot.in_use or slot.owner != index or slot.ready is None:
            raise RuntimeError(f"teacher slot not staged for {self.names[index]}: {name}")
        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(slot.ready)
        slot.tensor.record_stream(stream)
        self.modules[index][name]._handle.flat_param.data = slot.tensor
        slot.in_use = True
        self.seen.add(name)
        self.use_counts[index] += 1
        # FSDP now makes its unshard stream wait for this pre-unshard stream.
        return True

    def _release(self, index, name, module, args, output):
        handle = module._handle
        if handle._training_state != HandleTrainingState.IDLE or handle.flat_param.device.type != "cpu":
            raise RuntimeError("teacher slot released before FSDP reshard completed")
        slot = self.slots[name]
        if not slot.in_use or slot.owner != index:
            raise RuntimeError("teacher slot released without acquisition")
        # Python's forward has returned, but its CUDA kernels may still read weights.
        compute = torch.cuda.current_stream(self.device)
        slot.tensor.record_stream(compute)
        last_use = torch.cuda.Event()
        last_use.record(compute)
        handle._use_unsharded_views(as_params=False)
        slot.in_use = False
        self.released.add(name)
        if self.sequential:
            slot.ready = last_use
            slot.owner = None
        else:
            self._stage((index + 1) % len(self.names), name, last_use)

    def _finish(self, index, module, args, output):
        if self.active != (index, threading.get_ident()):
            return  # The pre-hook rejected this invocation before it acquired the ring.
        try:
            if output is None:
                self.broken = True  # Do not mask the original forward exception.
            elif self.released != set(self.slots):
                self.broken = True
                raise RuntimeError("teacher forward skipped FSDP layers; fixed layer execution is required")
            else:
                self.expected = (index + 1) % len(self.names)
        finally:
            torch.cuda.nvtx.range_pop()
            self.active = None
            self.lock.release()

    def close(self):
        """Drain CUDA work before restoring ordinary FSDP; failed models must be discarded."""
        if self.closed:
            return
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("cannot close teacher slots during a teacher forward")
        try:
            torch.cuda.synchronize(self.device)
            for hook in self.hooks:
                hook.remove()
            for handle, _ in self.originals:
                del handle.pre_unshard
            self.closed = True
            self.slots.clear()
        finally:
            self.lock.release()
