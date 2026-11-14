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

"""Synchronous multi-rank AdamW with named FP32 masters/moments.

Independent of CPUAdamPipeline and its single-GPU subprocess/slot allocation.
Only the optimizer owner writes; all ranks read after the update barrier.
"""
import os
import time

import torch
import torch.distributed as dist

from .cpu_state_store import CPUStateStore, distributed_tensor


class SharedCPUAdamW(torch.optim.Optimizer):
    _persistent_shared_cpu_state = True
    gradient_sync = "cpu_reduce"

    def __init__(self, optimizer, owner_rank, publisher, *, chunk_bytes=256 * 1024**2, reduce_mode=None):
        if type(chunk_bytes) is not int or chunk_bytes <= 0 or chunk_bytes % 4:
            raise ValueError("CPU reduce chunk_bytes must be a positive multiple of four")
        self.reduce_mode = reduce_mode or os.environ.get("OPENMOPD_CPU_REDUCE_MODE", "async")
        if self.reduce_mode not in ("serial", "double_buffer", "async"):
            raise ValueError("CPU reduce mode must be serial, double_buffer, or async")
        if optimizer.state:
            raise ValueError("SharedCPUAdamW must be constructed before the first update")
        super().__init__(optimizer.param_groups, optimizer.defaults)
        self.owner_rank = owner_rank
        self.publisher = publisher
        self.store = CPUStateStore.from_environment()
        self.pairs = []
        self.grad_buffers = []
        self.world_size = dist.get_world_size()
        self._prepared = False
        self.chunk_elements = chunk_bytes // 4
        self.workspace = distributed_tensor(self.store, "gradient/workspace", "rank_slots",
                                            (self.world_size, self.chunk_elements), torch.float32,
                                            owner_rank=owner_rank)
        self.workspaces = [self.workspace]
        self.staging = []
        self.copy_stream = None
        self.last_reduce_metrics = {}
        if self.reduce_mode != "serial":
            self.workspaces.append(distributed_tensor(
                self.store, "gradient/workspace", "rank_slots_1",
                (self.world_size, self.chunk_elements), torch.float32, owner_rank=owner_rank))
            self.staging = [torch.empty(self.chunk_elements, dtype=torch.float32, pin_memory=True) for _ in range(2)]
            devices = {p.device for group in self.param_groups for p in group["params"]}
            if len(devices) != 1 or next(iter(devices)).type != "cuda":
                raise ValueError("buffered CPU reduce requires parameters on one rank-local CUDA device")
            self.device = next(iter(devices))
            self.copy_stream = torch.cuda.Stream(device=self.device)
            self.copy_events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                                for _ in range(2)]
        groups = []
        for group_index, group in enumerate(self.param_groups):
            if group.get("amsgrad", False) or group.get("maximize", False):
                raise ValueError("shared CPU Adam supports ordinary AdamW only")
            masters = []
            for param_index, param in enumerate(group["params"]):
                name = f"group{group_index}/param{param_index}"
                def allocate(component, dtype, shape=param.shape, source=None):
                    return distributed_tensor(self.store, component, name, shape, dtype,
                                              source=source, owner_rank=owner_rank)
                host = allocate("student", param.dtype, source=param.detach())
                master = torch.nn.Parameter(allocate("optimizer/master", torch.float32, source=host),
                                            requires_grad=False)
                state = {
                    "step": allocate("optimizer/step", torch.float32, shape=()),
                    "exp_avg": allocate("optimizer/exp_avg", torch.float32),
                    "exp_avg_sq": allocate("optimizer/exp_avg_sq", torch.float32),
                    "cpu_master": master.detach(), "student": host,
                }
                self.state[param] = state
                self.pairs.append((param, master, host))
                masters.append(master)
                present = allocate("gradient/present", torch.bool, shape=(self.world_size,))
                reduced = allocate("gradient/reduced", torch.float32)
                self.grad_buffers.append((present, reduced))
            groups.append(dict(params=masters, **self._options(group)))
        # Pack parameters into fixed-capacity chunks; a chunk may cross parameter
        # boundaries and a large parameter may span multiple chunks.
        self.chunks = []
        segments, used = [], 0
        for index, (param, _, _) in enumerate(self.pairs):
            offset = 0
            while offset < param.numel():
                size = min(param.numel() - offset, self.chunk_elements - used)
                segments.append((index, offset, used, size))
                offset += size
                used += size
                if used == self.chunk_elements:
                    self.chunks.append(segments)
                    segments, used = [], 0
        if segments:
            self.chunks.append(segments)
        if dist.get_rank() == owner_rank:
            print(f"CPU_REDUCE_WORKSPACE chunk_bytes_per_rank={chunk_bytes} "
                  f"workspace_bytes={sum(w.numel() * 4 for w in self.workspaces)} "
                  f"mode={self.reduce_mode} pinned_bytes_per_rank={sum(w.numel() * 4 for w in self.staging)} "
                  f"reduced_gradient_bytes={sum(p.numel() for p, _, _ in self.pairs) * 4} "
                  f"chunks={len(self.chunks)}", flush=True)
        self.grad_norm = distributed_tensor(self.store, "gradient", "norm", (), torch.float32,
                                            owner_rank=owner_rank)
        self.cpu_optimizer = torch.optim.AdamW(groups, foreach=False)
        for param, master, _ in self.pairs:
            self.cpu_optimizer.state[master] = {
                key: self.state[param][key] for key in ("step", "exp_avg", "exp_avg_sq")
            }

    @staticmethod
    def _options(group):
        return {key: group[key] for key in ("lr", "betas", "eps", "weight_decay") if key in group}

    @torch.no_grad()
    def clip_grad_norm_(self, max_norm):
        """Complete transport/reduction before global clipping and owner AdamW.

        CUDA events protect staging reads. Ready/consumed barriers protect shared
        workspace reads/reuse. Neither staging nor workspace enters checkpoints.
        """
        started = time.perf_counter()
        self.last_reduce_metrics = dict(d2h_s=0.0, d2h_wait_s=0.0, staging_copy_s=0.0,
                                        cpu_reduce_s=0.0, barrier_s=0.0)
        rank = dist.get_rank()
        for (param, _, _), (present, _) in zip(self.pairs, self.grad_buffers, strict=True):
            present[rank] = param.grad is not None
        if self.reduce_mode == "serial":
            for segments in self.chunks:
                stamp = time.perf_counter()
                self._copy_chunk_to_cpu(segments, rank)
                self.last_reduce_metrics["d2h_s"] += time.perf_counter() - stamp
                self._transport_barrier()
                self._timed_reduce(segments, rank)
                self._transport_barrier()
        else:
            self._buffered_reduce(rank)
        self.last_reduce_metrics["transport_reduce_s"] = time.perf_counter() - started
        # Also synchronize presence flags for empty-parameter models.
        dist.barrier()
        if rank == self.owner_rank:
            for (_, master, _), (present, reduced) in zip(self.pairs, self.grad_buffers, strict=True):
                master.grad = reduced if present.any() else None
            norm = torch.nn.utils.clip_grad_norm_([master for _, master, _ in self.pairs], max_norm)
            self.grad_norm.copy_(norm)
        dist.barrier()
        self._prepared = True
        self.last_reduce_metrics["clip_total_s"] = time.perf_counter() - started
        return self.grad_norm.clone()

    def _transport_barrier(self):
        stamp = time.perf_counter()
        dist.barrier()
        self.last_reduce_metrics["barrier_s"] += time.perf_counter() - stamp

    def _timed_reduce(self, segments, rank):
        if rank == self.owner_rank:
            stamp = time.perf_counter()
            self._reduce_chunk(segments)
            self.last_reduce_metrics["cpu_reduce_s"] += time.perf_counter() - stamp

    def _enqueue_chunk(self, segments, slot):
        """D2H only; gradients remain live and immutable until clip returns."""
        begin, done = self.copy_events[slot]
        with torch.cuda.stream(self.copy_stream):
            begin.record(self.copy_stream)
            for index, offset, start, size in segments:
                grad = self.pairs[index][0].grad
                destination = self.staging[slot][start:start + size]
                if grad is None:
                    destination.zero_()
                else:
                    destination.copy_(grad.detach().view(-1)[offset:offset + size],
                                      non_blocking=self.reduce_mode == "async")
            done.record(self.copy_stream)

    def _buffered_reduce(self, rank):
        # Backward may have been submitted on a non-default stream. Establish
        # the dependency once, without modifying actor backward or synchronizing
        # the entire device. References in param.grad keep GPU sources alive.
        self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
        try:
            if self.chunks:
                self._enqueue_chunk(self.chunks[0], 0)
            for index, segments in enumerate(self.chunks):
                slot = index % 2
                begin, done = self.copy_events[slot]
                stamp = time.perf_counter()
                done.synchronize()  # CPU must not read staging before D2H completion.
                self.last_reduce_metrics["d2h_wait_s"] += time.perf_counter() - stamp
                self.last_reduce_metrics["d2h_s"] += begin.elapsed_time(done) / 1000
                self.workspace = self.workspaces[slot]
                used = segments[-1][2] + segments[-1][3]
                stamp = time.perf_counter()
                self.workspace[rank, :used].copy_(self.staging[slot][:used])
                self.last_reduce_metrics["staging_copy_s"] += time.perf_counter() - stamp
                self._transport_barrier()  # All ranks have published this chunk.
                # Previous iteration's consumed barrier freed the other slot.
                # Async mode overlaps this transfer with owner CPU reduction.
                if index + 1 < len(self.chunks):
                    self._enqueue_chunk(self.chunks[index + 1], 1 - slot)
                self._timed_reduce(segments, rank)
                self._transport_barrier()  # Owner has finished reading this workspace.
        finally:
            # Also drain on local exceptions: never allow pending writes into
            # staging after unwinding, zero_grad, or a subsequent optimizer call.
            self.copy_stream.synchronize()

    def _copy_chunk_to_cpu(self, segments, rank):
        """Blocking transport interface; workspace ownership is rank-local."""
        for index, offset, start, size in segments:
            grad = self.pairs[index][0].grad
            destination = self.workspace[rank, start:start + size]
            if grad is None:
                destination.zero_()
            else:
                # FSDP NO_SHARD gradients are contiguous; fail instead of making
                # an unbounded full-size contiguous copy for unsupported layouts.
                destination.copy_(grad.detach().view(-1)[offset:offset + size], non_blocking=False)

    def _reduce_chunk(self, segments):
        """Owner-only CPU mean directly into persistent reduced-gradient views."""
        for index, offset, start, size in segments:
            destination = self.grad_buffers[index][1].view(-1)[offset:offset + size]
            torch.sum(self.workspace[:, start:start + size], dim=0, out=destination)
            destination.div_(self.world_size)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("SharedCPUAdamW does not support closures")
        if not self._prepared:
            self.clip_grad_norm_(float("inf"))
        if not torch.isfinite(self.grad_norm):
            self.zero_grad()
            return
        if dist.get_rank() == self.owner_rank:
            for source, target in zip(self.param_groups, self.cpu_optimizer.param_groups, strict=True):
                target.update(self._options(source))
            self.cpu_optimizer.step()
            for _, master, host in self.pairs:
                host.copy_(master)
                master.grad = None
        dist.barrier()  # Completed shared weights are visible before each rank's H2D.
        for param, _, host in self.pairs:
            param.copy_(host)
        self.publisher.publish((param for param, _, _ in self.pairs), broadcast_parameters=False)
        self._prepared = False

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        for _, master, _ in self.pairs:
            master.grad = None
        self._prepared = False

    def state_dict(self):
        state = super().state_dict()
        state["published_version"] = self.publisher.version
        return state

    def load_state_dict(self, state_dict):
        # Never let Optimizer.load_state_dict replace named mappings or cast FP32
        # moments to BF16. Restore into existing shared views in-place.
        for group, saved in zip(self.param_groups, state_dict["param_groups"], strict=True):
            for param, index in zip(group["params"], saved["params"], strict=True):
                if dist.get_rank() == self.owner_rank:
                    for key, destination in self.state[param].items():
                        destination.copy_(state_dict["state"][index][key])
            group.update({key: value for key, value in saved.items() if key != "params"})
        dist.barrier()
        with torch.no_grad():
            for param, _, host in self.pairs:
                param.copy_(host)

        self.publisher.version = int(state_dict.get("published_version", max(
            (state["step"].item() for state in self.state.values()), default=0)))
        self.publisher.assert_version(param for param, _, _ in self.pairs)
