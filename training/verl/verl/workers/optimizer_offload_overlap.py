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
"""Defer optimizer D2H until rollout's synchronizing preparation has finished."""

from concurrent.futures import ThreadPoolExecutor

import torch


class OptimizerOffloadOverlap:
    """Event-ordered copies to reusable pinned memory during rollout.

    prepare() runs after the update; start() runs after rollout weights/KV wake.
    finish() is required before optimizer reuse, checkpointing or training mode.
    State stays on GPU until finish(), so partially written CPU tensors are not
    exposed through optimizer.state. GPU memory must fit both optimizer state
    and the live rollout KV cache; pinned CPU memory holds one state copy.
    """

    def __init__(self, device, chunk_mb: int = 32):
        if not isinstance(chunk_mb, int) or isinstance(chunk_mb, bool) or chunk_mb <= 0:
            raise ValueError("optimizer offload chunk_mb must be a positive integer")
        self.chunk_bytes = chunk_mb * 1024 * 1024
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device)
        self.buffers = {}
        self.entries = []
        self.ready = None
        self.future = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="optimizer-offload")

    @property
    def pending(self):
        return self.ready is not None

    @torch.no_grad()
    def prepare(self, optimizer):
        if self.pending:
            raise RuntimeError("previous optimizer offload must finish before another update")
        entries = []
        buffers = {}
        for param, state in optimizer.state.items():
            for key, value in state.items():
                if not isinstance(value, torch.Tensor) or value.device.type == "cpu":
                    continue
                if value.device != self.device or value.layout != torch.strided or not value.is_contiguous():
                    raise ValueError("optimizer overlap requires contiguous state on the configured CUDA device")
                identity = (param, key)
                buffer = self.buffers.get(identity)
                if buffer is None or buffer.shape != value.shape or buffer.dtype != value.dtype:
                    buffer = torch.empty_like(value, device="cpu", pin_memory=True)
                buffers[identity] = buffer
                entries.append((state, key, value, buffer))
        self.buffers = buffers
        self.entries = entries
        if entries:
            self.ready = torch.cuda.Event()
            self.ready.record(torch.cuda.current_stream(self.device))

    def start(self):
        if not self.pending or self.future is not None:
            return
        self.future = self.executor.submit(self._copy_chunks)

    @torch.no_grad()
    def _copy_chunks(self):
        # A whole-state enqueue blocks vLLM's sampled-token D2H behind gigabytes
        # of optimizer traffic even on different streams. Keep just one bounded
        # DMA chunk in flight; only this background thread waits between chunks.
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            with torch.cuda.nvtx.range("openmopd::io::d2h::optimizer_state"):
                try:
                    self.stream.wait_event(self.ready)
                    copied = torch.cuda.Event()
                    for _, _, source, destination in self.entries:
                        source_flat = source.view(-1)
                        destination_flat = destination.view(-1)
                        chunk_elements = max(1, self.chunk_bytes // source.element_size())
                        for offset in range(0, source.numel(), chunk_elements):
                            end = min(offset + chunk_elements, source.numel())
                            destination_flat[offset:end].copy_(source_flat[offset:end], non_blocking=True)
                            copied.record(self.stream)
                            copied.synchronize()
                except BaseException:
                    # Drain partial DMA before finish() can surface the error.
                    self.stream.synchronize()
                    raise

    def finish(self):
        if not self.pending:
            return
        self.start()
        with torch.cuda.nvtx.range("openmopd::io::optimizer_offload_join"):
            try:
                self.future.result()
            except BaseException:
                self.future = None
                raise
            for state, key, source, _ in self.entries:
                if state[key] is not source:
                    raise RuntimeError("optimizer state changed before asynchronous offload completed")
            for state, key, _, destination in self.entries:
                state[key] = destination
            self.entries.clear()
            self.ready = None
            self.future = None

    def close(self):
        try:
            self.finish()
        finally:
            self.executor.shutdown(wait=True)
