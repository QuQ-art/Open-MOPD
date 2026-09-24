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
"""One in-flight teacher model forward on a dedicated CUDA submission thread."""

from concurrent.futures import ThreadPoolExecutor

import torch


class TeacherForwardOverlap:
    """Keep CPU submission independent; transfer logits ownership with an event.

    The caller owns all FSDP model state and must submit only one forward at a
    time. This is deliberately a single-slot executor, not a multi-batch queue.
    """

    def __init__(self, device, name="Math"):
        self.device = torch.device(device)
        self.name = name
        self.stream = torch.cuda.Stream(device=self.device)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="teacher-forward")
        self.future = None
        self.last_done = None

    def start(self, forward, inputs_ready):
        if self.future is not None:
            raise RuntimeError("previous teacher forward has not been collected")
        self.future = self.executor.submit(self._run, forward, inputs_ready)

    def _run(self, forward, inputs_ready):
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream), torch.no_grad():
            self.stream.wait_event(inputs_ready)
            with torch.cuda.nvtx.range(f"openmopd::compute::teacher_model_forward::{self.name}"):
                logits = forward()
                done = torch.cuda.Event()
                done.record(self.stream)
            return logits, done

    def finish(self):
        if self.future is None:
            raise RuntimeError("teacher forward was not started")
        try:
            logits, done = self.future.result()
            self.last_done = done
            consumer = torch.cuda.current_stream(self.device)
            consumer.wait_event(done)
            logits.record_stream(consumer)
            return logits
        except BaseException:
            # Do not leave kernels using model state after a failed task.
            self.stream.synchronize()
            raise
        finally:
            self.future = None

    def drain(self):
        """Join pending CPU/GPU work when student execution raises."""
        try:
            if self.future is not None:
                self.future.result()
        finally:
            self.stream.synchronize()
            self.future = None

    def close(self):
        try:
            self.drain()
        finally:
            self.executor.shutdown(wait=True)
