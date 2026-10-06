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

import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP

from verl.workers.teacher_param_prefetch import TeacherHandlePrefetch


class TestTeacherParamPrefetch(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_teacher_forward_reuses_prefetched_shard(self):
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        with tempfile.TemporaryDirectory() if world_size == 1 else nullcontext() as directory:
            if world_size == 1:
                dist.init_process_group("nccl", init_method=Path(directory, "init").as_uri(), rank=0, world_size=1)
            else:
                torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
                dist.init_process_group("nccl", init_method="env://")
            try:
                torch.manual_seed(42)
                model = FSDP(
                    torch.nn.Linear(16, 16, bias=False),
                    cpu_offload=CPUOffload(offload_params=True),
                    device_id=torch.cuda.current_device(),
                ).eval()
                x = torch.randn(2, 16, device="cuda")
                with torch.no_grad():
                    baseline = model(x)

                prefetch = TeacherHandlePrefetch(model, max_mb=1)
                original_calls = 0
                original = prefetch.original_pre_unshards[0]

                def counted_original():
                    nonlocal original_calls
                    original_calls += 1
                    return original()

                prefetch.original_pre_unshards[0] = counted_original
                staged_ptr = prefetch.staged.data_ptr()
                prefetch.start()
                self.assertIsNotNone(prefetch.pending)
                self.assertEqual(prefetch.buffers[0].data_ptr(), staged_ptr)
                with torch.no_grad():
                    result = model(x)
                prefetch.assert_consumed(previous_reuse_count=0)
                prefetch.record_use_done()
                torch.testing.assert_close(result, baseline)
                self.assertEqual(original_calls, 0)  # no second HtoD through FSDP pre_unshard

                prefetch.start()
                self.assertEqual(prefetch.buffers[0].data_ptr(), staged_ptr)
                with torch.no_grad():
                    result = model(x)
                prefetch.assert_consumed(previous_reuse_count=1)
                prefetch.record_use_done()
                torch.testing.assert_close(result, baseline)
                self.assertEqual(original_calls, 0)

                with torch.no_grad():
                    model(x)
                self.assertEqual(original_calls, 1)  # normal path resumes without a pending shard
            finally:
                dist.destroy_process_group()
