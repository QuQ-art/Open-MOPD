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

"""Paired transport/reduce benchmark; identical BF16 gradients, no backward changes."""
import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.distributed as dist
from verl.workers.multi_gpu import MultiGPUWeightPublisher
from verl.workers.multi_gpu.shared_adam import SharedCPUAdamW

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--elements', type=int, default=3075098624)
parser.add_argument('--rounds', type=int, default=3)
args = parser.parse_args()
device = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(device)
torch.set_num_threads(8)
dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=600))
rank = dist.get_rank()
try:
    parameter = torch.nn.Parameter(torch.zeros(args.elements, device=device, dtype=torch.bfloat16))
    parameter.grad = torch.full_like(parameter, rank + 1)
    opt = SharedCPUAdamW(torch.optim.AdamW([parameter]), 0, MultiGPUWeightPublisher(0), reduce_mode='async')
    modes = ['serial', 'double_buffer', 'async']
    records = []
    # Warm every mode, then rotate order to reduce systematic warm-cache/order bias.
    for round_id in range(-1, args.rounds):
        order = modes if round_id < 0 else modes[round_id % 3:] + modes[:round_id % 3]
        for mode in order:
            opt.reduce_mode = mode
            opt.workspace = opt.workspaces[0]
            dist.barrier()
            torch.cuda.synchronize()
            start = time.perf_counter()
            norm = opt.clip_grad_norm_(float('inf'))
            elapsed = time.perf_counter() - start
            assert torch.isfinite(norm)
            for offset in range(0, args.elements, 1024 * 1024):
                torch.testing.assert_close(opt.grad_buffers[0][1][offset:offset + 1024 * 1024],
                                           torch.full_like(opt.grad_buffers[0][1][offset:offset + 1024 * 1024], 2.5),
                                           atol=0, rtol=0)
            entry = dict(round=round_id, mode=mode, rank=rank, elapsed_s=elapsed, **opt.last_reduce_metrics)
            entries = [None] * dist.get_world_size()
            dist.all_gather_object(entries, entry)
            if rank == 0:
                records.extend(entries)
                print(json.dumps(entries), flush=True)
    if rank == 0:
        Path(args.output).write_text(json.dumps(dict(elements=args.elements, chunk_bytes=256 * 1024**2,
            shared_workspace_bytes=sum(w.nbytes for w in opt.workspaces),
            pinned_bytes_per_rank=sum(w.nbytes for w in opt.staging), records=records), indent=2))
finally:
    dist.destroy_process_group()
