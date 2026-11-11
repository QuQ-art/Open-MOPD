# Copyright 2026 Open-MOPD contributors
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
"""Four-GPU Student DP acceptance test, without model downloads or Ray actors.

Run with the training Python environment:
    python tests/multi_gpu/test_student_dp.py
or:
    torchrun --standalone --nproc_per_node=4 tests/multi_gpu/test_student_dp.py --worker

Uses the production DataProto DP dispatcher and NO_SHARD strategy. A small
embedding/MLP student keeps this a gradient test rather than a full OPD run.
Equal sequence lengths make mean(local losses) exactly the global-batch loss.
No manual gradient reduction, optimizer step, or gradient clipping is performed.
FP32 parameters/reduction are compared to a CPU FP64 global-batch reference.
With --normalization, also exercise production update_policy on uneven batches,
micro-batch accumulation, entropy/KL, top-k and empty masks. Capture gradients
before production clipping; an optimizer with lr=0 keeps model weights fixed.
The parent enforces a 180s wall timeout; NCCL collectives have a 60s timeout.
"""
import copy
import datetime
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import sys
import unittest

# Also work when executed directly from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

WORLD_SIZE = 4
ATOL = 2e-6
RTOL = 2e-5
CONFIG = {
    "trainer": {"n_gpus_per_node": 4, "nnodes": 1},
    "data": {"train_batch_size": 4},
    "actor_rollout_ref": {
        "actor": {"strategy": "fsdp", "ppo_mini_batch_size": 4,
                  "ppo_micro_batch_size_per_gpu": 1, "ppo_epochs": 1},
        "rollout": {"tensor_model_parallel_size": 1, "data_parallel_size": 1, "n": 1,
                    "multi_gpu": {"enable": True, "mode": "replicated_dp"}},
    },
}


def worker():
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from torch import nn
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision

    from verl import DataProto
    from unittest.mock import patch

    from verl.single_controller.base.worker_group import WorkerGroup
    from verl.single_controller.base.decorator import dispatch_nd_compute_dataproto
    from verl.workers.config.multi_gpu import validate_training_config
    from verl.workers.multi_gpu import MultiGPUGradientSync

    validate_training_config(OmegaConf.create(CONFIG))
    assert int(os.environ['WORLD_SIZE']) == WORLD_SIZE, 'This acceptance test requires exactly four ranks'
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=60))
    rank = dist.get_rank()

    def gather(tensor):
        result = [torch.empty_like(tensor) for _ in range(WORLD_SIZE)]
        dist.all_gather(result, tensor.contiguous())
        return result

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(32, 16)
            self.block = nn.Sequential(nn.Linear(16, 24), nn.Tanh(), nn.Linear(24, 16))
            self.head = nn.Linear(16, 32)

        def forward(self, tokens):
            hidden = self.embedding(tokens)
            return self.head(hidden + self.block(hidden))

    def loss(model, tokens, targets):
        return nn.functional.cross_entropy(model(tokens).flatten(0, 1), targets.flatten())

    try:
        # Deliberately different seeds: sync_module_states must establish equality.
        torch.manual_seed(1234 + rank)
        student = Student().to(device)
        fsdp = FSDP(student, device_id=device,
                    sharding_strategy=MultiGPUGradientSync.sharding_strategy(),
                    sync_module_states=True, use_orig_params=True,
                    mixed_precision=MixedPrecision(param_dtype=torch.float32,
                                                   reduce_dtype=torch.float32,
                                                   buffer_dtype=torch.float32))
        assert fsdp.sharding_strategy.name == 'NO_SHARD'
        for name, parameter in student.named_parameters():
            replicas = gather(parameter.detach())
            for other in replicas:
                torch.testing.assert_close(parameter, other, atol=0, rtol=0, msg=f'initial Student: {name}')
        initial = {name: value.detach().cpu().clone() for name, value in student.state_dict().items()}
        reference = Student().double()
        reference.load_state_dict(initial)

        tokens = torch.arange(4 * 6).reshape(4, 6) % 32
        targets = (tokens * 3 + 7) % 32
        global_batch = DataProto.from_dict(tensors={
            'input_ids': tokens, 'targets': targets, 'sample_id': torch.arange(WORLD_SIZE),
        })

        class Group(WorkerGroup):
            world_size = WORLD_SIZE

            def __init__(self):
                pass  # Only rank routing is needed, not live Ray workers.

        # The same rank mapping registered by ActorRolloutRefWorker for SP=1.
        # Bypass only Ray object-store transport; retain real splitting/rank routing.
        with patch("verl.utils.ray_utils.parallel_put", side_effect=lambda values, **kwargs: values):
            args, _ = dispatch_nd_compute_dataproto(list(range(WORLD_SIZE)), WORLD_SIZE, Group(), global_batch)
        local_batch = args[0][rank].batch
        assert len(local_batch) == 1
        local_ids = gather(local_batch['sample_id'].to(device))
        assert torch.cat(local_ids).cpu().tolist() == [0, 1, 2, 3], 'samples duplicated or omitted by DP dispatch'
        actual_inputs = gather(local_batch['input_ids'].to(device))
        torch.testing.assert_close(torch.cat(actual_inputs).cpu(), tokens, atol=0, rtol=0)
        assert len({tuple(item.cpu().flatten().tolist()) for item in actual_inputs}) == WORLD_SIZE

        # Negative control: independent local gradients must differ. Otherwise
        # cross-rank equality alone could falsely pass without all-reduce.
        local_reference = copy.deepcopy(reference)
        loss(local_reference, local_batch['input_ids'], local_batch['targets']).backward()
        local_grad = torch.cat([p.grad.flatten() for p in local_reference.parameters()]).to(device)
        raw_gradients = gather(local_grad)
        assert all(not torch.allclose(raw_gradients[0], other) for other in raw_gradients[1:])

        # Only FSDP backward performs gradient synchronization.
        loss(fsdp, local_batch['input_ids'].to(device), local_batch['targets'].to(device)).backward()
        torch.cuda.synchronize()
        completed = gather(torch.ones(1, device=device, dtype=torch.int32))
        assert sum(int(value.item()) for value in completed) == WORLD_SIZE

        # Single-process, full global batch, no distributed wrapper, FP64 arithmetic.
        loss(reference, tokens, targets).backward()
        expected = dict(reference.named_parameters())
        max_error = 0.0
        checked = 0
        for name, parameter in student.named_parameters():
            assert parameter.grad is not None, f'missing gradient: {name}'
            assert torch.isfinite(parameter.grad).all(), f'nonfinite gradient: {name}'
            gradients = gather(parameter.grad)
            for other in gradients:
                torch.testing.assert_close(parameter.grad, other, atol=0, rtol=0, msg=f'rank gradients: {name}')
            actual = parameter.grad.detach().cpu().double()
            target = expected[name].grad
            torch.testing.assert_close(actual, target, atol=ATOL, rtol=RTOL, msg=f'global FP64 reference: {name}')
            max_error = max(max_error, (actual - target).abs().max().item())
            checked += parameter.numel()
        # Ensure this tests gradients before any parameter update/publication.
        for name, parameter in student.named_parameters():
            torch.testing.assert_close(parameter.detach().cpu(), initial[name], atol=0, rtol=0)
        if '--normalization' in sys.argv:
            from verl.workers.actor.dp_actor import DataParallelPPOActor
            from verl.workers.config.actor import FSDPActorConfig

            class ToyActor(DataParallelPPOActor):
                # Substitute only model-specific token/log-prob extraction; use
                # the production update_policy, losses, accumulation and clipping.
                def _forward_micro_batch(self, inputs, temperature, calculate_entropy=False,
                                         top_k=0, student_top_k_ids=None):
                    log_probs = self.actor_module(inputs['input_ids']).log_softmax(-1)
                    chosen = log_probs.gather(-1, inputs['responses'].unsqueeze(-1)).squeeze(-1)
                    entropy = -(log_probs.exp() * log_probs).sum(-1) if calculate_entropy else None
                    topk = log_probs.gather(-1, student_top_k_ids) if top_k else None
                    return entropy, chosen, None, topk

            cases = [
                ('unequal', [1, 2, 4, 6], 1, 'token-mean', False, False, False, False),
                ('sequence_mean', [1, 2, 4, 6], 1, 'seq-mean-token-mean', False, False, False, False),
                ('empty_rank', [0, 2, 4, 6], 1, 'token-mean', False, False, False, False),
                ('accumulation', [1, 6, 2, 4, 3, 2, 6, 1], 1, 'token-mean', True, False, False, False),
                ('whole_local_batch', [1, 6, 2, 4, 3, 2, 6, 1], 2, 'token-mean', True, False, False, False),
                ('dynamic_batch', [1, 6, 2, 4, 3, 2, 6, 1], 1, 'token-mean', True, False, False, True),
                ('topk_accumulation', [1, 6, 2, 4, 3, 2, 6, 1], 1, 'token-mean', True, True, True, False),
                ('topk_format_mask', [1, 6, 2, 4, 3, 2, 6, 1], 2, 'token-mean', True, True, True, False),
                ('all_masked', [0, 0, 0, 0], 1, 'token-mean', True, False, False, False),
            ]
            normalization_reports = []
            for name, valid_lengths, micro_size, mode, auxiliary, topk, format_filter, dynamic in cases:
                fsdp.zero_grad(set_to_none=True)
                lengths = torch.tensor(valid_lengths)
                count = len(valid_lengths)
                per_rank = count // WORLD_SIZE
                inputs = torch.arange(count * 6).reshape(count, 6) % 32
                responses = (inputs * 3 + 7) % 32
                mask = torch.arange(6).unsqueeze(0) < lengths.unsqueeze(1)
                ids = (responses.unsqueeze(-1) + torch.arange(3)) % 32
                shape = ids.shape if topk else responses.shape
                advantages = (torch.arange(math.prod(shape)).reshape(shape) % 11 - 5).float() / 7
                tensors = dict(input_ids=inputs, responses=responses, response_mask=mask,
                               attention_mask=torch.ones_like(inputs),
                               position_ids=torch.arange(6).expand(count, -1),
                               old_log_probs=torch.zeros_like(responses, dtype=torch.float32), advantages=advantages)
                if topk:
                    tensors['student_top_k_ids'] = ids
                if auxiliary:
                    tensors['ref_log_prob'] = torch.full_like(responses, -3.0, dtype=torch.float32)
                policy_mask = mask
                if format_filter:
                    tensors['format_mask'] = (torch.arange(count) % 3 != 0).long().unsqueeze(-1)
                    policy_mask = mask * tensors['format_mask']
                batch = DataProto.from_dict(tensors=tensors, meta_info={'temperature': 1.0})
                local_data = batch[rank * per_rank:(rank + 1) * per_rank]
                config = FSDPActorConfig(ppo_mini_batch_size=per_rank,
                                        ppo_micro_batch_size_per_gpu=micro_size,
                                        use_torch_compile=False, use_dynamic_bsz=dynamic,
                                        ppo_max_token_len_per_gpu=6,
                                        loss_agg_mode=mode, entropy_coeff=0.13 if auxiliary else 0,
                                        use_kl_loss=auxiliary, kl_loss_type='mse', kl_loss_coef=0.2,
                                        grad_clip=1e6)
                learner = ToyActor(config, fsdp, torch.optim.SGD(fsdp.parameters(), lr=0), global_token_mean=True)
                captured = []

                def capture():
                    captured.append([p.grad.detach().clone() for p in student.parameters()])
                    return DataParallelPPOActor._optimizer_step(learner)

                with patch.object(learner, '_optimizer_step', side_effect=capture), \
                        patch.object(dist, 'all_reduce', wraps=dist.all_reduce) as reductions:
                    learner.update_policy(local_data)
                assert len(captured) == 1, 'expected one update after all local micro-batches'
                count_reductions = [call for call in reductions.call_args_list
                                    if call.args[0].dtype == torch.int64 and call.args[0].numel() == 2]
                assert len(count_reductions) == int(mode == 'token-mean')
                reference.zero_grad(set_to_none=True)
                log_probs = reference(inputs).log_softmax(-1)
                chosen = log_probs.gather(-1, responses.unsqueeze(-1)).squeeze(-1)
                selected = log_probs.gather(-1, ids) if topk else chosen
                # The derivative of the on-policy PPO ratio at 1 is d(log pi).
                # Compute that gradient independently, without production agg_loss.
                policy_losses = -selected * advantages.double()
                if topk:
                    policy_losses = policy_losses.sum(-1)
                if mode == 'token-mean':
                    reference_loss = (policy_losses * policy_mask).sum() / policy_mask.sum().clamp_min(1)
                else:
                    reference_loss = ((policy_losses * mask).sum(-1) / lengths.clamp_min(1)).mean()
                if auxiliary:
                    entropy = -(log_probs.exp() * log_probs).sum(-1)
                    kld = 0.5 * (chosen - tensors['ref_log_prob']).square()
                    reference_loss += ((-0.13 * entropy + 0.2 * kld) * mask).sum() / mask.sum().clamp_min(1)
                reference_loss.backward()
                matches = True
                error = 0.0
                for actual, target in zip(captured[0], reference.parameters(), strict=True):
                    replicas = gather(actual)
                    for replica in replicas:
                        torch.testing.assert_close(actual, replica, atol=0, rtol=0)
                    gradient = actual.cpu().double()
                    error = max(error, (gradient - target.grad).abs().max().item())
                    matches &= torch.allclose(gradient, target.grad, atol=ATOL, rtol=RTOL)
                normalization_reports.append({'case': name, 'loss_agg_mode': mode,
                                              'valid_tokens_per_rank': lengths.reshape(4, -1).sum(-1).tolist(),
                                              'micro_batch_size': micro_size, 'auxiliary_losses': auxiliary,
                                              'token_count_reductions': len(count_reductions),
                                              'global_reference_match': matches, 'max_abs_error': error})
            if rank == 0:
                print('LOSS_NORMALIZATION_RESULT ' + json.dumps(normalization_reports), flush=True)
            assert all(item['global_reference_match'] for item in normalization_reports), (
                'global-batch normalization mismatch; see LOSS_NORMALIZATION_RESULT'
            )
        dist.barrier()
        if rank == 0:
            print('STUDENT_DP_PASS ' + json.dumps({
                'world_size': WORLD_SIZE, 'sample_ids': [0, 1, 2, 3],
                'initial_students_identical': True, 'backward_completed': True,
                'rank_gradients_identical': True, 'reference_dtype': 'float64',
                'checked_parameters': checked, 'max_abs_error': max_error,
                'atol': ATOL, 'rtol': RTOL,
            }), flush=True)
    finally:
        dist.destroy_process_group()


def launch():
    import torch
    if torch.cuda.device_count() < WORLD_SIZE:
        raise RuntimeError('Four visible CUDA GPUs are required; set CUDA_VISIBLE_DEVICES appropriately')
    env = dict(os.environ, TORCH_NCCL_ASYNC_ERROR_HANDLING='1', OMP_NUM_THREADS='1')
    process = subprocess.Popen(
        [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
         str(Path(__file__).resolve()), '--worker',
         *(['--normalization'] if '--normalization' in sys.argv else [])], env=env, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        # Kill the entire torchrun process group, including stuck NCCL workers.
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        raise AssertionError(f'Four-GPU test timed out after 180s\n{stdout}\n{stderr}') from None
    if process.returncode or 'STUDENT_DP_PASS ' not in stdout:
        raise AssertionError(f'Four-GPU Student DP failed ({process.returncode})\n{stdout}\n{stderr}')
    print(stdout, end='')


class StudentDPTest(unittest.TestCase):
    def test_four_gpu_student_gradients(self):
        import torch
        if torch.cuda.device_count() < WORLD_SIZE:
            self.skipTest('requires four CUDA GPUs')
        launch()


if __name__ == '__main__':
    if '--worker' in sys.argv:
        worker()
    else:
        launch()
