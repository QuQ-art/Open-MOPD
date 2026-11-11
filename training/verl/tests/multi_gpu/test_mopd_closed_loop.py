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

"""Four-sample MOPD acceptance, invoked by test_shared_cpu_state --gpu --rollout --mopd."""
import json
from contextlib import nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM

from verl.trainer.ppo.core_algos import compute_token_reward_direct_advantage
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.actor.mt_opd import build_domain_weights, select_routed_teacher_logprobs
from verl.workers.config.actor import FSDPActorConfig
from verl.workers.fsdp_workers import RewardModelWorker
from verl.workers.multi_gpu import MultiGPUGradientSync
from verl.workers.multi_gpu.teacher_store import SharedTeacherStore


def run_mopd_step(actor, optimizer, batch, model_config, device):
    rank = dist.get_rank()
    domains = ['math', 'code', 'if', 'math']
    # The fixture has four fixed, distinct prompts; preserve row identity through scoring/update.
    assert batch.non_tensor_batch['domain'].tolist() == [domains[rank]]
    assert batch.non_tensor_batch['sample_id'].tolist() == [rank]
    batch.to(device)
    reports = [None] * 4
    dist.all_gather_object(reports, (rank, domains[rank], batch.batch['input_ids'].tolist(),
                                    batch.batch['responses'].tolist()))
    assert [r[0] for r in reports] == list(range(4))
    assert [r[1] for r in reports] == domains
    assert len({str(r[2]) for r in reports}) == 4, 'rollouts must belong to different prompts'
    assert len({str(r[3]) for r in reports}) == 4, 'expected four distinct generated responses'
    config = FSDPActorConfig(ppo_mini_batch_size=1, ppo_micro_batch_size_per_gpu=1, ppo_epochs=1,
                            use_torch_compile=False, use_dynamic_bsz=False,
                            loss_agg_mode='token-mean', entropy_coeff=0, grad_clip=1.0)
    learner = DataParallelPPOActor(config, actor, optimizer, global_token_mean=True)
    batch.batch['response_mask'] = torch.ones_like(batch.batch['responses'])
    batch.meta_info.update(temperature=1.0, micro_batch_size=1, use_dynamic_bsz=False,
                           log_prob_top_k=8, top_k_strategy='only_stu', reward_weight_mode='student_p',
                           reward_mode='mt_opd', teacher_temperature=1.0)
    with torch.no_grad():
        _, chosen, ids, logps = learner._forward_micro_batch(batch.batch, 1.0, top_k=8)
    batch.batch['old_log_probs'] = chosen
    batch.batch['student_top_k_ids'] = ids
    batch.batch['student_top_k_log_probs'] = logps
    for parameter in actor.parameters():
        replicas = [torch.empty_like(parameter) for _ in range(4)]
        dist.all_gather(replicas, parameter.detach())
        for replica in replicas:
            torch.testing.assert_close(parameter, replica, atol=0, rtol=0)
    teacher_logps, errors, scorers = [], [], []
    for index in range(3):
        store = SharedTeacherStore(f'teacher/closed_loop/{index}')

        def make_teacher():
            torch.manual_seed(9100 + index)
            return AutoModelForCausalLM.from_config(model_config.hf_config, torch_dtype=torch.bfloat16)

        teacher = store.load_model(make_teacher, make_teacher)
        teacher.requires_grad_(False)
        teacher = FSDP(teacher, device_id=device, use_orig_params=False, cpu_offload=CPUOffload(False),
                       sharding_strategy=MultiGPUGradientSync.sharding_strategy())
        teacher.eval()
        store.bind_fsdp(teacher)
        scorer = SimpleNamespace(
            config=OmegaConf.create({'micro_batch_size_per_gpu': 1, 'use_dynamic_bsz': False,
                                     'model': {'path': f'teacher-{index}'}}),
            rank=rank, world_size=4, reward_module=teacher, _shared_teacher_store=store,
            _do_switch_chat_template=False, use_remove_padding=False, use_fused_kernels=False,
            ulysses_sharding_manager=nullcontext(), ulysses_sequence_parallel_size=1,
        )
        for name in ('_forward_micro_batch', '_forward_model_logits', '_compute_entropy_safe',
                     '_compute_teacher_top_k_log_probs', '_compute_rm_score'):
            setattr(scorer, name, MethodType(getattr(RewardModelWorker, name), scorer))
        scorers.append(scorer)
        output = scorer._compute_rm_score(batch)
        actual = output.batch['teacher_on_student_log_probs']
        # Independent unwrapped single-GPU forward on precisely the same tokens/IDs.
        reference = make_teacher().to(device).eval()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            logits = reference(input_ids=batch.batch['input_ids'], attention_mask=batch.batch['attention_mask'],
                               position_ids=batch.batch['position_ids'], use_cache=False).logits[:, -3:-1]
            expected = logits.float().log_softmax(-1).gather(-1, ids)
        # BF16 logits -> FP32 log-softmax; tolerance declared before execution.
        torch.testing.assert_close(actual.float(), expected, atol=2e-6, rtol=2e-5)
        errors.append((actual.float() - expected).abs().max().item())
        store.assert_shared_sources(teacher)
        teacher_logps.append(actual)
        del reference
    weights = build_domain_weights(batch.non_tensor_batch['domain'], ['math', 'code', 'if'], device)
    routed = select_routed_teacher_logprobs(teacher_logps, weights)
    torch.testing.assert_close(routed, teacher_logps[['math', 'code', 'if'].index(domains[rank])], atol=0, rtol=0)
    batch.batch['teacher_on_student_log_probs'] = routed
    signal = learner.compute_distillation_reward(batch).batch['rm_scores']
    expected_signal = (routed - logps) * torch.softmax(logps, dim=-1)
    torch.testing.assert_close(signal, expected_signal, atol=2e-6, rtol=2e-5)
    advantages, _ = compute_token_reward_direct_advantage(signal, batch.batch['response_mask'])
    torch.testing.assert_close(advantages, expected_signal, atol=2e-6, rtol=2e-5)
    assert torch.isfinite(advantages).all() and advantages.abs().sum() > 0
    batch.batch['advantages'] = advantages
    captured = []

    def checked_step():
        for parameter in actor.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            replicas = [torch.empty_like(parameter.grad) for _ in range(4)]
            dist.all_gather(replicas, parameter.grad)
            for replica in replicas:
                torch.testing.assert_close(parameter.grad, replica, atol=0, rtol=0)
        captured.append(True)
        return DataParallelPPOActor._optimizer_step(learner)

    initial = [p.detach().clone() for p in actor.parameters()]
    with patch.object(learner, '_optimizer_step', side_effect=checked_step), \
            patch.object(optimizer.cpu_optimizer, 'step', wraps=optimizer.cpu_optimizer.step) as cpu_steps:
        learner.update_policy(batch)
    assert len(captured) == 1
    calls = torch.tensor(cpu_steps.call_count, device=device)
    dist.all_reduce(calls)
    assert calls.item() == 1
    assert all(state['step'].item() == 1 for state in optimizer.state.values())
    assert any(not torch.equal(old, new) for old, new in zip(initial, actor.parameters()))
    for parameter in actor.parameters():
        replicas = [torch.empty_like(parameter) for _ in range(4)]
        dist.all_gather(replicas, parameter.detach())
        for replica in replicas:
            torch.testing.assert_close(parameter, replica, atol=0, rtol=0)
    assert batch.non_tensor_batch['domain'].tolist() == [domains[rank]]
    max_error = torch.tensor(max(errors), device=device)
    dist.all_reduce(max_error, op=dist.ReduceOp.MAX)
    if rank == 0:
        print('GPU_MOPD_PASS ' + json.dumps({'samples': reports, 'teacher_max_abs_error_all_ranks': max_error.item(),
              'sequential_teachers': 3, 'signal_reference': True, 'synchronized_gradients': True,
              'cpu_optimizer_calls': calls.item(), 'student_replicas_equal': True}), flush=True)
