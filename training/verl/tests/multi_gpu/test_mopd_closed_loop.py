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
                            loss_agg_mode='token-mean', entropy_coeff=0, grad_clip=1e-4)
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
    assert optimizer.gradient_sync == 'cpu_reduce'
    world_size = dist.get_world_size()
    # Independent unwrapped backward on this rank's tokens. With on-policy PPO
    # ratio=1, -A * log(pi) has the same derivative, without calling the PPO loss.
    # The rollout fixture's actor is loaded from_pretrained, whose attention
    # backend may differ from HFModelConfig's teacher override (eager).
    local_reference = AutoModelForCausalLM.from_config(
        actor.module.config, torch_dtype=torch.bfloat16,
        attn_implementation=actor.module.config._attn_implementation,
    ).to(device)
    with torch.no_grad():
        for reference_param, parameter in zip(local_reference.parameters(), actor.parameters(), strict=True):
            reference_param.copy_(parameter)
    local_reference.train()
    mask = batch.batch['response_mask']
    global_tokens = mask.sum().clone()
    dist.all_reduce(global_tokens)
    response_length = batch.batch['responses'].shape[-1]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = local_reference(input_ids=batch.batch['input_ids'], attention_mask=batch.batch['attention_mask'],
                                 position_ids=batch.batch['position_ids'], use_cache=False).logits
        selected = logits[:, -response_length - 1:-1].log_softmax(-1).gather(-1, ids)
        loss = -(selected * advantages.detach() * mask.unsqueeze(-1)).sum() * world_size / global_tokens
    loss.backward()
    local_expected = [p.grad.detach().cpu().float().clone() for p in local_reference.parameters()]
    del local_reference, logits, selected, loss
    captured, reductions = [], []
    initial = [p.detach().clone() for p in actor.parameters()]
    reference_masters = [torch.nn.Parameter(p.cpu().float()) for p in initial]
    reference_optimizer = torch.optim.AdamW(reference_masters, foreach=False,
                                           **optimizer._options(optimizer.param_groups[0]))
    raw_clip = torch.nn.utils.clip_grad_norm_

    def checked_step():
        expected_means, snapshots = [], []
        ranks_differ = False
        local_errors = []
        for (parameter, _, _), expected in zip(optimizer.pairs, local_expected, strict=True):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            # BF16 backward comparison; these tolerances are fixed before execution.
            try:
                torch.testing.assert_close(parameter.grad.cpu().float(), expected, atol=2e-5, rtol=2e-2)
            except AssertionError as error:
                local_errors.append(str(error))
            replicas = [torch.empty_like(parameter.grad) for _ in range(world_size)]
            dist.all_gather(replicas, parameter.grad)
            values = torch.stack([g.cpu().float() for g in replicas])
            ranks_differ |= any(not torch.equal(values[0], value) for value in values[1:])
            snapshots.append(values)
            # FP64 independent mean, compared against the FP32 CPU reducer.
            expected_means.append(values.double().mean(0).float())
        rank_errors = [None] * world_size
        dist.all_gather_object(rank_errors, local_errors)
        assert not any(rank_errors), f'Local gradient reference mismatches: {rank_errors}'
        assert ranks_differ, 'rank-local gradients unexpectedly identical before CPU reduce'
        for parameter, mean in zip(reference_masters, expected_means, strict=True):
            parameter.grad = mean.clone()
        reference_norm = raw_clip(reference_masters, config.grad_clip)
        assert reference_norm > config.grad_clip, 'fixture must exercise nontrivial clipping'

        def check_cpu_clip(parameters, max_norm, *args, **kwargs):
            parameters = list(parameters)
            assert rank == optimizer.owner_rank, 'only the owner may reduce/clip CPU gradients'
            assert max_norm == config.grad_clip
            for parameter, (_, master, _), (present, reduced), values, mean in zip(
                    parameters, optimizer.pairs, optimizer.grad_buffers, snapshots, expected_means, strict=True):
                assert parameter is master and parameter.device.type == 'cpu'
                assert parameter.grad.data_ptr() == reduced.data_ptr(), 'Adam must consume the shared reduced buffer'
                assert present.all()
                torch.testing.assert_close(reduced, mean, atol=2e-7, rtol=2e-5)
            norm = raw_clip(parameters, max_norm, *args, **kwargs)
            torch.testing.assert_close(norm, reference_norm, atol=2e-7, rtol=2e-5)
            for parameter, reference_param in zip(parameters, reference_masters, strict=True):
                torch.testing.assert_close(parameter.grad, reference_param.grad, atol=2e-7, rtol=2e-5)
            reductions.append(True)
            return norm

        raw_reduce = optimizer._reduce_chunk
        def check_workspace(segments):
            for index, offset, start, size in segments:
                expected = snapshots[index].view(world_size, -1)[:, offset:offset + size]
                torch.testing.assert_close(optimizer.workspace[:, start:start + size], expected, atol=0, rtol=0)
            return raw_reduce(segments)

        with patch.object(optimizer, '_reduce_chunk', side_effect=check_workspace), \
                patch('torch.nn.utils.clip_grad_norm_', side_effect=check_cpu_clip), \
                patch.object(dist, 'broadcast', side_effect=AssertionError('GPU parameter broadcast during update')):
            norm = DataParallelPPOActor._optimizer_step(learner)
        torch.testing.assert_close(norm.cpu(), reference_norm, atol=2e-7, rtol=2e-5)
        assert len(reductions) == int(rank == optimizer.owner_rank)
        # Local GPU gradients remain local even after the CPU reduction/update.
        for (parameter, _, _), values, (_, reduced), reference_param in zip(
                optimizer.pairs, snapshots, optimizer.grad_buffers, reference_masters, strict=True):
            torch.testing.assert_close(parameter.grad.cpu().float(), values[rank], atol=0, rtol=0)
            torch.testing.assert_close(reduced, reference_param.grad, atol=2e-7, rtol=2e-5)
        reference_optimizer.step()
        for (parameter, master, host), reference_param in zip(optimizer.pairs, reference_masters, strict=True):
            torch.testing.assert_close(master, reference_param, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(host, master.to(host.dtype), atol=0, rtol=0)
            torch.testing.assert_close(parameter.cpu(), host, atol=0, rtol=0)
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(optimizer.state[parameter][key], reference_optimizer.state[reference_param][key],
                                           atol=2e-7, rtol=2e-5)
        captured.append(True)
        return norm

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
              'sequential_teachers': 3, 'signal_reference': True, 'gradient_sync': 'cpu_reduce', 'local_gradients_match_reference': True,
              'local_gradients_differ': True, 'cpu_mean_before_clip_matches': True,
              'cpu_clip_and_adam_reference': True, 'gpu_parameter_broadcast': False,
              'cpu_optimizer_calls': calls.item(), 'student_replicas_equal': True}), flush=True)
