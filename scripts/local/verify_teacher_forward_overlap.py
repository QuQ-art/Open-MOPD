#!/usr/bin/env python3
"""Fixed-input real-model scoring comparison; no rollout or optimizer update.

Run with PYTHONPATH=training/verl and the training environment. The lightweight
worker adapters invoke the production scoring methods without starting Ray.
Timings cover local scoring through GPU completion, not a full training step.
"""

import argparse
import hashlib
import json
import os
import pickle
import statistics
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM

from verl import DataProto
from verl.utils.fsdp_utils import get_fsdp_wrap_policy
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.fsdp_workers import ActorRolloutRefWorker, RewardModelWorker


def bind(target, cls, names):
    for name in names:
        setattr(target, name, MethodType(getattr(cls, name), target))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--student', required=True)
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=8)
    parser.add_argument('--sequence-length', type=int, default=1248)
    parser.add_argument('--response-length', type=int, default=1024)
    args = parser.parse_args()
    if args.pairs < 1 or not 0 < args.response_length < args.sequence_length:
        parser.error('require positive pairs and 0 < response-length < sequence-length')
    torch.cuda.set_device(0)
    torch.manual_seed(31)
    teacher = None
    with tempfile.TemporaryDirectory() as directory:
        dist.init_process_group('nccl', init_method=Path(directory, 'init').as_uri(), rank=0, world_size=1)
        try:
            def load(path, offload):
                print(f'Loading {path}', flush=True)
                raw = AutoModelForCausalLM.from_pretrained(
                    path, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2',
                    local_files_only=True,
                ).eval()
                wrap = get_fsdp_wrap_policy(raw, OmegaConf.create({'wrap_policy': {'min_num_params': 0}}))
                return FSDP(raw, auto_wrap_policy=wrap, device_id=0, use_orig_params=False,
                            cpu_offload=CPUOffload(offload_params=offload))

            student_model = load(args.student, False)
            teacher_model = load(args.teacher, True)
            actor = SimpleNamespace(
                actor_module=student_model, config=SimpleNamespace(entropy_checkpointing=False),
                use_remove_padding=False, use_fused_kernels=False, use_ulysses_sp=False, device_name='cuda',
            )
            bind(actor, DataParallelPPOActor, ('_log_prob_model_forward', '_forward_micro_batch', 'compute_log_prob'))
            teacher = SimpleNamespace(
                reward_module=teacher_model, use_remove_padding=False, use_fused_kernels=False,
                _do_switch_chat_template=False, world_size=1, ulysses_sequence_parallel_size=1,
                ulysses_sharding_manager=nullcontext(), _teacher_forward_overlap=None, _teacher_handle_prefetch=None,
                config=OmegaConf.create({'use_dynamic_bsz': False, 'micro_batch_size_per_gpu': 1,
                                         'model': {'path': args.teacher}}),
            )
            bind(teacher, RewardModelWorker, (
                '_forward_model_logits', '_forward_micro_batch', '_compute_entropy_safe',
                '_compute_teacher_top_k_log_probs', 'prepare_forward_overlap', '_compute_rm_score',
                'prepare_param_prefetch', 'start_param_prefetch',
            ))
            teacher.prepare_param_prefetch(768)
            worker = SimpleNamespace(
                actor=actor, _is_actor=True, _is_offload_param=False, world_size=1,
                ulysses_sharding_manager=nullcontext(), get_fused_worker_by_name=lambda name: teacher,
                config=OmegaConf.create({'rollout': {
                    'teacher_forward_overlap': True, 'teacher_param_prefetch': True,
                    'teacher_param_prefetch_max_mb': 768, 'log_prob_micro_batch_size_per_gpu': 1,
                    'log_prob_max_token_len_per_gpu': args.sequence_length, 'log_prob_use_dynamic_bsz': False,
                    'temperature': 1.0, 'log_prob_top_k': 256,
                }}),
            )
            bind(worker, ActorRolloutRefWorker, ('_compute_log_prob', 'compute_log_prob_and_teacher'))
            inputs = torch.randint(0, student_model.config.vocab_size, (1, args.sequence_length))
            source = DataProto.from_dict({
                'input_ids': inputs, 'responses': inputs[:, -args.response_length:].clone(),
                'attention_mask': torch.ones_like(inputs),
                'position_ids': torch.arange(args.sequence_length).unsqueeze(0),
                'response_mask': torch.ones(1, args.response_length, dtype=torch.long),
            }, meta_info={'reward_mode': 'mt_opd', 'log_prob_top_k': 256, 'top_k_strategy': 'only_stu'})
            serialized = pickle.dumps(source)

            def score(mode):
                data = pickle.loads(serialized)
                # Control allocator starting state and exclude cleanup from timing.
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                if mode == 'overlap':
                    out = worker.compute_log_prob_and_teacher(data)
                else:
                    data = DataProto.from_dict(dict(data.batch.items()), meta_info=dict(data.meta_info))
                    student = worker._compute_log_prob(data)
                    data.union(student)
                    out = student.union(teacher._compute_rm_score(data))
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - started) * 1000
                memory = {name: getattr(torch.cuda, name)() / 2**30
                          for name in ('max_memory_allocated', 'max_memory_reserved')}
                # Materialize only after the measured GPU-completion boundary.
                tensors = {key: value.detach().cpu().clone() for key, value in out.batch.items()}
                return tensors, {'mode': mode, 'elapsed_ms': elapsed, 'memory_GiB': memory}

            reference, _ = score('serial')
            warm, _ = score('overlap')
            for key in reference:
                torch.testing.assert_close(warm[key], reference[key], rtol=0, atol=0)
            records = []
            for pair in range(args.pairs):
                for mode in (('serial', 'overlap') if pair % 2 == 0 else ('overlap', 'serial')):
                    actual, record = score(mode)
                    for key in reference:
                        torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)
                    record.update(pair=pair, exact_outputs=True)
                    records.append(record)
                    print(json.dumps(record), flush=True)
            summary = {
                mode: {'median_ms': statistics.median(r['elapsed_ms'] for r in records if r['mode'] == mode),
                       'peak_allocated_GiB': max(r['memory_GiB']['max_memory_allocated']
                                                for r in records if r['mode'] == mode),
                       'peak_reserved_GiB': max(r['memory_GiB']['max_memory_reserved']
                                               for r in records if r['mode'] == mode)}
                for mode in ('serial', 'overlap')
            }
            result = {'scope': 'local fixed-input scoring; frozen checkpoints; no Ray, rollout, Code teacher or update',
                      'synthetic_input': True, 'sequence_length': args.sequence_length,
                      'response_length': args.response_length, 'student': args.student, 'teacher': args.teacher,
                      'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                      'cuda_environment': {key: os.environ.get(key) for key in (
                          'CUDA_DEVICE_MAX_CONNECTIONS', 'CUDA_DEVICE_MAX_COPY_CONNECTIONS',
                          'CUDA_VISIBLE_DEVICES', 'CUDA_LAUNCH_BLOCKING')},
                      'input_sha256': hashlib.sha256(inputs.numpy().tobytes()).hexdigest(),
                      'output_sha256': {
                          key: hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                          for key, value in reference.items()
                      },
                      'checked_output_keys': sorted(reference), 'records': records, 'summary': summary,
                      'copy_count': teacher._teacher_handle_prefetch.copy_count,
                      'reuse_count': teacher._teacher_handle_prefetch.reuse_count}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(summary, indent=2), flush=True)
        finally:
            if teacher is not None and teacher._teacher_forward_overlap is not None:
                teacher._teacher_forward_overlap.close()
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
