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
"""Rank-local teacher slots: original FSDP logits versus two complete slot cycles.

Run via experiments/run_teacher_slots.py; all artifacts live in its output root.
Optional --model-root loads the actual Math/Code/IF checkpoints.
"""
import argparse
import datetime
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP, ShardingStrategy
    from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config
    from verl.utils.fsdp_utils import get_fsdp_wrap_policy
    from verl.workers.fsdp_workers import ActorRolloutRefWorker, RewardModelWorker
    from verl.workers.multi_gpu.teacher_store import SharedTeacherStore

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--model-root')
    args = parser.parse_args()
    device = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=600))
    rank = dist.get_rank()
    teachers = []
    try:
        for index, name in enumerate(('Math', 'Code', 'IF')):
            path = str(Path(args.model_root) / name) if args.model_root else name
            cfg = (AutoConfig.from_pretrained(path) if args.model_root else Qwen2Config(
                vocab_size=64, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128,
                tie_word_embeddings=True, attention_dropout=0.0))
            cfg._attn_implementation = 'eager'

            def empty():
                torch.manual_seed(100 + index)
                return AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)

            store = SharedTeacherStore('teacher/slots/' + name)
            model = store.load_model(
                (lambda: AutoModelForCausalLM.from_pretrained(
                    path, config=cfg, torch_dtype=torch.bfloat16)) if args.model_root else empty, empty)
            model.requires_grad_(False)
            model = FSDP(model, device_id=device, use_orig_params=False, cpu_offload=CPUOffload(False),
                         auto_wrap_policy=get_fsdp_wrap_policy(model),
                         sharding_strategy=ShardingStrategy.NO_SHARD)
            model.eval()
            store.bind_fsdp(model)
            teachers.append(SimpleNamespace(
                world_size=4, ulysses_sequence_parallel_size=1, use_fused_kernels=False,
                _do_switch_chat_template=False, _teacher_layer_pipeline=None,
                _shared_teacher_store=store, reward_module=model,
                config=OmegaConf.create({'model': {'path': path}, 'use_dynamic_bsz': False,
                                        'micro_batch_size_per_gpu': 1})))
        tokens = torch.arange(4 + rank * 8, 12 + rank * 8, device=device).reshape(1, -1)
        batch = {'input_ids': tokens, 'attention_mask': torch.ones_like(tokens),
                 'position_ids': torch.arange(8, device=device).reshape(1, -1)}

        def score(teacher):
            return RewardModelWorker._forward_model_logits(teacher, batch)

        # Drop allocator leftovers from initialization equally for both phases.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = []
        for teacher in teachers:
            logits = score(teacher)
            baseline.append(logits.cpu())
            del logits
        torch.cuda.synchronize()
        before = {'peak_allocated': torch.cuda.max_memory_allocated(),
                  'peak_reserved': torch.cuda.max_memory_reserved(),
                  'idle_allocated': torch.cuda.memory_allocated()}
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        roles = dict(zip(('rm', 'mt_rm_1', 'mt_rm_2'), teachers))
        actor = SimpleNamespace(
            world_size=4, _multi_gpu=True, ulysses_sequence_parallel_size=1, fused_worker_dict=roles,
            get_fused_worker_by_name=roles.get,
            config=OmegaConf.create({'rollout': {'teacher_layer_pipeline_max_mb': 8192}}))
        inspect.unwrap(ActorRolloutRefWorker.prepare_teacher_layer_pipeline)(actor)
        pipeline = actor._scoring_layer_pipeline
        assert pipeline.sequential and pipeline.stream is None
        pointers = {name: slot.tensor.data_ptr() for name, slot in pipeline.slots.items()}
        # No copy is allowed during student scoring in the sequential path.
        pipeline.prefetch_first()
        assert pipeline.copy_counts == [0, 0, 0]
        errors = []
        hooks = []
        for index, modules in enumerate(pipeline.modules):
            for name, module in modules.items():
                def check_slot(mod, inputs, wrapper=module, n=name):
                    assert wrapper._handle.flat_param.data_ptr() == pointers[n]
                hooks.append(module._fsdp_wrapped_module.register_forward_pre_hook(check_slot))
        for cycle in range(2):
            for index, teacher in enumerate(teachers):
                teacher._shared_teacher_store.assert_shared_sources(teacher.reward_module)
                actual = score(teacher).cpu()
                torch.testing.assert_close(actual, baseline[index], atol=0, rtol=0)
                errors.append(float((actual.float() - baseline[index].float()).abs().max()))
                for module in pipeline.modules[index].values():
                    flat = module._handle.flat_param
                    assert flat.device.type == 'cpu'
                    assert flat.data_ptr() == flat._openmopd_shared_cpu.data_ptr()
                assert pointers == {name: slot.tensor.data_ptr() for name, slot in pipeline.slots.items()}
        torch.cuda.synchronize()
        after = {'peak_allocated': torch.cuda.max_memory_allocated(),
                 'peak_reserved': torch.cuda.max_memory_reserved(),
                 'idle_allocated': torch.cuda.memory_allocated()}
        assert pipeline.copy_counts == [2 * len(pointers)] * 3
        for hook in hooks:
            hook.remove()
        pipeline.close()
        # Restoring ordinary FSDP must retain valid mmap sources and logits.
        for index, teacher in enumerate(teachers):
            torch.testing.assert_close(score(teacher).cpu(), baseline[index], atol=0, rtol=0)
            teacher._shared_teacher_store.assert_shared_sources(teacher.reward_module)
        report = {'rank': rank, 'baseline': before, 'slots': after, 'slot_bytes': pipeline.bytes,
                  'slots_count': len(pointers), 'max_logit_errors': errors,
                  'copies': pipeline.copy_counts, 'model_root': args.model_root,
                  'peak_allocated_delta': after['peak_allocated'] - before['peak_allocated']}
        reports = [None] * 4
        dist.all_gather_object(reports, report)
        if rank == 0:
            Path(args.output).write_text(json.dumps({'passed': True, 'ranks': reports}, indent=2) + '\n')
            print('TEACHER_SLOTS_PASS ' + json.dumps(reports), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
