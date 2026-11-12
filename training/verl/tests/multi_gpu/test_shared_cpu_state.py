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

"""Shared-memory acceptance: python this_file.py [--ray | --gpu]."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from verl.workers.multi_gpu.cpu_state_store import CPUStateStore, ENV_NAME


def child_mutate(root, namespace, descriptor):
    import torch
    store = CPUStateStore(namespace, root)
    tensor = store.attach(descriptor)
    tensor.add_(3)
    assert tensor.dtype == torch.float32


class SharedMemoryTests(unittest.TestCase):
    def test_spawn_maps_same_writable_storage(self):
        import multiprocessing
        import torch
        with tempfile.TemporaryDirectory() as root:
            owner = CPUStateStore('spawn-test', root, owner=True)
            tensor, desc = owner.allocate('student', 'weight', (2, 3), torch.float32, fill=2)
            process = multiprocessing.get_context('spawn').Process(target=child_mutate, args=(root, owner.namespace, desc))
            process.start()
            process.join(30)
            if process.is_alive():
                process.kill()
                process.join()
                self.fail('shared memory child timed out')
            self.assertEqual(process.exitcode, 0)
            torch.testing.assert_close(tensor, torch.full_like(tensor, 5))
            self.assertEqual(owner.inventory(), [desc])
            self.assertEqual(desc['model'], 'student')
            with self.assertRaises(ValueError):
                owner.allocate('student', 'weight', (4,), torch.float32)
            with self.assertRaises(PermissionError):
                CPUStateStore(owner.namespace, root).unlink()
            owner.unlink()
            # Unlinking the name does not invalidate existing live mappings.
            torch.testing.assert_close(tensor, torch.full_like(tensor, 5))


def ray_test():
    import ray
    import torch

    @ray.remote(num_cpus=1)
    class Reader:
        def __init__(self, descriptor):
            self.store = CPUStateStore(descriptor['namespace'], descriptor['root'])
            self.tensor = self.store.attach(descriptor)
            self.descriptor = descriptor

        def write(self, index, value):
            self.tensor[index] = value
            return os.getpid()

        def read(self):
            return {'pid': os.getpid(), 'values': self.tensor.tolist(),
                    'inode': os.stat(self.store.path / self.descriptor['key']).st_ino}

    with tempfile.TemporaryDirectory() as root:
        owner = CPUStateStore('ray-test', root, owner=True)
        tensor, descriptor = owner.allocate('student', 'ray-weight', (4,), torch.float32)
        ray.init(num_cpus=4, include_dashboard=False, runtime_env={"env_vars": {
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]) + os.pathsep + str(Path(__file__).parent),
        }})
        try:
            workers = [Reader.remote(descriptor) for _ in range(4)]
            ray.get([worker.write.remote(index, index + 10) for index, worker in enumerate(workers)], timeout=60)
            results = ray.get([worker.read.remote() for worker in workers], timeout=60)
            assert len({result['pid'] for result in results}) == 4
            assert len({result['inode'] for result in results}) == 1
            assert all(result['values'] == [10, 11, 12, 13] for result in results)
            assert tensor.tolist() == [10, 11, 12, 13]
            print('RAY_SHARED_CPU_PASS ' + json.dumps(results), flush=True)
        finally:
            ray.shutdown()
            owner.unlink()


def gpu_worker():
    import copy
    import datetime
    from types import SimpleNamespace
    from unittest.mock import patch
    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload
    from verl.workers.multi_gpu import MultiGPUGradientSync, MultiGPUWeightPublisher
    from verl.workers.multi_gpu.shared_adam import SharedCPUAdamW
    from verl.workers.multi_gpu.teacher_store import SharedTeacherStore
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    device = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=60))
    rank = dist.get_rank()
    assert dist.get_world_size() == 4
    try:
        def make_model():
            return nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2))

        def owner_load():
            assert rank == 0, 'non-owner must not load checkpoints'
            torch.manual_seed(4321)
            return make_model()

        student_store = SharedTeacherStore('student')
        student = student_store.load_model(owner_load, make_model)
        reference = make_model()
        reference.load_state_dict(student.state_dict())
        actor = FSDP(student, device_id=device, use_orig_params=True,
                     sharding_strategy=MultiGPUGradientSync.sharding_strategy(), sync_module_states=True)
        student_store.finish_initialization()
        optimizer = SharedCPUAdamW(torch.optim.AdamW(actor.parameters(), lr=0.01), 1,
                                   MultiGPUWeightPublisher(1))
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01, foreach=False)
        learner = SimpleNamespace(config=SimpleNamespace(grad_clip=0.1), actor_module=actor,
                                  actor_optimizer=optimizer)
        data = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 12
        step_reports = []
        for expected_step in (1, 2):
            with actor.no_sync():
                actor(data[rank:rank + 1].cuda()).square().mean().backward()
            reference(data).square().mean().backward()
            # FSDP must leave rank-local gradients untouched before CPU reduction.
            local = make_model()
            local.load_state_dict(reference.state_dict())
            local(data[rank:rank + 1]).square().mean().backward()
            for actual, target in zip(student.parameters(), local.parameters(), strict=True):
                torch.testing.assert_close(actual.grad.cpu(), target.grad, atol=2e-6, rtol=2e-5)
            with patch.object(optimizer.cpu_optimizer, 'step', wraps=optimizer.cpu_optimizer.step) as step, \
                    patch.object(dist, 'broadcast', side_effect=AssertionError('GPU parameter broadcast')):
                norm = DataParallelPPOActor._optimizer_step(learner)
            calls = torch.tensor([step.call_count], device=device)
            dist.all_reduce(calls)
            assert calls.item() == 1, 'AdamW executed more or less than once across four ranks'
            reference_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.1)
            torch.testing.assert_close(norm.cpu(), reference_norm, atol=2e-6, rtol=2e-5)
            for (_, _, reduced), target in zip(optimizer.grad_buffers, reference.parameters(), strict=True):
                torch.testing.assert_close(reduced, target.grad, atol=2e-6, rtol=2e-5)
            reference_optimizer.step()
            for actual, target in zip(student.parameters(), reference.parameters(), strict=True):
                torch.testing.assert_close(actual.cpu(), target, atol=2e-6, rtol=2e-5)
                state = optimizer.state[actual]
                expected_state = reference_optimizer.state[target]
                assert state['step'].item() == expected_step
                torch.testing.assert_close(state['cpu_master'], target.detach(), atol=2e-6, rtol=2e-5)
                for key in ('exp_avg', 'exp_avg_sq'):
                    torch.testing.assert_close(state[key], expected_state[key], atol=2e-6, rtol=2e-5)
                replicas = [torch.empty_like(actual) for _ in range(4)]
                dist.all_gather(replicas, actual.detach())
                for replica in replicas:
                    torch.testing.assert_close(replica, actual, atol=0, rtol=0)
            assert optimizer.publisher.version == expected_step
            step_reports.append({'step': expected_step, 'adamw_calls_across_ranks': calls.item(),
                                 'grad_norm': norm.item(), 'policy_version': optimizer.publisher.version})
            optimizer.zero_grad()
            reference_optimizer.zero_grad()
        # A single bad rank must poison the REDUCED gradient and make all ranks
        # skip through the production clipping/finite-check path.
        skipped = []
        for bad_value in (float('nan'), float('inf')):
            parameters_before = [p.detach().clone() for p in student.parameters()]
            state_before = copy.deepcopy(optimizer.state_dict())
            version_before = optimizer.publisher.version
            with actor.no_sync():
                output = actor(data[rank:rank + 1].cuda())
                if rank == 3:
                    def poison(gradient):
                        gradient = gradient.clone()
                        gradient.reshape(-1)[0] = bad_value
                        return gradient
                    output.register_hook(poison)
                output.square().mean().backward()
            with patch.object(optimizer.cpu_optimizer, 'step', wraps=optimizer.cpu_optimizer.step) as step, \
                    patch.object(dist, 'broadcast', side_effect=AssertionError('GPU parameter broadcast')):
                norm = DataParallelPPOActor._optimizer_step(learner)
            finite = torch.isfinite(norm).to(dtype=torch.int32).reshape(1)
            dist.all_reduce(finite.to(device))
            assert finite.item() == 0, 'ranks disagree about skipping nonfinite gradients'
            assert step.call_count == 0
            assert optimizer.publisher.version == version_before
            for actual, before in zip(student.parameters(), parameters_before, strict=True):
                torch.testing.assert_close(actual, before, atol=0, rtol=0)
            for index, state in optimizer.state_dict()['state'].items():
                for key, value in state.items():
                    torch.testing.assert_close(value, state_before['state'][index][key], atol=0, rtol=0)
            assert all(p.grad is None for p in student.parameters())
            skipped.append('nan' if bad_value != bad_value else 'inf')
        # Non-owner writable mapping test on a moment, then restore it.
        state = next(iter(optimizer.state.values()))
        before = state['exp_avg'].clone()
        dist.barrier()
        if rank == 3:
            state['exp_avg'].add_(1)
        dist.barrier()
        torch.testing.assert_close(state['exp_avg'], before + 1)
        dist.barrier()  # All readers finish before the writer restores shared bytes.
        if rank == 3:
            state['exp_avg'].copy_(before)
        dist.barrier()
        # Restore checkpoint by copying into mappings, never replacing them.
        checkpoint = copy.deepcopy(optimizer.state_dict())
        pointers = [value.data_ptr() for item in optimizer.state.values() for value in item.values()]
        dist.barrier()  # Each rank must finish taking its checkpoint snapshot.
        if rank == 1:
            state['exp_avg'].add_(10)
        dist.barrier()
        optimizer.publisher.version = 999
        optimizer.load_state_dict(checkpoint)
        assert optimizer.publisher.version == checkpoint['published_version']
        torch.testing.assert_close(state['exp_avg'], before)
        assert pointers == [value.data_ptr() for item in optimizer.state.values() for value in item.values()]

        teachers = []
        for index in range(3):
            store = SharedTeacherStore(f'teacher/{index}')
            model = store.load_model(owner_load, make_model)
            teacher = FSDP(model, device_id=device, use_orig_params=False,
                           sharding_strategy=MultiGPUGradientSync.sharding_strategy(), sync_module_states=True,
                           cpu_offload=CPUOffload(offload_params=False))
            teacher.requires_grad_(False)
            store.bind_fsdp(teacher)
            flat = teacher._handle.flat_param
            assert flat._local_shard.data_ptr() == flat._openmopd_shared_cpu.data_ptr()
            with torch.no_grad():
                first = teacher(data.cuda()).clone()
            torch.cuda.synchronize()
            dist.barrier()
            if rank == 0:
                flat._openmopd_shared_cpu.add_(0.1)
            dist.barrier()
            with torch.no_grad():
                second = teacher(data.cuda())
            assert not torch.equal(first, second), 'H2D did not read shared teacher weights'
            assert flat._local_shard.data_ptr() == flat._openmopd_shared_cpu.data_ptr()
            outputs = [torch.empty_like(second) for _ in range(4)]
            dist.all_gather(outputs, second)
            for output in outputs:
                torch.testing.assert_close(second, output, atol=0, rtol=0)
            teachers.append(teacher)
        inventory = optimizer.store.inventory()
        assert not any(item['model'].endswith('/init') for item in inventory)
        models = sorted({item['model'] for item in inventory})
        dist.barrier()
        if rank == 0:
            print('GPU_SHARED_CPU_PASS ' + json.dumps({'models': models, 'allocations': len(inventory),
                  'bytes': sum(item['nbytes'] for item in inventory), 'ranks': 4,
                  'teacher_h2d_shared': True, 'adam_reference_pass': True,
                  'steps': step_reports, 'nonfinite_skipped_all_ranks': skipped,
                  'student_replicas_bitwise_equal': True,
                  'master_and_moments_match_reference': True}), flush=True)
    finally:
        dist.destroy_process_group()


def rollout_worker():
    """Exercise real worker weight publication and vLLM generation on four ranks."""
    import datetime
    import inspect
    from types import MethodType, SimpleNamespace

    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from transformers import AutoModelForCausalLM

    from verl import DataProto
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.workers.config import HFModelConfig, RolloutConfig
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.workers.multi_gpu import MultiGPUGradientSync, MultiGPUWeightPublisher
    from verl.workers.multi_gpu.shared_adam import SharedCPUAdamW
    from verl.workers.multi_gpu.teacher_store import SharedTeacherStore
    from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout

    device = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=180))
    rank = dist.get_rank()
    assert dist.get_world_size() == 4
    try:
        model_config = HFModelConfig(path=os.environ['OPENMOPD_TEST_MODEL'],
                                     override_config={'attn_implementation': 'eager'})
        store = SharedTeacherStore('student')
        student = store.load_model(
            lambda: AutoModelForCausalLM.from_pretrained(model_config.local_path, torch_dtype=torch.bfloat16),
            lambda: AutoModelForCausalLM.from_config(model_config.hf_config, torch_dtype=torch.bfloat16),
        )
        actor = FSDP(student, device_id=device, use_orig_params=True,
                     sharding_strategy=MultiGPUGradientSync.sharding_strategy(), sync_module_states=True)
        store.finish_initialization()
        optimizer = SharedCPUAdamW(torch.optim.AdamW(actor.parameters(), lr=0.05), 1,
                                   MultiGPUWeightPublisher(1))
        config = RolloutConfig(
            name='vllm', dtype='bfloat16', prompt_length=3, response_length=2, max_model_len=16,
            tensor_model_parallel_size=1, data_parallel_size=1, pipeline_model_parallel_size=1,
            max_num_seqs=1, max_num_batched_tokens=16, enforce_eager=True, free_cache_engine=False,
            enable_chunked_prefill=False, enable_prefix_caching=False, gpu_memory_utilization=0.1,
            load_format='dummy', temperature=0, ignore_eos=True, verify_weight_sync=True,
            multi_gpu={'enable': True, 'cpu_state_mode': 'shared'},
            engine_kwargs={'vllm': {'kv_cache_memory_bytes': 16 * 1024 * 1024}},
        )
        mesh = init_device_mesh('cuda', (4, 1, 1), mesh_dim_names=('dp', 'tp', 'pp'))
        rollout = vLLMRollout(config, model_config, mesh)
        # Bypass Ray scheduling and expensive production model construction only;
        # use the actual worker rollout_mode/generate_sequences/trainer_mode bodies.
        worker = SimpleNamespace(
            config=SimpleNamespace(rollout=config), rollout=rollout,
            actor_module_fsdp=actor, actor_optimizer=optimizer, _is_actor=True, _is_rollout=True,
            _multi_gpu=True, _share_rollout_weights=False, _is_offload_param=False,
            base_sync_done=True, _rollout_weight_sync_count=0,
            generation_config=model_config.generation_config, gen_random_states=torch.cuda.get_rng_state(),
        )
        for name in ('_uses_cpu_adam', '_synchronous_cpu_adam', '_assert_published_policy', 'rollout_mode', 'trainer_mode'):
            setattr(worker, name, MethodType(getattr(ActorRolloutRefWorker, name), worker))
        generate = inspect.unwrap(ActorRolloutRefWorker.generate_sequences)
        prompt = torch.tensor([[4, 5, 6]], device=device)
        fixed_prompts = None
        if '--mopd' in sys.argv:
            import numpy as np
            from unittest.mock import patch
            from verl.single_controller.base.worker_group import WorkerGroup
            from verl.single_controller.base.decorator import dispatch_nd_compute_dataproto

            class Group(WorkerGroup):
                world_size = 4

                def __init__(self):
                    pass

            global_prompts = torch.arange(4, 16).reshape(4, 3)
            global_batch = DataProto.from_dict(tensors={
                'input_ids': global_prompts, 'attention_mask': torch.ones_like(global_prompts),
                'position_ids': torch.arange(3).expand(4, -1),
            }, non_tensors={'domain': np.array(['math', 'code', 'if', 'math'], dtype=object),
                            'sample_id': np.arange(4)})
            # Use production DP splitting; substitute only Ray object transport.
            with patch('verl.utils.ray_utils.parallel_put', side_effect=lambda values, **kwargs: values):
                args, _ = dispatch_nd_compute_dataproto(list(range(4)), 4, Group(), global_batch)
            fixed_prompts = args[0][rank]
            prompt = fixed_prompts.batch['input_ids'].to(device)

        def run_rollout():
            import copy
            prompts = copy.deepcopy(fixed_prompts) if fixed_prompts is not None else DataProto.from_dict(tensors={
                'input_ids': prompt.clone(), 'attention_mask': torch.ones_like(prompt),
                'position_ids': torch.arange(3, device=device).unsqueeze(0),
            })
            result = generate(worker, prompts)
            # Trainer retains original prompt metadata when unioning generated trajectories.
            if fixed_prompts is not None:
                result.non_tensor_batch.update(copy.deepcopy(fixed_prompts.non_tensor_batch))
            return result

        before = run_rollout()
        assert before.meta_info['rollout_policy_version'] == 0
        old_weights = {name: value.detach().clone() for name, value in rollout._get_model().named_parameters()}
        if '--mopd' in sys.argv:
            from test_mopd_closed_loop import run_mopd_step
            run_mopd_step(actor, optimizer, before, model_config, device)
        else:
            with actor.no_sync():
                logits = actor(prompt).logits
                # Different per-rank targets ensure this is a real distributed update.
                torch.nn.functional.cross_entropy(logits[:, -1].float(),
                                                 torch.tensor([7 + rank], device=device)).backward()
            learner = SimpleNamespace(config=SimpleNamespace(grad_clip=1.0), actor_module=actor,
                                      actor_optimizer=optimizer)
            norm = DataParallelPPOActor._optimizer_step(learner)
            assert torch.isfinite(norm)
            assert all(state['step'].item() == 1 for state in optimizer.state.values())
        after = run_rollout()
        assert after.meta_info['rollout_policy_version'] == optimizer.publisher.version == 1
        new_weights = dict(rollout._get_model().named_parameters())
        assert any(not torch.equal(old, new_weights[name]) for name, old in old_weights.items())
        # Check every runtime weight, including fused QKV and gate/up projections.
        source = dict(student.named_parameters())
        packed = {'qkv_proj': ('q_proj', 'k_proj', 'v_proj'), 'gate_up_proj': ('gate_proj', 'up_proj')}
        for name, actual in new_weights.items():
            expected = source.get(name)
            if expected is None:
                for fused, originals in packed.items():
                    if fused in name:
                        expected = torch.cat([source[name.replace(fused, original)] for original in originals])
                        break
            assert expected is not None, f'unchecked vLLM parameter: {name}'
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        # Independently decode with the newly updated HF student.
        actor.eval()
        with torch.no_grad():
            expected = prompt.clone()
            for _ in range(2):
                token = actor(expected).logits[:, -1].argmax(-1, keepdim=True)
                expected = torch.cat((expected, token), dim=-1)
        torch.testing.assert_close(after.batch['responses'].to(device), expected[:, 3:], atol=0, rtol=0)
        for parameter in student.parameters():
            replicas = [torch.empty_like(parameter) for _ in range(4)]
            dist.all_gather(replicas, parameter.detach())
            for replica in replicas:
                torch.testing.assert_close(replica, replicas[0], atol=0, rtol=0)
        if rank == 0:
            print('GPU_ROLLOUT_PASS ' + json.dumps({
                'ranks': 4, 'versions': [before.meta_info['rollout_policy_version'],
                                       after.meta_info['rollout_policy_version']],
                'real_vllm': True, 'all_runtime_weights_match_student': True,
                'next_tokens_match_updated_hf': True, 'adam_step': 1,
                'before_tokens': before.batch['responses'].tolist(),
                'after_tokens': after.batch['responses'].tolist(),
            }), flush=True)
    finally:
        dist.destroy_process_group()


def create_rollout_fixture(path):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(4321)
    config = Qwen2Config(vocab_size=64, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                         max_position_embeddings=32, bos_token_id=1, eos_token_id=2, pad_token_id=0,
                         tie_word_embeddings=False, attention_dropout=0.0)
    Qwen2ForCausalLM(config).to(dtype=torch.bfloat16).save_pretrained(path)
    vocab = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2, '[UNK]': 3,
             **{f'token{i}': i for i in range(4, 64)}}
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab, unk_token='[UNK]')),
                                       pad_token='[PAD]', bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]')
    tokenizer.save_pretrained(path)


def gpu_test():
    import signal
    owner = CPUStateStore.create()
    env = dict(os.environ, **{ENV_NAME: owner.namespace}, OMP_NUM_THREADS='1', TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    fixture = tempfile.TemporaryDirectory(prefix='openmopd-rollout-') if '--rollout' in sys.argv else None
    try:
        if fixture is not None:
            create_rollout_fixture(fixture.name)
            env['OPENMOPD_TEST_MODEL'] = fixture.name
            env['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
        worker_flag = '--rollout-worker' if fixture is not None else '--gpu-worker'
        process = subprocess.Popen([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                                    '--nproc_per_node=4', __file__, worker_flag] + (['--mopd'] if '--mopd' in sys.argv else []), env=env, start_new_session=True)
        try:
            result = process.wait(timeout=600 if fixture is not None else 180)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise AssertionError('shared CPU GPU test timed out') from None
        assert result == 0, f'GPU worker failed: {result}'
    finally:
        owner.unlink()
        if fixture is not None:
            fixture.cleanup()


if __name__ == '__main__':
    if '--ray' in sys.argv:
        ray_test()
    elif '--gpu-worker' in sys.argv:
        gpu_worker()
    elif '--rollout-worker' in sys.argv:
        rollout_worker()
    elif '--gpu' in sys.argv:
        gpu_test()
    else:
        unittest.main()
