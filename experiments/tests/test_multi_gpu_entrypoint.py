"""CPU-only acceptance tests, runnable with unittest without training dependencies."""
import importlib.util
import os
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = ROOT / 'training/verl/verl/workers/config/multi_gpu.py'
spec = importlib.util.spec_from_file_location('multi_gpu_config', VALIDATOR)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MultiGPUEntrypointTests(unittest.TestCase):
    def launch(self, *overrides, **env):
        environment = dict(os.environ)
        environment.pop('CUDA_VISIBLE_DEVICES', None)
        environment.update(env)
        return subprocess.run(['bash', str(ROOT / 'r2.sh'), '--dry-run', *overrides],
                              env=environment, capture_output=True, text=True)

    def test_single_gpu_preserves_defaults(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ('multi_gpu.enable=false', 'share_weights=true',
                       'student_teacher_pipeline=true', 'pipeline_overlap=true'):
            self.assertIn(option, result.stdout)

    def test_multi_gpu_defaults(self):
        for count in ('2', '4'):
            result = self.launch(OPENMOPD_N_GPUS_PER_NODE=count)
            self.assertEqual(result.returncode, 0, result.stderr)
            for option in ('multi_gpu.enable=true', 'multi_gpu.gradient_sync=cpu_reduce', 'share_weights=false',
                           'student_teacher_pipeline=false', 'pipeline_overlap=false'):
                self.assertIn(option, result.stdout)

    def test_topology_override_selects_multi_defaults(self):
        result = self.launch('trainer.n_gpus_per_node=2')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('multi_gpu.enable=true', result.stdout)

    def test_invalid_final_overrides(self):
        cases = {
            'actor_rollout_ref.rollout.multi_gpu.mode=sharded_dp': 'only mode=replicated_dp',
            'actor_rollout_ref.rollout.multi_gpu.mode=pipeline_parallel': 'only mode=replicated_dp',
            'actor_rollout_ref.rollout.multi_gpu.optimizer_owner_rank=2': 'optimizer_owner_rank',
            'actor_rollout_ref.rollout.multi_gpu.optimizer_owner_rank=-1': 'optimizer_owner_rank',
            'actor_rollout_ref.rollout.multi_gpu.synchronous_update=false': 'synchronous_update',
            'actor_rollout_ref.rollout.multi_gpu.gradient_sync=none': 'gradient_sync',
            'actor_rollout_ref.rollout.multi_gpu.gradient_sync=auto_all_reduce': 'requires gradient_sync=cpu_reduce',
            'actor_rollout_ref.rollout.multi_gpu.cpu_state_mode=invalid': 'cpu_state_mode',
            'actor_rollout_ref.rollout.pipeline_overlap=true': 'pipeline_overlap',
            'actor_rollout_ref.rollout.tensor_model_parallel_size=2': 'tensor_model_parallel_size',
            'data.train_batch_size=3': 'divisible',
            'data.token_batching.enabled=true': 'token_batching.enabled',
            'trainer.nnodes=2': 'one node',
        }
        for override, message in cases.items():
            with self.subTest(override=override):
                result = self.launch(override, OPENMOPD_N_GPUS_PER_NODE='2')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertNotIn('-m verl.trainer.main_ppo', result.stdout)

    def test_sequential_teacher_slots(self):
        result = self.launch('actor_rollout_ref.rollout.teacher_layer_pipeline=true',
                             OPENMOPD_N_GPUS_PER_NODE='4')
        self.assertEqual(result.returncode, 0, result.stderr)
        for override, message in (
            ('actor_rollout_ref.rollout.teacher_param_prefetch=true', 'teacher_param_prefetch'),
            ('actor_rollout_ref.rollout.teacher_forward_overlap=true', 'teacher_forward_overlap'),
            ('reward_model.use_dynamic_bsz=true', 'fixed micro-batches'),
        ):
            result = self.launch('actor_rollout_ref.rollout.teacher_layer_pipeline=true', override,
                                 OPENMOPD_N_GPUS_PER_NODE='4')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(message, result.stderr)

    def test_visibility(self):
        result = self.launch(OPENMOPD_N_GPUS_PER_NODE='2', CUDA_VISIBLE_DEVICES='0')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('CUDA_VISIBLE_DEVICES', result.stderr)

    def test_requires_multiple_gpus(self):
        result = self.launch(MULTI_GPU_ENABLED='true')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('greater than 1', result.stderr)

    def test_separate_states_and_nonzero_owner(self):
        result = self.launch(OPENMOPD_N_GPUS_PER_NODE='2', MULTI_GPU_CPU_STATE_MODE='separate',
                             MULTI_GPU_OPTIMIZER_OWNER_RANK='1')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_final_override_can_repair_environment(self):
        result = self.launch('actor_rollout_ref.rollout.share_weights=false',
                             OPENMOPD_N_GPUS_PER_NODE='2', SHARE_STUDENT_WEIGHTS='true')
        self.assertEqual(result.returncode, 0, result.stderr)


class MultiGPUComponentTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch is unavailable in this interpreter')
        self.torch = torch
        import sys
        sys.path.insert(0, str(ROOT / 'training/verl'))
        import verl.workers.multi_gpu as components
        self.components = components

    def test_owner_only_allocates_state_and_publishes(self):
        from unittest.mock import patch
        m = self.components
        for rank in (0, 1):
            parameter = self.torch.nn.Parameter(self.torch.ones(2))
            base = self.torch.optim.AdamW([parameter], lr=0.1)
            with patch.object(m.dist, 'get_rank', return_value=rank), \
                    patch.object(m.dist, 'broadcast') as broadcast, \
                    patch.object(m.dist, 'all_reduce'):
                optimizer = m.MultiGPUOptimizer(base, m.MultiGPUStateStore('shared', 1), m.MultiGPUWeightPublisher(1))
                parameter.grad = self.torch.ones_like(parameter)
                optimizer.step()
                self.assertEqual(bool(base.state), rank == 1)
                self.assertEqual(optimizer.publisher.version, 1)
                broadcast.assert_called_once()
                self.assertEqual(broadcast.call_args.kwargs['src'], 1)
                optimizer.zero_grad()
                self.assertIsNone(parameter.grad)

    def test_policy_version_mismatch_is_rejected(self):
        from unittest.mock import patch
        m = self.components
        parameter = self.torch.nn.Parameter(self.torch.ones(2))
        def mismatch(version, **kwargs):
            version[0] += 1
        with patch.object(m.dist, 'broadcast'), patch.object(m.dist, 'all_reduce', side_effect=mismatch):
            with self.assertRaisesRegex(RuntimeError, 'policy version mismatch'):
                m.MultiGPUWeightPublisher(0).publish([parameter])

    def test_separate_state_is_allocated_on_each_rank(self):
        from unittest.mock import patch
        m = self.components
        with patch.object(m.dist, 'get_rank', return_value=1):
            self.assertTrue(m.MultiGPUStateStore('separate', 0).owns_state)


if __name__ == '__main__':
    unittest.main()
