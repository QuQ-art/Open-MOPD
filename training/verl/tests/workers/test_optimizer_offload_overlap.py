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
import asyncio
import inspect
import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

from verl.workers.optimizer_offload_overlap import OptimizerOffloadOverlap


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestOptimizerOffloadOverlap(unittest.TestCase):
    def test_consecutive_updates_match_and_reuse_pinned_buffers(self):
        self._check_updates(torch.float32, torch.optim.AdamW)

    def test_bf16_stochastic_updates_match(self):
        from verl.utils.bf16_optimizer import BF16StochasticAdamW

        self._check_updates(torch.bfloat16, BF16StochasticAdamW)

    def _check_updates(self, dtype, optimizer_type):
        torch.manual_seed(42)
        reference = torch.nn.Parameter(torch.randn(768, 768, device="cuda", dtype=dtype))
        actual = torch.nn.Parameter(reference.detach().clone())
        baseline = optimizer_type([reference], lr=0.01, foreach=False)
        optimizer = optimizer_type([actual], lr=0.01, foreach=False)
        overlap = OptimizerOffloadOverlap(actual.device, chunk_mb=1)
        self.addCleanup(overlap.close)
        pointers = None
        for _ in range(3):
            # Production reload follows finish(), on the current compute stream.
            for key, value in optimizer.state[actual].items():
                if isinstance(value, torch.Tensor) and key != "step":
                    optimizer.state[actual][key] = value.to(actual.device, non_blocking=True)
            gradient = torch.randn_like(actual)
            reference.grad = gradient.clone()
            actual.grad = gradient.clone()
            rng = torch.cuda.get_rng_state()
            baseline.step()
            torch.cuda.set_rng_state(rng)
            optimizer.step()
            overlap.prepare(optimizer)
            self.assertTrue(overlap.pending)
            with self.assertRaisesRegex(RuntimeError, "previous optimizer offload"):
                overlap.prepare(optimizer)
            overlap.start()
            event = overlap.future
            overlap.start()
            self.assertIs(overlap.future, event)
            # Rollout is allowed to read updated parameters during D2H.
            expected = reference.detach() @ reference.detach().T
            observed = actual.detach() @ actual.detach().T
            self.assertEqual(optimizer.state[actual]["exp_avg"].device.type, "cuda")
            overlap.finish()
            self.assertFalse(overlap.pending)
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            for key in ("exp_avg", "exp_avg_sq"):
                cpu_state = optimizer.state[actual][key]
                self.assertTrue(cpu_state.is_pinned())
                torch.testing.assert_close(cpu_state, baseline.state[reference][key].cpu(), rtol=0, atol=0)
            current = [x.data_ptr() for x in overlap.buffers.values()]
            if pointers is not None:
                self.assertEqual(current, pointers)
            pointers = current

    def test_finish_without_rollout_and_producer_stream_dependency(self):
        parameter = torch.nn.Parameter(torch.zeros(1024, device="cuda"))
        optimizer = torch.optim.AdamW([parameter])
        producer = torch.cuda.Stream()
        overlap = OptimizerOffloadOverlap(parameter.device, chunk_mb=1)
        self.addCleanup(overlap.close)
        producer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(producer):
            state = torch.empty_like(parameter)
            torch.cuda._sleep(10_000_000)
            state.fill_(7)
            optimizer.state[parameter]["exp_avg"] = state
            optimizer.state[parameter]["step"] = 12
            overlap.prepare(optimizer)
        # Checkpoint/no-next-rollout path must launch and wait, without start().
        overlap.finish()
        torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], torch.full((1024,), 7.0))
        self.assertEqual(optimizer.state[parameter]["step"], 12)
        self.assertFalse(overlap.entries)
        overlap.finish()

    def test_empty_cpu_state_and_replaced_state(self):
        parameter = torch.nn.Parameter(torch.zeros(8, device="cuda"))
        optimizer = torch.optim.AdamW([parameter])
        overlap = OptimizerOffloadOverlap(parameter.device, chunk_mb=1)
        self.addCleanup(overlap.close)
        overlap.prepare(optimizer)
        overlap.start()
        overlap.finish()
        self.assertFalse(overlap.pending)
        optimizer.state[parameter]["step"] = torch.tensor(1.0)
        overlap.prepare(optimizer)
        self.assertFalse(overlap.pending)
        source = torch.ones_like(parameter)
        optimizer.state[parameter]["exp_avg"] = source
        overlap.prepare(optimizer)
        optimizer.state[parameter]["exp_avg"] = source.clone()
        with self.assertRaisesRegex(RuntimeError, "state changed"):
            overlap.finish()
        optimizer.state[parameter]["exp_avg"] = source
        overlap.finish()
        torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], torch.ones(8))

    def test_background_failure_keeps_gpu_state_for_retry(self):
        parameter = torch.nn.Parameter(torch.zeros(8, device="cuda"))
        optimizer = torch.optim.AdamW([parameter])
        source = torch.ones_like(parameter)
        optimizer.state[parameter]["exp_avg"] = source
        overlap = OptimizerOffloadOverlap(parameter.device)
        self.addCleanup(overlap.close)
        overlap.prepare(optimizer)
        with patch.object(overlap, "_copy_chunks", side_effect=RuntimeError("injected DMA failure")):
            overlap.start()
            with self.assertRaisesRegex(RuntimeError, "injected DMA failure"):
                overlap.finish()
        self.assertIs(optimizer.state[parameter]["exp_avg"], source)
        overlap.finish()
        torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], torch.ones(8))

    def test_worker_rollout_failure_and_checkpoint_boundaries(self):
        from verl.workers.fsdp_workers import ActorRolloutRefWorker

        parameter = torch.nn.Parameter(torch.zeros(1024, device="cuda"))
        optimizer = torch.optim.AdamW([parameter])
        optimizer.state[parameter]["exp_avg"] = torch.ones_like(parameter)
        overlap = OptimizerOffloadOverlap(parameter.device, chunk_mb=1)
        self.addCleanup(overlap.close)
        overlap.prepare(optimizer)
        order = []

        async def rollout_mode():
            self.assertTrue(overlap.pending)
            self.assertIsNone(overlap.future)
            order.append("weights_and_kv_ready")

        def generate_sequences(**kwargs):
            self.assertEqual(order, ["weights_and_kv_ready"])
            self.assertIsNotNone(overlap.future)
            raise RuntimeError("injected rollout failure")

        worker = SimpleNamespace(
            _is_actor=True, _is_rollout=True, _is_offload_param=False,
            _optimizer_offload_overlap=overlap,
            generation_config=SimpleNamespace(eos_token_id=2, pad_token_id=0),
            rollout_mode=rollout_mode, rollout=SimpleNamespace(generate_sequences=generate_sequences),
        )
        worker._finish_optimizer_offload = MethodType(ActorRolloutRefWorker._finish_optimizer_offload, worker)
        prompts = SimpleNamespace(meta_info={})
        prompts.to = lambda device: prompts
        loop = asyncio.new_event_loop()
        self.addCleanup(loop.close)
        with patch("verl.workers.fsdp_workers.get_event_loop", return_value=loop):
            with self.assertRaisesRegex(RuntimeError, "injected rollout failure"):
                inspect.unwrap(ActorRolloutRefWorker.generate_sequences)(worker, prompts)
        self.assertFalse(overlap.pending)
        torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], torch.ones(1024))

        def checkpoint_boundary(**kwargs):
            self.assertFalse(overlap.pending)
            torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], torch.ones(1024))
            raise RuntimeError("checkpoint boundary reached")

        worker.checkpoint_manager = SimpleNamespace(
            save_checkpoint=checkpoint_boundary, load_checkpoint=checkpoint_boundary
        )
        for method in (ActorRolloutRefWorker.save_checkpoint, ActorRolloutRefWorker.load_checkpoint):
            optimizer.state[parameter]["exp_avg"] = torch.ones_like(parameter)
            overlap.prepare(optimizer)
            with self.assertRaisesRegex(RuntimeError, "checkpoint boundary reached"):
                inspect.unwrap(method)(worker, local_path="unused")
