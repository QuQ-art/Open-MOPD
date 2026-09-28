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
import pickle
import tempfile
import threading
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP

from verl.workers.teacher_forward_overlap import TeacherForwardOverlap
from verl.workers.teacher_param_prefetch import TeacherHandlePrefetch, after_teacher_parameter_copies


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(64, 16)
        self.head = torch.nn.Linear(16, 64, bias=False)
        self.calls = 0

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids, **kwargs):
        self.calls += 1
        logits = self.head(self.embedding(input_ids))
        return (logits,) if kwargs.get("return_dict") is False else SimpleNamespace(logits=logits)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestTeacherForwardOverlap(unittest.TestCase):
    def test_bounded_prefix_reuse_discard_and_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            dist.init_process_group("nccl", init_method=Path(directory, "init").as_uri(), rank=0, world_size=1)
            try:
                torch.manual_seed(42)
                kwargs = dict(cpu_offload=CPUOffload(offload_params=True), device_id=0)
                model = FSDP(torch.nn.Sequential(*[
                    FSDP(torch.nn.Linear(512, 512, bias=False).to(dtype=torch.bfloat16).eval(), **kwargs)
                    for _ in range(3)
                ]), **kwargs)
                inputs = torch.randn(2, 512, device="cuda", dtype=torch.bfloat16)
                with torch.no_grad():
                    reference = model(inputs).clone()
                    # Three half-MiB shards exceed a one-MiB cap; no hooks may change.
                    originals = [h.pre_unshard for h in model._all_handles]
                    with self.assertRaisesRegex(ValueError, "above"):
                        TeacherHandlePrefetch(model, max_mb=1, num_handles=3)
                    self.assertEqual(originals, [h.pre_unshard for h in model._all_handles])
                    prefetch = TeacherHandlePrefetch(model, max_mb=1, num_handles=2)
                    self.assertEqual(len(prefetch.handles), 2)
                    self.assertEqual(model._all_handles[2].pre_unshard, originals[2])
                    pointers = [x.data_ptr() for x in prefetch.buffers]
                    for _ in range(3):
                        before = prefetch.reuse_count
                        prefetch.start()
                        with self.assertRaises(RuntimeError):
                            prefetch.start()
                        actual = model(inputs)
                        prefetch.assert_consumed(before)
                        prefetch.record_use_done()
                        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                    self.assertEqual((prefetch.copy_count, prefetch.reuse_count), (6, 6))
                    prefetch.start()
                    prefetch.discard()
                    self.assertIsNone(prefetch.pending)
                    before = prefetch.reuse_count
                    prefetch.start()
                    torch.testing.assert_close(model(inputs), reference, rtol=0, atol=0)
                    prefetch.assert_consumed(before)
                    prefetch.record_use_done()
                    self.assertEqual(pointers, [x.data_ptr() for x in prefetch.buffers])
            finally:
                dist.destroy_process_group()

    def test_parameter_copy_trigger_stream_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            dist.init_process_group("nccl", init_method=Path(directory, "init").as_uri(), rank=0, world_size=1)
            try:
                kwargs = dict(cpu_offload=CPUOffload(offload_params=True), device_id=0)
                model = FSDP(torch.nn.Sequential(*[
                    FSDP(torch.nn.Linear(16, 16).to(dtype=torch.bfloat16).eval(), **kwargs)
                    for _ in range(3)
                ]), **kwargs)
                inputs = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)
                with torch.no_grad():
                    expected = model(inputs).clone()
                    submitted = []
                    for index, handle in enumerate(model._all_handles):
                        original = handle.pre_unshard
                        def tracked(index=index, original=original):
                            copied = original()
                            submitted.append(index)
                            return copied
                        handle.pre_unshard = tracked
                    originals = [h.pre_unshard for h in model._all_handles]
                    calls = []
                    def callback():
                        self.assertEqual(torch.cuda.current_stream(), model._pre_unshard_stream)
                        self.assertEqual(submitted, [0, 1, 2])
                        calls.append(torch.cuda.Event())
                        calls[-1].record()
                    with after_teacher_parameter_copies(model, callback):
                        actual = model(inputs)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(originals, [h.pre_unshard for h in model._all_handles])
                    def fail():
                        raise RuntimeError("injected copy-trigger failure")
                    with self.assertRaisesRegex(RuntimeError, "injected copy-trigger"):
                        with after_teacher_parameter_copies(model, fail):
                            model(inputs)
                    torch.cuda.synchronize()
                    self.assertEqual(originals, [h.pre_unshard for h in model._all_handles])
            finally:
                dist.destroy_process_group()

    def test_joint_worker_matches_serial_and_serialization(self):
        from omegaconf import OmegaConf
        from verl import DataProto
        from verl.workers.actor.dp_actor import DataParallelPPOActor
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, RewardModelWorker

        with tempfile.TemporaryDirectory() as directory:
            dist.init_process_group("nccl", init_method=Path(directory, "init").as_uri(), rank=0, world_size=1)
            teacher = None
            try:
                torch.manual_seed(31)
                student_model = FSDP(TinyLM().to(dtype=torch.bfloat16).eval(), device_id=torch.cuda.current_device())
                teacher_model = FSDP(TinyLM().to(dtype=torch.bfloat16).eval(),
                                     cpu_offload=CPUOffload(offload_params=True), device_id=torch.cuda.current_device())
                actor = SimpleNamespace(actor_module=student_model, config=SimpleNamespace(entropy_checkpointing=False),
                                        use_remove_padding=False, use_fused_kernels=False,
                                        use_ulysses_sp=False, device_name="cuda")
                for name in ("_log_prob_model_forward", "_forward_micro_batch", "compute_log_prob"):
                    setattr(actor, name, MethodType(getattr(DataParallelPPOActor, name), actor))
                teacher_config = OmegaConf.create({
                    "use_dynamic_bsz": False, "micro_batch_size_per_gpu": 2, "model": {"path": "Math"},
                })
                teacher = SimpleNamespace(
                    reward_module=teacher_model, use_remove_padding=False, use_fused_kernels=False,
                    _do_switch_chat_template=False, world_size=1, ulysses_sequence_parallel_size=1,
                    ulysses_sharding_manager=nullcontext(), _teacher_forward_overlap=None, config=teacher_config,
                )
                for name in ("_forward_model_logits", "_forward_micro_batch", "_compute_entropy_safe",
                             "_compute_teacher_top_k_log_probs", "prepare_forward_overlap", "compute_rm_score",
                             "_compute_rm_score"):
                    setattr(teacher, name, MethodType(getattr(RewardModelWorker, name), teacher))
                teacher._teacher_handle_prefetch = TeacherHandlePrefetch(teacher_model, max_mb=1)
                code = SimpleNamespace(
                    reward_module=FSDP(TinyLM().to(dtype=torch.bfloat16).eval(), device_id=0,
                                       cpu_offload=CPUOffload(offload_params=True)),
                    use_remove_padding=False, use_fused_kernels=False, _do_switch_chat_template=False,
                    world_size=1, ulysses_sequence_parallel_size=1, ulysses_sharding_manager=nullcontext(),
                    config=OmegaConf.create({"use_dynamic_bsz": False, "micro_batch_size_per_gpu": 2,
                                             "model": {"path": "Code"}}),
                )
                for name in ("_forward_model_logits", "_forward_micro_batch", "_compute_entropy_safe",
                             "_compute_teacher_top_k_log_probs", "_compute_rm_score"):
                    setattr(code, name, MethodType(getattr(RewardModelWorker, name), code))
                code._teacher_handle_prefetch = TeacherHandlePrefetch(
                    code.reward_module, max_mb=1, nvtx_name="openmopd::io::h2d::code_teacher_prefetch")
                worker_config = OmegaConf.create({"rollout": {
                    "teacher_forward_overlap": True, "teacher_param_prefetch": True, "code_teacher_param_prefetch": True,
                    "log_prob_micro_batch_size_per_gpu": 2, "log_prob_max_token_len_per_gpu": 8,
                    "log_prob_use_dynamic_bsz": False, "temperature": 1.0, "log_prob_top_k": 4,
                }})
                worker = SimpleNamespace(
                    actor=actor, _is_actor=True, _is_offload_param=False, world_size=1,
                    ulysses_sharding_manager=nullcontext(), get_fused_worker_by_name=lambda name: {"rm": teacher, "mt_rm_1": code}[name],
                    config=worker_config,
                )
                for name in ("compute_log_prob", "_compute_log_prob", "compute_log_prob_and_teacher"):
                    setattr(worker, name, MethodType(getattr(ActorRolloutRefWorker, name), worker))
                inputs = torch.randint(0, 64, (2, 8))

                def batch():
                    result = DataProto.from_dict({"input_ids": inputs.clone(), "responses": inputs[:, -5:].clone(),
                                                "attention_mask": torch.ones(2, 8, dtype=torch.long),
                                                "position_ids": torch.arange(8).unsqueeze(0).expand(2, -1).clone(),
                                                "response_mask": torch.ones(2, 5, dtype=torch.long)},
                                               meta_info={"reward_mode": "mt_opd", "log_prob_top_k": 4})
                    # Real Ray input has consolidated storage with device=None.
                    return pickle.loads(pickle.dumps(result))

                original = batch()
                self.assertTrue(original.batch.is_consolidated())
                # Construct the serial reference independently of the joint
                # container path, retaining the correct student IDs.
                reference_data = DataProto.from_dict(dict(original.batch.items()), meta_info=dict(original.meta_info))
                student = worker._compute_log_prob(reference_data, teacher_forward_callback=lambda: None)
                reference_data.union(student)
                expected = student.union(teacher.compute_rm_score(reference_data)).batch.cpu()
                code_expected = code._compute_rm_score(reference_data).batch.cpu()
                for trigger in ("output_head", "math_h2d_done"):
                    worker.config.rollout.code_teacher_param_prefetch_trigger = trigger
                    actual = worker.compute_log_prob_and_teacher(batch())
                    # Match the real worker boundary as well as the direct outputs.
                    restored = pickle.loads(pickle.dumps(actual))
                    for key in expected.keys():
                        torch.testing.assert_close(actual.batch[key].cpu(), expected[key], rtol=0, atol=0)
                        torch.testing.assert_close(restored.batch[key].cpu(), expected[key], rtol=0, atol=0)
                    code_data = DataProto.from_dict(dict(batch().batch.items()), meta_info=dict(actual.meta_info))
                    code_data.meta_info.update(reward_mode="mt_opd", log_prob_top_k=4)
                    code_actual = code._compute_rm_score(code_data.union(actual))
                    for key in code_expected.keys():
                        torch.testing.assert_close(code_actual.batch[key].cpu(), code_expected[key], rtol=0, atol=0)
                self.assertEqual(code._teacher_handle_prefetch.copy_count, 2)
                self.assertEqual(code._teacher_handle_prefetch.reuse_count, 2)
                self.assertEqual(teacher._teacher_handle_prefetch.reuse_count, 2)
                normal_score = teacher._compute_rm_score
                def fail_postprocess(*args, **kwargs):
                    raise RuntimeError("injected Math postprocess failure")
                teacher._compute_rm_score = fail_postprocess
                with self.assertRaisesRegex(RuntimeError, "injected Math"):
                    worker.compute_log_prob_and_teacher(batch())
                self.assertIsNone(code._teacher_handle_prefetch.pending)
                teacher._compute_rm_score = normal_score
                # A discarded copy must not poison counters on the next scoring call.
                recovered = worker.compute_log_prob_and_teacher(batch())
                code_data = DataProto.from_dict(dict(batch().batch.items()), meta_info=dict(recovered.meta_info))
                code_data.meta_info.update(reward_mode="mt_opd", log_prob_top_k=4)
                code_actual = code._compute_rm_score(code_data.union(recovered))
                for key in code_expected.keys():
                    torch.testing.assert_close(code_actual.batch[key].cpu(), code_expected[key], rtol=0, atol=0)
                self.assertEqual(code._teacher_handle_prefetch.copy_count, 4)
                self.assertEqual(code._teacher_handle_prefetch.reuse_count, 3)
                self.assertIsNotNone(code._teacher_handle_prefetch.last_use_done)
                self.assertFalse(teacher_model.get_output_embeddings()._forward_pre_hooks)
                worker.config.rollout.log_prob_micro_batch_size_per_gpu = 1
                with self.assertRaisesRegex(ValueError, "student micro-batch"):
                    worker.compute_log_prob_and_teacher(batch())
                worker.config.rollout.log_prob_micro_batch_size_per_gpu = 2
                teacher.config.micro_batch_size_per_gpu = 1
                with self.assertRaisesRegex(ValueError, "teacher micro-batch"):
                    worker.compute_log_prob_and_teacher(batch())
            finally:
                if teacher is not None and teacher._teacher_forward_overlap is not None:
                    teacher._teacher_forward_overlap.close()
                dist.destroy_process_group()

    def test_split_scoring_matches_serial_with_fsdp_prefetch(self):
        from verl.workers.fsdp_workers import RewardModelWorker

        with tempfile.TemporaryDirectory() as directory:
            dist.init_process_group("nccl", init_method=Path(directory, "init").as_uri(), rank=0, world_size=1)
            runner = TeacherForwardOverlap(torch.device("cuda", torch.cuda.current_device()), "test")
            try:
                torch.manual_seed(17)
                raw = TinyLM().to(dtype=torch.bfloat16).eval()
                model = FSDP(raw, cpu_offload=CPUOffload(offload_params=True), device_id=torch.cuda.current_device())
                worker = SimpleNamespace(reward_module=model, use_remove_padding=False, use_fused_kernels=False)
                for name in (
                    "_forward_model_logits", "_forward_micro_batch", "_compute_entropy_safe",
                    "_compute_teacher_top_k_log_probs",
                ):
                    setattr(worker, name, MethodType(getattr(RewardModelWorker, name), worker))
                mb = {
                    "input_ids": torch.randint(0, 64, (1, 8), device="cuda"),
                    "attention_mask": torch.ones(1, 8, dtype=torch.long, device="cuda"),
                    "position_ids": torch.arange(8, device="cuda").unsqueeze(0),
                    "responses": torch.randint(0, 64, (1, 5), device="cuda"),
                }
                ids = torch.arange(4, device="cuda").expand(1, 5, 4).clone()
                # Lazy-init FSDP and establish a serial reference before wrapping pre_unshard.
                worker._forward_micro_batch(mb, student_top_k_ids=ids, compute_entropy=True, top_k=4)
                prefetch = TeacherHandlePrefetch(model, max_mb=1)
                staged_ptr = prefetch.staged.data_ptr()
                for i, strategy in enumerate(("only_stu", "union", "intersection", "top_p_intersec")):
                    with self.subTest(strategy=strategy):
                        kwargs = dict(student_top_k_ids=ids, compute_entropy=True, top_k=4,
                                      strategy=strategy, teacher_temperature=0.7 if i % 2 else 1.3)
                        expected = worker._forward_micro_batch(mb, **kwargs)
                        if runner.last_done is not None:
                            prefetch.stream.wait_event(runner.last_done)
                        prefetch.start()
                        ready = torch.cuda.Event()
                        ready.record()
                        calls = raw.calls
                        entered = threading.Event()

                        def forward():
                            self.assertEqual(torch.cuda.current_stream(), runner.stream)
                            entered.set()
                            return worker._forward_model_logits(mb)

                        runner.start(forward, ready)
                        self.assertTrue(entered.wait(timeout=10))
                        # Student-side IDs are produced independently of the teacher model.
                        live_ids = ids.clone()
                        logits = runner.finish()
                        prefetch.assert_consumed(previous_reuse_count=i)
                        kwargs["student_top_k_ids"] = live_ids
                        actual = worker._forward_micro_batch(mb, precomputed_logits=logits, **kwargs)
                        self.assertEqual(raw.calls, calls + 1, "postprocess must not run the teacher model again")
                        self.assertEqual(prefetch.staged.data_ptr(), staged_ptr)
                        for reference, result in zip(expected, actual):
                            if reference is None:
                                self.assertIsNone(result)
                            else:
                                torch.testing.assert_close(result, reference, rtol=0, atol=0)
                self.assertEqual(prefetch.copy_count, 4)
                self.assertEqual(prefetch.reuse_count, 4)
            finally:
                runner.close()
                dist.destroy_process_group()

    def test_single_slot_and_exception_cleanup(self):
        runner = TeacherForwardOverlap(torch.device("cuda", torch.cuda.current_device()), "test")
        release = threading.Event()
        ready = torch.cuda.Event()
        ready.record()
        try:
            def forward():
                if not release.wait(timeout=10):
                    raise TimeoutError("test release was not signaled")
                return torch.ones(4, device="cuda")

            runner.start(forward, ready)
            with self.assertRaisesRegex(RuntimeError, "not been collected"):
                runner.start(forward, ready)
            release.set()
            torch.testing.assert_close(runner.finish(), torch.ones(4, device="cuda"))

            def fail():
                raise ValueError("teacher failure")

            runner.start(fail, ready)
            with self.assertRaisesRegex(ValueError, "teacher failure"):
                runner.finish()
            self.assertIsNone(runner.future)
            runner.start(lambda: torch.full((4,), 2.0, device="cuda"), ready)
            torch.testing.assert_close(runner.finish(), torch.full((4,), 2.0, device="cuda"))
        finally:
            release.set()
            runner.close()
