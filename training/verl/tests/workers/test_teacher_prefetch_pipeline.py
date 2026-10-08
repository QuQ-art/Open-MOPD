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
import ast
import pickle
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP



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


def worker_method(class_name, method_name):
    """Load a production control method without initializing Ray/vLLM/CUDA."""
    source = Path(__file__).parents[2] / "verl/workers/fsdp_workers.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    method.decorator_list = []
    namespace = {"DataProto": object, "torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[method_name]


class TestTeacherPrefetchControl(unittest.TestCase):
    def test_initialization_wires_both_teachers_without_forward(self):
        teacher = SimpleNamespace(use_remove_padding=False, use_fused_kernels=False,
                                  prepare_param_prefetch=Mock(), forward=Mock())
        code = SimpleNamespace(prepare_param_prefetch=Mock(), _teacher_handle_prefetch=object(), forward=Mock())
        config = SimpleNamespace(teacher_param_prefetch_max_mb=6144, teacher_param_prefetch_handles=37,
                                 code_teacher_param_prefetch_max_mb=6144, code_teacher_param_prefetch_handles=37)
        worker = SimpleNamespace(world_size=1, ulysses_sequence_parallel_size=1,
                                 config=SimpleNamespace(rollout=config),
                                 get_fused_worker_by_name=lambda name: {"rm": teacher, "mt_rm_1": code}[name])
        prepare = worker_method("ActorRolloutRefWorker", "_prepare_teacher_prefetch")
        prepare(worker)
        teacher.prepare_param_prefetch.assert_called_once_with(6144, num_handles=37)
        code.prepare_param_prefetch.assert_called_once_with(
            6144, nvtx_name="openmopd::io::h2d::code_teacher_prefetch", num_handles=37)
        self.assertIs(teacher._next_teacher_prefetch, code._teacher_handle_prefetch)
        teacher.forward.assert_not_called()
        code.forward.assert_not_called()
        worker.get_fused_worker_by_name = lambda name: teacher if name == "rm" else None
        with self.assertRaisesRegex(ValueError, "colocated Math and Code"):
            prepare(worker)

    def test_repeated_math_start_preserves_full_prefetch_configuration(self):
        prefetch = SimpleNamespace(max_bytes=6144 * 1024 * 1024, num_handles=37,
                                   nvtx_name="openmopd::io::h2d::teacher_prefetch", start=Mock())
        worker = SimpleNamespace(_teacher_handle_prefetch=prefetch)
        worker.prepare_param_prefetch = MethodType(
            worker_method("RewardModelWorker", "prepare_param_prefetch"), worker)
        start = worker_method("RewardModelWorker", "start_param_prefetch")
        for _ in range(2):
            start(worker, 6144, num_handles=37)
        self.assertEqual(prefetch.start.call_count, 2)
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            start(worker, 6144, num_handles=1)
        self.assertEqual(prefetch.start.call_count, 2)

    def test_student_failure_discards_only_enabled_prefetch(self):
        prefetch = Mock()
        worker = SimpleNamespace(config=SimpleNamespace(rollout={"teacher_param_prefetch": True}),
                                 get_fused_worker_by_name=lambda name: SimpleNamespace(_teacher_handle_prefetch=prefetch),
                                 _compute_log_prob=Mock(side_effect=RuntimeError("student failed")))
        score = worker_method("ActorRolloutRefWorker", "compute_log_prob")
        with self.assertRaisesRegex(RuntimeError, "student failed"):
            score(worker, object())
        prefetch.discard.assert_called_once()
        prefetch.reset_mock()
        worker.config.rollout["teacher_param_prefetch"] = False
        with self.assertRaisesRegex(RuntimeError, "student failed"):
            score(worker, object())
        prefetch.discard.assert_not_called()

    def test_math_failure_drains_both_and_preserves_error(self):
        order = []
        worker = SimpleNamespace(
            _teacher_handle_prefetch=SimpleNamespace(discard=lambda: order.append("math")),
            _next_teacher_prefetch=SimpleNamespace(discard=lambda: order.append("code")),
            _compute_rm_score_impl=Mock(side_effect=ValueError("postprocess failed")))
        score = worker_method("RewardModelWorker", "_compute_rm_score")
        with patch.object(torch.cuda, "synchronize", side_effect=lambda: order.append("drain")):
            with self.assertRaisesRegex(ValueError, "postprocess failed"):
                score(worker, object())
        self.assertEqual(order, ["drain", "math", "code"])
        worker._teacher_handle_prefetch = worker._next_teacher_prefetch = None
        with patch.object(torch.cuda, "synchronize") as synchronize:
            with self.assertRaisesRegex(ValueError, "postprocess failed"):
                score(worker, object())
            synchronize.assert_not_called()


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestTeacherPrefetchPipeline(unittest.TestCase):
    def test_bounded_prefix_reuse_discard_and_cap(self):
        from verl.workers.teacher_param_prefetch import TeacherHandlePrefetch

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
        from verl.workers.teacher_param_prefetch import after_teacher_parameter_copies

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

    def test_sequential_workers_match_serial_and_serialization(self):
        from verl.workers.teacher_param_prefetch import TeacherHandlePrefetch

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
                    ulysses_sharding_manager=nullcontext(), config=teacher_config,
                )
                for name in ("_forward_model_logits", "_forward_micro_batch", "_compute_entropy_safe",
                             "_compute_teacher_top_k_log_probs", "prepare_param_prefetch", "start_param_prefetch", "compute_rm_score",
                             "_compute_rm_score", "_compute_rm_score_impl"):
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
                             "_compute_teacher_top_k_log_probs", "_compute_rm_score", "_compute_rm_score_impl"):
                    setattr(code, name, MethodType(getattr(RewardModelWorker, name), code))
                code._teacher_handle_prefetch = TeacherHandlePrefetch(
                    code.reward_module, max_mb=1, nvtx_name="openmopd::io::h2d::code_teacher_prefetch")
                worker_config = OmegaConf.create({"rollout": {
                    "teacher_param_prefetch": False, "teacher_param_prefetch_max_mb": 1, "teacher_param_prefetch_handles": 1,
                    "log_prob_micro_batch_size_per_gpu": 2, "log_prob_max_token_len_per_gpu": 8,
                    "log_prob_use_dynamic_bsz": False, "temperature": 1.0, "log_prob_top_k": 4,
                }})
                worker = SimpleNamespace(
                    actor=actor, _is_actor=True, _is_offload_param=False, world_size=1,
                    ulysses_sharding_manager=nullcontext(), get_fused_worker_by_name=lambda name: {"rm": teacher, "mt_rm_1": code}[name],
                    config=worker_config,
                )
                for name in ("compute_log_prob", "_compute_log_prob", "_prepare_teacher_prefetch"):
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
                student = worker._compute_log_prob(reference_data)
                reference_data.union(student)
                expected = student.union(teacher.compute_rm_score(reference_data)).batch.cpu()
                code_expected = code._compute_rm_score(reference_data).batch.cpu()
                worker.config.rollout.teacher_param_prefetch = True
                teacher._next_teacher_prefetch = code._teacher_handle_prefetch
                order = []
                original_student_forward = student_model.module.forward
                original_math_forward = teacher_model.module.forward

                def student_forward(*args, **kwargs):
                    order.append("student")
                    return original_student_forward(*args, **kwargs)

                def math_forward(*args, **kwargs):
                    self.assertEqual(order[-1], "student_done")
                    return original_math_forward(*args, **kwargs)

                student_model.module.forward = student_forward
                teacher_model.module.forward = math_forward

                def score():
                    data = batch()
                    # Same separate RPC order as the trainer, with no joint forward.
                    student_result = worker.compute_log_prob(data)
                    order.append("student_done")
                    self.assertIsNotNone(teacher._teacher_handle_prefetch.pending)
                    result = student_result.union(teacher.compute_rm_score(data.union(student_result)))
                    self.assertIsNotNone(code._teacher_handle_prefetch.pending)
                    return result

                for _ in range(2):
                    actual = score()
                    restored = pickle.loads(pickle.dumps(actual))
                    for key in expected.keys():
                        torch.testing.assert_close(actual.batch[key].cpu(), expected[key], rtol=0, atol=0, msg=key)
                        torch.testing.assert_close(restored.batch[key].cpu(), expected[key], rtol=0, atol=0, msg=key)
                    code_data = batch().union(actual)
                    code_actual = code._compute_rm_score(code_data)
                    for key in code_expected.keys():
                        torch.testing.assert_close(code_actual.batch[key].cpu(), code_expected[key], rtol=0, atol=0, msg=key)
                self.assertEqual(code._teacher_handle_prefetch.copy_count, 2)
                self.assertEqual(code._teacher_handle_prefetch.reuse_count, 2)
                self.assertEqual(teacher._teacher_handle_prefetch.reuse_count, 2)
                normal_entropy = teacher._compute_entropy_safe

                def fail_postprocess(*args, **kwargs):
                    raise RuntimeError("injected Math postprocess failure")

                teacher._compute_entropy_safe = fail_postprocess
                with self.assertRaisesRegex(RuntimeError, "injected Math"):
                    score()
                self.assertIsNone(code._teacher_handle_prefetch.pending)
                teacher._compute_entropy_safe = normal_entropy
                recovered = score()
                code_actual = code._compute_rm_score(batch().union(recovered))
                for key in code_expected.keys():
                    torch.testing.assert_close(code_actual.batch[key].cpu(), code_expected[key], rtol=0, atol=0, msg=key)
                self.assertEqual(code._teacher_handle_prefetch.copy_count, 4)
                self.assertEqual(code._teacher_handle_prefetch.reuse_count, 3)
                self.assertIsNotNone(code._teacher_handle_prefetch.last_use_done)
                # More than one Math micro-batch must still stage Code just once.
                teacher.config.micro_batch_size_per_gpu = 1
                actual = score()
                code._compute_rm_score(batch().union(actual))
                self.assertEqual(code._teacher_handle_prefetch.copy_count, 5)
                self.assertEqual(code._teacher_handle_prefetch.reuse_count, 5 - 1)
            finally:
                dist.destroy_process_group()
