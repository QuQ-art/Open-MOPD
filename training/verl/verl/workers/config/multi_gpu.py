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

"""Dependency-free preflight shared by r2.sh and the trainer entry point."""


def validate_multi_gpu(get):
    prefix = "actor_rollout_ref.rollout."
    multi = prefix + "multi_gpu."
    enabled = get(multi + "enable", False)
    if type(enabled) is not bool:
        raise ValueError("multi_gpu.enable must be true or false")
    if not enabled:
        return
    def require(condition, message):
        if not condition:
            raise ValueError("multi_gpu: " + message)

    count = get("trainer.n_gpus_per_node", 1)
    require(type(count) is int and count > 1, "n_gpus_per_node must be an integer greater than 1")
    require(get("trainer.nnodes", 1) == 1, "only one node is supported")
    require(get(multi + "mode", "replicated_dp") == "replicated_dp",
            "only mode=replicated_dp is implemented; sharded_dp/pipeline_parallel are unsupported")
    require(get(multi + "gradient_sync", "auto_all_reduce") == "auto_all_reduce",
            "gradient_sync must be auto_all_reduce (FSDP NO_SHARD owns reduction)")
    require(get(multi + "cpu_state_mode", "shared") in ("shared", "separate"),
            "cpu_state_mode must be shared or separate")
    require(get("data.token_batching.enabled", False) is False,
            "data.token_batching.enabled must be false (single-GPU slot pipeline only)")
    if get(multi + "cpu_state_mode", "shared") == "shared":
        require(get("actor_rollout_ref.actor.fsdp_config.param_offload", False) is False,
                "shared CPU state requires actor param_offload=false; student host weights live in the shared store")
        require(get("actor_rollout_ref.model.lora_rank", 0) == 0,
                "shared CPU state currently does not support LoRA")
        for teacher in ("reward_model", "mt_reward_model_1", "mt_reward_model_2"):
            require(get(teacher + ".model.fsdp_config.param_offload", True) is True,
                    teacher + " requires param_offload=true for shared teacher CPU weights")
            require(get(teacher + ".strategy", "fsdp") == "fsdp", teacher + " shared CPU state requires FSDP1")
    owner = get(multi + "optimizer_owner_rank", 0)
    require(type(owner) is int and 0 <= owner < count, "optimizer_owner_rank must be in [0, world_size)")
    require(get(multi + "synchronous_update", True) is True, "synchronous_update must be true")
    require(type(get(multi + "verify_policy_version", True)) is bool, "verify_policy_version must be boolean")
    for key in ("student_teacher_pipeline", "teacher_layer_pipeline",
                "teacher_forward_overlap", "teacher_param_prefetch", "pipeline_overlap", "cpu_adam_enabled",
                "pipeline_fp32_gradients", "skip_initial_weight_sync"):
        require(get(prefix + key, False) is False, key + " must be false in multi-GPU mode")
    if get(prefix + "share_weights", False):
        validate_replicated_weight_sharing({"mode": get(multi + "mode", "replicated_dp")}, count)
        require(get("actor_rollout_ref.actor.fsdp_config.param_offload", False) is False,
                "shared weights require actor.fsdp_config.param_offload=false")
        require(get("actor_rollout_ref.model.lora_rank", 0) == 0
                and get("actor_rollout_ref.model.lora_adapter_path", None) is None,
                "shared weights do not support LoRA")
    require(get(prefix + "cpu_adam_async", None) is None, "cpu_adam_async must be null")
    require(get(prefix + "pipeline_optimizer_backend", "cpu") == "cpu", "pipeline_optimizer_backend must be cpu")
    require(get(prefix + "name", "vllm") == "vllm" and get(prefix + "mode", "sync") == "sync",
            "requires synchronous vllm rollout")
    for key in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "data_parallel_size"):
        require(get(prefix + key, 1) == 1, key + " must be 1")
    require(get("actor_rollout_ref.actor.strategy", "fsdp") == "fsdp", "requires FSDP1 actor")
    require(get("actor_rollout_ref.actor.ulysses_sequence_parallel_size", 1) == 1, "requires Ulysses size 1")
    require(get("actor_rollout_ref.actor.fsdp_config.optimizer_offload_per_layer", False) is False,
            "optimizer_offload_per_layer must be false")
    for key in ("data.train_batch_size", "actor_rollout_ref.actor.ppo_mini_batch_size"):
        batch = get(key, count)
        require(type(batch) is int and batch > 0 and batch % count == 0, key + " must be divisible by GPU count")


def validate_replicated_weight_sharing(config, world_size):
    """Each rank aliases only its own NO_SHARD actor and TP=1 rollout."""
    if config.get("mode", "replicated_dp") != "replicated_dp" or world_size < 2:
        raise ValueError("replicated shared weights require replicated_dp and world_size >= 2")


def validate_training_config(config):
    from omegaconf import OmegaConf
    validate_multi_gpu(lambda key, default: OmegaConf.select(config, key, default=default))


def main():
    import ast
    import sys
    values = {}
    for arg in sys.argv[1:]:
        key, sep, value = arg.partition("=")
        if not sep:
            continue
        key = key.lstrip("+")
        if key.startswith("~"):
            raise SystemExit("r2 preflight: deletion overrides are unsupported")
        lowered = value.lower()
        if lowered in ("true", "false", "null"):
            value = {"true": True, "false": False, "null": None}[lowered]
        else:
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
        values[key] = value
    try:
        validate_multi_gpu(lambda key, default: values.get(key, default))
        count = values.get("trainer.n_gpus_per_node", 1)
        batch = values.get("data.train_batch_size", 1)
        if type(count) is not int or count < 1:
            raise ValueError("trainer.n_gpus_per_node must be a positive integer")
        if type(batch) is not int or batch < 1 or batch % count:
            raise ValueError("Train batch size must be positive and divisible by GPU count")
        if values.get("actor_rollout_ref.rollout.teacher_forward_overlap") is True and (count != 1 or batch != 1):
            raise ValueError("Teacher forward overlap requires one GPU and batch size 1")
        import os
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and count > len([item for item in visible.split(",") if item.strip() and item != "-1"]):
            raise ValueError("CUDA_VISIBLE_DEVICES exposes fewer devices than trainer.n_gpus_per_node")
        if count != 1 and values.get("actor_rollout_ref.rollout.multi_gpu.enable") is not True:
            raise ValueError("r2 multi-GPU launch requires multi_gpu.enable=true")
    except ValueError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
