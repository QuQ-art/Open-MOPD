#!/usr/bin/env bash
# Normal training with the workload settings from scripts/local/profile_native_common.sh.
set -euo pipefail

die() { printf '%s\n' "$*" >&2; exit 2; }

dry_run=false
while (($# > 0)); do
    case "$1" in
        --dry-run) dry_run=true; shift ;;
        --help|-h)
            printf 'Usage: bash %s [--dry-run] [Hydra overrides...]\n' "$0"
            printf 'Activate conda environment mopd first. Defaults to one epoch without profiling.\n'
            printf 'Token batching: TOKEN_BATCHING_ENABLED=true, TOKEN_BATCHING_MIN_TOKENS=8192, TOKEN_BATCHING_MAX_BATCH_SIZE=64.\n'
            printf 'Merge whole rollouts until prompt+response tokens strictly exceed the floor; the cap counts sequences after rollout.n.\n'
            printf 'Token batching requires positive OPENMOPD_NUM_STEPS or trainer.total_training_steps; provide enough epochs/data.\n'
            printf 'Trailing Hydra overrides take precedence; TOKEN_BATCHING_ENABLED=false restores ordinary batching.\n'
            printf 'USE_CU13=false; true prefers the CUDA 13 toolkit bundled with the selected Python environment.\n'
            printf 'Environment: PYTHONNOUSERSITE=1; VLLM_USE_V2_MODEL_RUNNER defaults to 0.\n'
            printf 'Multi-GPU disables single-GPU overlap, shared actor/vLLM weights, CPUAdam slots, and token batching by default.\n'
            printf 'Multi-GPU: OPENMOPD_N_GPUS_PER_NODE=2 auto-enables replicated_dp; MULTI_GPU_CPU_STATE_MODE=shared|separate, MULTI_GPU_OPTIMIZER_OWNER_RANK=0.\n'
            printf 'CPUAdam: CPU_ADAM_ENABLED=false, CPU_ADAM_ASYNC=null (true/false/null).\n'
            printf 'Gradients/backend: PIPELINE_FP32_GRADIENTS=false, PIPELINE_OPTIMIZER_BACKEND=cpu (cpu/cuda).\n'
            printf 'TEACHER_PARAM_OFFLOAD=true applies to all three teachers; use Hydra overrides per teacher.\n'
            printf 'Teacher slot cap: TEACHER_LAYER_PIPELINE_MAX_MB=8192.\n'
            printf 'Sync diagnostics: VERIFY_WEIGHT_SYNC=false, SKIP_INITIAL_WEIGHT_SYNC=false (non-shared weights).\n'
            printf 'OPD: TOP_K_STRATEGY=only_stu, REWARD_WEIGHT_MODE=student_p, TEACHER_TEMPERATURE=1.0.\n'
            printf 'Rollout: ENFORCE_EAGER=false, ENABLE_CHUNKED_PREFILL=true, ENABLE_PREFIX_CACHING=true, FREE_CACHE_ENGINE=true.\n'
            printf 'Set STEP_MEMORY_ENABLED=false to disable step/phase memory telemetry for speed tests.\n'
            printf 'Pinned pipeline buffers round to 2 MiB; OPENMOPD_PINNED_ALIGNMENT_MB=0 restores PyTorch allocation.\n'
            exit 0 ;;
        --) shift; break ;;
        -*) die "Unknown option: $1" ;;
        *) break ;;
    esac
done

repo_root="${OPENMOPD_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)}"
python_bin="${OPENMOPD_PYTHON:-python3}"
checkpoint_dir="${repo_root}/ckpt"

# Support both models/OPD/... and models/... layouts.
model_root="${OPENMOPD_MODEL_ROOT:-/home/xxf/Distill/models}"
if [[ -d "${model_root}/OPD" ]]; then
    model_root="${model_root}/OPD"
fi

resolve_executable() {
  local executable="$1"
  if [[ "${executable}" != */* ]]; then executable="$(command -v "${executable}" || true)"; fi
  [[ -f "${executable}" && -x "${executable}" ]] || die "Executable not found: $1"
  # Preserve the executable symlink: resolving a virtualenv's python to its
  # base interpreter would lose that environment's packages.
  printf '%s/%s\n' "$(cd "$(dirname "${executable}")" && pwd -P)" "$(basename "${executable}")"
}

hydra_string() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "${value}"
}

# Inspect topology overrides before selecting defaults; final validation still
# runs after every trailing override has been appended.
num_gpus="${OPENMOPD_N_GPUS_PER_NODE:-1}"
MULTI_GPU_ENABLED="${MULTI_GPU_ENABLED:-auto}"
for override in "$@"; do
    case "${override}" in
        trainer.n_gpus_per_node=*) num_gpus="${override#*=}" ;;
        actor_rollout_ref.rollout.multi_gpu.enable=*|+actor_rollout_ref.rollout.multi_gpu.enable=*|++actor_rollout_ref.rollout.multi_gpu.enable=*) MULTI_GPU_ENABLED="${override#*=}" ;;
    esac
done
[[ "${num_gpus}" =~ ^[1-9][0-9]*$ ]] || die "OPENMOPD_N_GPUS_PER_NODE must be a positive integer"
if [[ "${MULTI_GPU_ENABLED}" == auto ]]; then
    MULTI_GPU_ENABLED=false
    if ((num_gpus > 1)); then MULTI_GPU_ENABLED=true; fi
fi
[[ "${MULTI_GPU_ENABLED}" == true || "${MULTI_GPU_ENABLED}" == false ]] || die "MULTI_GPU_ENABLED must be auto, true, or false"
single_gpu_default=true
if [[ "${MULTI_GPU_ENABLED}" == true ]]; then single_gpu_default=false; fi
MULTI_GPU_MODE="${MULTI_GPU_MODE:-replicated_dp}"
MULTI_GPU_CPU_STATE_MODE="${MULTI_GPU_CPU_STATE_MODE:-shared}"
MULTI_GPU_OPTIMIZER_OWNER_RANK="${MULTI_GPU_OPTIMIZER_OWNER_RANK:-0}"
USE_CU13="${USE_CU13:-false}"
SHARE_STUDENT_WEIGHTS="${SHARE_STUDENT_WEIGHTS:-${single_gpu_default}}"
# Weight-copy diagnostics apply to the non-shared rollout path.
VERIFY_WEIGHT_SYNC="${VERIFY_WEIGHT_SYNC:-false}"
SKIP_INITIAL_WEIGHT_SYNC="${SKIP_INITIAL_WEIGHT_SYNC:-false}"
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-true}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"
FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-true}"
# CPUAdam requires only_stu; the other strategies support non-CPUAdam runs.
TOP_K_STRATEGY="${TOP_K_STRATEGY:-only_stu}"
REWARD_WEIGHT_MODE="${REWARD_WEIGHT_MODE:-student_p}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-1.0}"
BF16_STUDENT_WEIGHTS="${BF16_STUDENT_WEIGHTS:-true}"
STUDENT_TEACHER_PIPELINE="${STUDENT_TEACHER_PIPELINE:-${single_gpu_default}}"
# Match the Hydra defaults; slot reuse also enables CPUAdam.
CPU_ADAM_ENABLED="${CPU_ADAM_ENABLED:-false}"
# null preserves legacy publication timing; false publishes in step().
CPU_ADAM_ASYNC="${CPU_ADAM_ASYNC:-null}"
PIPELINE_FP32_GRADIENTS="${PIPELINE_FP32_GRADIENTS:-false}"
PIPELINE_OPTIMIZER_BACKEND="${PIPELINE_OPTIMIZER_BACKEND:-cpu}"
PIPELINE_OVERLAP="${PIPELINE_OVERLAP:-${single_gpu_default}}"
CPU_OPTIMIZER_THREADS="${CPU_OPTIMIZER_THREADS:-8}"
PIPELINE_GRADIENT_OFFLOAD="${PIPELINE_GRADIENT_OFFLOAD:-true}"
STEP_MEMORY_ENABLED="${STEP_MEMORY_ENABLED:-true}"
TEACHER_LAYER_PIPELINE="${TEACHER_LAYER_PIPELINE:-false}"
TEACHER_LAYER_PIPELINE_MAX_MB="${TEACHER_LAYER_PIPELINE_MAX_MB:-8192}"
optimizer_layer_default="${single_gpu_default}"
if [[ "${STUDENT_TEACHER_PIPELINE}" == true || "${CPU_ADAM_ENABLED}" == true ]]; then
    optimizer_layer_default=false
fi
OPTIMIZER_OFFLOAD_PER_LAYER="${OPTIMIZER_OFFLOAD_PER_LAYER:-${optimizer_layer_default}}"
# Shared default for the primary and both additional teachers.
TEACHER_PARAM_OFFLOAD="${TEACHER_PARAM_OFFLOAD:-true}"
TEACHER_PARAM_PREFETCH="${TEACHER_PARAM_PREFETCH:-${single_gpu_default}}"
TEACHER_PARAM_PREFETCH_MAX_MB="${TEACHER_PARAM_PREFETCH_MAX_MB:-768}"
TEACHER_FORWARD_OVERLAP="${TEACHER_FORWARD_OVERLAP:-${single_gpu_default}}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"
export OPENMOPD_PINNED_ALIGNMENT_MB="${OPENMOPD_PINNED_ALIGNMENT_MB:-2}"
[[ "${OPENMOPD_PINNED_ALIGNMENT_MB}" =~ ^[0-9]+$ ]] || die "OPENMOPD_PINNED_ALIGNMENT_MB must be a nonnegative integer"
train_batch_size="${OPENMOPD_TRAIN_BATCH_SIZE:-${num_gpus}}"
# Keep rollout chunks fixed; merge actual prompt+response tokens before scoring/training.
TOKEN_BATCHING_ENABLED="${TOKEN_BATCHING_ENABLED:-${single_gpu_default}}"
TOKEN_BATCHING_MIN_TOKENS="${TOKEN_BATCHING_MIN_TOKENS:-8192}"
TOKEN_BATCHING_MAX_BATCH_SIZE="${TOKEN_BATCHING_MAX_BATCH_SIZE:-64}"
gpu_memory_limit_gb="${OPENMOPD_GPU_MEMORY_LIMIT_GB:-24}"
kv_cache_mb="${OPENMOPD_KV_CACHE_MB:-4096}"
max_num_seqs="${OPENMOPD_MAX_NUM_SEQS:-8}"
val_batch_size="${OPENMOPD_VAL_BATCH_SIZE:-8}"
val_reward_workers="${OPENMOPD_VAL_REWARD_WORKERS:-8}"
val_before_train="${OPENMOPD_VAL_BEFORE_TRAIN:-true}"
# Token batching requires an explicit optimizer-step budget (environment or trailing Hydra override).
# null keeps the full-epoch behavior when token batching is disabled.
num_steps="${OPENMOPD_NUM_STEPS:-null}"

for var in USE_CU13 SHARE_STUDENT_WEIGHTS BF16_STUDENT_WEIGHTS OPTIMIZER_OFFLOAD_PER_LAYER \
  TEACHER_PARAM_PREFETCH TEACHER_FORWARD_OVERLAP STUDENT_TEACHER_PIPELINE PIPELINE_OVERLAP \
  TEACHER_LAYER_PIPELINE PIPELINE_GRADIENT_OFFLOAD STEP_MEMORY_ENABLED val_before_train \
  CPU_ADAM_ENABLED PIPELINE_FP32_GRADIENTS TEACHER_PARAM_OFFLOAD \
  VERIFY_WEIGHT_SYNC SKIP_INITIAL_WEIGHT_SYNC ENFORCE_EAGER ENABLE_CHUNKED_PREFILL \
  ENABLE_PREFIX_CACHING FREE_CACHE_ENGINE TOKEN_BATCHING_ENABLED; do
  [[ "${!var}" == true || "${!var}" == false ]] || die "${var} must be true or false"
done
[[ "${CPU_ADAM_ASYNC}" == true || "${CPU_ADAM_ASYNC}" == false || "${CPU_ADAM_ASYNC}" == null ]] || \
  die "CPU_ADAM_ASYNC must be true, false, or null"
[[ "${PIPELINE_OPTIMIZER_BACKEND}" == cpu || "${PIPELINE_OPTIMIZER_BACKEND}" == cuda ]] || \
  die "PIPELINE_OPTIMIZER_BACKEND must be cpu or cuda"
case "${TOP_K_STRATEGY}" in
  only_stu|only_tch|intersection|union|top_p_intersec) ;;
  *) die "TOP_K_STRATEGY must be only_stu, only_tch, intersection, union, or top_p_intersec" ;;
esac
case "${REWARD_WEIGHT_MODE}" in
  student_p|teacher_p|none) ;;
  *) die "REWARD_WEIGHT_MODE must be student_p, teacher_p, or none" ;;
esac
for var in num_gpus train_batch_size TEACHER_LAYER_PIPELINE_MAX_MB \
  TEACHER_PARAM_PREFETCH_MAX_MB CUDA_DEVICE_MAX_CONNECTIONS CPU_OPTIMIZER_THREADS \
  gpu_memory_limit_gb kv_cache_mb max_num_seqs val_batch_size val_reward_workers \
  TOKEN_BATCHING_MIN_TOKENS TOKEN_BATCHING_MAX_BATCH_SIZE; do
  [[ "${!var}" =~ ^[1-9][0-9]*$ ]] || die "${var} must be a positive integer"
done
((gpu_memory_limit_gb > 1 && kv_cache_mb < (gpu_memory_limit_gb - 1) * 1024)) || \
  die "GPU budget must exceed 1 GiB plus the KV cache"

cudagraph_sizes_csv=1
for ((capture_size = 2; capture_size < max_num_seqs; capture_size *= 2)); do
  cudagraph_sizes_csv+=",${capture_size}"
done
if ((max_num_seqs > 1)); then cudagraph_sizes_csv+=",${max_num_seqs}"; fi

[[ "${num_steps}" == null || "${num_steps}" =~ ^[1-9][0-9]*$ ]] || \
    die "OPENMOPD_NUM_STEPS must be a positive integer"

python_bin="$(resolve_executable "${python_bin}")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
    CUDA_VISIBLE_DEVICES=0
    for ((gpu = 1; gpu < num_gpus; gpu++)); do CUDA_VISIBLE_DEVICES+=",${gpu}"; done
fi
export CUDA_VISIBLE_DEVICES
export VLLM_USE_V1=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export PYTHONNOUSERSITE=1
# The shared-weight integration accesses the original v1 model runner.
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-0}"
if [[ "${USE_CU13}" == true ]]; then
    # Prefer the CUDA toolkit packaged with the selected Python environment.
    python_prefix="$(cd "$(dirname "${python_bin}")/.." && pwd -P)"
    for cuda_candidate in "${python_prefix}"/lib/python*/site-packages/nvidia/cu13; do
        if [[ -x "${cuda_candidate}/bin/nvcc" ]]; then
            export CUDA_HOME="${cuda_candidate}"
            export PATH="${CUDA_HOME}/bin:${PATH}"
            export LD_LIBRARY_PATH="${CUDA_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
            break
        fi
    done
fi
export PATH="$(dirname "${python_bin}"):${PATH}"
export PYTHONPATH="${repo_root}/training/verl${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${MULTI_GPU_CPU_STATE_MODE}" == "separate" ]]; then
  MULTI_GPU_GRADIENT_SYNC="${MULTI_GPU_GRADIENT_SYNC:-auto_all_reduce}"
else
  MULTI_GPU_GRADIENT_SYNC="${MULTI_GPU_GRADIENT_SYNC:-cpu_reduce}"
fi

cmd=("${python_bin}" -m verl.trainer.main_ppo
  trainer.step_memory.enabled="${STEP_MEMORY_ENABLED}"
  algorithm.adv_estimator=token_reward_direct
  "data.train_files=$(hydra_string "${model_root}/data/rl_prompt_mix/train.parquet")"
  "data.val_files=$(hydra_string "${model_root}/data/rl_prompt_mix/eval.parquet")"
  data.train_batch_size="${train_batch_size}" data.max_prompt_length=1024 data.max_response_length=4096
  data.token_batching.enabled="${TOKEN_BATCHING_ENABLED}"
  data.token_batching.min_tokens="${TOKEN_BATCHING_MIN_TOKENS}"
  data.token_batching.max_batch_size="${TOKEN_BATCHING_MAX_BATCH_SIZE}"
  data.val_batch_size="${val_batch_size}"
  data.filter_overlong_prompts=True data.truncation=error
  "actor_rollout_ref.model.path=$(hydra_string "${model_root}/MixSFT")"
  actor_rollout_ref.rollout.multi_gpu.enable="${MULTI_GPU_ENABLED}"
  actor_rollout_ref.rollout.multi_gpu.mode="${MULTI_GPU_MODE}"
  actor_rollout_ref.rollout.multi_gpu.cpu_state_mode="${MULTI_GPU_CPU_STATE_MODE}"
  actor_rollout_ref.rollout.multi_gpu.optimizer_owner_rank="${MULTI_GPU_OPTIMIZER_OWNER_RANK}"
  actor_rollout_ref.rollout.multi_gpu.gradient_sync="${MULTI_GPU_GRADIENT_SYNC:-cpu_reduce}"
  actor_rollout_ref.rollout.multi_gpu.synchronous_update=true
  actor_rollout_ref.rollout.multi_gpu.verify_policy_version=true
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.share_weights="${SHARE_STUDENT_WEIGHTS}"
  actor_rollout_ref.rollout.verify_weight_sync="${VERIFY_WEIGHT_SYNC}"
  actor_rollout_ref.rollout.skip_initial_weight_sync="${SKIP_INITIAL_WEIGHT_SYNC}"
  actor_rollout_ref.rollout.enforce_eager="${ENFORCE_EAGER}"
  actor_rollout_ref.rollout.enable_chunked_prefill="${ENABLE_CHUNKED_PREFILL}"
  actor_rollout_ref.rollout.enable_prefix_caching="${ENABLE_PREFIX_CACHING}"
  actor_rollout_ref.rollout.free_cache_engine="${FREE_CACHE_ENGINE}"
  actor_rollout_ref.rollout.gpu_memory_limit_gb="${gpu_memory_limit_gb}"
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=$((kv_cache_mb * 1024 * 1024))"
  actor_rollout_ref.rollout.max_num_seqs="${max_num_seqs}"
  actor_rollout_ref.rollout.cudagraph_capture_sizes="[${cudagraph_sizes_csv}]"
  actor_rollout_ref.rollout.max_num_batched_tokens=5120
  actor_rollout_ref.rollout.teacher_param_prefetch="${TEACHER_PARAM_PREFETCH}"
  actor_rollout_ref.rollout.teacher_param_prefetch_max_mb="${TEACHER_PARAM_PREFETCH_MAX_MB}"
  actor_rollout_ref.rollout.teacher_forward_overlap="${TEACHER_FORWARD_OVERLAP}"
  actor_rollout_ref.rollout.student_teacher_pipeline="${STUDENT_TEACHER_PIPELINE}"
  actor_rollout_ref.rollout.cpu_adam_enabled="${CPU_ADAM_ENABLED}"
  actor_rollout_ref.rollout.cpu_adam_async="${CPU_ADAM_ASYNC}"
  actor_rollout_ref.rollout.pipeline_fp32_gradients="${PIPELINE_FP32_GRADIENTS}"
  actor_rollout_ref.rollout.pipeline_optimizer_backend="${PIPELINE_OPTIMIZER_BACKEND}"
  actor_rollout_ref.rollout.pipeline_overlap="${PIPELINE_OVERLAP}"
  actor_rollout_ref.rollout.cpu_optimizer_threads="${CPU_OPTIMIZER_THREADS}"
  actor_rollout_ref.rollout.pipeline_gradient_offload="${PIPELINE_GRADIENT_OFFLOAD}"
  actor_rollout_ref.rollout.teacher_layer_pipeline="${TEACHER_LAYER_PIPELINE}"
  actor_rollout_ref.rollout.teacher_layer_pipeline_max_mb="${TEACHER_LAYER_PIPELINE_MAX_MB}"
  +actor_rollout_ref.rollout.reward_mode=mt_opd
  actor_rollout_ref.rollout.n=1 actor_rollout_ref.rollout.max_model_len=5120
  actor_rollout_ref.actor.ppo_mini_batch_size="${train_batch_size}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload_per_layer="${OPTIMIZER_OFFLOAD_PER_LAYER}"
  actor_rollout_ref.actor.optim.override_optimizer_config='{foreach:false}'
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.data_parallel_size=1
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  +actor_rollout_ref.rollout.log_prob_top_k=256
  +actor_rollout_ref.rollout.top_k_strategy="${TOP_K_STRATEGY}"
  +actor_rollout_ref.rollout.reward_weight_mode="${REWARD_WEIGHT_MODE}"
  +actor_rollout_ref.rollout.teacher_temperature="${TEACHER_TEMPERATURE}"
  "custom_reward_function.path=$(hydra_string "${repo_root}/training/verl/verl/utils/reward_score/opd_val_dispatch.py")"
  custom_reward_function.name=reward_func
  reward_model.micro_batch_size_per_gpu=1 reward_model.enable=True
  "reward_model.model.path=$(hydra_string "${model_root}/Math")"
  reward_model.model.input_tokenizer=null
  reward_model.model.fsdp_config.param_offload="${TEACHER_PARAM_OFFLOAD}"
  +reward_model.reward_kwargs.compute_true_reward=false
  +reward_model.reward_kwargs.distributed_validation=true
  +reward_model.reward_kwargs.distributed_num_shards="${val_reward_workers}"
  +reward_model.reward_kwargs.distributed_cpus_per_shard=1
  +reward_model.reward_kwargs.distributed_max_workers_per_shard=1
  '+mt_opd.teacher_domains=[math,code,if]' +mt_opd.n_additional_teachers=2
  trainer.n_gpus_per_node="${num_gpus}" trainer.nnodes=1 trainer.total_epochs=1
  trainer.total_training_steps="${num_steps}" trainer.val_before_train="${val_before_train}"
  trainer.resume_mode=disable
  "trainer.default_local_dir=$(hydra_string "${checkpoint_dir}")"
  trainer.project_name=OpenOPD-local trainer.experiment_name=mt-opd-local
  "trainer.logger=['console','tensorboard']"
  "+ray_kwargs.ray_init.runtime_env.env_vars.CUDA_DEVICE_MAX_CONNECTIONS='${CUDA_DEVICE_MAX_CONNECTIONS}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.OPENMOPD_PINNED_ALIGNMENT_MB='${OPENMOPD_PINNED_ALIGNMENT_MB}'"
  +mt_reward_model_1.enable=True
  "+mt_reward_model_1.model.path=$(hydra_string "${model_root}/Code")"
  +mt_reward_model_1.model.input_tokenizer=null
  +mt_reward_model_1.model.use_remove_padding=True
  +mt_reward_model_1.model.fsdp_config.param_offload="${TEACHER_PARAM_OFFLOAD}"
  +mt_reward_model_2.enable=True
  "+mt_reward_model_2.model.path=$(hydra_string "${model_root}/IF")"
  +mt_reward_model_2.model.input_tokenizer=null
  +mt_reward_model_2.model.use_remove_padding=True
  +mt_reward_model_2.model.fsdp_config.param_offload="${TEACHER_PARAM_OFFLOAD}"
)
if [[ "${BF16_STUDENT_WEIGHTS}" == true && "${SHARE_STUDENT_WEIGHTS}" != true ]]; then
  cmd+=(actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
    actor_rollout_ref.actor.optim.optimizer_impl=verl.utils.bf16_optimizer
    actor_rollout_ref.actor.optim.optimizer=BF16StochasticAdamW)
fi
cmd+=("$@")
# Dependency-free validation also applies to dry-run and final Hydra overrides.
"${python_bin}" "${repo_root}/training/verl/verl/workers/config/multi_gpu.py" "${cmd[@]:3}"
if [[ "${dry_run}" == true ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    exit 0
fi
exec "${cmd[@]}"
