# Sequential teachers with parameter prefetch

`actor_rollout_ref.rollout.teacher_param_prefetch=true` enables both transfers:

1. Stage the configured Math FSDP shards while student log-probs compute.
2. Run Math scoring after student scoring returns, on the ordinary scoring stream.
3. After Math has submitted all parameter HtoD copies, stage a bounded prefix of
   Code shards on a separate copy stream while Math forward continues.
4. Run Code scoring after Math scoring. Each staged shard is consumed once.

No teacher model forward runs on a background thread. The trainer always calls
student, Math and then Code in sequence. The pre-unshard event for Code waits
for Math parameter copies, not Math computation. Reusable staging buffers retain
per-shard copy events and a last-use event to prevent overwriting live parameters.
Only the first Math micro-batch triggers Code prefetch. Scoring failure drains
pending copies; a failed Math call also clears unused Code staging.

## Configuration

```text
actor_rollout_ref.rollout.teacher_param_prefetch=true
actor_rollout_ref.rollout.teacher_param_prefetch_handles=37
actor_rollout_ref.rollout.teacher_param_prefetch_max_mb=6144
actor_rollout_ref.rollout.code_teacher_param_prefetch_handles=37
actor_rollout_ref.rollout.code_teacher_param_prefetch_max_mb=6144
```

The switch defaults to false. Generic configuration defaults to one shard per
teacher with a 768 MiB cap. The four-case profile launcher explicitly selects
37 shards and a 6144 MiB cap for each teacher, covering all parameters of the
current Math and Code models. Initialization and subsequent Math starts use the
same configured shard count. Choose the count for the actual model. Caps are independent and
staging consumes additional GPU memory. Requires colocated Math (`rm`) and Code
(`mt_rm_1`), one GPU, SP=1, CPU-offloaded FSDP1 teachers, and padded/unfused Math.

Remove the obsolete `teacher_forward_overlap`, `code_teacher_param_prefetch`,
and `code_teacher_param_prefetch_trigger` overrides. Code now always uses the
Math-parameter-copy completion boundary. The local profiling launcher accepts
`TEACHER_PARAM_PREFETCH` as the single switch and retains the size/count variables.

`teacher_attention_metadata_cache` remains an independent optional optimization.
`optimizer_offload_overlap` and its implementation are unchanged and independent.

## Verification

```bash
PYTHONPATH=training/verl:training/verl/tests/workers \
  python -m unittest -v test_teacher_prefetch_pipeline test_teacher_param_prefetch

PYTHONPATH=training/verl python scripts/local/verify_teacher_prefetch.py \
  --student /path/to/student --teacher /path/to/Math --code-teacher /path/to/Code \
  --batch-size 4 --math-handles 37 --math-max-mb 6144 \
  --code-handles 37 --code-max-mb 6144 --output /tmp/prefetch-check.json
```

The fixed-input verifier compares both teacher outputs and student outputs with
prefetch disabled/enabled, including repeated staging reuse. It needs CUDA and
local checkpoints. Historical concurrent-forward profiles do not validate this
sequential-forward refactor. The model-only NVTX range is nested inside the full
teacher scoring range, keeping model execution distinguishable from logits
postprocessing; parameter DMA must still be identified separately in the trace.
