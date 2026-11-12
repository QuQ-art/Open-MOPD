# Shared CPU state acceptance

Enable `rollout.multi_gpu.enable=true`, `mode=replicated_dp`,
`cpu_state_mode=shared`, with one node and FSDP1. Actor parameter offload must
be disabled; teacher parameter offload must be enabled. Single-GPU CPUAdamPipeline
is unchanged. The multi-GPU path uses synchronous SharedCPUAdamW with FP32 masters
and moments, and a single optimizer writer (`optimizer_owner_rank`).

The driver creates a unique `/dev/shm/openmopd-<namespace>` directory and passes
its namespace through Ray runtime environments. Each tensor has a named mmap
and JSON metadata (model, tensor name, dtype, shape, byte offset and length).
RPC/collectives carry descriptors only. The driver unlinks the namespace after
the training task returns, including failures. Existing mappings remain valid
until their last tensor reference is released. A forcibly killed driver can
leave a namespace behind; only remove it after its workers have stopped.

Only rank zero loads each checkpoint; other ranks create meta model skeletons.
Initialization staging is released after FSDP consumes it. Student host weights,
FP32 masters, Adam moments/steps and the three teacher flat weights persist in
the shared namespace. GPU replicas and temporary checkpoint-loading/GPU flattening
buffers still exist. FSDP teachers initialize on GPU, then bind their actual CPU
`flat_param._local_shard` to shared storage; no per-rank CPU-offload pinned replica
is retained or allocated during lazy init. Shared mappings are pageable: calling
`pin_memory()` would create a private copy and must not be used on these sources.
This implementation intentionally favors correct shared ownership over asynchronous
pinned H2D performance. Teacher inference remains under `torch.no_grad()`.

Run from the repository root with the training environment's Python:

```bash
python training/verl/tests/multi_gpu/test_shared_cpu_state.py
python training/verl/tests/multi_gpu/test_shared_cpu_state.py --ray
python training/verl/tests/multi_gpu/test_shared_cpu_state.py --gpu
python training/verl/tests/multi_gpu/test_shared_cpu_state.py --gpu --rollout
python training/verl/tests/multi_gpu/test_student_dp.py
python training/verl/tests/multi_gpu/test_student_dp.py --normalization
```

The CPU test uses an independent spawned process. The Ray test checks four
independent actor PIDs, the same backing inode, and cross-process writes. The GPU
test uses real four-rank NCCL/FSDP, compares two shared AdamW steps with a reference,
checks pre-clipping reduced gradients, FP32 masters and moments, exactly one
underlying AdamW call across all ranks per update, and step counters 0 -> 1 -> 2.
It calls the production actor's gradient clipping/finite-check/update method and
requires bitwise-identical student replicas. Injecting NaN or Inf on rank 3
before backward must make every rank skip, without modifying parameters, shared
optimizer state, or the policy version. It also
checks non-owner moment writes and checkpoint restore without rebinding, and proves
that all three teachers' next H2D reads changed shared weights. It also verifies
that no initialization allocations remain. These are small-model component tests,
not a full production-checkpoint OPD training run. Numerical comparisons use the
predeclared `atol=2e-6`, `rtol=2e-5`; replica equality uses zero tolerance.

`--gpu --rollout` creates a temporary local one-layer Qwen2 checkpoint/tokenizer
and runs four real BF16 vLLM engines with TP=1. It uses the production worker
rollout-mode, generation and trainer-mode method bodies, bypassing only Ray
dispatch/profiling wrappers and full worker initialization. After one shared CPU
AdamW update it checks rollout version 0 -> 1, every vLLM runtime weight against
the new HF student (including packed projections), and greedy output tokens
against the updated HF model. Runtime weights and tokens require exact equality.
The parent timeout for this optional integration test is 600 seconds.

In replicated-DP mode, `token-mean` counts valid tokens across the complete
optimizer mini-batch using one NCCL all-reduce. Every micro-batch computes a
masked loss sum and backpropagates `world_size * local_loss_sum / global_tokens`;
there is no additional division by the number of accumulated micro-batches.
Policy format filtering uses its effective token count, while entropy and KL
use the response-mask token count. Domain and importance weights modify the
numerator only and are not included in the token counts. Zero-token
ranks contribute zero gradients. Sequence-mean objectives, including GSPO, and
the single-GPU pipeline retain their existing normalization.

`--normalization` exercises the production actor's `update_policy` method with
only the toy model's log-prob extraction substituted. It compares reduced
gradients against an independent single-process CPU FP64 global-batch formula.
Cases cover unequal valid token counts `[1, 2, 4, 6]`, a zero-token rank, entirely
masked batches, fixed/dynamic micro-batch accumulation, a whole local mini-batch,
entropy/KL terms, top-k policy losses, and policy format filtering. Accumulation
cases have rank token counts `[7, 6, 5, 7]`. Tests also assert one token-count
all-reduce per optimizer mini-batch and unchanged sequence-mean behavior. The
predeclared tolerances remain `atol=2e-6`, `rtol=2e-5`.

Four-sample, one-step three-teacher MOPD integration acceptance:

```bash
python training/verl/tests/multi_gpu/test_shared_cpu_state.py --gpu --rollout --mopd
```

This extends the real four-engine vLLM test. Four fixed prompts with domains
`[math, code, if, math]` pass through the production DP dispatcher (only Ray
object transport is substituted). Each rank generates a distinct response.
The test uses three different deterministic, local Qwen2 teachers, owner-loaded
into shared CPU mappings, and calls `RewardModelWorker._compute_rm_score`
sequentially. Each scorer checks that FSDP's actual H2D source still references
its shared mapping both before and after scoring; precomputed/overlapped logits
are rejected on this path. The existing trainer's sequential teacher ordering
is retained, with all multi-GPU overlap switches disabled by preflight.

For every teacher and rank, an independent unwrapped single-GPU forward on the
same tokens provides the reference (`atol=2e-6`, `rtol=2e-5`). Domain routing,
`compute_distillation_reward`, `compute_token_reward_direct_advantage`, and
`DataParallelPPOActor.update_policy` use their production implementations.
The signal is checked against `(teacher_logp - student_logp) * softmax(student_logp)`.
The actor receives the local mini-batch size 1, corresponding to configured
global mini-batch size 4, with one micro-batch and one epoch.

Before clipping, reduced gradients must be finite and bitwise identical on all
ranks. The underlying CPU AdamW must run exactly once globally, every state step
must become 1, and the new Student parameters must change and match bitwise on
all four GPUs. The subsequent real vLLM weight publication and generation are
also checked against the new Student. Successful runs print `GPU_MOPD_PASS`
followed by `GPU_ROLLOUT_PASS`; the parent enforces a 600-second timeout.
This is a small-model integration acceptance, not a full Ray trainer launch or
a run with the production teacher checkpoints; full worker initialization and
Ray RPC wrappers are bypassed, but scoring, actor update, FSDP/NCCL, shared Adam,
and vLLM generation execute on real GPUs.

## Shared Adam CPU gradient reduction

Shared mode requires `multi_gpu.gradient_sync=cpu_reduce` (the r2.sh default).
All forward/backward micro-batches run under FSDP `no_sync()`. Each rank writes
its local accumulated gradients to its own FP32 named shared CPU buffer using
blocking D2H. After a barrier the owner averages all rank buffers, computes the
global norm and clips on CPU, then performs one AdamW update. Missing gradients
contribute zero; parameters absent on every rank skip AdamW/weight decay.
Nonfinite reduced norms cause all ranks to skip the update and publication.
Each rank copies updated shared Student weights H2D; no GPU parameter broadcast
is performed. Scalar version checks, token counts and barriers still use the
existing process group. Separate-state mode continues to use FSDP all-reduce.

The shared-state GPU test checks local gradients before reduction, CPU reduced
and clipped gradients against a global-batch reference, nonfinite skipping,
owner-only AdamW, and fails if parameter broadcast is called during updates.
Rank buffers require `world_size * parameter_count * 4` CPU bytes, plus one
FP32 reduced-gradient buffer. They are transient workspace, excluded from
optimizer checkpoints; masters/moments remain shared and restored in place.
