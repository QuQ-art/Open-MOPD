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

Before CPU reduction, each rank's finite local gradients must match an independent
unwrapped-model backward of the on-policy loss `-sum(A * log(pi))`, using the
global token denominator (`atol=2e-5`, `rtol=2e-2` for BF16 backward). At least
one parameter must have different gradients across ranks. The test gathers those
local gradients for verification only and calculates their mean independently
in FP64. Inside the owner's CPU clipping call, the named rank buffers must match
the captured gradients exactly and the unclipped shared gradient must match that
mean (`atol=2e-7`, `rtol=2e-5`). The fixture uses `grad_clip=1e-4` and requires
clipping to be active; clipped gradients, norm, Adam moments and FP32 masters are
checked against a CPU reference. Local GPU gradients must remain unchanged, and
GPU parameter broadcast during the update fails the test.
The underlying CPU AdamW must run exactly once globally, every state step
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
its local accumulated gradients to its own fixed-capacity 256 MiB FP32 named
shared CPU workspace using blocking D2H. Parameters are packed across chunk
boundaries; each chunk has a ready barrier and a consumed barrier before reuse. After a barrier the owner averages all rank buffers, computes the
global norm and clips on CPU, then performs one AdamW update. Missing gradients
contribute zero; parameters absent on every rank skip AdamW/weight decay.
Nonfinite reduced norms cause all ranks to skip the update and publication.
Each rank copies updated shared Student weights H2D; no GPU parameter broadcast
is performed. Scalar version checks, token counts and barriers still use the
existing process group. Separate-state mode continues to use FSDP all-reduce.

The shared-state GPU test uses 28-byte rank workspaces to force parameter splits,
packed chunks and a short final chunk. It compares chunks against GPU all-reduce,
checks last-chunk Inf skipping and a further update after checkpoint restore.
It checks local gradients before reduction, CPU reduced
and clipped gradients against a global-batch reference, nonfinite skipping,
owner-only AdamW, and fails if parameter broadcast is called during updates.
Rank workspaces require `world_size * 256 MiB`, plus one full FP32 reduced-gradient
buffer (`parameter_count * 4` bytes). Global norm/clipping happens only after all
chunks complete; no partial Adam updates are allowed. They are transient workspace, excluded from
optimizer checkpoints; masters/moments remain shared and restored in place.

### 双缓冲与异步 CPU reduce

`SharedCPUAdamW(..., reduce_mode=...)` 支持 `serial`、`double_buffer`、`async`。
也可在所有 rank 统一设置 `OPENMOPD_CPU_REDUCE_MODE`；默认 `async`。
`serial` 保留原来的阻塞 D2H 到单个 mmap workspace；`double_buffer` 使用
两个私有 pinned staging 与两个共享 workspace，但 D2H 阻塞；`async` 在独立
CUDA stream 上把下一块 D2H 与 owner 当前块 CPU reduce 重叠。

CPU 必须等待 D2H completion event 后才能读 staging，并将有效区间复制到
当前共享 workspace。ready barrier 确认所有 rank 发布完成，consumed barrier
保证 owner reduce 完成后才复用槽。所有 chunks 完成后统一计算全局 norm、裁剪
和执行 AdamW；backward 与无 GPU 参数广播的权重发布路径不变。

256 MiB/rank 时共享 workspace 从 1 GiB 增至 2 GiB，另有四 rank 合计 2 GiB
的 pinned staging；完整 reduced-gradient 缓冲不变。scratch 不进 checkpoint。
`last_reduce_metrics` 包含 D2H、completion 等待、staging→shared memcpy、owner
reduce、barrier、transport 总耗时及含 norm/clip 的总耗时；overlap 分项不可相加。

运行 `experiments/run_cpu_reduce_async.py --output /mnt/sda1/zst/RONDO/experiments/<run>`，
等待四卡空闲 60 秒后依次验证三模式，再以正式 Student 参数量做配对性能测试。
`experiments/summarize_cpu_reduce_async.py <run>` 生成报告。该 microbenchmark 不包含
rollout/backward/AdamW，不能替代完整训练吞吐验收。
