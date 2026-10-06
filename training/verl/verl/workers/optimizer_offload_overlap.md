# Optimizer offload during rollout

This opt-in path moves optimizer-state device-to-host (D2H) transfers out of the
next rollout's cache-cleanup wait and into generation. It leaves cache cleanup
and weight/KV preparation in their original order. Defaults remain unchanged.

## Enable

Pass these Hydra overrides to the training command:

```text
actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
actor_rollout_ref.actor.fsdp_config.param_offload=false
actor_rollout_ref.rollout.optimizer_offload_overlap=true
actor_rollout_ref.rollout.optimizer_offload_overlap_chunk_mb=32
```

Supported scope: one CUDA GPU, colocated FSDP1 actor and synchronous vLLM rollout,
with contiguous optimizer-state tensors. Each DMA is at most `chunk_mb` MiB;
this must be a positive integer. CPU state and non-tensor metadata stay unchanged.

The GPU must fit optimizer state **at the same time as the rollout weights and
KV cache**. One reusable pinned CPU copy of the state is also retained. Reduce
`actor_rollout_ref.rollout.gpu_memory_utilization` when this does not fit.
In the tested 72 GiB-class GPU configuration below, `0.5` failed during KV wake;
`0.35` completed. This value is workload-specific, not a universal memory bound.

## Ordering and lifetime

1. After actor update, allocate/reuse pinned CPU buffers and record a producer
   event. Keep the updated optimizer state on GPU.
2. Complete rollout cache cleanup, weight synchronization and KV wake.
3. Start a background thread using a separate CUDA stream. Wait for the producer
   event and submit one bounded D2H chunk at a time. Only the background thread
   waits for each chunk, allowing vLLM token transfers to interleave.
4. After generation (including its exception path), wait for completion before
   replacing optimizer state with the CPU buffers and releasing GPU references.
   The next actor update, checkpoint save/load, trainer-mode transition and
   final profiler drain also finish any pending offload. A checkpoint or another
   update without an intervening rollout starts and finishes the transfer itself.

Submitting the entire state at once can block vLLM's token D2H behind optimizer
traffic even on separate streams. An initial full-enqueue trial showed a 437 ms
blocking copy inside rollout. Chunking removed that long stall. GPU sources stay
referenced until completion; CPU destinations are not published early. Background
copy errors propagate to the caller after draining partial DMA.

## Validation (2026-09-29)

Configuration: batch 4, BF16 stochastic AdamW, student/Math forward overlap,
37 Code-prefetch handles, `math_h2d_done` trigger, attention metadata reuse,
CUDA connections 8, rollout memory utilization 0.35. Both modes ran four training
steps; Nsight captured steps 3 and 4. Source hashes matched throughout both runs.

For the step 3 -> 4 transition:

| Metric | Serial offload | 32 MiB chunk overlap |
|---|---:|---:|
| Optimizer D2H | 22.911 GiB | 22.911 GiB |
| D2H DMA busy time | 460.852 ms | 472.921 ms |
| D2H inside next rollout | 0 ms | 472.921 ms |
| D2H concurrent with standalone rollout kernels | 0 ms | 17.136 ms (3.62%) |
| D2H concurrent with rollout CUDA Graphs or standalone kernels | 0 ms | 148.145 ms (31.33%) |
| Entry `empty_cache_before_rollout` | 585.507 ms | 88.545 ms |
| Post-rollout offload join | N/A | 0.444 ms |
| Sampled whole-run GPU peak | 67010 MiB | 70738 MiB |

The chunked transfers span 1.640 s because they interleave with rollout; the
GPU DMA busy time above excludes gaps. The largest DMA was 32 MiB (784 copies).
The largest rollout-thread CUDA API in its first 700 ms was 17.15 ms, compared
with 17.43 ms in the serial baseline and 437.22 ms in the full-enqueue trial.
The original kernel-only analysis omitted `CUPTI_ACTIVITY_KIND_GRAPH_TRACE`,
which contains the main vLLM forward executions. The 3.62% value covers only
standalone kernels and must not be presented as total GPU execution overlap.
Graph intervals include their internal dependencies; a graph/node-level capture
would be needed to measure exact SM-active kernel overlap inside those intervals.

Online generations diverged: step-4 mean response length was 876.75 vs 1024
and step duration was 15.184 vs 15.266 s including the final profiling drain.
These runs establish transfer placement and removal of the long blocking call;
they do **not** establish a stable end-to-end speedup or full-model bitwise
training equivalence. The sampled peak difference also does not measure the
additional resident optimizer state: peaks can occur in different phases.

Six focused CUDA tests cover repeated FP32/BF16 updates against bitwise-equal
serial references, pinned-buffer reuse, producer-stream ordering, no-rollout
completion, copy-failure recovery, rollout exceptions and checkpoint boundaries.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=training/verl:training/verl/tests/workers \
  python -m unittest -v test_optimizer_offload_overlap
```

Local raw reports, comparison scripts, source hashes and logs are under
`output/optimizer_rollout_overlap_20260929/` (ignored experiment artifacts).
`chunk_comparison.json` contains CUDA correlation-ID-based overlap statistics;
`validation_summary.json` contains trace SHA256 hashes and sampled memory peaks.

## Why graph gaps increase (diagnosis, 2026-09-30)

The later profile (`output/optimizer_overlap_profile_20260929/`) confirms two
coupled bottlenecks. The wrapper requests `logprobs=0`, which still returns the
sampled token's logprob. vLLM `LogprobsTensors.tolists()` performs three pageable
D2H readbacks; these host-blocking CUDA calls also stall the optimizer thread's
CUDA submissions in this trace. One optimizer `cudaMemcpyAsync` took 6.986 ms
inside a 7.721 ms graph execution, but its DMA began only after the graph ended.
Across step 4, 153 slow background memcpy calls consumed 1110.636 ms of host API
time. That time overlaps graph computation and is not additive rollout latency.

Once submission resumes, 32 MiB optimizer chunks and critical 16/32-byte
logprob/token transfers serialize on the D2H path. Two chunks in one observed
gap each delayed a tiny readback by about 0.5–0.65 ms. A separate stream does not
remove host-side CUDA API serialization or grant those tiny copies priority.
Waiting for every optimizer chunk bounds queued traffic but cannot prevent this
competition. The background thread also contributes CPU/API overhead; the trace
does not identify a specific driver lock or prove GIL contention as the cause.

For the first 153 graph-to-graph gaps of step 4, the latest overlap run averaged
3.045 ms versus 1.533 ms in the same-configuration serial baseline. The next 153
gaps, after optimizer DMA ended, averaged 1.555 ms. The first-window gap excess
is 231.291 ms; it must not be extrapolated to all 1024 generated-token iterations.
The corrected GPU-execution overlap is 158.111/475.135 ms = 33.28%; the rest of
DMA falls outside the recorded graph/standalone-kernel execution intervals.

An isolated two-repeat probe on idle GPU 1 (A6000, not the training GPU) held
synthetic graph work and 4 GiB/32 MiB-chunk D2H fixed. Replacing pageable logprob
readbacks with reusable pinned buffers and an event changed DMA/graph overlap
from 30–31% to about 81% and removed background memcpy API calls longer than
1 ms. Removing logprob readback in the probe gave 83–85%. This supports the
mechanism, not an end-to-end training speedup estimate.

The rollout wrapper now avoids requesting unused rollout logprobs when
`calculate_log_probs=false` (`logprobs=None`, not `0`). If logprobs are needed,
use pinned asynchronous readbacks with an event before CPU consumption. Critical
token D2H still competes with optimizer DMA, so chunk scheduling/size remains a
secondary limit. This diagnosis did not change training or installed vLLM code.

Detailed gap decomposition and probe artifacts are in
`output/optimizer_overlap_diagnosis_20260929/`.

## Rollout logprob fix and profile (2026-09-30)

The synchronous vLLM wrapper now defaults to `logprobs=None` when
`calculate_log_probs=False`, eliminating unused logprob computation and pageable
CPU readbacks. With `calculate_log_probs=True`, it retains `logprobs=0` and the
sampled-token logprobs consumed by the output path. Explicit sampling-parameter
overrides continue to follow the existing configuration behavior.

A real MixSFT-model check compared `logprobs=0` and `None` on two fixed prompts,
using greedy decoding and seeded sampling. Generated token IDs matched exactly;
enabled logprobs contained every sampled token, and disabled results were absent.
This bounded generation check is not full-training bitwise equivalence.

One new four-step profile used the same GPU, batch, teacher prefetch,
32 MiB optimizer chunks and 0.35 rollout-memory configuration. Step 4:

| Metric | Before logprob fix | After logprob fix |
|---|---:|---:|
| Optimizer D2H | 22.911 GiB | 22.911 GiB |
| DMA busy time | 475.135 ms | 475.517 ms |
| First-to-last DMA span | 1669.243 ms | 502.430 ms |
| DMA with Graph/standalone-kernel execution | 33.28% | 87.98% |
| Pageable rollout D2H calls | 3072 | 0 |
| Background memcpy APIs >1 ms | 153 | 0 |
| First 153 graph-gap mean | 3.045 ms | 1.480 ms |
| Gaps affected by optimizer DMA | 153 | 47 |
| Gap mean during optimizer DMA | 3.045 ms | 1.808 ms |
| Gap mean after optimizer DMA | 1.556 ms | 1.339 ms |

Step 3 GPU-execution overlap improved from 32.14% to 85.88%. Peak sampled GPU
memory was 70728 MiB, with no OOM; source hashes stayed unchanged during capture.
The remaining gap cost includes the sampled-token D2H dependency and its
competition with bounded optimizer chunks. This is not claimed to be 100%
SM-active overlap: CUDA Graph intervals include internal dependencies.
Online outputs differ across runs, so total rollout/step timings are descriptive
rather than a controlled end-to-end speedup measurement.

Artifacts: `output/optimizer_overlap_logprob_fix_20260930/`, including raw Nsight
report, `comparison.json`, `logprob_verification.json`, `manifest.json`, source
snapshot, GPU-memory samples and before/after gap plots.
