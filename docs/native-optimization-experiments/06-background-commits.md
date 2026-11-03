# Bounded background shard commits

This follows the observation-cache profile. Shard validation, packing, NPY writes,
SHA256 hashing, fsync and atomic manifest commits run on the CPU between GPU point
batches. The experiment overlaps that work with the next full-timeline GPU batch.
It changes scheduling and ownership, preserving the store format and arithmetic.

## Design

`BoundedShardCommitter` owns either synchronous writes or one `ThreadPoolExecutor`
worker. The main thread computes a batch, copies its output to independent CPU
arrays, and submits those arrays. It computes the next batch while that commit
runs. Before submitting again, it waits for the previous future. There is at most
one pending shard; the executor cannot accumulate an output queue. No CUDA work
runs in the commit thread.

CPU buffers remain owned by the future until its write finishes. The inference
loop never mutates or reuses them. The main thread snapshots verified resume
coverage before starting the writer and uses that immutable set while computing.
Only the worker mutates the store manifest during writes. The main thread drains
the writer before final verification and publication of the completed store.

The existing `TrajectoryStore.write_shard` handles dtype/shape checks, finiteness,
hashing, fsync and atomic commits unchanged. A background failure propagates to
the caller. Cleanup joins the worker even if computation also fails; a secondary
write failure is attached to the original exception. A verified committed prefix
can be resumed, and incomplete shards are recomputed. These remain process-level
recovery guarantees; parent-directory fsync and machine-crash durability are not
added by this experiment.

`storage.commits = "sync"` remains the default. Select the `background-commits`
profile, or inherit another profile and set `[storage] commits = "threaded"`.
`InferenceComponents.shard_committer_factory` accepts a fingerprinted factory
`(sink) -> committer` with `submit`, `drain`, `metrics` and a context-manager
lifetime. Implementations must honor the one-pending-shard ownership contract.
There is no shared global pool or monkey-patching.

The RAM staging check reserves a second conservative workspace for a pending
writer, including packing and finite-check temporaries. This is separate from the
retained payload counter, and remains an allocation target rather than an OS RSS
limit. GPU allocation budgets and scratch checks keep their existing meanings.

## Measurement plan

Use one frozen source for both modes. Alternate synchronous/threaded ordering over
five fresh correction pairs on the full 84-frame swing-tire clip, at the original
504 processing resolution, 1,024-point groups, batch 8 and seed 0. Keep caches,
prefetch, precision and other backends at control defaults. The process scope
includes model load, mapped preparation, correction, hashing, durable commits,
verification and cleanup; it excludes provisioning and viewer export.

If correction improves, measure five fresh full-pipeline pairs. Report correction
stage timing separately from DA3, tracking and total process timing. Changes in
unrelated provider stages cannot be attributed to commit overlap. Retain ranges,
paired observations and failed trials. Sample process-tree RSS/PSS, total device
memory, physical disk I/O and output allocation alongside CUDA allocator peaks.
Retain the OS page cache and do not claim cold-storage measurements.

The runner now refuses an occupied compute GPU before creating a trial and records
NVML compute-process ownership at each 100 ms telemetry sample. Only the trial's
process tree is allowed. An unrelated compute process stops our worker and marks
the trial `gpu_interference`; it never terminates the unrelated process. Interrupts
also reap our worker. Very short activity between samples can still be missed.

Check every corrected position, raw-lift value and mask over every frame and
reference pixel, including padded repeated pixels. Run separate composition checks
for observation caching, prefetch, eager CUDA rays and native packing/writing;
these single runs establish compatibility rather than latency distributions.
Test queue bounds, retained buffer lifetimes, exceptions, staging budgets, verified
resume and corrupted-shard rejection before interpreting performance.

## Isolated correction measurements

The five fresh matched pairs use the same frozen implementation at
`97ff732956de`, selecting only synchronous versus threaded commits. All ten trials
completed with exclusive-GPU monitoring enabled and no detected competing compute
processes. These are the final isolated runs, distinct from the excluded attempts
below. Medians and observed ranges come from the persisted trial records.

| Metric | Synchronous | Threaded |
|---|---:|---:|
| Fresh correction process, seconds | 18.754 (17.020–21.019) | 12.060 (11.947–13.883) |
| Correction stage, seconds | 16.846 | 10.132 |
| Sampled process-tree RSS, GiB | 1.521 | 1.542 |
| Sampled whole-device GPU peak, GiB | 3.967 | 3.967 |
| CUDA correction allocation peak, GiB | 1.929 | 1.929 |
| Physical process-tree writes, GiB | 0.3880 | 0.3881 |
| Sampled output allocation, GiB | 0.5313 | 0.5313 |
| Maximum pending payload, MiB | 0 | 12.469 |

Median process time decreases **35.7%**, with threaded commits faster in all five
pairs. The correction-stage median decreases **39.9%**. This is evidence for this
durable-write workload on the local storage device; it is not a claim about other
clips or faster storage. The measured RAM increase is small, while the exact
pending-buffer counter establishes the additional payload bound. RSS and device
peaks are sampled and include broader process/context allocations; whole-device
memory also includes desktop graphics. Output allocation includes prepared input
fixtures and mappings, separately from cumulative physical I/O.

Median inclusive write time is 7.115 seconds synchronously and 4.708 seconds in the
worker; median main-thread wait for background commits is 0.625 seconds. These
overlapping measurements cannot be added or interpreted as a stage-isolated disk
benchmark. Scheduling also changes observed writer latency; the experiment does
not attribute every saved second to one particular syscall or transfer.

## Full-pipeline measurements

All five full-pipeline pairs completed, also without detected competing compute.
They include fresh decoding, DA3 construction/inference, AllTracker, correction,
durable outputs and cleanup. Every pair favors threaded commits.

| Median metric | Synchronous | Threaded |
|---|---:|---:|
| Full process, seconds (observed range) | 58.041 (57.858–58.538) | 53.592 (53.437–53.802) |
| DA3 stage, seconds | 35.772 | 35.702 |
| AllTracker stage, seconds | 4.206 | 4.156 |
| Streamed correction stage, seconds | 13.861 | 9.459 |
| Sampled process-tree RSS, GiB | 12.578 | 13.256 |
| OS main-process RSS high water, GiB | 13.545 | 13.545 |
| Sampled whole-device GPU peak, GiB | 12.489 | 12.489 |
| Physical process-tree writes, GiB | 0.5857 | 0.5857 |
| Sampled output allocation, GiB | 0.5854 | 0.5854 |

The full-process median decreases **7.7%** (4.448 seconds). Its correction-stage
median decreases **31.8%** (4.402 seconds), while DA3 and tracker medians remain
close. This supports a pipeline gain from the changed commit schedule. Stage
medians are not additive decompositions of process medians, and timings from the
isolated correction context should not be substituted for pipeline stage timings.

The sampled RSS medians differ because the 100 ms sampler catches the brief DA3
peak to varying degrees. OS high-water records are essentially identical and
already reached in DA3, before the commit owner exists. The sampled median
difference is not evidence that the writer retains an additional 0.68 GiB. The
exact pending-payload bound and correction-only memory measurements above describe
its local storage cost more directly.

## Correctness and decision

All **22 full-output comparisons pass with bitwise equality**, covering ordinary
and repeated-point XYZ, raw lift, masks, image pixels and geometry. This includes
all timing repeats, the previous implementation's default behavior, and the three
composition checks (observation caching; caching plus prefetch; and caching,
prefetch, eager CUDA rays plus native packing/writing). These composition runs
establish compatibility, not five-trial performance distributions.

The CPU suite passes 157 tests, including queue bounds, buffer ownership,
exceptions, verified resume, corrupted-shard rejection and GPU-monitor mocks.
The CUDA suite passes 54 tests, including eight threaded-commit combinations.
Ruff formatting/lint and Pyrefly checking pass for all first-party Python with zero
type errors or warnings. Live WebGL interaction is not established by these CPU,
CUDA and Node checks; the charts are exported as standalone Plotly HTML.

Keep synchronous commits as the default and expose threaded commits as an
independent, tested option. The measured benefit is substantial here, with a
bounded additional host buffer and unchanged GPU footprint, storage format and
integrity policy. More workloads and storage devices are needed before making it
a universal default. DA3 now accounts for roughly two thirds of pipeline runtime,
so profiling its construction and execution is the next larger performance target.

Technical evidence: [summary](../benchmarks/background-commits/summary.json),
[trial records](../benchmarks/background-commits/trials.json),
[paired observations](../benchmarks/background-commits/pairs.json),
[commit counters](../benchmarks/background-commits/commits.json),
[numerical checks](../benchmarks/background-commits/quality.json) and
[source fingerprints](../benchmarks/background-commits/provenance.json).
The [standalone chart viewer](../benchmarks/background-commits/index.html) compares
runtime, host/device memory, disk writes, allocated output and OS RSS high water.

## Reproduce

Historical sources and raw trial directories are immutable. On a fresh checkout
containing the published stack history (use `--repository outputs/pr-stack` for
this workspace's isolated Git clone):

```bash
uv run python scripts/prepare_optimization_sources.py \
  --revisions configs/experiments/background-commits-revisions.json
uv run python scripts/run_optimization_experiments.py \
  --spec configs/experiments/background-commits.json --experiment correction
```

Provision the pinned weights and prepared window first, and choose a fresh output
root. Run `pipeline`, `combinations` and
`defaults` as separate experiments. `--resume` reuses existing records and is not a
new measurement.

To run every scope sequentially and scan outputs only after timing finishes:

```bash
uv run python scripts/run_background_campaign.py \
  --spec configs/experiments/background-commits.json
uv run python scripts/publish_background_commits.py \
  --campaign outputs/experiments/background-commits-isolated \
  --roles configs/experiments/background-commits-roles.json \
  --out docs/benchmarks/background-commits
uv run python scripts/build_optimization_charts.py \
  --summaries docs/benchmarks/background-commits/summary.json \
  --analysis docs/native-optimization-experiments/06-background-commits.md \
  --out-dir docs/benchmarks/background-commits
```

## Excluded attempts

The earlier directories `outputs/experiments/background-commits` and
`outputs/experiments/background-commits-measured` are retained exploratory records.
The first also spans edits to reporting-only scripts. Both ran while separate host
vLLM and Ollama processes occupied GPU memory, discovered during the first
full-pipeline pair. That pipeline attempt was interrupted and retains partial
outputs. Its correction comparisons passed, but its timings are excluded from
isolated performance conclusions. The user authorized stopping both host servers.
The final campaign uses `outputs/experiments/background-commits-isolated` with the
ownership guard enabled throughout.
