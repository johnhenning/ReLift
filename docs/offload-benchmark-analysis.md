# Analysis of the offload validation campaign

This is a reviewed interpretation of the records under `outputs/benchmarks/offload-primary`,
`offload-ablations` and `offload-capacity`. It is separate from the technical report renderer.
The generated report supplies the underlying trial dates, hardware metadata, settings, timings,
memory measurements, numerical comparisons and source links. Reassess this analysis before
combining it with measurements from a different campaign.

## What the paired comparison establishes

The primary experiment compares the frozen implementation with RAM offload on swing-tire,
using five fresh-process trials per mode, the complete 84-frame input, processing resolution
504, groups of 1,024 and eight groups per corrector batch. Mode order alternates between repeats.
Inputs and weights were provisioned beforehand; the OS page cache was retained.

The original median is about 36.2 seconds and the RAM median about 60.9 seconds: approximately
1.68 times the wall time. Sampled total device peak falls from about 19.5 GiB to 12.4 GiB,
a reduction of about 36%. This implementation improves capacity and incurs transfer and storage
overhead. These measurements do not establish a speed improvement or a general latency
distribution across different videos.

One initial baseline attempt briefly overlapped with copying completed gallery data. That
attempt is retained under `outputs/benchmarks/excluded-gallery-copy` and was replaced with an
isolated run. Prototype measurements in `offload-pilot` are excluded from the primary timing table.

The candidate includes mapped conversion, integrity scans and per-shard fsync costs. The original
NPZ writer has a weaker commit policy. The measured difference therefore includes both placement
and storage/commit costs; a storage-only ablation would be needed to isolate them fully.

The exploratory streamed-only run takes about 49.2 seconds, forced disk paging 197.4 seconds,
and automatic placement 51.8 seconds. Automatic placement selects streamed correction for this
short clip. Forced disk paging writes about 12.7 GiB of cold DA3 tensor archives and provides
the same sampled GPU peak as RAM paging. On this SSD it is a costly capacity option. These are
single runs, so the differences between streamed and automatic mode are not a repeatable speed claim.
System-wide swap counters also show traffic during the disk experiment; sampled benchmark-process
swap is zero, which does not establish that the host experienced no memory pressure.

## Long-clip capacity

The fresh original running and forest trials fail in DA3 with CUDA OOM. RAM offload completes
running's 206 frames and forest's 294 frames at the full processing resolution. Their H×W grids
are 504×280 and 280×504 respectively, retaining 141,120 reference-pixel trajectories in each clip.
Neither temporal frames nor inference pixels were dropped to obtain these results.

These are single capacity trials, not paired timing distributions. Their captured source hashes
precede the final provider cleanup change, which releases model references deterministically after
the DA3 stage. An OOM duration is not a successful baseline runtime and does not define a speedup.
The previously completed gallery scenes are reused artifacts, not additional controlled offload trials.

## Numerical behavior

The initial complete swing-tire comparison found bitwise equality for ordinary corrected tracks,
raw lift, visibility masks, processed images, cameras, intrinsics, prepared depth and tracker arrays.
The report's numerical tables contain the later validations and their exact provenance.

The old CUDA scatter has undefined ownership when padding repeats a point into another group.
Streamed correction keeps the first occurrence deterministically. In the pilot this affects
192 of 141,120 reference pixels (0.136%). Their coordinates can differ substantially: the maximum
pilot discrepancy is 0.207 times the reference depth scale. They are explicitly reported separately
from the ordinary-point parity gate. This is a behavior change; the complete dense result is not
claimed to be bitwise identical. A baseline-repeat comparison using the same recorded pixel policy
distinguishes that existing nondeterminism from changes to ordinary trajectories.

Long clips exceed the published 84-frame evaluation setting. Completing them does not itself
establish trajectory accuracy, even when all arrays are finite and consistent.

## Memory and recovery limits

The budgets govern retained cold tensor storage, estimated staging/output capacity and CUDA
allocation checks. They do not impose OS process-RSS or GPU-reservation ceilings. Forest's sampled
device and process-RSS peaks exceed the nominal placement targets because reserved CUDA workspace,
provider outputs and file-backed pages contribute additional memory. Its retained cold tensors
and CUDA allocations fit their configured limits. The measured total peaks remain important when
deciding whether another machine can run the same workload.

Disk mode retains the original tensor dtypes in mapped archives. It still consumes RAM through
resident file pages and the OS page cache. No cold-cache conclusion follows from these experiments.
The active DA3 attention layer and AllTracker must still fit on CUDA.

The new trajectory store commits temporary files with flush/fsync and atomic rename, verifies
committed shards on resume and rejects mismatching stored provenance. Parent-directory fsync
and a machine-crash durability guarantee are outside this implementation. Provider window caching
inherits the existing pipeline's reuse assumptions: use a fresh output root when changing provider
configuration or weights, and treat an explicitly supplied prepared window as a chosen input.
The frozen benchmark uses fresh inference outputs with fixed pinned providers.

CPU tests cover mapped round trips, corruption rejection, partial resume and provenance, decoder
identity, viewer coverage and the actual pinned DA3 tiny backbone in RAM and disk modes. Node tests
cover viewer controls and component-to-code selection. Live browser/WebGL rendering was unavailable
for automated verification in this session.

## Remaining experiments and performance work

The disk, streamed and automatic-mode runs are exploratory ablations. The larger workload-by-mode
matrix, storage-only isolation, prepared-stage timing, cold-cache experiments and detailed profiler
traces remain follow-up work. Single-trial results should not be described as repeatable speedups.

The initial streamed correction stage is substantially slower than dense correction. Profile repeated
depth-image transfers, archive-to-NPY preparation, integrity scans and per-shard commits first.
Then test bounded depth reuse and one-buffer transfer overlap while keeping the same point groups.
Also measure DA3 model construction and AllTracker's discarded pretrained initialization before its
released checkpoint is loaded. Keep each performance patch separate from the placement comparison;
`docs/performance-opportunities.md` describes those experiments.

## Reproduction commands for this campaign

Use fresh output roots for new measurements; the harness refuses to overwrite measured directories.
`--resume` reuses recorded trials and is not a new inference measurement.

```bash
uv run python scripts/benchmark_offload.py --scenes swing-tire --modes existing ram \
  --repeats 5 --output-root outputs/benchmarks/offload-primary
uv run python scripts/benchmark_offload.py --scenes swing-tire --modes streamed disk auto \
  --repeats 1 --output-root outputs/benchmarks/offload-ablations
uv run python scripts/benchmark_offload.py --scenes running forest --modes existing ram \
  --repeats 1 --output-root outputs/benchmarks/offload-capacity
uv run python scripts/compare_offload.py BASELINE_OUTPUT CANDIDATE_OUTPUT --out parity.json
uv run python scripts/publish_offload_results.py --scenes running forest
uv run python scripts/generate_all_views.py --videos running forest
uv run python scripts/capture_benchmark_environment.py
uv run python scripts/report_offload_results.py \
  --benchmark-roots outputs/benchmarks/offload-primary outputs/benchmarks/offload-ablations outputs/benchmarks/offload-capacity \
  --parity-files outputs/benchmarks/offload-pilot/parity*.json outputs/benchmarks/offload-primary/parity*.json outputs/benchmarks/offload-ablations/parity*.json \
  --environment outputs/benchmarks/shared-environment.json \
  --analysis docs/offload-benchmark-analysis.md --out outputs/benchmarks/offload-report.md
```

The frozen runnable original is under `outputs/benchmarks/baseline-source`. Each trial records its
command, source hashes, lockfile digest, seed, log, telemetry, stage metrics and outcome. Both variants
share the same locked environment and pinned provider checkpoints. The shared environment inventory
records actual checkpoint checksums outside the timed experiments.
