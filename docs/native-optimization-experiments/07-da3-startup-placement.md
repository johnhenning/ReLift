# CPU-first DA3 startup placement

This experiment follows bounded background shard commits. It changes initial
memory placement, independently of weight packing, feature archives, observation
caching, prefetch and output commits. The reference is stack revision
`46ae687a6c6e4d12a9c1bf655c06875269178984` (PR #17).

## Hypothesis and scope

The current provider loads the checkpoint on CPU, moves the complete model to
CUDA, then installs the pager. The pager immediately copies every transformer
block back to CPU. Those blocks are uploaded again when their forward hooks run.
The initial round trip costs time and temporarily allocates the complete model
on CUDA even though most weights are meant to remain off-device.

The refreshed reference completed a full 84-frame DA3 run and full pipeline before
implementation. Its separate startup profile attributes approximately 4 seconds
to module transfers and 10.5 seconds to random initialization. These inclusive
diagnostic spans overlap; they are not headline timings. Avoiding random
initialization is a separate future experiment, with additional checkpoint and
buffer-materialization risks.

The candidate loads the same complete checkpoint through the same upstream API.
It establishes cold block storage and hooks while the model is on CPU, then moves
only unpaged parameters and buffers to CUDA. It keeps the same parameter objects,
head slices, attention, dtype boundaries, reference selection and full temporal
context. There is no new native operator or change to numerical arithmetic.

## Configuration, injection and ownership

`[da3] startup = "cpu-first"` selects the candidate. `"current"` remains the
default. The `cpu-first-da3` profile changes only this field relative to `ram`;
`cpu-first-combined` inherits observation caching and threaded commits from
`background-cached`. Legacy experiment settings expose `da3_startup` as the
adapter to the grouped configuration. Unsupported non-paged modes fail explicitly.

The existing fingerprinted `InferenceComponents.pager_factory` receives the
startup policy; provider injection and session ownership remain available.
Cold parameters retain their CPU/mapped owners until an individual block runs.
Unpaged placement excludes both cold parameter identities and owned buffers.
Installation failure removes hooks and temporary archives through pager cleanup.
Parking and reactivation retain the existing per-clip lifecycle.

Size counters report cold parameter/buffer bytes and the initial cold tensor
round trip implied by the placement path. They are allocation/transfer accounting,
not a PCIe hardware measurement. Actual latency and CUDA allocator peaks are
recorded separately.

## Measurement and acceptance plan

Keep weights locally provisioned, HF offline and OS page cache retained. Freeze
the candidate source, then use that one source for both configurations. Run one
full-clip screening pair and require exact DA3 artifacts before timing five
fresh-process pairs, alternating order. Compare provider construction, forward,
cleanup and complete DA3 process time separately.

Measure five fresh full-pipeline pairs at resolution 504, all 84 frames,
1,024-point groups, batch 8 and seed 0. Check every output, including padded
repeated pixels, raw lift, masks, prepared arrays, cameras and image pixels. A
comparison against the preceding source verifies unchanged default behavior.
Single full-length running/forest pairs establish capacity and output parity,
not latency distributions. Exercise disk placement, packed storage, reuse and
composition separately from headline timing.

Record sampled process-tree RSS/PSS, OS RSS high water, whole-device GPU memory,
CUDA allocation/reservation, physical disk I/O and peak/final output allocation.
Keep expensive profiles and parity scans outside unprofiled timing. An occupied
compute GPU prevents a trial; detected interference invalidates it. Retain failed
and exploratory records. Report medians, ranges and paired differences; do not
combine historical timings into the fresh comparison.

Tests cover exact tiny-backbone state and repeated outputs, auxiliary features,
reference selection, CPU/CUDA residency, RAM/disk and packed layouts, configuration
round trips, injected factories and partial-installation cleanup. The measured
results and decision below describe the accepted scope; technical report scripts
contain no experiment conclusions.

## Repeated DA3-only measurements

Both configurations use frozen revision `35d34aca396836e42a9e52747e0ca714abd93111`.
The only selected optimization difference is startup placement. All five fresh
control/candidate pairs completed with exclusive-GPU monitoring. Order alternates
between repeats, and the OS page cache is retained.

| Median metric | Current startup | CPU-first startup |
|---|---:|---:|
| Full DA3 process, seconds (range) | 38.868 (38.702–39.011) | 34.015 (33.673–34.207) |
| Provider construction/placement, seconds | 17.976 | 13.081 |
| Forward stage, seconds | 17.613 | 17.562 |
| Sampled whole-device GPU peak, GiB | 12.460 | 7.243 |
| CUDA allocated peak during construction, GiB | 6.358 | 0.957 |
| CUDA reserved peak during forward, GiB | 11.156 | 5.939 |
| Sampled process-tree RSS peak, GiB | 11.414 | 11.486 |
| Main-process OS RSS high water, GiB | 13.594 | 13.594 |
| Physical writes, GiB | 0.07730 | 0.07730 |
| Sampled peak allocated output, GiB | 0.07729 | 0.07730 |

Median complete-process time decreases **12.5%** (4.853 seconds). CPU-first wins
all five pairs. Construction/placement falls **27.2%**, while forward times remain
close. The result supports avoiding redundant placement work, not faster model
arithmetic. Sampled total GPU peak decreases **41.9%**. It includes the desktop
and allocator reservation; the separate allocated/reserved metrics show why it
must not be interpreted as the model's live tensor footprint.
Forward allocated peaks remain approximately 4.45 GiB in both configurations.

Host RSS high water stays effectively identical. CPU-first still constructs and
loads the complete CPU checkpoint through the upstream API. The small sampled
RSS difference does not establish a host-memory regression or saving. Output
contents, write policy and final storage format are unchanged. Physical reads
are small with these warm checkpoint pages; this is not a cold-cache study.

Both variants retain **5,743,324,160** cold parameter bytes. Their size counters
account for an initial **11,486,648,320-byte** CPU→CUDA→CPU cold-weight round trip
in the current path and zero in CPU-first. Forward hook transfer counts and bytes
are unchanged. This accounting counts logical tensors, not measured bus traffic.

## Repeated full-pipeline measurements

Five fresh pairs include decoding, DA3, AllTracker, correction, hashing, durable
output commits and cleanup. All clips retain all 84 frames at the original 504
processing resolution. Only DA3 startup placement differs between configurations.

| Median metric | Current startup | CPU-first startup |
|---|---:|---:|
| Complete pipeline, seconds (range) | 59.623 (58.393–62.745) | 55.312 (53.150–55.984) |
| DA3 stage, seconds | 35.970 | 31.215 |
| AllTracker stage, seconds | 4.250 | 4.408 |
| Streamed correction stage, seconds | 15.002 | 15.356 |
| Sampled whole-device GPU peak, GiB | 12.460 | 7.243 |
| Sampled process-tree RSS peak, GiB | 11.325 | 11.521 |
| Main-process OS RSS high water, GiB | 13.545 | 13.545 |
| Physical writes, GiB | 0.58574 | 0.58574 |
| Sampled peak allocated output, GiB | 0.58536 | 0.58536 |

Median process time decreases **7.2%** (4.310 seconds), with CPU-first faster in
all five matched pairs. The DA3-stage median decreases by 4.756 seconds. Other
stages vary, particularly synchronous correction/commit time. Stage medians do
not sum to a decomposition of the process median; changes outside DA3 are not
attributed to the placement mechanism. Runtime ranges and the individual paired
observations remain available alongside the summary.

All ten DA3-only artifact checks pass with byte equality. All **nine** full-output
pipeline comparisons pass with bitwise equality for ordinary and padded-point XYZ,
raw lift, masks, images, geometry and prepared arrays. These compare every
completed repeat against the first completed control, rather than selecting one
candidate output. Screening adds two exact provider checks outside timing groups.

## Compatibility, capacity and decision

All additional pairs pass exact output checks. Disk, reuse and long-clip checks
compare complete DA3 artifact bytes and verify the stored files independently.
Composition checks every corrected trajectory, raw lift and mask. These are
single validation pairs and do not establish workload-specific speedups.

| Validation | Coverage | Outcome |
|---|---|---|
| Earlier observation cache + threaded commits | All 84 frames and 141,120 trajectories | Bitwise full-pipeline equality |
| Forced disk placement | All 84 frames, same 1,020 scratch files and 13,677,707,984 archive bytes | Byte-identical DA3 outputs |
| Reused provider | Swing-tire 84 frames followed by swing 60 frames | Both clips byte-identical |
| Running | All 206 frames, 504×280 H×W | Byte-identical DA3 outputs |
| Forest | All 294 frames, 280×504 H×W | Byte-identical DA3 outputs |
| Default placement versus preceding PR #17 | Full 84-frame pipeline | Bitwise equality, including padded points |

The long inputs also complete with reduced observed device peaks: running is
19.386 GiB with current startup versus 13.993 GiB with CPU-first; forest is
21.949 versus 18.986 GiB. These are sampled peaks from single trials, not memory
distributions. They establish unchanged full-length DA3 outputs and observed
capacity in the affected component. Full running/forest tracker/corrector
pipelines were not rerun in this experiment. Active attention and features must
still fit on CUDA.

Keep **CPU-first startup as an independent opt-in improvement**. It exceeds the
proposed component and pipeline targets in the five-pair swing-tire comparison,
reduces observed GPU memory, and preserves exact outputs in every tested scope.
The default remains `current`; select `--profile cpu-first-da3`, or compose
`[da3] startup = "cpu-first"` into an existing RAM/disk profile. These results
come from one GPU/storage environment and one repeated performance workload.
Other workloads/storage policies need their own repeated timing comparisons.

The final CPU suite passes **172** tests with the existing native build. The
relevant CUDA suite passes **60**, including startup with Python/C++ weight and
feature packing, cold-buffer ownership and repeated activation. Ruff formatting,
lint, Pyrefly with warnings enforced, and the Node viewer checks pass. Static
chart images are inspected separately; these tests do not establish browser WebGL
rendering. Checkpoint verification and the hardware inventory are retained with
the raw campaign.

## Evidence and reproduction

Technical exports provide [summary rows](../benchmarks/da3-startup/summary.json),
[raw trial records](../benchmarks/da3-startup/trials.json),
[paired observations](../benchmarks/da3-startup/pairs.json),
[startup size accounting](../benchmarks/da3-startup/startup.json),
[quality records](../benchmarks/da3-startup/quality.json),
[diagnostic excerpts](../benchmarks/da3-startup/diagnostics.json),
[environment](../benchmarks/da3-startup/environment.json) and
[source fingerprints](../benchmarks/da3-startup/provenance.json).
The [standalone viewer](../benchmarks/da3-startup/index.html) compares runtime,
CPU/GPU memory, physical writes and output allocation. It embeds Plotly and the
analysis, so it works offline. Validation groups are excluded from performance
charts by the explicit role configuration.
Published diagnostic tables retain the top 100 functions by inclusive duration,
with full function counts and hashes of the complete raw profiles. Use
`--profile-functions` to select a different technical-table size. Full JSON and
binary cProfile records remain under the recorded local diagnostic paths.

Provision the pinned checkpoints and demo inputs first. On a fresh checkout,
reconstruct source variants from the revision map. Use
`--repository outputs/pr-stack` in this workspace's isolated Git clone:

```sh
uv run python scripts/prepare_optimization_sources.py \
  --revisions configs/experiments/da3-startup-revisions.json
uv run python scripts/capture_benchmark_environment.py \
  --out outputs/experiments/da3-startup-final/environment.json
uv run python scripts/run_optimization_experiments.py \
  --spec configs/experiments/da3-startup.json --experiment baseline-da3
uv run python scripts/run_optimization_experiments.py \
  --spec configs/experiments/da3-startup.json --experiment baseline-pipeline
uv run python scripts/run_da3_startup_campaign.py --scope screening
uv run python scripts/run_da3_startup_campaign.py --scope measurements
uv run python scripts/run_da3_startup_campaign.py --scope validation
```

The first two tasks are the preceding PR's baseline, before selecting the new
implementation. Our original local refresh lives under
`outputs/experiments/da3-startup/baseline-da3` and `baseline-pipeline`; the final
reproduction spec writes fresh baseline tasks into its selected output root.
They are individual validation runs and are excluded from headline distributions.
The repeated comparison uses one frozen source, selecting only startup placement.

Choose a fresh output root for new measurements. `--resume` reuses recorded
trials and does not produce a fresh benchmark. Historical sources and trial
directories are never overwritten. Profiles are diagnostic runs, separately:

```sh
uv run python scripts/profile_da3_startup.py \
  --source outputs/experiments/da3-startup-baseline-source \
  --profile ram --out outputs/experiments/da3-startup-profile-control
uv run python scripts/profile_da3_startup.py \
  --source outputs/experiments/da3-startup-source \
  --profile cpu-first-da3 --out outputs/experiments/da3-startup-profile-candidate
```

The publication command below uses the retained local campaign paths. Adjust
explicit input paths for a new campaign, including its baseline records and
default-behavior comparison. Report scripts combine recorded technical outputs;
interpretation stays in this document.

```sh
uv run python scripts/publish_da3_startup.py \
  --campaign outputs/experiments/da3-startup-final \
  --roles configs/experiments/da3-startup-roles.json \
  --diagnostics outputs/experiments/da3-startup-profile-control/profile.json \
    outputs/experiments/da3-startup-profile-candidate/profile.json \
  --environment outputs/experiments/da3-startup-final/environment.json \
  --extra-quality outputs/experiments/da3-startup-final/defaults-quality.json \
  --additional-trials outputs/experiments/da3-startup/baseline-da3/control/trial-01/trial.json \
    outputs/experiments/da3-startup/baseline-pipeline/control/trial-01/trial.json \
  --out docs/benchmarks/da3-startup
uv run python scripts/build_optimization_charts.py \
  --summaries docs/benchmarks/da3-startup/summary.json \
  --analysis docs/native-optimization-experiments/07-da3-startup-placement.md \
  --out-dir docs/benchmarks/da3-startup
```

## Excluded attempts

The early `outputs/experiments/da3-startup/screening` and
`outputs/experiments/da3-startup-isolated/screening` pairs were interrupted when
an unrelated `nsight-dl-designer/onnxruntime_perf_test` process appeared on CUDA.
The guard stopped our worker, retaining `gpu_interference` records and partial
outputs. Those timings are excluded. The external benchmark subsequently ended;
the complete final campaign uses `outputs/experiments/da3-startup-final` and
retains ownership monitoring. Brief activity between samples can still be missed.
