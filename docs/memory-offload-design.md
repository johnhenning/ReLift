# ReLift Memory Offload Design and Benchmark Plan

Status: Approved design; core implementation and initial capacity validation completed November 2, 2026.

Implementation: [DA3 placement](../relift/da3_offload.py), [bounded correction](../relift/streaming.py),
[mapped storage](../relift/storage.py), [benchmark harness](../scripts/benchmark_offload.py).
Measured outcomes and remaining experiments are tracked in [benchmark results](offload-benchmark-results.md).
The sections below preserve the reviewed design and planned experimental scope; they are not a claim
that every proposed optimization or benchmark cell has been completed.

ReLift should keep active computation on the GPU, stage upcoming work in RAM, and keep reusable arrays and inactive state on SSD. The aim is to finish longer clips at the existing spatial resolution with bounded memory, while retaining the current inference semantics. Offloading may increase runtime; capacity and speed must be measured separately.

This proposal covers the DA3 plus AllTracker demo pipeline, dense trajectory correction, storage, and Plotly export. It does not change training, model weights, precision, tracker settings, or temporal context. Implementation and controlled benchmark runs follow review. The separate [performance investigation](performance-opportunities.md) ranks improvements that can be tested after establishing the baseline.

## Current behavior and evidence

The default pipeline uses DA3 at long side 504, AllTracker window 16 with four iterations, and corrector groups of 1,024 tracks with eight groups per batch and seed zero. It decodes all frames, runs each provider over the clip, loads complete observations onto CUDA, allocates a complete corrected trajectory tensor there, and writes a `tracks.npz` archive. Plotly subsequently loads complete trajectory arrays before selecting a small display subset. See [demo generation](../scripts/demo_video.py), [ray construction](../relift/rays.py), [dense correction](../relift/infer.py), and [Plotly export](../scripts/plotly_demo.py).

The local batch completed 16 of 18 clips at the default resolution. Running, with 206 frames, and forest, with 294 frames, failed inside DA3 with CUDA out-of-memory errors. Their logs reported process GPU use of approximately 22.78 and 22.97 GiB respectively at failure. These values describe failure snapshots, not measured peak memory for successful runs. Saving trajectory outputs to disk cannot by itself resolve this earlier model-layer allocation failure.

The host has an RTX 3090 Ti with approximately 24 GB VRAM, approximately 32 GB system RAM, and a Samsung 860 QVO SATA SSD. The environment uses Python 3.10.22 and PyTorch 2.11.0 with CUDA 12.8 wheels. This is a single-GPU design; SSD transfer performance should be measured on this machine rather than extrapolated from NVMe results.

Existing logs provide the following observations. The stage timers are single-run measurements, mostly rounded to whole seconds, and omit some decoding, output I/O, process startup, and HTML export. They are useful for choosing experiments, not for claiming a speedup.

| Clip | Frames | Processed grid H by W | DA3 stage seconds | DA3 forward seconds | AllTracker stage seconds | Corrector stage seconds |
|---|---:|---|---:|---:|---:|---:|
| Swing | 60 | 280 by 504 | 19 | 6.46 | 3 | 2.7 |
| Swing tire | 84 | 280 by 504 | 23 | 9.94 | 5 | 3.6 |
| Spin | 80 | 504 by 378 | 27 | 13.76 | 5 | 4.7 |
| Kangaroo | 100 | 280 by 504 | 26 | 12.56 | 5 | 4.2 |

The [observational inventory](../outputs/analysis/current-demo-performance.json) records input metadata, log hashes, working-tree hashes, versions, failures, and available timings. Regenerate it without running models:

```bash
uv run python scripts/analyze_demo_performance.py
```

The repository has uncommitted migration and demo changes. A Git commit ID alone does not identify the current implementation; benchmark provenance must include a runnable snapshot of the working tree and its dependencies.

## Goals and boundaries

The first implementation should preserve every source frame, the existing 504 processing resolution, camera conventions, dtypes, visibility gate, CUDA ray sampling, valid-point handling, seeded group membership, and model arithmetic. Smaller corrector batches can be explored separately after an equivalent fixed-batch comparison. Resolution reduction, temporal windows, frame dropping, quantization, and different checkpoints are separate quality tradeoffs and must not activate silently.

The default batch script currently contains a spatial-resolution retry policy. Controlled offload experiments must bypass it. A successful lower-resolution retry is not evidence that offloading solved the original workload.

Initial scheduling targets are 20 GiB GPU memory, 18 GiB RAM, and 64 GiB scratch space per active job. These are proposed budgets, not measured guarantees. GPU workspace, other processes, file-backed resident pages, and the OS page cache require headroom and telemetry. If a job cannot fit the configured policy, it should stop with its allocation estimates and failing stage.

## Storage and data access

Introduce a versioned `TrajectoryStore` interface with readers for existing `.npz` outputs and new `.npy` shards. Memory-mapped `.npy` files support reading and writing slices without loading a complete array into a Python allocation. Existing `.npz` members do not provide the same ordinary memory-mapping access path. [NumPy reference](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.open_memmap.html)

Store images and depth in frame order. Store correction inputs and outputs in track-major shards, with each track's full timeline contiguous. Persist both the reference pixel IDs and the exact seeded grouping permutation. This allows sequential group reads and writes without changing point identity. The reader exposes frame and pixel selections so consumers do not depend on the physical layout.

Preserve the baseline's storage conversions: corrected positions remain float32, raw lift retains its existing float16 export representation, observations remain boolean, and images remain uint8. The raw lift's float16 representation is for export; corrector initialization must still interpolate in the baseline computation dtype before that final cast. Preserve intermediate depth and tracker rounding at the same stage as the baseline. Avoid a new lossy conversion to make offloading fit.

The following sizes are calculated array payloads at a hypothetical 504 by 280 grid. They exclude headers, cameras, intermediate arrays, model weights, copies, and scratch state.

| Frames | Corrected float32 XYZ MiB | XYZ plus raw lift plus observation mask plus images MiB |
|---:|---:|---:|
| 84 | 135.7 | 248.7 |
| 100 | 161.5 | 296.1 |
| 206 | 332.7 | 609.9 |
| 294 | 474.8 | 870.5 |

The calculation is `T × H × W × 12` bytes for corrected XYZ and `T × H × W × 22` bytes for the four listed arrays together. Actual model activation memory is additional and can dominate these storage costs.

## Execution and offload policy

The three memory tiers have different roles:

| Tier | Contents | Transfer policy |
|---|---|---|
| GPU | Active model blocks, active trajectory group, necessary workspace | Retain frequently reused data while the job fits the GPU budget |
| RAM | Inactive weights when practical, current and next input buffers, recently used features | Bound staging buffers and reuse allocations |
| SSD | Source caches, completed stages, trajectory shards, inactive state that exceeds the RAM budget | Favor sequential reads and writes; avoid repeatedly paging hot tensors |

Implement the pipeline in this order:

1. **Decode and cache.** Decode into bounded buffers and persist frames once. A mapped source limits RAM use but does not automatically reduce GPU use when an upstream API materializes the entire clip. Record the complete frame count and FPS before inference.
2. **Prepare geometry and tracking.** Persist provider outputs as completed stages and release provider objects explicitly between stages. Integrate offload inside the upstream model adapters where necessary; replacing only the wrapper's input array is insufficient. AllTracker already extracts CNN features in chunks of 64 frames, but retains those features and receives the complete CUDA image tensor, so its total working set still needs profiling.
3. **Build observations by group.** Read the coordinates for the current group, construct rays on CUDA, and retain its full timeline. Keep the original full-image normalization, nearest-neighbor sampling, border handling, and camera transforms. CPU substitution for `grid_sample` is excluded because this repository documents different rounding behavior.
4. **Correct and commit.** Transfer the current group, run correction, then write its result to a track-major shard. Compute the raw interpolated lift in the same bounded grouping. Retain the full reference-frame depth statistics needed for identical clip normalization, without allocating all output trajectories on CUDA.
5. **Render from selections.** Read only the sampled display tracks and reference colours for Plotly. Validate dense data in bounded scans. Keep all temporal frames; spatial display sampling remains independent of inference density. Offer `.npz` as a streamed compatibility export and measure its cost separately.

Start with synchronous transfers to establish parity. Add one-buffer lookahead with pinned RAM and CUDA events only after correctness passes. A staging buffer cannot be recycled until the associated transfer or kernel finishes. Excess pinned memory can harm host memory availability; it counts toward the RAM budget. [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.11/notes/cuda.html)

## DA3 integration

DA3 currently moves the complete model to CUDA and retains selected backbone features for its heads. It alternates local and global attention, so independent temporal windows would alter the computation. The first adapter should move inactive weights and retained features to CPU, restore them when needed, and maintain the original dtype and operation order. Accelerate's CPU and disk weight-offload mechanisms are candidates, not a proven drop-in solution for this custom model. [Accelerate reference](https://huggingface.co/docs/accelerate/en/package_reference/big_modeling)

Weight placement and activation placement require separate implementations and measurements. Weight offload can free VRAM but does not automatically spill the active layer's Q, K, V, residuals, or workspace. Retained feature tensors can be moved after their producing operation only when their consumers and lifetimes are understood. Operations that access parameters directly outside a child module need explicit handling in any hook-based integration.

Use CPU RAM first for inactive weights. Enable disk-backed weight storage only when the configured RAM budget cannot hold them. Compare this forced disk mode against a CPU mode with the same model arithmetic. If an active layer still cannot fit after placement changes, report that limit; exact operation tiling or changed temporal context would be a subsequent proposal.

Inference already disables gradients and uses autocast in DA3. Training activation checkpointing is not an automatic solution to retained inference tensors.

## Completion and recovery

Each run manifest should include source hashes and FPS, provider and weight revisions, code and lockfile fingerprints, resolution, dtypes, normalization, grouping seed and permutation hash, budgets, array layouts, completion markers, and checksums. Cache reuse requires these inputs to match; frame counts and filenames alone are insufficient.

Write shards to temporary names, flush and validate them, rename them into place, then publish the manifest update atomically. Resume only committed shards; recompute partial work. Keep logical completion time and durable completion time distinct: mapped-array flushes and filesystem writes do not by themselves establish a power-loss durability contract. If durable commits are enabled, include data and manifest synchronization costs in both sides of that comparison.

Keep scratch state separate from checkpoints and permanent outputs. Preflight required space and remaining free space. Scratch cleanup may remove only files owned by the run. Test interruption, full-disk errors, checksum failures, stale cache reuse, and recovery after a partially written shard.

## Benchmark variants

Benchmark the unchanged implementation before adding other speed improvements. The following variants isolate storage, streaming, and model placement effects:

| Variant | Dense data path | DA3 model placement | Purpose |
|---|---|---|---|
| Existing | Current arrays and `.npz` | Current CUDA placement | Frozen reference |
| Storage | New store and mapped output, otherwise unchanged execution | Current CUDA placement | Isolate file format and serialization overhead |
| Streamed | Grouped observations and bounded corrector output | Current CUDA placement | Isolate dense-data GPU and RAM savings |
| RAM | Streamed store | Inactive weights and supported features in CPU RAM | Test original-resolution capacity and transfer overhead |
| Disk | Streamed store | Inactive state spills to SSD under a fixed RAM budget | Quantify physical disk traffic and capacity tradeoffs |
| Automatic | Budget-driven selection of validated modes | Chosen by policy | Validate practical defaults after individual modes are understood |

Use fixed group size 1,024 and batch eight in the primary comparison. Run a separate tuned-policy comparison for smaller batches, prefetch, or other changes. Keep original model and precision settings in every primary variant. A placement mode that cannot support a requested workload remains a recorded failure.

## Workloads and protocol

Use swing at 60 frames, cartwheel or swing-tire at 84, spin at 80 with a larger portrait grid, kangaroo at 100, running at 206, and forest at 294. The short cases support direct parity and latency comparisons; running and forest test whether full resolution becomes possible. The remaining catalog clips provide a final capacity and artifact-coverage check.

Begin with three repeated baseline runs and storage/streaming trials on one 84-frame clip. Once parity holds, use five paired trials per successful workload and variant, with randomized or alternating execution order. Report medians, individual trials, and spread; five trials do not support a meaningful tail-latency claim. Confirm baseline capacity failures in fresh processes, retaining failure time and stage rather than assigning an invented successful runtime.

Before running trials:

1. Freeze a runnable baseline snapshot including untracked project files, lockfile, configs, pinned upstream sources, and weight hashes. Keep the snapshot and candidate outputs in different roots. Record the working-tree fingerprint rather than identifying the baseline only by `HEAD`.
2. Provision weights and videos once, then perform timing without network downloads. Record GPU, driver, CUDA, CPU, RAM, storage, thread counts, environment variables, and background GPU use. Do not benchmark concurrently with demo generation or export.
3. Disable automatic resolution and temporal changes. Use the same input bytes, full frame count, processing resolution, checkpoint, precision, group seed, and CUDA backend settings for matching trials.
4. Separate fresh-process end-to-end latency from a long-lived worker's steady-state throughput. Record model initialization and compilation costs explicitly. Do not subtract startup from one mode while including it in another.
5. Measure fresh inference into an empty output root, stage-isolated trials on identical prepared inputs, viewer-only export, and validated resume as distinct scenarios. A cache hit is not a faster inference run.

For example, the existing baseline can be invoked explicitly with:

```bash
uv run python scripts/demo_video.py cache/demo_inputs/swing-tire.mp4 \
  --num_frames 0 --process-res 504 --corrector-batch 8 --no_video \
  --out outputs/benchmarks/baseline/swing-tire/trial-01
```

The benchmark runner, storage modes and budget CLI are now implemented. The storage-only ablation remains
separate future work; the first candidate combines mapped storage with streamed correction. RAM and disk
modes additionally page DA3 blocks and retained features, using a custom adapter for the pinned upstream.

Automatic placement uses conservative estimates of DA3 resident and cold state to choose streamed,
RAM or disk mode. The estimates are scheduling heuristics, not hard OS limits. RAM mode checks live
retained cold tensor storage; disk mode bounds cold archive bytes and checks free space. Streamed
correction checks its batch estimate, output size and CUDA allocated high water. GPU reserved memory,
other libraries, initialization, OS file caches, decoded inputs and provider outputs require additional
headroom. A configured 18 GiB RAM budget does not promise process RSS below 18 GiB. AllTracker and the
active DA3 attention layer still require their live computation to fit on CUDA.

## Metrics and instrumentation

| Area | Record | Interpretation |
|---|---|---|
| Completion | Outcome, failing stage, OOM or timeout, frames and grid | Capacity precedes speed comparisons |
| Time | Process wall time; load, decode, provider, ray, correction, write, hash, export and durable-commit time | Distinguish computation from transfer and startup |
| GPU | Peak allocated and reserved CUDA memory; sampled total device memory, utilization and temperature | Allocator peaks and sampled device use describe different things |
| RAM | Process-tree RSS and PSS, file-backed and anonymous memory, system available RAM | Include mapped pages, workers, and staging buffers |
| Host pressure | Major page faults, swap traffic, CPU time and threads | Offloading should not merely move failure into host swapping |
| I/O | Logical bytes requested, physical read/write bytes, scratch peak, durable bytes and disk busy time | Warm OS cache can make a disk-backed mode behave like RAM |
| Transfers | Host-to-device and device-to-host bytes and time, overlap, GPU idle time | Explain the cost of placement choices |
| Output | Data size, HTML size, full frame coverage, sample identity and integrity | Preserve usable, comparable deliverables |

Use a monotonic wall clock and synchronize CUDA at the measured stage boundaries. For overlapped execution, use CUDA events with explicit stream dependencies and a final synchronization; summing overlapping stage times overstates end-to-end time. PyTorch's profiler can capture CPU/CUDA activities and memory, but profiled runs should be separate from headline timing runs because instrumentation adds overhead. [CUDA timing guidance](https://docs.pytorch.org/docs/2.11/notes/cuda.html), [profiler reference](https://docs.pytorch.org/docs/2.11/profiler.html)

Sample process and device telemetry at a documented interval, initially 100 ms. Sample the aggregate process tree at each timestamp rather than adding peaks reached by different processes at different times. Report this as sampled telemetry; CUDA allocator peaks provide a complementary unsampled high-water mark for PyTorch allocations.

Run with a warm filesystem cache first. For disk-offload experiments, add a separate cold-cache study only with a reproducible, recorded cache policy; do not silently drop global system caches. Compare logical reads with physical bytes and faults. Distinguish filesystem cache, prepared inference caches, model caches, and compilation caches in every result.

Write proposed benchmark artifacts as a manifest, per-trial JSON records, a flat CSV summary, raw logs, telemetry samples, profiler traces for diagnostic trials, and the numerical parity report. A successful-run speed ratio is `existing wall time / candidate wall time`, evaluated only for matching completed workloads. Report OOM-to-success transitions separately as capacity gains.

## Correctness and quality comparison

Require exact frame counts, reference pixel IDs, grid dimensions, dtypes, visibility masks, grouping, and cache provenance. Storage-only round trips should preserve serialized values bit for bit. Check depth, cameras, intrinsics, raw lift, and corrected coordinates separately so a camera or preprocessing change cannot hide inside final errors.

For placement and streaming, target identical numerical outputs on the same GPU and settings. First measure repeated-baseline variation, then freeze tolerances before inspecting candidate results. A proposed upper bound for ordinary points is `1e-4` of the baseline reference median depth for maximum coordinate-vector error, with median and p99 errors also reported; this is an engineering parity threshold, not a claim about metric-world accuracy. Any weaker bound requires review.

The current corrector repeats some points to pad the final batch and documents undefined write order for those duplicates. Use `repeated_pixels` to report those points separately; do not hide them in an aggregate error statistic. Define deterministic ownership for the new store, document the difference, and compare the unaffected points strictly. Preserve the CUDA-generated permutation rather than replacing it with a CPU random generator.

Do not align predictions with a fitted transform before parity comparison; that could hide a coordinate-frame error. Do not compare an 84-frame baseline with the first 84 frames of a 206-frame candidate: DA3's global temporal context differs. Where the baseline cannot finish the complete clip, test both paths on identical shorter windows for parity, then assess full-length capacity and internal consistency separately.

The demo videos have no ground-truth 3D trajectories. Absolute tracking quality requires a fixed WorldTrack subset with the same scoring seeds and metrics, including all-point and dynamic-point APD3D and trajectory error. Run that quality check if any change alters numerical behavior, precision, context, or sampling; agreement on demonstration videos alone is insufficient.

## Proposed acceptance gates

- Every previously successful clip retains its original frame coverage and resolution, and passes the agreed parity checks.
- Running and forest either finish at the original resolution within the configured budgets or produce a precise remaining capacity limit. Smaller-resolution success does not satisfy the original-resolution capacity gate.
- Automatic mode targets no more than 10 percent median fresh-run latency overhead on the ordinary 84-frame workloads that already fit. This is an initial review target, not a measured result. Forced offload modes report their slowdown rather than promising a speedup.
- Report actual VRAM, host memory, scratch usage, physical I/O, and recovery behavior. Budget overruns require an explanation or a policy failure; allocator reservation and OS page cache must not be omitted.
- Interrupted jobs resume to a valid complete result, stale inputs are rejected, and viewers contain every temporal frame and the same selected pixel identities.

## Implementation sequence and review decisions

1. Freeze the reference and add the benchmark harness and telemetry, without changing inference.
2. Add the storage interface and mapped readers, preserving `.npz` compatibility; measure storage and renderer behavior.
3. Stream CUDA observation construction and corrector groups; validate normalization, grouping, duplicate padding, and resume behavior.
4. Add DA3 CPU placement and retained-feature handling; test a short clip before the long failures.
5. Add optional disk backing, bounded prefetch and automatic selection after synchronous modes pass.
6. Run the comparison matrix and evaluate the separate performance experiments individually.

The main decisions for review are the initial budgets, the parity threshold, the acceptable automatic-mode overhead, and whether disk-backed model weights belong in the first implementation or follow CPU offload. The recommended first delivery is storage plus streaming plus CPU offload, with disk spill for arrays and cold retained state; model-weight paging to SSD follows only if RAM capacity requires it.
