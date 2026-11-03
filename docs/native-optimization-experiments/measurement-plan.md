# Shared measurement and acceptance plan

Status: proposed. This plan adds no executable benchmark variants yet.

## Reference and isolation

Freeze the current optimized working source before changing it. Retain skipped discarded AllTracker initialization, float32 output views and hashing during NPY writes. Keep eager execution, group 1024, batch 8, seed 0, full clips and processing resolution 504. Depth caching and prefetch stay disabled. Use identical prepared inputs and checkpoint bytes across variants.

Record the frozen source manifest and actual Git revision where available. Do not rely on a branch name or the word `HEAD` to identify measured bytes. Freeze candidate variants separately. Use the same Python 3.13 environment and lockfile for every arm; capture the actual interpreter, installed package versions, GPU/driver, CPU, RAM, filesystem and storage device.

Native builds additionally record compiler and linker versions, source hashes, build flags, extension binary hash, build duration, architecture target and any extra libraries. Pin build dependencies with uv. Test against the project's installed PyTorch/CUDA versions rather than assuming the newest documentation describes that release. Keep a working Python fallback; choose the backend explicitly in trials and record which backend actually ran. A missing extension must not silently turn a native trial into a Python trial.

## Variants

Every experiment starts with these arms:

| Arm | Purpose |
|---|---|
| Current optimized Python | Measures the implementation users have now |
| Improved Python | Uses the proposed buffers/layout/packing through existing NumPy or PyTorch operations |
| Native equivalent | Uses the same layout, algorithm, memory bounds and synchronization as improved Python |

For a CUDA fusion experiment, distinguish a C++ wrapper around unchanged ATen operations from an actually fused kernel. For DA3, first compare storage layouts with Python, then compare implementations of the selected layout. Keep format/layout comparisons separate from language comparisons.

Do not combine experiments during screening. If a previously accepted change becomes the next control, freeze that control and rerun all arms against it. Final combined confirmation is a new measurement, not a sum of isolated savings.

## Workloads and timing

1. Refresh a diagnostic profile of the current control. Split gathering, host copies, transfers, CUDA work, packing, hashing and commit waits where relevant. Count allocations, bytes and calls. Profiled runs do not enter timing tables; inclusive spans cannot be summed.
2. Use full swing-tire: 84 frames, 280×504 H×W, all 141,120 reference pixels. Run the relevant component on identical prepared fixtures. Copy or provision fixtures outside timing, then include any preparation the component itself requires.
3. Run five fresh-process repeats per arm. Balance three-arm order with a recorded rotation, rather than always placing one candidate last. Retain all trials, medians, ranges and paired differences. Keep model jobs sequential on the GPU.
4. Confirm a passing candidate in five fresh full-pipeline pairs against its matching control, including decode, construction, inference, conversion and durable output commits. Exclude downloads, compilation and viewer export, reporting their costs separately. A first import or runtime initialization required by the backend stays in fresh-process timing.
5. Check full running (206 frames, 504×280) and forest (294 frames, 280×504) for coverage, capacity and numerical parity. Begin with one control/candidate pair each; repeat five pairs before making workload-specific speed claims. These longer runs are validation, not a reason to reduce resolution or temporal context.

Keep local weights and input files provisioned, HF offline and the OS page cache policy equal. Fresh output directories do not imply cold input pages. Record first creation and warm reuse separately for mapped archives. A true cold-cache study requires a separate reproducible procedure; do not label a run cold merely because it starts a new process. Avoid flushing global caches as part of this campaign.

## Measurements and charts

| Metric | Required measurement and interpretation |
|---|---|
| Runtime | Fresh-process wall time, component/stage time, first-use cost and steady-state time separately; synchronize at CUDA timing boundaries |
| CPU memory | Process-tree RSS/PSS sampled every 100 ms, OS main-process RSS high water, pinned bytes and explicit staging-buffer peaks; sample misses remain possible |
| GPU memory | CUDA allocated/reserved peaks and sampled whole-device usage; the latter includes desktop use and can miss brief peaks |
| Disk traffic | Process-tree physical reads/writes; count archive/shard bytes separately from I/O served by page cache |
| Disk capacity | Sampled peak logical and allocated bytes under **every** trial-owned output/scratch/cache directory, plus final retained bytes; sampling is a lower bound on short-lived peaks |
| Mechanism | Bytes copied/read/hashed, transfer and syscall counts, kernel launches and allocations where relevant; expensive tracing only in diagnostics |

Add OS RSS high-water reporting to workers that currently omit it. The existing correction worker's phase metrics record CUDA peaks but do not provide this CPU measurement. Native helpers must expose their owned buffer sizes; hidden native allocation cannot be inferred from Python allocation tracing. Explicit scratch counters should supplement sampled directory sizes when files are short-lived.

Produce standalone comparison charts for runtime, CPU memory, GPU memory, physical disk traffic and peak/final disk allocation. Show units, trial counts, ranges, failures and cache policy. Do not present an OOM duration as successful runtime. Keep the old original/offload comparison visibly separate from the fresh native experiment controls.

## Correctness and recovery gates

For data movement, packing and storage changes, require bitwise equality of array values, shapes and dtypes. Check all frames and pixels, including padded repeated pixels, invalid-reference pixels and the deterministic first-occurrence policy. Preserve masks, images, prepared tracker/depth arrays, poses and intrinsics exactly.

For fused floating-point arithmetic, target bitwise equality first. The existing engineering gate allows maximum vector error ≤1e-4 of reference median depth; apply it separately to ordinary XYZ, repeated XYZ and raw lift, with exact masks and unchanged dtypes/images. Report maximum, median and p99 errors, not just a pass label. Any change to that tolerance requires a separate quality decision. The earlier compilation and small-batch failures show that a faster time alone is insufficient.

Validate every completed timing repeat. Use `check_optimization_parity.py --all-trials` for supported tasks, extending it for new tasks rather than assuming provider artifact hashes can compare different archive formats. Compare decoded logical tensors when container bytes legitimately differ, and verify each container's recorded integrity checks independently.

Storage/lifecycle changes also require corruption, interruption, resume, missing/truncated file, budget failure and exception cleanup checks. Preserve existing flush/fsync, atomic replacement, final verification and provenance rules. Do not add a machine-crash durability claim: parent-directory fsync is not part of the present contract.

## Decision and implementation sequence

Initial review targets are a repeatable 10% reduction in the targeted component and at least a 1% fresh-pipeline reduction, both distinguishable from observed repeat variation. These are proposed decision thresholds, not predicted gains. A meaningful capacity result can justify a separate decision even without a speed gain; document its runtime cost. Stop early if the fresh profile makes the target implausible.

Keep native code only if it passes all gates and improves on the improved Python arm enough to justify build and maintenance cost. A component-only win without pipeline confirmation remains experimental. If the result is inconclusive, add balanced repeats rather than select the best run. Report regressions and rejected variants.

First add technical experiment options, worker tasks, provenance and metrics; then implement and test one candidate; then run correctness screening, component trials and pipeline confirmation. Reuse these existing tools where their task support fits:

- [Source freezing](../../scripts/freeze_benchmark_baseline.py)
- [Sequential trial runner](../../scripts/run_optimization_experiments.py) and [worker](../../scripts/experiment_worker.py)
- [Parity checker](../../scripts/check_optimization_parity.py)
- [Technical summary](../../scripts/summarize_optimization_experiments.py) and [standalone chart builder](../../scripts/build_optimization_charts.py)

The current runner hashes Python sources under `relift`; extend provenance to native sources, headers, build configuration and binaries before collecting trials. Its pipeline task currently fixes RAM offload; DA3 disk trials need an explicit mode option. New component tasks and three-arm order balancing are planned additions, not capabilities already implemented.

Store commands, hashes, telemetry, diagnostics, outputs and quality records in a fresh experiment directory. Technical summaries consume those records. Each proposal gets a separate results document describing interpretation and the keep/discard decision; do not bake narrative, dates, hardware descriptions or conclusions into report generators.
