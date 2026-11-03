# Experiment 3: fused CUDA ray construction

Status: proposed, after lower-risk CPU/storage experiments. Follow the [shared measurement plan](measurement-plan.md).

## Hypothesis and code scope

Ray construction currently normalizes tracked coordinates, samples depth with `grid_sample`, constructs normalized camera directions, rotates them into the reference coordinates, computes range and applies `nan_to_num`. A custom CUDA operator could reduce launches and intermediate tensor traffic.

The target is the GPU arithmetic in [MappedObservations.rays](../../relift/streaming.py), after coordinates arrive on CUDA. CPU coordinate/visibility gathering belongs to [experiment 1](01-cpu-batch-gathering.md). Keep that backend identical across this experiment's arms. Simply invoking the same ATen operations from C++ is a dispatch experiment, not a fused kernel.

The existing 0.82-second diagnostic rays span includes host gathering and depth uploads, so it does not establish the GPU fusion opportunity. First use a fresh CUDA trace to separate kernel time, launch overhead and transfer stalls.

## Candidate design and comparisons

| Variant | Change |
|---|---|
| Current | Existing eager PyTorch expressions and 16-frame depth blocks |
| Improved Python | Reduce unnecessary allocations or reuse outputs through existing supported operations, preserving operation order and the same depth blocks |
| C++/ATen control | Optional diagnostic arm dispatching equivalent unfused operations from a binding |
| Fused CUDA | One sampling/ray kernel per existing depth block, with only necessary outputs retained |

Register an optional custom PyTorch operator with shape/device/dtype validation and run it on the caller's current CUDA stream. Preserve device ownership and tensor lifetimes; no implicit device-wide synchronization or persistent full-clip depth cache. Record build flags and reject implicit fast-math transformations. Implement inference only initially. PyTorch describes the custom C++/CUDA operator route in its [official tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html); verify the APIs against the installed project version.

Preserve the existing `nearest`, `padding_mode="border"`, `align_corners=True` sampling behavior, including the normalization and unnormalization rounding path. Directly rounding original pixel coordinates is not assumed equivalent. Preserve normalization epsilon, rotation arithmetic, range calculation and `nan_to_num` semantics. Do not combine this change with another precision, batch, compiler or interpolation change.

Keep the same 16-frame source-depth staging. Return direction and distance arrays with the expected timeline/point order, while reusing the unchanged origin and visibility handling. The exact output layout can be optimized only if every arm pays the same required layout conversion and callers see identical semantics.

## Measurements and checks

Use already-uploaded fixtures for a kernel-only benchmark, then measure rays including upload, correction and fresh full inference. Include reference-frame all-pixel requests, normal batches, padded tails and long full timelines. Report CUDA-event kernel timings separately from synchronized wall time. Record launches, intermediate allocated/reserved bytes, host staging and transferred bytes.

Test boundary pixels, half-pixel nearest ties, out-of-bounds coordinates, single-pixel dimensions, noncontiguous inputs, zero/negative depths, NaN/Inf coordinates and depth, camera transforms and intrinsics. Derive expected behavior from the current PyTorch path, including exceptional values; do not substitute a cleaner mathematical convention.

Target bitwise intermediate parity. If arithmetic differs, apply the shared ≤1e-4 normalized maximum-vector gate to all complete outputs, including repeated pixels and raw lift, while retaining exact masks and metadata arrays. Check visibility and valid-reference selection exactly; a small sampling discrepancy can change group membership and invalidate the comparison. Use op validation where supported, plus full-output checks; operator registration tests alone do not establish numerical parity.

## Decision and deliverables

Stop if transfer or CPU gathering dominates and fusion cannot meet the shared targets. Reject a fast kernel that changes sampling, masks or the full-output gate. Earlier compilation and small-batch experiments failed numerical checks; they do not establish the cause of differences here, but make this gate essential.

Deliver an optional registered operator and Python fallback, build/import and CUDA stream tests, edge-case equivalence tests, kernel and pipeline results, and numerical error records. Do not add ray-stage savings to CPU gather savings; measure their eventual combination in a separate integrated run.
