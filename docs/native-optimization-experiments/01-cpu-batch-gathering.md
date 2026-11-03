# Experiment 1: CPU batch gathering into bounded buffers

Status: proposed. Start here after refreshing the current profile. Follow the [shared measurement plan](measurement-plan.md).

## Hypothesis and exact code scope

Coordinates and visibility are advanced-indexed from mapped arrays inside `MappedObservations.rays`, then `_transfer` calls `np.array` before creating a float32 CUDA tensor. A destination-oriented gather may eliminate a temporary copy and repeated allocations. NumPy indexing already executes native work; a C++ rewrite has no assumed advantage unless it reduces passes or improves access to mapped pages.

The target is the coordinate/visibility gathering and host-buffer preparation inside [MappedObservations.rays and _transfer](../../relift/streaming.py). This proposal does **not** rewrite the CUDA ray arithmetic. Extract a helper with the existing gathering behavior before replacing it. Keep depth-image transfers, nearest sampling, camera transforms and normalization unchanged.

The older 0.82-second inclusive rays span also includes GPU work and transfers. It is not an estimate of CPU gathering cost or the available saving.

## Candidate design and comparisons

Proposed helper contract: read mapped tracks `(T,H,W,2)`, visibility `(T,H,W)`, selected int64 pixel IDs and an optional single-frame selection; fill caller-owned float32 coordinate `(selected_T,N,2)` and visibility `(selected_T,N)` buffers in the same order. Duplicate IDs remain duplicated. Do not sort IDs for locality unless results are restored to exactly the original order and the reorder cost is included.

| Variant | Change |
|---|---|
| Current | Advanced indexing, `np.array`, existing tensor conversion/transfer |
| Improved Python | Reuse bounded destination buffers; test flattened NumPy gathering into destinations where the real strides permit it; preserve current transfer and dtype conversion semantics |
| C++ binding | One validated strided gather fills the same destinations without intermediate indexed arrays; no extra worker threads initially |

Use a small optional pybind11 extension. Inspect dtype, shape, strides and bounds under the GIL; retain the mapped-array owners; release the GIL only around native buffer access. Avoid automatic force-casting that silently copies the complete input. Support the real mapped dtypes explicitly, rejecting or using an explicit Python fallback for unsupported inputs. [pybind11 NumPy/buffer access](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html), [GIL handling](https://pybind11.readthedocs.io/en/stable/advanced/misc.html).

Keep staging to one batch and record exact owned bytes. Buffer reuse must wait for the existing device copy to complete. Pageable versus pinned staging is a separate factor: use identical memory type in the Python/native comparison. Test pinning or lookahead only in a later arm if the profile supports it; the earlier prefetch experiment did not improve runtime.

## Measurements and checks

Add a gather-only worker using actual saved IDs, including the reference-frame all-pixel call, normal 8192-point batches, the padded tail and invalid-reference batches. Record host gather time, allocations, copied bytes, page faults and destination size. Then measure gathering plus the unchanged upload and full correction; a CPU improvement may disappear behind transfers or page faults.

Run both fresh mapping/open and repeated access cases with an explicitly recorded warm page-cache policy. Do not mistake a warm mapped read for reduced physical disk demand. Include destination allocation once in fresh-process timing, reporting reuse timing separately.

Require bitwise coordinate/visibility equality before CUDA, then bitwise equality of rays and complete trajectories. Cover random/duplicate/empty IDs where supported, first/last IDs, full timeline and single frame, supported source dtypes, read-only maps, noncontiguous views, invalid IDs and owner lifetime. Unsupported stride patterns must produce a clear error or recorded fallback, not a hidden full-clip copy.

## Decision and deliverables

Stop if a fresh profile shows negligible gather/copy cost, or if the best Python gather is as fast as native code. Adopt only after the shared component and pipeline targets pass without increased unbounded RAM, changed GPU work or disk-capacity regression.

Deliver an optional helper/backend switch, CPU buffer/lifetime tests, five-repeat component results, full-output parity, pipeline confirmation and a separate analysis document. A negative result should retain the measurements and leave the Python path as default.
