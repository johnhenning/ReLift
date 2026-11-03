# Experiment 4: DA3 archive packing and block loading

Status: proposed. This is a storage/layout experiment before it is a C++ experiment. Follow the [shared measurement plan](measurement-plan.md).

## Hypothesis and code scope

`ColdStorage.put` writes individual tensors with `torch.save` in disk mode and reopens them as mapped tensors. `DA3Pager.page_block` stages parameters and buffers separately before each block, then restores CPU references. A per-block packed representation may reduce file handling and transfer-call overhead while preserving tensor values and memory bounds.

Targets: [ColdStorage.put, DA3Pager.page_block and cleanup](../../relift/da3_offload.py). Preserve backbone traversal, camera/reference selection, active attention, head slices, retained feature values and model precision. Use the existing model construction path; this is not a new checkpoint format or a whole-model LibTorch port.

The original forced-disk offload trial was much slower than RAM placement, but it was a single exploratory trial. It motivates profiling, not an expected native speedup. Packed parameter loading and retained-feature archives have different access patterns and must be measured separately.

## Candidate design and comparisons

Run two sub-experiments sequentially:

1. **Block weights and buffers:** compare current per-tensor storage/transfers with Python-created per-block, per-dtype packed regions. Compare a native loader only after the packed Python layout passes.
2. **Retained features:** compare current per-tensor feature files with a bounded packed feature archive. Keep this separate from the weight result before testing a combination.

| Variant | Purpose |
|---|---|
| Current Python layout | Existing files, maps and per-tensor uploads |
| Packed Python layout | Same values in validated contiguous regions, mapped views and fewer transfers where possible |
| Native packed loader | Same packed layout and transfer plan, with native validation/view construction or staging only where profiling supports it |

A versioned index records dtype, shape, strides, offset, length and integrity information. Preserve real aliases/shared storage where present, alignment and buffer ownership; reject unsupported layouts rather than silently duplicating large storage. Keep CPU maps and index owners alive until all tensor views are released. Pybind11 supports explicit buffer shape/stride metadata, but ownership must be designed rather than inferred. [Buffer protocol documentation](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html)

Pack parameters and buffers by dtype into bounded block regions. A packed GPU region can provide parameter views for one active block; restore the exact original CPU references after the forward, including exceptions. Include all live region bytes in the VRAM budget. No lookahead block or pinned duplicate is added initially. Retained feature archives remain CPU/mapped until the current traversal requests them.

Packing is real work: include archive creation in first-use stage/process timing. Report later reopening separately. No compression, quantization, reduced context or weaker validation is allowed. Preserve scratch cleanup and fail before exceeding configured capacity; record actual archive overhead as well as tensor payload bytes.

## Measurements and checks

First profile archive creation/open/map calls, page faults, CPU packing, staging allocations, weight-transfer call count/bytes and block forwards. Separate warm filesystem reads from physical I/O. Diagnostic counters must include buffers, not only parameters, and must not double-count aliased storage.

Run five fresh-process comparisons per arm in RAM and forced-disk mode separately. Use the same input/checkpoints/scratch filesystem and fixed budgets. Begin with a tiny pinned DA3 backbone for lifecycle checks, then full swing-tire, followed by running/forest capacity validation. Archive construction is included for first-use trials even when inputs and checkpoint downloads are provisioned. Multi-clip provider reuse stays disabled during the main comparison; any later reuse study gets a separate throughput result.

Measure DA3 construction, archive preparation, forward and cleanup, then full inference. Record physical reads/writes, peak/final scratch, file/map counts, sampled RSS/PSS, OS RSS high water, pinned bytes, CUDA peaks and whole-device samples. A packed format can lower file count yet increase transient RAM or copied bytes; show both outcomes.

Require exact parameter/buffer values and logical retained features after packing and reopening. Compare reference selection, auxiliary features, processed images, depth, poses and intrinsics, then complete corrected trajectories. Packed container bytes will differ from per-tensor `.pt` files: compare tensor contents and independently verify each format's checksums rather than expecting identical archive hashes.

Extend the DA3 RAM/disk tests with shared storage, mixed dtypes, truncated/index-corrupt archives, invalid offsets/overlaps, checksum failures, budget rejection, map lifetime and exceptions inside a block. Confirm hook removal, CPU-reference restoration and scratch cleanup. The current temporary archives are not resumable checkpoints; preserve that lifecycle rather than claiming new recovery guarantees.

## Decision and deliverables

Accept a layout improvement in Python if it captures the gain. Add a native loader only if it beats the same packed Python layout under the shared targets without shifting cost into excluded archive preparation or exceeding memory budgets. A RAM-only win and a forced-disk win are separate conclusions.

Deliver a documented internal archive schema, optional storage/loader backend switches, lifecycle and integrity tests, separate weight/feature results, RAM/disk charts and final combined confirmation for any accepted pieces. Retain the old backend for compatibility until both full-clip modes pass.
