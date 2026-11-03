# Experiment 2: trajectory shard packing and writing

Status: proposed. Measure after experiment 1, against an explicitly frozen control. Follow the [shared measurement plan](measurement-plan.md).

## Hypothesis and code scope

The correction loop rearranges tensors, transfers results to CPU, casts raw lift to float16 and calls `TrajectoryStore.write_shard`. The writer checks values, writes three NPY members, hashes bytes, fsyncs each file and atomically updates the manifest. A bounded packer/writer may remove redundant allocations and passes over the same payload.

Targets: output preparation in [stream_correct](../../relift/streaming.py), and [TrajectoryStore.write_shard, atomic_npy and _HashingWriter](../../relift/storage.py). Final verification, resume verification and manifest ownership remain in Python initially.

Hashing during writes is already enabled and improved the earlier experiment. The old 3.15-second shard span predates that change. SHA256 itself is already native, and Python releases the GIL for sufficiently large hash updates, so replacing `hashlib` alone is not the hypothesis. [Python 3.13 hashing documentation](https://docs.python.org/3.13/library/hashlib.html)

## Candidate design and comparisons

| Variant | Change |
|---|---|
| Current | Existing strided CPU arrays, validation and hash-on-write NPY output |
| Improved Python | Reuse bounded contiguous packing buffers, avoid redundant casts where safe, and tune payload chunking while retaining the same file format and commit policy |
| Native equivalent | Pack the same source strides into the same buffers/chunks while validating finite values and feeding an incremental writer/hash sink |

Begin with packing/validation only, passing the packed arrays to the existing Python writer. Add a native payload writer as a separate sub-experiment only if serialization remains measurable. This separates a useful packer from unnecessary native filesystem code.

The proposed interface accepts one shard's array buffers and metadata and returns committed member digests only after successful writes. Preserve float32 XYZ, float16 lift and boolean observed values. Start with the existing NumPy float16 conversion; a native conversion must independently prove identical rounding and overflow behavior before replacing it.

Reuse Python-generated NPY headers and existing layout decisions initially, rather than introducing a second file-format implementation. Hash the exact header and payload bytes written. Handle short writes and errors explicitly. Never publish a manifest entry until all three members are committed. Invalid data may leave unreferenced temporary files, but must never become a committed shard. Retain flush/fsync, atomic rename and final re-read verification.

Keep one bounded packing workspace; no full-clip output buffer. Do not add a background write queue, change shard count, drop final scans or alter fsync cadence in this comparison. Those would change the algorithm or durability policy and require separate experiments.

## Measurements and checks

Use replayable CPU output fixtures for all three fields and their actual strides. Separate device-to-host time, packing/conversion, finite checking, serialization/hash, fsync/rename and final verification in diagnostic runs. Measure synchronous complete-shard latency in unprofiled trials; do not sum overlapping spans.

Run on the same local filesystem with fresh files and the same shard boundaries. Record bytes packed, bytes hashed/read, file/write/fsync counts, physical writes and scratch peaks. A tmpfs microbenchmark may isolate packing, but cannot establish an SSD end-to-end improvement.

Require exact decoded values, dtypes, shapes, pixel order and masks, plus byte-identical member files for the initial unchanged-header/layout variants. If a later layout changes NPY headers, report that change and compare logical arrays separately; never compare unrelated container hashes as a numerical check. Manifest provenance can differ between sources.

Exercise strided arrays, tail shards, nonfinite inputs, float16 boundary values, short writes, disk-full/write/fsync errors, interrupted writes and corrupt committed files. Verify that resume recomputes incomplete work and rejects corrupt committed bytes under both backends. An exception must not mark a partially written shard complete.

## Decision and deliverables

Stop if remaining time is dominated by required fsync or final scans and packing is too small to meet the shared targets. Keep a Python packing improvement without a C++ dependency if it captures the gain. Fewer integrity checks do not count as an equivalent speedup.

Deliver separate packer and optional writer comparisons, fault/recovery tests, exact-format compatibility results, five-repeat measurements and complete pipeline confirmation. Document whether the gain comes from fewer copies, fewer payload passes or native dispatch.
