# Native optimization experiment proposals

These are the review proposals for four separate experiments. Run them one at a time, checking correctness and results before proceeding. Implementation updates are recorded separately below.

Implementation updates: [updated baseline](updated-baseline-results.md),
[gathering screening](01-cpu-batch-gathering-results.md) and
[shared configuration and dependency injection](../optimization-configuration.md).
The [shard packing/writing result](02-shard-packing-and-writing-results.md) records
the second experiment without changing the original proposal.
The [ray construction result](03-cuda-ray-construction-results.md) includes the
initial failed CUDA gate; the [bug audit](cuda-ray-audit.md) records the fixes and
passing corrected comparison. The new backend remains disabled by default.
The [corrected CUDA profile](cuda-ray-profile.md) tests full-timeline rotation and
separates the fused kernel's time from dispatch, gathering, transfers and commits.
The [DA3 packing/archive result](04-da3-archive-and-block-loading-results.md)
records the repeated RAM comparison, disk pilot and composition/reuse checks.
The proposals below remain the original review plans; result documents record
actual implementation and measurements separately.

| Order | Proposal | Question | Main risk |
|---|---|---|---|
| 1 | [CPU batch gathering](01-cpu-batch-gathering.md) | Can we gather mapped coordinates and visibility directly into bounded destination buffers, avoiding temporary copies? | Hidden conversion, buffer lifetime and mapped-page access costs |
| 2 | [Trajectory shard packing and writing](02-shard-packing-and-writing.md) | Can one bounded pass pack, validate, serialize and hash output more efficiently than the current writer? | Changing precision, integrity checks or commit behavior |
| 3 | [CUDA ray construction](03-cuda-ray-construction.md) | Can a custom operator reduce kernel launches and intermediate GPU tensors? | Numerical differences in sampling and ray arithmetic |
| 4 | [DA3 archive packing and block loading](04-da3-archive-and-block-loading.md) | Can packed storage and fewer weight transfers reduce RAM/disk offload overhead? | Aliasing, mapped lifetimes, transient memory and format complexity |

Each proposal specifies a current control, an improved Python variant and a native variant. Comparing native code only with the current implementation would conflate language changes with changes to layout or allocation. If the improved Python version delivers the same benefit, prefer it.

Read the [shared measurement and acceptance plan](measurement-plan.md) first. It defines the reference, workloads, memory and disk metrics, correctness gates, build records and deliverables. Individual proposals add component-specific checks.

The current optimized offload implementation is the new control. The pre-optimization offload implementation and original dense implementation remain historical context; their saved runtimes are not interchangeable with fresh controls. The [existing experiment log](../optimization-experiment-log.md) records a five-pair full-pipeline median change from 58.903 to 57.109 seconds under Python 3.13, with exact output parity. That is evidence for the three retained Python changes, not a forecast for C++.

The older diagnostic profile measured roughly 0.82 seconds inside rays and 3.15 seconds inside shard commits. Those are overlapping, instrumented spans from before write hashing was improved. Refresh the profile before estimating any opportunity. Gathering is part of ray construction, so savings from proposals 1 and 3 must not be added. Measure any eventual combination directly.

Rewriting the complete models, SHA256 alone or `fsync` alone is outside these experiments. The models already dispatch heavy work to PyTorch/CUDA; hashing already uses a native implementation. The question is whether we can eliminate redundant passes, allocations and calls around that work. [Python 3.13 hashing documentation](https://docs.python.org/3.13/library/hashlib.html)

Viewer-only gathering/export could be a later experiment if a fresh exporter profile identifies a bottleneck. It must use an export benchmark and must not be reported as faster inference.

## Reproduce from the final stack checkout

Provision the pinned weights and demo videos using the setup/download instructions.
The measured working snapshots remain immutable locally. In a fresh checkout,
reconstruct the supported source variants from published Git history:

```bash
uv run python scripts/prepare_optimization_sources.py \
  --revisions configs/experiments/native-source-revisions.json
```

This preserves the algorithms under comparison, not every byte of the original
uncommitted snapshots. `HEAD` denotes the final stack checkout, including the DA3
experiment. Existing destinations cause an error. New runs record their own hashes
and environment; they are not relabeled as the earlier measurements. The original
fully fused ray source is intentionally retained for reproducing its failed gate.

Compile the selected source's native files explicitly before timing, for example:

```bash
uv run python scripts/build_native.py --source outputs/experiments/native-composable-source \
  --out outputs/native/current --cuda --cxx g++-13
```

Create a fresh prepared window with the demo or the updated-baseline-pipeline task.
Pass that window to `optimization_matrix.py --prepared-window ...` for correction
experiments. The recorded campaign spec refers to the original local fixture;
replace its preparation/output paths for a fresh campaign. Configuration flags
must match the selected source's schema; use the current source for new matrices.
Use the matching older sources/configurations only to reproduce historical arms.

The [configuration guide](../optimization-configuration.md) covers one-at-a-time
variants, Cartesian combinations, provider-only tasks and multi-clip reuse. Raw
technical records generate the standalone charts; analysis stays in these Markdown
documents. Functional validation groups are omitted from performance charts, and
exploratory groups are labeled separately through the role configuration.
