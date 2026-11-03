# Native optimization experiment proposals

These are the review proposals for four separate experiments. Run them one at a time, checking correctness and results before proceeding. Implementation updates are recorded separately below.

Implementation updates: [updated baseline](updated-baseline-results.md),
[gathering screening](01-cpu-batch-gathering-results.md) and
[shared configuration and dependency injection](../optimization-configuration.md).
The [shard packing/writing result](02-shard-packing-and-writing-results.md) records
the second experiment without changing the original proposal.
The [ray construction result](03-cuda-ray-construction-results.md) includes a failed
CUDA numerical gate; the new backend remains disabled.
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
