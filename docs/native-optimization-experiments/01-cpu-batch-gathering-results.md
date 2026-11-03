# CPU gathering screening results

The native helper and NumPy reusable-buffer variant are implemented as optional
backends. The current gather remains the default. Both are selectable through
`OptimizationConfig` and the demo, session and matrix interfaces described in
[optimization configuration](../optimization-configuration.md).

Two five-repeat, three-arm correction screenings were recorded. In the initial
helper comparison, process medians were 18.415 seconds current, 18.279 NumPy and
18.446 C++. In the configurable comparison, medians were 16.781, 17.954 and
18.373 seconds respectively. The ranges overlap and paired differences change
direction. Neither establishes a repeatable improvement. GPU peaks remain about
3.882 GiB and physical writes about 0.388 GiB for all arms.

The configurable comparison uses the same source for all three arms, with resolved
settings recorded per trial. It includes all 84 frames, the original groups and
padding, normal durable commits and final verification. The first comparison uses
the frozen pre-native control. They are separate groups, not pooled repeats.

CPU unit checks briefly ran during the configurable screening, so those timings
are exploratory and are not an uncontended latency claim. More generally, the
control's correction runtime shifted above the initial 10.901-second baseline.
The cause was not isolated. Comparing a candidate directly with that earlier number
would be misleading; matching controls and full trial ranges remain in the records.

All 14 comparisons in each gathering screening pass the full-output gate. The
configured default also passes a full-pipeline comparison against the frozen
pre-native control. That one-pair check overlapped the parity scanner and is a
functional check, not a latency result. Quality records remain under
`outputs/experiments/native-optimizations`. CPU tests cover supported dtypes, negative
strides, duplicate/empty/tail IDs, read-only mapped input and explicit build errors.
GPU composition tests additionally exercise depth caching, write hashing and
prefetch with each gather backend; all six CUDA composition checks pass.

No native speedup is claimed, and no native backend is enabled by default. The
implementation remains available to test combinations, different storage and
larger workloads. A clean repeated campaign and pipeline confirmation are required
before any future promotion.

The updated baseline, compiler manifest, trial records, summaries and quality
results are preserved. Public compact evidence is copied to
`docs/benchmarks/native-optimizations` in the review branch. Full trajectory arrays
and traces remain under ignored `outputs/`. Configuration, measured data and this
analysis are separate; the summary/chart scripts contain no conclusions.
