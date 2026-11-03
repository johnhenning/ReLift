# Offload benchmark results

Open the [standalone comparison charts](../outputs/benchmark-charts/index.html) for runtime,
sampled host/GPU memory, physical disk I/O and saved dense-output allocation across the measured
methods. The viewer links PNG/SVG figures and CSV/JSON data. Its charts are generated from the
selected trial records; the separately reviewed [optimization guide](performance-opportunities.md)
describes the proposed experiments.

The [technical report](../outputs/benchmarks/offload-report.md) is generated from selected trial,
parity and environment records, combined with the separately reviewed
[analysis](offload-benchmark-analysis.md).

Machine-specific measurements and generated reports live under `outputs/benchmarks/`.
The renderer does not contain campaign-specific claims, dates, hardware descriptions or conclusions.
It derives counts, medians, settings-matched ratios, memory and I/O metrics from the supplied records,
reports missing metadata explicitly, and links to the underlying evidence.

The [analysis document](offload-benchmark-analysis.md) contains the interpretation, limitations,
remaining experiments and exact commands used for this campaign. It must be reviewed when a different
set of measurements is supplied. The [approved design](memory-offload-design.md) records the original
scope and the broader experimental plan.

The [report renderer](../scripts/report_offload_results.py) accepts explicit benchmark roots,
parity files, an optional environment record and optional analysis documents. Its tests exercise
unrelated synthetic campaigns, missing fields, OOMs, changed settings and conflicting versions.
