# Updated optimized baseline

The new campaign measured the frozen current implementation before any native changes.
Its source manifest is `outputs/experiments/native-baseline-source/snapshot.json` and the
hardware/checkpoint inventory is `outputs/experiments/native-optimizations/environment.json`.
The campaign uses Python 3.13.16, PyTorch 2.11.0+cu128 and the local RTX 3090 Ti.

Five fresh full-pipeline swing-tire runs completed with a median of **50.811 seconds**
(50.480–51.335). Five prepared-input correction runs completed with a median of
**10.901 seconds** (10.793–11.119). Both retain all 84 frames at 280×504, group 1024,
batch 8, seed 0, RAM offload and hash-on-write shard commits. The four subsequent
outputs within each group pass the full-output comparison against its first control.

Median sampled whole-device memory is 12.401 GiB for full inference and 3.880 GiB
for correction. Sampled tree RSS is 11.184 and 1.537 GiB; physical writes are 0.586
and 0.388 GiB. These are different benchmark scopes. The correction baseline did
not yet report OS RSS high water; later native workers add it. Input fixtures are
copied outside correction timing, with mapping conversion inside that timing.

The separate diagnostic measured 0.884 seconds inclusively in rays and 2.361 seconds
in shard writes. CPU self time includes 0.401 seconds in mapped indexing, 2.154 in
fsync and 2.025 in hashing across stages. Profile spans overlap, include instrumentation
and are not added or substituted for unprofiled times. They suggest that gathering
has a small ceiling and that required storage waits may limit the writer experiment.

The earlier 57.109-second optimized result is historical context. This campaign's
50.811-second median is the fresh control; their difference is not attributed to a
code change. New experiment arms rerun their matching controls in the same campaign.

Reproduce with a fresh source/output directory by adapting the paths in
`configs/experiments/native-optimizations.json`. Run its updated-baseline-pipeline,
updated-baseline-correction and updated-baseline-profile experiments in that order,
then run the parity checker with `--all-trials` and the technical summary. Do not
overwrite these measured directories. Results and interpretation stay separate from
the generic report generators.
