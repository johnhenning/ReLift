# DA3 packing and archive results

The fourth experiment adds independent `da3_weights` and `da3_features` backends.
Weights share a contiguous CPU/mapped region per block and dtype, with views
restored after each forward. Feature archives append logical tensor bytes to one
private-mapped file with a versioned index. Python and C++ use the same layouts.
The current layouts remain the defaults.

## Repeated RAM weight comparison

Five balanced fresh-process runs per arm retain the entire 84-frame swing-tire
clip and released checkpoint. Decode, construction, packing, forward, serialization,
checksums and cleanup are timed. All 15 output artifact sets are byte-identical,
including depth, processed images, poses and intrinsics.

| Method | Process median seconds [min,max] | Paged-block upload calls | Sampled device GiB | OS RSS high water GiB |
|---|---:|---:|---:|---:|
| Current | 38.288 [38.053,107.031] | 1,004 | 12.403 | 13.601 |
| Packed Python | 37.594 [37.183,38.882] | 64 | 12.577 | 13.600 |
| Packed C++ | 36.513 [35.447,37.044] | 64 | 12.577 | 13.600 |

Native packing's median is **4.6% lower** than current and **2.9% lower** than
packed Python. Every native repeat is faster than its matching control. All arms
transfer the same **5,743,324,160 weight bytes** through the block hooks. The call
counters exclude image, feature and other CUDA transfers.

The first control read approximately 6.35 GiB from storage while loading the
checkpoint, making construction 84.78 seconds and the process 107.03 seconds.
That outlier remains in the records and range. Subsequent controls take about
38 seconds. OS caches were retained; this is not a uniform cold-cache study.

Sampled RSS varies, but OS RSS high water is approximately 13.6 GiB for every arm;
the samples do not establish a RAM saving. Packed CUDA allocation increases sampled
device peak by about 0.174 GiB. Writes and peak output allocation are about 0.077 GiB
in all RAM arms. This is a DA3-only result below the proposed 10% component target,
without five-pair full-pipeline confirmation, so no backend is promoted by default.

## Disk feature-archive pilot

One complete run per arm keeps current weight placement and varies feature storage.
Each output artifact set is byte-identical to the control.

| Method | Process seconds | Scratch files before cleanup | Sampled peak output GiB | Process write counters GiB |
|---|---:|---:|---:|---:|
| Current per-tensor archives | 147.700 | 1,020 | 12.817 | 12.817 |
| Python feature archive | 131.730 | 1,006 | 12.817 | 12.817 |
| C++ feature archive | 132.172 | 1,006 | 12.817 | 12.817 |

The Python/native times are similar: any promising effect here concerns the layout,
not demonstrated C++ acceleration. These single exploratory points are not latency
distributions. Cache history, checkpoint reads and writeback remain relevant; repeat
a balanced campaign before concluding a performance gain. GPU peaks stay near
12.403 GiB and OS RSS high water near 13.6 GiB. Peak scratch differs from final
retained output because the provider removes its temporary archives.

## Composition and lifecycle checks

A full-pipeline check combines native gathering, packing/writing, corrected CUDA
preparation, packed DA3 weights, depth caching and prefetch. Ordinary and padded
trajectories and raw lift are bitwise equal, with exact masks, prepared arrays,
images and cameras. A separate two-clip check over swing-tire and swing verifies
byte-identical outputs with session reuse disabled/enabled. The reused provider
parks on CPU between stages and restores unpaged CUDA state on activation. These
single pairs are functional checks, excluded from performance charts.

Tests use actual tiny pinned DA3 backbones with RAM/disk, weights/features separately
and together, auxiliary features, reference selection, repeated calls and buffers.
They also cover private mapping lifetime, invalid/index-corrupt/truncated archives,
checksum rejection, budget failure, append rollback and partial-installation cleanup.
Unsupported aliased block storage fails explicitly. Injected shard packers cannot
commit arrays with changed shape/dtype; packing and writing can be configured separately.

The archive is temporary internal storage, not a resumable checkpoint or crash
durability guarantee. Explicit integrity verification is available without adding
a full reread to every tensor access. Model precision, reference selection,
temporal context and decoder slices remain unchanged.

## Interfaces and reproduction

`InferenceComponents` accepts cold-storage, feature-archive, block-packing and pager
factories, in addition to providers, model loading, gathering, rays and shard handling.
Injected implementations require fingerprints. The [configuration guide](../optimization-configuration.md)
covers JSON switches, one-at-a-time variants and Cartesian matrices. Feature archives
require disk placement; unsupported selections fail instead of silently doing nothing.

The spec contains `da3-packed-weights-ram`, `da3-feature-archive-disk-pilot`,
`combined-native-validation` and `da3-session-reuse-validation`. Raw records, build
and source hashes, telemetry and quality checks remain under
`outputs/experiments/native-optimizations`; compact evidence accompanies the PR.
Use fresh output roots for new trials. Running/forest capacity checks, repeated disk
studies, cold-cache isolation and repeated full-pipeline confirmation of selected
combinations remain follow-up experiments. None of the new backends is enabled by default.
