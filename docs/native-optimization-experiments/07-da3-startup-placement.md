# CPU-first DA3 startup placement

This experiment follows bounded background shard commits. It changes initial
memory placement, independently of weight packing, feature archives, observation
caching, prefetch and output commits. The reference is stack revision
`46ae687a6c6e4d12a9c1bf655c06875269178984` (PR #17).

## Hypothesis and scope

The current provider loads the checkpoint on CPU, moves the complete model to
CUDA, then installs the pager. The pager immediately copies every transformer
block back to CPU. Those blocks are uploaded again when their forward hooks run.
The initial round trip costs time and temporarily allocates the complete model
on CUDA even though most weights are meant to remain off-device.

The refreshed reference completed a full 84-frame DA3 run and full pipeline before
implementation. Its separate startup profile attributes approximately 4 seconds
to module transfers and 10.5 seconds to random initialization. These inclusive
diagnostic spans overlap; they are not headline timings. Avoiding random
initialization is a separate future experiment, with additional checkpoint and
buffer-materialization risks.

The candidate loads the same complete checkpoint through the same upstream API.
It establishes cold block storage and hooks while the model is on CPU, then moves
only unpaged parameters and buffers to CUDA. It keeps the same parameter objects,
head slices, attention, dtype boundaries, reference selection and full temporal
context. There is no new native operator or change to numerical arithmetic.

## Configuration, injection and ownership

`[da3] startup = "cpu-first"` selects the candidate. `"current"` remains the
default. The `cpu-first-da3` profile changes only this field relative to `ram`;
`cpu-first-combined` inherits observation caching and threaded commits from
`background-cached`. Legacy experiment settings expose `da3_startup` as the
adapter to the grouped configuration. Unsupported non-paged modes fail explicitly.

The existing fingerprinted `InferenceComponents.pager_factory` receives the
startup policy; provider injection and session ownership remain available.
Cold parameters retain their CPU/mapped owners until an individual block runs.
Unpaged placement excludes both cold parameter identities and owned buffers.
Installation failure removes hooks and temporary archives through pager cleanup.
Parking and reactivation retain the existing per-clip lifecycle.

Size counters report cold parameter/buffer bytes and the initial cold tensor
round trip implied by the placement path. They are allocation/transfer accounting,
not a PCIe hardware measurement. Actual latency and CUDA allocator peaks are
recorded separately.

## Measurement and acceptance plan

Keep weights locally provisioned, HF offline and OS page cache retained. Freeze
the candidate source, then use that one source for both configurations. Run one
full-clip screening pair and require exact DA3 artifacts before timing five
fresh-process pairs, alternating order. Compare provider construction, forward,
cleanup and complete DA3 process time separately.

Measure five fresh full-pipeline pairs at resolution 504, all 84 frames,
1,024-point groups, batch 8 and seed 0. Check every output, including padded
repeated pixels, raw lift, masks, prepared arrays, cameras and image pixels. A
comparison against the preceding source verifies unchanged default behavior.
Single full-length running/forest pairs establish capacity and output parity,
not latency distributions. Exercise disk placement, packed storage, reuse and
composition separately from headline timing.

Record sampled process-tree RSS/PSS, OS RSS high water, whole-device GPU memory,
CUDA allocation/reservation, physical disk I/O and peak/final output allocation.
Keep expensive profiles and parity scans outside unprofiled timing. An occupied
compute GPU prevents a trial; detected interference invalidates it. Retain failed
and exploratory records. Report medians, ranges and paired differences; do not
combine historical timings into the fresh comparison.

Tests cover exact tiny-backbone state and repeated outputs, auxiliary features,
reference selection, CPU/CUDA residency, RAM/disk and packed layouts, configuration
round trips, injected factories and partial-installation cleanup. Keep the option
experimental unless measured gains and correctness support it. Results and the
keep/discard decision will be appended after the campaign; technical report
scripts contain no experiment conclusions.
