# Composable inference optimizations

`InferenceConfig` groups public configuration by component. `OptimizationConfig`
is the validated flat execution contract retained for compatibility. Unspecified fields retain the currently accepted defaults.
Unknown fields, invalid backend names and incorrectly typed values fail before use.
The resolved settings and injected component fingerprints are recorded in metrics
and trajectory-store provenance; changed settings invalidate resume.

| Setting | Default | Experiment controlled |
|---|---|---|
| `gather` | `current` | `current`, reusable-buffer `python`, native `cpp` gathering |
| `shards` | `current` | `current`, `python`, `cpp` shard packing and optional native writing |
| `shard_writer` | `null` | Inherit `shards`; explicit `current`/`cpp` isolates packing from payload writing |
| `rays` | `current` | `current`, blocked `python`, `cuda` per-block rotation, `cuda_timeline` full-timeline rotation |
| `da3_weights` | `current` | `current`, packed `python`, packed `cpp` |
| `da3_features` | `current` | `current`, archived `python`, archived `cpp`; disk mode only |
| `da3_startup` | `current` | `cpu-first` establishes cold owners before CUDA placement; RAM/disk only |
| `tracker_init_weights` | `false` | Skip discarded pretrained initialization; `true` restores old initialization |
| `tracker_copy_outputs` | `false` | Return float32 views; `true` restores copies |
| `hash_writes` | `true` | Hash NPY bytes during writes; `false` uses the subsequent initial read |
| `depth_cache_mb` | `0` | Bounded persistent CUDA depth cache; zero disables it |
| `observation_cache_mb` | `0` | Original-dtype tracker coordinates/visibility cached on-device; replaces CPU gathering when it fits |
| `prefetch` | `false` | One-batch ray lookahead with explicit ownership and cleanup |
| `corrector_batch` | `8` | Number of independent groups per model call |
| `padding_batch` | `null` | Fix padded-group membership independently of execution batch size |
| `compile_corrector` | `false` | Explicit experimental `torch.compile(dynamic=False)` |
| `reuse_da3` | `false` | Reuse within an explicitly owned multi-clip session |

The earlier depth-cache/prefetch trials did not improve runtime. Compilation and
batches 1/2 failed the strict numerical gate. They remain selectable experiments,
not accepted defaults. Batch and precision effects must pass quality checks for
each combination. New backends are optional and cannot silently fall back when a
build is absent. Streaming-only settings are rejected in legacy dense mode;
packed DA3 weights require RAM/disk placement and feature archives require disk.

The original fully fused CUDA ray prototype failed the full-output gate. The
corrected preparation path retains PyTorch rotation and passes the recorded full
comparison, but remains experimental because it is slower. See the ray bug audit.
The [profiling follow-up](native-optimization-experiments/cuda-ray-profile.md)
adds `cuda_timeline` as a separate opt-in experiment. It keeps bounded depth
uploads, performs the original PyTorch rotation once over the timeline, and uses
a validated eager native launch to avoid registered-operator first-dispatch cost.
Select it with `--opt rays='"cuda_timeline"'` or
`configs/optimizations/rays-cuda-timeline.json`; a `--cuda` native build is required.

The [observation-cache experiment](native-optimization-experiments/05-observation-cache.md)
uses existing PyTorch gathering on immutable resident tensors. Select
`--opt observation_cache_mb=128` independently or combine it with depth caching,
prefetch and a ray backend. The cache records whether it is active or declined
for budget/headroom, along with resident and uploaded bytes. Its budget covers
retained tracker tensors; depth and active batches have separate allocations.
Select `gather="current"` with a positive cache budget: the cache replaces the
Python/C++ CPU gathering strategies, and conflicting choices are rejected.
`InferenceComponents.observation_cache_factory` accepts a fingerprinted factory
`(tracks, visibility, device, budget_mb)` returning an owner with `active`,
`gather(pixel_ids, frame=None)`, `metrics` and `close()`.

## Component profiles

Use one named profile or TOML file instead of individual optimization flags:

```bash
uv run python scripts/demo_video.py --profile observation-cache --print-config
uv run python scripts/demo_video.py cache/demo_inputs/swing-tire.mp4 \
  --num_frames 0 --profile observation-cache --no_video --out outputs/profile-example
```

`--print-config` resolves settings and source-file SHA256 fingerprints without
requiring a video or GPU. `--profile` selects a file in `configs/inference/`;
`--inference-config FILE.toml` selects a custom file. Choose one. Legacy JSON
`--optimizations` and `--opt` arguments remain supported for archived commands,
but cannot be combined with the grouped interface.

A custom file can inherit a named profile or a TOML path relative to itself:

```toml
extends = "ram"

[observations]
cache_mb = 128
prefetch = false

[rays]
backend = "current"

[storage]
packing = "current"
hash_writes = true
```

Tables merge by field, so changing rays keeps inherited observation settings.
Unknown tables/fields, incorrect types, inheritance cycles, invalid backends and
incompatible settings fail before inference. A profile records every inherited
file's content hash. Relative runtime native-build and scratch paths use the
working directory, while inheritance paths use the declaring file's directory.
A changed source file invalidates store resume even if its effective values match.

| Component | Fields |
|---|---|
| `memory` | `offload`, `ram_gb`, `vram_gb`, `scratch_gb`, `scratch_dir` |
| `observations` | `gather`, `cache_mb`, `depth_cache_mb`, `prefetch` |
| `rays` | `backend` |
| `correction` | `batch`, `padding_batch`, `compile` |
| `storage` | `packing`, `writer`, `hash_writes`, `commits` |
| `da3` | `weights`, `features`, `reuse`, `startup` |
| `tracker` | `init_weights`, `copy_outputs` |
| `runtime` | `native_build` |

The flat field table above describes the corresponding engine switches and their
constraints. Every earlier optimization is mapped to a component. `schema_version`
is currently `1`. TOML omits optional fields to keep defaults; writer omission
inherits the packing backend, and padding-batch omission retains existing grouping.

Built-in profiles are `ram`, `disk`, `observation-cache`, `cached-rays`,
`cuda-cached-rays` and `reuse-da3`. Experimental combinations remain opt-in;
profiles do not imply that all combinations improve performance or pass every
numerical gate. Native profiles require a separately built binary, configured by
`[runtime] native_build = "outputs/native/BUILD"` or `--native-build PATH`.
Explicit CLI memory budgets, offload, scratch, native build and corrector batch
may override a profile; the effective values are recorded in metrics and stores.
Model/checkpoint, input and output arguments remain separate from engine selection.
`--config` retains the existing evaluation YAML meaning; component TOML uses
`--inference-config`.

## Reproducible experiment matrices

The experiment runner accepts `options.inference_profile`,
`options.inference_config` or an inline grouped `options.configuration`. It records
resolved component values and file fingerprints in each trial. Changed profile
contents are rejected by `--resume`. The existing diagnostic `options.profile`
boolean retains its profiling meaning. Legacy `options.optimizations` remains
available for historical experiment specs.

Use a JSON grid with dotted component fields:

```json
{"observations.cache_mb": [0, 128], "storage.hash_writes": [true, false]}
```

```bash
uv run python scripts/optimization_matrix.py \
  --source FROZEN_SOURCE --profile ram \
  --grid configs/inference/example-grid.json --mode one-at-a-time \
  --out outputs/profile-matrix.json --output-root outputs/experiments/profile-matrix
uv run python scripts/run_optimization_experiments.py \
  --spec outputs/profile-matrix.json --experiment matrix
```

`one-at-a-time` includes the control and each independent change; `cartesian`
generates combinations. Inspect the generated spec before running it. Freeze the
source before generating a matrix, and keep fresh output roots. `correction`
reuses prepared geometry and cannot measure provider switches; `pipeline` runs a
full clip; `da3-configured` isolates geometry and supports multiple `--clips` to
measure reuse. Profiles should use RAM/disk offload for provider tasks. Set
`correction.padding_batch = 8` when varying execution batch size while retaining
reference padding groups. CLI generation still supports legacy flat JSON grids.

## Dependency injection

Python callers can inject `InferenceComponents` into `stream_correct` or an
`InferenceSession`. Available overrides are gather construction, ray construction,
shard packing/writing, pager/cold storage/feature archive construction, block
packing, depth/tracker provider construction, corrector
loading and native module loading. Overrides take precedence over the corresponding
configured backend. Each supplied override requires a stable fingerprint identifying
its implementation and relevant settings. There is no module-wide registry mutation.

```python
from relift.configuration import load_configuration
from relift.optimization import InferenceComponents, InferenceSession

components = InferenceComponents(
    gather_factory=my_gather_factory,
    tracker_factory=my_tracker_factory,
    fingerprints=(("gather_factory", "gather-v1"), ("tracker_factory", "tracker-v2")),
)
selection = load_configuration(profile="ram")
with InferenceSession(selection, components) as session:
    # Pass session=session to scripts.demo_video.main(argv, session=session).
    # Direct correction accepts optimizations=selection.config.optimizations(), components=components.
    pass
```

A gather factory returns a callable `(tracks, visibility, pixel_ids)` producing
float32 `(T,N,2)` coordinates and `(T,N)` visibility in the original order. Provider
factories use the normal provider constructor contract. Pager factories accept the
same arguments as `DA3Pager`. A ray builder takes `(coords, depth_block, K, rotation)`
and returns `(direction, distance)` in original frame/point order. A shard packer
takes `(field_name, array)` and returns `(packed_array, finite_bool)` with unchanged
shape/dtype. The NPY writer takes `(stream, array, hash_write)` and returns a digest
or None without closing the stream. Storage, archive and block-packer overrides
use the contracts in `da3_offload.py` and `da3_packing.py`, with the cold-storage
owner controlling cleanup. Prefer injected fakes for lifecycle tests and real outputs
for numerical comparisons.

## Explicit multi-clip lifetime

```bash
uv run python scripts/run_inference_session.py \
  cache/demo_inputs/swing-tire.mp4 cache/demo_inputs/swing.mp4 \
  --out outputs/session-example --profile reuse-da3
```

The session owns the reusable DA3 provider, parks it on CPU between stages/clips and
closes it on exit or failure. Weights, device and provider settings form its reuse
key. A changed key closes the old provider. Scratch belongs to the session, not an
individual clip. The helper resets seeds per clip. AllTracker and corrector are
constructed independently for each clip; this setting does not claim their reuse.

The historical worker snapshots and monkey-patching prefetch driver remain intact
for reproducing old measurements. The configurable production path uses an explicit
`PrefetchedRays` object with one future and stream, and restores no global methods.
Timed trials record actual backends, source/binary hashes and the environment.

## Background shard commits

`storage.commits` selects `sync` (default) or `threaded`. The latter owns one
pending CPU shard and overlaps its commit with the next GPU batch. It reserves an
additional staging workspace under the RAM target and drains before final store
verification. Select `background-commits`, or compose the field into an existing
TOML profile. `background-cached`, `background-prefetch` and `background-native`
are explicit composition-validation profiles; they are not automatic defaults.

The legacy adapter exposes the same choice as `commit_backend`.
`InferenceComponents.shard_committer_factory` receives the store sink and returns
an owner implementing `submit(start, xyz=..., lift=..., observed=...)`, `drain()`,
`metrics`, `__enter__` and `__exit__`. Submission transfers CPU buffer ownership
until the next drain; at most one submitted shard may remain pending. The factory
requires its own stable fingerprint. See the [experiment design and results](native-optimization-experiments/06-background-commits.md).

## DA3 startup placement

Select `cpu-first-da3`, or add `[da3] startup = "cpu-first"` to an inherited
RAM/disk profile. This keeps transformer blocks on CPU while installing their
pager and moves only unpaged state to CUDA. It avoids the initial CUDA upload
and CPU return of weights that will be paged during forward. The checkpoint,
inference arithmetic and full clip remain the same.

`cpu-first-combined` adds the option to observation caching and background commits;
`cpu-first-disk` and `cpu-first-reuse` exercise disk placement and session reuse.
The field is independent of `weights`/`features` packing. Existing fingerprinted
pager factories receive `startup` along with the other pager arguments. Defaults
retain the prior placement path. See the [startup experiment](native-optimization-experiments/07-da3-startup-placement.md)
for measurements and limits.
