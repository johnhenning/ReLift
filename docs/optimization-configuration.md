# Composable inference optimizations

`OptimizationConfig` is the shared validated configuration for demos, sessions and
new experiment variants. Unspecified fields retain the currently accepted defaults.
Unknown fields, invalid backend names and incorrectly typed values fail before use.
The resolved settings and injected component fingerprints are recorded in metrics
and trajectory-store provenance; changed settings invalidate resume.

| Setting | Default | Experiment controlled |
|---|---|---|
| `gather` | `current` | `current`, reusable-buffer `python`, native `cpp` gathering |
| `shards` | `current` | `current`, `python`, `cpp` shard packing; later stack layer |
| `rays` | `current` | `current`, fused `cuda`; later stack layer |
| `da3_weights` | `current` | `current`, packed `python`, packed `cpp`; later stack layer |
| `da3_features` | `current` | `current`, archived `python`, archived `cpp`; later stack layer |
| `tracker_init_weights` | `false` | Skip discarded pretrained initialization; `true` restores old initialization |
| `tracker_copy_outputs` | `false` | Return float32 views; `true` restores copies |
| `hash_writes` | `true` | Hash NPY bytes during writes; `false` uses the subsequent initial read |
| `depth_cache_mb` | `0` | Bounded persistent CUDA depth cache; zero disables it |
| `prefetch` | `false` | One-batch ray lookahead with explicit ownership and cleanup |
| `corrector_batch` | `8` | Number of independent groups per model call |
| `padding_batch` | `null` | Fix padded-group membership independently of execution batch size |
| `compile_corrector` | `false` | Explicit experimental `torch.compile(dynamic=False)` |
| `reuse_da3` | `false` | Reuse within an explicitly owned multi-clip session |

The earlier depth-cache/prefetch trials did not improve runtime. Compilation and
batches 1/2 failed the strict numerical gate. They remain selectable experiments,
not accepted defaults. Batch and precision effects must pass quality checks for
each combination. New backends are optional and cannot silently fall back when a
build is absent. A backend from a later stack layer fails explicitly until installed.

## CLI configuration

Build a native helper outside timed inference:

```bash
uv run python scripts/build_native.py --out outputs/native/gather
uv run python scripts/demo_video.py cache/demo_inputs/swing-tire.mp4 \
  --num_frames 0 --offload ram --no_video --out outputs/gather-example \
  --optimizations configs/optimizations/gather-cpp.json \
  --native-build outputs/native/gather
```

Compose independent settings with a JSON file or repeated `--opt key=JSON_VALUE`:

```bash
uv run python scripts/demo_video.py cache/demo_inputs/swing-tire.mp4 \
  --num_frames 0 --offload ram --no_video --out outputs/combination-example \
  --opt gather='"python"' --opt hash_writes=true --opt depth_cache_mb=256
```

These examples describe test configurations, not measured improvements. Keep fresh
output directories and compare the exact settings against a matching control. The
older `--corrector-batch` flag takes precedence over the config's batch value and
the resolved value is persisted. Model/input settings and memory budgets retain
their existing arguments; optimization selection does not change them implicitly.

The experiment runner accepts the same dictionary under each variant's
`options.optimizations`, plus `options.native_build` for a native binary. Its
pipeline task accepts `options.offload` so RAM/disk comparisons can remain separate.
Use these dictionaries to enumerate a combination matrix; this does not require
a separate code patch for every combination. Three-arm runs rotate execution order.

Generate a full Cartesian matrix without starting inference:

```bash
uv run python scripts/optimization_matrix.py \
  --source outputs/experiments/native-composable-gather-source \
  --grid configs/optimizations/example-grid.json \
  --out outputs/combination-matrix.json \
  --output-root outputs/experiments/combination-matrix
uv run python scripts/run_optimization_experiments.py \
  --spec outputs/combination-matrix.json --experiment matrix
```

The grid's keys are optimization settings and its lists are values to combine.
The example expands to eight variants and five repeats each; inspect the generated
spec before running it. Supply `--native-build` when the selected grid needs it.

## Dependency injection

Python callers can inject `InferenceComponents` into `stream_correct` or an
`InferenceSession`. Available overrides are gather construction, ray construction,
shard packing, pager construction, depth/tracker provider construction, corrector
loading and native module loading. Overrides take precedence over the corresponding
configured backend. Each supplied override requires a stable fingerprint identifying
its implementation and relevant settings. There is no module-wide registry mutation.

```python
from relift.optimization import InferenceComponents, InferenceSession, OptimizationConfig

components = InferenceComponents(
    gather_factory=my_gather_factory,
    tracker_factory=my_tracker_factory,
    fingerprints=(("gather_factory", "gather-v1"), ("tracker_factory", "tracker-v2")),
)
settings = OptimizationConfig(hash_writes=True, depth_cache_mb=0)
with InferenceSession(settings, components) as session:
    # Pass session=session to scripts.demo_video.main(argv, session=session).
    # Direct correction accepts optimizations=settings, components=components.
    pass
```

A gather factory returns a callable `(tracks, visibility, pixel_ids)` producing
float32 `(T,N,2)` coordinates and `(T,N)` visibility in the original order. Provider
factories use the normal provider constructor contract. Pager factories accept the
same arguments as `DA3Pager`. Later layers document ray and packer contracts when
their implementation is introduced. Prefer injected fakes for lifecycle tests and
real outputs for numerical comparisons.

## Explicit multi-clip lifetime

```bash
uv run python scripts/run_inference_session.py \
  cache/demo_inputs/swing-tire.mp4 cache/demo_inputs/swing.mp4 \
  --out outputs/session-example --opt reuse_da3=true
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
