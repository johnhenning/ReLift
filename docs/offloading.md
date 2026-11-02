# Bounded inference and offloading

The demo accepts `--offload none|streamed|ram|disk|auto`.

| Mode | Geometry placement | Dense correction and output |
|---|---|---|
| `none` | Original CUDA placement | Original dense CUDA arrays and `tracks.npz` |
| `streamed` | Original CUDA placement | Full-timeline point batches and mapped output shards |
| `ram` | Inactive DA3 blocks/features in CPU RAM | Streamed correction and mapped shards |
| `disk` | Inactive DA3 blocks/features in mapped SSD archives | Streamed correction and mapped shards |
| `auto` | Conservative memory estimates select a placement | Streamed correction and mapped shards |

```bash
uv run python scripts/demo_video.py VIDEO --num_frames 0 --process-res 504 \
  --offload ram --no_video --out outputs/offloaded-demo
```

The pipeline keeps every source frame and the original processing grid. It retains full DA3 temporal
attention and each corrected point's complete timeline. CUDA nearest-neighbor ray sampling, dtypes,
reference-frame normalization and seeded groups remain the computation path.

Mapped output is stored under `tracks.store/manifest.json` with track-major NPY shards. The manifest
records original pixel IDs, grouping, source/checkpoint fingerprints and per-file checksums. Readers
can sample original pixel IDs without loading the complete trajectory tensor. `--export-npz` adds
a compatibility archive. Plotly and the catalog batch accept mapped stores and legacy NPZ results.

Defaults are `--ram-budget-gb 18 --vram-budget-gb 20 --scratch-budget-gb 64`. These are retained-tensor,
allocation and staging targets, not OS limits on process RSS or reserved GPU memory. Active attention
still needs to fit on CUDA; disk mappings consume resident pages and OS cache. The batch does not
silently reduce resolution. Legacy spatial retries require explicit `--offload none --allow-resolution-retry`.

Shards use temporary files, fsync and atomic rename. Resume verifies committed data and provenance.
Padded duplicate points use deterministic first-occurrence ownership; the original CUDA scatter had
undefined write order for those points. Numerical comparisons must report them explicitly.
