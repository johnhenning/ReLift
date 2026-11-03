<h1 align="center">ReLift</h1>
<h2 align="center">Dense 3D Tracking as Lifted Trajectory Repair</h2>

<p align="center">
  <a href="https://assafsinger94.github.io/">Assaf Singer</a><sup>&dagger;</sup> &middot;
  <a href="https://www.linkedin.com/in/maroun-khriesh-1451b4216/">Maroun Khriesh</a><sup>&dagger;</sup> &middot;
  <a href="https://rotsteinnoam.github.io/">Noam Rotstein</a> &middot;
  <a href="https://ykasten.github.io/">Yoni Kasten</a> &middot;
  <a href="https://orlitany.github.io/">Or Litany</a>
</p>
<p align="center"><sup>&dagger;</sup> Equal contribution</p>

<p align="center">
  <a href="https://relift3d.github.io/">
    <img src="assets/logo_page.svg" alt="Project Page" width="125">
  </a>
  <a href="https://arxiv.org/abs/2610.42136">
    <img src="assets/logo_arxiv.svg" alt="Arxiv" width="125">
  </a>
  <a href="https://arxiv.org/pdf/2610.42136">
    <img src="assets/logo_paper.svg" alt="Paper" width="125">
  </a>
</p>

<div align="center">
  <img src="assets/teaser.gif" width="720" alt="Input video (left) and the 3D tracks (right)" /><br/>
  <b>Input video</b>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<b>3D tracks</b>
</div>

<br>

ReLift turns a dense 2D point tracker and a video 3D geometry model into 3-D trajectories of every pixel: the tracked
pixels are lifted along their camera rays at the predicted depth, and a small transformer, the corrector, repairs
and completes the lifted trajectories. One released checkpoint serves every tracker / geometry combination.

## Table of Contents

- [Setup](#setup)
- [Demo](#demo)
- [Training](#training)
- [Evaluation](#evaluation)
- [Repository layout](#repository-layout)
- [Licence](#licence)
- [Citation](#citation)

## Setup

```
git clone https://github.com/johnhenning/ReLift.git
cd ReLift
bash scripts/setup.sh                  # uv environment .venv + all five front ends as Python dependencies
uv run python scripts/download_weights.py     # all weights into checkpoints/, md5-checked
```

Requires Linux, [uv](https://docs.astral.sh/uv/getting-started/installation/) and a CUDA GPU (24 GB; CoWTracker needs 48 GB). VGGT-Omega's weights are **gated** on
Hugging Face: request access on https://huggingface.co/facebook/VGGT-Omega, then `uv run hf auth login` (or set
`HF_TOKEN`) and rerun the download. VGGT-Omega, CoWTracker, 4RC and the Depth Anything 3 checkpoint are released under
non-commercial licences.

uv installs Python 3.13 automatically when needed. Runtime versions live in `pyproject.toml`,
with transitive dependencies pinned in `uv.lock`; PyTorch and torchvision use the CUDA 12.8 index.
Run commands with `uv run` (no activation needed), or `source .venv/bin/activate` to use plain `python`.
Rerun `bash scripts/setup.sh` to sync the locked environment; after intentional dependency changes, run
`uv lock` and commit both `pyproject.toml` and `uv.lock`. Set `UV_PROJECT_ENVIRONMENT` to use a custom
environment directory, or `UV_CACHE_DIR` to relocate downloads. Run CPU tests with
`uv run --extra dev pytest tests --ignore=tests/gpu -q`.

## Demo

```
uv run python scripts/demo_video.py assets/demo/cartwheel.mp4        # any video file or directory of frames
```

Runs Depth Anything 3 and AllTracker on the first 84 frames, lifts every pixel of frame 0 and corrects it. Writes
`outputs/demo/cartwheel/tracks.npz` (`xyz` (T,H,W,3): the 3-D trajectory of every pixel in the frame-0 camera's
coordinates, plus the raw lift, the observation mask and the cameras) and `tracks.mp4` (raw lift left, ReLift
right). `--geometry vggto` uses VGGT-Omega instead of Depth Anything 3.

To reproduce both local demos and the previews shown in chat:

```bash
uv run python scripts/run_demos.py
```

[`scripts/run_demos.py`](scripts/run_demos.py) downloads only the three default model weights,
fetches the basketball input from the project page with a SHA256 check, runs the cartwheel and
basketball clips through `demo_video.py` (84 frames each, DA3 + AllTracker), then renders the results.
Prepared geometry and tracking windows are reused by `demo_video.py` when present; use
`--output-root outputs/demo-fresh` for a new run without those cached windows.

To regenerate previews from existing trajectories without inference or downloads:

```bash
uv run python scripts/run_demos.py --render-only
# Or render any saved demo directories with adjustable sampling and playback:
uv run python scripts/render_demo_results.py outputs/demo/cartwheel outputs/demo/basketball \
    --frame-step 3 --pixel-stride 4 --fps 5 --preview-width 900
```

[`scripts/render_demo_results.py`](scripts/render_demo_results.py) contains all visualization code.
It writes `preview.png/gif` from the tracking video and `pointcloud.png/gif/mp4` from `tracks.npz`,
with the raw lift on the left and corrected tracks on the right. The 3D view uses reference-frame
RGB colours, every fourth grid pixel, a fixed camera (elevation 12°, azimuth −65°), and shared
axis limits at the reference frame's 2nd–98th percentiles. Coordinates are displayed as `(x, z, -y)`;
these are estimated geometry units. Every third frame is played at 5 fps. Each PNG shows the middle
sampled frame (frame 42 for these 84-frame clips).

`summary.json` reports frame counts, grid sizes, observation coverage, and coordinate finiteness;
observation coverage is the tracker's visibility fraction, not an accuracy measurement.
Each demo's `rendering.json` records the input trajectory file's SHA256, rendering settings, sampled
frame indices, and library versions. The defaults reproduce the previews from the local run;
neither script changes or restarts other GPU services.

For every source frame and interactive Plotly 3D viewers:

```bash
uv run python scripts/run_demos.py --num-frames 0 --plotly
# Rebuild only the interactive viewers from those saved full-length results:
uv run python scripts/run_demos.py --num-frames 0 --render-only --skip-previews --plotly
```

Full-length runs default to `outputs/demo-full/`, separate from the 84-frame recipe's cached windows.
Both supplied cartwheel and basketball videos contain exactly 84 frames (2.8 seconds at 30 FPS and
7 seconds at 12 FPS respectively); taking every frame therefore gives the same temporal extent as
the original demos. Longer videos run as one window, outside the corrector's published 84-frame
setting, and may need more GPU memory. The demo now records the source FPS and writes its comparison
video at that rate.

[`scripts/plotly_demo.py`](scripts/plotly_demo.py), [`scripts/plotly_viewer.js`](scripts/plotly_viewer.js),
and [`scripts/plotly_viewer.html`](scripts/plotly_viewer.html) generate `trajectories.html` and a
`plotly.json` manifest in each output directory. The HTML embeds Plotly and all its data for offline
viewing. Drag to orbit, scroll to zoom, play/pause or scrub the frame slider, and click any point to
see its entire path. Cameras stay synchronized between raw and corrected views. Every temporal frame
is included; a uniform subset of up to 2,500 reference pixels is shown for browser performance,
with complete paths for 60 representative moving points plus the selected point. The original
`tracks.npz` keeps trajectories for all 141,120 grid pixels in these runs. Axis ranges use the
0.5th–99.5th percentiles across both methods and all frames, with padding, to limit depth outliers.
These are estimated geometry coordinates, not measured metric accuracy.
Viewer export/temporal-coverage checks run with the Python suite; control-logic checks run with
`node tests/plotly_viewer.test.cjs` (no browser required). These control tests do not verify WebGL
rendering or visual layout.

Additional official input clips are selectable from [`assets/demo/catalog.json`](assets/demo/catalog.json):

```bash
uv run python scripts/run_demos.py --list-videos
uv run python scripts/download_demo_inputs.py running tennis motocross-bumps
uv run python scripts/run_demos.py --videos running tennis motocross-bumps --num-frames 0 --plotly
```

The catalog includes running, swing-tire, swing, spin, forest, tennis, motocross-bumps, drift-turn,
lady-running, Libby, motorbike, BMX-trees, kangaroo, horse jumping, schoolgirls, and camel in addition
to the two local demos. Downloads are cached under `cache/demo_inputs/` (with a checksum, frame count,
FPS, duration, and source URL recorded for each). They are project-page clips; the catalog does not
claim access to longer original recordings. The bundled cartwheel and existing basketball file stay
under `assets/demo/`.

To generate every catalog video's full-length interactive view and a gallery:

```bash
uv run python scripts/generate_all_views.py
```

[`scripts/generate_all_views.py`](scripts/generate_all_views.py) downloads and inspects inputs,
runs GPU inference sequentially, and exports each Plotly viewer. Valid completed results are reused;
rerun the same command to resume after interruption. It checks source frame counts, source FPS,
trajectory checksums, and viewer sizes before marking a clip complete. Per-clip `inference.log`
and `plotly.log` capture commands and output. This batch generates dense arrays and interactive
HTML; it skips the optional comparison MP4 and GIF generation.

Open `outputs/demo-full/index.html` to browse all scenes. `outputs/demo-full/batch.json` records
source metadata, completion status, and export metadata. The index updates as each clip finishes;
failed clips retain their error message while the remaining clips continue. Use `--download-only`
to inventory inputs or `--videos running forest` to process selected clips. Published clip lengths
vary: running has 206 frames and forest has 294; the full source timeline is retained.
After a CUDA out-of-memory error, the batch retries long-side resolutions 392, 336, and 280
(default 504), using smaller corrector batches. It retains every source frame and records each
attempt in `inference_res*.log` and `batch.json`. Reduced-resolution caches have separate
`window_res*` directories; successful default-resolution results are reused. Each viewer and
the gallery report the processing grid, and `plotly.json` records its resolution and corrector
batch. To select these settings manually, `demo_video.py` accepts `--process-res` and
`--corrector-batch`.

## Training

<details>
<summary>Prepare the PointOdyssey windows and train the corrector</summary>

Data: the annotation files of the official PointOdyssey training split
(`train.tar.gz`, from which only `<scene>/anno.npz` is needed), extracted to `data/pointodyssey/train/<scene>/anno.npz`.

```
uv run python scripts/prepare_pointodyssey.py --stream 0 &     # training windows -> cache/pointodyssey_windows
uv run python scripts/prepare_pointodyssey.py --stream 1 &     # (CPU; the three streams run in parallel)
uv run python scripts/prepare_pointodyssey.py --stream 2 &
uv run python scripts/train.py --run my_run                    # the released recipe -> outputs/runs/my_run/
uv run python scripts/export_checkpoint.py outputs/runs/my_run/final.pt checkpoints/my_run.pt
```

</details>

## Evaluation

<details>
<summary>Download WorldTrack, run the front ends and the corrector, print Table 1</summary>

Data: the 200 WorldTrack clips (ADT, Panoptic Studio, PointOdyssey, Dynamic Replica; 50 each), downloaded and
md5-checked by the first command. Then the front ends, the corrector and the scorer; the last command prints the
results table (APD3D, the layout of Table 1 of the paper).

```
uv run python scripts/download_worldtrack.py                                                 # -> data/worldtrack/
uv run python scripts/prepare_worldtrack.py --geometry da3 vggto rc4 --tracker alltracker
uv run python scripts/prepare_worldtrack.py --geometry --tracker cowtracker                  # own invocation (needs 48 GB);
                                                                                      # empty --geometry reuses the geometry above
uv run python scripts/run_corrector.py [--checkpoint checkpoints/my_run.pt]                  # the six tracker / geometry rows
uv run python scripts/score.py --variants alltracker_da3 alltracker_vggto alltracker_rc4 cowtracker_da3 cowtracker_vggto cowtracker_rc4
uv run python scripts/make_table1.py
```

Every GPU script takes `--shard i --nshards n` and skips finished clips.

</details>

## Repository layout

| Path | What it holds |
|---|---|
| `relift/` | The package: the corrector (`model.py`, `train.py`), the lift (`rays.py`), the five front-end wrappers (`providers/`), scoring (`eval/`) |
| `scripts/` | Every entry point: setup, downloads, demo, data preparation, training, inference, scoring |
| `configs/` | Paths, provider settings, weight sources with pinned revisions and md5s, the training recipe |
| `manifests/` | The WorldTrack file ids and md5s used by `download_worldtrack.py` |
| `packages/frontends/` | Python wheel packaging for the five pinned front ends; sources are downloaded and checksum-verified at build time |
| `tests/` | CPU unit tests (run in CI) and GPU tests that need weights and a clip |

Data, caches, weights and outputs live outside version control, under the roots in `configs/paths.yaml`; override
any of them with `RELIFT_<KEY>` (e.g. `RELIFT_DATA_ROOT=/mnt/big/relift_data`) or the matching command-line flag.

## Licence

Our code is MIT (`LICENSE`); the front-end Python dependencies keep their own licences, and four of the five
are non-commercial — see [`docs/frontend-dependencies.md`](docs/frontend-dependencies.md).
`configs/weights.yaml` records the licence, source and md5 of every weight the pipeline downloads.

## Citation

```bibtex
@misc{singer2026relift,
      title={ReLift: Dense 3D Tracking as Lifted Trajectory Repair},
      author={Assaf Singer and Maroun Khriesh and Noam Rotstein and Yoni Kasten and Or Litany},
      year={2026},
      eprint={2610.42136},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.42136}
}
```
