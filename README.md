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

uv installs Python 3.10 automatically when needed. Runtime versions live in `pyproject.toml`,
with transitive dependencies pinned in `uv.lock`; PyTorch and torchvision use the CUDA 12.8 index.
Run commands with `uv run` (no activation needed), or `source .venv/bin/activate` to use plain `python`.
Rerun `bash scripts/setup.sh` to sync the locked environment; after intentional dependency changes, run
`uv lock` and commit both `pyproject.toml` and `uv.lock`. Set `UV_PROJECT_ENVIRONMENT` to use a custom
environment directory, or `UV_CACHE_DIR` to relocate downloads. Run CPU tests with
`uv run --extra dev pytest tests --ignore=tests/gpu -q`.

## Demo

```
python scripts/demo_video.py assets/demo/cartwheel.mp4        # any video file or directory of frames
```

Runs Depth Anything 3 and AllTracker on the first 84 frames, lifts every pixel of frame 0 and corrects it. Writes
`outputs/demo/cartwheel/tracks.npz` (`xyz` (T,H,W,3): the 3-D trajectory of every pixel in the frame-0 camera's
coordinates, plus the raw lift, the observation mask and the cameras) and `tracks.mp4` (raw lift left, ReLift
right). `--geometry vggto` uses VGGT-Omega instead of Depth Anything 3.

## Training

<details>
<summary>Prepare the PointOdyssey windows and train the corrector</summary>

Data: the annotation files of the official PointOdyssey training split
(`train.tar.gz`, from which only `<scene>/anno.npz` is needed), extracted to `data/pointodyssey/train/<scene>/anno.npz`.

```
python scripts/prepare_pointodyssey.py --stream 0 &     # training windows -> cache/pointodyssey_windows
python scripts/prepare_pointodyssey.py --stream 1 &     # (CPU; the three streams run in parallel)
python scripts/prepare_pointodyssey.py --stream 2 &
python scripts/train.py --run my_run                    # the released recipe -> outputs/runs/my_run/
python scripts/export_checkpoint.py outputs/runs/my_run/final.pt checkpoints/my_run.pt
```

</details>

## Evaluation

<details>
<summary>Download WorldTrack, run the front ends and the corrector, print Table 1</summary>

Data: the 200 WorldTrack clips (ADT, Panoptic Studio, PointOdyssey, Dynamic Replica; 50 each), downloaded and
md5-checked by the first command. Then the front ends, the corrector and the scorer; the last command prints the
results table (APD3D, the layout of Table 1 of the paper).

```
python scripts/download_worldtrack.py                                                 # -> data/worldtrack/
python scripts/prepare_worldtrack.py --geometry da3 vggto rc4 --tracker alltracker
python scripts/prepare_worldtrack.py --geometry --tracker cowtracker                  # own invocation (needs 48 GB);
                                                                                      # empty --geometry reuses the geometry above
python scripts/run_corrector.py [--checkpoint checkpoints/my_run.pt]                  # the six tracker / geometry rows
python scripts/score.py --variants alltracker_da3 alltracker_vggto alltracker_rc4 cowtracker_da3 cowtracker_vggto cowtracker_rc4
python scripts/make_table1.py
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
