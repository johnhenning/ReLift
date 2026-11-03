#!/usr/bin/env python
"""Run the corrector on prepared WorldTrack windows and save the sparse predictions the scorer reads.

    python scripts/run_corrector.py [--rows alltracker_da3 ...] [--subsets adt pstudio po ds] [--shard i --nshards n]

Per clip of each subset and per row (<tracker>_<geometry>), the window directory
``<windows_root>/<subset_dir>/<clip>/`` must hold ``<tracker>_tracks.npz`` and ``<geometry>_depth_data.npz``
(written by prepare_worldtrack.py).  The raw lift is built from them (no cleaning of any kind), the corrector runs
over every pixel (``predict_dense``), and the trajectories at 4RC's query pixels are written to
``<preds_root>/<row>/<subset_dir>/<clip>.npz`` together with the tracker's visibility there and the GPU model that produced them.  Clips already written
are skipped, so several shards or restarts can share one output directory.

All settings come from the config (configs/eval/table1.yaml): visibility gate, group and batch sizes, the group
seed, and the per-tracker visibility remap.  CUDA is required (the pixel-to-group partition and the
depth sampling are defined on the CUDA path).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from relift.checkpoint import load  # noqa: E402
from relift.config import REPO, paths  # noqa: E402
from relift.data.worldtrack import clip_paths, load_clip, read_out, subset_dir  # noqa: E402
from relift.infer import observations, predict_dense  # noqa: E402
from relift.provenance import machine  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=str(REPO / "configs" / "eval" / "table1.yaml"))
    ap.add_argument(
        "--rows",
        nargs="+",
        default=None,
        help="<tracker>_<geometry>; default: every row of the config",
    )
    ap.add_argument("--subsets", nargs="+", default=None, help="default: the config's subsets")
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="default <checkpoints_root>/<config corrector.checkpoint>",
    )
    ap.add_argument("--windows_root", default=None, help="default <cache_root>/worldtrack_windows")
    ap.add_argument("--worldtrack_root", default=None, help="default <data_root>/worldtrack")
    ap.add_argument("--preds_root", default=None, help="default <outputs_root>/preds")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--clips", nargs="*", default=None, help="only these clip names")
    ap.add_argument("--limit", type=int, default=0, help="first N clips per subset")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    P = paths()
    ckpt = a.checkpoint or str(P["checkpoints_root"] / cfg["corrector"]["checkpoint"])
    windows = a.windows_root or P["cache_root"] / "worldtrack_windows"
    wt = a.worldtrack_root or P["data_root"] / "worldtrack"
    preds = a.preds_root or P["outputs_root"] / "preds"
    rows = a.rows or [f"{r['tracker']}_{r['geometry']}" for r in cfg["rows"]]
    subsets = a.subsets or cfg["subsets"]
    C = cfg["corrector"]
    T_max = int(cfg["num_frames"])
    model, info = load(ckpt, a.device)
    prov = {k: np.array(str(v)) for k, v in machine(a.device).items() if k in ("gpu", "torch")}
    print(
        f"{ckpt}: step {info['step']} | rows {rows} | subsets {subsets} | shard {a.shard}/{a.nshards} -> {preds}",
        flush=True,
    )
    n_done, times = 0, []
    for subset in subsets:
        paths_ = clip_paths(wt, subset, a.shard, a.nshards)
        if a.clips:
            paths_ = [p for p in paths_ if p.stem in set(a.clips)]
        for path in paths_[: a.limit or None]:
            name = path.stem
            W = os.path.join(windows, subset_dir(subset), name)
            dsts = {r: os.path.join(preds, r, subset_dir(subset), f"{name}.npz") for r in rows}
            if all(os.path.exists(d) for d in dsts.values()):
                continue
            if not os.path.isdir(W):
                print(f"MISSING WINDOW {subset_dir(subset)}/{name}", flush=True)
                continue
            t0 = time.perf_counter()
            clip = load_clip(path, T_max)
            T = len(clip["frames"])
            n_q = 0
            for row, dst in dsts.items():
                if os.path.exists(dst):
                    continue
                tracker, geometry = row.split("_", 1)
                tf, df = f"{tracker}_tracks.npz", f"{geometry}_depth_data.npz"
                if not (
                    os.path.exists(os.path.join(W, tf)) and os.path.exists(os.path.join(W, df))
                ):
                    print(
                        f"MISSING {subset_dir(subset)}/{name}: {tf if not os.path.exists(os.path.join(W, tf)) else df}",
                        flush=True,
                    )
                    continue
                with np.load(os.path.join(W, df)) as z:
                    geo = dict(poses=z["poses"], intrinsics=z["intrinsics"])
                o, d, r, vis = observations(
                    W, T, a.device, tf, df, vis_tau=cfg["trackers"][tracker]["vis_tau"]
                )
                visf = vis.float().cpu().numpy().astype(np.float16)
                field = (
                    predict_dense(
                        model, o, d, r, vis, group=C["group"], batch=C["batch"], seed=C["seed"]
                    )
                    .float()
                    .cpu()
                    .numpy()
                )
                n_q = read_out(field, visf, geo, clip, dst, **prov)
                del o, d, r, vis, field
                torch.cuda.empty_cache()
            times.append(time.perf_counter() - t0)
            n_done += 1
            print(f"{subset_dir(subset)}/{name} T={T} queries={n_q} {times[-1]:.1f}s", flush=True)
    print(
        f"RUN_CORRECTOR_DONE: {n_done} clips"
        + (f", mean {np.mean(times):.1f}s per clip over {len(rows)} rows" if times else ""),
        flush=True,
    )


if __name__ == "__main__":
    main()
