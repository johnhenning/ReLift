#!/usr/bin/env python
"""Build the prepared windows of the WorldTrack clips: geometry (depth, intrinsics, poses) and dense 2-D tracks.

    python scripts/prepare_worldtrack.py --geometry da3 vggto rc4 --tracker alltracker cowtracker \
        [--subsets adt pstudio po ds] [--shard i --nshards n] [--clips NAME ...] [--limit N]

Per clip (the first 84 frames of ``<worldtrack_root>/<subset_dir>/<clip>.npz``) writes into
``<windows_root>/<subset_dir>/<clip>/``:
    da3_depth_data.npz, vggto_depth_data.npz, rc4_depth_data.npz      depth f16, poses f32, intrinsics f32
    da3_frames.npz                                                    DA3's processed frames (the tracker's input)
    alltracker_tracks.npz, cowtracker_tracks.npz                      tracks_2d f16, visibility f16, confidence f16
    cowtracker_meta.json                                              size, timing, memory, visibility statistics
Files already present are skipped (shards and restarts share one tree).  ``da3`` must exist or be requested: it
defines the grid for every other provider.  Settings: configs/providers.yaml; weights: configs/weights.yaml.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from relift.config import REPO, paths  # noqa: E402
from relift.data.worldtrack import (  # noqa: E402
    NUM_FRAMES,
    clip_paths,
    load_clip,
    load_jpeg_frames_pillow,
    subset_dir,
)
from relift.provenance import note  # noqa: E402
from relift.providers import GEOMETRY, TRACKERS, depth_file, tracks_file  # noqa: E402


def savez(path: str | Path, **arrays: Any) -> None:
    tmp = str(path) + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def weight_path(cfg_w: Any, name: str, root: str | Path) -> Any:
    w = cfg_w[name]
    return (
        os.path.join(root, w.get("dir", ""), w["file"])
        if "file" in w
        else os.path.join(root, w["dir"])
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--geometry", nargs="*", default=["da3"], choices=GEOMETRY)
    ap.add_argument("--tracker", nargs="*", default=["alltracker"], choices=TRACKERS)
    ap.add_argument("--subsets", nargs="+", default=["adt", "pstudio", "po", "ds"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num_frames", type=int, default=NUM_FRAMES)
    ap.add_argument("--windows_root", default=None, help="default <cache_root>/worldtrack_windows")
    ap.add_argument("--worldtrack_root", default=None, help="default <data_root>/worldtrack")
    ap.add_argument("--checkpoints_root", default=None)
    ap.add_argument(
        "--weights",
        nargs="*",
        default=[],
        metavar="NAME=PATH",
        help="override a weight path, e.g. vggt_omega=/x/vggt_omega_1b_512.pt",
    )
    ap.add_argument("--providers_config", default=str(REPO / "configs" / "providers.yaml"))
    ap.add_argument(
        "--cow_max_pixels",
        type=int,
        default=None,
        help="override the config (changes the protocol)",
    )
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    P = paths()
    windows = a.windows_root or str(P["cache_root"] / "worldtrack_windows")
    wt = a.worldtrack_root or str(P["data_root"] / "worldtrack")
    ck_root = a.checkpoints_root or str(P["checkpoints_root"])
    pc = yaml.safe_load(open(a.providers_config))
    wc = yaml.safe_load(open(REPO / "configs" / "weights.yaml"))
    wpath = {n: weight_path(wc, n, ck_root) for n in wc}
    for kv in a.weights:
        n, p = kv.split("=", 1)
        wpath[n] = p
    want_g, want_t = list(a.geometry), list(a.tracker)
    models: dict[str, Any] = {}

    def model(name: str) -> Any:
        if name not in models:
            t0 = time.perf_counter()
            if name == "da3":
                from relift.providers.da3 import DepthAnything3Provider

                models[name] = DepthAnything3Provider(
                    wpath["da3"], a.device, **{k: v for k, v in pc["da3"].items() if k != "decoder"}
                )
            elif name == "alltracker":
                from relift.providers.alltracker import AllTrackerProvider

                models[name] = AllTrackerProvider(
                    wpath["alltracker"],
                    a.device,
                    pc["alltracker"]["window"],
                    pc["alltracker"]["iters"],
                )
            elif name == "vggto":
                from relift.providers.vggt_omega import VGGTOmegaProvider

                models[name] = VGGTOmegaProvider(
                    wpath["vggt_omega"], a.device, pc["vggto"]["resolution"], pc["vggto"]["patch"]
                )
            elif name == "rc4":
                from relift.providers.rc4 import RC4Provider

                models[name] = RC4Provider(
                    wpath["rc4"], a.device, **{k: v for k, v in pc["rc4"].items() if k != "decoder"}
                )
            elif name == "cowtracker":
                from relift.providers.cowtracker import CoWTrackerProvider

                models[name] = CoWTrackerProvider(
                    wpath["cowtracker"],
                    a.device,
                    pc["cowtracker"]["dtype"],
                    pc["cowtracker"]["attention"],
                )
            print(f"loaded {name} in {time.perf_counter() - t0:.0f}s", flush=True)
        return models[name]

    n_done = 0
    for subset in a.subsets:
        sd = subset_dir(subset)
        clips = clip_paths(wt, subset, a.shard, a.nshards)
        if a.clips:
            clips = [p for p in clips if p.stem in set(a.clips)]
        for path in clips[: a.limit or None]:
            name = path.stem
            W = os.path.join(windows, sd, name)
            os.makedirs(W, exist_ok=True)
            todo_g = [g for g in want_g if not os.path.exists(os.path.join(W, depth_file(g)))]
            todo_t = [t for t in want_t if not os.path.exists(os.path.join(W, tracks_file(t)))]
            need_da3_run = "da3" in todo_g or (
                "alltracker" in todo_t and not os.path.exists(os.path.join(W, "da3_frames.npz"))
            )
            if not todo_g and not todo_t:
                continue
            t0 = time.perf_counter()
            clip = (
                load_clip(path, a.num_frames)
                if (need_da3_run or set(todo_g) & {"vggto", "rc4"})
                else None
            )
            frames = (
                np.stack([np.array(im.convert("RGB")) for im in clip["frames"]]) if clip else None
            )
            T = len(clip["frames"]) if clip else a.num_frames
            log = []
            if need_da3_run:
                g = model("da3").run(frames)
                savez(
                    os.path.join(W, depth_file("da3")),
                    depth=g["depth"].astype(np.float16),
                    poses=g["poses"],
                    intrinsics=g["intrinsics"],
                )
                savez(os.path.join(W, "da3_frames.npz"), images=g["images"])
                note(os.path.join(W, "provenance.json"), "da3", a.device)
                log.append(f"da3 {g['depth'].shape[1]}x{g['depth'].shape[2]}")
            if not os.path.exists(os.path.join(W, depth_file("da3"))):
                raise SystemExit(
                    f"{sd}/{name}: da3_depth_data.npz is missing and da3 was not requested; it defines the grid"
                )
            with np.load(os.path.join(W, depth_file("da3"))) as z:
                H, Wd = z["depth"].shape[1:]
            if "alltracker" in todo_t:
                images = np.load(os.path.join(W, "da3_frames.npz"))["images"]
                tr, vis, conf = model("alltracker").run(images)
                savez(
                    os.path.join(W, tracks_file("alltracker")),
                    tracks_2d=tr.astype(np.float16),
                    visibility=vis.astype(np.float16),
                    confidence=conf.astype(np.float16),
                )
                note(os.path.join(W, "provenance.json"), "alltracker", a.device)
                log.append("alltracker")
            if "vggto" in todo_g:
                g = model("vggto").run(frames, H, Wd)
                savez(
                    os.path.join(W, depth_file("vggto")),
                    depth=g["depth"].astype(np.float16),
                    poses=g["poses"],
                    intrinsics=g["intrinsics"],
                )
                note(os.path.join(W, "provenance.json"), "vggto", a.device)
                log.append("vggto")
            if "rc4" in todo_g:
                g = model("rc4").run(clip, H, Wd)
                savez(
                    os.path.join(W, depth_file("rc4")),
                    depth=g["depth"].astype(np.float16),
                    poses=g["poses"],
                    intrinsics=g["intrinsics"],
                    model_hw=g["model_hw"],
                    valid_depth=g["valid_depth"],
                    check_px=g["check_px"],
                    check_rel=g["check_rel"],
                )
                note(os.path.join(W, "provenance.json"), "rc4", a.device)
                log.append(
                    f"rc4 (reproj {g['check_px']:.3f}px, track-vs-depth {g['check_rel']:.1e})"
                )
            if "cowtracker" in todo_t:
                from relift.providers.cowtracker import schema_check, vis_stats

                cf = load_jpeg_frames_pillow(path, a.num_frames)
                (tr, vis, conf), size, secs, peak = model("cowtracker").run(
                    cf, H, Wd, a.cow_max_pixels or pc["cowtracker"]["max_pixels"]
                )
                schema_check(tr, vis, conf, len(cf), H, Wd)
                tmp = os.path.join(W, tracks_file("cowtracker") + ".tmp.npz")
                np.savez_compressed(
                    tmp,
                    tracks_2d=tr.astype(np.float16),
                    visibility=vis.astype(np.float16),
                    confidence=conf.astype(np.float16),
                )
                os.replace(tmp, os.path.join(W, tracks_file("cowtracker")))
                meta = dict(
                    subset=sd,
                    clip=name,
                    T=len(cf),
                    native=[int(H), int(Wd)],
                    frame_hw=[int(cf.shape[1]), int(cf.shape[2])],
                    size=list(size),
                    max_pixels=a.cow_max_pixels or pc["cowtracker"]["max_pixels"],
                    dtype=pc["cowtracker"]["dtype"],
                    attention=pc["cowtracker"]["attention"],
                    seconds=secs,
                    peak_gib=peak,
                    cow=vis_stats(vis, "cowtracker"),
                    created=time.strftime("%Y-%m-%dT%H:%M:%S"),
                )
                if os.path.exists(os.path.join(W, tracks_file("alltracker"))):
                    with np.load(os.path.join(W, tracks_file("alltracker"))) as at:
                        meta["alltracker"] = vis_stats(
                            at["visibility"][: len(cf)].astype(np.float32), "alltracker"
                        )
                json.dump(
                    meta,
                    open(os.path.join(W, "cowtracker_meta.json"), "w"),
                    indent=1,
                    default=float,
                )
                note(os.path.join(W, "provenance.json"), "cowtracker", a.device, size=list(size))
                log.append(f"cowtracker {size[0]}x{size[1]} {secs:.1f}s {peak:.1f}GiB")
            n_done += 1
            print(
                f"{sd}/{name} T={T}: {', '.join(log)} ({time.perf_counter() - t0:.1f}s)", flush=True
            )
    print(f"PREPARE_DONE: {n_done} clips", flush=True)


if __name__ == "__main__":
    main()
