"""PointOdyssey training windows (paper App. A.3): ground-truth tracks -> 84-frame windows for training.

Per scene one file ``<cache>/<scene>.npz``:
    X       (n, T, P, 3) float32   ground-truth points in the reference (frame-0) camera coordinate system, metres
    vis     (n, T, P)    bool      PointOdyssey visibility AND inside the image AND in front of the camera
    origin  (n, T, 3)    float32   camera centre of every frame in that coordinate system
    dyn     (n, P)       bool      the point moves in the world by more than DYN_THR over the window
    step    (n,)         int8      temporal stride of the window (1 or 2)
    start   (n,)         int32

Windows start every T/2 strides; a window keeps the points that are valid, visible, inside the image and in front
of the camera in its reference frame, needs at least P/2 of them and stores P, at most half of them moving.

Reproducing the published cache exactly.  The cache the released checkpoint was trained on was built by three
processes, each taking every third scene of the sorted scene list and drawing its point subsets from its own
``numpy.random.default_rng(0)`` stream, consumed scene after scene.  ``build_stream`` replays one such stream;
running streams 0, 1 and 2 (of 3) reproduces the cache array for array.  Any other number of streams gives a
different, equally valid, point sampling.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..records.model import PointOdysseyWindows

T, P = 84, 2048
DYN_THR = 0.05  # metres of world displacement (max over the window)
STEPS = (1, 2)
EXCLUDE = {"cab_e_ego2"}  # the same recording as WorldTrack's cab_e_3rd
VAL_EVERY, VAL_FROM = 13, 6  # validation = every 13th scene name from the 7th


def windows_of(
    z: Any,
    rng: np.random.Generator,
    T: int = T,
    P: int = P,
    dyn_thr: float = DYN_THR,
    steps: tuple[int, ...] = STEPS,
) -> PointOdysseyWindows:
    """One scene's annotations (``anno.npz`` arrays) -> dict of lists (see the module docstring)."""
    Xw, uv, E, K = z["trajs_3d"], z["trajs_2d"], z["extrinsics"].astype(np.float64), z["intrinsics"]
    vis, val = z["visibs"], z["valids"]
    F = Xw.shape[0]
    W_img, H_img = 2.0 * K[0, 0, 2], 2.0 * K[0, 1, 2]
    out: PointOdysseyWindows = dict(X=[], vis=[], origin=[], dyn=[], step=[], start=[])
    for s in steps:
        span = (T - 1) * s + 1
        for t0 in range(0, F - span + 1, (T // 2) * s):
            idx = np.arange(t0, t0 + span, s)
            xw = Xw[idx].astype(np.float64)  # T,N,3
            ok = val[idx].all(0) & np.isfinite(xw).all((0, 2)) & (np.abs(xw).max((0, 2)) < 1e3)
            Rt = E[idx]
            xc = (
                np.einsum("tij,tnj->tni", Rt[:, :3, :3], xw) + Rt[:, None, :3, 3]
            )  # per-frame camera
            inside = (
                (uv[idx, :, 0] >= 0)
                & (uv[idx, :, 0] < W_img)
                & (uv[idx, :, 1] >= 0)
                & (uv[idx, :, 1] < H_img)
                & (xc[..., 2] > 0.1)
            )
            v = vis[idx] & inside
            cand = np.flatnonzero(ok & v[0])
            if len(cand) < P // 2:
                continue
            c2w0 = np.linalg.inv(Rt[0])
            M = np.einsum("tij,jk->tik", Rt, c2w0)  # frame-0 camera -> frame-t camera
            Minv = np.linalg.inv(M)  # frame-t camera -> frame-0 camera
            x0 = np.einsum("tij,tnj->tni", Minv[:, :3, :3], xc[:, cand]) + Minv[:, None, :3, 3]
            moved = np.linalg.norm(xw[:, cand] - xw[:1, cand], axis=-1).max(0) > dyn_thr
            d, st = np.flatnonzero(moved), np.flatnonzero(~moved)
            nd = min(len(d), P // 2) if len(st) >= P - min(len(d), P // 2) else min(len(d), P)
            pick = np.concatenate(
                [
                    rng.choice(d, nd, replace=False) if nd else d[:0],
                    rng.choice(st, P - nd, replace=len(st) < P - nd) if P - nd else st[:0],
                ]
            )
            out["X"].append(x0[:, pick].astype(np.float32))
            out["vis"].append(v[:, cand[pick]])
            out["origin"].append(Minv[:, :3, 3].astype(np.float32))
            out["dyn"].append(moved[pick])
            out["step"].append(s)
            out["start"].append(t0)
    return out


def scene_names(src: str | Path) -> list[str]:
    """Scenes with annotations, sorted: ``<src>/<scene>/anno.npz``."""
    return sorted(p.parent.name for p in Path(src).glob("*/anno.npz"))


def build_stream(
    src: str | Path,
    cache: str | Path,
    stream: int = 0,
    nstreams: int = 3,
    log: Callable[..., Any] = print,
    rewrite: bool = False,
) -> None:
    """Build the windows of every ``nstreams``-th scene starting at ``stream`` with one ``default_rng(0)`` stream.
    A scene whose file exists is still computed (the stream must advance) but not rewritten unless ``rewrite``."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for name in scene_names(src)[stream::nstreams]:
        dst = cache / f"{name}.npz"
        try:
            z = np.load(Path(src) / name / "anno.npz")
            o = windows_of(z, rng)
        except Exception as e:  # noqa: BLE001
            log(f"{name}: FAILED {e!r}")
            continue
        if not o["X"]:
            log(f"{name}: no usable window")
            continue
        if dst.exists() and not rewrite:
            log(f"{name}: {len(o['X'])} windows (file exists, kept)")
            continue
        tmp = cache / f".{name}.tmp.npz"
        np.savez(
            tmp,
            X=np.stack(o["X"]),
            vis=np.stack(o["vis"]),
            origin=np.stack(o["origin"]),
            dyn=np.stack(o["dyn"]),
            step=np.array(o["step"], np.int8),
            start=np.array(o["start"], np.int32),
        )
        os.replace(tmp, dst)
        log(
            f"{name}: {len(o['X'])} windows, dynamic share {np.mean(o['dyn']):.2f}, visible share {np.mean(o['vis']):.2f}"
        )


def split(cache: str | Path) -> tuple[list[str], list[str]]:
    """Training and validation scene names of a window cache: drop EXCLUDE, hold out names[VAL_FROM::VAL_EVERY]."""
    names = sorted(p.stem for p in Path(cache).glob("*.npz") if not p.name.startswith("."))
    names = [n for n in names if n not in EXCLUDE]
    val = names[VAL_FROM::VAL_EVERY]
    return [n for n in names if n not in val], val
