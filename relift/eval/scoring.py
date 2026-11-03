"""Scoring saved sparse predictions with 4RC's own alignment and metric functions (84 frames, seeded).

Alignment and metric are imported from 4RC's ``track_eval_util`` (never re-implemented):
  * sim3   : ``compute_average_pts_within_thresh(..., scaling="sim3")``: RANSAC Sim(3) over a random 16,384 sample of
             all point-frames, 1,000 iterations, 0.05 inlier threshold, closed-form refit on the inliers
  * global : ``scaling="global"``: median norm ratio, no rotation or translation
Per clip and alignment: APD3D over all point-frames at 4RC's four thresholds (0.1, 0.3, 0.5, 1.0 m), EPE, the
dynamic-point APD (ground-truth path longer than 1 cm, re-aligned on those points alone, as 4RC does), and, not in
4RC's code, APD over the occluded and the visible cells under the all-point alignment.

4RC's RANSAC generator is unseeded.  With ``seed_ransac`` (the paper's setting) the 16,384-point sample and the
RANSAC draws are seeded per clip and alignment, so every method is scored on the same draws:
``seed = sum(ord(c) for c in name + mode) % 2**31`` and ``seed + 1`` for the dynamic points.
"""

from __future__ import annotations

import glob
import json
import os
from collections.abc import Callable
from typing import Any

import numpy as np

from ..data.worldtrack import NUM_FRAMES, clip_paths, load_clip, subset_dir
from ..vendor import rc4_eval

MODES = ("sim3", "global")
DYN_PATH_LENGTH = 0.01


def clip_seed(name: str, mode: str) -> int:
    return sum(map(ord, name + mode)) % 2**31


def apd(d: np.ndarray[Any, Any], thresholds: Any) -> float:
    """Mean over thresholds of the fraction of distances within each; NaN for an empty set."""
    return float(np.mean([(d <= t).mean() for t in thresholds])) if d.size else float("nan")


def _seeded(seed: int, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run fn with ``np.random.default_rng()`` (4RC's RANSAC generator) seeded, then restore it."""
    rng = np.random.default_rng
    fixed_seed = seed

    def seeded(seed: Any = None) -> np.random.Generator:
        return rng(fixed_seed if seed is None else seed)

    np.random.default_rng = seeded
    try:
        return fn(*args, **kwargs)
    finally:
        np.random.default_rng = rng


def score_clip(
    name: str,
    gt: np.ndarray[Any, Any],
    pred: np.ndarray[Any, Any],
    occ: np.ndarray[Any, Any],
    intrinsics: np.ndarray[Any, Any],
    seed_ransac: bool = True,
) -> dict[str, Any]:
    """gt, pred (T, N, 3) float64, occ (T, N) bool, intrinsics (fx, fy, cx, cy) -> the per-clip record."""
    _, U = rc4_eval()
    thresholds = list(U.PIXEL_TO_FIXED_METRIC_THRESH.values())
    call = (
        (lambda s, *k, **kw: _seeded(s, U.compute_average_pts_within_thresh, *k, **kw))
        if seed_ransac
        else (lambda s, *k, **kw: U.compute_average_pts_within_thresh(*k, **kw))
    )
    pred = np.nan_to_num(pred, nan=1e6, posinf=1e6, neginf=-1e6)
    dyn = np.linalg.norm(gt[1:] - gt[:-1], axis=-1).sum(0) > DYN_PATH_LENGTH
    r: dict[str, Any] = dict(
        n=int(gt.shape[1]), dyn=int(dyn.sum()), occ_cells=int(occ.sum()), T=int(gt.shape[0])
    )
    for mode in MODES:
        s = clip_seed(name, mode)
        np.random.seed(s)  # the 16,384-point sample
        avg, aligned, _, _, epe = call(
            s, gt, pred, scaling=mode, intrinsics_params=intrinsics, compute_epe=True
        )
        d = np.linalg.norm(aligned - gt, axis=-1)
        r[mode] = dict(
            apd=float(avg),
            epe=float(epe),
            occ=apd(d[occ], thresholds),
            vis=apd(d[~occ], thresholds),
        )
        if dyn.sum() > 0:
            if seed_ransac:
                np.random.seed(s + 1)
            r[mode]["dyn"] = float(
                call(
                    s + 1,
                    gt[:, dyn],
                    pred[:, dyn],
                    scaling=mode,
                    use_fixed_metric_threshold=True,
                    intrinsics_params=intrinsics,
                    pred_aligned=None,
                )[0]
            )
    return r


def score_subset(
    preds_dir: Any,
    worldtrack_root: Any,
    subset: Any,
    dst: Any,
    num_frames: int = NUM_FRAMES,
    seed_ransac: bool = True,
    log: Callable[..., Any] = print,
) -> dict[str, Any]:
    """Score every ``<preds_dir>/<clip>.npz`` of one subset into the JSON ``dst`` ({clip: record}); clips already in
    ``dst`` are kept (resumable).  Clips without an in-image query point are skipped, as in 4RC's harness."""
    rows = json.load(open(dst)) if os.path.exists(dst) else {}
    gt_paths = {p.stem: p for p in clip_paths(worldtrack_root, subset)}
    for f in sorted(glob.glob(f"{preds_dir}/*.npz")):
        name = os.path.basename(f)[:-4]
        if name in rows:
            continue
        clip = load_clip(gt_paths[name], num_frames)
        z = np.load(f)
        cols = z["gt_columns"]
        pred = z["tracks"][:num_frames].astype(np.float64)
        gt = clip["world"][:, cols].astype(np.float64)
        occ = ~clip["visibility"][:, cols]
        T = min(len(gt), len(pred))
        gt, pred, occ = gt[:T], pred[:T], occ[:T]
        if gt.shape[1] == 0:
            log(f"skip (no query points): {subset_dir(subset)}/{name}")
            continue
        rows[name] = score_clip(name, gt, pred, occ, clip["intrinsics"], seed_ransac)
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        json.dump(rows, open(dst, "w"))
    return rows
