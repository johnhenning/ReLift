"""The WorldTrack evaluation clips (4 subsets x 50 clips, the release used by 4RC) and the 4RC read-out rule.

Layout: ``<worldtrack_root>/<subset>/<clip>.npz`` with subset in SUBSETS.  Each file holds ``images_jpeg_bytes``,
``tracks_XYZ`` (T, N, 3), ``visibility`` (T, N) and ``fx_fy_cx_cy``.  Clips are loaded with 4RC's own loader so that
frames (OpenCV JPEG decoding), projected query pixels and ground truth are exactly what the protocol scores.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from ..records.model import WorldTrackClip
from ..vendor import rc4_eval

SUBSETS = {"adt": "adt_mini", "pstudio": "pstudio_mini", "po": "po_mini", "ds": "ds_mini"}
NAMES = {"adt": "ADT", "pstudio": "PS", "po": "PO", "ds": "DR"}
NUM_FRAMES = 84


def subset_dir(key_or_dir: str) -> str:
    """'adt' -> 'adt_mini'; a directory name is returned unchanged."""
    return SUBSETS.get(key_or_dir, key_or_dir)


def _natural(path: str | Path) -> list[int | str]:
    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r"(\d+)", os.path.basename(str(path)))
    ]


def clip_paths(root: str | Path, subset: str, shard: int = 0, nshards: int = 1) -> list[Path]:
    """The clips of a subset in natural order (the order of the research runs), optionally one shard of them."""
    return sorted(Path(root).joinpath(subset_dir(subset)).glob("*.npz"), key=_natural)[
        shard::nshards
    ]


def load_clip(path: str | Path, num_frames: int = NUM_FRAMES) -> WorldTrackClip:
    """-> dict(frames: list of PIL images, tracks_uv (T,N,2), intrinsics (4,), world (T,N,3), visibility (T,N) bool)."""
    TE, _ = rc4_eval()
    out = TE.load_npz_data(str(path), num_frames=num_frames)
    return dict(
        frames=out[0][:num_frames],
        tracks_uv=np.asarray(out[2]),
        intrinsics=np.asarray(out[3]),
        world=np.asarray(out[4])[:num_frames],
        visibility=np.asarray(out[5])[:num_frames].astype(bool),
        name=Path(path).stem,
    )


def load_jpeg_frames_pillow(path: str | Path, num_frames: int = NUM_FRAMES) -> np.ndarray[Any, Any]:
    """The clip's frames decoded by Pillow, (T,H,W,3) uint8 RGB: the decoding CoWTracker's windows were built with."""
    import io

    from PIL import Image

    g = np.load(path, allow_pickle=True)
    return np.stack(
        [
            np.array(Image.open(io.BytesIO(bytes(b))).convert("RGB"))
            for b in g["images_jpeg_bytes"][:num_frames]
        ]
    )


def read_out(
    field: np.ndarray[Any, Any],
    visf: np.ndarray[Any, Any],
    geo: dict[str, np.ndarray[Any, Any]],
    clip: WorldTrackClip,
    dst: str | Path,
    **extra: Any,
) -> int:
    """4RC's read-out of a dense prediction: the ground-truth points visible in frame 0, scaled from the image to the
    prediction grid, dropped when outside it, truncated to integer pixels.  Writes the sparse prediction file the
    scorer reads and returns the number of scored points."""
    mask = clip["visibility"][0]
    q_full = clip["tracks_uv"][0, mask]
    ph, pw = field.shape[1:3]
    gw, gh = clip["frames"][0].size
    q = q_full * np.array([pw, ph]) / np.array([gw, gh])
    inb = (q[:, 0] >= 0) & (q[:, 0] < pw) & (q[:, 1] >= 0) & (q[:, 1] < ph)
    u, v = q[inb, 0].astype(int), q[inb, 1].astype(int)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        **extra,
        tracks=field[:, v, u].astype(np.float32),
        pred_visibility=visf[:, v, u],
        query_uv_pred=q[inb].astype(np.float32),
        query_uv_full=q_full[inb].astype(np.float32),
        in_image=inb,
        gt_columns=np.flatnonzero(mask)[inb].astype(np.int32),
        poses_w2c=np.asarray(geo["poses"], np.float32),
        intrinsics=np.asarray(geo["intrinsics"], np.float32),
        field_hw=np.array([ph, pw]),
        image_wh=np.array([gw, gh]),
        num_frames=field.shape[0],
    )
    os.replace(tmp, dst)
    return int(inb.sum())
