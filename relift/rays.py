"""Reading a prepared window and building its rays.

A prepared window is what the data-preparation step writes for one clip: the 2-D tracker's dense track field and
visibility, and the geometry model's depth, intrinsics and camera-from-world poses.  Nothing else is read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TRACKS_FILE = "alltracker_tracks.npz"
DEPTH_FILE = "depth_data.npz"


def load_window(
    window_dir: str | Path,
    num_frames: int = 84,
    device: str = "cuda",
    tracks_file: str = TRACKS_FILE,
    depth_file: str = DEPTH_FILE,
) -> dict:
    """Tracks (T,H,W,2), visibility (T,H,W), depth (T,H,W), intrinsics (T,3,3) and poses (T,4,4) of one window."""
    at = np.load(f"{window_dir}/{tracks_file}")
    vg = np.load(f"{window_dir}/{depth_file}")
    T = min(num_frames, at["tracks_2d"].shape[0], vg["depth"].shape[0])
    dev = torch.device(device)
    to = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)  # noqa: E731
    out = dict(
        tracks=to(at["tracks_2d"][:T]),
        vis=to(at["visibility"][:T]),
        depth=to(vg["depth"][:T]),
        intrinsics=to(vg["intrinsics"][:T]),
        poses=to(vg["poses"][:T]),
        T=T,
    )
    out["H"], out["W"] = out["depth"].shape[1:3]
    return out


def normalised_coords(
    tracks: torch.Tensor, intrinsics: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tracked pixels (T,H,W,2) in normalised camera coordinates under per-frame intrinsics (T,3,3)."""
    T = intrinsics.shape[0]
    fx, fy = intrinsics[:, 0, 0].view(T, 1, 1), intrinsics[:, 1, 1].view(T, 1, 1)
    cx, cy = intrinsics[:, 0, 2].view(T, 1, 1), intrinsics[:, 1, 2].view(T, 1, 1)
    return (tracks[..., 0] - cx) / fx, (tracks[..., 1] - cy) / fy


def build_rays(win: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rays of every pixel: camera centre per frame (T,3), unit direction (T,H,W,3), range along the ray (T,H,W).

    Depth is sampled at the tracked pixel (nearest neighbour, clamped to the border), poses are expressed relative to
    the reference frame, and directions are normalised in that coordinate system.  CUDA only: the nearest-neighbour
    sampling of ``grid_sample`` rounds differently on CPU, so CPU results would not match the published numbers.
    """
    T, H, W = win["T"], win["H"], win["W"]
    P = win["poses"].clone()
    inv0 = torch.eye(4, device=P.device, dtype=P.dtype)
    r0, t0 = P[0, :3, :3], P[0, :3, 3]
    inv0[:3, :3] = r0.T
    inv0[:3, 3] = -r0.T @ t0
    P = P @ inv0
    coords = win["tracks"]
    norm = torch.stack(
        [2.0 * coords[..., 0] / max(W - 1, 1) - 1.0, 2.0 * coords[..., 1] / max(H - 1, 1) - 1.0],
        dim=-1,
    )
    z = F.grid_sample(
        win["depth"][:, None],
        norm.reshape(T, H * W, 1, 2),
        mode="nearest",
        padding_mode="border",
        align_corners=True,
    )[:, 0, :, 0].reshape(T, H, W)
    xc, yc = normalised_coords(coords, win["intrinsics"])
    dcam = F.normalize(torch.stack([xc, yc, torch.ones_like(xc)], dim=-1), dim=-1)
    Rwc = P[:, :3, :3].transpose(-2, -1)
    origin = -torch.einsum("tij,tj->ti", Rwc, P[:, :3, 3])
    direction = torch.einsum("tij,thwj->thwi", Rwc, dcam)
    distance = z * torch.sqrt(xc * xc + yc * yc + 1.0)
    return origin, direction, distance


def lift(origin: torch.Tensor, direction: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    """The raw lift of every pixel and frame: o + r d, (T,H,W,3)."""
    return origin[:, None, None, :] + direction * distance[..., None]
