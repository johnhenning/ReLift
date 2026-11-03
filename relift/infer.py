"""Dense inference: from a prepared window to the corrected trajectory of every pixel.

``observations`` builds the raw lift's ingredients from the window; ``predict_dense`` runs the corrector over all
pixels in random groups and writes the corrected trajectories back to their pixels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .features import correct
from .model import Corrector
from .rays import DEPTH_FILE, TRACKS_FILE, build_rays, load_window

VIS_GATE = (
    0.85  # a cell is observed when the tracker's visibility exceeds this and the range is valid
)


def vis_remap(v: np.ndarray[Any, Any], tau: float, gate: float = VIS_GATE) -> np.ndarray[Any, Any]:
    """Monotone piecewise-linear map sending the calibrated threshold ``tau`` to the gate: v <= tau -> [0, gate],
    v > tau -> (gate, 1].  Used for CoWTracker (tau 0.65, App. A.1); the gate itself never changes."""
    v = np.asarray(v, np.float32)
    lo = gate * v / tau
    hi = gate + (1 - gate) * (v - tau) / max(1 - tau, 1e-6)
    return np.where(v <= tau, lo, hi).astype(np.float32)


def observations(
    window_dir: str | Path,
    num_frames: int,
    device: str = "cuda",
    tracks_file: Any = TRACKS_FILE,
    depth_file: Any = DEPTH_FILE,
    vis_tau: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Camera centres (T,3), ray directions (T,H,W,3), ranges (T,H,W) and tracker visibility (T,H,W) of a window.
    With ``vis_tau`` the visibility is remapped (``vis_remap``) and rounded to float16, as the stored tracker files
    are, before use."""
    win = load_window(window_dir, num_frames, device, tracks_file, depth_file)
    if vis_tau is not None:
        v = vis_remap(win["vis"].cpu().numpy(), vis_tau).astype(np.float16)
        win["vis"] = torch.as_tensor(v, dtype=torch.float32, device=win["vis"].device)
    origin, direction, distance = build_rays(win)
    return (
        origin,
        torch.nan_to_num(direction),
        torch.nan_to_num(distance),
        torch.nan_to_num(win["vis"]),
    )


def repeated_pixels(
    distance: torch.Tensor, group: int = 1024, batch: int = 8, seed: int = 0
) -> torch.Tensor:
    """The flat pixel indices that ``predict_dense`` repeats to fill its last batch (its partition rule replayed).
    Their prediction is one of two values written in undefined order; every other pixel is deterministic."""
    dev = distance.device
    T, H, W = distance.shape
    gen = torch.Generator(device=dev).manual_seed(seed)
    valid = torch.nonzero(distance.reshape(T, H * W)[0] > 0)[:, 0]
    perm = valid[torch.randperm(len(valid), device=dev, generator=gen)]
    out = []
    for s in range(0, len(perm), group * batch):
        sel = perm[s : s + group * batch]
        need = max(1, (len(sel) + group - 1) // group) * group - len(sel)
        if need > 0:
            out.append(sel[torch.arange(need, device=dev) % len(sel)])
    return torch.unique(torch.cat(out)) if out else valid[:0]


@torch.no_grad()
def predict_dense(
    model: Corrector,
    origin: torch.Tensor,
    direction: torch.Tensor,
    distance: torch.Tensor,
    visibility: torch.Tensor,
    group: int = 1024,
    batch: int = 8,
    seed: int = 0,
    scale: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Every reference pixel -> trajectories (T,H,W,3) in the reference camera's coordinate system (input units).

    Pixels with a valid reference range are shuffled once (CUDA generator, fixed seed) and cut into groups, so each
    group is a random scatter over the whole scene; ``batch`` groups run as one forward pass.  One depth scale for the
    whole clip.  Pixels whose reference range is invalid are not given to the model and keep their raw lift.
    """
    dev = distance.device
    T, H, W = distance.shape
    gen = torch.Generator(device=dev).manual_seed(seed)
    dirf = direction.reshape(T, H * W, 3)
    rf = distance.reshape(T, H * W)
    visf = visibility.reshape(T, H * W)
    out = (origin[:, None, :] + dirf * rf[..., None]).float()  # the lift, kept for invalid pixels
    valid = torch.nonzero(rf[0] > 0)[:, 0]
    if scale is None:
        scale = out[0, valid, 2].median().clamp_min(1e-6)
    perm = valid[torch.randperm(len(valid), device=dev, generator=gen)]
    K = model.chunk
    Tp = (T + K - 1) // K * K  # pad to a whole number of chunks
    pad = Tp - T
    for s in range(0, len(perm), group * batch):
        sel = perm[s : s + group * batch]
        nb = max(1, (len(sel) + group - 1) // group)
        need = nb * group - len(sel)
        if need > 0:  # fill the last batch by repeating pixels
            sel = torch.cat([sel, sel[torch.arange(need, device=dev) % len(sel)]])
        pix = sel.view(nb, group)
        d = dirf[:, pix].permute(1, 0, 2, 3)  # (nb, T, group, 3)
        r = rf[:, pix].permute(1, 0, 2)
        obs = (visf[:, pix].permute(1, 0, 2) > VIS_GATE) & (r > 0)
        obs[:, 0] = True
        Y = (origin[None, :, None, :] + d * r[..., None]) * obs[..., None]
        ray = d * obs[..., None]
        o = origin[None].expand(nb, T, 3)
        if pad:
            Y = torch.cat([Y, Y.new_zeros((nb, pad) + Y.shape[2:])], 1)
            ray = torch.cat([ray, ray.new_zeros((nb, pad) + ray.shape[2:])], 1)
            obs = torch.cat([obs, obs.new_zeros((nb, pad) + obs.shape[2:])], 1)
            o = torch.cat([o, o[:, -1:].expand(nb, pad, 3)], 1)
        x = correct(model, Y / scale, obs, ray, o / scale)
        out[:, pix.reshape(-1)] = (x[:, :T] * scale).permute(1, 0, 2, 3).reshape(T, -1, 3).float()
    return torch.nan_to_num(out).reshape(T, H, W, 3)
