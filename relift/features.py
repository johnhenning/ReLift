"""Shared by training and inference: normalisation, the initialised trajectory, the input features, the correction.

All tensors are batched: Y (B, T, P, 3) the lifted observation (zero where unobserved), obs (B, T, P) bool with
obs[:, 0] all True, ray (B, T, P, 3) the observed ray direction (zero where unobserved), origin (B, T, 3) the camera
centres, everything in the reference camera's coordinate system divided by the clip's scale.
"""

from __future__ import annotations

from typing import Any

import torch

S = 10.0  # feature / output scale: the network works in units of 10 % of the reference median depth
GAP_CAP = 40.0  # the gap-length feature saturates at this many frames


def normalise(
    Y: torch.Tensor, obs: torch.Tensor | None, origin: torch.Tensor, *more: Any
) -> tuple[torch.Tensor, ...]:
    """One scale per clip: the median lifted depth of the reference frame.  Returns (Y / s, origin / s, s, *more / s)."""
    s = Y[:, 0, :, 2].median(dim=1).values.clamp_min(1e-6)  # (B,)
    v = s.view(-1, 1, 1, 1)
    return (Y / v, origin / s.view(-1, 1, 1), s) + tuple(m / v for m in more)


def interp_init(Y: torch.Tensor, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per track: linear interpolation of Y between observed frames, held constant at the ends, and the distance in
    frames to the nearest observed frame (the gap length).  Y (B, T, P, 3), obs (B, T, P) with obs[:, 0] all True."""
    B, T, P, _ = Y.shape
    t = torch.arange(T, device=Y.device).view(1, T, 1).expand(B, T, P)
    last = torch.cummax(torch.where(obs, t, torch.full_like(t, -1)), 1).values
    nxt = torch.flip(
        torch.cummin(torch.flip(torch.where(obs, t, torch.full_like(t, T)), [1]), 1).values, [1]
    )
    has_n = nxt < T
    li, ni = last.clamp(0, T - 1), nxt.clamp(0, T - 1)
    Yl = torch.gather(Y, 1, li[..., None].expand(-1, -1, -1, 3))
    Yn = torch.gather(Y, 1, ni[..., None].expand(-1, -1, -1, 3))
    w = ((t - li).float() / (ni - li).clamp_min(1).float())[..., None]
    x = torch.where(has_n[..., None], (1 - w) * Yl + w * Yn, Yl)
    gap = torch.minimum(t - last, torch.where(has_n, nxt - t, torch.full_like(t, 10 * T)))
    return x, gap


start_field = interp_init


def features_clean(
    x_start: torch.Tensor,
    Y: torch.Tensor,
    obs: torch.Tensor,
    gap: torch.Tensor,
    ray: torch.Tensor,
    origin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The 14 input channels, relative to the anchor a = x_start[:, 0] (the reference-frame lift) and scaled by S:
    initialised trajectory (3), observed lift (3), observed flag (1), gap length (1), observed ray (3), camera centre
    relative to the reference camera (3).  Returns (features (B, T, P, 14), anchor (B, P, 3))."""
    a = x_start[:, :1]
    of = obs[..., None].float()
    B, T, P, _ = x_start.shape
    cam = ((origin - origin[:, :1]) * S)[:, :, None, :].expand(B, T, P, 3)
    f = torch.cat(
        [
            (x_start - a) * S,
            (Y - a) * S * of,
            of,
            (gap.float().clamp(max=GAP_CAP) / GAP_CAP)[..., None],
            ray * of,
            cam,
        ],
        -1,
    )
    return f, a[:, 0]


@torch.no_grad()
def correct(
    model: torch.nn.Module,
    Y: torch.Tensor,
    obs: torch.Tensor,
    ray: torch.Tensor,
    origin: torch.Tensor,
) -> torch.Tensor:
    """One pass of the corrector on normalised inputs -> trajectories (B, T, P, 3), bfloat16 autocast on CUDA."""
    x_start, gap = interp_init(Y, obs)
    f, a = features_clean(x_start, Y, obs, gap, ray, origin)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        u = model(f, a)
    return x_start + u.float() / S
