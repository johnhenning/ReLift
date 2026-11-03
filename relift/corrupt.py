"""The simulated lift used for training: a noisy lifted observation of clean 3-D tracks (paper App. A.3).

Batched torch on the training device.  Inputs are in the reference camera's coordinate system (any unit):
    X (B,T,P,3) clean points, vis (B,T,P) ground-truth visibility, origin (B,T,3) camera centres, dyn (B,P) moving.
Output: Y (B,T,P,3) lifted observation (0 where unobserved), obs (B,T,P) bool, ray (B,T,P,3) observed ray direction.

Along the ray, the log-range error is a per-frame flicker (AR1) + a per-frame image-plane tilt (AR1) + a per-track
slowly varying bias (AR1) + white noise + outlier jumps; across the ray, a small angular noise.  The observed set is
the ground-truth visibility minus synthetic occluders in front of a moving track and tracker dropouts, plus
false-visible cells at occluder borders that carry the occluder's depth.  The reference frame is always observed.
Every level is drawn once per sample from the ranges of Table A.2.
"""

from __future__ import annotations

import torch


def _ar1(
    shape: tuple[int, ...],
    rho: float | torch.Tensor,
    gen: torch.Generator | None,
    device: str | torch.device,
) -> torch.Tensor:
    """Unit-variance AR(1) along dim 1, started from the stationary distribution.  rho broadcastable without dim 1."""
    e = torch.randn(shape, device=device, generator=gen)
    if torch.is_tensor(rho):  # one rho per sample
        rho = rho.reshape((shape[0],) + (1,) * (len(shape) - 2))
    out = torch.empty_like(e)
    out[:, 0] = e[:, 0]
    c = (1.0 - rho**2) ** 0.5
    for t in range(1, shape[1]):
        out[:, t] = rho * out[:, t - 1] + c * e[:, t]
    return out


def _u(
    B: int,
    lo: float,
    hi: float,
    gen: torch.Generator | None,
    device: str | torch.device,
    extra: int = 1,
) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand((B,) + (1,) * extra, device=device, generator=gen)


def occluder_mask(
    X: torch.Tensor, dyn: torch.Tensor, gen: torch.Generator | None, p_apply: float = 0.6
) -> tuple[torch.Tensor, torch.Tensor]:
    """(B,T,P) bool hidden by a synthetic occluder, and the occluder's depth ratio z_occ / z (B,T,P) on those cells."""
    B, T, P, _ = X.shape
    dev = X.device
    z = X[..., 2].clamp_min(1e-3)
    q = X[..., :2] / z[..., None]  # projective image coordinates
    hidden = torch.zeros((B, T, P), dtype=torch.bool, device=dev)
    ratio = torch.ones((B, T, P), device=dev)
    tt = torch.arange(T, device=dev).view(1, T, 1)
    for _rep, p in enumerate((p_apply, 0.2)):
        w = dyn.float() + 1e-3  # prefer a moving track as the target
        j = torch.multinomial(w, 1, generator=gen)[:, 0]
        ts = torch.randint(T // 4, T, (B,), device=dev, generator=gen)
        bi = torch.arange(B, device=dev)
        c = q[bi, ts, j]  # (B, 2)
        zt = z[bi, ts, j]
        half = 0.03 + 0.22 * torch.rand((B, 2), device=dev, generator=gen)
        drift = 0.002 * torch.randn((B, 2), device=dev, generator=gen)
        centre = (
            c[:, None, None, :]
            + drift[:, None, None, :] * (tt[..., None] - ts.view(B, 1, 1, 1)).float()
        )
        inside = ((q - centre).abs() < half[:, None, None, :]).all(-1)
        z_occ = (zt * _u(B, 0.5, 0.9, gen, dev, 0))[:, None, None]
        on = (torch.rand(B, device=dev, generator=gen) < p)[:, None, None]
        h = inside & (z > z_occ) & on & (X[..., 2] > 1e-2)
        ratio = torch.where(h & ~hidden, z_occ / z, ratio)
        hidden |= h
    return hidden, ratio


def corrupt(
    X: torch.Tensor,
    vis: torch.Tensor,
    origin: torch.Tensor,
    dyn: torch.Tensor,
    gen: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, P, _ = X.shape
    dev = X.device
    rel = X - origin[:, :, None, :]
    r = torch.linalg.vector_norm(rel, dim=-1).clamp_min(1e-4)
    d = rel / r[..., None]

    # ---- observed set
    occ, ratio = occluder_mask(X, dyn, gen)
    drop_pt = torch.rand((B, P), device=dev, generator=gen) < 0.10
    s0 = torch.randint(1, T, (B, P), device=dev, generator=gen)
    ln = torch.randint(2, 21, (B, P), device=dev, generator=gen)
    tt = torch.arange(T, device=dev).view(1, T, 1)
    drop = drop_pt[:, None] & (tt >= s0[:, None]) & (tt < (s0 + ln)[:, None])
    obs = vis & ~occ & ~drop
    near = torch.zeros_like(obs)  # occluded cells next to an observed one
    near[:, 1:] |= obs[:, :-1]
    near[:, :-1] |= obs[:, 1:]
    false_vis = occ & vis & near & (torch.rand((B, T, P), device=dev, generator=gen) < 0.3)
    obs = obs | false_vis
    obs[:, 0] = True
    false_vis[:, 0] = False

    # ---- along-ray error (log range)
    sg, rg = _u(B, 0.0, 0.03, gen, dev), _u(B, 0.3, 0.95, gen, dev)
    flick = sg * _ar1((B, T), rg, gen, dev)  # (B, T)
    st = _u(B, 0.0, 0.03, gen, dev, 2)
    tilt = st * _ar1((B, T, 2), rg[..., None], gen, dev)  # (B, T, 2)
    qx = d[..., :2]  # bounded image-plane proxy
    sb = _u(B, 0.0, 0.02, gen, dev, 2)
    bias = sb * _ar1((B, T, P), 0.9, gen, dev)
    sw = _u(B, 0.002, 0.02, gen, dev, 2)
    white = sw * torch.randn((B, T, P), device=dev, generator=gen)
    p_out = _u(B, 0.0, 0.05, gen, dev, 2)
    m = torch.rand((B, T, P), device=dev, generator=gen) < p_out
    m[:, 1:] |= m[:, :-1] & (torch.rand((B, T - 1, P), device=dev, generator=gen) < 0.5)
    sign = torch.where(torch.rand((B, T, P), device=dev, generator=gen) < 0.5, -1.0, 1.0)
    jump = m.float() * sign * (0.1 + 0.4 * torch.rand((B, T, P), device=dev, generator=gen))
    log_err = flick[..., None] + (tilt[:, :, None, :] * qx).sum(-1) + bias + white + jump
    r_obs = r * torch.exp(log_err.clamp(-1.0, 1.0))
    r_obs = torch.where(false_vis, r * ratio, r_obs)  # the tracker sits on the occluder

    # ---- across-ray error
    sa = _u(B, 0.0005, 0.005, gen, dev, 3)
    n = sa * torch.randn((B, T, P, 3), device=dev, generator=gen)
    n = n - d * (n * d).sum(-1, keepdim=True)
    ray = torch.nn.functional.normalize(d + n, dim=-1)

    Y = origin[:, :, None, :] + ray * r_obs[..., None]
    of = obs[..., None].float()
    return Y * of, obs, ray * of
