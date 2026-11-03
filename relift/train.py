"""Training of the corrector (paper App. A.3): the regression objective on the simulated lift of PointOdyssey windows.

The pieces, in the order a step uses them: ``Windows`` (a window cache -> 512 random tracks per window, 1,024 fixed
tracks for validation), ``make_batch`` (optional mirror, ``corrupt``, ``normalise``), ``loss_fn`` (the position +
4 x velocity loss on the relative error, moving and hidden cells weighted up), ``lr_at`` (1,000 warm-up steps, then
a cosine to 3 % indexed by elapsed training time or by step), ``validate`` (the lift, its interpolation, a smoothed
interpolation, the model, and the model on another clip's observations, as % of depth).

Everything here reproduces the research code of the released run; the only additions are the optional seed and the
step-indexed schedule.  Randomness that the research run left unseeded (track sampling, window length, mirroring,
the simulated lift) stays unseeded unless ``seed`` is given.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .corrupt import corrupt
from .features import S, correct, features_clean, interp_init, normalise
from .records.model import LossMetrics, TrainingBatch

LENGTHS = (
    48,
    68,
    84,
)  # a training window is cut to one of these lengths (validation cycles through them)
P_TRAIN = 512  # tracks per window in training
P_VAL = 1024  # tracks per window in validation (every second of the 2,048)
VAL_PER_SCENE = 8  # validation windows per scene, evenly spaced


class Windows(torch.utils.data.Dataset):
    """Windows of the named scenes.  ``fixed``: the same 1,024 tracks every time (validation); else 512 random ones."""

    def __init__(
        self, cache: str | Path, names: list[str], fixed: bool = False, per_scene: int | None = None
    ) -> None:
        self.items, self.fixed = [], fixed
        for n in names:
            z = np.load(f"{cache}/{n}.npz")
            X, vis, origin, dyn = z["X"], z["vis"], z["origin"], z["dyn"]
            idx = (
                range(len(X))
                if per_scene is None
                else np.linspace(0, len(X) - 1, min(per_scene, len(X))).astype(int)
            )
            self.items += [(X[i], vis[i], origin[i], dyn[i]) for i in idx]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        X, vis, origin, dyn = self.items[index]
        p = np.arange(P_VAL) * 2 if self.fixed else np.random.permutation(X.shape[1])[:P_TRAIN]
        return (
            torch.from_numpy(X[:, p]),
            torch.from_numpy(vis[:, p]),
            torch.from_numpy(origin),
            torch.from_numpy(dyn[p]),
        )


def make_batch(
    batch: tuple[torch.Tensor, ...] | list[torch.Tensor],
    dev: str | torch.device,
    gen: torch.Generator | None = None,
    length: int | None = None,
    augment: bool = True,
) -> TrainingBatch:
    """-> dict(X clean tracks, Y lifted observation, obs, ray, origin, dyn, vis), normalised per clip."""
    X, vis, origin, dyn = (b.to(dev, non_blocking=True) for b in batch)
    if length is not None:
        X, vis, origin = X[:, :length], vis[:, :length], origin[:, :length]
    if augment and torch.rand((), device=dev, generator=gen) < 0.5:  # mirror the scene
        X, origin = X * X.new_tensor([-1.0, 1.0, 1.0]), origin * origin.new_tensor([-1.0, 1.0, 1.0])
    Y, obs, ray = corrupt(X, vis, origin, dyn, gen)
    Y, origin, _, X = normalise(Y, obs, origin, X)
    Y = Y * obs[..., None].float()
    assert torch.isfinite(Y).all() and torch.isfinite(X).all(), "non-finite training input"
    return dict(X=X, Y=Y, obs=obs, ray=ray, origin=origin, dyn=dyn, vis=vis)


def loss_fn(model: torch.nn.Module, b: TrainingBatch) -> tuple[torch.Tensor, LossMetrics]:
    """Position + 4 x velocity error of the correction, relative to the track's range, weighted 4x on moving tracks
    and 2x on hidden cells.  -> (loss, dict(pos, vel) per sample)."""
    x_start, gap = interp_init(b["Y"], b["obs"])
    f, a = features_clean(x_start, b["Y"], b["obs"], gap, b["ray"], b["origin"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        u = model(f, a)
    rho = torch.linalg.vector_norm(x_start[:, :1], dim=-1, keepdim=True).clamp_min(
        0.3
    )  # the track's own range
    e = (u.float() - S * (b["X"] - x_start)) / rho
    w = (1.0 + 3.0 * b["dyn"].float())[:, None, :] * (1.0 + (~b["obs"]).float())
    pos = ((e**2).sum(-1) * w).sum((1, 2)) / w.sum((1, 2))
    de = e[:, 1:] - e[:, :-1]
    vel = ((de**2).sum(-1) * w[:, 1:]).sum((1, 2)) / w[:, 1:].sum((1, 2))
    return (pos + 4.0 * vel).mean(), dict(pos=pos.detach(), vel=vel.detach())


def lr_at(step: int, progress: float, lr: float, warmup: int = 1000, floor: float = 0.03) -> float:
    """Linear warm-up over ``warmup`` steps, then a cosine from ``lr`` to ``floor * lr`` over ``progress`` in [0, 1]."""
    return (
        lr
        * min(1.0, (step + 1) / warmup)
        * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))
    )


def smooth(x: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    """Gaussian smoothing along time (B, T, P, 3), replicate padding; a classical baseline for validation."""
    k = torch.arange(-6, 7, device=x.device).float()
    k = torch.exp(-0.5 * (k / sigma) ** 2)
    k = k / k.sum()
    B, T, P, _ = x.shape
    y = x.permute(0, 2, 3, 1).reshape(B * P * 3, 1, T)
    y = torch.nn.functional.conv1d(
        torch.nn.functional.pad(y, (6, 6), mode="replicate"), k.view(1, 1, -1)
    )
    return y.view(B, P, 3, T).permute(0, 3, 1, 2)


@torch.no_grad()
def validate(model: torch.nn.Module, loader: Any, dev: str | torch.device) -> dict[str, float]:
    """Mean error as % of depth on observed / hidden / observed moving / hidden moving cells, and the velocity error on
    hidden moving cells, for the lift, its interpolation, the smoothed interpolation, the model, and the model given
    another clip's observations (must be far worse, or the input is ignored).  Deterministic: seeded per batch."""
    model.eval()
    acc = {}

    def add(name: str, x: torch.Tensor, b: TrainingBatch) -> None:
        d = 100.0 * torch.linalg.vector_norm(x - b["X"], dim=-1)
        dynm = b["dyn"][:, None, :].expand_as(b["obs"])
        for cell, m in (
            ("obs", b["obs"]),
            ("hid", ~b["obs"]),
            ("obs_dyn", b["obs"] & dynm),
            ("hid_dyn", ~b["obs"] & dynm),
        ):
            s, n = acc.setdefault(f"{name}/{cell}", [0.0, 0.0])
            acc[f"{name}/{cell}"] = [s + float(d[m].sum()), n + float(m.sum())]
        v = 100.0 * torch.linalg.vector_norm(
            (x[:, 1:] - x[:, :-1]) - (b["X"][:, 1:] - b["X"][:, :-1]), dim=-1
        )
        mh = (~b["obs"] & dynm)[:, 1:]
        s, n = acc.setdefault(f"{name}/vel_hid_dyn", [0.0, 0.0])
        acc[f"{name}/vel_hid_dyn"] = [s + float(v[mh].sum()), n + float(mh.sum())]

    for i, batch in enumerate(loader):
        gen = torch.Generator(device=dev).manual_seed(1234 + i)
        b = make_batch(batch, dev, gen, length=LENGTHS[i % len(LENGTHS)], augment=False)
        x_start, _ = interp_init(b["Y"], b["obs"])
        add("lift", torch.where(b["obs"][..., None], b["Y"], x_start), b)
        add("interp", x_start, b)
        add("smooth", smooth(x_start), b)
        add("model", correct(model, b["Y"], b["obs"], b["ray"], b["origin"]), b)
        r = lambda t: torch.roll(t, 1, 0)  # noqa: E731
        add(
            "model_wrongcond",
            correct(model, r(b["Y"]), r(b["obs"]), r(b["ray"]), r(b["origin"])),
            b,
        )
    model.train()
    return {k: s / max(n, 1.0) for k, (s, n) in acc.items()}


def ema_update(ema: torch.nn.Module, model: torch.nn.Module, step: int) -> None:
    d = min(0.999, (1 + step) / (10 + step))
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters(), strict=False):
            pe.lerp_(pm, 1 - d)
