from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from relift.features import S, interp_init
from relift.model import Corrector
from relift.train import (
    LENGTHS,
    P_TRAIN,
    P_VAL,
    Windows,
    loss_fn,
    lr_at,
    make_batch,
    smooth,
    validate,
)


def _cache(
    tmp_path: Path, scenes: Any = ("a", "b"), n: int = 3, T: int = 84, P: int = 2048, seed: int = 0
) -> Any:
    rng = np.random.default_rng(seed)
    for s in scenes:
        X = rng.uniform(-1, 1, (n, T, P, 3)).astype(np.float32) + np.array([0, 0, 5], np.float32)
        X[:, :, :300] += np.linspace(0, 1, T)[None, :, None, None] * np.array(
            [1.0, 0, 0], np.float32
        )
        origin = np.zeros((n, T, 3), np.float32)
        origin[:, :, 0] = 0.001 * np.arange(T)
        dyn = np.zeros((n, P), bool)
        dyn[:, :300] = True
        np.savez(
            tmp_path / f"{s}.npz",
            X=X,
            vis=rng.random((n, T, P)) > 0.2,
            origin=origin,
            dyn=dyn,
            step=np.ones(n, np.int8),
            start=np.zeros(n, np.int32),
        )
    return str(tmp_path)


def test_windows_and_batch(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    train, val = Windows(cache, ["a", "b"]), Windows(cache, ["a"], fixed=True, per_scene=2)
    assert len(train) == 6 and len(val) == 2
    X, vis, origin, dyn = train[0]
    assert (
        X.shape == (84, P_TRAIN, 3)
        and vis.shape == (84, P_TRAIN)
        and origin.shape == (84, 3)
        and dyn.shape == (P_TRAIN,)
    )
    Xv = val[0][0]
    assert Xv.shape == (84, P_VAL, 3) and torch.equal(
        Xv, val[0][0]
    )  # fixed tracks: the same every time
    batch = torch.utils.data.default_collate([train[0], train[1]])
    b = make_batch(batch, "cpu", torch.Generator().manual_seed(0), length=48, augment=False)
    assert (
        b["X"].shape == (2, 48, P_TRAIN, 3)
        and b["obs"][:, 0].all()
        and b["Y"].shape == b["X"].shape
    )
    assert torch.equal(
        b["Y"][~b["obs"]], torch.zeros_like(b["Y"][~b["obs"]])
    )  # unobserved cells are zero
    assert abs(float(b["Y"][:, 0, :, 2].median()) - 1.0) < 1e-5 or torch.allclose(
        b["Y"][0, 0, :, 2].median(), torch.tensor(1.0), atol=1e-5
    )
    # the same generator gives the same batch; a mirrored batch flips x
    b2 = make_batch(batch, "cpu", torch.Generator().manual_seed(0), length=48, augment=False)
    assert torch.equal(b["Y"], b2["Y"]) and torch.equal(b["obs"], b2["obs"])


def test_untrained_loss_is_the_interpolation_loss(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    train = Windows(cache, ["a"])
    batch = torch.utils.data.default_collate([train[0], train[1]])
    b = make_batch(batch, "cpu", torch.Generator().manual_seed(1), length=LENGTHS[0], augment=False)
    model = Corrector(dim=64, depth=1, heads=1)  # zero-initialised head: u = 0
    loss, parts = loss_fn(model, b)
    x_start, _ = interp_init(b["Y"], b["obs"])
    rho = torch.linalg.vector_norm(x_start[:, :1], dim=-1, keepdim=True).clamp_min(0.3)
    e = -S * (b["X"] - x_start) / rho
    w = (1.0 + 3.0 * b["dyn"].float())[:, None, :] * (1.0 + (~b["obs"]).float())
    pos = ((e**2).sum(-1) * w).sum((1, 2)) / w.sum((1, 2))
    de = e[:, 1:] - e[:, :-1]
    vel = ((de**2).sum(-1) * w[:, 1:]).sum((1, 2)) / w[:, 1:].sum((1, 2))
    assert (
        torch.allclose(parts["pos"], pos)
        and torch.allclose(parts["vel"], vel)
        and abs(float(loss) - float((pos + 4 * vel).mean())) < 1e-5
    )
    loss.backward()  # the head's gradient is defined
    assert model.head.weight.grad is not None
    assert torch.isfinite(model.head.weight.grad).all()


def test_schedule_smooth_and_validate(tmp_path: Path) -> None:
    assert abs(lr_at(0, 0.0, 3e-4) - 3e-7) < 1e-12 and abs(lr_at(999, 0.0, 3e-4) - 3e-4) < 1e-12
    assert (
        abs(lr_at(5000, 1.0, 3e-4) - 0.03 * 3e-4) < 1e-12
        and abs(lr_at(5000, 0.5, 3e-4) - 3e-4 * (0.03 + 0.97 * 0.5)) < 1e-12
    )
    x = torch.randn(1, 20, 3, 3)
    assert smooth(x).shape == x.shape and torch.allclose(
        smooth(torch.ones(1, 20, 2, 3)), torch.ones(1, 20, 2, 3), atol=1e-6
    )
    cache = _cache(tmp_path)
    val = Windows(cache, ["a"], fixed=True, per_scene=2)
    loader = torch.utils.data.DataLoader(val, batch_size=2)
    m = validate(Corrector(dim=64, depth=1, heads=1), loader, "cpu")
    for name in ("lift", "interp", "smooth", "model", "model_wrongcond"):
        for k in ("obs", "hid", "obs_dyn", "hid_dyn", "vel_hid_dyn"):
            assert np.isfinite(m[f"{name}/{k}"])
    assert (
        m["lift/obs"] == m["interp/obs"] and m["model/hid"] == m["interp/hid"]
    )  # untrained model = interpolation
    assert m["model_wrongcond/obs"] > 5 * m["model/obs"]  # another clip's observations: far worse
    json.dumps(m)
