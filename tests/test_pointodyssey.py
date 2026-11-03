from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from relift.data.pointodyssey import EXCLUDE, split, windows_of


def _scene(F: int = 200, N: int = 3000, seed: int = 0, moving: int = 600) -> Any:
    """A synthetic anno.npz: static points plus `moving` points drifting 1 m, a camera translating along x."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, (N, 3)) + np.array([0, 0, 6.0])
    traj = np.repeat(X[None], F, 0)
    traj[:, :moving] += np.linspace(0, 1, F)[:, None, None] * np.array([1.0, 0, 0])
    E = np.tile(np.eye(4, dtype=np.float32), (F, 1, 1))
    E[:, 0, 3] = -0.001 * np.arange(F)
    K = np.tile(np.array([[400, 0, 320], [0, 400, 240], [0, 0, 1]], np.float32), (F, 1, 1))
    xc = np.einsum("tij,tnj->tni", E[:, :3, :3], traj) + E[:, None, :3, 3]
    uv = np.stack(
        [400 * xc[..., 0] / xc[..., 2] + 320, 400 * xc[..., 1] / xc[..., 2] + 240], -1
    ).astype(np.float32)
    return dict(
        trajs_3d=traj.astype(np.float32),
        trajs_2d=uv,
        extrinsics=E,
        intrinsics=K,
        visibs=np.ones((F, N), bool),
        valids=np.ones((F, N), bool),
    )


def test_windows_layout_and_dynamic_cap() -> None:
    z = _scene()
    o = windows_of(z, np.random.default_rng(0))
    # stride 1: starts 0, 42, 84 (span 84 fits up to start 116); stride 2: span 167, starts 0 (84 does not fit)
    assert o["step"] == [1, 1, 1, 2] and o["start"] == [0, 42, 84, 0]
    for X, vis, origin, dyn in zip(o["X"], o["vis"], o["origin"], o["dyn"], strict=False):
        assert (
            X.shape == (84, 2048, 3)
            and vis.shape == (84, 2048)
            and origin.shape == (84, 3)
            and dyn.shape == (2048,)
        )
        assert dyn.sum() == 600  # fewer than half moving: all of them kept
        assert np.allclose(origin[0], 0) and X.dtype == np.float32
    # static points stay put in the reference camera coordinate system
    X, dyn = o["X"][0], o["dyn"][0]
    assert np.abs(X[:, ~dyn] - X[:1, ~dyn]).max() < 1e-3


def test_moving_points_capped_at_half_when_static_points_suffice() -> None:
    o = windows_of(_scene(moving=1500), np.random.default_rng(0))  # 1,500 moving, 1,500 static
    assert all(d.sum() == 1024 for d in o["dyn"])
    o = windows_of(
        _scene(moving=2500), np.random.default_rng(0)
    )  # only 500 static: the rest is moving
    assert all(d.sum() == 2048 for d in o["dyn"])


def test_too_few_visible_points_skips_the_window() -> None:
    z = _scene(N=1500)
    z["visibs"][:, :600] = False
    assert windows_of(z, np.random.default_rng(0))["X"] == []


def test_split_rule(tmp_path: Path) -> None:
    names = [f"scene{i:03d}" for i in range(30)] + ["cab_e_ego2"]
    for n in names:
        (tmp_path / f"{n}.npz").write_bytes(b"")
    train, val = split(tmp_path)
    kept = sorted(n for n in names if n not in EXCLUDE)
    assert (
        val == kept[6::13]
        and len(val) == 2
        and not set(train) & set(val)
        and "cab_e_ego2" not in train + val
    )
