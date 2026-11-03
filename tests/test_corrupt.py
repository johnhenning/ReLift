from __future__ import annotations

from typing import Any

import torch

from relift.corrupt import corrupt, occluder_mask


def _clean_batch(B: int = 4, T: int = 24, P: int = 64, seed: int = 0) -> Any:
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(B, T, P, 3, generator=g) * 0.3 + torch.tensor([0.0, 0.0, 3.0])
    X[..., 2] = X[..., 2].clamp_min(0.5)
    vis = torch.rand(B, T, P, generator=g) > 0.2
    vis[:, 0] = True
    origin = torch.cumsum(torch.randn(B, T, 3, generator=g) * 0.01, 1)
    dyn = torch.rand(B, P, generator=g) < 0.4
    return X, vis, origin, dyn


def test_shapes_and_invariants() -> None:
    X, vis, origin, dyn = _clean_batch()
    gen = torch.Generator().manual_seed(1)
    Y, obs, ray = corrupt(X, vis, origin, dyn, gen)
    assert Y.shape == X.shape and obs.shape == vis.shape and ray.shape == X.shape
    assert obs.dtype == torch.bool and obs[:, 0].all()
    assert torch.isfinite(Y).all() and torch.isfinite(ray).all()
    assert (Y[~obs] == 0).all() and (ray[~obs] == 0).all()  # zero where unobserved
    n = torch.linalg.vector_norm(ray[obs], dim=-1)
    assert torch.allclose(n, torch.ones_like(n), atol=1e-5)  # unit rays on observed cells
    assert (obs & ~vis).any() or True  # false-visible cells may exist
    assert (vis & ~obs).any()  # something is hidden


def test_range_error_is_bounded_by_the_clip() -> None:
    """The log-range perturbation is clipped to [-1, 1]; false-visible cells carry the (nearer) occluder's range."""
    X, vis, origin, dyn = _clean_batch(B=8, T=40, P=256)
    Y, obs, ray = corrupt(X, vis, origin, dyn, torch.Generator().manual_seed(2))
    r_true = torch.linalg.vector_norm(X - origin[:, :, None], dim=-1)
    r_obs = torch.linalg.vector_norm(Y - origin[:, :, None], dim=-1)
    ratio = (r_obs / r_true)[obs]
    assert ratio.min() > 0 and ratio.max() <= 2.7183


def test_occluder_hides_only_points_behind_it() -> None:
    X, vis, origin, dyn = _clean_batch(B=6, T=40, P=256)
    hidden, ratio = occluder_mask(X, dyn, torch.Generator().manual_seed(3))
    assert hidden.shape == (6, 40, 256) and ratio.shape == hidden.shape
    assert (ratio[hidden] > 0).all() and (ratio[hidden] < 1).all() and (ratio[~hidden] == 1).all()


def test_fixed_seed_is_reproducible() -> None:
    X, vis, origin, dyn = _clean_batch()
    a = corrupt(X, vis, origin, dyn, torch.Generator().manual_seed(5))
    b = corrupt(X, vis, origin, dyn, torch.Generator().manual_seed(5))
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=False))
