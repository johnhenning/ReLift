from __future__ import annotations

from typing import Any

import numpy as np
import torch

from relift.features import interp_init, normalise


def _reference(Y: Any, obs: Any) -> Any:
    """Plain NumPy interpolation, per track, held at the ends; gap = frames to the nearest observed frame."""
    B, T, P, _ = Y.shape
    x, gap = np.zeros_like(Y), np.zeros((B, T, P), np.int64)
    for b in range(B):
        for p in range(P):
            o = np.flatnonzero(obs[b, :, p])
            for t in range(T):
                prev, nxt = o[o <= t].max(), (o[o >= t].min() if (o >= t).any() else None)
                if nxt is None:
                    x[b, t, p] = Y[b, prev, p]
                    gap[b, t, p] = t - prev
                else:
                    w = 0.0 if nxt == prev else (t - prev) / (nxt - prev)
                    x[b, t, p] = (1 - w) * Y[b, prev, p] + w * Y[b, nxt, p]
                    gap[b, t, p] = min(t - prev, nxt - t)
    return x, gap


def test_interp_init_matches_reference() -> None:
    rng = np.random.default_rng(0)
    B, T, P = 3, 20, 7
    Y = rng.normal(size=(B, T, P, 3)).astype(np.float32)
    obs = rng.random((B, T, P)) > 0.6
    obs[:, 0] = True
    Y = Y * obs[..., None]
    x, gap = interp_init(torch.from_numpy(Y), torch.from_numpy(obs))
    xr, gr = _reference(Y, obs)
    assert np.allclose(x.numpy(), xr, atol=1e-6)
    assert np.array_equal(gap.numpy(), gr)
    assert (x[obs] == torch.from_numpy(Y)[obs]).all()


def test_normalise_uses_the_reference_median_depth() -> None:
    torch.manual_seed(0)
    Y = torch.rand(2, 5, 11, 3) + 0.5
    origin = torch.rand(2, 5, 3)
    Yn, on, s = normalise(Y, None, origin)
    assert torch.allclose(s, Y[:, 0, :, 2].median(dim=1).values)
    assert torch.allclose(Yn * s.view(-1, 1, 1, 1), Y) and torch.allclose(
        on * s.view(-1, 1, 1), origin
    )
