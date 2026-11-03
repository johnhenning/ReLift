from __future__ import annotations

import torch

from relift.features import S, correct, features_clean, interp_init
from relift.model import Corrector


def test_parameter_count_of_the_released_configuration() -> None:
    m = Corrector(dim=384, depth=8, heads=6, in_feats=14, chunk=4)
    assert sum(p.numel() for p in m.parameters()) == 19_145_100


def test_untrained_model_returns_the_initialised_trajectory() -> None:
    """The head is zero-initialised, so the correction is zero and `correct` returns the interpolated lift."""
    torch.manual_seed(0)
    m = Corrector(dim=64, depth=2, heads=2).eval()
    B, T, P = 2, 8, 16
    Y = torch.randn(B, T, P, 3)
    obs = torch.rand(B, T, P) > 0.3
    obs[:, 0] = True
    Y = Y * obs[..., None]
    ray = torch.nn.functional.normalize(torch.randn(B, T, P, 3), dim=-1) * obs[..., None]
    origin = torch.randn(B, T, 3)
    x_start, _ = interp_init(Y, obs)
    out = correct(m, Y, obs, ray, origin)
    assert torch.equal(out, x_start)


def test_permutation_equivariance_over_tracks() -> None:
    torch.manual_seed(0)
    m = Corrector(dim=64, depth=2, heads=2).eval()
    with torch.no_grad():
        m.head.weight.normal_()
        m.head.bias.normal_()
    B, T, P = 1, 8, 12
    f, a = torch.randn(B, T, P, 14), torch.randn(B, P, 3)
    perm = torch.randperm(P)
    with torch.no_grad():
        u = m(f, a)
        up = m(f[:, :, perm], a[:, perm])
    assert torch.allclose(u[:, :, perm], up, atol=1e-5)


def test_feature_layout() -> None:
    """14 channels: initialised trajectory, observed lift, observed flag, gap, observed ray, camera centre."""
    B, T, P = 1, 4, 3
    x_start = torch.arange(B * T * P * 3, dtype=torch.float32).view(B, T, P, 3)
    Y = x_start + 1
    obs = torch.tensor([[[True, True, False]] * T])
    gap = torch.tensor([[[0, 0, 50]] * T])
    ray = torch.ones(B, T, P, 3)
    origin = torch.arange(T, dtype=torch.float32)[None, :, None].expand(B, T, 3).clone()
    f, a = features_clean(x_start, Y, obs, gap, ray, origin)
    assert f.shape == (B, T, P, 14) and torch.equal(a, x_start[:, 0])
    assert torch.equal(f[..., :3], (x_start - x_start[:, :1]) * S)
    assert (
        torch.equal(f[:, :, :2, 3:6], (Y - x_start[:, :1])[:, :, :2] * S)
        and (f[:, :, 2, 3:6] == 0).all()
    )
    assert torch.equal(f[..., 6], obs.float())
    assert torch.equal(f[:, :, 2, 7], torch.ones(B, T)) and (f[:, :, :2, 7] == 0).all()
    assert torch.equal(f[:, :, :2, 8:11], torch.ones(B, T, 2, 3)) and (f[:, :, 2, 8:11] == 0).all()
    assert torch.equal(f[:, :, 0, 11:], (origin - origin[:, :1]) * S)
