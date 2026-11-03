from __future__ import annotations

import numpy as np
import torch

from relift.infer import VIS_GATE, repeated_pixels, vis_remap


def test_vis_remap_sends_tau_to_the_gate_and_is_monotone() -> None:
    v = np.linspace(0, 1, 101, dtype=np.float32)
    r = vis_remap(v, 0.65)
    assert r.dtype == np.float32 and r.min() == 0 and abs(r.max() - 1) < 1e-6
    assert np.all(np.diff(r) >= 0)
    assert abs(vis_remap(np.array([0.65]), 0.65)[0] - VIS_GATE) < 1e-6
    assert (vis_remap(v[v > 0.65], 0.65) > VIS_GATE).all() and (
        vis_remap(v[v <= 0.65], 0.65) <= VIS_GATE
    ).all()


def test_repeated_pixels_follow_the_partition_rule() -> None:
    """valid pixels = 2,300: batches of 8 x 32 = 256; the last batch holds 2,300 - 8 * 256 = 252 pixels,
    i.e. 8 groups with 4 missing -> 4 repeated pixels (the first 4 of that batch)."""
    T, H, W = 2, 50, 50
    distance = torch.zeros(T, H, W)
    distance.view(T, -1)[:, :2300] = 1.0
    rep = repeated_pixels(distance, group=32, batch=8, seed=0)
    assert rep.numel() == 4
    # replay: the repeated pixels are the first `need` of the last batch of the seeded permutation
    gen = torch.Generator(device=distance.device).manual_seed(0)
    valid = torch.nonzero(distance.view(T, -1)[0] > 0)[:, 0]
    perm = valid[torch.randperm(len(valid), device=distance.device, generator=gen)]
    assert torch.equal(rep, torch.sort(perm[8 * 256 : 8 * 256 + 4]).values)
    assert (
        repeated_pixels(torch.ones(T, 16, 16), group=32, batch=8).numel() == 0
    )  # 256 = exactly one batch
