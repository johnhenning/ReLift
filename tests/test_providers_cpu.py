"""Provider pieces that need no weights or GPU."""

from __future__ import annotations

import torch

from relift.providers import depth_file, tracks_file
from relift.providers.cowtracker import _sdpa, _to_native, cow_size


def test_cow_size_rule_matches_the_published_sizes() -> None:
    assert cow_size(280, 504, 240000) == (336, 672)
    assert cow_size(504, 504, 240000) == (448, 448)
    assert cow_size(280, 504, 420000) == (448, 896)
    assert cow_size(504, 504, 420000) == (560, 560)
    assert cow_size(280, 504, 240000, fixed=(336, 560)) == (336, 560)


def test_sdpa_matches_explicit_attention() -> None:
    torch.manual_seed(0)
    B, M, H, K = 2, 7, 3, 8
    q, k, v = (torch.randn(B, M, H, K) for _ in range(3))
    scale = K**-0.5
    ref = torch.softmax(torch.einsum("bmhk,bnhk->bhmn", q, k) * scale, -1)
    ref = torch.einsum("bhmn,bnhk->bmhk", ref, v)
    assert torch.allclose(_sdpa(q, k, v, None, 0.0, scale), ref, atol=1e-5)


def test_to_native_keeps_frame_zero_on_the_grid_and_identity_at_native_size() -> None:
    T, Hr, Wr, Hn, Wn = 3, 6, 8, 12, 16
    gy, gx = torch.meshgrid(
        torch.arange(Hr, dtype=torch.float32), torch.arange(Wr, dtype=torch.float32), indexing="ij"
    )
    tracks = torch.stack([gx, gy], -1)[None].repeat(T, 1, 1, 1) + torch.tensor(
        [1.0, 0.5]
    )  # a constant flow
    vis, conf = torch.full((T, Hr, Wr), 0.7), torch.full((T, Hr, Wr), 0.2)
    tr, v, c = _to_native(tracks, vis, conf, Hn, Wn)
    ny, nx = torch.meshgrid(
        torch.arange(Hn, dtype=torch.float32), torch.arange(Wn, dtype=torch.float32), indexing="ij"
    )
    assert torch.equal(tr[0], torch.stack([nx, ny], -1))
    sx, sy = (Wn - 1) / (Wr - 1), (Hn - 1) / (Hr - 1)
    assert torch.allclose(tr[1], torch.stack([nx + 1.0 * sx, ny + 0.5 * sy], -1), atol=1e-4)
    assert torch.allclose(v, torch.full((T, Hn, Wn), 0.7)) and torch.allclose(
        c, torch.full((T, Hn, Wn), 0.2)
    )
    same = _to_native(tracks, vis, conf, Hr, Wr)
    assert torch.equal(same[0], tracks)


def test_file_names() -> None:
    assert (
        depth_file("da3") == "da3_depth_data.npz"
        and tracks_file("cowtracker") == "cowtracker_tracks.npz"
    )
