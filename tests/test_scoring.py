from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from relift.eval.scoring import apd, clip_seed, score_clip

rc4 = pytest.importorskip("relift.vendor").rc4_eval if True else None


def test_clip_seed_is_the_published_rule() -> None:
    assert clip_seed("boxes_28", "sim3") == sum(map(ord, "boxes_28sim3")) % 2**31
    assert clip_seed("a", "sim3") != clip_seed("a", "global")


def test_apd_counts_fractions_within_thresholds() -> None:
    d = np.array([0.05, 0.2, 0.4, 0.8, 2.0])
    assert apd(d, [0.1, 0.3, 0.5, 1.0]) == pytest.approx(np.mean([1 / 5, 2 / 5, 3 / 5, 4 / 5]))
    assert np.isnan(apd(np.array([]), [0.1]))


def _toy_clip(seed: int = 0) -> Any:
    rng = np.random.default_rng(seed)
    T, N = 10, 40
    gt = rng.normal(size=(T, N, 3)) + np.array([0, 0, 4.0])
    gt[:, : N // 2] = gt[:1, : N // 2]  # half the points are static
    pred = 1.3 * gt + 0.05 * rng.normal(size=gt.shape)
    occ = rng.random((T, N)) < 0.2
    return gt, pred, occ, np.array([300.0, 300.0, 128.0, 128.0])


def test_score_clip_is_deterministic_and_well_formed() -> None:
    pytest.importorskip("torch")
    gt, pred, occ, intr = _toy_clip()
    a = score_clip("toy", gt, pred, occ, intr)
    b = score_clip("toy", gt, pred, occ, intr)
    assert a == b
    assert a["n"] == 40 and a["T"] == 10 and a["dyn"] == 20 and a["occ_cells"] == int(occ.sum())
    for mode in ("sim3", "global"):
        r = a[mode]
        assert set(r) == {"apd", "epe", "occ", "vis", "dyn"}
        assert 0 <= r["apd"] <= 1 and r["epe"] >= 0
    assert a["sim3"]["apd"] > 0.9  # a scaled copy is recovered by Sim(3)


def test_nan_predictions_score_as_far_away() -> None:
    pytest.importorskip("torch")
    gt, pred, occ, intr = _toy_clip()
    pred[:, :5] = np.nan
    r = score_clip("toy", gt, pred, occ, intr)
    assert np.isfinite(r["sim3"]["apd"]) and r["sim3"]["apd"] < 0.9
