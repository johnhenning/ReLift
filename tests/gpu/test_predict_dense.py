"""Dense inference on one prepared window reproduces a recorded prediction.

Needs CUDA, RELIFT_CHECKPOINT and RELIFT_TEST_WINDOW (a directory with ``alltracker_tracks.npz`` and
``da3_depth_data.npz``).  With RELIFT_TEST_REFERENCE (a ``.npz`` holding ``tracks`` (T,H,W,3), recorded on the GPU
type named in its ``gpu`` field) the prediction is compared: max abs difference <= 2 cm on the same GPU type.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from relift.checkpoint import load
from relift.infer import observations, predict_dense

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_predict_dense_on_a_window() -> None:
    ckpt, win = os.environ.get("RELIFT_CHECKPOINT"), os.environ.get("RELIFT_TEST_WINDOW")
    if not (ckpt and win):
        pytest.skip("set RELIFT_CHECKPOINT and RELIFT_TEST_WINDOW")
    model, _ = load(ckpt)
    o, d, r, vis = observations(
        win, 84, tracks_file="alltracker_tracks.npz", depth_file="da3_depth_data.npz"
    )
    p = predict_dense(model, o, d, r, vis)
    assert p.shape == (o.shape[0], d.shape[1], d.shape[2], 3) and torch.isfinite(p).all()
    lift = (o[:, None, None, :] + d * r[..., None]).float()
    invalid = r[0] <= 0
    if invalid.any():  # pixels without a valid reference range keep the lift
        assert torch.equal(p[:, invalid], torch.nan_to_num(lift[:, invalid]))
    ref = os.environ.get("RELIFT_TEST_REFERENCE")
    if ref:
        z = np.load(ref)
        same_gpu = str(z["gpu"]) == torch.cuda.get_device_name(0)
        diff = float(np.abs(p.cpu().numpy() - z["tracks"]).max())
        assert diff <= (0.02 if same_gpu else 0.5), (
            f"max abs diff {diff} m (same GPU type: {same_gpu})"
        )
