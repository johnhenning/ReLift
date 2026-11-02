"""Viewer exports preserve temporal coverage while sampling only reference pixels."""
import base64
import json
import re

import numpy as np

from scripts.plotly_demo import export_viewer, prepare_data, sample_grid


def test_uniform_spatial_sampling_is_bounded_and_unique():
    pixels = sample_grid(280, 504, 2500)
    assert 1 <= len(pixels) <= 2500
    assert len(np.unique(pixels, axis=0)) == len(pixels)
    assert (pixels >= 0).all() and (pixels[:, 0] < 280).all() and (pixels[:, 1] < 504).all()


def test_offline_viewer_keeps_first_last_and_all_intermediate_frames(tmp_path):
    total, height, width = 7, 3, 4
    xyz = np.arange(total * height * width * 3, dtype=np.float32).reshape(total, height, width, 3) / 10
    np.savez(tmp_path / "tracks.npz", xyz=xyz, lift=xyz.astype(np.float16),
             observed=np.ones((total, height, width), bool), images=np.full(xyz.shape, 127, np.uint8),
             source_fps=12., requested_frames=0)
    payload, _, displayed, _ = prepare_data(tmp_path / "tracks.npz", max_points=5)
    pixels = np.array(payload["pixels"])
    decoded = np.frombuffer(base64.b64decode(payload["corrected"]), dtype="<f4").reshape(total, -1, 3)
    expected = xyz[:, pixels[:, 1], pixels[:, 0]][:, :, [0, 2, 1]] * [1, 1, -1]
    np.testing.assert_array_equal(decoded, expected)
    np.testing.assert_array_equal(displayed, expected)
    metadata = export_viewer(tmp_path, max_points=5)
    assert metadata["frames"] == total and metadata["requested_frames"] == 0
    assert metadata["temporal_sampling"] == "none" and metadata["source_fps"] == 12
    page = (tmp_path / "trajectories.html").read_text()
    assert not re.search(r'<script[^>]+src=["\']https?://', page)
    encoded_payload = re.search(r'<script id="trajectory-data" type="application/json">(.*?)</script>', page, re.S)[1]
    assert json.loads(encoded_payload)["frames"] == total
    assert "__PAYLOAD__" not in page and "__VIEWER_JS__" not in page
