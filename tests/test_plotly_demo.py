"""Viewer exports preserve temporal coverage while sampling only reference pixels."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import numpy as np

from relift.storage import TrajectoryStore
from scripts.plotly_demo import export_viewer, prepare_data, sample_grid


def test_uniform_spatial_sampling_is_bounded_and_unique() -> None:
    pixels = sample_grid(280, 504, 2500)
    assert 1 <= len(pixels) <= 2500
    assert len(np.unique(pixels, axis=0)) == len(pixels)
    assert (pixels >= 0).all() and (pixels[:, 0] < 280).all() and (pixels[:, 1] < 504).all()


def test_offline_viewer_keeps_first_last_and_all_intermediate_frames(tmp_path: Path) -> None:
    total, height, width = 7, 3, 4
    xyz = (
        np.arange(total * height * width * 3, dtype=np.float32).reshape(total, height, width, 3)
        / 10
    )
    np.savez(
        tmp_path / "tracks.npz",
        xyz=xyz,
        lift=xyz.astype(np.float16),
        observed=np.ones((total, height, width), bool),
        images=np.full(xyz.shape, 127, np.uint8),
        source_fps=12.0,
        requested_frames=0,
    )
    payload, _, displayed, _ = prepare_data(tmp_path / "tracks.npz", max_points=5)
    pixels = np.array(payload["pixels"])
    decoded = np.frombuffer(base64.b64decode(payload["corrected"]), dtype="<f4").reshape(
        total, -1, 3
    )
    expected = xyz[:, pixels[:, 1], pixels[:, 0]][:, :, [0, 2, 1]] * [1, 1, -1]
    np.testing.assert_array_equal(decoded, expected)
    np.testing.assert_array_equal(displayed, expected)
    metadata = export_viewer(tmp_path, max_points=5)
    assert metadata["frames"] == total and metadata["requested_frames"] == 0
    assert metadata["temporal_sampling"] == "none" and metadata["source_fps"] == 12
    page = (tmp_path / "trajectories.html").read_text()
    assert not re.search(r'<script[^>]+src=["\']https?://', page)
    encoded_payload = re.search(
        r'<script id="trajectory-data" type="application/json">(.*?)</script>', page, re.S
    )
    assert encoded_payload is not None
    assert json.loads(encoded_payload[1])["frames"] == total
    assert "__PAYLOAD__" not in page and "__VIEWER_JS__" not in page


def test_mapped_viewer_samples_original_pixel_ids_without_a_dense_npz(tmp_path: Path) -> None:
    total, height, width = 7, 3, 4
    xyz = (
        np.arange(total * height * width * 3, dtype=np.float32).reshape(total, height * width, 3)
        / 10
    )
    order = np.arange(height * width)[::-1]
    store = TrajectoryStore.create(
        tmp_path,
        (total, height, width, 3),
        dict(
            source_fps=12.0,
            requested_frames=0,
            process_resolution=504,
            input_sha256="fixture-input",
        ),
        order,
        dict(
            images=np.full((total, height, width, 3), 127, np.uint8),
            poses=np.tile(np.eye(4), (total, 1, 1)),
            intrinsics=np.tile(np.eye(3), (total, 1, 1)),
        ),
    )
    for start in range(0, len(order), 5):
        selected = order[start : start + 5]
        store.write_shard(
            start,
            xyz=xyz[:, selected].transpose(1, 0, 2),
            lift=xyz[:, selected].transpose(1, 0, 2).astype(np.float16),
            observed=np.ones((len(selected), total), bool),
        )
    store.finish()
    payload, _, displayed, _ = prepare_data(tmp_path, max_points=5)
    pixels = np.array(payload["pixels"])
    ids = pixels[:, 1] * width + pixels[:, 0]
    expected = xyz[:, ids][:, :, [0, 2, 1]] * [1, 1, -1]
    np.testing.assert_array_equal(displayed, expected)
    metadata = export_viewer(tmp_path, max_points=5)
    assert metadata["source"] == "tracks.store/manifest.json"
    assert (
        metadata["storage"] == "mapped-track-shards" and metadata["input_sha256"] == "fixture-input"
    )
    assert not (tmp_path / "tracks.npz").exists()
    assert metadata["frames"] == total and metadata["displayed_trajectories"] <= 5
