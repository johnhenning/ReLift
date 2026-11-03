"""Batch resume rejects truncated exports and modified trajectory files."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from relift.records.reporting import GalleryState, GalleryVideo, VideoSource
from relift.storage import TrajectoryStore
from scripts.download_demo_inputs import sha256
from scripts.generate_all_views import complete_result, infer_full_resolution, write_index
from scripts.plotly_demo import export_viewer


def test_completed_view_requires_full_frame_coverage_and_matching_bytes(tmp_path: Path) -> None:
    (tmp_path / "tracks.npz").write_bytes(b"test trajectory archive")
    (tmp_path / "trajectories.html").write_text("<div>export</div>")
    metadata = dict(
        frames=206,
        source_fps=12.0,
        requested_frames=0,
        finite=True,
        temporal_sampling="none",
        source_sha256=sha256(tmp_path / "tracks.npz"),
        html_bytes=(tmp_path / "trajectories.html").stat().st_size,
    )
    (tmp_path / "plotly.json").write_text(json.dumps(metadata))
    assert complete_result(tmp_path, dict(frames=206, fps=12.0)) == metadata
    assert complete_result(tmp_path, dict(frames=294, fps=12.0)) is None
    (tmp_path / "tracks.npz").write_bytes(b"changed trajectory archive")
    assert complete_result(tmp_path, dict(frames=206, fps=12.0)) is None


def test_gallery_links_only_completed_outputs_and_escapes_errors(tmp_path: Path) -> None:
    root = tmp_path / "running"
    root.mkdir()
    (root / "trajectories.html").write_text("<div>export</div>")
    state: GalleryState = dict(
        videos={
            "running": dict(
                status="complete",
                source=dict(frames=206, duration_seconds=206 / 12),
                result=dict(dense_trajectories=141120),
            ),
            "forest": dict(status="failed", error="<unavailable>"),
        }
    )
    write_index(tmp_path, state)
    page = (tmp_path / "index.html").read_text()
    assert 'href="running/trajectories.html"' in page
    assert 'href="forest/trajectories.html"' not in page
    assert "&lt;unavailable&gt;" in page and "<unavailable>" not in page
    assert (
        json.loads((tmp_path / "batch.json").read_text())["videos"]["running"]["status"]
        == "complete"
    )


def test_mapped_completion_binds_source_resolution_and_shard_integrity(tmp_path: Path) -> None:
    T, H, W = 3, 2, 2
    store = TrajectoryStore.create(
        tmp_path,
        (T, H, W, 3),
        dict(source_fps=12.0, requested_frames=0, process_resolution=504, input_sha256="source-a"),
        np.arange(H * W),
        dict(
            images=np.full((T, H, W, 3), 127, np.uint8),
            poses=np.tile(np.eye(4), (T, 1, 1)),
            intrinsics=np.tile(np.eye(3), (T, 1, 1)),
        ),
    )
    xyz = np.arange(H * W * T * 3, dtype=np.float32).reshape(H * W, T, 3) / 10
    store.write_shard(0, xyz=xyz, lift=xyz.astype(np.float16), observed=np.ones((H * W, T), bool))
    store.finish()
    exported = export_viewer(tmp_path)
    source: VideoSource = dict(frames=T, fps=12.0, sha256="source-a")
    assert complete_result(tmp_path, source, 504) == exported
    assert complete_result(tmp_path, source, 392) is None
    changed_source = source.copy()
    changed_source["sha256"] = "source-b"
    assert complete_result(tmp_path, changed_source, 504) is None
    assert store.manifest is not None
    file = store.root / store.manifest["shards"][0]["arrays"]["xyz"]["file"]
    with file.open("r+b") as stream:
        stream.seek(-1, 2)
        value = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([value[0] ^ 1]))
    assert complete_result(tmp_path, source, 504) is None


def test_offload_oom_does_not_silently_lower_spatial_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fail(script: Any, arguments: Any, log: Path) -> None:
        calls.append(arguments)
        log.write_text("CUDA out of memory\n")
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr("scripts.generate_all_views.run_logged", fail)
    args = SimpleNamespace(
        offload="ram", ram_budget_gb=18, vram_budget_gb=20, scratch_budget_gb=64, scratch_dir=None
    )
    record: GalleryVideo = {}
    with pytest.raises(RuntimeError, match="out of memory"):
        infer_full_resolution("source.mp4", tmp_path, record, args)
    assert len(calls) == 1
    assert calls[0][calls[0].index("--process-res") + 1] == 504
    assert record["inference_attempts"][0]["status"] == "failed"
