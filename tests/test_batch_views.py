"""Batch resume rejects truncated exports and modified trajectory files."""
import json

from scripts.download_demo_inputs import sha256
from scripts.generate_all_views import complete_result, write_index


def test_completed_view_requires_full_frame_coverage_and_matching_bytes(tmp_path):
    (tmp_path / "tracks.npz").write_bytes(b"test trajectory archive")
    (tmp_path / "trajectories.html").write_text("<div>export</div>")
    metadata = dict(frames=206, source_fps=12., requested_frames=0, finite=True, temporal_sampling="none",
                    source_sha256=sha256(tmp_path / "tracks.npz"), html_bytes=(tmp_path / "trajectories.html").stat().st_size)
    (tmp_path / "plotly.json").write_text(json.dumps(metadata))
    assert complete_result(tmp_path, dict(frames=206, fps=12.)) == metadata
    assert complete_result(tmp_path, dict(frames=294, fps=12.)) is None
    (tmp_path / "tracks.npz").write_bytes(b"changed trajectory archive")
    assert complete_result(tmp_path, dict(frames=206, fps=12.)) is None


def test_gallery_links_only_completed_outputs_and_escapes_errors(tmp_path):
    root = tmp_path / "running"
    root.mkdir()
    (root / "trajectories.html").write_text("<div>export</div>")
    state = dict(videos={"running": dict(status="complete", source=dict(frames=206, duration_seconds=206 / 12),
                                         result=dict(dense_trajectories=141120)),
                         "forest": dict(status="failed", error="<unavailable>")})
    write_index(tmp_path, state)
    page = (tmp_path / "index.html").read_text()
    assert 'href="running/trajectories.html"' in page
    assert 'href="forest/trajectories.html"' not in page
    assert "&lt;unavailable&gt;" in page and "<unavailable>" not in page
    assert json.loads((tmp_path / "batch.json").read_text())["videos"]["running"]["status"] == "complete"
