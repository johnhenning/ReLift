#!/usr/bin/env python
"""Export self-contained Plotly viewers of full-length dense demo trajectories.

    uv run python scripts/plotly_demo.py outputs/demo-full/cartwheel outputs/demo-full/basketball

Every temporal frame is included. Only reference pixels are sampled for browser
performance; tracks.npz remains dense. Click a point to inspect its complete path.
The saved HTML works offline and needs no Python server.
"""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.io as pio

HERE = Path(__file__).resolve().parent


def sample_grid(height, width, max_points):
    if max_points < 1:
        raise ValueError("max_points must be positive")
    stride = max(1, int(np.ceil(np.sqrt(height * width / max_points))))
    rows, cols = np.mgrid[0:height:stride, 0:width:stride]
    pixels = np.stack([rows.ravel(), cols.ravel()], axis=-1)
    if len(pixels) > max_points:
        pixels = pixels[np.linspace(0, len(pixels) - 1, max_points, dtype=int)]
    return pixels


def encoded(array, dtype):
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def prepare_data(source, max_points=2500, trajectory_count=60, fps_override=None):
    with np.load(source) as archive:
        dense = archive["xyz"]
        shape = dense.shape
        if len(shape) != 4 or shape[-1] != 3 or min(shape[:3]) < 1:
            raise ValueError(f"Expected nonempty (T,H,W,3) trajectories, got {shape}")
        total, height, width, _ = shape
        pixels = sample_grid(height, width, max_points)
        rows, cols = pixels.T
        finite = bool(np.isfinite(dense).all())
        if not finite:
            raise ValueError("Dense corrected trajectories contain nonfinite positions")
        corrected = dense[:, rows, cols].astype(np.float32)
        raw = archive["lift"][:, rows, cols].astype(np.float32)
        observed = archive["observed"][:, rows, cols].astype(np.uint8)
        rgb = archive["images"][0, rows, cols]
        fps = float(fps_override if fps_override is not None else archive["source_fps"] if "source_fps" in archive else 15)
        requested = int(archive["requested_frames"]) if "requested_frames" in archive else None
        process_resolution = int(archive["process_resolution"]) if "process_resolution" in archive else max(height, width)
        corrector_batch = int(archive["corrector_batch"]) if "corrector_batch" in archive else None
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    if not np.isfinite(corrected).all() or not np.isfinite(raw).all():
        raise ValueError("Sampled trajectories contain nonfinite positions")
    # Rank motion relative to initial depth so distant sky points do not dominate.
    displacement = np.linalg.norm(corrected - corrected[:1], axis=-1).max(axis=0)
    relative_motion = displacement / np.maximum(np.abs(corrected[0, :, 2]), 1e-6)
    dynamic = np.argsort(relative_motion)[::-1][:max(1, min(trajectory_count, len(pixels)))].tolist()
    # Camera coordinates -> right, forward, up; keep floating-point precision.
    corrected = corrected[:, :, [0, 2, 1]] * np.array([1, 1, -1], np.float32)
    raw = raw[:, :, [0, 2, 1]] * np.array([1, 1, -1], np.float32)
    combined = np.concatenate([raw.reshape(-1, 3), corrected.reshape(-1, 3)])
    bounds = np.quantile(combined, [.005, .995], axis=0)
    padding = np.maximum((bounds[1] - bounds[0]) * .08, .01)
    bounds[0] -= padding
    bounds[1] += padding
    colors = [f"rgb({r},{g},{b})" for r, g, b in rgb]
    payload = dict(frames=total, points=len(pixels), dense_points=height * width, fps=fps, finite=finite,
                   requested_frames=requested, grid=[height, width], process_resolution=process_resolution,
                   corrector_batch=corrector_batch, raw=encoded(raw, "<f4"), corrected=encoded(corrected, "<f4"),
                   observed=encoded(observed, "u1"), pixels=pixels[:, ::-1].tolist(),
                   dynamic=dynamic, bounds=bounds.tolist())
    return payload, raw, corrected, colors


def make_figure(name, payload, raw, corrected, colors):
    figure = go.Figure()
    hover = "pixel (%{customdata[0]}, %{customdata[1]})<br>x=%{x:.3f}<br>depth=%{y:.3f}<br>up=%{z:.3f}<extra>%{fullData.name}</extra>"
    # Trace order is shared with plotly_viewer.js: clouds, complete paths,
    # selected paths, selected heads. Positions update without resetting cameras.
    for points, scene, label in ((raw, "scene", "Raw lift"), (corrected, "scene2", "ReLift")):
        figure.add_trace(go.Scatter3d(x=points[0, :, 0], y=points[0, :, 1], z=points[0, :, 2],
                                     mode="markers", scene=scene, name=label, customdata=payload["pixels"],
                                     marker=dict(size=2.2, color=colors, opacity=.95), hovertemplate=hover))
    for scene in ("scene", "scene2"):
        figure.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines", scene=scene, name="Complete paths",
                                     line=dict(color="#557a9b", width=2), opacity=.45, hoverinfo="skip"))
    for scene in ("scene", "scene2"):
        figure.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines", scene=scene, name="Selected trajectory",
                                     line=dict(color="#e67700", width=6), hoverinfo="skip"))
    for scene in ("scene", "scene2"):
        figure.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="markers", scene=scene, name="Selected point",
                                     marker=dict(color="#e67700", size=6), hoverinfo="skip"))
    axes = {}
    for index, (key, title) in enumerate((("xaxis", "X"), ("yaxis", "Forward depth (Z)"), ("zaxis", "Up (−Y)"))):
        axes[key] = dict(title=title, range=[payload["bounds"][0][index], payload["bounds"][1][index]],
                         autorange=False, gridcolor="#dce2ea", zeroline=False)
    camera = dict(eye=dict(x=1.4, y=-1.7, z=.65), up=dict(x=0, y=0, z=1))
    scene = dict(**axes, aspectmode="data", camera=camera, uirevision="retain-camera", bgcolor="#f7f9fc")
    figure.update_layout(
        template="plotly_white", showlegend=False, height=620, margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor="#ffffff", font=dict(family="system-ui, sans-serif", color="#243047"),
        scene=dict(**scene, domain=dict(x=[0, .49], y=[0, 1])),
        scene2=dict(**scene, domain=dict(x=[.51, 1], y=[0, 1])),
        annotations=[dict(text="Raw lift", x=.245, y=1.04, xref="paper", yref="paper", showarrow=False),
                     dict(text="ReLift", x=.755, y=1.04, xref="paper", yref="paper", showarrow=False)],
        uirevision="retain-camera", meta=dict(scene=name, frames=payload["frames"], temporal_sampling="none"),
    )
    return figure


def export_viewer(directory, max_points=2500, trajectory_count=60, fps_override=None):
    directory = Path(directory)
    source = directory / "tracks.npz"
    payload, raw, corrected, colors = prepare_data(source, max_points, trajectory_count, fps_override)
    figure = make_figure(directory.name, payload, raw, corrected, colors)
    fragment = pio.to_html(figure, full_html=False, include_plotlyjs=True, div_id="trajectory-plot",
                           config=dict(responsive=True, displaylogo=False, scrollZoom=True),
                           post_script="window.reliftPlotReady = true; window.dispatchEvent(new Event('relift-plot-ready'));")
    template = (HERE / "plotly_viewer.html").read_text()
    title = html.escape(directory.name.replace("-", " ").title())
    page = template.replace("__TITLE__", title).replace("__PLOT__", fragment)
    page = page.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    page = page.replace("__VIEWER_JS__", (HERE / "plotly_viewer.js").read_text())
    output = directory / "trajectories.html"
    output.write_text(page, encoding="utf-8")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    metadata = dict(generator="scripts/plotly_demo.py", plotly_version=plotly.__version__,
                    source="tracks.npz", source_sha256=digest.hexdigest(), frames=payload["frames"],
                    source_fps=payload["fps"], requested_frames=payload["requested_frames"],
                    dense_trajectories=payload["dense_points"], displayed_trajectories=payload["points"],
                    finite=payload["finite"], duration_seconds=payload["frames"] / payload["fps"],
                    grid=payload["grid"], process_resolution=payload["process_resolution"], corrector_batch=payload["corrector_batch"],
                    complete_paths_shown=len(payload["dynamic"]), temporal_sampling="none",
                    display_axes=["x", "z", "-y"], spatial_selection="uniform reference pixel grid",
                    axis_quantiles=[.005, .995], embedded_plotlyjs=True, html_bytes=output.stat().st_size)
    (directory / "plotly.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Plotly: {output} ({payload['frames']} frames, {payload['points']} displayed / {payload['dense_points']} dense tracks)", flush=True)
    return metadata


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directories", nargs="+", type=Path)
    ap.add_argument("--max-points", type=int, default=2500)
    ap.add_argument("--trajectory-count", type=int, default=60, help="complete motion paths shown per panel")
    ap.add_argument("--fps", type=float, help="override playback timebase; otherwise use stored source FPS")
    args = ap.parse_args()
    if args.max_points < 1 or args.trajectory_count < 1:
        ap.error("point and trajectory counts must be positive")
    for directory in args.directories:
        export_viewer(directory, args.max_points, args.trajectory_count, args.fps)


if __name__ == "__main__":
    main()
