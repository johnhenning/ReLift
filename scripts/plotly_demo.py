#!/usr/bin/env python
"""Export self-contained Plotly viewers of full-length dense demo trajectories.

    uv run python scripts/plotly_demo.py outputs/demo-full/cartwheel outputs/demo-full/basketball

Every temporal frame is included. Only reference pixels are sampled for browser
performance; the mapped store or legacy NPZ remains dense. Click a point to inspect its complete path.
The saved HTML works offline and needs no Python server.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.io as pio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from relift.records.artifacts import ViewerMetadata, ViewerPayload
from relift.storage import TrajectoryStore, file_sha256


def sample_grid(height: Any, width: Any, max_points: Any) -> Any:
    if max_points < 1:
        raise ValueError("max_points must be positive")
    stride = max(1, int(np.ceil(np.sqrt(height * width / max_points))))
    rows, cols = np.mgrid[0:height:stride, 0:width:stride]
    pixels = np.stack([rows.ravel(), cols.ravel()], axis=-1)
    if len(pixels) > max_points:
        pixels = pixels[np.linspace(0, len(pixels) - 1, max_points, dtype=int)]
    return pixels


def encoded(array: Any, dtype: Any) -> Any:
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def prepare_data(
    source: TrajectoryStore | str | Path,
    max_points: int = 2500,
    trajectory_count: int = 60,
    fps_override: Any = None,
) -> tuple[ViewerPayload, np.ndarray[Any, Any], np.ndarray[Any, Any], list[str]]:
    store = source if isinstance(source, TrajectoryStore) else TrajectoryStore.open(source)
    shape = store.shape
    if len(shape) != 4 or shape[-1] != 3 or min(shape[:3]) < 1:
        raise ValueError(f"Expected nonempty (T,H,W,3) trajectories, got {shape}")
    if store.manifest is not None:
        store.verify()
    finite = store.finite()
    if not finite:
        raise ValueError("Dense trajectories contain nonfinite positions")
    total, height, width, _ = shape
    pixels = sample_grid(height, width, max_points)
    rows, cols = pixels.T
    sampled = store.sample(rows * width + cols)
    corrected = sampled["xyz"].astype(np.float32, copy=False)
    raw = sampled["lift"].astype(np.float32)
    observed = sampled["observed"].astype(np.uint8)
    rgb = store.array("images")[0, rows, cols]
    metadata = store.metadata
    fps = float(fps_override if fps_override is not None else metadata.get("source_fps", 15))
    requested = int(metadata["requested_frames"]) if "requested_frames" in metadata else None
    process_resolution = int(metadata.get("process_resolution", max(height, width)))
    corrector_batch = (
        int(metadata["corrector_batch"]) if metadata.get("corrector_batch") is not None else None
    )
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    if not np.isfinite(corrected).all() or not np.isfinite(raw).all():
        raise ValueError("Sampled trajectories contain nonfinite positions")
    # Rank motion relative to initial depth so distant sky points do not dominate.
    displacement = np.linalg.norm(corrected - corrected[:1], axis=-1).max(axis=0)
    relative_motion = displacement / np.maximum(np.abs(corrected[0, :, 2]), 1e-6)
    dynamic = np.argsort(relative_motion)[::-1][
        : max(1, min(trajectory_count, len(pixels)))
    ].tolist()
    # Camera coordinates -> right, forward, up; keep floating-point precision.
    corrected = corrected[:, :, [0, 2, 1]] * np.array([1, 1, -1], np.float32)
    raw = raw[:, :, [0, 2, 1]] * np.array([1, 1, -1], np.float32)
    combined = np.concatenate([raw.reshape(-1, 3), corrected.reshape(-1, 3)])
    bounds = np.quantile(combined, [0.005, 0.995], axis=0)
    padding = np.maximum((bounds[1] - bounds[0]) * 0.08, 0.01)
    bounds[0] -= padding
    bounds[1] += padding
    colors = [f"rgb({r},{g},{b})" for r, g, b in rgb]
    payload: ViewerPayload = dict(
        frames=total,
        points=len(pixels),
        dense_points=height * width,
        fps=fps,
        finite=finite,
        requested_frames=requested,
        grid=[height, width],
        process_resolution=process_resolution,
        corrector_batch=corrector_batch,
        raw=encoded(raw, "<f4"),
        corrected=encoded(corrected, "<f4"),
        observed=encoded(observed, "u1"),
        pixels=pixels[:, ::-1].tolist(),
        dynamic=dynamic,
        bounds=bounds.tolist(),
    )
    return payload, raw, corrected, colors


def make_figure(name: str, payload: ViewerPayload, raw: Any, corrected: Any, colors: Any) -> Any:
    figure = go.Figure()
    hover = "pixel (%{customdata[0]}, %{customdata[1]})<br>x=%{x:.3f}<br>depth=%{y:.3f}<br>up=%{z:.3f}<extra>%{fullData.name}</extra>"
    # Trace order is shared with plotly_viewer.js: clouds, complete paths,
    # selected paths, selected heads. Positions update without resetting cameras.
    for points, scene, label in ((raw, "scene", "Raw lift"), (corrected, "scene2", "ReLift")):
        figure.add_trace(
            go.Scatter3d(
                x=points[0, :, 0],
                y=points[0, :, 1],
                z=points[0, :, 2],
                mode="markers",
                scene=scene,
                name=label,
                customdata=payload["pixels"],
                marker=dict(size=2.2, color=colors, opacity=0.95),
                hovertemplate=hover,
            )
        )
    for scene in ("scene", "scene2"):
        figure.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="lines",
                scene=scene,
                name="Complete paths",
                line=dict(color="#557a9b", width=2),
                opacity=0.45,
                hoverinfo="skip",
            )
        )
    for scene in ("scene", "scene2"):
        figure.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="lines",
                scene=scene,
                name="Selected trajectory",
                line=dict(color="#e67700", width=6),
                hoverinfo="skip",
            )
        )
    for scene in ("scene", "scene2"):
        figure.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="markers",
                scene=scene,
                name="Selected point",
                marker=dict(color="#e67700", size=6),
                hoverinfo="skip",
            )
        )
    axes = {}
    for index, (key, title) in enumerate(
        (("xaxis", "X"), ("yaxis", "Forward depth (Z)"), ("zaxis", "Up (−Y)"))
    ):
        axes[key] = dict(
            title=title,
            range=[payload["bounds"][0][index], payload["bounds"][1][index]],
            autorange=False,
            gridcolor="#dce2ea",
            zeroline=False,
        )
    camera = dict(eye=dict(x=1.4, y=-1.7, z=0.65), up=dict(x=0, y=0, z=1))
    scene = dict(
        **axes, aspectmode="data", camera=camera, uirevision="retain-camera", bgcolor="#f7f9fc"
    )
    figure.update_layout(
        template="plotly_white",
        showlegend=False,
        height=620,
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor="#ffffff",
        font=dict(family="system-ui, sans-serif", color="#243047"),
        scene=dict(**scene, domain=dict(x=[0, 0.49], y=[0, 1])),
        scene2=dict(**scene, domain=dict(x=[0.51, 1], y=[0, 1])),
        annotations=[
            dict(text="Raw lift", x=0.245, y=1.04, xref="paper", yref="paper", showarrow=False),
            dict(text="ReLift", x=0.755, y=1.04, xref="paper", yref="paper", showarrow=False),
        ],
        uirevision="retain-camera",
        meta=dict(scene=name, frames=payload["frames"], temporal_sampling="none"),
    )
    return figure


def export_viewer(
    directory: str | Path,
    max_points: int = 2500,
    trajectory_count: int = 60,
    fps_override: Any = None,
) -> ViewerMetadata:
    directory = Path(directory)
    store = TrajectoryStore.open(directory)
    source = (
        store.root / "manifest.json" if store.manifest is not None else directory / "tracks.npz"
    )
    payload, raw, corrected, colors = prepare_data(
        store, max_points, trajectory_count, fps_override
    )
    figure = make_figure(directory.name, payload, raw, corrected, colors)
    fragment = pio.to_html(
        figure,
        full_html=False,
        include_plotlyjs=True,
        div_id="trajectory-plot",
        config=dict(responsive=True, displaylogo=False, scrollZoom=True),
        post_script="window.reliftPlotReady = true; window.dispatchEvent(new Event('relift-plot-ready'));",
    )
    template = (HERE / "plotly_viewer.html").read_text()
    title = html.escape(directory.name.replace("-", " ").title())
    page = template.replace("__TITLE__", title).replace("__PLOT__", fragment)
    page = page.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    page = page.replace("__VIEWER_JS__", (HERE / "plotly_viewer.js").read_text())
    output = directory / "trajectories.html"
    output.write_text(page, encoding="utf-8")
    metadata: ViewerMetadata = dict(
        generator="scripts/plotly_demo.py",
        plotly_version=plotly.__version__,
        source=str(source.relative_to(directory)),
        source_sha256=file_sha256(source),
        frames=payload["frames"],
        storage="mapped-track-shards" if store.manifest is not None else "npz",
        input_sha256=store.metadata.get("input_sha256"),
        source_fps=payload["fps"],
        requested_frames=payload["requested_frames"],
        dense_trajectories=payload["dense_points"],
        displayed_trajectories=payload["points"],
        finite=payload["finite"],
        duration_seconds=payload["frames"] / payload["fps"],
        grid=payload["grid"],
        process_resolution=payload["process_resolution"],
        corrector_batch=payload["corrector_batch"],
        complete_paths_shown=len(payload["dynamic"]),
        temporal_sampling="none",
        display_axes=["x", "z", "-y"],
        spatial_selection="uniform reference pixel grid",
        axis_quantiles=[0.005, 0.995],
        embedded_plotlyjs=True,
        html_bytes=output.stat().st_size,
    )
    (directory / "plotly.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"Plotly: {output} ({payload['frames']} frames, {payload['points']} displayed / {payload['dense_points']} dense tracks)",
        flush=True,
    )
    return metadata


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("directories", nargs="+", type=Path)
    ap.add_argument("--max-points", type=int, default=2500)
    ap.add_argument(
        "--trajectory-count", type=int, default=60, help="complete motion paths shown per panel"
    )
    ap.add_argument(
        "--fps", type=float, help="override playback timebase; otherwise use stored source FPS"
    )
    args = ap.parse_args()
    if args.max_points < 1 or args.trajectory_count < 1:
        ap.error("point and trajectory counts must be positive")
    for directory in args.directories:
        export_viewer(directory, args.max_points, args.trajectory_count, args.fps)


if __name__ == "__main__":
    main()
