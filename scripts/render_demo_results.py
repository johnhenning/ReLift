#!/usr/bin/env python
"""Render saved demo results without running inference or using the GPU.

    uv run python scripts/render_demo_results.py outputs/demo/cartwheel outputs/demo/basketball

Each directory must contain tracks.npz and tracks.mp4 from demo_video.py.
Writes preview.png/gif, pointcloud.png/gif/mp4, and rendering.json in each
directory, plus summary.json beside the demo directories (or --summary PATH).
Defaults match the previews generated for the local cartwheel/basketball demos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import cv2
import imageio.v2 as imageio
import matplotlib
from imageio.core.format import Format

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

from relift.records.artifacts import RenderSummary


def positive_int(value: Any) -> Any:
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def sha256(path: Path) -> Any:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_animation(frames: Any, root: Path, stem: Any, fps: Any, video: bool = False) -> None:
    frames[len(frames) // 2].save(root / f"{stem}.png")
    frames[0].save(
        root / f"{stem}.gif",
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / fps),
        loop=0,
    )
    if video:
        with imageio.get_writer(
            str(root / f"{stem}.mp4"), fps=fps, codec="libx264", macro_block_size=2
        ) as writer:
            for frame in frames:
                cast(Format.Writer, writer).append_data(np.asarray(frame))


def video_preview(root: Path, indices: Any, width: Any, fps: Any) -> None:
    cap = cv2.VideoCapture(str(root / "tracks.mp4"))
    frames = []
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open {root / 'tracks.mp4'}")
        for t in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode frame {t} of {root / 'tracks.mp4'}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height = max(1, round(frame.shape[0] * width / frame.shape[1]))
            frames.append(Image.fromarray(cv2.resize(frame, (width, height))))
    finally:
        cap.release()
    save_animation(frames, root, "preview", fps)


def display_coordinates(points: Any) -> Any:
    """Camera (x, y-down, z-forward) -> plot (x, z-forward, y-up)."""
    return points[:, [0, 2, 1]] * np.array([1, 1, -1])


def pointcloud_preview(root: Path, data: Any, indices: Any, stride: Any, fps: Any) -> Any:
    total = len(data["xyz"])
    raw = data["lift"][:, ::stride, ::stride].reshape(total, -1, 3).astype(np.float32)
    fixed = data["xyz"][:, ::stride, ::stride].reshape(total, -1, 3)
    # Colour follows each reference pixel, rather than resampling a later frame.
    colors = data["images"][0, ::stride, ::stride].reshape(-1, 3) / 255
    reference = display_coordinates(fixed[0])
    bounds = np.quantile(reference, [0.02, 0.98], axis=0)
    extent = bounds[1] - bounds[0]
    # Avoid singular plot limits for planar or constant-coordinate scenes.
    for axis in range(3):
        if extent[axis] < 1e-6:
            bounds[:, axis] += [-0.005, 0.005]
    extent = bounds[1] - bounds[0]
    fig = plt.figure(figsize=(10, 5), dpi=100, facecolor="#f5f6fa")
    axes = [fig.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]
    heading = fig.text(0.5, 0.95, "", ha="center", fontsize=14)
    fig.text(0.25, 0.87, "Raw lift", ha="center", fontsize=12)
    fig.text(0.75, 0.87, "ReLift", ha="center", fontsize=12)
    frames = []
    try:
        for t in indices:
            for ax, points in zip(axes, (raw, fixed), strict=False):
                ax.clear()
                p = display_coordinates(points[t])
                keep = np.isfinite(p).all(1)
                ax.scatter(*p[keep].T, c=colors[keep], s=1.2, depthshade=False, rasterized=True)
                ax.set(xlim=bounds[:, 0], ylim=bounds[:, 1], zlim=bounds[:, 2])
                ax.set_box_aspect(np.maximum(extent, 0.01))
                ax.view_init(elev=12, azim=-65)
                ax.set_axis_off()
                ax.set_facecolor("#f5f6fa")
            heading.set_text(
                f"{root.name.title()} | reconstructed 3D points | frame {t}/{total - 1}"
            )
            fig.subplots_adjust(left=0, right=1, bottom=0, top=0.86, wspace=0)
            fig.canvas.draw()
            frames.append(
                Image.fromarray(
                    np.asarray(cast(FigureCanvasAgg, fig.canvas).buffer_rgba())[..., :3].copy()
                )
            )
        save_animation(frames, root, "pointcloud", fps, video=True)
    finally:
        plt.close(fig)
    return bounds.tolist()


def render(root: Path, args: Any) -> RenderSummary:
    root = Path(root)
    source = root / "tracks.npz"
    with np.load(source) as archive:
        data = {
            key: archive[key] for key in ("xyz", "lift", "observed", "images", "checkpoint_step")
        }
    xyz = data["xyz"]
    if xyz.ndim != 4 or xyz.shape[-1] != 3 or min(xyz.shape[:3]) == 0:
        raise ValueError(f"{source}: expected nonempty (T,H,W,3) trajectories, got {xyz.shape}")
    if (
        data["lift"].shape != xyz.shape
        or data["images"].shape != xyz.shape
        or data["observed"].shape != xyz.shape[:3]
    ):
        raise ValueError(f"{source}: lift, images, and observed must match xyz's grid")
    finite = bool(np.isfinite(xyz).all())
    if not finite:
        raise ValueError(f"{source}: corrected trajectories contain nonfinite coordinates")
    indices = list(range(0, len(xyz), args.frame_step))
    video_preview(root, indices, args.preview_width, args.fps)
    bounds = pointcloud_preview(root, data, indices, args.pixel_stride, args.fps)
    settings = dict(
        generator="scripts/render_demo_results.py",
        source="tracks.npz",
        source_sha256=sha256(source),
        sampled_frames=indices,
        fps=args.fps,
        preview_width=args.preview_width,
        pixel_stride=args.pixel_stride,
        color_source="reference-frame RGB",
        display_axes=["x", "z", "-y"],
        axis_bounds=bounds,
        axis_quantiles=[0.02, 0.98],
        elevation=12,
        azimuth=-65,
        python=platform.python_version(),
        packages={
            p: version(p)
            for p in (
                "numpy",
                "matplotlib",
                "pillow",
                "opencv-python-headless",
                "imageio",
                "imageio-ffmpeg",
            )
        },
    )
    (root / "rendering.json").write_text(json.dumps(settings, indent=2) + "\n")
    return dict(
        frames=len(xyz),
        grid=list(xyz.shape[1:3]),
        trajectories=int(np.prod(xyz.shape[1:3])),
        observed_percent=round(float(data["observed"].mean() * 100), 1),
        finite=finite,
        checkpoint_step=int(data["checkpoint_step"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("directories", nargs="+", type=Path, help="demo output directories")
    ap.add_argument("--frame-step", type=positive_int, default=3, help="render every Nth frame")
    ap.add_argument(
        "--pixel-stride", type=positive_int, default=4, help="3D grid subsampling stride"
    )
    ap.add_argument(
        "--fps", type=positive_int, default=5, help="animation playback frames per second"
    )
    ap.add_argument("--preview-width", type=positive_int, default=900)
    ap.add_argument(
        "--summary", type=Path, help="default: summary.json beside the demo directories"
    )
    args = ap.parse_args()
    names = [p.name for p in args.directories]
    if len(names) != len(set(names)):
        ap.error("demo directory names must be unique for the summary")
    if args.summary is None and len({p.resolve().parent for p in args.directories}) != 1:
        ap.error("provide --summary when demo directories have different parents")
    summary = {}
    for directory in args.directories:
        summary[directory.name] = render(directory, args)
        print(f"Rendered {directory}", flush=True)
    path = args.summary or args.directories[0].parent / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Summary: {path}")


if __name__ == "__main__":
    main()
