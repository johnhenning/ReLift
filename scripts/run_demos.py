#!/usr/bin/env python
"""Run selected official demo clips and reproduce their saved previews/viewers.

    uv run python scripts/run_demos.py
    uv run python scripts/run_demos.py --render-only
    uv run python scripts/run_demos.py --num-frames 0 --plotly
    uv run python scripts/run_demos.py --videos running tennis --num-frames 0 --plotly

Downloads the three default model weights and selected inputs, runs demo_video.py
for each clip, then calls the saved static/interactive visualization scripts.
--render-only uses existing results and needs no downloads or GPU.
"""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from relift.config import paths
from scripts.download_demo_inputs import catalog, fetch_input


def run(script, *arguments):
    command = [sys.executable, f"scripts/{script}", *map(str, arguments)]
    print("+ " + shlex.join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render-only", action="store_true", help="regenerate previews from existing results")
    ap.add_argument("--output-root", type=Path)
    ap.add_argument("--videos", nargs="+", choices=tuple(catalog()), default=["cartwheel", "basketball"])
    ap.add_argument("--list-videos", action="store_true", help="list official inputs and exit")
    ap.add_argument("--num-frames", type=int, default=84, help="0 generates trajectories for every source frame")
    ap.add_argument("--plotly", action="store_true", help="also generate interactive 3D viewers")
    ap.add_argument("--skip-previews", action="store_true", help="skip static images and GIFs")
    args = ap.parse_args()
    if args.list_videos:
        for name, entry in catalog().items():
            print(f"{name:20} {entry['description']}")
        return
    if args.num_frames < 0:
        ap.error("--num-frames must be nonnegative")
    root = (args.output_root or paths()["outputs_root"] / ("demo-full" if args.num_frames == 0 else "demo")).resolve()
    directories = [root / name for name in args.videos]
    if not args.render_only:
        run("download_weights.py", "--only", "relift", "da3", "alltracker")
        inputs = [fetch_input(name) for name in args.videos]
        for video, output in zip(inputs, directories):
            run("demo_video.py", video, "--num_frames", args.num_frames, "--geometry", "da3", "--out", output)
    if not args.skip_previews:
        run("render_demo_results.py", *directories, "--summary", root / "summary.json")
    if args.plotly:
        run("plotly_demo.py", *directories)


if __name__ == "__main__":
    main()
