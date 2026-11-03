#!/usr/bin/env python
"""Run multiple full clips through an explicitly owned composable session.

uv run python scripts/run_inference_session.py CLIP1 CLIP2 --out outputs/session \
    --opt reuse_da3=true --opt hash_writes=true
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relift.optimization import InferenceSession, OptimizationConfig
from scripts.demo_video import main as infer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--optimizations", type=Path)
    parser.add_argument("--opt", action="append", default=[])
    parser.add_argument("--native-build", type=Path)
    parser.add_argument("--offload", choices=("ram", "disk", "streamed"), default="ram")
    args = parser.parse_args()
    settings = OptimizationConfig.read(args.optimizations, args.opt)
    with InferenceSession(settings, native_build=args.native_build) as session:
        for index, video in enumerate(args.videos):
            import random

            import numpy as np
            import torch

            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            infer(
                [
                    str(video),
                    "--out",
                    str(args.out / f"{index:02}-{video.stem}"),
                    "--num_frames",
                    "0",
                    "--offload",
                    args.offload,
                    "--scratch-dir",
                    str(args.out / "scratch"),
                    "--no_video",
                ],
                session=session,
            )


if __name__ == "__main__":
    main()
