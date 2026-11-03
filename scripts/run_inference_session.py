#!/usr/bin/env python
"""Run multiple full clips through an explicitly owned composable session.

uv run python scripts/run_inference_session.py CLIP1 CLIP2 --out outputs/session \
    --profile reuse-da3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relift.configuration import load_configuration
from relift.optimization import InferenceSession, OptimizationConfig
from scripts.demo_video import main as infer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--optimizations", type=Path)
    parser.add_argument("--opt", action="append", default=[])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--profile")
    group.add_argument("--inference-config", type=Path)
    parser.add_argument("--native-build", type=Path)
    parser.add_argument("--offload", choices=("ram", "disk", "streamed"))
    args = parser.parse_args()
    structured = args.profile is not None or args.inference_config is not None
    if structured and (args.optimizations or args.opt):
        parser.error("Choose a component profile/config or legacy optimization flags")
    selection = (
        load_configuration(profile=args.profile, path=args.inference_config) if structured else None
    )
    settings = (
        selection.config.optimizations()
        if selection
        else OptimizationConfig.read(args.optimizations, args.opt)
    )
    offload = args.offload or (selection.config.memory.offload if selection else "ram")
    with InferenceSession(selection or settings, native_build=args.native_build) as session:
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
                    offload,
                    "--scratch-dir",
                    str(args.out / "scratch"),
                    "--no_video",
                ],
                session=session,
            )


if __name__ == "__main__":
    main()
