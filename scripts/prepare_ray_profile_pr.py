#!/usr/bin/env python
"""Stage the registered ablation or final eager follow-up in the isolated PR clone.

No commits, pushes or PR creation. The shared workspace Git index stays untouched.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "outputs/pr-stack"
BASE = "codex/09-native-da3-packing"
BRANCH = "codex/10-cuda-ray-profiling"
CODE = [
    "relift/cuda_rays.py",
    "relift/streaming.py",
    "relift/optimization.py",
    "tests/gpu/test_cuda_rays.py",
    "tests/gpu/test_ray_arithmetic.py",
    "tests/gpu/test_timeline_rays.py",
    "tests/gpu/test_configurable_optimizations.py",
    "tests/test_optimization_config.py",
    "configs/optimizations/rays-cuda-timeline.json",
]
FINAL = [
    ".gitattributes",
    "scripts/profile_cuda_rays.py",
    "scripts/benchmark_ray_startup.py",
    "scripts/audit_cuda_rays.py",
    "scripts/publish_ray_profile.py",
    "scripts/prepare_ray_profile_pr.py",
    "configs/experiments/cuda-ray-timeline.json",
    "configs/experiments/cuda-ray-source-revisions.json",
    "docs/native-optimization-experiments/cuda-ray-profile.md",
    "docs/native-optimization-experiments/cuda-ray-audit.md",
    "docs/native-optimization-experiments/README.md",
    "docs/optimization-configuration.md",
]


def git(*args: Any) -> Any:
    return subprocess.check_output(["git", "-C", str(DEST), *args], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["ablation", "final"])
    parser.add_argument(
        "--source",
        type=Path,
        help="Explicit source; ablation defaults to its measured frozen snapshot",
    )
    args = parser.parse_args()
    expected = BASE if args.phase == "ablation" else BRANCH
    if git("branch", "--show-current") != expected or git("status", "--porcelain"):
        raise ValueError(f"Expected clean isolated branch {expected}")
    source = (
        args.source
        or (
            ROOT / "outputs/experiments/rays-timeline-v2-source"
            if args.phase == "ablation"
            else ROOT
        )
    ).resolve()
    files = CODE + (FINAL if args.phase == "final" else [])
    if args.phase == "final":
        files += [
            str(path.relative_to(source))
            for path in sorted((source / "docs/benchmarks/cuda-ray-profile").rglob("*"))
            if path.is_file()
        ]
    for filename in files:
        if not (source / filename).is_file():
            raise FileNotFoundError(source / filename)
    if args.phase == "ablation":
        git("checkout", "-b", BRANCH)
    for filename in files:
        destination = DEST / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / filename, destination)
    subprocess.run([sys.executable, str(DEST / "scripts/build_architecture_view.py")], check=True)
    git("add", *files, "docs/architecture-view.html")
    print(git("diff", "--cached", "--stat"))


if __name__ == "__main__":
    main()
