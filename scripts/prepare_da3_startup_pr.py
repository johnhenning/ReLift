#!/usr/bin/env python
"""Stage DA3 startup code and later evidence in the isolated stack checkout."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "outputs/pr-stack"
BASE = "codex/16-typed-records"
BRANCH = "codex/17-da3-startup"
CORE = [
    "relift/optimization.py",
    "relift/configuration.py",
    "relift/da3_offload.py",
    "relift/providers/da3.py",
    "relift/records/configuration.py",
    "relift/records/runtime.py",
    "relift/records/benchmark.py",
    "relift/records/artifacts.py",
    "scripts/configured_depth_experiment.py",
    "scripts/profile_da3_startup.py",
    "scripts/prepare_da3_startup_pr.py",
    "tests/test_da3_startup.py",
    "tests/gpu/test_da3_startup.py",
    "configs/inference/cpu-first-da3.toml",
    "configs/inference/cpu-first-combined.toml",
    "configs/inference/cpu-first-disk.toml",
    "configs/inference/cpu-first-reuse.toml",
    "scripts/run_da3_startup_campaign.py",
    "scripts/publish_da3_startup.py",
    "configs/experiments/da3-startup-roles.json",
    "configs/experiments/da3-startup.json",
    "docs/native-optimization-experiments/07-da3-startup-placement.md",
]


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(DEST), *args], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("measured", "final"))
    args = parser.parse_args()
    expected = BASE if args.phase == "measured" else BRANCH
    if git("branch", "--show-current") != expected or git("status", "--porcelain"):
        raise ValueError(f"Expected clean branch {expected}")
    files = list(CORE)
    if args.phase == "measured":
        git("checkout", "-b", BRANCH)
    else:
        files += [
            "scripts/run_da3_startup_campaign.py",
            "scripts/publish_da3_startup.py",
            "configs/experiments/da3-startup-roles.json",
            "configs/experiments/da3-startup-revisions.json",
            "docs/optimization-configuration.md",
            "docs/native-optimization-experiments/README.md",
            "docs/architecture.md",
            "docs/architecture-view.html",
        ]
        files += [
            str(p.relative_to(ROOT))
            for p in (ROOT / "docs/benchmarks/da3-startup").rglob("*")
            if p.is_file()
        ]
    for name in files:
        target = DEST / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    git("add", *files)
    print(git("diff", "--cached", "--stat"))


if __name__ == "__main__":
    main()
