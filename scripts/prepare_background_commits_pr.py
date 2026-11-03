#!/usr/bin/env python
"""Stage measured code or final background-commit evidence in the isolated PR clone."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "outputs/pr-stack"
BASE = "codex/13-component-configuration"
BRANCH = "codex/14-background-commits"
MEASURED = [
    "relift/shard_commits.py",
    "relift/optimization.py",
    "relift/configuration.py",
    "relift/streaming.py",
    "scripts/experiment_worker.py",
    "scripts/pipeline_experiment.py",
    "scripts/prepare_background_commits_pr.py",
    "tests/test_shard_commits.py",
    "tests/gpu/test_shard_commits.py",
    "configs/experiments/background-commits.json",
    "configs/inference/background-commits.toml",
    "configs/inference/background-cached.toml",
    "configs/inference/background-prefetch.toml",
    "configs/inference/background-native.toml",
]


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(DEST), *args], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("measured", "final"))
    args = parser.parse_args()
    expected = BASE if args.phase == "measured" else BRANCH
    if git("branch", "--show-current") != expected or git("status", "--porcelain"):
        raise ValueError(f"Expected clean isolated branch {expected}")
    files = list(MEASURED)
    if args.phase == "measured":
        git("checkout", "-b", BRANCH)
    else:
        files += [
            "scripts/publish_background_commits.py",
            "docs/optimization-configuration.md",
            "docs/native-optimization-experiments/06-background-commits.md",
            "docs/native-optimization-experiments/README.md",
            "docs/architecture.md",
            "docs/architecture-view.html",
            ".gitattributes",
            "configs/experiments/background-commits-revisions.json",
            "configs/experiments/background-commits-roles.json",
        ]
        files += [
            str(p.relative_to(ROOT))
            for p in sorted((ROOT / "docs/benchmarks/background-commits").rglob("*"))
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
