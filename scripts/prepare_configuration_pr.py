#!/usr/bin/env python
"""Stage grouped configuration as its own next stack branch; never commit or push."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "outputs/pr-stack"
BASE = "codex/12-python-quality"
BRANCH = "codex/13-component-configuration"
FILES = [
    "relift/configuration.py",
    "relift/optimization.py",
    "relift/streaming.py",
    "scripts/demo_video.py",
    "scripts/run_inference_session.py",
    "scripts/experiment_worker.py",
    "scripts/pipeline_experiment.py",
    "scripts/configured_depth_experiment.py",
    "scripts/run_optimization_experiments.py",
    "scripts/optimization_matrix.py",
    "scripts/prepare_configuration_pr.py",
    "tests/test_configuration.py",
    "docs/optimization-configuration.md",
    "docs/architecture.md",
    "docs/architecture-view.html",
    "configs/experiments/configuration-validation.json",
]


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(DEST), *args], text=True).strip()


def main() -> None:
    if git("branch", "--show-current") != BASE or git("status", "--porcelain"):
        raise ValueError(f"Expected clean isolated branch {BASE}")
    files = FILES + [
        str(p.relative_to(ROOT)) for p in sorted((ROOT / "configs/inference").glob("*"))
    ]
    git("checkout", "-b", BRANCH)
    for name in files:
        target = DEST / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    git("add", *files)
    print(git("diff", "--cached", "--stat"))


if __name__ == "__main__":
    main()
