#!/usr/bin/env python
"""Copy one completed native experiment into the next isolated review branch.

This stages files but does not commit, push or open a PR. The shared workspace's
Git index is never changed. Validate the staged diff before publishing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "outputs/pr-stack"
BRANCHES = [
    "codex/05-optimization-experiments",
    "codex/06-native-batch-gather",
    "codex/07-native-shard-writer",
    "codex/08-native-cuda-rays",
    "codex/09-native-da3-packing",
]


def git(*args: Any) -> Any:
    return subprocess.check_output(["git", "-C", str(DEST), *args], text=True).strip()


def portable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, str):
        return value.replace(str(ROOT) + "/", "")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layer", type=int, choices=range(1, 5))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT,
        help="Immutable layer source; defaults to current workspace",
    )
    args = parser.parse_args()
    expected = BRANCHES[args.layer if args.refresh else args.layer - 1]
    if git("branch", "--show-current") != expected or git("status", "--porcelain"):
        raise ValueError(f"Expected clean isolated branch {expected}")
    if not args.refresh:
        git("checkout", "-b", BRANCHES[args.layer])
    source_root = args.source.resolve()
    if source_root != ROOT:
        files = sorted(json.loads((source_root / "snapshot.json").read_text())["files"])
    else:
        files = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True
        ).splitlines()
    for filename in files:
        source = source_root / filename
        if not source.is_file() or source.suffix in (".mp4", ".so"):
            continue
        target = DEST / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if filename == "README.md":
            target.write_text(
                source.read_text().replace(
                    "https://github.com/ReLift3D/ReLift.git",
                    "https://github.com/johnhenning/ReLift.git",
                )
            )
        else:
            shutil.copy2(source, target)
    results = [
        "01-cpu-batch-gathering-results.md",
        "02-shard-packing-and-writing-results.md",
        "03-cuda-ray-construction-results.md",
        "04-da3-archive-and-block-loading-results.md",
    ]
    for filename in (
        "scripts/prepare_native_pr.py",
        "configs/experiments/native-optimizations.json",
        "configs/experiments/native-measurement-roles.json",
        "scripts/build_optimization_charts.py",
        "scripts/summarize_optimization_experiments.py",
        "docs/optimization-configuration.md",
        "docs/native-optimization-experiments/README.md",
        "docs/native-optimization-experiments/" + results[args.layer - 1],
    ):
        if (ROOT / filename).is_file():
            shutil.copy2(ROOT / filename, DEST / filename)
    if args.layer >= 3:
        for filename in (
            "docs/native-optimization-experiments/cuda-ray-audit.md",
            "scripts/run_source_tests.py",
        ):
            shutil.copy2(ROOT / filename, DEST / filename)
    campaign = ROOT / "outputs/experiments/native-optimizations"
    evidence = DEST / "docs/benchmarks/native-optimizations"
    evidence.mkdir(parents=True, exist_ok=True)
    for filename in ("summary.json", "environment.json"):
        (evidence / filename).write_text(
            json.dumps(portable(json.loads((campaign / filename).read_text())), indent=2) + "\n"
        )
    for path in campaign.glob("ray-audit*.json"):
        (evidence / path.name).write_text(
            json.dumps(portable(json.loads(path.read_text())), indent=2) + "\n"
        )
    trials = [portable(json.loads(p.read_text())) for p in sorted(campaign.rglob("trial.json"))]
    (evidence / "trials.json").write_text(json.dumps(trials, indent=2) + "\n")
    quality = {
        str(p.relative_to(campaign)): portable(json.loads(p.read_text()))
        for p in sorted(campaign.rglob("quality.json"))
    }
    (evidence / "quality.json").write_text(json.dumps(quality, indent=2) + "\n")
    for image in (ROOT / "outputs/native-optimization-charts").glob("*.png"):
        shutil.copy2(image, evidence / image.name)
    git("add", "-A")
    print(git("diff", "--cached", "--stat"))


if __name__ == "__main__":
    main()
