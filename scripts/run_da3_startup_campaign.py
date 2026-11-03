#!/usr/bin/env python
"""Run one startup campaign scope sequentially; scan saved outputs after timing."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOPES = {
    "screening": ("screening",),
    "measurements": ("da3", "pipeline"),
    "validation": ("composition", "disk", "reuse", "running", "forest"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=ROOT / "configs/experiments/da3-startup.json")
    parser.add_argument("--scope", choices=SCOPES, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    for experiment in SCOPES[args.scope]:
        command = [
            sys.executable,
            str(ROOT / "scripts/run_optimization_experiments.py"),
            "--spec",
            str(args.spec),
            "--experiment",
            experiment,
        ]
        if args.resume:
            command.append("--resume")
        subprocess.run(command, cwd=ROOT, check=True)
        trials = list((ROOT / spec["output_root"] / experiment).rglob("trial.json"))
        if not trials or any(
            json.loads(path.read_text())["status"] != "complete" for path in trials
        ):
            raise RuntimeError(f"Incomplete campaign scope: {experiment}; inspect retained records")
    for experiment in SCOPES[args.scope]:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/check_optimization_parity.py"),
                str(ROOT / spec["output_root"] / experiment),
                "--all-trials",
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
