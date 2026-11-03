#!/usr/bin/env python
"""Run the background-commit scopes sequentially, then compare every saved trial."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOPES = ("correction", "pipeline", "combinations", "defaults")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--experiments", nargs="+", choices=SCOPES, default=list(SCOPES))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    for name in args.experiments:
        command = [
            sys.executable,
            str(ROOT / "scripts/run_optimization_experiments.py"),
            "--spec",
            str(args.spec.resolve()),
            "--experiment",
            name,
        ]
        if args.resume:
            command.append("--resume")
        subprocess.run(command, cwd=ROOT, check=True)
    # Full scans and hashing run after all timing trials, to avoid concurrent I/O.
    for name in args.experiments:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/check_optimization_parity.py"),
                str(ROOT / spec["output_root"] / name),
                "--all-trials",
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
