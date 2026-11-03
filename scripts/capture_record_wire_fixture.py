#!/usr/bin/env python
"""Capture report wire expectations from an immutable pre-migration source snapshot."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from relift.records.artifacts import SourceSnapshot
    from relift.records.comparison import ErrorReport


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Capture to a fresh file and review before replacing a wire fixture")
    snapshot: SourceSnapshot = json.loads((args.source / "snapshot.json").read_text())
    legacy = runpy.run_path(str(args.source / "scripts/compare_offload.py"))
    reference = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]], np.float32)
    regular = legacy["ErrorStats"](2.0)
    regular.add(reference, reference + np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], np.float32))
    nonfinite = legacy["ErrorStats"](2.0)
    nonfinite.add(reference, np.array([[np.nan, 0.0, 1.0], [1.0, 1.0, 1.0]], np.float32))
    regular_report: ErrorReport = regular.report()
    nonfinite_report: ErrorReport = nonfinite.report()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            dict(
                source_revision=snapshot.get("git_revision", snapshot["base_commit"]),
                regular=regular_report,
                nonfinite=nonfinite_report,
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(args.out)


if __name__ == "__main__":
    main()
