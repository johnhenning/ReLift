#!/usr/bin/env python
"""Reconstruct benchmark sources from an explicit revision map; never overwrite snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from freeze_benchmark_baseline import ROOT, freeze_revision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revisions", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=ROOT)
    args = parser.parse_args()
    sources = json.loads(args.revisions.read_text())
    for destination in sources:
        if (ROOT / destination).exists():
            raise FileExistsError(
                f"Snapshot already exists: {destination}; use a fresh checkout for historical reconstruction"
            )
    for destination, revision in sources.items():
        freeze_revision(ROOT / destination, revision, args.repository)


if __name__ == "__main__":
    main()
