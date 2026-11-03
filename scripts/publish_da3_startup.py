#!/usr/bin/env python
"""Export technical startup evidence; analysis is maintained in separate Markdown."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from relift.records.benchmark import TrialRecord

from publish_background_commits import paired_results, save
from summarize_optimization_experiments import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, nargs="*", default=[])
    parser.add_argument("--extra-quality", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    paths = sorted(args.campaign.rglob("trial.json"))
    records: list[TrialRecord] = [json.loads(path.read_text()) for path in paths]
    roles = json.loads(args.roles.read_text())
    save(args.out / "trials.json", records)
    save(args.out / "summary.json", summarize(args.campaign, roles))
    save(args.out / "pairs.json", paired_results(records, roles, "control", "cpu-first"))
    quality = sorted(args.campaign.rglob("quality.json")) + args.extra_quality
    save(args.out / "quality.json", {str(path): json.loads(path.read_text()) for path in quality})
    save(
        args.out / "startup.json",
        [
            dict(
                experiment=r["experiment"],
                mode=r["mode"],
                repeat=r["repeat"],
                startup=r["result"].get("startup", []),
            )
            for r in records
            if r["result"] is not None
        ],
    )
    save(
        args.out / "diagnostics.json",
        {str(path): json.loads(path.read_text()) for path in args.diagnostics},
    )
    sources = [*paths, *quality, *args.diagnostics, args.roles, Path(__file__)]
    save(
        args.out / "provenance.json",
        dict(
            source_sha256={
                str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
            }
        ),
    )
    print(args.out)


if __name__ == "__main__":
    main()
