#!/usr/bin/env python
"""Export technical startup evidence; analysis is maintained in separate Markdown."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from relift.records.artifacts import DA3StartupProfile, DA3StartupProfileExcerpt
    from relift.records.benchmark import DA3StartupExport, TrialRecord

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.publish_background_commits import paired_results, save
from scripts.summarize_optimization_experiments import summarize


def profile_excerpt(path: Path, limit: int) -> DA3StartupProfileExcerpt:
    """Publish bounded technical tables; provenance fingerprints the full raw file."""
    profile: DA3StartupProfile = json.loads(path.read_text())
    return {
        **profile,
        "functions": sorted(
            profile["functions"], key=lambda row: row["inclusive_seconds"], reverse=True
        )[:limit],
        "function_count_full": len(profile["functions"]),
        "functions_limit": limit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, nargs="*", default=[])
    parser.add_argument("--profile-functions", type=int, default=100)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--extra-quality", type=Path, nargs="*", default=[])
    parser.add_argument("--additional-trials", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.profile_functions < 0:
        parser.error("--profile-functions must be nonnegative")
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    paths = sorted(args.campaign.rglob("trial.json")) + args.additional_trials
    records: list[TrialRecord] = [json.loads(path.read_text()) for path in paths]
    roles = json.loads(args.roles.read_text())
    save(args.out / "trials.json", records)
    save(args.out / "summary.json", summarize(args.campaign, roles))
    save(args.out / "pairs.json", paired_results(records, roles, "control", "cpu-first"))
    quality = sorted(args.campaign.rglob("quality.json")) + args.extra_quality
    save(args.out / "quality.json", {str(path): json.loads(path.read_text()) for path in quality})
    startup: list[DA3StartupExport] = [
        dict(
            experiment=r["experiment"],
            mode=r["mode"],
            repeat=r["repeat"],
            startup=r["result"].get("startup", []),
        )
        for r in records
        if r["result"] is not None
    ]
    save(args.out / "startup.json", startup)
    save(
        args.out / "diagnostics.json",
        {str(path): profile_excerpt(path, args.profile_functions) for path in args.diagnostics},
    )
    (args.out / "environment.json").write_bytes(args.environment.read_bytes())
    sources = [
        *paths,
        *quality,
        *args.diagnostics,
        args.environment,
        args.roles,
        Path(__file__),
        ROOT / "scripts/publish_background_commits.py",
        ROOT / "scripts/summarize_optimization_experiments.py",
    ]
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
