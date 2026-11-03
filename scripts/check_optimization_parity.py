#!/usr/bin/env python
"""Validate isolated experiment outputs against a completed control trial.

    uv run python scripts/check_optimization_parity.py EXPERIMENT_DIR

Correction compares every frame/pixel; provider tasks compare loaded-state and
artifact hashes from every recorded run and verify the stored artifact bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_offload import compare


def sha(path: str | Path) -> Any:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--control", default="control")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--all-trials",
        action="store_true",
        help="Compare every completed repeat, including controls",
    )
    args = parser.parse_args()
    paths = sorted(args.experiment.glob("*/trial-*/trial.json"))
    records = [(path, json.loads(path.read_text())) for path in paths]
    controls = [
        (path, record)
        for path, record in records
        if record["mode"] == args.control and record["status"] == "complete"
    ]
    if not controls:
        raise ValueError("No completed control trial")
    control_path, control = controls[0]
    reports = []
    if control["task"] == "compilation":
        reference = Path(control["result"]["runs"][0]["output"])
        for path, record in records:
            if record["status"] != "complete":
                continue
            for run in record["result"]["runs"]:
                report = compare(reference, Path(run["output"]), tolerance=args.tolerance)
                for category in ("ordinary_xyz", "repeated_xyz", "raw_lift"):
                    error = report["errors"][category]
                    if error["nonfinite_vectors"] or (
                        error["max_normalized_vector_error"] is not None
                        and error["max_normalized_vector_error"] > args.tolerance
                    ):
                        report["passed"] = False
                report.update(
                    variant=record["mode"],
                    phase=run["phase"],
                    trial=str(path),
                    trial_sha256=sha(path),
                )
                reports.append(report)
    elif control["task"] in ("correction", "pipeline"):
        # Full checks on one output per distinct candidate. Timing repeats share
        # inputs/settings/seed; variant records list the exact checked run.
        seen = set()
        all_trials = args.all_trials or control["task"] == "pipeline"
        for path, record in records:
            if record["status"] != "complete" or path == control_path:
                continue
            if not all_trials and (record["mode"] == args.control or record["mode"] in seen):
                continue
            seen.add(record["mode"])
            report = compare(
                Path(control["output"]), Path(record["output"]), tolerance=args.tolerance
            )
            for category in ("ordinary_xyz", "repeated_xyz", "raw_lift"):
                error = report["errors"][category]
                if error["nonfinite_vectors"] or (
                    error["max_normalized_vector_error"] is not None
                    and error["max_normalized_vector_error"] > args.tolerance
                ):
                    report["passed"] = False
            for prepared in report.get("prepared_outputs", {}).values():
                if prepared.get("compared") and any(
                    not value["equal"] for value in prepared["arrays"].values()
                ):
                    report["passed"] = False
            report.update(
                variant=record["mode"],
                trial=str(path),
                trial_sha256=sha(path),
                control_trial=str(control_path),
                checker_sha256=sha(__file__),
            )
            reports.append(report)
    else:
        reference = control["result"]
        for path, record in records:
            if record["status"] != "complete":
                continue
            result = record["result"]
            artifacts = result.get("artifacts", {})
            verified = all(
                sha(Path(record["output"]) / name) == digest for name, digest in artifacts.items()
            )
            state_equal = (
                (
                    bool(result.get("state_sha256"))
                    and result["state_sha256"] == reference.get("state_sha256")
                )
                if control["task"] == "alltracker"
                else None
            )
            inputs_equal = result.get("input_sha256") == reference.get("input_sha256")
            reports.append(
                dict(
                    variant=record["mode"],
                    trial=str(path),
                    trial_sha256=sha(path),
                    state_equal=state_equal,
                    inputs_equal=inputs_equal,
                    artifacts_verified=verified,
                    artifacts_equal=artifacts == reference.get("artifacts"),
                    passed=bool(
                        verified
                        and artifacts
                        and artifacts == reference.get("artifacts")
                        and inputs_equal
                        and state_equal is not False
                    ),
                )
            )
    value = dict(
        passed=bool(reports) and all(report["passed"] for report in reports), reports=reports
    )
    (args.experiment / "quality.json").write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(dict(passed=value["passed"], comparisons=len(reports))))
    if not value["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
