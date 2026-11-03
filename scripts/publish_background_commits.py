#!/usr/bin/env python
"""Publish recorded commit measurements; interpretation lives in separate Markdown."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_optimization_experiments import summarize


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def paired_results(
    records: list[dict[str, Any]], roles: dict[str, str], control: str, candidate: str
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        if (
            record["status"] == "complete"
            and roles.get(record["experiment"], "benchmark") == "benchmark"
        ):
            groups[(record["experiment"], record["repeat"])][record["mode"]] = record
    result = []
    for (experiment, repeat), variants in sorted(groups.items()):
        if control not in variants or candidate not in variants:
            continue
        a, b = variants[control], variants[candidate]
        identity = (
            "code_hashes",
            "input_sha256",
            "task",
            "process_res",
            "seed",
            "environment",
            "driver_hashes",
        )
        if any(a.get(key) != b.get(key) for key in identity):
            raise ValueError(f"Unmatched paired trial conditions: {experiment}/{repeat}")
        result.append(
            dict(
                experiment=experiment,
                repeat=repeat,
                control_seconds=a["wall_seconds"],
                candidate_seconds=b["wall_seconds"],
                candidate_over_control=b["wall_seconds"] / a["wall_seconds"],
                seconds_saved=a["wall_seconds"] - b["wall_seconds"],
                control_phases=a["result"]["phases"],
                candidate_phases=b["result"]["phases"],
            )
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--control", default="control")
    parser.add_argument("--candidate", default="threaded")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.campaign.rglob("trial.json"))
    records = [json.loads(path.read_text()) for path in paths]
    roles = json.loads(args.roles.read_text())
    quality_paths = sorted(args.campaign.rglob("quality.json"))
    save(args.out / "trials.json", records)
    save(args.out / "summary.json", summarize(args.campaign, roles))
    save(
        args.out / "quality.json",
        {
            str(path.relative_to(args.campaign)): json.loads(path.read_text())
            for path in quality_paths
        },
    )
    pairs = paired_results(records, roles, args.control, args.candidate)
    save(args.out / "pairs.json", pairs)
    metrics: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        value = (record.get("result") or {}).get("shard_commits")
        if record["status"] == "complete" and value:
            metrics[(record["experiment"], record["mode"])].append(value)
    save(
        args.out / "commits.json",
        [
            dict(
                experiment=experiment,
                variant=variant,
                n=len(samples),
                samples=samples,
                medians={
                    key: median(sample[key] for sample in samples)
                    for key in samples[0]
                    if type(samples[0][key]) in (int, float)
                },
            )
            for (experiment, variant), samples in sorted(metrics.items())
        ],
    )
    sources = [*paths, *quality_paths, args.roles]
    save(
        args.out / "provenance.json",
        dict(
            source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ),
    )
    print(args.out)


if __name__ == "__main__":
    main()
