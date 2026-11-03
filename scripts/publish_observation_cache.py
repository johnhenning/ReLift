#!/usr/bin/env python
"""Publish cache measurements/figures; analysis remains in a separate document."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from publish_ray_profile import save

from scripts.summarize_optimization_experiments import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--audit", type=Path, action="append", default=[])
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    profile = json.loads(args.profile.read_text())
    save(args.out / "profile.json", profile)
    for path in args.audit:
        save(args.out / (path.parent.name + ".json"), json.loads(path.read_text()))
    records = [json.loads(path.read_text()) for path in sorted(args.campaign.rglob("trial.json"))]
    save(args.out / "trials.json", records)
    save(args.out / "summary.json", summarize(args.campaign, json.loads(args.roles.read_text())))
    save(
        args.out / "quality.json",
        {
            str(path.relative_to(args.campaign)): json.loads(path.read_text())
            for path in sorted(args.campaign.rglob("quality.json"))
        },
    )
    diagnostics = {}
    for path in sorted(args.campaign.rglob("diagnostic.json")):
        value = json.loads(path.read_text())
        diagnostics[str(path.relative_to(args.campaign))] = dict(
            spans=value["spans"], caveat=value["caveat"], functions=value["functions"][:100]
        )
    save(args.out / "diagnostics.json", diagnostics)
    sources = [args.profile, args.roles, *args.audit]
    save(
        args.out / "provenance.json",
        dict(
            sources={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = profile["results"]
    labels = list(rows)
    figure, axes = plt.subplots(1, 2, figsize=(14, 7), layout="constrained")
    samples = [[s["wall_ms"] for s in rows[name]["samples"]] for name in labels]
    medians = [statistics.median(s) for s in samples]
    axes[0].barh(labels, medians, color="#367b92")
    axes[0].errorbar(
        medians,
        range(len(labels)),
        xerr=[
            [m - min(s) for m, s in zip(medians, samples, strict=False)],
            [max(s) - m for m, s in zip(medians, samples, strict=False)],
        ],
        fmt="none",
        color="#263746",
        capsize=3,
    )
    for y, (median, values) in enumerate(zip(medians, samples, strict=False)):
        axes[0].text(max(values), y, f" {median:.3f}", va="center")
    axes[0].set_xlim(0, max(max(s) for s in samples) * 1.25)
    axes[0].set_title("Warm ray-batch runtime · observed min–max")
    axes[0].set_xlabel("milliseconds")
    resident = [profile["construction"][name]["cache"]["resident_bytes"] / 2**20 for name in labels]
    axes[1].barh(labels, resident, color="#278574")
    for y, value in enumerate(resident):
        axes[1].text(value, y, f" {value:.2f}", va="center")
    axes[1].set_xlim(0, max(resident) * 1.25 or 1)
    axes[1].set_title("Retained observation cache\nExcludes depth and active batch")
    axes[1].set_xlabel("MiB")
    for axis in axes:
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.2)
    T, H, W = profile["shape"]
    figure.suptitle(
        f"Recorded observation-cache component measurements · {T} frames, {H}×{W}", fontsize=15
    )
    figure.supxlabel(
        f"Warm component medians from {len(samples[0])} rotating rounds; construction/upload and commits are excluded here and included in fresh-process trials.\nCache resident bytes are explicit tensor storage, not whole-device allocation.",
        fontsize=10,
    )
    figure.savefig(args.out / "components.png", dpi=150)
    plt.close(figure)
    print(args.out)


if __name__ == "__main__":
    main()
