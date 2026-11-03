#!/usr/bin/env python
"""Derive experiment tables from technical records; analysis/decisions stay separate."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from statistics import median


def summarize(root):
    groups = defaultdict(list)
    for path in sorted(root.rglob("trial.json")):
        record = json.loads(path.read_text())
        groups[(record["experiment"], record["mode"], json.dumps(record["code_hashes"], sort_keys=True))].append((path, record))
    rows = []
    for (experiment, mode, _), records in groups.items():
        good = [r for _, r in records if r["status"] == "complete"]
        row = dict(experiment=experiment, variant=mode, n=len(good), attempts=len(records),
                   statuses=dict(Counter(r["status"] for _, r in records)),
                   diagnostic=any(r.get("diagnostic") for _, r in records),
                   trial_records=[str(path) for path, _ in records], options=records[0][1]["options"])
        wall = [r["wall_seconds"] for r in good]
        row.update(wall_median=median(wall) if wall else None, wall_min=min(wall) if wall else None, wall_max=max(wall) if wall else None)
        for key in ("sampled_peak_process_tree_rss_bytes", "sampled_peak_process_tree_pss_bytes", "sampled_peak_gpu_device_used_bytes",
                    "process_tree_read_bytes", "process_tree_write_bytes"):
            values = [r["telemetry"][key] for r in good if r["telemetry"].get(key) is not None]
            row[key] = median(values) if values else None
        phases = defaultdict(list)
        for record in good:
            for name, value in (record.get("result") or {}).get("phases", {}).items():
                phases[name].append(value["seconds"])
        row["phase_medians"] = {name: median(values) for name, values in phases.items()}
        peaks = [r["sampled_output_peak"]["allocated_bytes"] for r in good]
        row["sampled_output_peak_allocated_bytes"] = median(peaks) if peaks else None
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows = summarize(args.root)
    (args.root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["# Optimization experiment measurements", "", "Diagnostic profiles are separate from unprofiled measurements. Stage medians do not include process startup; process medians do. Failures contribute no successful timing.", "",
             "| Experiment | Variant | Completed | Process seconds median [min,max] | Stage median seconds | Sampled GPU GiB | Sampled RSS GiB | Physical writes GiB |", "|---|---|---:|---|---|---:|---:|---:|"]
    for row in rows:
        fmt = lambda value: "—" if value is None else f"{value:.3f}"
        timing = f"{fmt(row['wall_median'])} [{fmt(row['wall_min'])}, {fmt(row['wall_max'])}]"
        phases = ", ".join(f"{key}={value:.3f}" for key, value in row["phase_medians"].items())
        resource = [fmt(row[key] / 2**30) if row[key] is not None else "—" for key in
                    ("sampled_peak_gpu_device_used_bytes", "sampled_peak_process_tree_rss_bytes", "process_tree_write_bytes")]
        lines.append(f"| {row['experiment']} | {row['variant']}{' (diagnostic)' if row['diagnostic'] else ''} | {row['n']}/{row['attempts']} | {timing} | {phases} | " + " | ".join(resource) + " |")
    (args.root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
