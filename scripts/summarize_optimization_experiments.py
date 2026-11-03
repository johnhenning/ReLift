#!/usr/bin/env python
"""Derive experiment tables from technical records; analysis/decisions stay separate."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median


def summarize(root, roles=None):
    groups = defaultdict(list)
    for path in sorted(root.rglob("trial.json")):
        record = json.loads(path.read_text())
        signature = {key: record.get(key) for key in ("code_hashes", "native_build", "environment", "wrapper_sha256", "options", "task", "input_sha256", "process_res", "seed")}
        groups[(record["experiment"], record["mode"], json.dumps(signature, sort_keys=True))].append((path, record))
    rows = []
    for (experiment, mode, signature), records in groups.items():
        good = [r for _, r in records if r["status"] == "complete"]
        row = dict(experiment=experiment, variant=mode, n=len(good), attempts=len(records),
                   measurement_role=(roles or {}).get(experiment,"benchmark"),
                   statuses=dict(Counter(r["status"] for _, r in records)),
                   diagnostic=any(r.get("diagnostic") for _, r in records),
                   trial_records=[str(path) for path, _ in records], options=records[0][1]["options"],
                   source_fingerprint=hashlib.sha256(signature.encode()).hexdigest(),
                   environment=records[0][1].get("environment"))
        quality_path = root / experiment / "quality.json"
        checks = json.loads(quality_path.read_text()).get("reports", []) if quality_path.exists() else []
        checks = [check for check in checks if check.get("variant") == mode]
        row["quality"] = ("pass" if all(check["passed"] for check in checks) else "fail") if checks else None
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
        highwaters = [max(values) for record in good if (values := [phase["process_rss_highwater_bytes"]
                      for phase in (record.get("result") or {}).get("phases", {}).values()
                      if phase.get("process_rss_highwater_bytes") is not None])]
        row["process_rss_highwater_bytes"] = median(highwaters) if highwaters else None
        peaks = [r["sampled_output_peak"]["allocated_bytes"] for r in good]
        row["sampled_output_peak_allocated_bytes"] = median(peaks) if peaks else None
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--roles", type=Path, help="Separate technical classification of benchmark, screening and validation groups")
    args = parser.parse_args()
    rows = summarize(args.root,json.loads(args.roles.read_text()) if args.roles else None)
    (args.root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["# Optimization experiment measurements", "", "Diagnostic profiles are separate from unprofiled measurements. Stage medians do not include process startup; process medians do. Failures contribute no successful timing.", "",
             "| Experiment | Variant | Completed | Quality gate | Process seconds median [min,max] | Stage median seconds | Sampled GPU GiB | Sampled RSS GiB | Physical writes GiB |", "|---|---|---:|---|---|---|---:|---:|---:|"]
    for row in rows:
        fmt = lambda value: "—" if value is None else f"{value:.3f}"
        timing = f"{fmt(row['wall_median'])} [{fmt(row['wall_min'])}, {fmt(row['wall_max'])}]"
        phases = ", ".join(f"{key}={value:.3f}" for key, value in row["phase_medians"].items())
        resource = [fmt(row[key] / 2**30) if row[key] is not None else "—" for key in
                    ("sampled_peak_gpu_device_used_bytes", "sampled_peak_process_tree_rss_bytes", "process_tree_write_bytes")]
        lines.append(f"| {row['experiment']} | {row['variant']}{' (diagnostic)' if row['diagnostic'] else ''} | {row['n']}/{row['attempts']} | {row['quality'] or 'not compared'} | {timing} | {phases} | " + " | ".join(resource) + " |")
    (args.root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
