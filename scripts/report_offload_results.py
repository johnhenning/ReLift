#!/usr/bin/env python
"""Render a technical Markdown report from explicitly selected benchmark records.

This is a read-only renderer: trial records, not summary files or inferred hardware,
are the source of measurements. Optional analysis is supplied by the caller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any
from urllib.parse import quote

MATCH_FIELDS = (
    "scene",
    "scenario",
    "input_sha256",
    "process_res",
    "corrector_batch",
    "num_frames",
    "seed",
)
PLACEMENT_SETTINGS = {
    "offload_requested",
    "offload_resolved",
    "ram_budget_gb",
    "vram_budget_gb",
    "scratch_budget_gb",
}
MEASUREMENTS = {
    "wall_seconds": ("Wall s", 1),
    "telemetry.sampled_peak_gpu_device_used_bytes": ("Sampled device GiB", 2**30),
    "telemetry.sampled_peak_process_tree_rss_bytes": ("Sampled tree RSS GiB", 2**30),
    "telemetry.sampled_peak_process_tree_pss_bytes": ("Sampled tree PSS GiB", 2**30),
    "telemetry.process_tree_read_bytes": ("Physical read GiB", 2**30),
    "telemetry.process_tree_write_bytes": ("Physical write GiB", 2**30),
}


def canonical(value: Any) -> Any:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> Any:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def cell(value: Any) -> Any:
    if value is None:
        return "unknown"
    if isinstance(value, (dict, list, tuple)):
        value = canonical(value)
    return str(value).replace("|", "&#124;").replace("\n", " ")


def number(value: Any) -> Any:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fmt(value: Any) -> Any:
    return f"{value:.6g}" if number(value) else "unknown"


def lookup(data: Any, dotted: Any) -> Any:
    for key in dotted.split("."):
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def numeric_leaves(data: Any, prefix: str = "") -> Any:
    result = {}
    if isinstance(data, dict):
        for key, value in data.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                result.update(numeric_leaves(value, name))
            elif number(value):
                result[name] = value
    return result


def environment_conflicts(environments: Any) -> Any:
    """Compare only overlapping identity/version fields, never missing coverage.

    Environment collectors may add hardware details without contradicting a
    smaller record. Package collector aliases share a namespace. Runtime GPU
    utilization and other sampled counters are deliberately not identities.
    """
    observed = defaultdict(set)

    def collect(key: str, value: Any) -> None:
        if isinstance(value, dict):
            for name, item in value.items():
                collect(f"{key}.{name}", item)
        elif value is not None:
            observed[key].add(canonical(value))

    for environment in environments:
        python = environment.get("python")
        if isinstance(python, str):
            match = re.match(r"\d+\.\d+(?:\.\d+)?", python.strip())
            python = match.group() if match else python
        collect("python", python)
        for field in ("platform", "hardware"):
            collect(field, environment.get(field))
        for alias in ("packages", "versions"):
            packages = environment.get(alias)
            if isinstance(packages, dict):
                for name, value in packages.items():
                    collect("packages." + re.sub(r"[-_.]+", "-", name).lower(), value)
    return sorted(key for key, values in observed.items() if len(values) > 1)


def recorded_fields(data: Any, prefix: str = "", short_list_limit: int = 12) -> Any:
    """Flatten recorded evidence; keep lengthy arrays in their linked source."""
    rows: list[tuple[str, Any]] = []
    if isinstance(data, dict):
        if not data:
            rows.append((prefix or "record", {}))
        for key, value in data.items():
            name = f"{prefix}.{key}" if prefix else key
            rows.extend(recorded_fields(value, name, short_list_limit))
    elif isinstance(data, list):
        if len(data) > short_list_limit:
            rows.append((prefix or "record", f"{len(data)} items; see source"))
        elif all(not isinstance(value, (dict, list)) for value in data):
            rows.append((prefix or "record", data))
        else:
            if not data:
                rows.append((prefix or "record", []))
            for index, value in enumerate(data):
                rows.extend(recorded_fields(value, f"{prefix}[{index}]", short_list_limit))
    else:
        rows.append((prefix or "record", "null" if data is None else data))
    return rows


def link(path: str | Path, destination: str | Path, label: Any = None) -> Any:
    path = Path(path).resolve()
    relative = os.path.relpath(path, Path(destination).resolve().parent)
    return f"[{cell(label or path.name)}]({quote(relative, safe='/._-')})"


def table(headers: Any, rows: Any) -> Any:
    return ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"] + [
        "| " + " | ".join(cell(v) for v in row) + " |" for row in rows
    ]


def read_json(path: str | Path) -> Any:
    with Path(path).open() as stream:
        return json.load(stream)


def load_trials(roots: Any) -> Any:
    """Only load small trial.json files; deduplicate overlapping explicit roots."""
    records, seen = [], set()
    for root in roots:
        root = Path(root).resolve()
        if not root.is_dir():
            raise ValueError(f"Benchmark root is not a directory: {root}")
        for path in sorted(root.rglob("trial.json")):
            if path.resolve() not in seen:
                data = read_json(path)
                records.append({"path": path.resolve(), "root": root, "data": data})
                seen.add(path.resolve())
    return records


def workload(data: Any) -> Any:
    result = {key: data.get(key) for key in MATCH_FIELDS}
    command = data.get("command") or []
    # The benchmark worker prefixes the inference command with its own seed and
    # script arguments. Only compare flags belonging to the inference script.
    options = None
    if "--num_frames" in command:
        options = {}
        tokens = iter(command[command.index("--num_frames") :])
        key = None
        ignored = {
            "--out",
            "--offload",
            "--ram-budget-gb",
            "--vram-budget-gb",
            "--scratch-budget-gb",
            "--scratch-dir",
        }
        for token in tokens:
            if token.startswith("--"):
                key, separator, value = token.partition("=")
                if key not in ignored:
                    options[key] = [value] if separator else []
            elif key and key not in ignored:
                options[key].append(token)
    result["command_settings"] = options
    return result


def source_version(data: Any) -> Any:
    return {key: data.get(key) for key in ("code_hashes", "lockfile_sha256", "wrapper_sha256")}


def aggregate(records: Any) -> Any:
    """Keep changed settings/source versions separate, without rewriting summaries."""
    groups = defaultdict(list)
    for record in records:
        data = record["data"]
        key = (
            str(record["root"]),
            data.get("mode"),
            canonical(workload(data)),
            canonical(source_version(data)),
            canonical(lookup(data, "metrics.settings")),
        )
        groups[key].append(record)
    result: list[dict[str, Any]] = []
    for members in groups.values():
        valid = [
            r
            for r in members
            if r["data"].get("status") == "complete"
            and number(r["data"].get("wall_seconds"))
            and r["data"]["wall_seconds"] > 0
        ]
        values = {}
        counts = {}
        for field in MEASUREMENTS:
            observations = [
                lookup(r["data"], field) for r in valid if number(lookup(r["data"], field))
            ]
            values[field] = median(observations) if observations else None
            counts[field] = len(observations)
        result.append(dict(members=members, valid=valid, values=values, counts=counts))
    return result


def ratios(groups: Any) -> Any:
    """Within a campaign, match recorded workload and wrapper/environment versions.

    Unknown required settings or multiple matching baseline versions suppress a
    ratio. Placement budgets are displayed, but are not workload settings.
    Additional shared inference settings must agree when both sides record them.
    """
    results = []
    for group in groups:
        candidate = group["members"][0]
        data = candidate["data"]
        if data.get("mode") == "existing":
            continue
        reason = None
        matches = []
        if (
            any(data.get(key) is None for key in MATCH_FIELDS)
            or not data.get("wrapper_sha256")
            or not data.get("lockfile_sha256")
        ):
            reason = "required workload/version metadata missing"
        else:
            for baseline in groups:
                reference = baseline["members"][0]
                ref = reference["data"]
                if (
                    reference["root"] != candidate["root"]
                    or ref.get("mode") != "existing"
                    or workload(ref) != workload(data)
                ):
                    continue
                if any(ref.get(k) != data.get(k) for k in ("wrapper_sha256", "lockfile_sha256")):
                    continue
                a, b = lookup(ref, "metrics.settings") or {}, lookup(data, "metrics.settings") or {}
                if any(a[k] != b[k] for k in a.keys() & b.keys() - PLACEMENT_SETTINGS):
                    continue
                matches.append(baseline)
            if len(matches) != 1:
                reason = "no unique settings/version-matched existing group"
            elif not matches[0]["valid"] or not group["valid"]:
                reason = "successful timings unavailable"
        value = (
            None
            if reason
            else matches[0]["values"]["wall_seconds"] / group["values"]["wall_seconds"]
        )
        results.append((group, value, reason))
    return results


def render_report(
    benchmark_roots: Any,
    parity_files: Any = (),
    environment: Any = None,
    analysis: Any = (),
    destination: str | Path = Path("benchmark-report.md"),
) -> Any:
    records = load_trials(benchmark_roots)
    groups = aggregate(records)
    lines = [
        "# Benchmark technical report",
        "",
        "Measurements below are derived from the linked trial records. Unknown metadata is shown explicitly. Supplied analysis is reproduced separately.",
        "",
        "## Inputs",
        "",
    ]
    roots = list(dict.fromkeys(Path(p).resolve() for p in benchmark_roots))
    rows = []
    for root in roots:
        selected = [r for r in records if r["root"] == root]
        stamps = sorted(
            str(r["data"]["started_at"]) for r in selected if r["data"].get("started_at")
        )
        rows.append(
            (
                link(root, destination, str(root)),
                len(selected),
                stamps[0] if stamps else None,
                stamps[-1] if stamps else None,
            )
        )
    lines += table(
        ["Benchmark root", "Trial records", "First recorded start", "Last recorded start"], rows
    )
    warnings = []
    if not records:
        warnings.append("No trial records found in the selected roots.")
    for field in (*MATCH_FIELDS, "started_at", "code_hashes", "lockfile_sha256", "wrapper_sha256"):
        missing = sum(r["data"].get(field) is None for r in records)
        if missing:
            warnings.append(f"Missing {field}: {missing} trial record(s).")
    versions = defaultdict(set)
    for r in records:
        d = r["data"]
        versions[(str(r["root"]), d.get("scene"), d.get("mode"), d.get("scenario"))].add(
            canonical(source_version(d))
        )
    for key, values in versions.items():
        if len(values) > 1:
            warnings.append(
                f"Mixed source/lockfile/wrapper versions within {cell(key)}; versions are reported separately."
            )
    for field in ("lockfile_sha256", "wrapper_sha256"):
        if len({canonical(r["data"].get(field)) for r in records}) > 1:
            warnings.append(
                f"Mixed {field} across selected records; ratios require matching versions."
            )
    lines += [
        "",
        "## Environment",
        "",
        "Hardware identity is unknown unless explicitly identified in the environment records. Device capacity is reported only as recorded capacity.",
        "",
    ]
    env_paths = list(
        dict.fromkeys(
            ([Path(environment).resolve()] if environment else [])
            + [root / "environment.json" for root in roots if (root / "environment.json").is_file()]
        )
    )
    if not env_paths:
        lines += ["Environment metadata: unknown. Hardware identity: unknown."]
    envs = []
    for path in env_paths:
        data = read_json(path)
        envs.append(data)
        lines += [
            f"Recorded environment: {link(path, destination, str(path))}",
            "",
            "```json",
            json.dumps(data, indent=2, sort_keys=True),
            "```",
            "",
        ]
    conflicts = environment_conflicts(envs)
    if conflicts:
        warnings.append(
            "Mixed recorded environments in comparable fields: "
            + ", ".join(conflicts)
            + "; ratios are limited to each benchmark root."
        )
    lines += ["", "## Record warnings", ""] + (
        ["- " + w for w in warnings]
        or ["No mixed-version or missing required trial metadata detected."]
    )
    lines += [
        "",
        "## Aggregate measurements",
        "",
        "Each row has identical recorded workload, mode, source hashes and metric settings within one root. Medians use successful trials with finite positive wall time; failed and OOM trials remain in the status counts. Each measurement includes its available sample count. Missing measurements are unknown; sampled memory peaks are not allocator peaks.",
        "",
    ]
    headers = ["Group", "Root / scene / mode", "Statuses", "Settings", "Source version"] + [
        label + " (n)" for label, _ in MEASUREMENTS.values()
    ]
    rows = []
    for i, group in enumerate(groups, 1):
        group["id"] = f"G{i}"
        r = group["members"][0]
        d = r["data"]
        values = [
            f"{fmt(group['values'][k] / divisor) if group['values'][k] is not None else 'unknown'} ({group['counts'][k]})"
            for k, (_, divisor) in MEASUREMENTS.items()
        ]
        rows.append(
            (
                group["id"],
                f"{r['root'].name} / {d.get('scene', 'unknown')} / {d.get('mode', 'unknown')}",
                dict(Counter(x["data"].get("status", "unknown") for x in group["members"])),
                workload(d),
                digest(source_version(d)),
                *values,
            )
        )
    lines += table(headers, rows)
    lines += [
        "",
        "### Settings-matched timing ratios",
        "",
        "Ratio = median successful existing wall time / median successful candidate wall time. Values above one indicate shorter candidate wall time. Matching requires all listed workload fields, lockfile and seed-wrapper hashes, one existing version, and agreement of shared non-placement metric settings. Ratios do not compare failure duration with successful runtime. Raw commands and complete settings remain linked below.",
        "",
    ]
    lines += table(
        ["Candidate group", "Existing / candidate ratio", "Unavailable reason"],
        [(g["id"], fmt(value), reason or "—") for g, value, reason in ratios(groups)],
    )
    lines += [
        "",
        "## Stage metrics and configured settings",
        "",
        "Numeric stage fields are median values across the successful trials where recorded, with availability counts. Field names retain their recorded units; allocated and reserved CUDA peaks remain distinct.",
        "",
    ]
    rows = []
    for group in groups:
        members = group["members"]
        settings = lookup(members[0]["data"], "metrics.settings")
        lines += [
            f"{group['id']} metric settings (including recorded budgets): `{cell(settings)}`",
            "",
        ]
        flattened = [
            numeric_leaves(lookup(r["data"], "metrics.stages") or {}) for r in group["valid"]
        ]
        for key in sorted(set().union(*(set(d) for d in flattened))):
            values = [d[key] for d in flattened if key in d]
            rows.append((group["id"], key, fmt(median(values)), len(values)))
    lines += table(["Group", "Stage field", "Median", "n"], rows)
    lines += [
        "",
        "## Trial provenance",
        "",
        "Trial links contain recorded commands, settings, input fingerprints, per-file source hashes, status and metrics. The source-version digest hashes the recorded code-hash dictionary, lockfile hash and wrapper hash; it does not rehash the working tree.",
        "",
    ]
    rows = []
    for group in groups:
        for r in group["members"]:
            d = r["data"]
            log = Path(d["log"]) if d.get("log") else None
            telemetry = r["path"].parent / "telemetry.jsonl"
            rows.append(
                (
                    group["id"],
                    link(r["path"], destination, str(r["path"])),
                    d.get("started_at"),
                    d.get("status"),
                    link(log, destination) if log else None,
                    link(telemetry, destination) if telemetry.is_file() else None,
                    d.get("input_sha256"),
                    d.get("lockfile_sha256"),
                    d.get("wrapper_sha256"),
                )
            )
    lines += table(
        [
            "Group",
            "Trial source",
            "Started at",
            "Status",
            "Log",
            "Telemetry",
            "Input SHA256",
            "Lockfile SHA256",
            "Wrapper SHA256",
        ],
        rows,
    )
    lines += [
        "",
        "## Numerical parity records",
        "",
        "Parity results and policies below are reproduced from the selected JSON records; this renderer does not recompute array comparisons.",
        "",
    ]
    if not parity_files:
        lines += ["Parity records: unknown (none supplied)."]
    for path in parity_files:
        path = Path(path).resolve()
        data = read_json(path)
        lines += [f"Source: {link(path, destination, str(path))}", ""]
        lines += table(["Recorded field", "Value"], recorded_fields(data)) + [""]
    for path in analysis:
        path = Path(path).resolve()
        lines += [
            "",
            "## Supplied analysis",
            "",
            f"Verbatim source: {link(path, destination, str(path))}",
            "",
            path.read_text(),
        ]
    return "\n".join(lines) + "\n"


def main(argv: Any = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--parity-files", nargs="*", type=Path, default=[])
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--analysis", nargs="*", type=Path, default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = render_report(
        args.benchmark_roots, args.parity_files, args.environment, args.analysis, args.out
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(args.out)


if __name__ == "__main__":
    main()
