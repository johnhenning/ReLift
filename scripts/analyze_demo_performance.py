#!/usr/bin/env python
"""Summarize existing logs and code fingerprints without running inference.

    uv run python scripts/analyze_demo_performance.py

This is an observational inventory, not a controlled benchmark. It never runs
models, changes caches, or retries failed videos. Timings retain their original
log precision and do not include all end-to-end work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from relift.records.reporting import (
        ArrayMemoryEstimate,
        DemoInventoryRow,
        DemoPerformanceReport,
    )

REPO = Path(__file__).resolve().parents[1]
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(root: Path) -> list[DemoInventoryRow]:
    state = json.loads((root / "batch.json").read_text())
    rows = []
    for name, record in state["videos"].items():
        directory = root / name
        log = directory / "inference.log"
        text = ANSI.sub("", log.read_text()) if log.exists() else ""
        row: DemoInventoryRow = dict(
            name=name,
            status=record["status"],
            source=record.get("source", {}),
            export=record.get("result", {}),
        )
        if log.exists():
            row["log"] = str(log.relative_to(REPO)) if log.is_relative_to(REPO) else str(log)
            row["log_sha256"] = sha256(log)
        patterns = dict(
            da3_stage_seconds=r"da3: grid \d+x\d+, ([\d.]+)s",
            da3_forward_seconds=r"Model Forward Pass Done\. Time: ([\d.]+) seconds",
            alltracker_stage_seconds=r"alltracker: ([\d.]+)s",
            corrector_stage_seconds=r"corrector .*? in ([\d.]+)s;",
        )
        timings = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                timings[key] = float(match[1])
        row["logged_timings"] = timings
        if all(
            k in timings
            for k in ("da3_stage_seconds", "alltracker_stage_seconds", "corrector_stage_seconds")
        ):
            row["sum_of_logged_stage_seconds"] = sum(
                timings[k]
                for k in (
                    "da3_stage_seconds",
                    "alltracker_stage_seconds",
                    "corrector_stage_seconds",
                )
            )
        if "CUDA out of memory" in text:
            row["failure"] = dict(type="cuda_oom", detail=text.splitlines()[-1])
        for filename in ("tracks.npz", "trajectories.html"):
            file = directory / filename
            if file.exists():
                if filename == "tracks.npz":
                    row["tracks.npz_bytes"] = file.stat().st_size
                else:
                    row["trajectories.html_bytes"] = file.stat().st_size
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=REPO / "outputs/demo-full")
    ap.add_argument(
        "--out", type=Path, default=REPO / "outputs/analysis/current-demo-performance.json"
    )
    args = ap.parse_args()
    rows = summarize(args.root)
    paths = [
        REPO / p
        for p in (
            "pyproject.toml",
            "uv.lock",
            "configs/providers.yaml",
            "configs/eval/table1.yaml",
            "configs/weights.yaml",
            "packages/frontends/upstreams.json",
        )
    ]
    paths += list((REPO / "relift").rglob("*.py"))
    paths += [
        REPO / "scripts" / name
        for name in ("demo_video.py", "plotly_demo.py", "generate_all_views.py")
    ]
    fingerprints = {str(p.relative_to(REPO)): sha256(p) for p in sorted(set(paths)) if p.is_file()}
    memory: list[ArrayMemoryEstimate] = []
    for total in (84, 100, 206, 294):
        cells = total * 504 * 280
        memory.append(
            dict(
                frames=total,
                grid=[504, 280],
                xyz_float32_mib=cells * 12 / 2**20,
                raw_float16_mib=cells * 6 / 2**20,
                xyz_raw_mask_images_mib=cells * 22 / 2**20,
            )
        )
    report: DemoPerformanceReport = dict(
        kind="observational_inventory_not_controlled_benchmark",
        captured_at=datetime.now(UTC).isoformat(),
        python=platform.python_version(),
        versions={p: version(p) for p in ("torch", "torchvision", "numpy", "plotly")},
        base_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        working_tree_hashes=fingerprints,
        videos=rows,
        theoretical_array_memory=memory,
        timing_limits="Coarse single-run stage logs; omit decode, some I/O, process startup, and HTML export. No peak-memory telemetry was recorded for successful stages.",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Inventory: {args.out}")
    for row in rows:
        print(
            f"{row['name']:20} {row['status']:10} frames={row['source'].get('frames')} stages={row['logged_timings']}"
        )
    print("Theoretical storage at a 504x280 grid (MiB):")
    for row in memory:
        print(
            f"{row['frames']:3} frames: xyz={row['xyz_float32_mib']:.1f}, core payload={row['xyz_raw_mask_images_mib']:.1f}"
        )


if __name__ == "__main__":
    main()
