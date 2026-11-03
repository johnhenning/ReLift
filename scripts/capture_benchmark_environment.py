#!/usr/bin/env python
"""Record hardware and verify the exact shared checkpoint bytes outside timing runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def hashes(path: Path) -> Any:
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            md5.update(block)
            sha.update(block)
    return dict(
        md5=md5.hexdigest(),
        sha256=sha.hexdigest(),
        bytes=path.stat().st_size,
        mtime_ns=path.stat().st_mtime_ns,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "outputs/benchmarks/shared-environment.json"
    )
    parser.add_argument("--weights-config", type=Path, default=ROOT / "configs/weights.yaml")
    parser.add_argument("--checkpoints-root", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--weights", nargs="+", default=["relift", "da3", "alltracker"])
    args = parser.parse_args()
    config = yaml.safe_load(args.weights_config.read_text())
    paths = {
        name: args.checkpoints_root
        / config[name].get("dir", "")
        / config[name].get("file", "model.safetensors")
        for name in args.weights
    }
    checkpoints = {}
    for name, path in paths.items():
        row = hashes(path)
        row["expected_md5"] = config[name].get("md5", config[name].get("md5_model_safetensors"))
        row["matches_pinned_checkpoint"] = row["md5"] == row["expected_md5"]
        row["file"] = str(path.resolve())
        checkpoints[name] = row
        if not row["matches_pinned_checkpoint"]:
            raise ValueError(f"Checkpoint differs from pinned bytes: {path}")
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        text=True,
    ).strip()
    storage = json.loads(
        subprocess.check_output(
            [
                "lsblk",
                "--json",
                "--exclude",
                "7",
                "--output",
                "NAME,MODEL,TRAN,SIZE,TYPE,MOUNTPOINTS",
            ],
            text=True,
        )
    )
    processors = [
        dict(line.split(":", 1) for line in block.splitlines() if ":" in line)
        for block in Path("/proc/cpuinfo").read_text().strip().split("\n\n")
    ]
    models = sorted(
        {
            value.strip()
            for processor in processors
            for key, value in processor.items()
            if key.strip() == "model name"
        }
    )
    memory = {
        line.split(":", 1)[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.split(":", 1)[0] in {"MemTotal", "SwapTotal"}
    }
    result = dict(
        captured_at=datetime.now(UTC).isoformat(),
        platform=platform.platform(),
        python=sys.version,
        gpu=gpu,
        storage=storage,
        packages={
            name: version(name)
            for name in ("torch", "torchvision", "numpy", "plotly", "relift-frontends")
        },
        hardware=dict(cpu_models=models, logical_processors=len(processors), memory_bytes=memory),
        checkpoints=checkpoints,
        cache_policy="Local pinned checkpoints; OS page cache retained. Run this inventory outside timing trials.",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
