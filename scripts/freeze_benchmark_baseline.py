#!/usr/bin/env python
"""Freeze the complete current source tree before offload changes.

    uv run python scripts/freeze_benchmark_baseline.py

Weights, data caches, the virtual environment and Git metadata stay shared/read-only.
Use this checkout's Python with the frozen scripts and explicit checkpoint/input paths.
"""
import hashlib
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "outputs/benchmarks/baseline-source"
EXCLUDE = {".git", ".venv", "outputs", "checkpoints", "cache", "data", "__pycache__", ".pytest_cache", ".codex", ".agents"}


def freeze(destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Snapshot already exists; refusing to overwrite {destination}")
    destination.mkdir(parents=True)
    files = {}
    for directory, names, filenames in os.walk(ROOT):
        names[:] = sorted(n for n in names if n not in EXCLUDE and not n.endswith(".egg-info"))
        for name in sorted(filenames):
            source = Path(directory) / name
            relative = source.relative_to(ROOT)
            if not source.is_file():
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            files[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = dict(base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    files=files, python=str(ROOT / ".venv/bin/python"), checkpoints=str(ROOT / "checkpoints"))
    (destination / "snapshot.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Frozen {len(files)} source files at {destination}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEST)
    freeze(parser.parse_args().out)


if __name__ == "__main__":
    main()
