#!/usr/bin/env python
"""Freeze the complete current source tree before offload changes.

    uv run python scripts/freeze_benchmark_baseline.py

Weights, data caches, the virtual environment and Git metadata stay shared/read-only.
Use this checkout's Python with the frozen scripts and explicit checkpoint/input paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from relift.records.artifacts import SourceSnapshot

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "outputs/benchmarks/baseline-source"
EXCLUDE = {
    ".git",
    ".venv",
    "outputs",
    "checkpoints",
    "cache",
    "data",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".pyrefly_cache",
    ".codex",
    ".agents",
}
NESTED_EXCLUDE = EXCLUDE - {"outputs", "checkpoints", "cache", "data"}


def freeze(destination: str | Path) -> Path:
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Snapshot already exists; refusing to overwrite {destination}")
    destination.mkdir(parents=True)
    files = {}
    for directory, names, filenames in os.walk(ROOT):
        excluded = EXCLUDE if Path(directory) == ROOT else NESTED_EXCLUDE
        names[:] = sorted(n for n in names if n not in excluded and not n.endswith(".egg-info"))
        for name in sorted(filenames):
            if name in EXCLUDE:
                continue
            source = Path(directory) / name
            relative = source.relative_to(ROOT)
            if not source.is_file():
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            files[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest: SourceSnapshot = dict(
        base_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        files=files,
        python=str(ROOT / ".venv/bin/python"),
        checkpoints=str(ROOT / "checkpoints"),
    )
    (destination / "snapshot.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Frozen {len(files)} source files at {destination}")
    return destination


def freeze_revision(destination: str | Path, revision: str, repository: str | Path = ROOT) -> Path:
    """Materialize committed source without changing the checkout or Git index."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite source snapshot: {destination}")
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", f"{revision}^{{commit}}"], text=True
    ).strip()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="source-snapshot-", dir=destination.parent
    ) as temporary:
        temporary = Path(temporary)
        archive, tree = temporary / "source.tar", temporary / "tree"
        tree.mkdir()
        with archive.open("wb") as stream:
            subprocess.run(
                ["git", "-C", str(repository), "archive", "--format=tar", commit],
                stdout=stream,
                check=True,
            )
        files = {}
        with tarfile.open(archive) as members:
            for member in members:
                relative = Path(member.name)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not (member.isdir() or member.isfile())
                ):
                    raise ValueError(f"Unsupported archive member: {member.name}")
                if member.isdir():
                    continue
                target = tree / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted = members.extractfile(member)
                assert extracted is not None
                with extracted as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(member.mode & 0o777)
                files[member.name] = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest: SourceSnapshot = dict(
            base_commit=commit,
            git_revision=commit,
            files=files,
            python=str(ROOT / ".venv/bin/python"),
            checkpoints=str(ROOT / "checkpoints"),
        )
        (tree / "snapshot.json").write_text(json.dumps(manifest, indent=2) + "\n")
        tree.rename(destination)
    print(f"Frozen {len(files)} committed source files at {destination} ({commit[:12]})")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEST)
    parser.add_argument("--revision", help="Snapshot a Git commit/ref instead of the working tree")
    parser.add_argument("--repository", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.revision:
        freeze_revision(args.out, args.revision, args.repository)
    else:
        freeze(args.out)


if __name__ == "__main__":
    main()
