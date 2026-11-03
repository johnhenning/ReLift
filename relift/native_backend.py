"""Explicit optional native builds; never compile or silently fall back in inference."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType


def digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cpu(directory: str | Path | None = None) -> ModuleType:
    directory = directory or os.environ.get("RELIFT_NATIVE_BUILD")
    if not directory:
        raise RuntimeError(
            "Native backend requires RELIFT_NATIVE_BUILD; run scripts/build_native.py first"
        )
    root = Path(directory)
    manifest = json.loads((root / "build.json").read_text())
    source_root = Path(__file__).resolve().parents[1]
    for name, value in manifest["sources"].items():
        if digest(source_root / name) != value:
            raise RuntimeError(f"Native build source mismatch: {name}")
    binary = root / manifest["binary"]
    if digest(binary) != manifest["binary_sha256"]:
        raise RuntimeError("Native extension checksum mismatch")
    spec = importlib.util.spec_from_file_location("_relift_cpu", binary)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load native module: {binary}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
