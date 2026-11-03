"""Paths: ``configs/paths.yaml`` relative to the repository, overridable by environment variables and CLI flags."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
KEYS = ("data_root", "cache_root", "checkpoints_root", "outputs_root")


def paths(overrides: Any = None) -> dict[str, Path]:
    """Resolved absolute paths.  Precedence: explicit overrides > RELIFT_<KEY> environment variables > paths.yaml."""
    cfg = yaml.safe_load(open(REPO / "configs" / "paths.yaml")) or {}
    out = {}
    for k in KEYS:
        v = (
            (overrides or {}).get(k)
            or os.environ.get(f"RELIFT_{k.upper()}")
            or cfg.get(k, k.split("_")[0])
        )
        p = Path(os.path.expanduser(str(v)))
        out[k] = p if p.is_absolute() else REPO / p
    return out
