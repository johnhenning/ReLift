"""What produced an output: GPU model, torch version, date.  The GPU model matters (bfloat16 arithmetic differs between GPU models), so every
prepared window and every prediction file carries it and ``make_table1.py`` reports the models behind a table."""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch


def machine(device: str | torch.device = "cuda") -> dict[str, Any]:
    info: dict[str, Any] = dict(host=platform.node(), date=time.strftime("%Y-%m-%dT%H:%M:%S"))
    try:
        import torch

        info["torch"] = torch.__version__
        info["gpu"] = (
            torch.cuda.get_device_name(device)
            if str(device).startswith("cuda") and torch.cuda.is_available()
            else "cpu"
        )
    except Exception:  # noqa: BLE001
        info["gpu"] = None
    return info


def note(path: str | Path, key: str, device: str | torch.device = "cuda", **extra: Any) -> None:
    """Merge ``{key: machine() + extra}`` into the JSON file at ``path`` (one file per window directory)."""
    d = json.load(open(path)) if os.path.exists(path) else {}
    d[key] = dict(machine(device), **extra)
    tmp = str(path) + ".tmp"
    json.dump(d, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def gpus_of_windows(windows_root: Any) -> dict[str, list[str]]:
    """-> {provider: sorted set of GPU models} over every ``provenance.json`` below ``windows_root``."""
    out = {}
    for sub in sorted(os.listdir(windows_root)) if os.path.isdir(windows_root) else []:
        for clip in sorted(os.listdir(os.path.join(windows_root, sub))):
            f = os.path.join(windows_root, sub, clip, "provenance.json")
            if os.path.exists(f):
                for k, v in json.load(open(f)).items():
                    out.setdefault(k, set()).add(v.get("gpu"))
    return {k: sorted(str(g) for g in v) for k, v in out.items()}


def gpus_of_preds(preds_root: Any, variants: Any) -> dict[str, list[str]]:
    """-> {variant: sorted set of GPU models} from the ``gpu`` field of the prediction files (absent in old files)."""
    import numpy as np

    out = {}
    for var in variants:
        root = os.path.join(preds_root, var)
        found = set()
        for sub in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            for f in sorted(os.listdir(os.path.join(root, sub))):
                if f.endswith(".npz"):
                    with np.load(os.path.join(root, sub, f)) as z:
                        found.add(str(z["gpu"]) if "gpu" in z.files else "unrecorded")
        if found:
            out[var] = sorted(found)
    return out
