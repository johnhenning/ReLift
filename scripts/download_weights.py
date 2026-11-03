#!/usr/bin/env python
"""Download every weight file into <checkpoints_root> (configs/weights.yaml lists them).

    python scripts/download_weights.py [--only da3 alltracker ...] [--checkpoints_root DIR]

VGGT-Omega is gated on Hugging Face (CoWTracker was not at the time of writing, but may become so): request access
on the model page, then log in once with `hf auth login` (or set HF_TOKEN) and rerun.  A fine-grained token
must have the "read gated repositories" permission.  Files already present and matching the
recorded md5 are skipped.  The corrector checkpoint is fetched from the release asset URL in the config, or you may
place the file under <checkpoints_root> yourself.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from relift.config import REPO, paths  # noqa: E402

ACCESS = {
    "vggt_omega": "https://huggingface.co/facebook/VGGT-Omega",
    "cowtracker": "https://huggingface.co/facebook/cowtracker",
}


def md5(path: str | Path, chunk: int = 1 << 24) -> Any:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def fetch(name: str, w: dict[str, Any], root: str | Path) -> Any:
    from huggingface_hub import hf_hub_download, snapshot_download

    dst_dir = os.path.join(root, w.get("dir", ""))
    if w["kind"] == "hf_snapshot":
        path = snapshot_download(w["repo"], revision=w.get("revision"), local_dir=dst_dir)
        if w.get("md5_model_safetensors"):
            got = md5(os.path.join(path, "model.safetensors"))
            return path, (
                "md5 ok"
                if got == w["md5_model_safetensors"]
                else f"MD5 MISMATCH {got} != {w['md5_model_safetensors']}"
            )
        return path, "directory"
    if w["kind"] == "hf_file":
        path = hf_hub_download(w["repo"], w["file"], revision=w.get("revision"), local_dir=dst_dir)
    else:  # plain URL
        path = os.path.join(root, w["file"])
        if not w.get("url"):
            if os.path.exists(path):
                return path, "present"
            return None, "no URL yet: place the file under checkpoints_root yourself"
        if not os.path.exists(path):
            urllib.request.urlretrieve(w["url"], path + ".part")
            os.replace(path + ".part", path)
    if w.get("md5"):
        got = md5(path)
        if got != w["md5"]:
            return path, f"MD5 MISMATCH {got} != {w['md5']}"
        return path, "md5 ok"
    return path, "ok"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--checkpoints_root", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(REPO / "configs" / "weights.yaml"))
    root = a.checkpoints_root or str(paths()["checkpoints_root"])
    os.makedirs(root, exist_ok=True)
    failed = []
    for name, w in cfg.items():
        if a.only and name not in a.only:
            continue
        try:
            path, status = fetch(name, w, root)
            print(f"{name:12s} {status:40s} {path}", flush=True)
            if path is None or "MISMATCH" in status:
                failed.append(name)
        except Exception as e:  # noqa: BLE001
            msg = str(e).splitlines()[0][:160]
            if name in ACCESS and (
                "401" in msg
                or "403" in msg
                or "gated" in msg.lower()
                or "GatedRepo" in type(e).__name__
            ):
                print(
                    f"{name:12s} ACCESS REQUIRED: request access at {ACCESS[name]}, then `hf auth login` and rerun",
                    flush=True,
                )
            else:
                print(f"{name:12s} FAILED: {msg}", flush=True)
            failed.append(name)
    print("all weights present" if not failed else f"missing: {failed}")
    sys.exit(1 if failed else 0)
