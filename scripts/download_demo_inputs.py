#!/usr/bin/env python
"""List or download official demo inputs, recording hashes and video lengths.

    uv run python scripts/download_demo_inputs.py --list
    uv run python scripts/download_demo_inputs.py running tennis motocross-bumps
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request

import cv2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from relift.config import paths


def catalog():
    return json.loads((REPO / "assets/demo/catalog.json").read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_input(name):
    entry = catalog()[name]
    cache = paths()["cache_root"] / "demo_inputs"
    cache.mkdir(parents=True, exist_ok=True)
    path = REPO / entry["local_file"] if "local_file" in entry else cache / f"{name}.mp4"
    record = cache / f"{name}.json"
    # Once fetched, keep a local pin even when the upstream catalog has no published hash.
    previous = json.loads(record.read_text()) if record.exists() else {}
    expected = entry.get("sha256") or previous.get("sha256")
    if not path.exists() or (expected and sha256(path) != expected):
        temp = path.with_suffix(".mp4.part")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            print(f"Downloading {entry['source_url']}", flush=True)
            with urllib.request.urlopen(entry["source_url"], timeout=120) as source, temp.open("wb") as target:
                shutil.copyfileobj(source, target)
            if expected and sha256(temp) != expected:
                raise ValueError(f"{name}: source checksum changed; inspect the input before updating its pin")
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot decode {path}")
        metadata = dict(name=name, source_url=entry["source_url"], sha256=sha256(path),
                        frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), fps=cap.get(cv2.CAP_PROP_FPS),
                        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()
    metadata["duration_seconds"] = metadata["frames"] / metadata["fps"] if metadata["fps"] else None
    record.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{name}: {metadata['frames']} frames, {metadata['fps']:g} fps -> {path}", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="*", metavar="SCENE")
    ap.add_argument("--list", action="store_true", help="show available official inputs without downloading")
    args = ap.parse_args()
    unknown = set(args.videos) - set(catalog())
    if unknown:
        ap.error(f"unknown scenes: {', '.join(sorted(unknown))}; use --list")
    if args.list or not args.videos:
        for name, entry in catalog().items():
            print(f"{name:20} {entry['description']}")
    else:
        for name in args.videos:
            fetch_input(name)


if __name__ == "__main__":
    main()
