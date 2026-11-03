#!/usr/bin/env python
"""Download the 200 WorldTrack evaluation clips (4 subsets x 50) and verify them against the md5 manifest.

    python scripts/download_worldtrack.py [--subsets adt pstudio po ds] [--limit N] [--verify_only] [--workers 4]

Source: the WorldTrack release used by 4RC, a public Google Drive folder (``manifests/worldtrack_drive.json``
holds the per-file ids and sizes; the folder is linked from 4RC's README).  Files land in
``<data_root>/worldtrack/<subset>_mini/<clip>.npz``; a file whose size and md5 match the manifest is skipped.
Google Drive throttles bulk downloads at times: rerun until every file is reported ``ok``.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from relift.config import REPO, paths  # noqa: E402
from relift.data.worldtrack import SUBSETS  # noqa: E402

DRIVE = REPO / "manifests" / "worldtrack_drive.json"
MD5 = REPO / "manifests" / "worldtrack_md5.json"
URL = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"


def md5(path: str | Path, chunk: int = 1 << 24) -> Any:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def fetch(job: Any) -> Any:
    sub, e, root, want_md5, verify_only = job
    dst = os.path.join(root, sub, e["name"])
    key = f"{sub}/{e['name']}"
    if os.path.exists(dst) and os.path.getsize(dst) == e["size"]:
        return key, ("ok" if md5(dst) == want_md5 else "MD5 MISMATCH")
    if verify_only:
        return key, "missing"
    err = "unknown"
    for _ in range(4):
        try:
            r = urllib.request.urlopen(
                urllib.request.Request(
                    URL.format(id=e["id"]), headers={"User-Agent": "Mozilla/5.0"}
                ),
                timeout=120,
            )
            tmp = dst + ".part"
            with open(tmp, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            if os.path.getsize(tmp) == e["size"]:
                os.replace(tmp, dst)
                return key, ("ok" if md5(dst) == want_md5 else "MD5 MISMATCH")
            err = f"size {os.path.getsize(tmp)} != {e['size']} (Drive quota page?)"
        except Exception as ex:  # noqa: BLE001
            err = repr(ex)[:120]
    return key, f"FAILED {err}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS))
    ap.add_argument("--root", default=None, help="default <data_root>/worldtrack")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--verify_only", action="store_true")
    a = ap.parse_args()
    root = a.root or str(paths()["data_root"] / "worldtrack")
    drive, want = json.load(open(DRIVE)), json.load(open(MD5))["files"]
    jobs = []
    for s in a.subsets:
        sub = SUBSETS.get(s, s)
        os.makedirs(os.path.join(root, sub), exist_ok=True)
        jobs += [
            (sub, e, root, want[f"{sub}/{e['name']}"], a.verify_only)
            for e in drive[sub][: a.limit or None]
        ]
    with cf.ThreadPoolExecutor(a.workers) as ex:
        res = dict(ex.map(fetch, jobs))
    bad = {k: v for k, v in res.items() if v != "ok"}
    for k, v in sorted(bad.items()):
        print(f"{k}: {v}")
    counts = {}
    for v in res.values():
        counts[v.split()[0]] = counts.get(v.split()[0], 0) + 1
    print(f"{len(res)} files: {counts} -> {root}")
    sys.exit(1 if bad else 0)
