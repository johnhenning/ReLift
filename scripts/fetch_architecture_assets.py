#!/usr/bin/env python
"""Vendor the pinned Mermaid browser bundle and license for the offline viewer.

    uv run python scripts/fetch_architecture_assets.py

The first download records checksums; subsequent downloads must match them.
Normal viewer builds use the checked-in assets and never access the network.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
VERSION = "11.12.0"
DEST = ROOT / "docs/vendor/mermaid"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    lock = DEST / "source.json"
    previous = json.loads(lock.read_text()) if lock.exists() else {}
    records = {}
    for filename, upstream in (("mermaid.min.js", "dist/mermaid.min.js"), ("LICENSE", "LICENSE")):
        url = f"https://cdn.jsdelivr.net/npm/mermaid@{VERSION}/{upstream}"
        with urlopen(url, timeout=120) as response:
            data = response.read()
        digest = hashlib.sha256(data).hexdigest()
        expected = previous.get("files", {}).get(filename, {}).get("sha256")
        if expected is not None and expected != digest:
            raise ValueError(f"Downloaded bytes differ from the recorded asset: {filename}")
        (DEST / filename).write_bytes(data)
        records[filename] = dict(url=url, sha256=digest, bytes=len(data))
        print(f"Vendored {filename}: {len(data):,} bytes")
    lock.write_text(
        json.dumps(dict(package="mermaid", version=VERSION, files=records), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
