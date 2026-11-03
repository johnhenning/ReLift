#!/usr/bin/env python
"""PointOdyssey training annotations -> the window cache the corrector trains on.

    python scripts/prepare_pointodyssey.py --src data/pointodyssey/train --stream 0   # and --stream 1, --stream 2

Reads ``<src>/<scene>/anno.npz`` (the official PointOdyssey training release) and writes
``<cache_root>/pointodyssey_windows/<scene>.npz``.  The published cache was built by three independent streams
(every third scene of the sorted list, each with its own seeded generator); run the three streams, in parallel or one
after the other, to reproduce it array for array.  --nstreams other than 3 gives a different point sampling.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from relift.config import paths  # noqa: E402
from relift.data.pointodyssey import build_stream, split  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", default=None, help="default <data_root>/pointodyssey/train")
    ap.add_argument("--cache", default=None, help="default <cache_root>/pointodyssey_windows")
    ap.add_argument("--stream", type=int, default=0)
    ap.add_argument("--nstreams", type=int, default=3)
    ap.add_argument("--rewrite", action="store_true", help="overwrite existing scene files")
    a = ap.parse_args()
    P = paths()
    src = a.src or P["data_root"] / "pointodyssey" / "train"
    cache = a.cache or P["cache_root"] / "pointodyssey_windows"
    build_stream(
        src, cache, a.stream, a.nstreams, rewrite=a.rewrite, log=lambda s: print(s, flush=True)
    )
    train, val = split(cache)
    print(f"cache {cache}: {len(train)} training scenes, {len(val)} validation scenes: {val}")
    print("PREPARE_POINTODYSSEY_DONE")
