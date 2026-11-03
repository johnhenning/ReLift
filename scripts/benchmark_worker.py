#!/usr/bin/env python
"""Seed the identical process environment before invoking either implementation.

python scripts/benchmark_worker.py --seed 0 frozen/scripts/demo_video.py VIDEO ...
"""

from __future__ import annotations

import argparse
import random
import runpy
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("script", type=Path)
    ap.add_argument("arguments", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed % 2**32)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    script = args.script.resolve()
    # The frozen demo must import frozen relift modules, not the editable checkout.
    sys.path.insert(0, str(script.parent.parent))
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *args.arguments]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
