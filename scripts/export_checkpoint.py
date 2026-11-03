#!/usr/bin/env python
"""Training checkpoint -> release checkpoint (EMA weights only, with the constructor config and provenance).

python scripts/export_checkpoint.py runs/corrector/final.pt checkpoints/released_checkpoint.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from relift.checkpoint import export, md5  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("src")
    ap.add_argument("dst")
    a = ap.parse_args()
    out = export(a.src, a.dst)
    print(
        f"wrote {a.dst}: step {out['step']}, {out['num_parameters']:,} parameters, config {out['config']}, "
        f"source md5 {out['source_md5']}, release md5 {md5(a.dst)}"
    )
