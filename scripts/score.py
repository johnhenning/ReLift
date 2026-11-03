#!/usr/bin/env python
"""Score saved sparse predictions under 4RC's 84-frame protocol (seeded RANSAC Sim(3) and global scale).

    python scripts/score.py --preds_root outputs/preds --variants alltracker_da3 [...] --subsets adt pstudio po ds

Reads ``<preds_root>/<variant>/<subset_dir>/<clip>.npz`` (written by run_corrector.py) and the WorldTrack clips under
``<data_root>/worldtrack/<subset_dir>/<clip>.npz``; writes ``<scores_dir>/<variant>_<subset_dir>.json`` with one
record per clip (resumable: clips already scored are kept).  Then ``make_table1.py`` turns the JSONs into Table 1.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from relift.config import paths  # noqa: E402
from relift.data.worldtrack import NUM_FRAMES, subset_dir  # noqa: E402
from relift.eval.scoring import score_subset  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--variants", nargs="+", required=True, help="prediction directories under --preds_root"
    )
    ap.add_argument("--subsets", nargs="+", default=["adt", "pstudio", "po", "ds"])
    ap.add_argument("--preds_root", default=None, help="default <outputs_root>/preds")
    ap.add_argument("--scores_dir", default=None, help="default <outputs_root>/scores")
    ap.add_argument("--worldtrack_root", default=None, help="default <data_root>/worldtrack")
    ap.add_argument("--num_frames", type=int, default=NUM_FRAMES)
    ap.add_argument(
        "--unseeded", action="store_true", help="4RC's unseeded RANSAC (not the paper's setting)"
    )
    a = ap.parse_args()
    P = paths()
    preds_root = a.preds_root or P["outputs_root"] / "preds"
    scores_dir = a.scores_dir or P["outputs_root"] / "scores"
    wt = a.worldtrack_root or P["data_root"] / "worldtrack"
    for subset in a.subsets:
        for v in a.variants:
            dst = os.path.join(scores_dir, f"{v}_{subset_dir(subset)}.json")
            rows = score_subset(
                os.path.join(preds_root, v, subset_dir(subset)),
                wt,
                subset,
                dst,
                a.num_frames,
                seed_ransac=not a.unseeded,
            )
            print(f"{v} {subset_dir(subset)}: {len(rows)} clips -> {dst}", flush=True)
    print("SCORE_DONE")
