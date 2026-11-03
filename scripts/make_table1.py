#!/usr/bin/env python
"""The evaluation table from the per-clip score files written by score.py.

    python scripts/make_table1.py [--scores_dir outputs/scores] [--subsets po ds] [--record outputs/table1.json]

One row per <tracker>_<geometry> variant whose score files exist, one All / Dyn pair per subset (APD3D x100, mean
over the clips), and the mean over the subsets.  --record writes the table and the machine provenance (GPU models
recorded in the windows and prediction files) to a JSON file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from relift.config import paths  # noqa: E402
from relift.eval.tables import ROWS, SUBSET_ORDER, markdown, table  # noqa: E402
from relift.provenance import gpus_of_preds, gpus_of_windows, machine  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scores_dir", default=None, help="default <outputs_root>/scores")
    ap.add_argument(
        "--subsets",
        nargs="+",
        default=list(SUBSET_ORDER),
        choices=list(SUBSET_ORDER),
        help="restrict the table to these subsets",
    )
    ap.add_argument(
        "--preds_root",
        default=None,
        help="default <outputs_root>/preds; the GPU models behind the predictions are collected from it",
    )
    ap.add_argument(
        "--windows_root",
        default=None,
        help="default <cache_root>/worldtrack_windows; the GPU models behind the windows likewise",
    )
    ap.add_argument(
        "--record", default=None, help="write the table and its provenance to this JSON file"
    )
    a = ap.parse_args()
    P = paths()
    scores_dir = a.scores_dir or P["outputs_root"] / "scores"
    observed = table(scores_dir, subsets=a.subsets)
    if not observed:
        raise SystemExit(f"no score files in {scores_dir} (run score.py first)")
    print(markdown(observed, a.subsets))
    gpus = dict(
        corrector=gpus_of_preds(a.preds_root or P["outputs_root"] / "preds", [v for _, v in ROWS]),
        windows=gpus_of_windows(a.windows_root or P["cache_root"] / "worldtrack_windows"),
    )
    if any(gpus.values()):
        print("\nGPU models behind this table:", json.dumps(gpus))
    if a.record:
        os.makedirs(os.path.dirname(a.record) or ".", exist_ok=True)
        prov = machine("cpu")  # host, date, torch (no GPU in this step)
        json.dump(
            dict(
                provenance=prov,
                scores_dir=str(scores_dir),
                subsets=list(a.subsets),
                rows=observed,
                gpus=gpus,
            ),
            open(a.record, "w"),
            indent=1,
        )
        print(f"recorded {a.record}")


if __name__ == "__main__":
    main()
