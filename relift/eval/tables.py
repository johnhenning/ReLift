"""The evaluation table (the layout of Table 1 of the paper) from per-clip score files.

Score files: ``<scores_dir>/<variant>_<subset_dir>.json`` as written by ``scoring.score_subset``.  A cell is the mean
over the clips of a subset (x100); the Mean column is the mean of the four subset means.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..data.worldtrack import NAMES, subset_dir

ROWS = [
    ("AllTracker + DA3", "alltracker_da3"),
    ("AllTracker + VGGT-Omega", "alltracker_vggto"),
    ("AllTracker + 4RC", "alltracker_rc4"),
    ("CoWTracker + DA3", "cowtracker_da3"),
    ("CoWTracker + VGGT-Omega", "cowtracker_vggto"),
    ("CoWTracker + 4RC", "cowtracker_rc4"),
]
SUBSET_ORDER = ("adt", "pstudio", "po", "ds")
CELLS = ("all", "dyn")


def subset_means(
    scores_dir: Any, variant: Any, subset: Any, mode: str = "sim3"
) -> dict[str, float] | None:
    f = Path(scores_dir) / f"{variant}_{subset_dir(subset)}.json"
    if not f.exists():
        return None
    rows = json.load(open(f))
    vals = [r[mode]["apd"] for r in rows.values()]
    dyn = [r[mode]["dyn"] for r in rows.values() if "dyn" in r[mode]]
    return dict(all=100 * float(np.mean(vals)), dyn=100 * float(np.mean(dyn)), clips=len(rows))


def table(
    scores_dir: Any, rows: Any = ROWS, mode: str = "sim3", subsets: Any = SUBSET_ORDER
) -> dict[str, Any]:
    """-> {label: {subset: {all, dyn, clips}, "mean": {all, dyn}, "subsets": [...]}} for the rows whose files exist.
    ``subsets`` restricts the table (e.g. one subset on a test machine); the Mean column is then over those only."""
    out = {}
    for label, variant in rows:
        per = {s: subset_means(scores_dir, variant, s, mode) for s in subsets}
        if any(v is None for v in per.values()):
            missing = [s for s, v in per.items() if v is None]
            out[label] = dict(
                variant=variant,
                missing=missing,
                subsets=list(subsets),
                **{s: v for s, v in per.items() if v},
            )
            continue
        out[label] = dict(
            variant=variant,
            subsets=list(subsets),
            **per,
            mean={
                c: float(np.mean([value[c] for value in per.values() if value is not None]))
                for c in CELLS
            },
        )
    return out


def markdown(tab: Any, subsets: Any = SUBSET_ORDER) -> str:
    mean = "Mean" if tuple(subsets) == tuple(SUBSET_ORDER) else f"Mean({len(subsets)})"
    head = (
        "| Method | "
        + " | ".join(f"{NAMES[s]} All | {NAMES[s]} Dyn" for s in subsets)
        + f" | {mean} All | {mean} Dyn | clips |"
    )
    lines = [head, "|:---|" + "---:|" * (2 * len(subsets) + 3)]
    for label, r in tab.items():
        if "mean" not in r:
            lines.append(f"| {label} | missing: {', '.join(r['missing'])} |")
            continue
        cells = [f"{r[s][c]:.2f}" for s in subsets for c in CELLS] + [
            f"{r['mean'][c]:.2f}" for c in CELLS
        ]
        lines.append(
            f"| {label} | " + " | ".join(cells) + f" | {sum(r[s]['clips'] for s in subsets)} |"
        )
    return "\n".join(lines)
