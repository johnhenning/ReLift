"""The evaluation table (the layout of Table 1 of the paper) from per-clip score files.

Score files: ``<scores_dir>/<variant>_<subset_dir>.json`` as written by ``scoring.score_subset``.  A cell is the mean
over the clips of a subset (x100); the Mean column is the mean of the four subset means.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..data.worldtrack import NAMES, subset_dir
from ..records.model import AlignmentMode, EvaluationRow, SubsetMeans, SubsetName

ROWS = [
    ("AllTracker + DA3", "alltracker_da3"),
    ("AllTracker + VGGT-Omega", "alltracker_vggto"),
    ("AllTracker + 4RC", "alltracker_rc4"),
    ("CoWTracker + DA3", "cowtracker_da3"),
    ("CoWTracker + VGGT-Omega", "cowtracker_vggto"),
    ("CoWTracker + 4RC", "cowtracker_rc4"),
]
SUBSET_ORDER: tuple[SubsetName, ...] = ("adt", "pstudio", "po", "ds")
CELLS: tuple[Literal["all", "dyn"], ...] = ("all", "dyn")


def subset_means(
    scores_dir: Any, variant: Any, subset: Any, mode: AlignmentMode = "sim3"
) -> SubsetMeans | None:
    f = Path(scores_dir) / f"{variant}_{subset_dir(subset)}.json"
    if not f.exists():
        return None
    rows = json.load(open(f))
    vals = [r[mode]["apd"] for r in rows.values()]
    dyn = [r[mode]["dyn"] for r in rows.values() if "dyn" in r[mode]]
    return dict(all=100 * float(np.mean(vals)), dyn=100 * float(np.mean(dyn)), clips=len(rows))


def table(
    scores_dir: str | Path,
    rows: list[tuple[str, str]] = ROWS,
    mode: AlignmentMode = "sim3",
    subsets: tuple[SubsetName, ...] | list[SubsetName] = SUBSET_ORDER,
) -> dict[str, EvaluationRow]:
    """-> {label: {subset: {all, dyn, clips}, "mean": {all, dyn}, "subsets": [...]}} for the rows whose files exist.
    ``subsets`` restricts the table (e.g. one subset on a test machine); the Mean column is then over those only."""
    out: dict[str, EvaluationRow] = {}
    for label, variant in rows:
        per = {s: subset_means(scores_dir, variant, s, mode) for s in subsets}
        row: EvaluationRow = dict(variant=variant, subsets=list(subsets))
        missing = [s for s, value in per.items() if value is None]
        if missing:
            row["missing"] = missing
        for subset, value in per.items():
            if value is not None:
                row[subset] = value
        if not missing:
            row["mean"] = dict(
                all=float(np.mean([v["all"] for v in per.values() if v is not None])),
                dyn=float(np.mean([v["dyn"] for v in per.values() if v is not None])),
            )
        out[label] = row
    return out


def markdown(
    tab: dict[str, EvaluationRow], subsets: tuple[SubsetName, ...] | list[SubsetName] = SUBSET_ORDER
) -> str:
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
