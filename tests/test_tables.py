from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from relift.eval.tables import markdown, table


def _scores(d: Any, variant: Any, subset: Any, vals: Any, dyn_every: bool = True) -> None:
    rows = {
        f"clip{i}": {"sim3": dict(apd=v, **({"dyn": v - 0.1} if dyn_every or i % 2 else {}))}
        for i, v in enumerate(vals)
    }
    (d / f"{variant}_{subset}.json").write_text(json.dumps(rows))


def test_table_and_markdown(tmp_path: Path) -> None:
    rows = [("A + B", "a_b")]
    vals = np.linspace(0.8, 0.9, 10)
    for sub in ("adt_mini", "pstudio_mini", "po_mini", "ds_mini"):
        _scores(tmp_path, "a_b", sub, vals, dyn_every=(sub != "po_mini"))
    o = table(tmp_path, rows)
    r = o["A + B"]
    assert abs(r["adt"]["all"] - 100 * vals.mean()) < 1e-9 and r["adt"]["clips"] == 10
    assert (
        abs(r["adt"]["dyn"] - 100 * (vals.mean() - 0.1)) < 1e-9
        and abs(r["mean"]["all"] - 100 * vals.mean()) < 1e-9
    )
    md = markdown(o)
    assert "| A + B |" in md and "| Mean All | Mean Dyn |" in md and md.count("\n") == 2


def test_partial_subsets(tmp_path: Path) -> None:
    rows = [("A + B", "a_b")]
    vals = np.linspace(0.8, 0.9, 10)
    _scores(tmp_path, "a_b", "po_mini", vals)  # only one subset was computed
    full = table(tmp_path, rows)
    assert "mean" not in full["A + B"] and full["A + B"]["missing"] == ["adt", "pstudio", "ds"]
    assert "missing: adt, pstudio, ds" in markdown(full)
    o = table(tmp_path, rows, subsets=("po",))
    assert abs(o["A + B"]["mean"]["all"] - 100 * vals.mean()) < 1e-9 and o["A + B"]["subsets"] == [
        "po"
    ]
    assert "Mean(1) All" in markdown(o, ("po",)) and "| PO All |" in markdown(o, ("po",))
