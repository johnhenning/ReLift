from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from relift.provenance import gpus_of_preds, gpus_of_windows, machine, note


def test_note_merges_and_gpus_are_collected(tmp_path: Path) -> None:
    w = tmp_path / "windows" / "adt_mini" / "clip"
    w.mkdir(parents=True)
    note(str(w / "provenance.json"), "da3", "cpu")
    note(str(w / "provenance.json"), "cowtracker", "cpu", size=[336, 672])
    d = json.load(open(w / "provenance.json"))
    assert (
        set(d) == {"da3", "cowtracker"}
        and d["cowtracker"]["size"] == [336, 672]
        and d["da3"]["gpu"] == "cpu"
    )
    assert gpus_of_windows(str(tmp_path / "windows")) == {"da3": ["cpu"], "cowtracker": ["cpu"]}
    p = tmp_path / "preds" / "a_b" / "adt_mini"
    p.mkdir(parents=True)
    np.savez(p / "x.npz", tracks=np.zeros(3), gpu=np.array("GPU-X"))
    np.savez(p / "y.npz", tracks=np.zeros(3))
    assert gpus_of_preds(str(tmp_path / "preds"), ["a_b", "none"]) == {
        "a_b": ["GPU-X", "unrecorded"]
    }
    assert "date" in machine("cpu") and machine("cpu")["gpu"] == "cpu"
