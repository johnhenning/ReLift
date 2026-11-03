from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from relift.batch_gather import BatchGather


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_gather_strides_duplicates_empty_tail_and_reuse(dtype: Any, backend: str) -> None:
    if backend == "cpp" and not os.environ.get("RELIFT_NATIVE_BUILD"):
        pytest.skip("explicit native build required")
    rng = np.random.default_rng(4)
    tracks = rng.normal(size=(5, 3, 7, 2)).astype(dtype)[:, ::-1, ::2]
    vis = rng.normal(size=(5, 3, 7)).astype(dtype)[:, ::-1, ::2]
    gather = BatchGather(backend)
    for ids in (np.array([11, 0, 5, 5]), np.arange(12), np.array([], dtype=np.int64)):
        coords, actual_vis = gather(tracks, vis, ids)
        rows, cols = np.divmod(ids, tracks.shape[2])
        np.testing.assert_array_equal(coords, tracks[:, rows, cols].astype(np.float32))
        np.testing.assert_array_equal(actual_vis, vis[:, rows, cols].astype(np.float32))
    with pytest.raises(IndexError):
        gather(tracks, vis, np.array([12]))
    with pytest.raises(IndexError):
        gather(tracks, vis, np.array([-1]))


def test_missing_native_build_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RELIFT_NATIVE_BUILD", raising=False)
    with pytest.raises(RuntimeError, match="RELIFT_NATIVE_BUILD"):
        BatchGather("cpp")


def test_mapped_readonly_source_survives(tmp_path: Path) -> None:
    source = np.arange(2 * 3 * 4 * 2, dtype=np.float16).reshape(2, 3, 4, 2)
    np.save(tmp_path / "tracks.npy", source)
    tracks = np.load(tmp_path / "tracks.npy", mmap_mode="r")
    coords, _ = BatchGather("python")(tracks, source[..., 0], np.array([0, 11]))
    np.testing.assert_array_equal(coords, source.reshape(2, 12, 2)[:, [0, 11]])
