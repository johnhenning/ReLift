"""Optional one-batch CPU gathering, kept separate from CUDA ray arithmetic."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np


class BatchGather:
    def __init__(self, backend: str, native: ModuleType | None = None) -> None:
        if backend not in ("python", "cpp"):
            raise ValueError(f"Unknown gather backend {backend}")
        self.backend = backend
        self.buffers: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] | None = None
        self.peak_bytes = 0
        if backend == "cpp":
            from .native_backend import load_cpu

            self.native = native or load_cpu()

    def __call__(
        self,
        tracks: np.ndarray[Any, Any],
        visibility: np.ndarray[Any, Any],
        ids: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        if ids.ndim != 1 or ((ids < 0) | (ids >= tracks.shape[1] * tracks.shape[2])).any():
            raise IndexError("Pixel IDs must be a vector within the grid")
        if tracks.dtype not in (np.float16, np.float32) or visibility.dtype not in (
            np.float16,
            np.float32,
        ):
            raise TypeError("Gather supports float16/float32 sources only")
        T, N = len(tracks), len(ids)
        shape = (T, N, 2)
        if self.buffers is None or self.buffers[0].shape != shape:
            # Release the preceding batch before allocating a differently sized one.
            self.buffers = None
            self.buffers = (np.empty(shape, np.float32), np.empty((T, N), np.float32))
        assert self.buffers is not None
        coords, vis = self.buffers
        self.peak_bytes = max(self.peak_bytes, coords.nbytes + vis.nbytes)
        if self.backend == "cpp":
            self.native.gather(tracks, visibility, ids, coords, vis)
        else:
            rows, cols = np.divmod(ids, tracks.shape[2])
            # Existing native NumPy gather; copy directly into reusable destinations.
            np.copyto(coords, tracks[:, rows, cols], casting="safe")
            np.copyto(vis, visibility[:, rows, cols], casting="safe")
        return coords, vis
