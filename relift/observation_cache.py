"""Budgeted, immutable tracker tensors with one selected batch gathered on-device."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from .records.runtime import ObservationCacheMetrics


class ObservationCache:
    """Own original-dtype coordinates/visibility; preserve float32 gather results.

    Budget covers retained tensors, independently of the depth cache and active
    correction batch. Uploads use bounded frame blocks. A declined cache allocates
    nothing; its reason and requested size remain visible in metrics.
    """

    def __init__(
        self,
        tracks: np.ndarray[Any, Any],
        visibility: np.ndarray[Any, Any],
        device: str | torch.device,
        budget_mb: float,
    ) -> None:
        if type(budget_mb) not in (int, float) or not math.isfinite(budget_mb) or budget_mb < 0:
            raise ValueError("observation cache budget must be finite and nonnegative")
        if (
            tracks.ndim != 4
            or tracks.shape[-1] != 2
            or visibility.shape != tracks.shape[:3]
            or min(tracks.shape[:3]) < 1
        ):
            raise ValueError("observation cache shapes differ")
        self.device = torch.device(device)
        self.T, self.H, self.W = tracks.shape[:3]
        self.coords = self.visibility = None
        self.closed = False
        self._metrics: ObservationCacheMetrics = dict(
            state="disabled",
            budget_bytes=int(budget_mb * 2**20),
            required_bytes=tracks.nbytes + visibility.nbytes,
            resident_bytes=0,
            upload_bytes=0,
            upload_calls=0,
            gather_calls=0,
            pixel_id_upload_bytes=0,
            source_dtypes=[str(tracks.dtype), str(visibility.dtype)],
            device=str(self.device),
        )
        if budget_mb == 0:
            return
        if any(value.dtype not in (np.float16, np.float32) for value in (tracks, visibility)):
            raise TypeError("observation cache supports float16/float32 sources")
        if self._metrics["required_bytes"] > self._metrics["budget_bytes"]:
            self._metrics["state"] = "budget_declined"
            return
        if (
            self.device.type == "cuda"
            and torch.cuda.mem_get_info(self.device)[0]
            < self._metrics["required_bytes"] + 512 * 2**20
        ):
            self._metrics["state"] = "headroom_declined"
            return
        try:
            self.coords = torch.empty(
                (self.T, self.H * self.W, 2),
                device=self.device,
                dtype=torch.float16 if tracks.dtype == np.float16 else torch.float32,
            )
            self.visibility = torch.empty(
                (self.T, self.H * self.W),
                device=self.device,
                dtype=torch.float16 if visibility.dtype == np.float16 else torch.float32,
            )
            for start in range(0, self.T, 16):
                stop = min(start + 16, self.T)
                for source, target in ((tracks, self.coords), (visibility, self.visibility)):
                    # Private writable host block; no mapped-page mutation or
                    # lifetime dependence on a temporary read-only NumPy view.
                    block = np.array(source[start:stop], copy=True, order="C")
                    target[start:stop].copy_(
                        torch.from_numpy(block.reshape(target[start:stop].shape))
                    )
                    if self.device.type == "cuda":
                        self._metrics["upload_bytes"] += block.nbytes
                        self._metrics["upload_calls"] += 1
            self._metrics.update(
                state="active",
                resident_bytes=sum(
                    t.numel() * t.element_size() for t in (self.coords, self.visibility)
                ),
            )
        except BaseException:
            self.close()
            raise

    @property
    def active(self) -> bool:
        return not self.closed and self._metrics["state"] == "active"

    @property
    def metrics(self) -> ObservationCacheMetrics:
        return self._metrics.copy()

    def gather(
        self, pixel_ids: np.ndarray[Any, Any] | list[int], frame: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.active:
            raise RuntimeError("observation cache is not active")
        assert self.coords is not None and self.visibility is not None
        ids = np.asarray(pixel_ids, dtype=np.int64)
        if ids.ndim != 1 or ((ids < 0) | (ids >= self.H * self.W)).any():
            raise IndexError("pixel IDs must be a vector within the grid")
        if frame is not None and (type(frame) is not int or not 0 <= frame < self.T):
            raise IndexError("frame is outside the clip")
        times = slice(None) if frame is None else slice(frame, frame + 1)
        selected = torch.as_tensor(np.ascontiguousarray(ids), device=self.device)
        self._metrics["gather_calls"] += 1
        if self.device.type == "cuda":
            self._metrics["pixel_id_upload_bytes"] += ids.nbytes
            # The cache can be consumed by the lookahead stream and released
            # before that stream completes. Preserve allocator ownership then.
            self.coords.record_stream(torch.cuda.current_stream(self.device))
            self.visibility.record_stream(torch.cuda.current_stream(self.device))
        return (
            self.coords[times].index_select(1, selected).float(),
            self.visibility[times].index_select(1, selected).float(),
        )

    def close(self) -> None:
        self.coords = self.visibility = None
        self.closed = True
        self._metrics["resident_bytes"] = 0
        self._metrics["state"] = "closed"
