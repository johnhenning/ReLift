"""Cache precision, accounting, fallback, injection and lifecycle checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from test_streaming import prepared_window

from relift.model import Corrector
from relift.observation_cache import ObservationCache
from relift.optimization import InferenceComponents, OptimizationConfig
from relift.streaming import MappedObservations, stream_correct


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_original_dtype_budget_and_exact_selected_values(dtype: Any) -> None:
    tracks = np.arange(35 * 3 * 5 * 2, dtype=np.float32).reshape(35, 3, 5, 2).astype(dtype)[:, ::-1]
    visibility = (tracks[..., 0] / np.float32(100)).astype(dtype)
    needed = tracks.nbytes + visibility.nbytes
    cache = ObservationCache(tracks, visibility, "cpu", needed / 2**20)
    assert cache.active and cache.metrics["resident_bytes"] == needed
    assert cache.coords is not None
    assert cache.coords.element_size() == np.dtype(dtype).itemsize
    ids = np.array([14, 0, 3, 3, 10])
    for frame in (None, 0, 16, 34):
        times = slice(None) if frame is None else slice(frame, frame + 1)
        a, b = cache.gather(ids, frame=frame)
        np.testing.assert_array_equal(
            a.numpy(), tracks[times, ids // 5, ids % 5].astype(np.float32)
        )
        np.testing.assert_array_equal(
            b.numpy(), visibility[times, ids // 5, ids % 5].astype(np.float32)
        )
    for ids in ([-1], [15], [[1]]):
        with pytest.raises(IndexError):
            cache.gather(cast(Any, ids))
    with pytest.raises(IndexError):
        cache.gather([0], frame=35)
    assert cache.gather([])[0].shape == (35, 0, 2)
    cache.close()
    cache.close()
    assert not cache.active and cache.coords is None and cache.metrics["resident_bytes"] == 0
    with pytest.raises(RuntimeError):
        cache.gather([0])
    declined = ObservationCache(tracks, visibility, "cpu", (needed - 1) / 2**20)
    assert (
        not declined.active
        and declined.coords is None
        and declined.metrics["state"] == "budget_declined"
    )


def test_cache_configuration_and_injected_cleanup_on_exception(tmp_path: Path) -> None:
    prepared_window(tmp_path)
    for value in (-1, float("inf"), True):
        with pytest.raises(ValueError):
            OptimizationConfig(observation_cache_mb=value)
    with pytest.raises(ValueError, match="replaces CPU"):
        OptimizationConfig(observation_cache_mb=1, gather="cpp")
    with pytest.raises(ValueError, match="Streaming"):
        OptimizationConfig(observation_cache_mb=1).validate_mode("none")
    with pytest.raises(ValueError, match="fingerprints"):
        InferenceComponents(observation_cache_factory=ObservationCache)
    owned = []

    def factory(*args: Any) -> Any:
        cache = ObservationCache(*args)
        owned.append(cache)
        return cache

    components = InferenceComponents(
        observation_cache_factory=factory, fingerprints=(("observation_cache_factory", "test-v1"),)
    )
    source = MappedObservations(tmp_path, "cpu", observation_cache_mb=1, components=components)
    reference = MappedObservations(tmp_path, "cpu")
    ids = np.array([0, 3, 3, 14])
    for a, b in zip(reference.rays(ids), source.rays(ids), strict=False):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    source.close()
    assert owned[-1].closed
    model = Corrector(dim=16, depth=1, heads=2, chunk=4).eval()
    with pytest.raises(MemoryError, match="scratch budget"):
        stream_correct(
            tmp_path,
            tmp_path / "out",
            model,
            {},
            device="cpu",
            scratch_budget_gb=0,
            optimizations=OptimizationConfig(observation_cache_mb=1),
            components=components,
        )
    assert owned[-1].closed


def test_partial_cache_installation_releases_owned_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    original = torch.empty
    calls = []

    def fail(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("allocation failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", fail)
    cache = ObservationCache.__new__(ObservationCache)
    with pytest.raises(RuntimeError, match="allocation failed"):
        cache.__init__(np.ones((2, 3, 5, 2), np.float16), np.ones((2, 3, 5), np.float16), "cpu", 1)
    assert cache.coords is None and cache.visibility is None and cache.closed


def test_disabled_cache_keeps_legacy_source_conversion(tmp_path: Path) -> None:
    T, H, W = prepared_window(tmp_path)
    path = tmp_path / "alltracker_tracks.npz"
    with np.load(path) as archive:
        arrays = {key: archive[key].astype(np.float64) for key in archive.files}
    np.savez(path, **arrays)
    source = MappedObservations(tmp_path, "cpu")
    assert not source.observation_cache.active
    assert source.rays(np.array([0, 3, 14]))[1].dtype == torch.float32
    with pytest.raises(TypeError, match="float16/float32"):
        MappedObservations(tmp_path, "cpu", observation_cache_mb=1)
