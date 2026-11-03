"""Real-device cache streams, exact rays, decline policy and composed inference."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from test_streaming import prepared_window

from relift.model import Corrector
from relift.observation_cache import ObservationCache
from relift.optimization import InferenceComponents, OptimizationConfig
from relift.streaming import MappedObservations, stream_correct

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def test_nondefault_stream_close_and_headroom_decline(monkeypatch: pytest.MonkeyPatch) -> None:
    coords = np.arange(35 * 9 * 11 * 2, dtype=np.float16).reshape(35, 9, 11, 2)
    vis = np.ones((35, 9, 11), np.float16)
    cache = ObservationCache(coords, vis, "cuda", 1)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = cache.gather([0, 98, 7, 7], frame=16)
    cache.close()
    torch.cuda.current_stream().wait_stream(stream)
    for a, b in zip(
        actual,
        (
            coords[16:17, [0, 8, 0, 0], [0, 10, 7, 7]].astype(np.float32),
            vis[16:17, [0, 8, 0, 0], [0, 10, 7, 7]].astype(np.float32),
        ),
        strict=False,
    ):
        np.testing.assert_array_equal(a.cpu().numpy(), b)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args: (0, 1))
    declined = ObservationCache(coords, vis, "cuda", 1)
    assert declined.metrics["state"] == "headroom_declined" and declined.coords is None


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("rays", ["current", "cuda_timeline"])
@pytest.mark.parametrize("depth", [0, 1])
def test_cache_combinations_are_bitwise_and_resume(
    tmp_path: Path, prefetch: bool, rays: Any, depth: Any
) -> None:
    if rays != "current" and not os.environ.get("RELIFT_NATIVE_BUILD"):
        pytest.skip("explicit native build required")
    T, H, W = prepared_window(tmp_path)
    components = InferenceComponents()
    if rays != "current":
        components = components.with_native_build(os.environ["RELIFT_NATIVE_BUILD"])
    ids = np.arange(H * W)
    reference = MappedObservations(tmp_path, "cuda")
    source = MappedObservations(
        tmp_path,
        "cuda",
        components=components,
        ray_backend=rays,
        depth_cache_mb=depth,
        observation_cache_mb=1,
    )
    for frame in (None, 0, T - 1):
        for a, b in zip(
            reference.rays(ids, frame=frame), source.rays(ids, frame=frame), strict=False
        ):
            assert torch.equal(a.view(torch.int32), b.view(torch.int32))
    source.close()
    torch.manual_seed(7)
    model = Corrector(dim=16, depth=1, heads=2, chunk=4).eval().cuda()
    reference = stream_correct(tmp_path, tmp_path / "reference", model, {}, group=4, batch=2)
    config = OptimizationConfig(
        observation_cache_mb=1,
        depth_cache_mb=depth,
        rays=rays,
        prefetch=prefetch,
        corrector_batch=2,
    )
    actual = stream_correct(
        tmp_path,
        tmp_path / "candidate",
        model,
        {},
        group=4,
        optimizations=config,
        components=components,
    )
    for key, value in reference.sample(ids).items():
        np.testing.assert_array_equal(actual.sample(ids)[key], value)
    resumed = stream_correct(
        tmp_path,
        tmp_path / "candidate",
        model,
        {},
        group=4,
        optimizations=config,
        components=components,
    )
    assert resumed.manifest is not None
    assert resumed.manifest["state"] == "complete"
