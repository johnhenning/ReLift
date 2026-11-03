"""Threaded commits compose with caches, ray prefetch and native writers on CUDA."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from test_streaming import prepared_window

from relift.model import Corrector
from relift.optimization import InferenceComponents, OptimizationConfig
from relift.streaming import stream_correct

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("rays", ["current", "cuda_timeline"])
@pytest.mark.parametrize("writer", ["current", "cpp"])
def test_threaded_commits_match_sync_combinations_and_resume(
    tmp_path: Path,
    prefetch: bool,
    rays: str,
    writer: str,
) -> None:
    if (rays != "current" or writer == "cpp") and not os.environ.get("RELIFT_NATIVE_BUILD"):
        pytest.skip("Explicit native build required")
    T, H, W = prepared_window(tmp_path)
    torch.manual_seed(7)
    model = Corrector(dim=16, depth=1, heads=2, chunk=4).cuda().eval()
    components = InferenceComponents()
    if rays != "current" or writer == "cpp":
        components = components.with_native_build(os.environ["RELIFT_NATIVE_BUILD"])
    common = dict(
        corrector_batch=2,
        observation_cache_mb=1,
        depth_cache_mb=1,
        rays=rays,
        prefetch=prefetch,
        shard_writer=writer,
    )
    baseline = stream_correct(
        tmp_path,
        tmp_path / "sync",
        model,
        {},
        group=4,
        optimizations=OptimizationConfig(**common),
        components=components,
    )
    selected = OptimizationConfig(**common, commit_backend="threaded")
    output = tmp_path / "threaded"
    candidate = stream_correct(
        tmp_path, output, model, {}, group=4, optimizations=selected, components=components
    )
    for name in ("xyz", "lift", "observed"):
        for frame in range(T):
            np.testing.assert_array_equal(
                candidate.read_frame(name, frame), baseline.read_frame(name, frame)
            )
    counters = json.loads((output / "shard-commits.json").read_text())
    assert counters["backend"] == "threaded" and counters["shards_committed"] == 3
    assert counters["peak_pending_payload_bytes"] <= T * min(H * W, 8) * 19
    assert counters["pending_payload_bytes"] == 0
    resumed = stream_correct(
        tmp_path, output, model, {}, group=4, optimizations=selected, components=components
    )
    assert resumed.verify() and resumed.manifest == candidate.manifest
    assert counters == json.loads((output / "shard-commits.json").read_text())
