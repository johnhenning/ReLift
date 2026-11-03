from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from test_streaming import prepared_window

from relift.optimization import InferenceComponents, InferenceSession, OptimizationConfig
from relift.streaming import MappedObservations
from scripts.optimization_matrix import expand, isolate


def test_strict_settings_and_independent_combinations(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(dict(gather="python", hash_writes=False, depth_cache_mb=256)))
    settings = OptimizationConfig.read(path, ["prefetch=true", "corrector_batch=4"])
    assert settings.gather == "python" and settings.prefetch and not settings.hash_writes
    assert settings.corrector_batch == 4
    for values in (
        {"typo": True},
        {"gather": "missing"},
        {"prefetch": "true"},
        {"corrector_batch": True},
        {"depth_cache_mb": float("nan")},
    ):
        with pytest.raises((TypeError, ValueError)):
            OptimizationConfig.from_dict(values)
    matrix = expand(
        {"corrector_batch": 8}, {"hash_writes": [True, False], "gather": ["current", "python"]}
    )
    assert len(matrix) == 4 and len({(row["hash_writes"], row["gather"]) for row in matrix}) == 4
    with pytest.raises(ValueError):
        expand({}, {"gather": []})
    isolated = isolate({}, {"hash_writes": [True, False], "gather": ["current", "python"]})
    assert len(isolated) == 3
    assert all(
        sum(dict(row)[key] != dict(OptimizationConfig().as_dict())[key] for key in row) <= 1
        for row in isolated
    )


def test_injected_gather_is_used_and_fingerprinted(tmp_path: Path) -> None:
    prepared_window(tmp_path)
    calls = []

    def gather(tracks: Any, visibility: Any, ids: Any) -> Any:
        calls.append(ids.copy())
        rows, cols = np.divmod(ids, tracks.shape[2])
        return tracks[:, rows, cols].astype(np.float32), visibility[:, rows, cols].astype(
            np.float32
        )

    components = InferenceComponents(
        gather_factory=lambda: gather, fingerprints=(("gather_factory", "test-v1"),)
    )
    current = MappedObservations(tmp_path, "cpu").rays(np.array([0, 3, 3, 14]))
    actual = MappedObservations(tmp_path, "cpu", components=components).rays(
        np.array([0, 3, 3, 14])
    )
    assert len(calls) == 1
    for a, b in zip(current, actual, strict=False):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match="fingerprints"):
        InferenceComponents(gather_factory=lambda: gather)


def test_inapplicable_optimizations_fail_instead_of_silently_running_control() -> None:
    OptimizationConfig().validate_mode("none")
    OptimizationConfig(tracker_copy_outputs=True, corrector_batch=4).validate_mode("none")
    with pytest.raises(ValueError, match="Streaming"):
        OptimizationConfig(prefetch=True).validate_mode("none")
    with pytest.raises(ValueError, match="disk"):
        OptimizationConfig(da3_features="cpp").validate_mode("ram")
    OptimizationConfig(gather="cpp", da3_weights="cpp", da3_features="cpp").validate_mode("disk")


def test_injected_camera_builder_uses_full_timeline_rotation(tmp_path: Path) -> None:
    from relift.cuda_rays import torch_ray_builder

    T, _, _ = prepared_window(tmp_path)
    calls = []

    class Builder:
        def __call__(self, *args: Any) -> None:
            raise AssertionError("camera protocol should be used")

        def camera(self, coords: Any, depth: Any, K: Any) -> Any:
            calls.append(("camera", len(coords)))
            identity = torch.eye(3).expand(len(coords), 3, 3)
            return torch_ray_builder(coords, depth, K, identity)

        def rotate(self, rotation: Any, camera: Any) -> Any:
            calls.append(("rotate", len(camera)))
            return torch.einsum("tij,tnj->tni", rotation, camera)

    builder = Builder()
    components = InferenceComponents(
        ray_builder=builder, fingerprints=(("ray_builder", "camera-test-v1"),)
    )
    ids = np.array([0, 3, 3, 14])
    reference = MappedObservations(tmp_path, "cpu").rays(ids)
    actual = MappedObservations(tmp_path, "cpu", components=components).rays(ids)
    assert calls == [("camera", T), ("rotate", T)]
    for a, b in zip(actual, reference, strict=False):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert OptimizationConfig.read(overrides=['rays="cuda_timeline"']).rays == "cuda_timeline"
    with pytest.raises(ValueError, match="Streaming"):
        OptimizationConfig(rays="cuda_timeline").validate_mode("none")


def test_session_reuses_only_matching_provider_and_closes_on_exception(tmp_path: Path) -> None:
    events = []

    class Provider:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            events.append("create")

        def park(self) -> None:
            events.append("park")

        def activate(self) -> None:
            events.append("activate")

        def close(self) -> None:
            events.append("close")

    components = InferenceComponents(
        depth_provider_factory=Provider, fingerprints=(("depth_provider_factory", "fake-v1"),)
    )
    with pytest.raises(RuntimeError, match="intentional"):
        with InferenceSession(OptimizationConfig(reuse_da3=True), components) as session:
            provider = session.depth(tmp_path, "cpu", scratch_dir=str(tmp_path / "a"))
            session.release_depth(provider)
            assert session.depth(tmp_path, "cpu", scratch_dir=str(tmp_path / "b")) is provider
            raise RuntimeError("intentional")
    assert events == ["create", "park", "activate", "close"]
