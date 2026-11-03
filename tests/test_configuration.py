"""Profiles compose validated component settings and preserve provenance and injection."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from relift.configuration import (
    ConfigurationSelection,
    InferenceConfig,
    Memory,
    experiment_configuration,
    load_configuration,
)
from relift.optimization import InferenceComponents, InferenceSession, OptimizationConfig
from scripts.optimization_matrix import component_matrix

ROOT = Path(__file__).resolve().parents[1]


def test_inheritance_and_content_fingerprints(tmp_path: Path) -> None:
    parent = tmp_path / "parent.toml"
    parent.write_text('[memory]\noffload="ram"\n[observations]\ncache_mb=128\n')
    child = tmp_path / "child.toml"
    child.write_text('extends="parent.toml"\n[observations]\ndepth_cache_mb=256\n')
    selection = load_configuration(path=child)
    assert selection.config.memory.offload == "ram"
    assert selection.config.observations.cache_mb == 128
    assert selection.config.observations.depth_cache_mb == 256
    assert [s.path for s in selection.sources] == [str(parent), str(child)]
    original = selection.as_dict()
    parent.write_text(parent.read_text() + "# provenance-only change\n")
    revised = load_configuration(path=child)
    assert revised.config == selection.config
    assert revised.sources != selection.sources
    assert original == selection.as_dict()
    child.write_text('extends="child.toml"\n')
    with pytest.raises(ValueError, match="cycle"):
        load_configuration(path=child)


@pytest.mark.parametrize(
    "values",
    [
        {"observation": {}},
        {"memory": {"typo": 3}},
        {"correction": {"batch": True}},
        {"observations": {"prefetch": "false"}},
        {"rays": {"backend": "missing"}},
        {"schema_version": True},
        {"schema_version": 2},
        {"memory": {"ram_gb": float("inf")}},
        {"memory": {"ram_gb": 0}},
        {"memory": []},
        {"storage": {"commits": "unbounded"}},
        {"memory": {"offload": "ram"}, "observations": {"cache_mb": 128, "gather": "cpp"}},
    ],
)
def test_bad_settings_fail_before_inference(values: dict[str, Any]) -> None:
    with pytest.raises((ValueError, TypeError)):
        InferenceConfig.from_dict(values)


def test_all_existing_optimizations_round_trip() -> None:
    legacy = OptimizationConfig(
        shards="cpp",
        shard_writer="current",
        rays="cuda_timeline",
        da3_weights="python",
        da3_features="cpp",
        tracker_init_weights=True,
        tracker_copy_outputs=True,
        hash_writes=False,
        depth_cache_mb=256,
        observation_cache_mb=128,
        prefetch=True,
        corrector_batch=4,
        padding_batch=8,
        compile_corrector=True,
        reuse_da3=True,
    )
    config = InferenceConfig.from_optimizations(legacy, Memory(offload="disk"))
    assert config.optimizations() == legacy
    assert InferenceConfig.from_dict(config.as_dict()) == config
    assert InferenceConfig().optimizations() == OptimizationConfig()
    assert load_configuration(profile="cuda-cached-rays").config.rays.backend == "cuda_timeline"
    with pytest.raises(ValueError):
        load_configuration(profile="../ram")
    with pytest.raises(ValueError):
        load_configuration(profile="ram", values={})


def test_session_uses_selected_configuration_and_injected_provider(tmp_path: Path) -> None:
    calls: list[str] = []

    class Provider:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.append("construct")
            assert kwargs["optimizations"].reuse_da3

        def activate(self) -> None:
            calls.append("activate")

        def park(self) -> None:
            calls.append("park")

        def close(self) -> None:
            calls.append("close")

    selection = load_configuration(profile="reuse-da3")
    components = InferenceComponents(
        depth_provider_factory=Provider,
        fingerprints=(("depth_provider_factory", "test-v1"),),
    )
    with InferenceSession(selection, components) as session:
        assert session.configuration == selection
        first = session.depth(tmp_path, "cpu")
        session.release_depth(first)
        assert session.depth(tmp_path, "cpu") is first
    assert calls == ["construct", "park", "activate", "close"]
    with InferenceSession(selection.config, components) as session:
        assert session.config == selection.config.optimizations()


def test_cli_prints_effective_configuration_without_video_or_cuda() -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts/demo_video.py"),
        "--profile",
        "observation-cache",
        "--ram-budget-gb",
        "12.5",
        "--corrector-batch",
        "4",
        "--print-config",
    ]
    output = subprocess.check_output(command, cwd=ROOT, text=True)
    data = json.loads(output)
    assert data["resolved"]["memory"]["ram_gb"] == 12.5
    assert data["resolved"]["observations"]["cache_mb"] == 128
    assert data["resolved"]["correction"]["batch"] == 4
    assert len(data["sources"]) == 2
    failed = subprocess.run([*command, "--opt", "prefetch=true"], cwd=ROOT, capture_output=True)
    assert failed.returncode != 0


def test_component_matrix_and_experiment_selectors(tmp_path: Path) -> None:
    base = load_configuration(profile="ram").config.as_dict()
    grid = {"observations.cache_mb": [0, 128], "storage.hash_writes": [True, False]}
    matrix = component_matrix(base, grid)
    assert len(matrix) == 4
    isolated = component_matrix(base, grid, isolated=True)
    assert len(isolated) == 3
    assert base == load_configuration(profile="ram").config.as_dict()
    selection = experiment_configuration({"configuration": matrix[1]}, tmp_path)
    assert isinstance(selection, ConfigurationSelection)
    assert selection.config == InferenceConfig.from_dict(matrix[1])
    assert experiment_configuration({"profile": True}, tmp_path) is None  # legacy diagnostic flag
    with pytest.raises(ValueError):
        experiment_configuration({"configuration": base, "optimizations": {}}, tmp_path)
    with pytest.raises(ValueError):
        component_matrix(base, {"observations.typo": [0]})
    with pytest.raises(ValueError):
        component_matrix(base, {"cache_mb": [0]})
    config = replace(selection.config, memory=Memory(offload="disk"))
    assert config.optimizations() == selection.config.optimizations()
