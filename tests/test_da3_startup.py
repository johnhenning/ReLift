"""CPU-first pager setup preserves state, ownership, cleanup and configuration."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
from test_da3_offload import assert_nested_equal, tiny_backbone

from relift.configuration import InferenceConfig, Memory, load_configuration
from relift.da3_offload import DA3Pager
from relift.optimization import OptimizationConfig
from relift.records.artifacts import DA3StartupProfile
from relift.records.runtime import ProfileFunction
from scripts.publish_da3_startup import profile_excerpt


def test_profile_publication_selects_leading_functions_without_changing_raw_evidence(
    tmp_path: Path,
) -> None:
    slow: ProfileFunction = dict(
        file="sample.py", line=1, function="slow", calls=1, self_seconds=3.0, inclusive_seconds=4.0
    )
    fast: ProfileFunction = {**slow, "function": "fast", "inclusive_seconds": 0.5}
    profile: DA3StartupProfile = dict(
        source="snapshot",
        source_manifest_sha256="source-sha",
        configuration=load_configuration(profile="cpu-first-da3").as_dict(),
        startup_seconds=8.0,
        cuda_allocated_peak_bytes=100,
        cuda_reserved_peak_bytes=200,
        functions=[fast, slow],
        caveat="diagnostic",
    )
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    original = path.read_bytes()
    excerpt = profile_excerpt(path, 1)
    assert excerpt["functions"] == [slow]
    assert excerpt["function_count_full"] == 2 and excerpt["functions_limit"] == 1
    assert path.read_bytes() == original


@pytest.mark.parametrize("mode", ["ram", "disk"])
@pytest.mark.parametrize("packing", ["current", "python"])
def test_cpu_first_matches_current_state_and_repeated_backbone_outputs(
    tmp_path: Path, mode: str, packing: str
) -> None:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(19)
    original = tiny_backbone()
    candidate = deepcopy(original)
    images = torch.randn(1, 5, 3, 28, 28)
    control = DA3Pager(
        original,
        mode=mode,
        device="cpu",
        scratch_dir=tmp_path / "control",
        weight_backend=packing,
        feature_backend=packing if mode == "disk" else "current",
    )
    owner = DA3Pager(
        candidate,
        mode=mode,
        device="cpu",
        scratch_dir=tmp_path / "candidate",
        weight_backend=packing,
        startup="cpu-first",
        feature_backend=packing if mode == "disk" else "current",
    )
    try:
        for key, value in original.state_dict().items():
            assert torch.equal(value, candidate.state_dict()[key]), key
        assert owner.startup_metrics["cold_parameter_bytes"] > 0
        assert owner.startup_metrics["cold_buffer_bytes"] == 4 * 8
        assert owner.startup_metrics["initial_cold_roundtrip_tensor_bytes"] == 0
        with torch.inference_mode():
            for _ in range(2):
                expected = original.get_intermediate_layers(
                    images, n=[1, 3], export_feat_layers=[0, 3], ref_view_strategy="saddle_balanced"
                )
                actual = candidate.get_intermediate_layers(
                    images, n=[1, 3], export_feat_layers=[0, 3], ref_view_strategy="saddle_balanced"
                )
                assert_nested_equal(expected, actual)
        assert owner.transferred_weight_bytes == control.transferred_weight_bytes
        assert owner.transferred_buffer_bytes == control.transferred_buffer_bytes
    finally:
        owner.close()
        control.close()
        torch.set_num_threads(previous)
    assert not list(tmp_path.rglob("da3-*"))


def test_startup_validation_round_trip_and_composition() -> None:
    settings = OptimizationConfig(da3_startup="cpu-first", da3_weights="python", reuse_da3=True)
    config = InferenceConfig.from_optimizations(settings, Memory(offload="ram"))
    assert config.da3.startup == "cpu-first"
    assert config.optimizations() == settings
    assert InferenceConfig.from_dict(config.as_dict()) == config
    assert load_configuration(profile="cpu-first-combined").config.storage.commits == "threaded"
    for mode in ("none", "streamed"):
        with pytest.raises(ValueError, match="requires RAM or disk"):
            OptimizationConfig(da3_startup="cpu-first").validate_mode(mode)
    with pytest.raises(ValueError, match="da3_startup"):
        OptimizationConfig(da3_startup="typo")


def test_failure_placing_unpaged_state_cleans_all_hooks_and_cold_archives(tmp_path: Path) -> None:
    class FailingPager(DA3Pager):
        def activate_unpaged(self, model: torch.nn.Module) -> None:
            raise RuntimeError("injected placement failure")

    model = tiny_backbone()
    original = model.get_intermediate_layers.__func__
    with pytest.raises(RuntimeError, match="injected placement failure"):
        FailingPager(model, mode="disk", device="cpu", scratch_dir=tmp_path, startup="cpu-first")
    assert model.get_intermediate_layers.__func__ is original
    assert all(not block._forward_hooks and not block._forward_pre_hooks for block in model.blocks)
    assert not list(tmp_path.glob("da3-*"))


def test_injected_pager_receives_startup_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from depth_anything_3.api import DepthAnything3

    from relift.optimization import InferenceComponents
    from relift.providers.da3 import DepthAnything3Provider

    model = tiny_backbone()
    monkeypatch.setattr(DepthAnything3, "from_pretrained", lambda *args, **kwargs: model)
    selected: list[str] = []

    def factory(*args: Any, **kwargs: Any) -> DA3Pager:
        selected.append(kwargs["startup"])
        return DA3Pager(*args, **kwargs)

    provider = DepthAnything3Provider(
        tmp_path,
        device="cpu",
        offload="ram",
        scratch_dir=str(tmp_path),
        optimizations=OptimizationConfig(da3_startup="cpu-first"),
        components=InferenceComponents(
            pager_factory=factory, fingerprints=(("pager_factory", "test"),)
        ),
    )
    try:
        assert selected == ["cpu-first"]
        assert provider.pager is not None
        assert provider.pager.startup_metrics["cold_parameter_bytes"] > 0
    finally:
        provider.close()
