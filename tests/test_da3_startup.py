"""CPU-first pager setup preserves state, ownership, cleanup and configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
from test_da3_offload import assert_nested_equal, tiny_backbone

from relift.configuration import InferenceConfig, Memory, load_configuration
from relift.da3_offload import DA3Pager
from relift.optimization import OptimizationConfig


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
        original, mode=mode, device="cpu", scratch_dir=tmp_path / "control", weight_backend=packing
    )
    owner = DA3Pager(
        candidate,
        mode=mode,
        device="cpu",
        scratch_dir=tmp_path / "candidate",
        weight_backend=packing,
        startup="cpu-first",
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
