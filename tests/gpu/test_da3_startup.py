"""Real CUDA ownership and exact tiny-backbone outputs for startup placement."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from test_da3_offload import tiny_backbone

from relift.da3_offload import DA3Pager

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("mode", ["ram", "disk"])
@pytest.mark.parametrize("packing", ["current", "python", "cpp"])
def test_cpu_first_cuda_residency_outputs_and_reactivation(
    tmp_path: Path, mode: str, packing: str
) -> None:
    from relift.native_backend import load_cpu

    torch.manual_seed(23)
    original = tiny_backbone()
    candidate = deepcopy(original)
    native = load_cpu() if packing == "cpp" else None
    control = DA3Pager(
        original,
        mode=mode,
        device="cuda",
        scratch_dir=tmp_path / "control",
        weight_backend=packing,
        native=native,
    )
    owner = DA3Pager(
        candidate,
        mode=mode,
        device="cuda",
        scratch_dir=tmp_path / "candidate",
        weight_backend=packing,
        native=native,
        startup="cpu-first",
    )
    images = torch.randn(1, 5, 3, 28, 28, device="cuda")
    try:
        assert control.startup_metrics["initial_cold_roundtrip_tensor_bytes"] > 0
        assert owner.startup_metrics["initial_cold_roundtrip_tensor_bytes"] == 0
        cold = {id(parameter) for weights in owner.weights for parameter, _ in weights}
        for parameter in candidate.parameters():
            assert parameter.device.type == ("cpu" if id(parameter) in cold else "cuda")
        for key, value in original.state_dict().items():
            assert torch.equal(value, candidate.state_dict()[key]), key
        with torch.inference_mode():
            for _ in range(2):
                expected, expected_aux = original.get_intermediate_layers(
                    images, n=[1, 3], export_feat_layers=[0, 3], ref_view_strategy="saddle_balanced"
                )
                actual, actual_aux = candidate.get_intermediate_layers(
                    images, n=[1, 3], export_feat_layers=[0, 3], ref_view_strategy="saddle_balanced"
                )
                for a, b in zip(expected, actual, strict=True):
                    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
                assert all(torch.equal(a, b) for a, b in zip(expected_aux, actual_aux, strict=True))
                # Simulate parking a reused provider, then reactivate only its hot state.
                candidate.to("cpu")
                owner.activate_unpaged(candidate)
        assert owner.transferred_weight_bytes == control.transferred_weight_bytes
    finally:
        owner.close()
        control.close()
