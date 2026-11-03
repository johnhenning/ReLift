"""Exact regression for the norm reduction and cuBLAS FMA accumulation order."""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch

from relift.cuda_rays import TimelineRayBuilder, make_ray_builder, torch_ray_builder
from relift.native_backend import load_cpu

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("count", [1, 7, 8192])
@pytest.mark.parametrize("strided_rotation", [False, True])
@pytest.mark.parametrize("factory", [make_ray_builder, TimelineRayBuilder])
def test_exact_ray_arithmetic_for_nontrivial_intrinsics_and_rotation(
    count: Any, strided_rotation: Any, factory: Any
) -> None:
    if not os.environ.get("RELIFT_NATIVE_BUILD"):
        pytest.skip("explicit native build required")
    native = load_cpu()
    if not hasattr(native, "launch_rays"):
        pytest.skip("CUDA build required")
    generator = torch.Generator(device="cuda").manual_seed(42)
    coords = (
        (
            torch.rand((3, count, 2), generator=generator, device="cuda")
            * torch.tensor([504, 280], device="cuda")
        )
        .half()
        .float()
    )
    depth = torch.rand((3, 280, 504), generator=generator, device="cuda") + 1
    K = torch.tensor([[[459.3, 0, 251.2], [0, 463.7, 139.9], [0, 0, 1]]] * 3, device="cuda")
    rotation = torch.linalg.qr(torch.randn((3, 3, 3), generator=generator, device="cuda"))[0]
    rotation = rotation.transpose(-1, -2) if strided_rotation else rotation.contiguous()
    expected = torch_ray_builder(coords, depth, K, rotation)
    actual = factory(native)(coords, depth, K, rotation)
    for a, b in zip(actual, expected, strict=False):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
