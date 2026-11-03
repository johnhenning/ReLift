"""CPU parity checks use the actual pinned DA3 backbone with tiny dimensions.

These validate traversal, reference restoration, cold features and callbacks.
They do not establish CUDA memory capacity or end-to-end geometry accuracy.
"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
from depth_anything_3.model.dinov2.vision_transformer import DinoVisionTransformer
from torch import nn

from relift.da3_offload import ColdStorage, DA3Pager


@pytest.fixture(autouse=True)
def one_cpu_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_backbone(cat_token: bool = True) -> Any:
    model = DinoVisionTransformer(
        img_size=28,
        patch_size=14,
        embed_dim=32,
        depth=4,
        num_heads=4,
        mlp_ratio=2,
        alt_start=2,
        qknorm_start=2,
        rope_start=2,
        cat_token=cat_token,
        drop_path_rate=0.0,
    )
    for index, block in enumerate(model.blocks):
        block.register_buffer("relift_test_buffer", torch.tensor([index], dtype=torch.int64))
    return model.eval()


def assert_nested_equal(left: Any, right: Any) -> None:
    if torch.is_tensor(left):
        assert left.device.type == right.device.type == "cpu"
        assert torch.equal(left, right)
    else:
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=False):
            assert_nested_equal(a, b)


@pytest.mark.parametrize("mode", ["ram", "disk"])
@pytest.mark.parametrize("strategy", ["middle", "saddle_balanced"])
def test_real_backbone_matches_original_with_reference_and_aux_features(
    tmp_path: Path, mode: str, strategy: Any
) -> None:
    torch.manual_seed(7)
    original = tiny_backbone()
    candidate = deepcopy(original)
    images = torch.randn(1, 5, 3, 28, 28)
    state = {name: value.clone() for name, value in candidate.state_dict().items()}
    original_method = candidate.get_intermediate_layers.__func__
    with torch.inference_mode():
        expected = original.get_intermediate_layers(
            images.clone(), n=[1, 3], export_feat_layers=[0, 3], ref_view_strategy=strategy
        )
        pager = DA3Pager(candidate, mode=mode, device="cpu", scratch_dir=tmp_path)
        cold_root = pager.storage.root
        try:
            actual = candidate.get_intermediate_layers(
                images.clone(), n=[1, 3], export_feat_layers=[0, 3], ref_view_strategy=strategy
            )
            assert_nested_equal(expected, actual)
            assert pager.transferred_weight_bytes > 0
            if mode == "disk":
                assert pager.storage.bytes_written > 0
                assert list(cold_root.glob("tensor-*.pt"))
            else:
                assert pager.storage.bytes_written == 0
            for name, value in candidate.state_dict().items():
                assert torch.equal(value, state[name]), name
            # A repeated invocation must not carry state from the previous traversal.
            again = candidate.get_intermediate_layers(
                images.clone(), n=[1, 3], export_feat_layers=[0, 3], ref_view_strategy=strategy
            )
            assert_nested_equal(expected, again)
        finally:
            pager.close()
    assert not cold_root.exists()
    assert candidate.get_intermediate_layers.__func__ is original_method
    assert not hasattr(candidate, "_relift_cold_storage")
    assert all(
        not block._forward_pre_hooks and not block._forward_hooks for block in candidate.blocks
    )
    for name, value in candidate.state_dict().items():
        assert torch.equal(value, state[name]), name


@pytest.mark.parametrize("mode", ["ram", "disk"])
def test_integer_layer_selection_without_concatenation_preserves_values(
    tmp_path: Path, mode: str
) -> None:
    torch.manual_seed(12)
    original = tiny_backbone(cat_token=False)
    candidate = deepcopy(original)
    images = torch.randn(1, 2, 3, 28, 28)  # Below the reference-selection threshold.
    with torch.inference_mode():
        expected = original.get_intermediate_layers(images.clone(), n=2, export_feat_layers=[])
        pager = DA3Pager(candidate, mode=mode, device="cpu", scratch_dir=tmp_path)
        try:
            actual = candidate.get_intermediate_layers(images.clone(), n=2, export_feat_layers=[])
            assert_nested_equal(expected, actual)
        finally:
            pager.close()


def test_cold_disk_storage_is_mapped_and_cleanup_preserves_live_values(tmp_path: Path) -> None:
    storage = ColdStorage("disk", tmp_path)
    tensor = torch.arange(24, dtype=torch.float32).reshape(3, 8).transpose(0, 1)
    mapped = storage.put(tensor)
    assert mapped.is_contiguous()
    assert torch.equal(mapped, tensor)
    assert storage.bytes_written > tensor.numel() * tensor.element_size()
    files = list(storage.root.glob("tensor-*.pt"))
    assert len(files) == 1
    maps = Path("/proc/self/maps")
    if maps.exists():
        assert str(files[0]) in maps.read_text()
    root = storage.root
    storage.close()
    assert not root.exists()
    # POSIX mappings retain the tensor until its own storage is released.
    assert torch.equal(mapped, tensor)


def test_weight_and_buffer_callbacks_restore_values_after_forward_error(tmp_path: Path) -> None:
    class FailingBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([3.0]))
            self.register_buffer("offset", torch.tensor([2.0]))

        def forward(self, values: Any) -> None:
            raise RuntimeError("intentional test failure")

    model = nn.Module()
    block = FailingBlock()
    model.add_module("block", block)
    before = {name: value.clone() for name, value in block.state_dict().items()}
    pager = DA3Pager(model, mode="disk", device="cpu", scratch_dir=tmp_path)
    pager.page_block(block)
    try:
        with pytest.raises(RuntimeError, match="intentional test failure"):
            block(torch.ones(1))
        for name, value in block.state_dict().items():
            assert torch.equal(value, before[name])
        assert pager.transferred_weight_bytes == block.weight.numel() * block.weight.element_size()
    finally:
        pager.close()
    assert not block._forward_pre_hooks and not block._forward_hooks
