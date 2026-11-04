"""Depth Anything 3 (DA3NESTED-GIANT-LARGE-1.1): depth, intrinsics and camera-from-world poses of all frames jointly."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..optimization import InferenceComponents, OptimizationConfig
from ..records.model import DA3Geometry


class DepthAnything3Provider:
    def __init__(
        self,
        weights: str | Path,
        device: str | torch.device = "cuda",
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        ref_view_strategy: str = "saddle_balanced",
        offload: str = "none",
        scratch_dir: str = "cache/offload",
        ram_budget_gb: float = 18,
        scratch_budget_gb: float = 64,
        optimizations: OptimizationConfig | None = None,
        components: InferenceComponents | None = None,
    ) -> None:
        from depth_anything_3.api import DepthAnything3

        startup = optimizations.da3_startup if optimizations else "current"
        if startup != "current" and offload not in ("ram", "disk"):
            raise ValueError("CPU-first DA3 startup requires RAM or disk offload")
        if optimizations and (
            optimizations.da3_weights != "current" or optimizations.da3_features != "current"
        ):
            if offload not in ("ram", "disk"):
                raise ValueError("Packed DA3 backends require RAM or disk offload")
            if optimizations.da3_features != "current" and offload != "disk":
                raise ValueError("Packed feature archives require disk offload")
        self.model: DepthAnything3 | None = DepthAnything3.from_pretrained(weights)
        if startup == "current":
            self.model.to(device)
        self.device = torch.device(device)
        self.pager = None
        if offload in ("ram", "disk"):
            from ..da3_offload import DA3Pager

            factory = (
                components.pager_factory if components and components.pager_factory else DA3Pager
            )
            native = None
            if optimizations and "cpp" in (optimizations.da3_weights, optimizations.da3_features):
                from ..native_backend import load_cpu

                native = (
                    components.native_loader()
                    if components and components.native_loader
                    else load_cpu()
                )
            self.pager = factory(
                self.model,
                mode=offload,
                device=device,
                scratch_dir=scratch_dir,
                ram_budget_gb=ram_budget_gb,
                scratch_budget_gb=scratch_budget_gb,
                weight_backend=optimizations.da3_weights if optimizations else "current",
                feature_backend=optimizations.da3_features if optimizations else "current",
                native=native,
                storage_factory=components.cold_storage_factory if components else None,
                feature_archive_factory=components.feature_archive_factory if components else None,
                block_packer=components.block_packer if components else None,
                startup=startup,
            )
        self.kw = dict(
            process_res=process_res,
            process_res_method=process_res_method,
            ref_view_strategy=ref_view_strategy,
        )

    def run(self, frames: Any) -> DA3Geometry:
        """frames (T,H,W,3) uint8 RGB -> dict(depth (T,h,w) f32, poses (T,4,4) f32, intrinsics (T,3,3) f32,
        images (T,h,w,3) uint8: the frames as DA3 processed them, which the tracker runs on)."""
        # Autocast's weight cache would retain paged blocks on CUDA until the
        # entire model returns; disable only that cache in placement modes.
        if self.model is None:
            raise RuntimeError("Depth provider is closed")
        with torch.autocast("cuda", enabled=False, cache_enabled=self.pager is None):
            pred = self.model.inference(image=[f for f in frames], **self.kw)
        if pred.extrinsics is None or pred.intrinsics is None or pred.processed_images is None:
            raise ValueError("DA3 did not return cameras, intrinsics and processed images")
        ext = pred.extrinsics.astype(np.float32)
        if ext.shape[1:] == (3, 4):
            ext = np.concatenate(
                [ext, np.tile(np.array([[[0, 0, 0, 1]]], np.float32), (len(ext), 1, 1))], 1
            )
        return dict(
            depth=pred.depth.astype(np.float32),
            poses=ext,
            intrinsics=pred.intrinsics.astype(np.float32),
            images=pred.processed_images,
        )

    def close(self) -> None:
        if self.pager is not None:
            self.pager.close()
            self.pager = None
        self.model = None

    def park(self) -> None:
        if self.model is None:
            raise RuntimeError("Depth provider is closed")
        self.model.to("cpu")

    def activate(self) -> None:
        if self.model is None:
            raise RuntimeError("Depth provider is closed")
        # Move only modules that are not explicitly paged; cold parameter storage
        # is restored by block hooks and must retain its CPU/mapped ownership.
        cold = (
            {id(p) for weights in self.pager.weights for p, _ in weights} if self.pager else set()
        )
        for parameter in self.model.parameters():
            if id(parameter) not in cold:
                parameter.data = parameter.data.to(self.device)
        paged_buffers = getattr(self.pager, "buffer_owners", set()) if self.pager else set()
        for module in self.model.modules():
            for name, value in module._buffers.items():
                if value is not None and (id(module), name) not in paged_buffers:
                    module._buffers[name] = value.to(self.device)
