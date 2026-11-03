"""VGGT-Omega (vggt_omega_1b_512): depth, intrinsics and poses of all frames jointly with its released "balanced"
loader at 512, resampled to the DA3 grid; intrinsics rescaled accordingly."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..records.model import VGGTGeometry


class VGGTOmegaProvider:
    def __init__(
        self,
        weights: str | Path,
        device: str | torch.device = "cuda",
        resolution: int = 512,
        patch: int = 16,
    ) -> None:
        from vggt_omega.models import VGGTOmega

        self.device, self.res, self.patch = torch.device(device), resolution, patch
        self.model = VGGTOmega().to(self.device).eval()
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))

    def run(self, frames: np.ndarray[Any, Any], H: int, W: int) -> VGGTGeometry:
        """frames (T,h0,w0,3) uint8 RGB, (H, W) the DA3 grid -> dict(depth (T,H,W) f32, poses (T,4,4) f32
        camera-from-world, intrinsics (T,3,3) f32, depth_conf (T,H,W) f32)."""
        from PIL import Image
        from torchvision import transforms as TF
        from vggt_omega.utils.load_fn import _balanced_target_shape, _crop_to_supported_aspect_ratio
        from vggt_omega.utils.pose_enc import encoding_to_camera

        images = []
        for frame in frames:  # load_and_preprocess_images(mode="balanced"), in memory
            image = Image.fromarray(frame)
            cropped = _crop_to_supported_aspect_ratio(image)
            if (
                cropped.size != image.size
            ):  # never happens at the WorldTrack aspect ratios; a crop would
                raise RuntimeError(
                    f"VGGT-Omega loader cropped {image.size} -> {cropped.size}; "  # misalign the tracks
                    "unsupported aspect ratio for this pipeline"
                )
            th, tw = _balanced_target_shape(
                cropped.size[1] / max(cropped.size[0], 1), self.res, self.patch
            )
            images.append(TF.ToTensor()(cropped.resize((tw, th), Image.Resampling.BICUBIC)))
        with torch.no_grad():
            pred = self.model(torch.stack(images)[None].to(self.device))
            Hi, Wi = pred["images"].shape[-2:]
            extr, intr = encoding_to_camera(pred["pose_enc"], (Hi, Wi))
            depth = F.interpolate(
                pred["depth"][0, ..., 0].float()[:, None],
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            conf = F.interpolate(
                pred["depth_conf"][0].float()[:, None],
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
        if intr is None:
            raise ValueError("VGGT-Omega did not return intrinsics")
        K = intr[0].cpu().numpy().astype(np.float64)
        K[:, 0, 0] *= W / Wi
        K[:, 0, 2] *= W / Wi
        K[:, 1, 1] *= H / Hi
        K[:, 1, 2] *= H / Hi
        poses = np.tile(np.eye(4), (len(frames), 1, 1))
        poses[:, :3, :4] = extr[0].cpu().numpy().astype(np.float64)
        return dict(
            depth=depth.cpu().numpy().astype(np.float32),
            poses=poses.astype(np.float32),
            intrinsics=K.astype(np.float32),
            depth_conf=conf.cpu().numpy().astype(np.float32),
            internal_hw=[int(Hi), int(Wi)],
        )
