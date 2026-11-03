"""AllTracker: dense 2-D tracks of every pixel of frame 0 (released weights, temporal window 16, 4 refinement
iterations, the whole sequence in one ``forward`` call) on DA3's processed frames."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..vendor import front_ends


class AllTrackerProvider:
    def __init__(
        self,
        weights: str | Path,
        device: str | torch.device = "cuda",
        window: int = 16,
        iters: int = 4,
        init_weights: bool = False,
        copy_outputs: bool = False,
    ) -> None:
        front_ends()
        from nets.alltracker import Net

        self.device, self.iters = torch.device(device), iters
        net = Net(window, init_weights=init_weights)
        self.copy_outputs = copy_outputs
        state = torch.load(weights, map_location="cpu")
        net.load_state_dict(state["model"], strict=True)
        self.net = net.eval().to(self.device)
        for p in self.net.parameters():
            p.requires_grad = False

    def run(
        self, images: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """images (T,H,W,3) uint8 -> tracks (T,H,W,2) f32 pixel positions (frame 0 = the pixel grid),
        visibility (T,H,W) f32, confidence (T,H,W) f32."""
        x = (
            torch.from_numpy(images).float().permute(0, 3, 1, 2)[None].to(self.device)
        )  # (1,T,3,H,W), 0..255
        _, T, _, H, W = x.shape
        ys, xs = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        grid = torch.stack([xs, ys])[None, None]  # (1,1,2,H,W), (x, y)
        with torch.no_grad():
            flows, visconf, _, _ = self.net(x, iters=self.iters, sw=None, is_training=False)
            flows[:, 0] = 0.0
            traj = (flows + grid).permute(0, 1, 3, 4, 2)[0]  # (T,H,W,2)
            visconf = visconf.permute(0, 1, 3, 4, 2)[0]  # (T,H,W,2)
        return (
            traj.cpu().numpy().astype(np.float32, copy=self.copy_outputs),
            visconf[..., 0].cpu().numpy().astype(np.float32, copy=self.copy_outputs),
            visconf[..., 1].cpu().numpy().astype(np.float32, copy=self.copy_outputs),
        )
