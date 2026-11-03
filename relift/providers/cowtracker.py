"""CoWTracker as the dense 2-D tracker: one joint pass over the clip in fp16 at the largest size with both sides
multiples of 112 within a pixel budget, the flow field resampled to the DA3 grid (frame 0 = exact grid).

Attention path.  CoWTracker's ViT blocks call xformers' flash attention.  The published CoWTracker windows were
produced on GPUs for which that kernel is not built, through a fallback to ``torch.nn.functional.scaled_dot_product_attention``
(a local patch of the research checkout, logged in every run).  This provider installs that fallback
unconditionally, so the released code follows the same arithmetic on every GPU and does not need xformers.
"""

from __future__ import annotations

import math
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..records.model import VisibilityStats

MULT = 112  # lcm of the ViT patch (14) and the side ResNet's stride (16)
ASPECT_TOL = math.log(1.15)


def cow_size(
    Hn: int, Wn: int, max_pixels: int = 240000, fixed: tuple[int, int] | None = None
) -> tuple[int, int]:
    """Among the 112-multiple sizes with H*W <= max_pixels and |log((H/W)/(Hn/Wn))| <= log(1.15), the one with the
    most pixels (ties: least aspect distortion).  240k px: 280x504 -> 336x672, 504x504 -> 448x448."""
    if fixed:
        H, W = fixed
        assert H % MULT == 0 and W % MULT == 0, fixed
        return H, W
    best = None
    kmax = int(math.sqrt(max_pixels) // MULT) * 4 + 4
    for a in range(1, kmax):
        for b in range(1, kmax):
            H, W = a * MULT, b * MULT
            if H * W > max_pixels:
                continue
            d = abs(math.log((H / W) / (Hn / Wn)))
            if d > ASPECT_TOL:
                continue
            key = (H * W, -d)
            if best is None or key > best[0]:
                best = (key, (H, W))
    assert best is not None, (Hn, Wn, max_pixels)
    return best[1]


def _sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    scale: float | None,
) -> torch.Tensor:
    """torch equivalent of xformers' memory_efficient_attention on (B, M, H, K) inputs."""
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        scale=scale,
    )
    return out.transpose(1, 2)


def _install_sdpa_attention() -> None:
    """Make ``cowtracker.layers.video_transformer`` importable without xformers and route its attention through
    ``_sdpa`` (the path of the published runs)."""
    try:
        import xformers.ops  # noqa: F401
    except ImportError:
        stub, ops = types.ModuleType("xformers"), types.ModuleType("xformers.ops")
        stub.__dict__["ops"] = ops
        sys.modules.setdefault("xformers", stub)
        sys.modules.setdefault("xformers.ops", ops)
    import cowtracker.layers.video_transformer as vt

    vt._get_flash_attention_ops = lambda: None

    def forward(
        self: Any, x: torch.Tensor, attn_mask: torch.Tensor | None = None, is_causal: bool = False
    ) -> torch.Tensor:
        assert not is_causal, "causal attention is not used by CoWTracker's ViT blocks"
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        x = _sdpa(q, k, v, attn_mask, self.attn_drop if self.training else 0.0, self.scale)
        x = x.reshape(B, N, C)
        return self.proj_drop(self.proj(x))

    vt.FlashAttention3.forward = forward


class CoWTrackerProvider:
    def __init__(
        self,
        weights: str | Path,
        device: str = "cuda",
        dtype: str = "fp16",
        attention: str = "sdpa",
    ) -> None:
        assert attention == "sdpa", "only the published attention path is supported"
        _install_sdpa_attention()
        from cowtracker import CoWTrackerWindowed

        self.device = device
        self.dtype = dict(fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32)[dtype]
        self.model: Any = CoWTrackerWindowed.from_checkpoint(
            str(weights), device=device, dtype=self.dtype
        )

    def run(
        self,
        frames: np.ndarray[Any, Any],
        Hn: int,
        Wn: int,
        max_pixels: int = 240000,
        fixed: tuple[int, int] | None = None,
    ) -> tuple[
        tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]],
        tuple[int, int],
        float,
        float,
    ]:
        """frames (T,h,w,3) uint8 RGB (Pillow-decoded), native grid (Hn, Wn) -> (tracks (T,Hn,Wn,2) f32 in native
        pixels with frame 0 = exact grid, visibility, confidence (T,Hn,Wn) f32 in [0,1]), size used, seconds, peak GiB."""
        size = cow_size(Hn, Wn, max_pixels, fixed)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        video = (
            torch.as_tensor(frames, device=self.device).permute(0, 3, 1, 2).float()
        )  # (T,3,h,w) in 0..255
        if video.shape[-2:] != size:
            video = F.interpolate(
                video, size=size, mode="bilinear", align_corners=False, antialias=True
            )
        with (
            torch.no_grad(),
            torch.amp.autocast(
                device_type="cuda", dtype=self.dtype, enabled=self.device.startswith("cuda")
            ),
        ):
            pred = self.model.model(video)  # one joint pass
        tr, vis, conf = pred["track"][0].float(), pred["vis"][0].float(), pred["conf"][0].float()
        torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        tr, vis, conf = _to_native(tr, vis, conf, Hn, Wn)
        out = (
            tr.cpu().numpy().astype(np.float32),
            vis.cpu().numpy().astype(np.float32),
            conf.cpu().numpy().astype(np.float32),
        )
        peak = torch.cuda.max_memory_allocated() / 2**30
        del tr, vis, conf, pred, video
        torch.cuda.empty_cache()
        return out, size, secs, peak


def _to_native(
    tracks: torch.Tensor,
    vis: torch.Tensor,
    conf: torch.Tensor,
    Hn: int,
    Wn: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resample the flow field to the native grid (bilinear, corner-aligned) and rescale it to native pixel units."""
    T, Hr, Wr, _ = tracks.shape
    dev = tracks.device
    gy, gx = torch.meshgrid(
        torch.arange(Hr, device=dev, dtype=torch.float32),
        torch.arange(Wr, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    flow = tracks - torch.stack([gx, gy], -1)[None]  # (T,Hr,Wr,2)
    if (Hr, Wr) == (Hn, Wn):
        return tracks, vis, conf
    sx, sy = (Wn - 1) / max(Wr - 1, 1), (Hn - 1) / max(Hr - 1, 1)
    stack = torch.cat([flow.permute(0, 3, 1, 2), vis[:, None], conf[:, None]], 1)  # (T,4,Hr,Wr)
    out = torch.cat(
        [
            F.interpolate(stack[i : i + 8], size=(Hn, Wn), mode="bilinear", align_corners=True)
            for i in range(0, T, 8)
        ]
    )
    gy, gx = torch.meshgrid(
        torch.arange(Hn, device=dev, dtype=torch.float32),
        torch.arange(Wn, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    tr = torch.stack([gx + out[:, 0] * sx, gy + out[:, 1] * sy], -1)
    tr[0] = torch.stack([gx, gy], -1)  # frame 0: exact grid
    return tr, out[:, 2].clamp(0, 1), out[:, 3].clamp(0, 1)


def vis_stats(vis: np.ndarray[Any, Any], name: str = "") -> VisibilityStats:
    v = np.asarray(vis, np.float32).ravel()
    q = np.percentile(v, [10, 25, 50, 75, 90])
    return dict(
        name=name,
        mean=float(v.mean()),
        p10=float(q[0]),
        p25=float(q[1]),
        p50=float(q[2]),
        p75=float(q[3]),
        p90=float(q[4]),
        frac_gt_085=float((v > 0.85).mean()),
        frac_gt_05=float((v > 0.5).mean()),
        hist=np.histogram(v, bins=10, range=(0, 1))[0].tolist(),
    )


def schema_check(
    tracks: np.ndarray[Any, Any],
    vis: np.ndarray[Any, Any],
    conf: np.ndarray[Any, Any],
    T: int,
    H: int,
    W: int,
) -> None:
    assert tracks.shape == (T, H, W, 2), tracks.shape
    assert vis.shape == (T, H, W) and conf.shape == (T, H, W), (vis.shape, conf.shape)
    gy, gx = np.mgrid[0:H, 0:W]
    assert (
        np.abs(tracks[0, ..., 0] - gx).max() < 1e-3 and np.abs(tracks[0, ..., 1] - gy).max() < 1e-3
    ), "frame 0 is not the grid"
    assert (
        np.isfinite(tracks).all()
        and vis.min() >= 0
        and vis.max() <= 1
        and conf.min() >= 0
        and conf.max() <= 1
    )
