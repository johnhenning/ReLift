"""4RC as a geometry provider: the frames go once through 4RC with the settings of its own evaluation (size 512,
patch 14, no crop, square_ok), and from each view the camera-to-world extrinsic -> camera-from-world pose relative
to frame 0, the world point map -> z-depth in that frame's camera, and the intrinsics are resampled onto the DA3
grid (nearest neighbour; the identity when 4RC's grid at size 512 equals the DA3 grid).  Nothing else of 4RC's
output is used.  Two checks per clip: ``check_px`` (median reprojection error of 4RC's frame-0 point map through
its own intrinsics) and ``check_rel`` (median relative difference between the frame-0 depth and the z of 4RC's own
track field at the scored query pixels; must be about 0: the same map)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..records.model import RC4Geometry, WorldTrackClip
from ..vendor import rc4_eval


class RC4Provider:
    def __init__(
        self,
        weights: str | Path,
        device: str = "cuda",
        size: int = 512,
        patch_size: int = 14,
        crop: bool = False,
        square_ok: bool = True,
    ) -> None:
        rc4_eval()
        from arc.models.arc import Arc

        self.device = device
        self.model = Arc.from_pretrained(weights).to(device).eval()
        self.view_kw = dict(size=size, patch_size=patch_size, crop=crop, square_ok=square_ok)

    def run(self, clip: WorldTrackClip, h: int, w: int) -> RC4Geometry:
        """clip: relift.data.worldtrack.load_clip output (PIL frames, tracks_uv, visibility); (h, w): the DA3 grid.
        -> dict(depth (T,h,w) f32, poses (T,4,4) f32, intrinsics (T,3,3) f32, model_hw, valid_depth, check_px, check_rel)."""
        import track_eval_dense as TD

        frames = clip["frames"]
        views = TD._prepare_views_from_pil_list(frames, **self.view_kw)
        for v in views:
            v["track_query_idx"] = np.array([0], dtype=np.int64)
        views = TD.collate_with_cat([tuple(views)])
        for v in views:
            v["img"] = v["img"].to(self.device)
        with torch.no_grad():
            pr = self.model(views, inference_track=True)
        c2w = np.stack([p["extrinsic"].float().cpu().numpy().reshape(4, 4) for p in pr]).astype(
            np.float64
        )
        w2c = np.linalg.inv(c2w)
        w2c = w2c @ np.linalg.inv(w2c[0])  # frame 0 = world
        pts = np.concatenate([p["pts"].float().cpu().numpy() for p in pr], 0).astype(
            np.float64
        )  # (T,H',W',3)
        w0 = np.linalg.inv(c2w[0])  # 4RC world -> frame-0 camera
        pts = np.einsum("ij,thwj->thwi", w0[:3, :3], pts) + w0[:3, 3]
        depth = np.einsum("tij,thwj->thwi", w2c[:, :3, :3], pts)[..., 2] + w2c[:, None, None, 2, 3]
        K = np.stack([p["intrinsic"].float().cpu().numpy().reshape(3, 3) for p in pr]).astype(
            np.float64
        )
        Hm, Wm = depth.shape[1:]
        gv, gu = np.mgrid[0:Hm, 0:Wm]
        P0 = pts[0]
        zz = np.clip(P0[..., 2], 1e-6, None)
        pu, pv = (
            K[0, 0, 0] * P0[..., 0] / zz + K[0, 0, 2],
            K[0, 1, 1] * P0[..., 1] / zz + K[0, 1, 2],
        )
        ok = P0[..., 2] > 0
        check_px = float(np.median(np.hypot(pu - gu, pv - gv)[ok])) if ok.any() else float("nan")
        tr0 = np.concatenate([p["track"].float().cpu().numpy() for p in pr[:1]], 0)[
            0
        ]  # (H',W',3) frame-0 field
        mask = np.asarray(clip["visibility"])[0].astype(bool)
        q = np.array(clip["tracks_uv"])[0, mask] * np.array([Wm, Hm]) / np.array(frames[0].size)
        inb = (q[:, 0] >= 0) & (q[:, 0] < Wm) & (q[:, 1] >= 0) & (q[:, 1] < Hm)
        u, v = q[inb, 0].astype(int), q[inb, 1].astype(int)
        tz = (w0[:3, :3] @ tr0[v, u].T).T[:, 2] + w0[2, 3]
        check_rel = (
            float(np.median(np.abs(depth[0][v, u] - tz) / np.clip(depth[0][v, u], 1e-6, None)))
            if inb.any()
            else float("nan")
        )
        vi = np.clip(((np.arange(h) + 0.5) * Hm / h).astype(int), 0, Hm - 1)
        ui = np.clip(((np.arange(w) + 0.5) * Wm / w).astype(int), 0, Wm - 1)
        dg = depth[:, vi][:, :, ui]
        Kg = K.copy()
        Kg[:, 0, 0] *= w / Wm
        Kg[:, 0, 2] = (K[:, 0, 2] + 0.5) * w / Wm - 0.5
        Kg[:, 1, 1] *= h / Hm
        Kg[:, 1, 2] = (K[:, 1, 2] + 0.5) * h / Hm - 0.5
        del pr
        torch.cuda.empty_cache()
        return dict(
            depth=dg.astype(np.float32),
            poses=w2c.astype(np.float32),
            intrinsics=Kg.astype(np.float32),
            model_hw=np.array([Hm, Wm]),
            valid_depth=float((dg > 0).mean()),
            check_px=check_px,
            check_rel=check_rel,
        )
