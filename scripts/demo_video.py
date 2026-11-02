#!/usr/bin/env python
"""Run ReLift on a video: every pixel of the first frame -> its corrected 3-D trajectory through the clip.

    python scripts/demo_video.py assets/demo/cartwheel.mp4 [--out DIR] [--num_frames 84] [--geometry da3|vggto]
                                 [--checkpoint PATH] [--checkpoints_root DIR] [--device cuda]

VIDEO is a video file (anything OpenCV decodes) or a directory of image frames (sorted by name).  The pipeline is the
one behind Table 1: Depth Anything 3 (depth, intrinsics, cameras; it also fixes the working grid, long side 504 px),
AllTracker (dense 2-D tracks of every grid pixel on DA3's processed frames), the raw lift of the tracked pixels along
their rays, and the corrector over every pixel (``relift.infer.predict_dense``).  With ``--geometry vggto`` the lift
uses VGGT-Omega's depth and cameras instead (AllTracker still runs on DA3's frames, as in the paper).

Writes into DIR (default ``<outputs_root>/demo/<video name>/``):
    window/      the prepared window, as prepare_worldtrack.py writes it (depth, cameras, frames, tracks, provenance)
    tracks.npz   xyz (T,H,W,3) float32: the corrected trajectory of every frame-0 pixel, in the frame-0 camera's
                 coordinate system and the geometry model's units; lift (T,H,W,3) float16: the raw lift, interpolated
                 where the tracker does not observe the pixel; observed (T,H,W) bool; images (T,H,W,3) uint8;
                 poses (T,4,4) camera-from-world relative to frame 0; intrinsics (T,3,3)
    tracks.mp4   a sub-grid of the pixels projected into every frame with the geometry model's cameras: the lift on
                 the left, the corrected tracks on the right; filled dot = observed by the tracker, ring = filled in
                 (playback uses the source video's FPS; image directories default to 15 FPS).

Training uses 48, 68 and 84 frames; published evaluation uses 84. Longer clips run as one window and are
outside the published evaluation setting (``--num_frames 0`` takes the whole clip).
"""
import argparse
import os
import sys
import time
from pathlib import Path
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from relift.checkpoint import load  # noqa: E402
from relift.config import REPO, paths  # noqa: E402
from relift.features import interp_init  # noqa: E402
from relift.infer import VIS_GATE, observations, predict_dense  # noqa: E402
from relift.providers import depth_file, tracks_file  # noqa: E402
from relift.provenance import note  # noqa: E402
from relift.telemetry import Metrics  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_frames(src, num_frames):
    """(T,H,W,3) uint8 RGB from a video file or a directory of images; ``num_frames`` 0 = all."""
    frames = []
    if os.path.isdir(src):
        names = sorted(n for n in os.listdir(src) if os.path.splitext(n)[1].lower() in IMAGE_EXT)
        for n in names[: num_frames or None]:
            im = cv2.imread(os.path.join(src, n), cv2.IMREAD_COLOR)
            if im is None:
                raise SystemExit(f"cannot read {os.path.join(src, n)}")
            frames.append(im[..., ::-1])
    else:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise SystemExit(f"cannot open {src}")
        while num_frames == 0 or len(frames) < num_frames:
            ok, im = cap.read()
            if not ok:
                break
            frames.append(im[..., ::-1])
        cap.release()
    if len(frames) < 2:
        raise SystemExit(f"{src}: {len(frames)} frame(s) read; at least 2 are needed")
    if any(f.shape != frames[0].shape for f in frames):
        raise SystemExit(f"{src}: frames differ in size")
    return np.stack(frames).astype(np.uint8)


def weight_path(cfg_w, name, root):
    w = cfg_w[name]
    return os.path.join(root, w.get("dir", ""), w["file"]) if "file" in w else os.path.join(root, w["dir"])


def savez(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def prepare_window(frames, out, geometry, wpath, pc, device, offload="none", scratch_dir="cache/offload", metrics=None,
                   ram_budget_gb=18, scratch_budget_gb=64):
    """Geometry, DA3's frames and AllTracker's tracks into ``out`` (files already present are kept)."""
    os.makedirs(out, exist_ok=True)
    da3_f, frames_f, tr_f = os.path.join(out, depth_file("da3")), os.path.join(out, "da3_frames.npz"), os.path.join(out, tracks_file("alltracker"))
    if not (os.path.exists(da3_f) and os.path.exists(frames_f)):
        from relift.providers.da3 import DepthAnything3Provider
        t0 = time.perf_counter()
        with metrics.stage("da3") if metrics else nullcontext() as stage:
            provider = DepthAnything3Provider(wpath["da3"], device, offload=offload, scratch_dir=scratch_dir,
                                               ram_budget_gb=ram_budget_gb, scratch_budget_gb=scratch_budget_gb,
                                               **{k: v for k, v in pc["da3"].items() if k != "decoder"})
            try:
                g = provider.run(frames)
            finally:
                if stage is not None and provider.pager is not None:
                    stage.update(cold_tensor_bytes_written=provider.pager.storage.bytes_written,
                                 peak_retained_cpu_tensor_bytes=provider.pager.storage.peak_retained_cpu_bytes,
                                 weight_host_to_device_bytes=provider.pager.transferred_weight_bytes)
                provider.close()
                del provider
            savez(da3_f, depth=g["depth"].astype(np.float16), poses=g["poses"], intrinsics=g["intrinsics"])
            savez(frames_f, images=g["images"])
        note(os.path.join(out, "provenance.json"), "da3", device)
        print(f"da3: grid {g['depth'].shape[1]}x{g['depth'].shape[2]}, {time.perf_counter() - t0:.0f}s", flush=True)
        del g
        torch.cuda.empty_cache()
    images = np.load(frames_f)["images"]
    if len(images) != len(frames):
        raise ValueError(f"Cached window has {len(images)} frames, input has {len(frames)}; choose a fresh --out directory")
    H, W = images.shape[1:3]
    if not os.path.exists(tr_f):
        from relift.providers.alltracker import AllTrackerProvider
        t0 = time.perf_counter()
        with metrics.stage("alltracker") if metrics else nullcontext():
            tr, vis, conf = AllTrackerProvider(wpath["alltracker"], device, pc["alltracker"]["window"], pc["alltracker"]["iters"]).run(images)
            savez(tr_f, tracks_2d=tr.astype(np.float16), visibility=vis.astype(np.float16), confidence=conf.astype(np.float16))
        note(os.path.join(out, "provenance.json"), "alltracker", device)
        print(f"alltracker: {time.perf_counter() - t0:.0f}s", flush=True)
        torch.cuda.empty_cache()
    if geometry == "vggto" and not os.path.exists(os.path.join(out, depth_file("vggto"))):
        from relift.providers.vggt_omega import VGGTOmegaProvider
        t0 = time.perf_counter()
        g = VGGTOmegaProvider(wpath["vggt_omega"], device, pc["vggto"]["resolution"], pc["vggto"]["patch"]).run(frames, H, W)
        savez(os.path.join(out, depth_file("vggto")), depth=g["depth"].astype(np.float16), poses=g["poses"], intrinsics=g["intrinsics"])
        note(os.path.join(out, "provenance.json"), "vggto", device)
        print(f"vggto: {time.perf_counter() - t0:.0f}s", flush=True)
        torch.cuda.empty_cache()
    return images


def project(X, poses, K):
    """X (T,N,3) in the frame-0 camera's coordinates -> pixels (T,N,2) and depths (T,N) in every frame."""
    Xc = np.einsum("tij,tnj->tni", poses[:, :3, :3], X) + poses[:, :3, 3][:, None]
    z = Xc[..., 2]
    u = K[:, 0, 0][:, None] * Xc[..., 0] / np.where(z > 1e-6, z, np.nan) + K[:, 0, 2][:, None]
    v = K[:, 1, 1][:, None] * Xc[..., 1] / np.where(z > 1e-6, z, np.nan) + K[:, 1, 2][:, None]
    return np.stack([u, v], -1), z


def write_video(path, images, lift, xyz, observed, poses, K, stride, fps=15, up=2):
    T, H, W = images.shape[:3]
    vs, us = np.mgrid[stride // 2:H:stride, stride // 2:W:stride]
    vs, us = vs.ravel(), us.ravel()
    hue = (0.5 * us / W + 0.5 * vs / H) * 179.0                               # colour = position in frame 0
    cols = cv2.cvtColor(np.stack([hue, np.full_like(hue, 230), np.full_like(hue, 255)], -1).astype(np.uint8)[None], cv2.COLOR_HSV2BGR)[0]
    cols = [tuple(int(c) for c in col) for col in cols]
    uv = {name: project(f[:, vs, us].astype(np.float64), poses.astype(np.float64), K.astype(np.float64))[0] * up
          for name, f in (("lift", lift), ("ours", xyz))}
    ph, pw = H * up, W * up
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * pw, ph + 28))
    for t in range(T):
        base = cv2.resize(images[t][..., ::-1], (pw, ph), interpolation=cv2.INTER_LINEAR)
        panels = []
        for name, label in (("lift", "raw lift"), ("ours", "ReLift")):
            img = base.copy()
            for j, (u, v) in enumerate(uv[name][t]):
                if not (np.isfinite(u) and np.isfinite(v)):
                    continue
                p = (int(round(u * 16)), int(round(v * 16)))
                if observed[t, vs[j], us[j]]:
                    cv2.circle(img, p, 3 * 16, cols[j], -1, cv2.LINE_AA, shift=4)
                else:
                    cv2.circle(img, p, 3 * 16, (255, 255, 255), -1, cv2.LINE_AA, shift=4)
                    cv2.circle(img, p, 3 * 16, cols[j], 1, cv2.LINE_AA, shift=4)
            bar = np.full((28, pw, 3), 30, np.uint8)
            cv2.putText(bar, f"{label}   frame {t}/{T - 1}   observed {100 * observed[t].mean():.0f}%", (8, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
            panels.append(np.vstack([bar, img]))
        wr.write(np.hstack(panels))
    wr.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="a video file or a directory of image frames")
    ap.add_argument("--out", default=None, help="default <outputs_root>/demo/<video name>")
    ap.add_argument("--num_frames", type=int, default=84, help="frames taken from the start (0 = all)")
    ap.add_argument("--geometry", default="da3", choices=["da3", "vggto"])
    ap.add_argument("--checkpoint", default=None, help="default <checkpoints_root>/<configs/eval/table1.yaml corrector.checkpoint>")
    ap.add_argument("--checkpoints_root", default=None)
    ap.add_argument("--config", default=str(REPO / "configs" / "eval" / "table1.yaml"), help="corrector settings (group, batch, seed)")
    ap.add_argument("--providers_config", default=str(REPO / "configs" / "providers.yaml"))
    ap.add_argument("--process-res", type=int, help="override DA3's long-side resolution (default 504); retains all temporal frames")
    ap.add_argument("--corrector-batch", type=int, help="override the number of trajectory groups per corrector forward pass")
    ap.add_argument("--stride", type=int, default=12, help="pixel stride of the points drawn in tracks.mp4")
    ap.add_argument("--no_video", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--offload", choices=["none", "streamed", "ram", "disk", "auto"], default="none")
    ap.add_argument("--ram-budget-gb", type=float, default=18)
    ap.add_argument("--vram-budget-gb", type=float, default=20)
    ap.add_argument("--scratch-budget-gb", type=float, default=64)
    ap.add_argument("--scratch-dir", default=None)
    ap.add_argument("--export-npz", action="store_true", help="also export compatibility tracks.npz from a mapped store")
    ap.add_argument("--prepared-window", help="reuse an explicitly chosen prepared window for stage-isolated correction")
    a = ap.parse_args()
    if not all(np.isfinite(x) and x > 0 for x in (a.ram_budget_gb, a.vram_budget_gb, a.scratch_budget_gb)):
        ap.error("memory and scratch budgets must be positive and finite")
    if a.num_frames < 0:
        ap.error("--num_frames must be nonnegative")
    if a.process_res is not None and (a.process_res < 28 or a.process_res % 14):
        ap.error("--process-res must be a multiple of 14 and at least 28")
    if a.corrector_batch is not None and a.corrector_batch < 1:
        ap.error("--corrector-batch must be positive")
    if not a.device.startswith("cuda"):
        raise SystemExit("CUDA is required: the published numbers correspond to the CUDA path")
    P = paths()
    cfg = yaml.safe_load(open(a.config))
    pc = yaml.safe_load(open(a.providers_config))
    if a.process_res is not None:
        pc["da3"]["process_res"] = a.process_res
    if a.corrector_batch is not None:
        cfg["corrector"]["batch"] = a.corrector_batch
    wc = yaml.safe_load(open(REPO / "configs" / "weights.yaml"))
    ck_root = a.checkpoints_root or str(P["checkpoints_root"])
    wpath = {n: weight_path(wc, n, ck_root) for n in wc}
    ckpt = a.checkpoint or os.path.join(ck_root, cfg["corrector"]["checkpoint"])
    name = os.path.splitext(os.path.basename(os.path.normpath(a.video)))[0]
    out = a.out or str(P["outputs_root"] / "demo" / name)
    os.makedirs(out, exist_ok=True)
    metrics = Metrics(out, dict(offload_requested=a.offload, process_resolution=pc["da3"]["process_res"],
                               corrector_batch=cfg["corrector"]["batch"], ram_budget_gb=a.ram_budget_gb,
                               vram_budget_gb=a.vram_budget_gb, scratch_budget_gb=a.scratch_budget_gb))
    with metrics.stage("decode", cuda=False):
        if a.offload == "none":
            frames = read_frames(a.video, a.num_frames)
        else:
            from relift.storage import read_frames_mapped
            frames = read_frames_mapped(a.video, a.num_frames, Path(out) / "decoded")
    T = len(frames)
    source_fps = 15.0
    if not os.path.isdir(a.video):
        cap = cv2.VideoCapture(a.video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if np.isfinite(fps) and fps > 0:
            source_fps = float(fps)
    print(f"{a.video}: {T} frames of {frames.shape[2]}x{frames.shape[1]} -> {out}", flush=True)
    resolution = pc["da3"]["process_res"]
    resolved = a.offload
    if resolved == "auto":
        estimated_points = resolution * max(14, int(resolution * min(frames.shape[1:3]) / max(frames.shape[1:3]) / 14) * 14)
        estimated_gb = 6 + .12 * T * estimated_points / (504 * 280)
        estimated_cold_gb = 6 + .05 * T * estimated_points / (504 * 280)
        resolved = ("disk" if estimated_cold_gb > a.ram_budget_gb else "ram") if estimated_gb > a.vram_budget_gb else "streamed"
        metrics.data["settings"]["estimated_da3_vram_gb"] = estimated_gb
        metrics.data["settings"]["estimated_da3_cold_gb"] = estimated_cold_gb
    metrics.data["settings"]["offload_resolved"] = resolved
    metrics.save()
    window = a.prepared_window or os.path.join(out, "window" if resolution == 504 else f"window_res{resolution}")
    images = prepare_window(frames, window, a.geometry, wpath, pc, a.device,
                            offload=resolved, scratch_dir=a.scratch_dir or str(P["cache_root"] / "offload"), metrics=metrics,
                            ram_budget_gb=a.ram_budget_gb, scratch_budget_gb=a.scratch_budget_gb)
    del frames

    with metrics.stage("corrector_load"):
        model, info = load(ckpt, a.device)
    C = cfg["corrector"]
    df = depth_file(a.geometry)
    if resolved in ("streamed", "ram", "disk"):
        from relift.streaming import stream_correct
        from relift.storage import file_sha256
        metadata = dict(geometry=a.geometry, checkpoint_step=info["step"], checkpoint_sha256=file_sha256(ckpt),
                        source_fps=source_fps, source_frames=T, requested_frames=a.num_frames,
                        process_resolution=resolution, corrector_batch=C["batch"], offload=resolved)
        if os.path.isfile(a.video):
            metadata["input_sha256"] = file_sha256(a.video)
        with metrics.stage("corrector_stream"):
            store = stream_correct(window, out, model, metadata, group=C["group"], batch=C["batch"], seed=C["seed"],
                                   device=a.device, depth_file=df, ram_budget_gb=a.ram_budget_gb,
                                   vram_budget_gb=a.vram_budget_gb, scratch_budget_gb=a.scratch_budget_gb)
        if a.export_npz or not a.no_video:
            with metrics.stage("export_npz", cuda=False):
                store.export_npz(Path(out) / "tracks.npz")
        if not a.no_video:
            with np.load(Path(out) / "tracks.npz") as arrays:
                write_video(os.path.join(out, "tracks.mp4"), arrays["images"], arrays["lift"], arrays["xyz"], arrays["observed"],
                            arrays["poses"], arrays["intrinsics"], a.stride, fps=source_fps)
        print(f"DEMO_DONE: {out}/tracks.store/manifest.json", flush=True)
        return
    t0 = time.perf_counter()
    o, d, r, vis = observations(window, T, a.device, tracks_file("alltracker"), df)
    observed = (vis > VIS_GATE) & (r > 0)
    observed[0] = True
    xyz = predict_dense(model, o, d, r, vis, group=C["group"], batch=C["batch"], seed=C["seed"])
    Y = (o[:, None, None, :] + d * r[..., None]) * observed[..., None]
    H, W = r.shape[1:]
    lift = interp_init(Y.reshape(1, T, H * W, 3), observed.reshape(1, T, H * W))[0].reshape(T, H, W, 3)
    print(f"corrector (step {info['step']}): {T}x{H}x{W} pixels in {time.perf_counter() - t0:.1f}s; "
          f"observed {100 * observed.float().mean():.0f}% of the cells", flush=True)

    with np.load(os.path.join(window, df)) as z:
        poses, K = z["poses"].astype(np.float32), z["intrinsics"].astype(np.float32)
    poses = poses @ np.linalg.inv(poses[0])                                   # camera-from-(frame-0 camera)
    xyz, lift, observed = xyz.float().cpu().numpy(), lift.float().cpu().numpy(), observed.cpu().numpy()
    np.savez(os.path.join(out, "tracks.npz"), xyz=xyz.astype(np.float32), lift=lift.astype(np.float16), observed=observed,
             images=images, poses=poses, intrinsics=K, geometry=a.geometry, checkpoint_step=info["step"],
             source_fps=source_fps, source_frames=T, requested_frames=a.num_frames,
             process_resolution=resolution, corrector_batch=C["batch"])
    if not a.no_video:
        write_video(os.path.join(out, "tracks.mp4"), images, lift, xyz, observed, poses, K, a.stride, fps=source_fps)
    print(f"DEMO_DONE: {out}/tracks.npz" + ("" if a.no_video else f", {out}/tracks.mp4"), flush=True)


if __name__ == "__main__":
    main()
