"""Dense correction with mapped inputs and one full-timeline point batch on CUDA."""
from __future__ import annotations

from pathlib import Path
import shutil
from contextlib import ExitStack
from functools import wraps

import numpy as np
import torch
import torch.nn.functional as F

from .features import correct, interp_init
from .infer import VIS_GATE
from .storage import TrajectoryStore, file_sha256, mapped_archive


class MappedObservations:
    """CUDA ray construction for selected original pixels, retaining baseline math."""

    def __init__(self, window_dir, device="cuda", depth_file="da3_depth_data.npz", tracks_file="alltracker_tracks.npz", depth_cache_mb=0, gather_backend="current", components=None, ray_backend="current"):
        window = Path(window_dir)
        self.depth_source, self.track_source = window / depth_file, window / tracks_file
        depth = mapped_archive(self.depth_source)
        tracks = mapped_archive(self.track_source)
        self.depth, self.tracks, self.visibility = depth["depth"], tracks["tracks_2d"], tracks["visibility"]
        self.T, self.H, self.W = self.depth.shape
        if self.tracks.shape[:3] != self.depth.shape or self.visibility.shape != self.depth.shape:
            raise ValueError("Geometry and tracker frames/grids do not match")
        self.device = torch.device(device)
        self.ray_builder = components.ray_builder if components else None
        if self.ray_builder is None and ray_backend in ("cuda", "cuda_timeline"):
            from .cuda_rays import make_ray_builder, TimelineRayBuilder
            from .native_backend import load_cpu
            factory = TimelineRayBuilder if ray_backend == "cuda_timeline" else make_ray_builder
            self.ray_builder = factory(components.native_loader() if components and components.native_loader else load_cpu())
        elif self.ray_builder is None and ray_backend == "python":
            from .cuda_rays import torch_ray_builder
            self.ray_builder = torch_ray_builder
        self.gather = None
        if components is not None and components.gather_factory is not None:
            self.gather = components.gather_factory()
        elif gather_backend != "current":
            from .batch_gather import BatchGather
            native = components.native_loader() if gather_backend == "cpp" and components and components.native_loader else None
            self.gather = BatchGather(gather_backend, native=native)
        self.depth_cuda = None
        needed = self.T * self.H * self.W * 4
        if depth_cache_mb > 0 and needed <= depth_cache_mb * 2**20:
            if self.device.type != "cuda" or torch.cuda.mem_get_info(self.device)[0] > needed + 512 * 2**20:
                self.depth_cuda = torch.as_tensor(np.array(self.depth), dtype=torch.float32, device=self.device)
        self.poses_cpu, self.intrinsics_cpu = depth["poses"], depth["intrinsics"]
        self.poses = torch.as_tensor(np.array(self.poses_cpu), dtype=torch.float32, device=self.device)
        self.K = torch.as_tensor(np.array(self.intrinsics_cpu), dtype=torch.float32, device=self.device)
        P = self.poses.clone()
        inv0 = torch.eye(4, device=self.device, dtype=P.dtype)
        r0, t0 = P[0, :3, :3], P[0, :3, 3]
        inv0[:3, :3], inv0[:3, 3] = r0.T, -r0.T @ t0
        P = P @ inv0
        self.Rwc = P[:, :3, :3].transpose(-2, -1)
        self.origin = -torch.einsum("tij,tj->ti", self.Rwc, P[:, :3, 3])

    @torch.no_grad()
    def rays(self, pixel_ids, frame=None):
        """Return origin, direction, distance, visibility; never construct all-T/all-P rays."""
        ids = np.asarray(pixel_ids, dtype=np.int64)
        rows, cols = np.divmod(ids, self.W)
        times = slice(None) if frame is None else slice(frame, frame + 1)
        if self.gather is None:
            coords = self._transfer(self.tracks[times, rows, cols])
            gathered_vis = None
        else:
            coords_cpu, gathered_vis = self.gather(self.tracks[times], self.visibility[times], ids)
            # Synchronous pageable upload completes before these buffers are reused.
            coords = torch.as_tensor(coords_cpu, dtype=torch.float32, device=self.device)
        K, Rwc, origin = self.K[times], self.Rwc[times], self.origin[times]
        T, count = coords.shape[:2]
        camera_builder = getattr(self.ray_builder, "camera", None)
        if camera_builder is not None:
            # Advanced indexing can produce time-strided coordinates. Copy once
            # for the full timeline instead of once per 16-frame native call.
            coords = coords.contiguous()
        # Only image-depth slices enter CUDA. grid_sample still uses original full H/W.
        # Small frame blocks limit source-image memory and transfer call overhead.
        norm = (torch.stack([2.0 * coords[..., 0] / max(self.W - 1, 1) - 1.0,
                            2.0 * coords[..., 1] / max(self.H - 1, 1) - 1.0], -1)
                if self.ray_builder is None else None)
        sampled, built_directions, built_distances = [], [], []
        offset = 0 if frame is None else frame
        frame_block = T if camera_builder is not None and self.depth_cuda is not None else 16
        for t in range(0, T, frame_block):
            length = min(frame_block, T - t)
            depth = (self.depth_cuda[offset + t:offset + t + length] if self.depth_cuda is not None else
                     self._transfer(self.depth[offset + t:offset + t + length]))
            if self.ray_builder is None:
                z = F.grid_sample(depth[:, None], norm[t:t + length].reshape(length, count, 1, 2), mode="nearest",
                                  padding_mode="border", align_corners=True)[:, 0, :, 0]
                sampled.append(z)
            else:
                if camera_builder is not None:
                    d,r = camera_builder(coords[t:t+length],depth,K[t:t+length])
                else:
                    d,r = self.ray_builder(coords[t:t+length],depth,K[t:t+length],Rwc[t:t+length])
                built_directions.append(d); built_distances.append(r)
        if self.ray_builder is None:
            z = torch.cat(sampled)
            fx, fy = K[:, 0, 0, None], K[:, 1, 1, None]
            cx, cy = K[:, 0, 2, None], K[:, 1, 2, None]
            xc, yc = (coords[..., 0] - cx) / fx, (coords[..., 1] - cy) / fy
            dcam = F.normalize(torch.stack([xc, yc, torch.ones_like(xc)], dim=-1), dim=-1)
            direction = torch.einsum("tij,tnj->tni", Rwc, dcam)
            distance = z * torch.sqrt(xc * xc + yc * yc + 1.0)
        else:
            direction,distance = torch.cat(built_directions),torch.cat(built_distances)
            if camera_builder is not None:
                direction = self.ray_builder.rotate(Rwc, direction)
        vis = (self._transfer(self.visibility[times, rows, cols]) if gathered_vis is None else
               torch.as_tensor(gathered_vis, dtype=torch.float32, device=self.device))
        return origin, torch.nan_to_num(direction), torch.nan_to_num(distance), torch.nan_to_num(vis)

    def _transfer(self, values):
        array = np.array(values)
        hosts = getattr(self, "_pinned_inputs", None)
        if hosts is None:
            return torch.as_tensor(array, dtype=torch.float32, device=self.device)
        host = torch.from_numpy(array).pin_memory()
        hosts.append(host)
        return host.to(device=self.device, dtype=torch.float32, non_blocking=True)


def _batch_plan(valid, group, batch, seed, device, padding_batch=None):
    gen = torch.Generator(device=device).manual_seed(seed)
    perm = valid[torch.randperm(len(valid), device=device, generator=gen)]
    repeated = []
    planning_batch = batch if padding_batch is None else padding_batch
    for start in range(0, len(perm), group * planning_batch):
        sel = perm[start:start + group * planning_batch]
        need = max(1, (len(sel) + group - 1) // group) * group - len(sel)
        if need:
            repeated.append(sel[torch.arange(need, device=device) % len(sel)])
    duplicates = torch.unique(torch.cat(repeated)) if repeated else perm[:0]
    return perm, duplicates


def _selected_ids(order, start, group, batch, padding_batch=None):
    """Keep the reference padded group intact while varying execution batching."""
    original = order[start:start + group * batch]
    need = (-len(original)) % group
    if not need:
        return original, original
    source = original
    if padding_batch is not None:
        reference_start = ((len(order) - 1) // (group * padding_batch)) * group * padding_batch
        source = order[reference_start:]
    return original, np.concatenate([original, source[np.arange(need) % len(source)]])


def managed_resources(function):
    @wraps(function)
    def run(*args, **kwargs):
        with ExitStack() as resources:
            return function(*args, resources=resources, **kwargs)
    return run


@managed_resources
@torch.no_grad()
def stream_correct(window_dir, output_dir, model, metadata, group=1024, batch=8, seed=0,
                   device="cuda", depth_file="da3_depth_data.npz", tracks_file="alltracker_tracks.npz",
                   ram_budget_gb=18, vram_budget_gb=20, scratch_budget_gb=64, depth_cache_mb=0, hash_writes=True,
                   padding_batch=None, gather_backend="current", optimizations=None, components=None, resources=None):
    """Correct every source pixel using full timelines and atomic mapped result shards.

    ``metadata`` must identify the checkpoint/settings for safe resume. The output
    is a ``TrajectoryStore``; compatibility NPZ export is an explicit later step.
    Padded duplicate results use their first occurrence and are listed for parity.
    """
    from .optimization import OptimizationConfig, InferenceComponents
    components = components or InferenceComponents()
    if optimizations is not None:
        batch, padding_batch = optimizations.corrector_batch, optimizations.padding_batch
        gather_backend = optimizations.gather
        depth_cache_mb, hash_writes = optimizations.depth_cache_mb, optimizations.hash_writes
        if optimizations.compile_corrector:
            model = torch.compile(model, dynamic=False)
    if min(group, batch) < 1:
        raise ValueError("group and batch must be positive")
    if padding_batch is not None and padding_batch < 1:
        raise ValueError("padding_batch must be positive")
    dev = torch.device(device)
    obs_source = MappedObservations(window_dir, dev, depth_file, tracks_file, depth_cache_mb=depth_cache_mb, gather_backend=gather_backend, components=components,
                                    ray_backend=optimizations.rays if optimizations else "current")
    T, H, W = obs_source.T, obs_source.H, obs_source.W
    total = H * W
    # Only the reference frame needs all points for the exact clip-wide median.
    reference_ids = np.arange(total, dtype=np.int64)
    origin0, direction0, ranges0, _ = obs_source.rays(reference_ids, frame=0)
    valid = torch.nonzero(ranges0[0] > 0)[:, 0]
    raw0 = (origin0[:, None, :] + direction0 * ranges0[..., None]).float()[0]
    scale = raw0[valid, 2].median().clamp_min(1e-6) if len(valid) else raw0.new_tensor(1.)
    perm, repeated = _batch_plan(valid, group, batch, seed, dev, padding_batch=padding_batch)
    valid_ids = perm.cpu().numpy()
    invalid_ids = np.flatnonzero(~(ranges0[0] > 0).cpu().numpy())
    order = np.concatenate([valid_ids, invalid_ids])
    repeated_ids = repeated.cpu().numpy()
    del direction0, ranges0, raw0
    images = mapped_archive(Path(window_dir) / "da3_frames.npz")["images"]
    if images.shape != (T, H, W, 3):
        raise ValueError("Prepared images do not match the trajectory grid")
    # 22 bytes per cell for exported arrays, plus a conservative one-batch workspace estimate.
    stored_bytes = T * total * 22
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output_dir).free
    if stored_bytes > scratch_budget_gb * 2**30 or stored_bytes > free * .9:
        raise MemoryError("Trajectory store would exceed scratch budget or available disk space")
    batch_bytes = min(total, group * batch) * T * 64
    if batch_bytes > ram_budget_gb * 2**30:
        raise MemoryError("Trajectory staging batch exceeds RAM budget; lower corrector batch")
    poses = np.asarray(obs_source.poses_cpu, dtype=np.float32)
    poses = poses @ np.linalg.inv(poses[0])
    provenance = dict(metadata, group=group, batch=batch, seed=seed, vis_gate=VIS_GATE,
                      depth_sha256=file_sha256(obs_source.depth_source), tracker_sha256=file_sha256(obs_source.track_source),
                      frames_sha256=file_sha256(Path(window_dir) / "da3_frames.npz"),
                      scale=float(scale), storage="mapped-track-shards", duplicate_policy="first occurrence",
                      streaming_sha256=file_sha256(__file__),
                      storage_sha256=file_sha256(Path(__file__).with_name("storage.py")),
                      device=str(dev), torch_version=torch.__version__, numpy_version=np.__version__,
                      ram_budget_gb=ram_budget_gb, vram_budget_gb=vram_budget_gb,
                      scratch_budget_gb=scratch_budget_gb)
    provenance.update(depth_cache_mb=depth_cache_mb, depth_cache_bytes=obs_source.depth_cuda.numel() * 4 if obs_source.depth_cuda is not None else 0,
                      hash_writes=hash_writes, padding_batch=padding_batch, gather_backend=gather_backend,
                      optimizations=optimizations.as_dict() if optimizations else None, components=components.describe())
    if optimizations is not None:
        provenance["implementation_hashes"] = {str(p.relative_to(Path(__file__).parent)): file_sha256(p)
            for p in sorted(Path(__file__).parent.rglob("*.py"))}
    packer, payload_writer = components.shard_packer, components.npy_writer
    if optimizations:
        from .shard_packing import ShardPacker
        writer_backend = optimizations.shard_writer or ("cpp" if optimizations.shards == "cpp" else "current")
        native = None
        if ((optimizations.shards == "cpp" and packer is None) or (writer_backend == "cpp" and payload_writer is None)):
            from .native_backend import load_cpu
            native = components.native_loader() if components.native_loader else load_cpu()
        if optimizations.shards != "current" and packer is None:
            packer = ShardPacker(optimizations.shards,native=native)
        if writer_backend == "cpp" and payload_writer is None:
            payload_writer = ShardPacker("cpp",native=native).write
    store = TrajectoryStore.create(output_dir, (T, H, W, 3), provenance, order,
                                   dict(images=images, poses=poses, intrinsics=np.asarray(obs_source.intrinsics_cpu)), repeated_ids,
                                   hash_writes=hash_writes, packer=packer, payload_writer=payload_writer)
    if store.manifest["state"] == "complete":
        return store
    scheduler = None
    if optimizations is not None and optimizations.prefetch:
        from .prefetch import PrefetchedRays
        scheduler = resources.enter_context(PrefetchedRays(obs_source))
    # Shared source matrices and one depth image remain small. The active corrector
    # batch is the only full-timeline CUDA allocation that scales with point count.
    for start in range(0, len(valid_ids), group * batch):
        original, selected = _selected_ids(valid_ids, start, group, batch, padding_batch)
        stop = start + len(original)
        if store.has_shard(start, stop):
            continue
        nb = max(1, (len(original) + group - 1) // group)
        origin, direction, distance, vis = (scheduler.rays(selected) if scheduler else obs_source.rays(selected))
        d = direction.reshape(T, nb, group, 3).permute(1, 0, 2, 3)
        r = distance.reshape(T, nb, group).permute(1, 0, 2)
        observed = (vis.reshape(T, nb, group).permute(1, 0, 2) > VIS_GATE) & (r > 0)
        observed[:, 0] = True
        Y = (origin[None, :, None, :] + d * r[..., None]) * observed[..., None]
        ray = d * observed[..., None]
        o = origin[None].expand(nb, T, 3)
        pad = (-T) % model.chunk
        unpadded_Y, unpadded_observed = Y, observed
        if pad:
            Y = torch.cat([Y, Y.new_zeros((nb, pad, group, 3))], 1)
            ray = torch.cat([ray, ray.new_zeros((nb, pad, group, 3))], 1)
            observed = torch.cat([observed, observed.new_zeros((nb, pad, group))], 1)
            o = torch.cat([o, o[:, -1:].expand(nb, pad, 3)], 1)
        result = correct(model, Y / scale, observed, ray, o / scale)[:, :T] * scale
        if scheduler:
            for following_start in range(start + group * batch, len(valid_ids), group * batch):
                following_original, following = _selected_ids(valid_ids, following_start, group, batch, padding_batch)
                if not store.has_shard(following_start, following_start + len(following_original)):
                    scheduler.schedule(following)
                    break
        result = torch.nan_to_num(result.permute(1, 0, 2, 3).reshape(T, -1, 3))[:, :len(original)]
        raw = interp_init(unpadded_Y, unpadded_observed)[0].permute(1, 0, 2, 3).reshape(T, -1, 3)[:, :len(original)]
        observed_export = unpadded_observed.permute(1, 0, 2).reshape(T, -1)[:, :len(original)]
        store.write_shard(start, xyz=result.permute(1, 0, 2).float().cpu().numpy(),
                          lift=raw.permute(1, 0, 2).float().cpu().numpy().astype(np.float16),
                          observed=observed_export.T.cpu().numpy())
        del origin, direction, distance, vis, d, r, observed, Y, ray, o, result, raw, observed_export, unpadded_Y, unpadded_observed
        if dev.type == "cuda" and torch.cuda.max_memory_allocated(dev) > vram_budget_gb * 2**30:
            raise MemoryError("Corrector exceeded the configured GPU allocation budget; lower batch size")
    # Pixels without a valid reference range retain their un-interpolated raw lift
    # in xyz, exactly as predict_dense does. The exported lift still interpolates.
    for offset in range(0, len(invalid_ids), group * batch):
        ids = invalid_ids[offset:offset + group * batch]
        start = len(valid_ids) + offset
        if store.has_shard(start, start + len(ids)):
            continue
        origin, d, r, vis = obs_source.rays(ids)
        observed = (vis > VIS_GATE) & (r > 0)
        observed[0] = True
        xyz = torch.nan_to_num(origin[:, None, :] + d * r[..., None])
        Y = xyz * observed[..., None]
        raw = interp_init(Y[None], observed[None])[0][0]
        store.write_shard(start, xyz=xyz.permute(1, 0, 2).float().cpu().numpy(),
                          lift=raw.permute(1, 0, 2).float().cpu().numpy().astype(np.float16),
                          observed=observed.T.cpu().numpy())
        del origin, d, r, vis, observed, xyz, Y, raw
    return store.finish()
