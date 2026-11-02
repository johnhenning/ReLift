"""Mapped trajectory shards and a compatibility reader for legacy NPZ results.

Trajectory shards are (points, time, coordinates); their original reference-pixel
IDs are recorded separately. Readers request original IDs and receive (time,
points, ...) arrays, so physical grouping never changes point identity.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

import numpy as np

FORMAT = "relift.trajectory-store"
VERSION = 1
FIELDS = {"xyz": np.dtype("float32"), "lift": np.dtype("float16"), "observed": np.dtype("bool")}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        json.dump(_json_value(value), stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def atomic_npy(path, array):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def mapped_archive(source, cache=None):
    """Stream NPZ members into independently mapped NPY files, never loading all members.

    Conversion is a one-time preparation cost and source bytes are fingerprinted.
    A changed archive gets a distinct cache, avoiding stale prepared-window reuse.
    """
    source = Path(source)
    if source.is_dir():
        return {p.stem: np.load(p, mmap_mode="r", allow_pickle=False) for p in source.glob("*.npy")}
    digest = file_sha256(source)
    root = Path(cache) if cache else source.parent / ".mapped" / (source.stem + "-" + digest[:16])
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "archive.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if previous.get("source_sha256") not in (None, digest):
        raise ValueError(f"Mapped archive cache belongs to different source bytes: {root}")
    members, member_records = {}, {}
    with zipfile.ZipFile(source) as archive:
        for info in archive.infolist():
            # NPZ names are flat array names; reject nested or traversal members.
            name = info.filename
            if not name.endswith(".npy") or Path(name).name != name:
                raise ValueError(f"Unexpected NPZ member: {name}")
            target = root / name
            old_members = previous.get("members", {})
            old = old_members.get(name[:-4], {}) if isinstance(old_members, dict) else {}
            valid = (target.exists() and target.stat().st_size == info.file_size
                     and old.get("sha256") == file_sha256(target))
            if not valid:
                temp = target.with_suffix(".npy.tmp")
                with archive.open(info) as src, temp.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                    dst.flush()
                    os.fsync(dst.fileno())
                temp.replace(target)
            members[name[:-4]] = np.load(target, mmap_mode="r", allow_pickle=False)
            member_records[name[:-4]] = dict(file=name, sha256=file_sha256(target))
    atomic_json(manifest_path, dict(source_sha256=digest, members=member_records))
    return members


def read_frames_mapped(source, num_frames, cache_dir):
    """Decode RGB frames into a disk spool with at most one decoded frame in RAM.

    The spool is committed after the decoder reaches the requested limit or EOF;
    actual decoded count, not container metadata, determines its mapped shape.
    Existing caches require matching input bytes, requested frame count and spool
    integrity. Benchmark callers use a fresh cache directory per trial.
    """
    import cv2

    if num_frames < 0:
        raise ValueError("num_frames must be nonnegative")
    source, root = Path(source), Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    image_paths = None
    if source.is_dir():
        image_paths = sorted((p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS), key=lambda p: p.name)
        h = hashlib.sha256()
        for path in image_paths:
            h.update(path.name.encode("utf-8"))
            h.update(b"\0")
            h.update(file_sha256(path).encode("ascii"))
            h.update(b"\0")
        source_hash = h.hexdigest()
    else:
        source_hash = file_sha256(source)
    manifest_path, spool = root / "frames.json", root / "frames.rgb"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("source_sha256") != source_hash or manifest.get("requested_frames") != num_frames:
            raise ValueError("Mapped frame cache input or frame limit changed; choose a fresh cache directory")
        shape = tuple(manifest["shape"])
        if not spool.exists() or spool.stat().st_size != int(np.prod(shape)) or file_sha256(spool) != manifest.get("sha256"):
            raise ValueError("Mapped frame spool integrity check failed")
        return np.memmap(spool, mode="r", dtype=np.uint8, shape=shape)
    temporary = spool.with_suffix(".rgb.tmp")
    count, frame_shape = 0, None
    cap = None
    try:
        if image_paths is None:
            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                raise ValueError(f"Cannot open {source}")
        with temporary.open("wb") as stream:
            while num_frames == 0 or count < num_frames:
                if image_paths is not None:
                    if count >= len(image_paths):
                        break
                    image = cv2.imread(str(image_paths[count]), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError(f"Cannot read {image_paths[count]}")
                else:
                    ok, image = cap.read()
                    if not ok:
                        break
                if frame_shape is None:
                    frame_shape = image.shape
                elif image.shape != frame_shape:
                    raise ValueError(f"{source}: frames differ in size")
                # BGR->RGB matches demo_video.read_frames exactly.
                stream.write(np.ascontiguousarray(image[..., ::-1]).tobytes())
                count += 1
            stream.flush()
            os.fsync(stream.fileno())
        if count < 2:
            raise ValueError(f"{source}: {count} decoded frames; at least two are required")
        shape = (count,) + frame_shape
        temporary.replace(spool)
        manifest = dict(format="relift.mapped-frames", version=1, source_sha256=source_hash,
                        requested_frames=num_frames, shape=shape, dtype="uint8", color_order="RGB",
                        sha256=file_sha256(spool))
        atomic_json(manifest_path, manifest)
    finally:
        if cap is not None:
            cap.release()
        temporary.unlink(missing_ok=True)
    return np.memmap(spool, mode="r", dtype=np.uint8, shape=shape)


class TrajectoryStore:
    """Read a completed mapped store or legacy ``tracks.npz`` without API drift."""

    def __init__(self, path, *, allow_incomplete=False):
        path = Path(path)
        self._legacy = None
        if path.name == "manifest.json" and path.is_file():
            path = path.parent
        if path.is_dir() and (path / "tracks.store/manifest.json").exists():
            path = path / "tracks.store"
        if path.is_dir() and (path / "manifest.json").exists():
            self.root = path
            self.manifest = json.loads((path / "manifest.json").read_text())
            if self.manifest.get("format") != FORMAT or self.manifest.get("version") != VERSION:
                raise ValueError(f"Unsupported trajectory store: {path}")
            if self.manifest["state"] != "complete" and not allow_incomplete:
                raise ValueError(f"Trajectory store is incomplete: {path}")
            self.shape = tuple(self.manifest["shape"])
            self.metadata = dict(self.manifest["metadata"])
            self.metadata["repeated_pixel_ids"] = self.manifest.get("repeated_pixel_ids", [])
            self._order = np.load(path / "pixel_ids.npy", mmap_mode="r")
            self._inverse = np.empty(len(self._order), np.int64)
            self._inverse[self._order] = np.arange(len(self._order))
        else:
            if path.is_dir():
                path = path / "tracks.npz"
            self.root = path.parent
            self._legacy = mapped_archive(path)
            self.shape = tuple(self._legacy["xyz"].shape)
            self.metadata = {k: v.item() if v.shape == () else v for k, v in self._legacy.items()
                             if k not in (*FIELDS, "images", "poses", "intrinsics")}
            self.metadata.setdefault("repeated_pixel_ids", [])
            self.manifest = None

    @classmethod
    def open(cls, path, **kwargs):
        return cls(path, **kwargs)

    @classmethod
    def create(cls, output_dir, shape, metadata, pixel_ids, arrays, repeated_pixel_ids=()):
        root = Path(output_dir) / "tracks.store"
        root.mkdir(parents=True, exist_ok=True)
        shape = tuple(map(int, shape))
        T, H, W, C = shape
        if C != 3 or min(T, H, W) < 1:
            raise ValueError("Trajectory shape must be nonempty (T,H,W,3)")
        order = np.asarray(pixel_ids, dtype=np.int64)
        if not np.array_equal(np.sort(order), np.arange(H * W)):
            raise ValueError("Pixel order must contain every reference pixel exactly once")
        metadata = _json_value(metadata)
        existing = root / "manifest.json"
        if existing.exists():
            store = cls(root, allow_incomplete=True)
            if store.shape != shape or store.manifest["metadata"] != metadata or not np.array_equal(store._order, order):
                raise ValueError("Store provenance/grouping changed; choose a fresh output directory")
            store.verify(complete=False)
            return store
        atomic_npy(root / "pixel_ids.npy", order)
        (root / "shards").mkdir(exist_ok=True)
        saved = {}
        for name, array in arrays.items():
            if name not in {"images", "poses", "intrinsics"}:
                raise ValueError(f"Unexpected shared array {name}")
            atomic_npy(root / f"{name}.npy", array)
            saved[name] = dict(file=f"{name}.npy", sha256=file_sha256(root / f"{name}.npy"))
        manifest = dict(format=FORMAT, version=VERSION, state="writing", shape=shape,
                        metadata=metadata, pixel_order_sha256=file_sha256(root / "pixel_ids.npy"),
                        repeated_pixel_ids=np.unique(repeated_pixel_ids).astype(np.int64).tolist(),
                        arrays=saved, shards=[])
        atomic_json(existing, manifest)
        return cls(root, allow_incomplete=True)

    def has_shard(self, start, stop):
        return any(s["start"] == start and s["stop"] == stop for s in self.manifest["shards"])

    def write_shard(self, start, *, xyz, lift, observed):
        if self._legacy is not None or self.manifest["state"] == "complete":
            raise ValueError("Cannot append to a completed or legacy store")
        count = len(xyz)
        stop = start + count
        if start < 0 or stop > self.shape[1] * self.shape[2] or count < 1:
            raise ValueError("Invalid shard range")
        for shard in self.manifest["shards"]:
            if max(start, shard["start"]) < min(stop, shard["stop"]):
                raise ValueError("Shard overlaps committed data")
        files = {}
        for name, array in dict(xyz=xyz, lift=lift, observed=observed).items():
            value = np.asarray(array, dtype=FIELDS[name])
            expected = (count, self.shape[0]) + (() if name == "observed" else (3,))
            if value.shape != expected:
                raise ValueError(f"{name}: expected {expected}, got {value.shape}")
            if name != "observed" and not np.isfinite(value).all():
                raise ValueError(f"{name}: nonfinite values cannot be committed")
            relative = f"shards/{start:09d}-{stop:09d}.{name}.npy"
            target = self.root / relative
            atomic_npy(target, value)
            files[name] = dict(file=relative, sha256=file_sha256(target))
        self.manifest["shards"].append(dict(start=start, stop=stop, arrays=files))
        self.manifest["shards"].sort(key=lambda s: s["start"])
        atomic_json(self.root / "manifest.json", self.manifest)

    def finish(self):
        self.verify(complete=True)
        self.manifest["state"] = "complete"
        self.manifest["finite"] = True
        atomic_json(self.root / "manifest.json", self.manifest)
        return self

    def verify(self, complete=True):
        if self._legacy is not None:
            return self.finite()
        if file_sha256(self.root / "pixel_ids.npy") != self.manifest["pixel_order_sha256"]:
            raise ValueError("Pixel mapping checksum mismatch")
        next_start = 0
        for shard in self.manifest["shards"]:
            if complete and shard["start"] != next_start:
                raise ValueError("Trajectory shards do not cover every reference pixel")
            next_start = shard["stop"]
            for entry in shard["arrays"].values():
                if file_sha256(self.root / entry["file"]) != entry["sha256"]:
                    raise ValueError(f"Trajectory shard checksum mismatch: {entry['file']}")
        for entry in self.manifest["arrays"].values():
            if file_sha256(self.root / entry["file"]) != entry["sha256"]:
                raise ValueError(f"Shared array checksum mismatch: {entry['file']}")
        if complete and next_start != self.shape[1] * self.shape[2]:
            raise ValueError("Trajectory shards do not cover every reference pixel")
        return True

    def sample(self, pixel_ids, fields=("xyz", "lift", "observed")):
        ids = np.asarray(pixel_ids, dtype=np.int64).reshape(-1)
        T, H, W, _ = self.shape
        if ((ids < 0) | (ids >= H * W)).any():
            raise IndexError("Reference pixel ID outside the processing grid")
        if any(name not in FIELDS for name in fields):
            raise ValueError("sample fields must be xyz, lift, or observed")
        if self._legacy is not None:
            return {name: np.asarray(self._legacy[name].reshape(T, H * W, *self._legacy[name].shape[3:])[:, ids]) for name in fields}
        positions = self._inverse[ids]
        result = {name: np.empty((T, len(ids)) + (() if name == "observed" else (3,)), dtype=FIELDS[name]) for name in fields}
        covered = np.zeros(len(ids), dtype=bool)
        for shard in self.manifest["shards"]:
            selected = np.flatnonzero((positions >= shard["start"]) & (positions < shard["stop"]))
            if not len(selected):
                continue
            rows = positions[selected] - shard["start"]
            for name in fields:
                values = np.load(self.root / shard["arrays"][name]["file"], mmap_mode="r")
                result[name][:, selected] = np.swapaxes(values[rows], 0, 1)
            covered[selected] = True
        if not covered.all():
            raise ValueError("Requested pixels are not committed yet")
        return result

    sample_pixels = sample

    def array(self, name):
        if self._legacy is not None:
            return self._legacy[name]
        if name not in self.manifest["arrays"]:
            if name in self.metadata:
                return np.asarray(self.metadata[name])
            raise KeyError(f"{name} is sharded; use sample or read_frame")
        return np.load(self.root / self.manifest["arrays"][name]["file"], mmap_mode="r")

    def read_frame(self, name, frame):
        if name not in FIELDS:
            return np.asarray(self.array(name)[frame])
        T, H, W, _ = self.shape
        if not 0 <= frame < T:
            raise IndexError(frame)
        if self._legacy is not None:
            return np.asarray(self._legacy[name][frame])
        output = np.empty((H * W,) + (() if name == "observed" else (3,)), dtype=FIELDS[name])
        for shard in self.manifest["shards"]:
            values = np.load(self.root / shard["arrays"][name]["file"], mmap_mode="r")
            ids = self._order[shard["start"]:shard["stop"]]
            output[ids] = values[:, frame]
        return output.reshape((H, W) + output.shape[1:])

    def finite(self, block_points=4096):
        if self._legacy is not None:
            arrays = [self._legacy["xyz"], self._legacy["lift"]]
            return all(np.isfinite(values[t:t + 1]).all() for values in arrays for t in range(self.shape[0]))
        for shard in self.manifest["shards"]:
            for name in ("xyz", "lift"):
                array = np.load(self.root / shard["arrays"][name]["file"], mmap_mode="r")
                for start in range(0, len(array), block_points):
                    if not np.isfinite(array[start:start + block_points]).all():
                        return False
        return True

    def export_npz(self, path):
        """Stream a legacy-shaped compatibility archive via temporary mapped arrays."""
        import tempfile
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with tempfile.TemporaryDirectory(prefix="relift-export-", dir=path.parent) as staging:
            stage = Path(staging)
            with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                for name, dtype in FIELDS.items():
                    shape = self.shape[:-1] if name == "observed" else self.shape
                    target = stage / f"{name}.npy"
                    mapped = np.lib.format.open_memmap(target, mode="w+", dtype=dtype, shape=shape)
                    for t in range(self.shape[0]):
                        mapped[t] = self.read_frame(name, t)
                    mapped.flush()
                    del mapped
                    archive.write(target, f"{name}.npy")
                    target.unlink()
                for name in ("images", "poses", "intrinsics"):
                    target = stage / f"{name}.npy"
                    atomic_npy(target, self.array(name))
                    archive.write(target, f"{name}.npy")
                    target.unlink()
                for name, value in self.metadata.items():
                    if isinstance(value, (dict, list)) or value is None:
                        continue
                    target = stage / f"{name}.npy"
                    atomic_npy(target, np.asarray(value))
                    archive.write(target, f"{name}.npy")
                    target.unlink()
        temp_path.replace(path)
        return path


def open_store(path, **kwargs):
    return TrajectoryStore.open(path, **kwargs)
