"""Mapped shards retain original pixel identity, support resume and detect corruption."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from relift.storage import (
    TrajectoryStore,
    atomic_npy,
    file_sha256,
    mapped_archive,
    read_frames_mapped,
)


def fixture_arrays() -> Any:
    T, H, W = 5, 3, 4
    xyz = np.arange(T * H * W * 3, dtype=np.float32).reshape(T, H, W, 3) / 20
    return dict(
        xyz=xyz,
        lift=xyz.astype(np.float16),
        observed=xyz[..., 0] % 2 > 0.5,
        images=np.arange(T * H * W * 3, dtype=np.uint8).reshape(T, H, W, 3),
        poses=np.tile(np.eye(4, dtype=np.float32), (T, 1, 1)),
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (T, 1, 1)),
    )


@pytest.mark.parametrize("transpose", [False, True])
def test_hash_during_write_matches_exact_archive_bytes_and_detects_later_corruption(
    tmp_path: Path, transpose: Any
) -> None:
    data = np.arange(120, dtype=np.float32).reshape(10, 12)
    if transpose:
        data = data.T
    baseline, candidate = tmp_path / "baseline.npy", tmp_path / "candidate.npy"
    atomic_npy(baseline, data)
    digest = atomic_npy(candidate, data, hash_write=True)
    assert baseline.read_bytes() == candidate.read_bytes()
    assert digest == file_sha256(candidate)
    with candidate.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"x")
    assert digest != file_sha256(candidate)


def write_store(root: str | Path, arrays: Any) -> Any:
    T, H, W, _ = arrays["xyz"].shape
    order = np.array([7, 2, 9, 3, 1, 6, 5, 11, 4, 8, 0, 10])
    store = TrajectoryStore.create(
        root,
        arrays["xyz"].shape,
        dict(source_fps=12, requested_frames=0),
        order,
        {name: arrays[name] for name in ("images", "poses", "intrinsics")},
        [7],
    )
    for start in range(0, len(order), 5):
        ids = order[start : start + 5]
        fields = {
            name: np.swapaxes(
                arrays[name].reshape((T, H * W) + arrays[name].shape[3:])[:, ids], 0, 1
            )
            for name in ("xyz", "lift", "observed")
        }
        store.write_shard(start, **fields)
    return store.finish()


def test_sampling_and_compatibility_export_preserve_all_values(tmp_path: Path) -> None:
    arrays = fixture_arrays()
    store = write_store(tmp_path, arrays)
    ids = np.array([11, 0, 7, 7, 2])
    selected = store.sample(ids)
    for name in ("xyz", "lift", "observed"):
        expected = arrays[name].reshape((5, 12) + arrays[name].shape[3:])[:, ids]
        np.testing.assert_array_equal(selected[name], expected)
        np.testing.assert_array_equal(store.read_frame(name, 4), arrays[name][4])
    assert isinstance(store.array("images"), np.memmap)
    assert store.metadata["repeated_pixel_ids"] == [7]
    assert store.finite()
    path = store.export_npz(tmp_path / "compatibility.npz")
    with np.load(path) as archived:
        for name, expected in arrays.items():
            np.testing.assert_array_equal(archived[name], expected)
    legacy = TrajectoryStore.open(path)
    for name, expected in selected.items():
        np.testing.assert_array_equal(legacy.sample(ids)[name], expected)


def test_partial_resume_rejects_changed_provenance_and_incomplete_publication(
    tmp_path: Path,
) -> None:
    arrays = fixture_arrays()
    shared = {name: arrays[name] for name in ("images", "poses", "intrinsics")}
    order = np.arange(12)
    store = TrajectoryStore.create(tmp_path, arrays["xyz"].shape, dict(seed=0), order, shared)
    first = {
        name: np.swapaxes(arrays[name].reshape((5, 12) + arrays[name].shape[3:])[:, :4], 0, 1)
        for name in ("xyz", "lift", "observed")
    }
    store.write_shard(0, **first)
    with pytest.raises(ValueError, match="incomplete"):
        TrajectoryStore.open(tmp_path)
    with pytest.raises(ValueError, match="cover every"):
        store.finish()
    resumed = TrajectoryStore.create(tmp_path, arrays["xyz"].shape, dict(seed=0), order, shared)
    assert resumed.has_shard(0, 4)
    with pytest.raises(ValueError, match="provenance"):
        TrajectoryStore.create(tmp_path, arrays["xyz"].shape, dict(seed=1), order, shared)


def test_checksum_detects_modified_shard(tmp_path: Path) -> None:
    store = write_store(tmp_path, fixture_arrays())
    first = store.manifest["shards"][0]["arrays"]["xyz"]["file"]
    path = store.root / first
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        byte = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([byte[0] ^ 1]))
    with pytest.raises(ValueError, match="checksum"):
        store.verify()


def test_mapped_archive_rejects_wrong_cache_source(tmp_path: Path) -> None:
    source = tmp_path / "prepared.npz"
    np.savez(source, a=np.arange(10))
    arrays = mapped_archive(source, tmp_path / "mapped")
    assert isinstance(arrays["a"], np.memmap)
    np.savez(source, a=np.arange(11))
    with pytest.raises(ValueError, match="different source"):
        mapped_archive(source, tmp_path / "mapped")


def test_frame_decode_spool_matches_rgb_values_and_rejects_stale_limits(tmp_path: Path) -> None:
    import cv2

    source = tmp_path / "frames"
    source.mkdir()
    bgr = []
    for index in range(4):
        frame = np.full((4, 5, 3), [index, 40 + index, 90 + index], np.uint8)
        cv2.imwrite(str(source / f"{index:03d}.png"), frame)
        bgr.append(frame)
    mapped = read_frames_mapped(source, 0, tmp_path / "cache")
    assert isinstance(mapped, np.memmap) and mapped.shape == (4, 4, 5, 3)
    np.testing.assert_array_equal(mapped, np.stack(bgr)[..., ::-1])
    np.testing.assert_array_equal(read_frames_mapped(source, 0, tmp_path / "cache"), mapped)
    with pytest.raises(ValueError, match="frame limit changed"):
        read_frames_mapped(source, 2, tmp_path / "cache")
    partial = read_frames_mapped(source, 2, tmp_path / "other-cache")
    np.testing.assert_array_equal(partial, mapped[:2])
