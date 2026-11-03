"""Benchmark reporting retains failures and only compares equivalent workloads."""
from argparse import Namespace
import json
from pathlib import Path

import numpy as np

from scripts.benchmark_offload import build_command, process_stats, write_summary
from scripts.compare_offload import compare
from relift.storage import TrajectoryStore


def test_commands_use_identical_seed_wrapper_and_fixed_full_resolution(tmp_path):
    args = Namespace(baseline=tmp_path / "frozen", num_frames=0, process_res=504, corrector_batch=8,
                     candidate_args=[], seed=17)
    a = build_command("existing", tmp_path / "video.mp4", tmp_path / "a", args)
    b = build_command("ram", tmp_path / "video.mp4", tmp_path / "b", args)
    assert "benchmark_worker.py" in a[1] and a[:4] == b[:4]
    assert a[3] == "17" and a[4] == str(args.baseline / "scripts/demo_video.py")
    assert "--offload" not in a and b[-2:] == ["--offload", "ram"]
    assert a[a.index("--num_frames") + 1] == b[b.index("--num_frames") + 1] == "0"
    assert a[a.index("--process-res") + 1] == b[b.index("--process-res") + 1] == "504"


def test_summary_does_not_turn_oom_into_a_speedup(tmp_path):
    for mode, status, seconds in (("existing", "cuda_oom", 4.), ("ram", "complete", 30.)):
        root = tmp_path / "forest" / mode / "trial-01"
        root.mkdir(parents=True)
        record = dict(scene="forest", mode=mode, repeat=1, status=status, wall_seconds=seconds,
                      input_sha256="same", process_res=504, corrector_batch=8, num_frames=0,
                      scenario="fresh_inference_process", telemetry={})
        (root / "trial.json").write_text(json.dumps(record))
    rows = write_summary(tmp_path)
    existing, ram = sorted(rows, key=lambda row: row["mode"])
    assert existing["completed"] == 0 and existing["median_wall_seconds"] is None
    assert ram["completed"] == 1 and ram["median_wall_seconds"] == 30.
    assert ram["existing_over_candidate_speed_ratio"] is None
    assert (tmp_path / "summary.csv").exists()


def test_proc_stats_can_read_own_process():
    import os
    row = process_stats(os.getpid())
    assert row["pid"] == os.getpid() and row["rss_bytes"] > 0
    assert row["cpu_seconds"] >= 0


def test_parity_separates_padded_pixels_and_detects_ordinary_changes(tmp_path):
    T, H, W = 3, 2, 3
    xyz = np.arange(T * H * W * 3, dtype=np.float32).reshape(T, H, W, 3) / 20 + 1
    shared = dict(images=np.zeros((T, H, W, 3), np.uint8), poses=np.tile(np.eye(4, dtype=np.float32), (T, 1, 1)),
                  intrinsics=np.tile(np.eye(3, dtype=np.float32), (T, 1, 1)))
    reference = tmp_path / "reference"
    reference.mkdir()
    np.savez(reference / "tracks.npz", xyz=xyz, lift=xyz.astype(np.float16), observed=np.ones((T, H, W), bool), **shared)
    candidate = tmp_path / "candidate"
    order = np.array([3, 1, 5, 0, 2, 4])
    modified = xyz.copy().reshape(T, -1, 3)
    modified[:, 3] += 100  # Defined separately from ordinary point parity.
    store = TrajectoryStore.create(candidate, xyz.shape, dict(scale=1.), order, shared, repeated_pixel_ids=[3])
    store.write_shard(0, xyz=modified[:, order].swapaxes(0, 1),
                       lift=xyz.reshape(T, -1, 3)[:, order].astype(np.float16).swapaxes(0, 1),
                       observed=np.ones((H * W, T), bool))
    store.finish()
    report = compare(reference, candidate, chunk_points=2, prepared=False)
    assert report["passed"]
    assert report["errors"]["ordinary_xyz"]["max_normalized_vector_error"] == 0
    assert report["errors"]["repeated_xyz"]["max_normalized_vector_error"] > 100
    assert report["compared_cells"] == T * H * W
    # The same numerical difference in a nonpadded point must fail.
    with np.load(reference / "tracks.npz") as z:
        changed = {key: z[key] for key in z.files}
    changed["xyz"] = changed["xyz"].copy()
    changed["xyz"][:, 0, 0] += 1
    np.savez(reference / "tracks.npz", **changed)
    assert not compare(reference, candidate, chunk_points=2, prepared=False)["passed"]


def test_legacy_repeat_comparison_can_use_explicit_matching_pixel_policy(tmp_path):
    T, H, W = 3, 2, 2
    xyz = np.arange(T * H * W * 3, dtype=np.float32).reshape(T, H, W, 3) / 10 + 1
    shared = dict(images=np.zeros((T, H, W, 3), np.uint8), poses=np.tile(np.eye(4, dtype=np.float32), (T, 1, 1)),
                  intrinsics=np.tile(np.eye(3, dtype=np.float32), (T, 1, 1)))
    arrays = dict(xyz=xyz, lift=xyz.astype(np.float16), observed=np.ones((T, H, W), bool), **shared)
    reference, repeat, policy_root = (tmp_path / name for name in ("reference", "repeat", "policy"))
    reference.mkdir()
    repeat.mkdir()
    np.savez(reference / "tracks.npz", **arrays)
    changed = dict(arrays)
    changed["xyz"] = xyz.copy()
    changed["xyz"][:, 0, 1] += 1  # The known repeated/padded pixel, flattened ID 1.
    np.savez(repeat / "tracks.npz", **changed)
    order = np.arange(H * W)
    store = TrajectoryStore.create(policy_root, xyz.shape, dict(scale=2., seed=0, group=4, batch=1),
                                   order, shared, repeated_pixel_ids=[1])
    store.write_shard(0, xyz=xyz.reshape(T, -1, 3).swapaxes(0, 1),
                       lift=arrays["lift"].reshape(T, -1, 3).swapaxes(0, 1),
                       observed=np.ones((H * W, T), bool))
    store.finish()
    assert not compare(reference, repeat, prepared=False)["passed"]
    report = compare(reference, repeat, prepared=False, pixel_policy_from=policy_root)
    assert report["passed"] and report["repeated_pixel_ids"] == [1]
    assert report["normalization_scale"] == 2.
    assert report["errors"]["ordinary_xyz"]["max_normalized_vector_error"] == 0.
    assert report["errors"]["repeated_xyz"]["max_normalized_vector_error"] > 0.
    assert report["pixel_policy"]["source"] == str(policy_root)
    assert report["pixel_policy"]["grouping"] == dict(seed=0, group=4, batch=1)
    assert len(report["pixel_policy"]["manifest_sha256"]) == 64
