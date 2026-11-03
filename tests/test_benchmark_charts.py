"""Chart summaries preserve measured settings, missing values and capacity failures."""
import json

import numpy as np

from scripts.build_benchmark_charts import collect, inventory_output, write_outputs


def trial(root, mode, repeat, seconds, status="complete", **changes):
    directory = root / "arbitrary-scene" / mode / f"trial-{repeat:02}"
    directory.mkdir(parents=True)
    data = dict(scene="arbitrary-scene", mode=mode, status=status, wall_seconds=seconds,
                scenario="fresh", input_sha256="same-input", process_res=210, corrector_batch=3,
                num_frames=0, seed=7, wrapper_sha256="wrapper", lockfile_sha256="lock",
                code_hashes={"algorithm.py": mode},
                telemetry=dict(sampled_peak_process_tree_rss_bytes=2**30,
                               sampled_peak_gpu_device_used_bytes=2 * 2**30,
                               process_tree_write_bytes=0))
    data.update(changes)
    (directory / "trial.json").write_text(json.dumps(data))
    return directory


def test_medians_ranges_and_zero_io_are_distinct_from_missing_measurements(tmp_path):
    root = tmp_path / "campaign"
    trial(root, "existing", 1, 10)
    trial(root, "existing", 2, 14)
    trial(root, "ram", 1, 20)
    item, = collect([root])
    original, ram = item["rows"]
    assert original["metrics"]["runtime_s"] == dict(values=[10., 14.], median=12., min=10., max=14., n=2)
    assert original["metrics"]["disk_writes_gib"]["median"] == 0
    assert original["metrics"]["disk_reads_gib"]["median"] is None
    assert ram["metrics"]["host_rss_gib"]["median"] == 1
    assert ram["metrics"]["gpu_gib"]["median"] == 2


def test_failures_are_capacity_outcomes_not_fast_runtimes_or_zero_memory(tmp_path):
    root = tmp_path / "campaign"
    trial(root, "existing", 1, 1, status="cuda_oom")
    trial(root, "ram", 1, 30)
    item, = collect([root])
    original, ram = item["rows"]
    assert original["n"] == 0 and original["statuses"] == {"cuda_oom": 1}
    assert original["metrics"]["runtime_s"]["median"] is None
    assert original["metrics"]["gpu_gib"]["median"] is None
    assert ram["metrics"]["runtime_s"]["median"] == 30


def test_changed_workloads_and_source_versions_are_not_pooled(tmp_path):
    root = tmp_path / "campaign"
    trial(root, "ram", 1, 10)
    trial(root, "ram", 2, 20, code_hashes={"algorithm.py": "changed"})
    trial(root, "ram", 3, 30, process_res=224)
    items = collect([root])
    assert len(items) == 2
    assert sorted(len(item["rows"]) for item in items) == [1, 2]
    assert all(len({row["label"] for row in item["rows"]}) == len(item["rows"]) for item in items)


def test_actual_runtime_separates_interpreters_and_allows_frozen_source_locks(tmp_path):
    root = tmp_path / "campaign"
    runtime = dict(python="3.13", environment_lockfile_sha256="actual-environment")
    trial(root, "control", 1, 10, lockfile_sha256="old-source-lock", environment=runtime)
    trial(root, "candidate", 1, 9, lockfile_sha256="new-source-lock", environment=runtime)
    trial(root, "candidate", 2, 8, environment=dict(runtime, python="3.10"))
    items = collect([root])
    assert sorted(len(item["rows"]) for item in items) == [1, 2]


def test_dense_inventory_ignores_validation_caches_and_never_extracts_archives(tmp_path):
    np.savez(tmp_path / "tracks.npz", xyz=np.zeros((3, 2, 4, 3), np.float32))
    mapped = tmp_path / ".mapped"
    mapped.mkdir()
    (mapped / "later-validation-cache").write_bytes(b"x" * 1234)
    before = sorted(tmp_path.rglob("*"))
    data = inventory_output(tmp_path)
    assert data["dense_bytes"] == (tmp_path / "tracks.npz").stat().st_size
    assert data["shape"] == [3, 2, 4, 3]
    assert data["files"] == 1
    assert sorted(tmp_path.rglob("*")) == before


def test_chart_outputs_and_raw_data_are_generated_from_arbitrary_records(tmp_path):
    root = tmp_path / "campaign"
    trial(root, "existing", 1, 10)
    trial(root, "ram", 1, 20)
    items = collect([root])
    output = write_outputs(items, [root], tmp_path / "charts")
    assert output.is_file()
    page = output.read_text()
    assert "arbitrary-scene" in page and "plotly.js" in page
    assert '<script src="http' not in page
    assert list(output.parent.glob("*.png")) and list(output.parent.glob("*.svg"))
    data = json.loads((output.parent / "comparison.json").read_text())
    assert data["workloads"][0]["rows"][0]["metrics"]["runtime_s"]["median"] == 10
    assert data["definitions"]["scratch_peak"].startswith("Not recorded")
