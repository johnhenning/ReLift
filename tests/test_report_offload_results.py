"""Reports follow arbitrary persisted campaigns, not repository-specific claims."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.report_offload_results import aggregate, load_trials, main, ratios, render_report


def write_trial(
    root: str | Path, scene: Any, mode: str, index: int, seconds: Any, **changes: Any
) -> Any:
    folder = root / scene / mode / f"trial-{index:02}"
    folder.mkdir(parents=True)
    data = dict(
        scene=scene,
        mode=mode,
        repeat=index,
        status="complete",
        wall_seconds=seconds,
        started_at=f"2031-04-12T11:00:{index:02}+00:00",
        scenario="synthetic-process",
        input_sha256="input-a",
        process_res=210,
        corrector_batch=3,
        num_frames=17,
        seed=23,
        code_hashes={"model.py": f"code-{mode}"},
        wrapper_sha256="wrapper-a",
        lockfile_sha256="lock-a",
        telemetry=dict(
            sampled_peak_gpu_device_used_bytes=4 * 2**30,
            sampled_peak_process_tree_rss_bytes=3 * 2**30,
            sampled_peak_process_tree_pss_bytes=2 * 2**30,
            process_tree_read_bytes=2**30,
            process_tree_write_bytes=2**29,
        ),
        metrics=dict(
            settings=dict(ram_budget_gb=7),
            stages=dict(
                encode=dict(
                    seconds=seconds / 2, cuda_allocated_peak_bytes=123, cuda_reserved_peak_bytes=456
                )
            ),
        ),
        log=str(folder / "run.log"),
    )
    data.update(changes)
    (folder / "trial.json").write_text(json.dumps(data))
    (folder / "run.log").write_text("synthetic log")
    (folder / "telemetry.jsonl").write_text("{}\n")
    return folder / "trial.json"


@pytest.mark.parametrize(
    "campaign,scene,mode,existing,candidate,expected_ratio,model",
    [
        ("campaign-amber", "turntable", "host", [12, 16], [6, 8], 2.0, "Device Alpha"),
        ("campaign-violet", "waterfall", "archive", [21, 39, 30], [10, 14, 12], 2.5, "Device Beta"),
    ],
)
def test_report_is_derived_from_two_unrelated_campaigns(
    tmp_path: Path,
    campaign: Any,
    scene: Any,
    mode: str,
    existing: Any,
    candidate: Any,
    expected_ratio: Any,
    model: Any,
) -> None:
    root = tmp_path / campaign
    for name, times in (("existing", existing), (mode, candidate)):
        for i, seconds in enumerate(times, 1):
            write_trial(root, scene, name, i, seconds)
    environment = dict(hardware=dict(gpu_name=model), versions=dict(torch="synthetic-version"))
    (root / "environment.json").write_text(json.dumps(environment))
    parity = tmp_path / "arbitrary-parity.json"
    parity_data = dict(
        passed=False,
        repeated_pixel_count=5,
        shape=[17, 10, 21, 3],
        errors=dict(ordinary_xyz=dict(max_normalized_vector_error=0.125)),
    )
    parity.write_text(json.dumps(parity_data))
    analysis = tmp_path / "reviewed.md"
    verbatim = "Reviewed prose with exact **formatting**.\n\nDo not rewrite this.\n"
    analysis.write_text(verbatim)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    output = tmp_path / "reports" / "result.md"
    main(
        [
            "--benchmark-roots",
            str(root),
            "--parity-files",
            str(parity),
            "--analysis",
            str(analysis),
            "--out",
            str(output),
        ]
    )
    report = output.read_text()
    groups = aggregate(load_trials([root]))
    assert ratios(groups)[0][1] == expected_ratio
    assert scene in report and mode in report and model in report
    assert "2031-04-12T11:00:01+00:00" in report
    assert (
        "cuda_allocated_peak_bytes | 123" in report and "cuda_reserved_peak_bytes | 456" in report
    )
    assert "Sampled tree PSS GiB" in report and "Physical read GiB" in report
    assert '"ram_budget_gb":7' in report
    assert "errors.ordinary_xyz.max_normalized_vector_error | 0.125" in report
    assert "passed | False" in report and "shape | [17,10,21,3]" in report
    assert report.endswith(verbatim + "\n")
    assert "../arbitrary-parity.json" in report and "../reviewed.md" in report
    assert "run.log)" in report and "telemetry.jsonl)" in report and "trial.json)" in report
    assert all(p.read_bytes() == value for p, value in before.items())
    assert not (root / "summary.json").exists()
    for unsupported in ("RTX", "504", "swing-tire", "forest", "November", "206", "294", "141,120"):
        assert unsupported not in report


def test_missing_metadata_is_unknown_and_failure_has_no_ratio(tmp_path: Path) -> None:
    root = tmp_path / "minimal"
    root.mkdir()
    (root / "trial.json").write_text(
        json.dumps(dict(mode="existing", status="cuda_oom", wall_seconds=1))
    )
    report = render_report([root], destination=tmp_path / "report.md")
    assert "Hardware identity: unknown" in report
    assert "Environment metadata: unknown" in report
    assert "Missing started_at: 1" in report and "Missing code_hashes: 1" in report
    assert "cuda_oom" in report
    assert aggregate(load_trials([root]))[0]["values"]["wall_seconds"] is None
    assert "successful timings unavailable" not in report  # No invented candidate comparison.
    assert "Parity records: unknown" in report


def test_ratio_suppresses_oom_mismatched_settings_and_ambiguous_versions(tmp_path: Path) -> None:
    root = tmp_path / "changes"
    write_trial(root, "motion", "existing", 1, 1, status="cuda_oom")
    write_trial(root, "motion", "ram", 1, 5)
    assert ratios(aggregate(load_trials([root])))[0][1:] == (None, "successful timings unavailable")
    write_trial(root, "motion", "different-res", 1, 2, process_res=211)
    results = ratios(aggregate(load_trials([root])))
    assert all(value is None for _, value, _ in results)
    write_trial(root, "motion", "existing", 2, 8, code_hashes={"model.py": "different-code"})
    report = render_report([root])
    assert "Mixed source/lockfile/wrapper versions" in report
    assert "no unique settings/version-matched existing group" in report
    assert all(value is None for _, value, _ in ratios(aggregate(load_trials([root]))))


def test_roots_are_explicit_and_overlaps_do_not_duplicate_records(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    path = write_trial(root, "clip", "existing", 1, 10)
    assert len(load_trials([root, path.parent])) == 1
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path / "report.md")])


def test_capacity_metadata_does_not_invent_device_identity_and_environment_is_optional(
    tmp_path: Path,
) -> None:
    root = tmp_path / "capacity"
    write_trial(root, "clip", "existing", 1, 10)
    environment = tmp_path / "separate-environment.json"
    environment.write_text(json.dumps(dict(cuda_sampling=dict(gpu_device_total_bytes=10 * 2**30))))
    report = render_report([root], environment=environment)
    assert "10737418240" in report
    assert "Hardware identity is unknown unless explicitly identified" in report
    assert "RTX" not in report and "Device Alpha" not in report


def test_additional_inference_flags_prevent_an_unmatched_ratio(tmp_path: Path) -> None:
    root = tmp_path / "flags"
    command = [
        "interpreter",
        "worker",
        "--seed",
        "23",
        "script",
        "video",
        "--num_frames",
        "17",
        "--no_video",
    ]
    write_trial(root, "clip", "existing", 1, 10, command=command + ["--out", "baseline-output"])
    write_trial(
        root,
        "clip",
        "host",
        1,
        5,
        command=command + ["--out", "candidate-output", "--offload", "host"],
    )
    assert ratios(aggregate(load_trials([root])))[0][1] == 2
    write_trial(root, "clip", "tuned", 1, 3, command=command + ["--geometry", "another-provider"])
    result = ratios(aggregate(load_trials([root])))
    assert result[1][1] is None


def test_supplementary_environment_metadata_is_not_a_conflicting_environment(
    tmp_path: Path,
) -> None:
    root = tmp_path / "supplemented"
    write_trial(root, "clip", "existing", 1, 10)
    harness = dict(
        python="3.12.9 (build details)",
        platform="synthetic-platform",
        versions={"torch": "8.0", "Some_Package": "1.2"},
        cuda_sampling=dict(gpu_device_used_bytes=120),
    )
    (root / "environment.json").write_text(json.dumps(harness))
    shared = tmp_path / "shared.json"
    detailed = dict(
        python="3.12.9",
        platform="synthetic-platform",
        packages={"torch": "8.0", "some-package": "1.2", "extra": "4.5"},
        hardware=dict(cpu_models=["Synthetic CPU"], memory_bytes=123456),
    )
    shared.write_text(json.dumps(detailed))
    report = render_report([root], environment=shared)
    assert "Mixed recorded environments" not in report
    aggregates = report.split("## Aggregate measurements", 1)[1].split("### Settings-matched", 1)[0]
    assert "supplemented / clip / existing" in aggregates
    assert str(root) not in aggregates
    # Comparable values, rather than field absence or package alias coverage,
    # establish an actual contradiction.
    detailed["packages"]["torch"] = "9.0"
    shared.write_text(json.dumps(detailed))
    report = render_report([root], environment=shared)
    assert "Mixed recorded environments in comparable fields: packages.torch" in report
    assert "packages.extra;" not in report and "hardware.memory_bytes;" not in report


def test_parity_tables_preserve_small_values_and_link_long_lists(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    write_trial(root, "clip", "existing", 1, 10)
    parity = tmp_path / "parity.json"
    values = dict(
        passed=True,
        shape=[4, 8, 9, 3],
        repeated_pixel_ids=list(range(127)),
        pixel_policy=dict(source="policy-store", grouping=dict(seed=7, group=16)),
        errors=dict(ordinary_xyz=dict(bitwise_equal=False, max_normalized_vector_error=1e-7)),
        missing_value=None,
    )
    parity.write_text(json.dumps(values))
    output = tmp_path / "docs" / "report.md"
    report = render_report([root], parity_files=[parity], destination=output)
    section = report.split("## Numerical parity records", 1)[1]
    assert "../parity.json" in section
    assert "passed | True" in section and "shape | [4,8,9,3]" in section
    assert "repeated_pixel_ids | 127 items; see source" in section
    assert "pixel_policy.source | policy-store" in section
    assert "pixel_policy.grouping.seed | 7" in section
    assert "errors.ordinary_xyz.bitwise_equal | False" in section
    assert "errors.ordinary_xyz.max_normalized_vector_error | 1e-07" in section
    assert "missing_value | null" in section
    assert json.dumps(values["repeated_pixel_ids"]) not in section
    assert json.loads(parity.read_text()) == values
