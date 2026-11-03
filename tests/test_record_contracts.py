"""Structural typing must reject record mistakes and preserve serialized reports."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import assert_type

import numpy as np
import pytest

from relift.configuration import PROFILES, load_configuration
from relift.records.comparison import ErrorReport, is_trajectory_comparison
from relift.records.configuration import ConfigurationSelectionRecord
from scripts.check_optimization_parity import enforce_parity
from scripts.compare_offload import ErrorStats, compare

ROOT = Path(__file__).resolve().parents[1]


def test_typed_configuration_serializers_preserve_every_dataclass_field() -> None:
    for profile in sorted(PROFILES.glob("*.toml")):
        config = load_configuration(profile=profile.stem).config
        assert config.as_dict() == asdict(config), profile.stem
        optimizations = config.optimizations()
        assert optimizations.as_dict() == asdict(optimizations), profile.stem


@pytest.mark.parametrize("nonfinite", [False, True])
def test_error_report_preserves_pre_migration_wire_values(nonfinite: bool) -> None:
    fixture = json.loads((ROOT / "tests/fixtures/error-report-wire.json").read_text())
    reference = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]], np.float32)
    candidate = reference + np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], np.float32)
    if nonfinite:
        candidate = np.array([[np.nan, 0.0, 1.0], [1.0, 1.0, 1.0]], np.float32)
    stats = ErrorStats(2.0)
    stats.add(reference, candidate)
    report = stats.report()
    assert_type(report, ErrorReport)
    assert (
        json.loads(json.dumps(report, allow_nan=False))
        == fixture["nonfinite" if nonfinite else "regular"]
    )
    assert_type(load_configuration(profile="ram").as_dict(), ConfigurationSelectionRecord)


def test_shape_mismatch_remains_a_failed_report_in_campaign_validation(tmp_path: Path) -> None:
    for name, shape in (("reference", (2, 1, 1, 3)), ("candidate", (3, 1, 1, 3))):
        folder = tmp_path / name
        folder.mkdir()
        xyz = np.ones(shape, np.float32)
        np.savez(folder / "tracks.npz", xyz=xyz)
    report = compare(tmp_path / "reference", tmp_path / "candidate")
    assert not is_trajectory_comparison(report)
    enforce_parity(report, 1e-4)
    assert report == dict(
        passed=False,
        reason="shape mismatch",
        reference_shape=(2, 1, 1, 3),
        candidate_shape=(3, 1, 1, 3),
    )


@pytest.mark.parametrize(
    "source,expected",
    [
        (
            'from relift.records.comparison import ErrorReport\ndef invalid(r: ErrorReport) -> None:\n    r["vector_count"] = "two"\n',
            "vector_count",
        ),
        (
            'from relift.records.comparison import ErrorReport\ndef invalid(r: ErrorReport) -> object:\n    return r["vector_cout"]\n',
            "vector_cout",
        ),
        (
            'from relift.records.runtime import FileReference\ndef invalid() -> FileReference:\n    return {"file": "data.npy"}\n',
            "sha256",
        ),
        (
            'from relift.records.configuration import OptimizationRecord\ndef invalid(r: OptimizationRecord) -> None:\n    r["prefetch"] = "yes"\n',
            "prefetch",
        ),
    ],
)
def test_pyrefly_rejects_invalid_record_fields(tmp_path: Path, source: str, expected: str) -> None:
    fixture = tmp_path / "invalid.py"
    fixture.write_text(source)
    output = tmp_path / "diagnostics.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyrefly",
            "check",
            "--config",
            str(ROOT / "pyproject.toml"),
            str(fixture),
            "--output-format",
            "json",
            "-o",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    errors = json.loads(output.read_text())["errors"]
    assert any(expected in error["description"] for error in errors)
    assert not any(error["name"] in ("import-error", "missing-import") for error in errors)


def test_worker_type_imports_do_not_preload_the_current_relift_package() -> None:
    modules = [
        "experiment_worker.py",
        "pipeline_experiment.py",
        "compilation_experiment.py",
        "configured_depth_experiment.py",
        "provider_reuse_experiment.py",
        "profile_cuda_rays.py",
    ]
    program = (
        f"import runpy,sys; sys.path.insert(0, {str(ROOT / 'scripts')!r}); "
        + "; ".join(f"runpy.run_path({str(ROOT / 'scripts' / name)!r})" for name in modules)
        + "; assert 'relift' not in sys.modules, 'type imports contaminated source selection'"
    )
    subprocess.run([sys.executable, "-c", program], cwd=ROOT, check=True)


def test_working_snapshot_keeps_first_party_data_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import freeze_benchmark_baseline as snapshots

    source = tmp_path / "source"
    (source / "relift/data").mkdir(parents=True)
    (source / "relift/data/worldtrack.py").write_text('"""First-party source."""\n')
    (source / "data").mkdir()
    (source / "data/large-download.bin").write_bytes(b"exclude downloaded data")
    monkeypatch.setattr(snapshots, "ROOT", source)
    monkeypatch.setattr(snapshots.subprocess, "check_output", lambda *args, **kwargs: "revision\n")
    destination = snapshots.freeze(tmp_path / "snapshot")
    assert (destination / "relift/data/worldtrack.py").is_file()
    assert not (destination / "data").exists()
