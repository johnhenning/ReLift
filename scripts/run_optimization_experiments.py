#!/usr/bin/env python
"""Run explicit experiment variants sequentially, with fresh outputs and telemetry.

    uv run python scripts/run_optimization_experiments.py --spec configs/experiments/optimization.json --experiment profile

Only variants listed in the selected experiment run. Existing trials require
--resume; no benchmark or frozen source is overwritten. Reports consume trial.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_offload import NvidiaSampler, Telemetry, sha256, swap_counters, write_json


def tree_size(root: str | Path) -> Any:
    logical = allocated = 0
    for directory, _, files in os.walk(root):
        for filename in files:
            try:
                stat = (Path(directory) / filename).stat()
                logical += stat.st_size
                allocated += stat.st_blocks * 512
            except FileNotFoundError:
                pass
    return dict(logical_bytes=logical, allocated_bytes=allocated)


def run_variant(
    spec: dict[str, Any], name: str, definition: Any, variant: Any, repeat: Any, resume: Any
) -> None:
    source = (ROOT / variant["source"]).resolve()
    directory = ROOT / spec["output_root"] / name / variant["name"] / f"trial-{repeat:02}"
    record = directory / "trial.json"
    options = variant.get("options", {})
    from relift.configuration import experiment_configuration

    selection = experiment_configuration(options, ROOT, source / "configs/inference")
    configuration = selection.as_dict() if selection else None
    if record.exists() and resume:
        previous = json.loads(record.read_text())
        if (
            previous.get("options") != options
            or previous.get("source") != str(source)
            or previous.get("configuration") != configuration
        ):
            raise ValueError(f"Changed trial settings: {record}")
        return
    if directory.exists():
        raise FileExistsError(directory)
    output = directory / "output"
    output.mkdir(parents=True)
    if definition["task"] in ("correction", "alltracker", "compilation"):
        fixture = ROOT / spec["prepared_window"]
        shutil.copytree(fixture, output / "window", ignore=shutil.ignore_patterns(".mapped"))
    hashes = {
        str(p.relative_to(source)): sha256(p)
        for folder in (source / "relift", source / "native")
        if folder.exists()
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.suffix in (".py", ".cpp", ".cu", ".h")
    }
    command = [
        sys.executable,
        str(ROOT / "scripts/experiment_worker.py"),
        "--source",
        str(source),
        "--root",
        str(ROOT),
        "--output",
        str(output),
        "--task",
        definition["task"],
        "--options",
        json.dumps(options),
    ]
    wrapper_hash = sha256(ROOT / "scripts/experiment_worker.py")
    driver_hashes = {
        str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "scripts").glob("*.py"))
    }
    runtime = dict(
        python=sys.version,
        executable=sys.executable,
        packages={
            name: version(name) for name in ("torch", "numpy", "matplotlib", "relift-frontends")
        },
        environment_lockfile_sha256=sha256(ROOT / "uv.lock"),
    )
    gpu = NvidiaSampler()
    telemetry = Telemetry(gpu)
    output / "scratch"
    environment = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TORCHINDUCTOR_COMPILE_THREADS="1",
        MAX_JOBS="1",
        TORCHINDUCTOR_CACHE_DIR=str(output / "compiler-cache"),
        TRITON_CACHE_DIR=str(output / "triton-cache"),
    )
    native_build = options.get("native_build") or (
        selection.config.runtime.native_build if selection else None
    )
    build_record = None
    if native_build:
        build_root = (ROOT / native_build).resolve()
        build_record = json.loads((build_root / "build.json").read_text())
        if sha256(build_root / build_record["binary"]) != build_record["binary_sha256"]:
            raise ValueError("Native binary checksum mismatch")
        environment["RELIFT_NATIVE_BUILD"] = str(build_root)
    log = directory / "run.log"
    swap_before = swap_counters()
    print(f"START {name}/{variant['name']} repeat {repeat}", flush=True)
    peaks = dict(logical_bytes=0, allocated_bytes=0)
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    timed_out = False
    with log.open("w") as stream:
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
            env=environment,
            start_new_session=True,
        )
        while process.poll() is None:
            elapsed = time.perf_counter() - started
            telemetry.capture(process.pid, elapsed)
            sizes = tree_size(output)
            for key, value in sizes.items():
                peaks[key] = max(peaks[key], value)
            if elapsed > definition.get("timeout_seconds", 900):
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(0.1)
        returncode = process.wait()
    wall = time.perf_counter() - started
    text = log.read_text(errors="replace")
    result_file = output / "result.json"
    result = json.loads(result_file.read_text()) if result_file.exists() else None
    status = (
        "timeout"
        if timed_out
        else "complete"
        if returncode == 0 and result
        else "cuda_oom"
        if "out of memory" in text.lower()
        else "failed"
    )
    record_data = dict(
        experiment=name,
        mode=variant["name"],
        source=str(source),
        options=options,
        configuration=configuration,
        task=definition["task"],
        diagnostic=bool(options.get("profile")),
        repeat=repeat,
        status=status,
        started_at=started_at,
        wall_seconds=wall,
        command=command,
        output=str(output),
        log=str(log),
        returncode=returncode,
        scenario=f"{name}_{definition['task']}_process",
        scene=spec["scene"],
        input_sha256=sha256(ROOT / spec["input"]),
        num_frames=0,
        process_res=spec["process_res"],
        corrector_batch=options.get("optimizations", {}).get(
            "corrector_batch", options.get("batch", 8)
        ),
        seed=0,
        code_hashes=hashes,
        native_build=build_record,
        driver_hashes=driver_hashes,
        lockfile_sha256=sha256(source / "uv.lock"),
        wrapper_sha256=wrapper_hash,
        environment=runtime,
        telemetry=telemetry.summary(),
        telemetry_interval_seconds=0.1,
        sampled_output_peak=peaks,
        final_output_size=tree_size(output),
        system_swap_deltas={
            key: value - swap_before[key] for key, value in swap_counters().items()
        },
        result=result,
        metrics=dict(settings=options, stages=result["phases"]) if result else None,
    )
    write_json(record, record_data)
    (directory / "telemetry.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in telemetry.samples)
    )
    print(f"FINISH {name}/{variant['name']} repeat {repeat}: {status}, {wall:.3f}s", flush=True)
    if status != "complete":
        print(log, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    experiment = spec["experiments"][args.experiment]
    for repeat in range(1, experiment.get("repeats", 5) + 1):
        variants = experiment["variants"]
        if len(variants) > 2:
            offset = (repeat - 1) % len(variants)
            variants = variants[offset:] + variants[:offset]
        elif not repeat % 2:
            variants = list(reversed(variants))
        for variant in variants:
            run_variant(spec, args.experiment, experiment, variant, repeat, args.resume)


if __name__ == "__main__":
    main()
