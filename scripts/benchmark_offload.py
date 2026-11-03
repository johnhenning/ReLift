#!/usr/bin/env python
"""Benchmark frozen existing and candidate offload paths in isolated processes.

    uv run python scripts/benchmark_offload.py --scenes swing-tire --modes existing ram --repeats 3
    uv run python scripts/benchmark_offload.py --scenes running forest --modes existing ram --repeats 1

The caller must schedule this without other GPU jobs. Every measured trial has a
fresh output directory, full log, sampled telemetry and trial.json. Existing OOMs
are capacity failures, never successful runtimes or speedups. Provisioning and
optional prepared-window copies occur outside measured subprocess wall time.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "outputs/benchmarks/baseline-source"
MODES = ("existing", "none", "streamed", "ram", "disk", "auto")


def sha256(path: str | Path) -> Any:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class NvidiaSampler:
    """NVML device totals sampled in-process; no repeated nvidia-smi subprocesses."""

    class Memory(ctypes.Structure):
        _fields_ = [
            ("total", ctypes.c_ulonglong),
            ("free", ctypes.c_ulonglong),
            ("used", ctypes.c_ulonglong),
        ]

    class Utilization(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    def __init__(self, index: int = 0) -> None:
        self.library, self.handle, self.error = None, None, None
        try:
            self.library = ctypes.CDLL("libnvidia-ml.so.1")
            if self.library.nvmlInit_v2():
                raise RuntimeError("nvmlInit_v2 failed")
            self.handle = ctypes.c_void_p()
            result = self.library.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(self.handle))
            if result:
                raise RuntimeError(f"nvmlDeviceGetHandleByIndex_v2 returned {result}")
        except (OSError, AttributeError, RuntimeError) as error:
            self.error = str(error)
            self.library, self.handle = None, None

    def sample(self) -> Any:
        if self.library is None:
            return {}
        memory, utilization = self.Memory(), self.Utilization()
        out = {}
        if self.library.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(memory)) == 0:
            out.update(gpu_device_used_bytes=memory.used, gpu_device_total_bytes=memory.total)
        if self.library.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(utilization)) == 0:
            out.update(
                gpu_utilization_percent=utilization.gpu,
                gpu_memory_utilization_percent=utilization.memory,
            )
        temperature = ctypes.c_uint()
        if self.library.nvmlDeviceGetTemperature(self.handle, 0, ctypes.byref(temperature)) == 0:
            out["gpu_temperature_celsius"] = temperature.value
        return out


def process_stats(pid: Any) -> Any:
    """Read one process without importing psutil or initializing CUDA."""
    root = Path("/proc") / str(pid)
    stat = (root / "stat").read_text()
    fields = stat[stat.rfind(")") + 2 :].split()
    values = dict(
        pid=int(pid),
        ppid=int(fields[1]),
        started_ticks=int(fields[19]),
        rss_bytes=int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
        minor_faults=int(fields[7]),
        major_faults=int(fields[9]),
        cpu_seconds=(int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK"),
    )
    try:
        rollup = {}
        for line in (root / "smaps_rollup").read_text().splitlines():
            key, _, tail = line.partition(":")
            if key in ("Pss", "Anonymous", "Swap"):
                rollup[key] = int(tail.split()[0]) * 1024
        values.update(
            pss_bytes=rollup.get("Pss", 0),
            anonymous_bytes=rollup.get("Anonymous", 0),
            swap_bytes=rollup.get("Swap", 0),
        )
    except (OSError, ValueError):
        values["pss_unavailable"] = True
    try:
        counters = dict(line.split(": ", 1) for line in (root / "io").read_text().splitlines())
        for key in ("read_bytes", "write_bytes", "rchar", "wchar"):
            values[key] = int(counters[key])
    except (OSError, ValueError, KeyError):
        values["io_unavailable"] = True
    return values


def process_tree(root_pid: Any) -> Any:
    stats = {}
    for directory in Path("/proc").iterdir():
        if directory.name.isdigit():
            try:
                stat = (directory / "stat").read_text()
                fields = stat[stat.rfind(")") + 2 :].split()
                stats[int(directory.name)] = int(fields[1])
            except (OSError, ValueError, IndexError):
                pass
    selected = {root_pid}
    while True:
        children = {pid for pid, ppid in stats.items() if ppid in selected}
        if children <= selected:
            break
        selected.update(children)
    results = []
    for pid in selected:
        try:
            results.append(process_stats(pid))
        except (OSError, ValueError, IndexError):
            pass
    return results


def swap_counters() -> Any:
    counters = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
    size = os.sysconf("SC_PAGE_SIZE")
    return {key + "_bytes": int(counters[key]) * size for key in ("pswpin", "pswpout")}


class Telemetry:
    def __init__(self, gpu: Any) -> None:
        self.gpu = gpu
        self.samples, self.counters = [], {}

    def capture(self, root_pid: Any, elapsed: Any) -> None:
        processes = process_tree(root_pid)
        for proc in processes:
            key = (proc["pid"], proc["started_ticks"])
            previous = self.counters.setdefault(key, {})
            for counter in (
                "read_bytes",
                "write_bytes",
                "rchar",
                "wchar",
                "minor_faults",
                "major_faults",
                "cpu_seconds",
            ):
                previous[counter] = max(previous.get(counter, 0), proc.get(counter, 0))
        row = dict(elapsed_seconds=elapsed, process_count=len(processes))
        for key in ("rss_bytes", "pss_bytes", "anonymous_bytes", "swap_bytes"):
            row["process_tree_" + key] = sum(proc.get(key, 0) for proc in processes)
        row["process_detail_unavailable"] = any(
            p.get("pss_unavailable") or p.get("io_unavailable") for p in processes
        )
        row.update(self.gpu.sample())
        self.samples.append(row)

    def summary(self) -> Any:
        peaks = {}
        for key in (
            "process_tree_rss_bytes",
            "process_tree_pss_bytes",
            "process_tree_anonymous_bytes",
            "process_tree_swap_bytes",
            "gpu_device_used_bytes",
        ):
            values = [row[key] for row in self.samples if key in row]
            peaks["sampled_peak_" + key] = max(values) if values else None
        for counter in (
            "read_bytes",
            "write_bytes",
            "rchar",
            "wchar",
            "minor_faults",
            "major_faults",
            "cpu_seconds",
        ):
            peaks["process_tree_" + counter] = sum(
                proc.get(counter, 0) for proc in self.counters.values()
            )
        peaks["sample_count"] = len(self.samples)
        peaks["gpu_sampling_error"] = self.gpu.error
        peaks["sampling_limits"] = (
            "RSS/PSS/device memory are sampled; short peaks and exited-before-sampling children can be missed. GPU totals include desktop/other processes. I/O counters retain last sampled values per process."
        )
        return peaks


def resolve_video(scene: Any) -> Any:
    catalog = json.loads((REPO / "assets/demo/catalog.json").read_text())
    if scene not in catalog:
        raise ValueError(f"Unknown scene {scene}")
    entry = catalog[scene]
    file = (
        REPO / entry["local_file"]
        if "local_file" in entry
        else REPO / "cache/demo_inputs" / f"{scene}.mp4"
    )
    if not file.exists():
        raise FileNotFoundError(f"Provision {scene} first with scripts/download_demo_inputs.py")
    return file


def build_command(mode: str, video: Any, output: Any, args: Any) -> Any:
    if mode == "existing":
        script = args.baseline / "scripts/demo_video.py"
    else:
        script = REPO / "scripts/demo_video.py"
    command = [
        sys.executable,
        str(REPO / "scripts/benchmark_worker.py"),
        "--seed",
        str(args.seed),
        str(script),
        str(video),
        "--num_frames",
        str(args.num_frames),
        "--process-res",
        str(args.process_res),
        "--corrector-batch",
        str(args.corrector_batch),
        "--checkpoints_root",
        str(REPO / "checkpoints"),
        "--out",
        str(output),
        "--no_video",
    ]
    if mode != "existing":
        command += ["--offload", mode]
        if args.candidate_args:
            command += args.candidate_args
    return command


def benchmark_trial(mode: str, scene: Any, repeat: Any, args: Any, gpu: Any) -> Any:
    directory = args.output_root / scene / mode / f"trial-{repeat:02}"
    result_file = directory / "trial.json"
    if result_file.exists() and args.resume:
        result = json.loads(result_file.read_text())
        expected = (args.process_res, args.corrector_batch, args.num_frames, args.seed)
        actual = (
            result["process_res"],
            result["corrector_batch"],
            result["num_frames"],
            result.get("seed", 0),
        )
        if expected != actual or result["input_sha256"] != sha256(resolve_video(scene)):
            raise ValueError(f"Refusing to reuse trial with changed settings/input: {result_file}")
        result["reused_trial_record"] = True
        return result
    if directory.exists():
        raise FileExistsError(
            f"Refusing to reuse measured output {directory}; use a new root or --resume"
        )
    directory.mkdir(parents=True)
    output = directory / "output"
    output.mkdir()
    fixture_copy = 0.0
    if args.prepared_window:
        start = time.perf_counter()
        shutil.copytree(args.prepared_window, output / "window")
        fixture_copy = time.perf_counter() - start
    video = resolve_video(scene)
    command = build_command(mode, video, output, args)
    fingerprint_root = args.baseline if mode == "existing" else REPO
    code_hashes = {
        str(p.relative_to(fingerprint_root)): sha256(p)
        for p in sorted((fingerprint_root / "relift").rglob("*.py"))
    }
    code_hashes["scripts/demo_video.py"] = sha256(fingerprint_root / "scripts/demo_video.py")
    before_swap = swap_counters()
    telemetry = Telemetry(gpu)
    idle_gpu = gpu.sample()
    environment = dict(os.environ)
    environment.update(
        HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", RELIFT_BENCHMARK_SEED=str(args.seed)
    )
    log = directory / "run.log"
    timed_out = False
    print(f"START {scene} {mode} trial {repeat}", flush=True)
    started = datetime.now(UTC).isoformat()
    with log.open("w") as stream:
        start = time.perf_counter()
        process = subprocess.Popen(
            command,
            cwd=REPO,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        while process.poll() is None:
            elapsed = time.perf_counter() - start
            telemetry.capture(process.pid, elapsed)
            if elapsed > args.timeout:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(args.interval)
        returncode = process.wait()
        elapsed = time.perf_counter() - start
    text = log.read_text(errors="replace")
    has_output = (output / "tracks.npz").exists() or (
        output / "tracks.store/manifest.json"
    ).exists()
    status = (
        "timeout"
        if timed_out
        else "complete"
        if returncode == 0 and has_output
        else "cuda_oom"
        if "CUDA out of memory" in text or "OutOfMemoryError" in text
        else "failed"
    )
    result = dict(
        scene=scene,
        mode=mode,
        repeat=repeat,
        status=status,
        returncode=returncode,
        started_at=started,
        wall_seconds=elapsed,
        command=command,
        output=str(output),
        input=str(video),
        input_sha256=sha256(video),
        process_res=args.process_res,
        corrector_batch=args.corrector_batch,
        num_frames=args.num_frames,
        seed=args.seed,
        wrapper_sha256=sha256(REPO / "scripts/benchmark_worker.py"),
        scenario="prepared_window_process" if args.prepared_window else "fresh_inference_process",
        fixture_copy_seconds_outside_measurement=fixture_copy,
        idle_gpu=idle_gpu,
        telemetry_interval_seconds=args.interval,
        telemetry=telemetry.summary(),
        log=str(log),
        baseline_snapshot=str(args.baseline / "snapshot.json"),
    )
    after_swap = swap_counters()
    result["system_swap_deltas"] = {key: after_swap[key] - before_swap[key] for key in before_swap}
    for file in (output / "metrics.json", output / "offload.json"):
        if file.exists():
            result[file.stem] = json.loads(file.read_text())
    result["code_hashes"] = code_hashes
    result["lockfile_sha256"] = sha256(fingerprint_root / "uv.lock")
    (directory / "telemetry.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in telemetry.samples)
    )
    write_json(result_file, result)
    print(f"FINISH {scene} {mode} trial {repeat}: {status}, {elapsed:.2f}s", flush=True)
    return result


def write_summary(root: Path) -> Any:
    records = [json.loads(p.read_text()) for p in sorted(root.glob("*/*/trial-*/trial.json"))]
    groups = defaultdict(list)
    for record in records:
        groups[(record["scene"], record["mode"], record["scenario"])].append(record)
    summaries = []
    for (scene, mode, scenario), trials in sorted(groups.items()):
        valid = [r for r in trials if r["status"] == "complete"]
        row = dict(
            scene=scene,
            mode=mode,
            scenario=scenario,
            trials=len(trials),
            completed=len(valid),
            failures=";".join(r["status"] for r in trials if r["status"] != "complete"),
        )
        for key, getter in (
            ("wall_seconds", lambda r: r["wall_seconds"]),
            (
                "sampled_peak_gpu_device_mib",
                lambda r: r["telemetry"].get("sampled_peak_gpu_device_used_bytes", None),
            ),
            (
                "sampled_peak_tree_rss_mib",
                lambda r: r["telemetry"].get("sampled_peak_process_tree_rss_bytes", None),
            ),
            (
                "sampled_peak_tree_pss_mib",
                lambda r: r["telemetry"].get("sampled_peak_process_tree_pss_bytes", None),
            ),
            ("physical_read_mib", lambda r: r["telemetry"].get("process_tree_read_bytes", None)),
            ("physical_write_mib", lambda r: r["telemetry"].get("process_tree_write_bytes", None)),
        ):
            values = [getter(r) for r in valid if getter(r) is not None]
            if key != "wall_seconds":
                values = [v / 2**20 for v in values]
            row["median_" + key] = statistics.median(values) if values else None
            if key == "wall_seconds":
                row["min_wall_seconds"] = min(values) if values else None
                row["max_wall_seconds"] = max(values) if values else None
        summaries.append(row)
    # Only completed, matching scenario/resolution/frame-limit/batch comparisons get ratios.
    for row in summaries:
        baselines = [
            b
            for b in summaries
            if b["scene"] == row["scene"]
            and b["scenario"] == row["scenario"]
            and b["mode"] == "existing"
        ]
        row["existing_over_candidate_speed_ratio"] = None
        if baselines and row["median_wall_seconds"] and baselines[0]["median_wall_seconds"]:
            candidate = groups[(row["scene"], row["mode"], row["scenario"])]
            reference = groups[(row["scene"], "existing", row["scenario"])]

            def settings(r: dict[str, Any]) -> tuple[Any, ...]:
                return (
                    r["input_sha256"],
                    r["process_res"],
                    r["corrector_batch"],
                    r["num_frames"],
                    r.get("seed", 0),
                )

            if len({settings(r) for r in candidate + reference}) == 1:
                row["existing_over_candidate_speed_ratio"] = (
                    baselines[0]["median_wall_seconds"] / row["median_wall_seconds"]
                )
    write_json(
        root / "summary.json",
        dict(
            trials=records,
            groups=summaries,
            notes="Only successful runs contribute time/memory medians. Failures remain capacity outcomes. GPU device peaks are sampled totals including nonbenchmark processes; warm OS caches are not flushed.",
        ),
    )
    if summaries:
        with (root / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    lines = [
        "# Offload benchmark results",
        "",
        "Fresh-process measurements; no global cache flushing. Failures are excluded from successful-run medians.",
        "",
        "| Scene | Mode | Completed | Median seconds | Device GPU MiB sampled | Tree RSS MiB sampled | Existing / candidate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    def fmt(value: float | None) -> str:
        return "—" if value is None else f"{value:.2f}"

    for row in summaries:
        lines.append(
            f"| {row['scene']} | {row['mode']} | {row['completed']}/{row['trials']} | {fmt(row['median_wall_seconds'])} | {fmt(row['median_sampled_peak_gpu_device_mib'])} | {fmt(row['median_sampled_peak_tree_rss_mib'])} | {fmt(row['existing_over_candidate_speed_ratio'])} |"
        )
    lines += [
        "",
        "CUDA allocator metrics, if emitted by a candidate, are preserved in each trial.json separately from sampled device memory.",
        "Sampling may miss short peaks and processes that exit between samples. No tail latency or ground-truth accuracy claim is implied.",
        "",
    ]
    (root / "summary.md").write_text("\n".join(lines))
    return summaries


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scenes", nargs="+", default=["swing-tire"])
    ap.add_argument("--modes", nargs="+", choices=MODES, default=["existing", "ram"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--output-root", type=Path, default=REPO / "outputs/benchmarks/offload")
    ap.add_argument("--baseline", type=Path, default=BASELINE)
    ap.add_argument("--num-frames", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--process-res", type=int, default=504)
    ap.add_argument("--corrector-batch", type=int, default=8)
    ap.add_argument("--interval", type=float, default=0.1)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument(
        "--prepared-window",
        type=Path,
        help="copy the same prepared window before each trial; one scene only",
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--only-summary", action="store_true")
    ap.add_argument("--candidate-args", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()
    args.output_root = args.output_root.resolve()
    args.baseline = args.baseline.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.only_summary:
        write_summary(args.output_root)
        return
    if args.repeats < 1 or args.interval <= 0 or args.timeout <= 0 or args.num_frames < 0:
        ap.error("repeats, interval and timeout must be positive; frames must be nonnegative")
    if args.prepared_window and len(args.scenes) != 1:
        ap.error("--prepared-window is valid for one scene per invocation")
    gpu = NvidiaSampler()
    environment = dict(
        python=sys.version,
        platform=platform.platform(),
        versions={p: version(p) for p in ("torch", "numpy", "plotly")},
        cuda_sampling=gpu.sample(),
        gpu_sampling_error=gpu.error,
        baseline_snapshot_sha256=sha256(args.baseline / "snapshot.json"),
        cache_policy="Fresh process and output for each trial; provisioned local weights and inputs; OS page cache retained; HF offline.",
    )
    write_json(args.output_root / "environment.json", environment)
    # Alternate mode order to reduce systematic effects from temperature/cache order.
    for scene in args.scenes:
        for repeat in range(1, args.repeats + 1):
            modes = args.modes if repeat % 2 else list(reversed(args.modes))
            for mode in modes:
                benchmark_trial(mode, scene, repeat, args, gpu)
                write_summary(args.output_root)
    print(f"Summary: {args.output_root / 'summary.md'}")


if __name__ == "__main__":
    main()
