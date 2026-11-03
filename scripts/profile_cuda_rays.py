#!/usr/bin/env python
"""Profile ray construction separately from the model and from fresh-process trials.

Uses real prepared coordinates, CUDA events, synchronized wall timings, and a
separate instrumented trace. Never treats profiler timings as benchmark results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from relift.records.artifacts import (
        OperatorProfile,
        RayProfileReport,
        TimingDistribution,
        TimingSample,
    )

import numpy as np
import torch


def digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure_many(
    functions: Mapping[str, Callable[[], object]], repeats: int, iterations: int
) -> dict[str, TimingDistribution]:
    for _ in range(3):
        for function in functions.values():
            function()
    torch.cuda.synchronize()
    samples: dict[str, list[TimingSample]] = {name: [] for name in functions}
    names = list(functions)
    for repeat in range(repeats):
        offset = repeat % len(names)
        for name in names[offset:] + names[:offset]:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            before = time.perf_counter()
            result: Any = None
            start.record()
            for _ in range(iterations):
                result = functions[name]()
            end.record()
            end.synchronize()
            samples[name].append(
                dict(
                    wall_ms=(time.perf_counter() - before) * 1000 / iterations,
                    stream_elapsed_ms=start.elapsed_time(end) / iterations,
                )
            )
            del result
    return {
        name: dict(
            samples=rows,
            median_wall_ms=statistics.median(s["wall_ms"] for s in rows),
            median_stream_elapsed_ms=statistics.median(s["stream_elapsed_ms"] for s in rows),
        )
        for name, rows in samples.items()
    }


def trace(function: Callable[..., object], destination: str | Path) -> list[OperatorProfile]:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as profiler:
        with torch.profiler.record_function("ray_profile"):
            function()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(destination))
    rows: list[OperatorProfile] = [
        dict(
            name=e.key,
            calls=e.count,
            self_cpu_us=e.self_cpu_time_total,
            self_device_us=e.self_device_time_total,
            device_us=e.device_time_total,
        )
        for e in profiler.key_averages()
    ]
    return sorted(rows, key=lambda e: e["self_cpu_us"] + e["self_device_us"], reverse=True)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--window", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--points", type=int, default=8192)
    args = parser.parse_args()
    if min(args.repeats, args.iterations, args.points) < 1:
        raise ValueError("positive counts required")
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source.resolve()))
    from relift.cuda_rays import make_ray_builder, torch_ray_builder
    from relift.native_backend import load_cpu
    from relift.optimization import InferenceComponents
    from relift.streaming import MappedObservations

    native = load_cpu(args.build)
    builder = make_ray_builder(native)
    components = InferenceComponents().with_native_build(str(args.build.resolve()))
    obs = MappedObservations(args.window, "cuda")
    ids = np.random.default_rng(0).permutation(obs.H * obs.W)[: args.points]
    coords = torch.as_tensor(
        np.array(obs.tracks[:, ids // obs.W, ids % obs.W]), device="cuda", dtype=torch.float32
    )
    depth = torch.as_tensor(np.array(obs.depth), device="cuda", dtype=torch.float32)
    results = {}
    for frames in sorted({1, min(16, obs.T), obs.T}):
        values = (coords[:frames], depth[:frames], obs.K[:frames], obs.Rwc[:frames])
        camera = torch.empty((*values[0].shape[:2], 3), device="cuda")
        distance = torch.empty(values[0].shape[:2], device="cuda")
        contiguous = tuple(v.contiguous() for v in values[:3])

        def kernel(
            contiguous: tuple[torch.Tensor, ...] = contiguous,
            camera: torch.Tensor = camera,
            distance: torch.Tensor = distance,
            frames: int = frames,
            height: int = obs.H,
            width: int = obs.W,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            c, d, k = contiguous
            native.launch_camera_rays(
                c.data_ptr(),
                d.data_ptr(),
                k.data_ptr(),
                camera.data_ptr(),
                distance.data_ptr(),
                frames,
                c.shape[1],
                height,
                width,
                torch.cuda.current_stream().cuda_stream,
            )
            return camera, distance

        kernel()
        functions = {
            "torch": lambda values=values: torch_ray_builder(*values),
            "cuda": lambda values=values: builder(*values),
            "camera_kernel": kernel,
            "rotation": lambda values=values, camera=camera: torch.einsum(
                "tij,tnj->tni", values[3], camera
            ),
            "rotation_clean": lambda values=values, camera=camera: torch.nan_to_num(
                torch.einsum("tij,tnj->tni", values[3], camera)
            ),
        }
        from relift import cuda_rays

        if hasattr(cuda_rays, "TimelineRayBuilder"):
            timeline = cuda_rays.TimelineRayBuilder(native)
            functions["timeline"] = lambda values=values, timeline=timeline: timeline(*values)
        timings = measure_many(functions, args.repeats, args.iterations)
        for name, function in functions.items():
            label = f"resident_{frames}_{name}"
            results[label] = timings[name]
            results[label]["profile"] = trace(function, args.out / f"{label}.json")
            print(label, results[label]["median_wall_ms"], flush=True)
    del coords, depth, obs
    for cache in (0, 256):
        from relift import cuda_rays

        backends = (
            ("current", "python", "cuda", "cuda_timeline")
            if hasattr(cuda_rays, "TimelineRayBuilder")
            else ("current", "python", "cuda")
        )
        sources = {
            backend: MappedObservations(
                args.window,
                "cuda",
                depth_cache_mb=cache,
                ray_backend=backend,
                components=components,
            )
            for backend in backends
        }
        functions = {
            backend: (lambda source=source: source.rays(ids)) for backend, source in sources.items()
        }
        timings = measure_many(functions, args.repeats, args.iterations)
        for backend, function in functions.items():
            label = f"mapped_cache{cache}_{backend}"
            results[label] = timings[backend]
            results[label]["profile"] = trace(function, args.out / f"{label}.json")
            print(label, results[label]["median_wall_ms"], flush=True)
        del sources, functions
    record: RayProfileReport = dict(
        created_at=datetime.now(UTC).isoformat(),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        source_hashes={
            str(p.relative_to(args.source)): digest(p)
            for folder in ("relift", "native")
            for p in sorted((args.source / folder).rglob("*"))
            if p.suffix in (".py", ".cpp", ".cu")
        },
        script_sha256=digest(__file__),
        native_build=json.loads((args.build / "build.json").read_text()),
        caveat="Warm isolated rays; arm order rotates each repeat; event spans include host launch gaps. Profiler traces are separate diagnostic runs. No model or output commits included.",
        results=results,
    )
    (args.out / "summary.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
