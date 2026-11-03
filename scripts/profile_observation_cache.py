#!/usr/bin/env python
"""Isolate budgeted observation gathering, full rays, and exact real-input parity.

Uses rotating warm timings separately from traces and all-frame/all-pixel scans.
Cache construction, resident bytes and uploads are explicitly recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from profile_cuda_rays import measure_many, trace


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--window", type=Path, required=True)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--budget-mb", type=float, default=128)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.budget_mb <= 0 or min(args.repeats, args.iterations) < 1:
        raise ValueError("positive budget/counts required")
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source.resolve()))
    from relift.optimization import InferenceComponents
    from relift.streaming import MappedObservations

    components = InferenceComponents()
    if args.build:
        components = components.with_native_build(str(args.build.resolve()))
    settings: dict[str, dict[str, Any]] = {
        "control": {},
        "observations": dict(observation_cache_mb=args.budget_mb),
        "depth": dict(depth_cache_mb=256),
        "observations-depth": dict(observation_cache_mb=args.budget_mb, depth_cache_mb=256),
    }
    if args.build:
        settings["eager"] = dict(ray_backend="cuda_timeline")
        settings["observations-eager"] = dict(
            observation_cache_mb=args.budget_mb, ray_backend="cuda_timeline"
        )
        settings["observations-depth-eager"] = dict(
            observation_cache_mb=args.budget_mb, depth_cache_mb=256, ray_backend="cuda_timeline"
        )
    sources = {}
    construction = {}
    try:
        for name, options in settings.items():
            torch.cuda.synchronize()
            start = time.perf_counter()
            sources[name] = MappedObservations(
                args.window, "cuda", components=components, **options
            )
            torch.cuda.synchronize()
            construction[name] = dict(
                seconds=time.perf_counter() - start, cache=sources[name].observation_cache.metrics
            )
            if options.get("observation_cache_mb") and not sources[name].observation_cache.active:
                raise MemoryError(f"Requested audit cache did not fit: {construction[name]}")
        reference = sources["control"]
        T, H, W = reference.T, reference.H, reference.W
        ids = np.random.default_rng(0).permutation(H * W)[:8192]
        results = {}
        if not args.audit_only:
            functions = {
                name: (lambda source=source: source.rays(ids)) for name, source in sources.items()
            }
            results = measure_many(functions, args.repeats, args.iterations)
            for name, function in functions.items():
                results[name]["profile"] = trace(function, args.out / (name + ".json"))
                print(name, results[name]["median_wall_ms"], flush=True)
        differences = {
            name: {key: 0 for key in ["origins", "directions", "ranges", "visibility"]}
            for name in sources
            if name != "control"
        }
        for start in range(0, H * W, 8192):
            selected = np.arange(start, min(start + 8192, H * W))
            expected = reference.rays(selected)
            for name, source in sources.items():
                if name == "control":
                    continue
                actual = source.rays(selected)
                for key, a, b in zip(differences[name], expected, actual, strict=False):
                    differences[name][key] += int(
                        torch.count_nonzero(a.view(torch.int32) != b.view(torch.int32))
                    )
        report = dict(
            created_at=datetime.now(UTC).isoformat(),
            shape=[T, H, W],
            points=T * H * W,
            results=results,
            construction=construction,
            bitwise_differences=differences,
            passed=not any(n for row in differences.values() for n in row.values()),
            final_cache_metrics={
                name: source.observation_cache.metrics for name, source in sources.items()
            },
            input_hashes={
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [reference.depth_source, reference.track_source]
            },
            source_hashes={
                str(path.relative_to(args.source)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted((args.source / "relift").glob("*.py"))
            },
            command=sys.argv,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            torch=torch.__version__,
            gpu=torch.cuda.get_device_name(),
            native_build=json.loads((args.build / "build.json").read_text())
            if args.build
            else None,
            caveat="Warm rotating component timings; separate traces; constructors and audit are excluded from warm timings. Full-process trials include preparation/upload costs.",
        )
        (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                dict(points=report["points"], passed=report["passed"], differences=differences)
            ),
            flush=True,
        )
        if not report["passed"]:
            raise SystemExit(1)
    finally:
        for source in sources.values():
            source.close()


if __name__ == "__main__":
    main()
