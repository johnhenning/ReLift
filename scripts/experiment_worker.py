#!/usr/bin/env python
"""One isolated optimization trial. Invoked by run_optimization_experiments.py.

Task results separate model construction, preparation, execution and serialization.
The source checkout and options are explicit; diagnostic profiles are not timings.
"""

from __future__ import annotations

import argparse
import cProfile
import functools
import hashlib
import json
import os
import pstats
import random
import resource
import sys
import time
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch


class ProfileStats(Protocol):
    """CPython pstats data, absent from its installed stubs."""

    stats: dict[tuple[str, int, str], tuple[int, int, float, float, Any]]


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


class Phases:
    def __init__(self) -> None:
        self.results = {}

    @contextmanager
    def stage(self, name: str) -> Generator[None]:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.results[name] = dict(
                seconds=time.perf_counter() - started,
                process_rss_highwater_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                cuda_allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                cuda_reserved_peak_bytes=torch.cuda.max_memory_reserved(),
            )


class Diagnostic:
    """Explicit synchronized inclusive spans plus cProfile and a short CUDA trace."""

    def __init__(self, output: Any) -> None:
        self.output = output
        self.spans = defaultdict(lambda: dict(calls=0, seconds=0.0))
        self.originals = []
        self.cpu = cProfile.Profile()

    def wrap(self, owner: Any, name: str, label: Any, cuda: bool = False) -> Any:
        original = getattr(owner, name)

        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if cuda:
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.profiler.record_function(label):
                result = original(*args, **kwargs)
            if cuda:
                torch.cuda.synchronize()
            record = self.spans[label]
            record["calls"] += 1
            record["seconds"] += time.perf_counter() - started
            return result

        self.originals.append((owner, name, original))
        setattr(owner, name, wrapped)

    def __enter__(self) -> Any:
        import relift.storage as storage
        import relift.streaming as streaming

        for owner, name, label, cuda in (
            (streaming.MappedObservations, "rays", "rays", True),
            (streaming, "correct", "corrector", True),
            (streaming, "interp_init", "raw_interpolation", True),
            (streaming, "mapped_archive", "mapped_archive", False),
            (storage, "file_sha256", "file_hash", False),
            (streaming, "file_sha256", "input_hash", False),
            (storage, "atomic_npy", "npy_commit", False),
            (storage, "atomic_json", "manifest_commit", False),
            (storage.TrajectoryStore, "write_shard", "shard_write_inclusive", False),
            (storage.TrajectoryStore, "finish", "finish_verify", False),
        ):
            self.wrap(owner, name, label, cuda)
        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
            record_shapes=False,
            profile_memory=True,
            on_trace_ready=lambda profiler: profiler.export_chrome_trace(
                str(self.output / "cuda-trace.json")
            ),
        )
        self.profiler.__enter__()
        original = storage.TrajectoryStore.write_shard

        def step(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            self.profiler.step()
            return result

        self.originals.append((storage.TrajectoryStore, "write_shard", original))
        storage.TrajectoryStore.write_shard = step
        self.cpu.enable()
        return self

    def __exit__(self, *error: Any) -> None:
        self.cpu.disable()
        self.profiler.__exit__(*error)
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)
        self.cpu.dump_stats(str(self.output / "cpu-profile.pstats"))
        stats = pstats.Stats(self.cpu)
        functions = [
            dict(
                file=key[0],
                line=key[1],
                function=key[2],
                calls=value[1],
                self_seconds=value[2],
                inclusive_seconds=value[3],
            )
            for key, value in cast(ProfileStats, stats).stats.items()
        ]
        save(
            self.output / "diagnostic.json",
            dict(
                spans=dict(self.spans),
                caveat="Diagnostic synchronized inclusive spans overlap and alter execution. Never use their sum or runtime as an unprofiled speed measurement.",
                functions=sorted(functions, key=lambda row: row["self_seconds"], reverse=True),
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="correction")
    parser.add_argument("--options", default="{}")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    options = json.loads(args.options)
    options.pop("native_build", None)  # runner supplies the verified build through the environment
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    phases = Phases()
    configuration = None
    if any(key in options for key in ("inference_profile", "inference_config", "configuration")):
        from relift.configuration import experiment_configuration

        configuration = experiment_configuration(options, args.root)
        for key in ("inference_profile", "inference_config", "configuration"):
            options.pop(key, None)
    if configuration is not None and args.task not in ("pipeline", "correction", "da3-configured"):
        raise ValueError(
            "Grouped configuration applies to correction, pipeline or da3-configured tasks"
        )
    if args.task == "pipeline":
        from pipeline_experiment import run

        result = run(args, options, configuration=configuration)
        save(output / "result.json", result)
        print(json.dumps(dict(shape=result["shape"], finite=result["finite"])), flush=True)
        return
    if args.task == "da3-reuse":
        from provider_reuse_experiment import run

        result = run(args, options, phases)
        save(output / "result.json", result)
        print(json.dumps(result["totals"]), flush=True)
        return
    if args.task == "da3-configured":
        from configured_depth_experiment import run

        result = run(args, options, phases, configuration=configuration)
        save(output / "result.json", result)
        print(json.dumps(result["counters"]), flush=True)
        return
    if args.task == "alltracker":
        from relift.providers.alltracker import AllTrackerProvider
        from relift.storage import file_sha256

        with phases.stage("input_load"):
            with np.load(output / "window/da3_frames.npz") as archive:
                images = archive["images"]
        with phases.stage("provider_load"):
            provider = AllTrackerProvider(
                args.root / "checkpoints/alltracker/alltracker.pth", **options
            )
        state = hashlib.sha256()
        for name, value in sorted(provider.net.state_dict().items()):
            state.update(name.encode())
            state.update(value.detach().cpu().contiguous().numpy().tobytes())
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        with phases.stage("provider_forward"):
            tracks, visibility, confidence = provider.run(images)
        with phases.stage("serialize"):
            np.savez(
                output / "tracker.npz", tracks=tracks, visibility=visibility, confidence=confidence
            )
        save(
            output / "result.json",
            dict(
                task=args.task,
                phases=phases.results,
                state_sha256=state.hexdigest(),
                artifacts={"tracker.npz": file_sha256(output / "tracker.npz")},
                source=str(args.source.resolve()),
                shape=tracks.shape,
            ),
        )
        print(json.dumps(phases.results), flush=True)
        return
    from relift.checkpoint import load
    from relift.storage import file_sha256
    from relift.streaming import stream_correct

    with phases.stage("model_load"):
        model, info = load(args.root / "checkpoints/released_checkpoint.pt")
    if args.task == "compilation":
        from compilation_experiment import run

        result = run(args, options, phases, model, info)
        save(output / "result.json", result)
        print(json.dumps(result["phases"]), flush=True)
        return
    if args.task != "correction":
        raise ValueError(f"Unknown task: {args.task}")
    diagnostic = Diagnostic(output) if options.pop("profile", False) else None
    lookahead = None
    if options.pop("prefetch", False):
        from relift.prefetch import RayLookahead

        lookahead = RayLookahead(batch=options.get("batch", 8))
    if diagnostic:
        diagnostic.__enter__()
    try:
        with phases.stage("correction"), lookahead if lookahead else nullcontext():
            if configuration is not None:
                options["optimizations"] = configuration.config.optimizations().as_dict()
                options.setdefault("ram_budget_gb", configuration.config.memory.ram_gb)
                options.setdefault("vram_budget_gb", configuration.config.memory.vram_gb)
                options.setdefault("scratch_budget_gb", configuration.config.memory.scratch_gb)
            optimization_values = options.pop("optimizations", None)
            if optimization_values is not None:
                from relift.optimization import InferenceComponents, OptimizationConfig

                options["optimizations"] = OptimizationConfig.from_dict(optimization_values)
                components = InferenceComponents()
                native_build = os.environ.get("RELIFT_NATIVE_BUILD") or (
                    configuration.config.runtime.native_build if configuration else None
                )
                if native_build:
                    components = components.with_native_build(native_build)
                options["components"] = components
            store = stream_correct(
                output / "window",
                output,
                model,
                dict(
                    checkpoint_sha256=file_sha256(args.root / "checkpoints/released_checkpoint.pt"),
                    checkpoint_step=info["step"],
                    **({"configuration": configuration.as_dict()} if configuration else {}),
                ),
                group=1024,
                batch=options.pop("batch", 8),
                seed=0,
                **options,
            )
    finally:
        if diagnostic:
            diagnostic.__exit__(*sys.exc_info())
    execution = (
        dict(
            peak_pinned_bytes=lookahead.peak_pinned_bytes,
            prefetched_host_to_device_bytes=lookahead.host_to_device_bytes,
        )
        if lookahead
        else {}
    )
    save(
        output / "result.json",
        dict(
            task=args.task,
            phases=phases.results,
            shape=store.shape,
            execution=execution,
            configuration=configuration.as_dict() if configuration else None,
            observation_cache=json.loads((output / "observation-cache.json").read_text())
            if (output / "observation-cache.json").exists()
            else None,
            finite=store.finite(),
            source=str(args.source.resolve()),
            started_at=datetime.now(UTC).isoformat(),
        ),
    )
    print(json.dumps(dict(phases=phases.results, shape=store.shape)), flush=True)


if __name__ == "__main__":
    main()
