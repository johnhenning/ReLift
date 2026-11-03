#!/usr/bin/env python
"""Separate fresh-process factory/first-call cost from warm ray math.

No warmup precedes the first measured builder call. Inputs and native loading are
provisioned first and timed separately. Raw per-process records are never replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from relift.records.artifacts import StartupRecord


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--window", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--backend", choices=["torch", "cuda", "timeline", "timeline_registered"])
    args = parser.parse_args()
    if args.backend:
        import numpy as np
        import torch

        sys.path.insert(0, str(args.source.resolve()))
        from relift.cuda_rays import make_ray_builder, torch_ray_builder
        from relift.native_backend import load_cpu
        from relift.streaming import MappedObservations

        torch.set_grad_enabled(False)
        start = time.perf_counter()
        obs = MappedObservations(args.window, "cuda")
        ids = np.arange(8192)
        coords = torch.as_tensor(
            np.array(obs.tracks[:16, ids // obs.W, ids % obs.W]), device="cuda", dtype=torch.float32
        )
        depth = torch.as_tensor(np.array(obs.depth[:16]), device="cuda", dtype=torch.float32)
        values = (coords, depth, obs.K[:16], obs.Rwc[:16])
        native = load_cpu(args.build) if args.backend != "torch" else None
        torch.cuda.synchronize()
        provision = time.perf_counter() - start
        start = time.perf_counter()
        if args.backend == "torch":
            builder = torch_ray_builder
        elif args.backend == "cuda":
            assert native is not None
            builder = make_ray_builder(native)
        else:
            assert native is not None
            from relift.cuda_rays import TimelineRayBuilder

            builder = (
                TimelineRayBuilder(native, registered=True)
                if args.backend == "timeline_registered"
                else TimelineRayBuilder(native)
            )
        factory = time.perf_counter() - start
        samples = []
        for _ in range(6):
            torch.cuda.synchronize()
            start = time.perf_counter()
            builder(*values)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - start)
        record: StartupRecord = dict(
            backend=args.backend,
            provision_seconds=provision,
            factory_seconds=factory,
            first_call_seconds=samples[0],
            subsequent_call_seconds=samples[1:],
            torch=torch.__version__,
            gpu=torch.cuda.get_device_name(),
            created_at=datetime.now(UTC).isoformat(),
            source=str(args.source.resolve()),
            ray_source_sha256=hashlib.sha256(
                (args.source / "relift/cuda_rays.py").read_bytes()
            ).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            native_build=json.loads((args.build / "build.json").read_text()),
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as stream:
            json.dump(record, stream, indent=2)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    # Only request registered timeline mode if the selected source supports it.
    backends = ["torch", "cuda", "timeline"]
    if "registered=" in (args.source / "relift/cuda_rays.py").read_text():
        backends.append("timeline_registered")
    records = []
    for repeat in range(args.repeats):
        offset = repeat % len(backends)
        for backend in backends[offset:] + backends[:offset]:
            path = args.out / f"{backend}-{repeat + 1:02}.json"
            command = [
                sys.executable,
                __file__,
                "--source",
                str(args.source),
                "--build",
                str(args.build),
                "--window",
                str(args.window),
                "--out",
                str(path),
                "--backend",
                backend,
            ]
            subprocess.run(command, check=True)
            record = json.loads(path.read_text())
            record["command"] = command
            records.append(record)
            print(
                backend,
                repeat + 1,
                record["factory_seconds"],
                record["first_call_seconds"],
                flush=True,
            )
    summary = {
        backend: dict(
            provision_seconds=statistics.median(
                r["provision_seconds"] for r in records if r["backend"] == backend
            ),
            factory_seconds=statistics.median(
                r["factory_seconds"] for r in records if r["backend"] == backend
            ),
            first_call_seconds=statistics.median(
                r["first_call_seconds"] for r in records if r["backend"] == backend
            ),
        )
        for backend in backends
    }
    (args.out / "summary.json").write_text(
        json.dumps(dict(records=records, medians=summary), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
