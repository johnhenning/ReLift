#!/usr/bin/env python
"""Profile DA3 construction separately from forward inference, against explicit source.

Run outside headline timing. The profile includes checkpoint loading and pager
installation; weights are locally provisioned and the filesystem cache is retained.
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import hashlib
import json
import os
import pstats
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from relift.records.artifacts import DA3StartupProfile
    from relift.records.runtime import ProfileFunction


class ProfileStats(Protocol):
    stats: dict[tuple[str, int, str], tuple[int, int, float, float, object]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", default="ram")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, str(args.source.resolve()))
    sys.path.insert(0, str(args.root / "scripts"))
    import torch

    from relift.configuration import load_configuration
    from relift.optimization import InferenceSession
    from scripts.benchmark_offload import NvidiaSampler

    gpu = NvidiaSampler()
    if gpu.compute_processes():
        raise RuntimeError("Startup diagnostic requires an idle compute GPU")
    args.out.mkdir(parents=True)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    profiler = cProfile.Profile()
    selection = load_configuration(profile=args.profile)
    started = time.perf_counter()
    with InferenceSession(selection) as session:
        profiler.enable()
        provider = session.depth(
            args.root / "checkpoints/da3",
            "cuda",
            offload=selection.config.memory.offload,
            scratch_dir=args.out / "scratch",
        )
        torch.cuda.synchronize()
        profiler.disable()
        seconds = time.perf_counter() - started
        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        session.release_depth(provider)
        del provider
    gc.collect()
    profiler.dump_stats(str(args.out / "startup.pstats"))
    # pstats' public stats table uses (filename, line, function) tuple keys.
    functions: list[ProfileFunction] = [
        dict(
            file=key[0],
            line=key[1],
            function=key[2],
            calls=value[1],
            self_seconds=value[2],
            inclusive_seconds=value[3],
        )
        for key, value in cast(ProfileStats, pstats.Stats(profiler)).stats.items()
    ]
    report: DA3StartupProfile = dict(
        source=str(args.source.resolve()),
        source_manifest_sha256=hashlib.sha256(
            (args.source / "snapshot.json").read_bytes()
        ).hexdigest(),
        configuration=selection.as_dict(),
        startup_seconds=seconds,
        cuda_allocated_peak_bytes=allocated,
        cuda_reserved_peak_bytes=reserved,
        functions=sorted(functions, key=lambda row: row["inclusive_seconds"], reverse=True),
        caveat="cProfile diagnostic; inclusive spans overlap. Not an unprofiled latency sample.",
    )
    (args.out / "profile.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
