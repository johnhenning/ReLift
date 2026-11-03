"""DA3-only trials through the same config and injected session used by demos."""

from __future__ import annotations

import gc
import os
from typing import Any

import numpy as np
import torch
from provider_reuse_experiment import frames, reset_seed


def run(args: Any, options: dict[str, Any], phases: Any) -> Any:
    from relift.optimization import InferenceSession, OptimizationConfig
    from relift.storage import file_sha256

    settings = OptimizationConfig.from_dict(options.pop("optimizations", {}))
    clips = options.pop("clips")
    mode = options.pop("offload", "ram")
    if options or mode not in ("ram", "disk") or not clips:
        raise ValueError("Expected clips, RAM/disk mode and optimization settings")
    artifacts, shapes, inputs, counters = {}, {}, {}, []
    with InferenceSession(settings, native_build=os.environ.get("RELIFT_NATIVE_BUILD")) as session:
        for index, filename in enumerate(clips):
            path = (args.root / filename).resolve()
            inputs[filename] = file_sha256(path)
            with phases.stage(f"clip_{index}_decode"):
                images = frames(path)
            reset_seed()
            with phases.stage(f"clip_{index}_provider_load"):
                provider = session.depth(
                    args.root / "checkpoints/da3",
                    "cuda",
                    offload=mode,
                    scratch_dir=args.output / "scratch",
                    ram_budget_gb=18,
                    scratch_budget_gb=64,
                )
            try:
                reset_seed()
                with phases.stage(f"clip_{index}_forward"):
                    geometry = provider.run(images)
                with phases.stage(f"clip_{index}_serialize"):
                    name = f"clip-{index}.npz"
                    np.savez(args.output / name, **geometry)
                artifacts[name] = file_sha256(args.output / name)
                shapes[name] = {key: list(value.shape) for key, value in geometry.items()}
                counters.append(
                    dict(
                        cold_bytes_written=provider.pager.storage.bytes_written,
                        transfer_calls=provider.pager.transfer_calls,
                        weight_transfer_bytes=provider.pager.transferred_weight_bytes,
                        buffer_transfer_bytes=provider.pager.transferred_buffer_bytes,
                        scratch_files=sum(
                            1 for p in provider.pager.storage.root.rglob("*") if p.is_file()
                        ),
                    )
                )
                del images, geometry
            finally:
                with phases.stage(f"clip_{index}_release"):
                    session.release_depth(provider)
                    gc.collect()
                    torch.cuda.empty_cache()
    return dict(
        task="da3-configured",
        phases=phases.results,
        artifacts=artifacts,
        shapes=shapes,
        input_sha256=inputs,
        optimizations=settings.as_dict(),
        counters=counters,
        scope="Full-clip DA3 decode, construction, archive preparation, forward, serialization and cleanup; excludes tracking/correction.",
    )
