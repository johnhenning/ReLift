"""Two-clip DA3 lifecycle experiment, invoked by the isolated trial worker.

Only RAM placement is tested here. Provider disposal remains explicit; this does
not introduce a process-global model cache into ordinary demo inference.
"""
import gc
import random

import cv2
import numpy as np
import torch


def reset_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)


def frames(path):
    cap = cv2.VideoCapture(str(path))
    values = []
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            values.append(image[..., ::-1])
    finally:
        cap.release()
    if len(values) < 2:
        raise ValueError(f"Cannot decode a full clip: {path}")
    return np.stack(values)


def run(args, options, phases):
    from relift.providers.da3 import DepthAnything3Provider
    from relift.storage import file_sha256

    reuse = options.pop("reuse")
    clips = options.pop("clips")
    if options or not isinstance(reuse, bool) or len(clips) < 2:
        raise ValueError("Expected reuse flag and at least two complete clips")
    output = args.output.resolve()
    provider = None
    artifacts, shapes, inputs = {}, {}, {}
    try:
        for index, filename in enumerate(clips):
            path = (args.root / filename).resolve()
            with phases.stage(f"clip_{index}_decode"):
                images = frames(path)
            inputs[filename] = file_sha256(path)
            if provider is None:
                reset_seed()
                with phases.stage(f"clip_{index}_provider_load"):
                    provider = DepthAnything3Provider(args.root / "checkpoints/da3", offload="ram",
                        scratch_dir=output / "scratch", ram_budget_gb=18, scratch_budget_gb=64)
            reset_seed()
            with phases.stage(f"clip_{index}_forward"):
                geometry = provider.run(images)
            with phases.stage(f"clip_{index}_serialize"):
                name = f"clip-{index}.npz"
                np.savez(output / name, **geometry)
            artifacts[name] = file_sha256(output / name)
            shapes[name] = {key: list(value.shape) for key, value in geometry.items()}
            del images, geometry
            if not reuse:
                with phases.stage(f"clip_{index}_cleanup"):
                    provider.close()
                    provider = None
                    gc.collect()
                    torch.cuda.empty_cache()
    finally:
        if provider is not None:
            with phases.stage("final_cleanup"):
                provider.close()
                provider = None
                gc.collect()
                torch.cuda.empty_cache()
    totals = {label: sum(row["seconds"] for name, row in phases.results.items() if name.endswith(suffix))
              for label, suffix in (("provider_load_seconds", "_provider_load"), ("forward_seconds", "_forward"))}
    return dict(task="da3-reuse", phases=phases.results, artifacts=artifacts, shapes=shapes,
                input_sha256=inputs, totals=totals, reused=reuse,
                source=str(args.source.resolve()),
                scope="Two complete clips in one process; same RAM offload, resolution, weights and per-clip seeds. No AllTracker/corrector timing.")
