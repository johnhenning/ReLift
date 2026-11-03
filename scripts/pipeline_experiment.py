"""Fresh full-pipeline validation shared by frozen and optimized source trials."""

from __future__ import annotations

import json
import os
import runpy
import sys
from typing import Any


def run(args: Any, options: dict[str, Any]) -> Any:
    from relift.storage import TrajectoryStore

    video = (args.root / options.pop("video")).resolve()
    optimizations = options.pop("optimizations", None)
    offload = options.pop("offload", "ram")
    if options:
        raise ValueError(f"Unknown pipeline options: {options}")
    output = args.output.resolve()
    if (output / "window").exists():
        raise ValueError("Full-pipeline trials must not reuse a prepared window")
    script = args.source.resolve() / "scripts/demo_video.py"
    previous = sys.argv
    sys.argv = [
        str(script),
        str(video),
        "--num_frames",
        "0",
        "--process-res",
        "504",
        "--corrector-batch",
        str((optimizations or {}).get("corrector_batch", 8)),
        "--checkpoints_root",
        str(args.root / "checkpoints"),
        "--out",
        str(output),
        "--offload",
        offload,
        "--scratch-dir",
        str(output / "scratch"),
        "--no_video",
    ]
    if optimizations is not None:
        for key, value in optimizations.items():
            sys.argv += ["--opt", f"{key}={json.dumps(value)}"]
    if os.environ.get("RELIFT_NATIVE_BUILD") and optimizations is not None:
        sys.argv += ["--native-build", os.environ["RELIFT_NATIVE_BUILD"]]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = previous
    metrics = json.loads((output / "metrics.json").read_text())
    store = TrajectoryStore.open(output)
    return dict(
        task="pipeline",
        phases=metrics["stages"],
        shape=store.shape,
        finite=store.finite(),
        observation_cache=json.loads((output / "observation-cache.json").read_text())
        if (output / "observation-cache.json").exists()
        else None,
        source=str(args.source.resolve()),
        scope="Fresh full clip decode, DA3, AllTracker, correction and durable output commits; excludes viewer/video export and provisioning.",
    )
