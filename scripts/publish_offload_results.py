#!/usr/bin/env python
"""Copy validated full-resolution benchmark data into the demo gallery.

    uv run python scripts/publish_offload_results.py --scenes running forest
    uv run python scripts/generate_all_views.py --videos running forest

No inference runs here. The second command exports viewers from the copied store.
Existing destination stores are never replaced. Benchmark originals remain intact.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relift.storage import TrajectoryStore, file_sha256
from scripts.generate_all_views import dense_result_ready


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, default=ROOT / "outputs/benchmarks/offload-capacity"
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/demo-full")
    parser.add_argument("--scenes", nargs="+", default=["running", "forest"])
    parser.add_argument("--mode", default="ram")
    parser.add_argument("--trial", type=int, default=1)
    args = parser.parse_args()
    for scene in args.scenes:
        trial_dir = args.source_root / scene / args.mode / f"trial-{args.trial:02}"
        trial = json.loads((trial_dir / "trial.json").read_text())
        source = json.loads((ROOT / "cache/demo_inputs" / f"{scene}.json").read_text())
        output = trial_dir / "output"
        if (
            trial["status"] != "complete"
            or trial["process_res"] != 504
            or trial["num_frames"] != 0
            or not dense_result_ready(output, source, 504)
        ):
            raise ValueError(f"Not a validated full-resolution, full-length result: {trial_dir}")
        store = TrajectoryStore.open(output)
        if not store.finite():
            raise ValueError(f"Nonfinite trajectories: {output}")
        destination = args.output_root / scene
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / "tracks.store"
        if target.exists():
            if file_sha256(target / "manifest.json") != file_sha256(
                output / "tracks.store/manifest.json"
            ):
                raise FileExistsError(f"Refusing to replace different data: {target}")
            TrajectoryStore.open(destination).verify()
        else:
            temporary = destination / "tracks.store.copying"
            if temporary.exists():
                raise FileExistsError(f"Inspect interrupted copy before retrying: {temporary}")
            shutil.copytree(output / "tracks.store", temporary)
            TrajectoryStore.open(temporary).verify()
            temporary.rename(target)
        shutil.copy2(trial_dir / "run.log", destination / "inference.log")
        shutil.copy2(output / "metrics.json", destination / "metrics.json")
        shutil.copy2(trial_dir / "trial.json", destination / "benchmark-trial.json")
        print(f"Published {scene}: {store.shape[0]} frames, {store.shape[2]}×{store.shape[1]} grid")


if __name__ == "__main__":
    main()
