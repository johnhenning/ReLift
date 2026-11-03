"""Fresh full-pipeline validation shared by frozen and optimized source trials."""
import json
import runpy
import sys


def run(args, options):
    from relift.storage import TrajectoryStore

    video = (args.root / options.pop("video")).resolve()
    if options:
        raise ValueError(f"Unknown pipeline options: {options}")
    output = args.output.resolve()
    if (output / "window").exists():
        raise ValueError("Full-pipeline trials must not reuse a prepared window")
    script = args.source.resolve() / "scripts/demo_video.py"
    previous = sys.argv
    sys.argv = [str(script), str(video), "--num_frames", "0", "--process-res", "504",
                "--corrector-batch", "8", "--checkpoints_root", str(args.root / "checkpoints"),
                "--out", str(output), "--offload", "ram", "--scratch-dir", str(output / "scratch"), "--no_video"]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = previous
    metrics = json.loads((output / "metrics.json").read_text())
    store = TrajectoryStore.open(output)
    return dict(task="pipeline", phases=metrics["stages"], shape=store.shape, finite=bool(store.finite()),
                source=str(args.source.resolve()),
                scope="Fresh full clip decode, DA3, AllTracker, correction and durable output commits; excludes viewer/video export and provisioning.")
