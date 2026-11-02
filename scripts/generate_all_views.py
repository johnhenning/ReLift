#!/usr/bin/env python
"""Generate full-length offline Plotly viewers for every official demo input.

    uv run python scripts/generate_all_views.py
    uv run python scripts/generate_all_views.py --download-only

GPU jobs run sequentially. Completed results are validated and reused; failed
clips do not prevent the others from running. Rerun the same command to resume.
Each clip gets inference/export logs. index.html and batch.json track progress.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shlex
import subprocess
import sys
from urllib.parse import quote

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from relift.config import paths
from relift.storage import TrajectoryStore
from scripts.download_demo_inputs import catalog, fetch_input, sha256


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def complete_result(output, source, process_resolution=None):
    """Never reuse a short window or an HTML export of different trajectory data."""
    output = Path(output)
    arrays, manifest, page = output / "tracks.npz", output / "plotly.json", output / "trajectories.html"
    store_manifest = output / "tracks.store/manifest.json"
    if not manifest.is_file() or not page.is_file() or not (arrays.is_file() or store_manifest.is_file()):
        return None
    metadata = json.loads(manifest.read_text())
    if metadata.get("frames") != source["frames"] or metadata.get("requested_frames") != 0:
        return None
    if not np.isclose(metadata.get("source_fps", 0), source["fps"]):
        return None
    if not metadata.get("finite") or metadata.get("temporal_sampling") != "none":
        return None
    if process_resolution is not None and metadata.get("process_resolution") != process_resolution:
        return None
    exported_source = metadata.get("source", "tracks.npz")
    if exported_source == "tracks.store/manifest.json":
        try:
            store = TrajectoryStore.open(output)
            store.verify()
            if (store.shape[0] != source["frames"] or store.metadata.get("requested_frames") != 0
                    or not np.isclose(store.metadata.get("source_fps", 0), source["fps"])):
                return None
            if process_resolution is not None and store.metadata.get("process_resolution") != process_resolution:
                return None
            if source.get("sha256") is not None and store.metadata.get("input_sha256") != source["sha256"]:
                return None
        except (ValueError, OSError, KeyError):
            return None
        arrays = store_manifest
    elif exported_source != "tracks.npz":
        return None
    if not arrays.is_file() or metadata.get("source_sha256") != sha256(arrays):
        return None
    if metadata.get("html_bytes") != page.stat().st_size:
        return None
    return metadata


def dense_result_ready(output, source, process_resolution=504):
    """Check complete dense data even when a viewer still needs to be exported."""
    try:
        store = TrajectoryStore.open(output)
        if store.manifest is not None:
            store.verify()
            if source.get("sha256") is not None and store.metadata.get("input_sha256") != source["sha256"]:
                return False
        return (store.shape[0] == source["frames"] and store.metadata.get("requested_frames") == 0
                and np.isclose(store.metadata.get("source_fps", 0), source["fps"])
                and store.metadata.get("process_resolution", max(store.shape[1:3])) == process_resolution)
    except (ValueError, OSError, KeyError):
        return False


def run_logged(script, arguments, log):
    command = [sys.executable, f"scripts/{script}", *map(str, arguments)]
    print("+ " + shlex.join(command), flush=True)
    with log.open("w") as stream:
        stream.write("+ " + shlex.join(command) + "\n")
        stream.flush()
        process = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)
        result = process.wait()
    if result:
        raise RuntimeError(f"{script} exited {result}; see {log.name}")


def infer_with_memory_retries(video, output, record):
    """Explicit legacy opt-in only: reduce spatial grids after CUDA OOM."""
    record["inference_attempts"] = []
    for resolution, batch in ((504, 8), (392, 4), (336, 4), (280, 2)):
        log = output / f"inference_res{resolution}.log"
        attempt = dict(resolution=resolution, corrector_batch=batch, log=log.name, started_at=now())
        record["inference_attempts"].append(attempt)
        try:
            run_logged("demo_video.py", [video, "--num_frames", 0, "--geometry", "da3", "--out", output,
                                        "--no_video", "--process-res", resolution, "--corrector-batch", batch], log)
            attempt.update(status="complete", finished_at=now())
            # Keep a predictable log path as well as every individual attempt.
            (output / "inference.log").write_text(log.read_text())
            record["processing_resolution"] = resolution
            return
        except RuntimeError:
            attempt.update(status="failed", finished_at=now())
            text = log.read_text()
            if "CUDA out of memory" not in text and "OutOfMemoryError" not in text:
                raise
            print(f"GPU memory retry: {output.name}, long side {resolution}; retaining all frames", flush=True)
    raise RuntimeError("CUDA memory exhausted at every configured spatial resolution; see inference_res*.log")


def infer_full_resolution(video, output, record, args):
    """Offload storage/placement, retaining the full source timeline and 504 grid."""
    log = output / f"inference_{args.offload}_res504.log"
    attempt = dict(resolution=504, corrector_batch=8, offload=args.offload, log=log.name, started_at=now())
    record["inference_attempts"] = [attempt]
    arguments = [video, "--num_frames", 0, "--geometry", "da3", "--out", output, "--no_video",
                 "--process-res", 504, "--corrector-batch", 8, "--offload", args.offload,
                 "--ram-budget-gb", args.ram_budget_gb, "--vram-budget-gb", args.vram_budget_gb,
                 "--scratch-budget-gb", args.scratch_budget_gb]
    if args.scratch_dir:
        arguments += ["--scratch-dir", args.scratch_dir]
    try:
        run_logged("demo_video.py", arguments, log)
        attempt.update(status="complete", finished_at=now())
        record["processing_resolution"] = 504
    except RuntimeError:
        attempt.update(status="failed", finished_at=now())
        raise
    finally:
        if log.exists():
            (output / "inference.log").write_text(log.read_text())


def write_index(root, state):
    rows = []
    for name, entry in catalog().items():
        record = state["videos"].get(name, {})
        source = record.get("source", {})
        status = record.get("status", "pending")
        page = root / name / "trajectories.html"
        link = f'<a href="{quote(name)}/trajectories.html">Open 3D viewer</a>' if status == "complete" and page.exists() else html.escape(status)
        image = f'<img src="{quote(name)}/input.jpg" alt="{html.escape(entry["description"])}" loading="lazy">' if (root / name / "input.jpg").exists() else ""
        frames = source.get("frames", "—")
        seconds = f'{source["duration_seconds"]:.2f} s' if source.get("duration_seconds") is not None else "—"
        metadata = record.get("result", {})
        tracks = f'{metadata["dense_trajectories"]:,}' if "dense_trajectories" in metadata else "—"
        if metadata.get("grid"):
            tracks += f'<p>{metadata["grid"][1]}×{metadata["grid"][0]} grid</p>'
        error = f'<p class="error">{html.escape(record["error"])}</p>' if record.get("error") else ""
        rows.append(f'<tr><td>{image}</td><td><strong>{html.escape(name)}</strong><p>{html.escape(entry["description"])}</p>{error}</td>'
                    f'<td>{frames}</td><td>{seconds}</td><td>{tracks}</td><td>{link}</td></tr>')
    finished = sum(v.get("status") == "complete" for v in state["videos"].values())
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>ReLift interactive demo gallery</title>
<style>body{font:15px/1.5 system-ui,sans-serif;color:#243047;background:#fff;margin:0}main{max-width:1200px;margin:auto;padding:24px}
h1{margin:0 0 8px}p{margin:4px 0;color:#57647a}table{width:100%;border-collapse:collapse;margin-top:24px}
th,td{text-align:left;padding:12px;border-bottom:1px solid #dce2ea;vertical-align:middle}a{color:#165da1}img{width:128px;height:84px;object-fit:contain;background:#f3f5f8}
.error{color:#a82222}.table-wrap{overflow-x:auto}th{white-space:nowrap}@media(max-width:600px){main{padding:12px}img{width:80px;height:64px}th,td{padding:8px}}</style>
</head><body><main><h1>ReLift · interactive demo gallery</h1>
<p>__COUNT__ completed videos. Every viewer contains all source frames, synchronized raw/ReLift 3D views, playback, frame scrubbing, and clickable complete trajectories.</p>
<p>Plots sample reference pixels for browser performance. Saved tracks.store directories or legacy tracks.npz files keep dense trajectories. Source clips are the published demonstration videos.</p>
<div class="table-wrap"><table><thead><tr><th>Input</th><th>Scene</th><th>Frames</th><th>Duration</th><th>Dense tracks</th><th>Interactive view</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<p>Generated by scripts/generate_all_views.py. Detailed progress and provenance are in batch.json; each output directory contains inference.log, plotly.log, and plotly.json.</p>
</main></body></html>'''
    (root / "index.html").write_text(page.replace("__COUNT__", str(finished)).replace("__ROWS__", "\n".join(rows)))
    state["updated_at"] = now()
    save_json(root / "batch.json", state)


def prepare_input(name, output):
    path = fetch_input(name)
    source = json.loads((paths()["cache_root"] / "demo_inputs" / f"{name}.json").read_text())
    if source["frames"] < 2 or source["fps"] <= 0:
        raise ValueError("input must have at least two frames and a positive FPS")
    output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    try:
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(output / "input.jpg"), frame)
    finally:
        cap.release()
    return path, source


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root", type=Path, default=paths()["outputs_root"] / "demo-full")
    ap.add_argument("--videos", nargs="+", choices=tuple(catalog()), help="default: all catalog videos")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--status", action="store_true", help="print saved progress without downloading or running models")
    ap.add_argument("--offload", choices=["none", "streamed", "ram", "disk", "auto"], default="auto")
    ap.add_argument("--scratch-dir", type=Path)
    ap.add_argument("--ram-budget-gb", type=float, default=18)
    ap.add_argument("--vram-budget-gb", type=float, default=20)
    ap.add_argument("--scratch-budget-gb", type=float, default=64)
    ap.add_argument("--allow-resolution-retry", action="store_true", help="explicit legacy opt-in; only with --offload none")
    args = ap.parse_args()
    if min(args.ram_budget_gb, args.vram_budget_gb, args.scratch_budget_gb) <= 0:
        ap.error("memory and scratch budgets must be positive")
    if args.allow_resolution_retry and args.offload != "none":
        ap.error("--allow-resolution-retry is only supported with --offload none")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    names = args.videos or list(catalog())
    manifest = root / "batch.json"
    state = json.loads(manifest.read_text()) if manifest.exists() else dict(started_at=now(), videos={})
    if args.status:
        for name in names:
            record = state["videos"].get(name, {})
            print(f"{name:20} {record.get('status', 'pending'):10} {record.get('source', {}).get('frames', '—')} frames")
        return
    inputs = {}
    # Downloads and input inspection are independent; GPU inference stays sequential.
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(prepare_input, name, root / name): name for name in names}
        for future in as_completed(futures):
            name = futures[future]
            try:
                path, source = future.result()
                inputs[name] = path
                record = state["videos"].get(name, {})
                if record.get("source", {}).get("sha256") not in (None, source["sha256"]):
                    raise ValueError("source input changed; choose a fresh output root to avoid mixing cached windows")
                state["videos"][name] = dict(status="ready", source=source)
            except Exception as error:
                state["videos"][name] = dict(status="failed", stage="download", error=str(error))
                print(f"FAILED {name}: {error}", flush=True)
            write_index(root, state)
    if args.download_only:
        print(f"Input inventory: {root / 'batch.json'}", flush=True)
        return
    run_logged("download_weights.py", ["--only", "relift", "da3", "alltracker"], root / "weights.log")
    for name in names:
        record = state["videos"][name]
        if name not in inputs or record["status"] == "failed":
            continue
        output = root / name
        try:
            source = record["source"]
            required_resolution = None if args.allow_resolution_retry else 504
            metadata = complete_result(output, source, required_resolution)
            if metadata:
                record.update(status="complete", result=metadata, reused=True)
                print(f"REUSE {name}: {metadata['frames']} full-length frames", flush=True)
            else:
                record.update(status="running", stage="inference", started_at=now())
                write_index(root, state)
                # Existing complete dense arrays can be re-exported without GPU work.
                dense_ready = dense_result_ready(output, source)
                if not dense_ready:
                    if args.allow_resolution_retry:
                        infer_with_memory_retries(inputs[name], output, record)
                    else:
                        infer_full_resolution(inputs[name], output, record, args)
                record["stage"] = "plotly"
                write_index(root, state)
                run_logged("plotly_demo.py", [output], output / "plotly.log")
                metadata = complete_result(output, source, required_resolution)
                if metadata is None:
                    raise ValueError("export does not match the full source frame count or trajectory checksum")
                record.update(status="complete", result=metadata, finished_at=now(), reused=False)
                record.pop("error", None)
                print(f"COMPLETE {name}: {metadata['frames']} frames, {metadata['dense_trajectories']} dense tracks", flush=True)
        except Exception as error:
            record.update(status="failed", error=str(error), finished_at=now())
            print(f"FAILED {name}: {error}", flush=True)
        write_index(root, state)
    failed = [name for name in names if state["videos"][name]["status"] != "complete"]
    print(f"Gallery: {root / 'index.html'}; failed: {failed}", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
