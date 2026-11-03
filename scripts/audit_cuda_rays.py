#!/usr/bin/env python
"""Compare real rays and encoded nearest-pixel probes without running the model.

uv run python scripts/audit_cuda_rays.py --source outputs/experiments/native-rays-source \
    --build outputs/native/rays --window PREPARED_WINDOW --out outputs/ray-audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--window", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--mapped-backend",
        choices=("cuda", "cuda_timeline"),
        help="Also compare complete timelines through MappedObservations, with cache off/on",
    )
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    sys.path.insert(0, str(args.source.resolve()))
    from relift.cuda_rays import make_ray_builder, torch_ray_builder
    from relift.native_backend import load_cpu
    from relift.streaming import MappedObservations

    build = make_ray_builder(load_cpu(args.build))
    source = MappedObservations(args.window, "cuda")
    H, W = source.H, source.W
    code = torch.arange(H * W, device="cuda", dtype=torch.float32).reshape(H, W)
    totals = dict(
        points=0,
        nearest_mismatches=0,
        direction_components_different=0,
        max_direction_absolute_error=0.0,
        max_range_absolute_error=0.0,
        range_values_different=0,
    )
    examples = []
    decomposition = {}
    # Scan every original pixel and temporal frame, using the input's fp16
    # tracker coordinates, rather than only easy ties near the origin.
    for start in range(0, source.T, 16):
        stop = min(start + 16, source.T)
        T = stop - start
        coords = torch.as_tensor(
            np.array(source.tracks[start:stop]).reshape(T, H * W, 2),
            device="cuda",
            dtype=torch.float32,
        )
        depth = torch.as_tensor(
            np.array(source.depth[start:stop]), device="cuda", dtype=torch.float32
        )
        K, R = source.K[start:stop], source.Rwc[start:stop]
        eager = torch_ray_builder(coords, depth, K, R)
        fused = build(coords, depth, K, R)
        if start == 0:
            x = (coords[..., 0] - K[:, 0, 2, None]) / K[:, 0, 0, None]
            y = (coords[..., 1] - K[:, 1, 2, None]) / K[:, 1, 1, None]
            camera = torch.stack([x, y, torch.ones_like(x)], -1)
            length = torch.linalg.vector_norm(camera, dim=-1)
            for name, value in {
                "xy_then_one": torch.sqrt((x * x + y * y) + 1),
                "x_one_then_y": torch.sqrt((x * x + 1) + y * y),
                "x_then_y_one": torch.sqrt(x * x + (y * y + 1)),
            }.items():
                decomposition["norm_" + name] = int(torch.count_nonzero(length != value))
            identity = torch.eye(3, device="cuda").repeat(T, 1, 1)
            native_camera = build(coords, depth, K, identity)[0]
            expected_camera = F.normalize(camera, dim=-1)
            decomposition["camera_components_different"] = int(
                torch.count_nonzero(native_camera != expected_camera)
            )
            native_rotated = torch.einsum("tij,tnj->tni", R, native_camera)
            decomposition["rotation_components_different_given_native_camera"] = int(
                torch.count_nonzero(native_rotated != fused[0])
            )

            def fma(a: Any, b: Any, c: Any) -> Any:
                return (a.double() * b.double() + c.double()).float()

            for order in ((0, 1, 2), (2, 1, 0), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1)):
                acc = R[:, None, :, order[0]] * expected_camera[:, :, None, order[0]]
                for k in order[1:]:
                    acc = fma(R[:, None, :, k], expected_camera[:, :, None, k], acc)
                decomposition["rotation_fma_" + "".join(map(str, order))] = int(
                    torch.count_nonzero(acc != eager[0])
                )
        totals["points"] += T * H * W
        totals["direction_components_different"] += int(torch.count_nonzero(eager[0] != fused[0]))
        totals["range_values_different"] += int(torch.count_nonzero(eager[1] != fused[1]))
        totals["max_direction_absolute_error"] = max(
            totals["max_direction_absolute_error"], float((eager[0] - fused[0]).abs().max())
        )
        totals["max_range_absolute_error"] = max(
            totals["max_range_absolute_error"], float((eager[1] - fused[1]).abs().max())
        )
        # Huge focal lengths make the radial factor round to exactly one for
        # finite ordinary coordinates, so range encodes the chosen pixel ID.
        probe_K = torch.eye(3, device="cuda").repeat(T, 1, 1)
        probe_K[:, 0, 0] = probe_K[:, 1, 1] = 1e15
        identity = torch.eye(3, device="cuda").repeat(T, 1, 1)
        encoded = code[None].expand(T, -1, -1).contiguous()
        expected = torch_ray_builder(coords, encoded, probe_K, identity)[1]
        actual = build(coords, encoded, probe_K, identity)[1]
        mismatch = (expected != actual) & torch.isfinite(coords).all(-1)
        totals["nearest_mismatches"] += int(mismatch.sum())
        for t, p in torch.nonzero(mismatch)[: max(0, 12 - len(examples))].tolist():
            examples.append(
                dict(
                    frame=start + t,
                    reference_pixel=p,
                    coordinate=coords[t, p].tolist(),
                    expected_pixel=int(expected[t, p]),
                    actual_pixel=int(actual[t, p]),
                )
            )
    mapped = {}
    if args.mapped_backend:
        from relift.optimization import InferenceComponents

        components = InferenceComponents().with_native_build(str(args.build.resolve()))
        for cache in (0, 256):
            candidate = MappedObservations(
                args.window,
                "cuda",
                depth_cache_mb=cache,
                ray_backend=args.mapped_backend,
                components=components,
            )
            reference = MappedObservations(args.window, "cuda", depth_cache_mb=cache)
            differences = {key: 0 for key in ("origins", "directions", "ranges", "visibility")}
            for p in range(0, H * W, 8192):
                ids = np.arange(p, min(p + 8192, H * W))
                expected = reference.rays(ids)
                actual = candidate.rays(ids)
                for key, a, b in zip(differences, expected, actual, strict=False):
                    differences[key] += int(
                        torch.count_nonzero(a.view(torch.int32) != b.view(torch.int32))
                    )
            mapped[str(cache)] = dict(
                points=source.T * H * W,
                bitwise_differences=differences,
                passed=not any(differences.values()),
            )
            del candidate, reference
    record = dict(
        source=str(args.source.resolve()),
        build=json.loads((args.build / "build.json").read_text()),
        shape=[source.T, H, W],
        totals=totals,
        first_block_decomposition=decomposition,
        nearest_examples=examples,
        tracker_sha256=hashlib.sha256(source.track_source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        mapped_backend=args.mapped_backend,
        mapped=mapped,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(totals, indent=2))
    print(json.dumps(decomposition, indent=2))
    print(json.dumps(examples[:4], indent=2))
    if mapped:
        print(json.dumps(mapped, indent=2))
        if not all(row["passed"] for row in mapped.values()):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
