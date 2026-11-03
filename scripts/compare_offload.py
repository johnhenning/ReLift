#!/usr/bin/env python
"""Compare frozen NPZ and mapped offload trajectories by original pixel identity.

    uv run python scripts/compare_offload.py BASELINE_DIR CANDIDATE_DIR --out parity.json

All pixels and frames are examined in bounded point groups. Repeated padded
points are reported separately because the old CUDA scatter has undefined write
order for those duplicates. Reported maximum errors are exact; median/p99 error
upper bounds use a fixed logarithmic histogram rather than retaining all errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from relift.records.comparison import (
    ArrayComparison,
    ChunkedArrayComparison,
    ComparisonReport,
    ErrorGroups,
    ErrorReport,
    PixelGrouping,
    PixelPolicy,
    PreparedComparison,
    SharedArrays,
    TrajectoryComparison,
)
from relift.storage import TrajectoryStore, file_sha256, mapped_archive


class ErrorStats:
    def __init__(self, scale: float) -> None:
        self.scale = scale
        self.edges = np.concatenate(([0.0], np.geomspace(1e-12, 1e3, 256), [np.inf]))
        self.histogram = np.zeros(len(self.edges) - 1, np.int64)
        self.count = self.nonfinite = self.zeros = 0
        self.maximum = self.total = 0.0
        self.bitwise_equal = True

    def add(self, reference: ArrayLike, candidate: ArrayLike) -> None:
        reference, candidate = np.asarray(reference), np.asarray(candidate)
        self.bitwise_equal &= reference.dtype == candidate.dtype and np.array_equal(
            np.ascontiguousarray(reference).view(np.uint8),
            np.ascontiguousarray(candidate).view(np.uint8),
        )
        error = (
            np.linalg.norm(candidate.astype(np.float64) - reference.astype(np.float64), axis=-1)
            / self.scale
        )
        finite = np.isfinite(error)
        self.nonfinite += int((~finite).sum())
        error = error[finite]
        self.count += error.size
        self.zeros += int((error == 0).sum())
        if error.size:
            self.maximum = max(self.maximum, float(error.max()))
            self.total += float(error.sum())
            self.histogram += np.histogram(error, self.edges)[0]

    def percentile_bound(self, fraction: float) -> float | None:
        if not self.count:
            return None
        if self.zeros >= np.ceil(self.count * fraction):
            return 0.0
        index = int(np.searchsorted(np.cumsum(self.histogram), np.ceil(self.count * fraction)))
        result = self.edges[min(index + 1, len(self.edges) - 1)]
        return float(result) if np.isfinite(result) else None

    def report(self) -> ErrorReport:
        return dict(
            vector_count=self.count,
            nonfinite_vectors=self.nonfinite,
            bitwise_equal=self.bitwise_equal,
            max_normalized_vector_error=self.maximum if not self.nonfinite else None,
            mean_normalized_vector_error=self.total / self.count if self.count else None,
            median_normalized_error_upper_bound=self.percentile_bound(0.5),
            p99_normalized_error_upper_bound=self.percentile_bound(0.99),
            quantile_method="fixed logarithmic histogram upper bounds; maximum is exact",
        )


def small_array_comparison(reference: ArrayLike, candidate: ArrayLike) -> ArrayComparison:
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    result: ArrayComparison = dict(
        reference_shape=list(reference.shape),
        candidate_shape=list(candidate.shape),
        reference_dtype=str(reference.dtype),
        candidate_dtype=str(candidate.dtype),
        equal=False,
        max_absolute_error=None,
    )
    if reference.shape != candidate.shape:
        result.update(equal=False, max_absolute_error=None)
        return result
    result["equal"] = reference.dtype == candidate.dtype and np.array_equal(
        np.ascontiguousarray(reference).view(np.uint8),
        np.ascontiguousarray(candidate).view(np.uint8),
    )
    diff = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    result["max_absolute_error"] = float(diff.max(initial=0)) if np.isfinite(diff).all() else None
    return result


def compare_prepared(reference: Path, candidate: Path) -> dict[str, PreparedComparison]:
    reports: dict[str, PreparedComparison] = {}
    for filename, fields in (
        ("da3_depth_data.npz", ("depth", "poses", "intrinsics")),
        ("alltracker_tracks.npz", ("tracks_2d", "visibility", "confidence")),
    ):
        left, right = reference / "window" / filename, candidate / "window" / filename
        if not left.exists() or not right.exists():
            reports[filename] = dict(
                compared=False, reason="prepared fixture is not available on both sides"
            )
            continue
        a, b = mapped_archive(left), mapped_archive(right)
        result: dict[str, ArrayComparison | ChunkedArrayComparison] = {}
        for field in fields:
            if a[field].shape != b[field].shape:
                result[field] = small_array_comparison(a[field], b[field])
                continue
            equal = True
            maximum: float | None = 0.0
            for start in range(0, len(a[field]), 8):
                row = small_array_comparison(
                    a[field][start : start + 8], b[field][start : start + 8]
                )
                equal &= row["equal"]
                if row["max_absolute_error"] is None:
                    maximum = None
                elif maximum is not None:
                    maximum = max(maximum, row["max_absolute_error"])
            result[field] = dict(
                equal=equal,
                max_absolute_error=maximum,
                shape=list(a[field].shape),
                dtype=str(a[field].dtype),
            )
        reports[filename] = dict(compared=True, arrays=result)
    return reports


def compare(
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    chunk_points: int = 2048,
    tolerance: float = 1e-4,
    prepared: bool = True,
    pixel_policy_from: str | Path | None = None,
) -> ComparisonReport:
    reference, candidate = (
        TrajectoryStore.open(reference_path),
        TrajectoryStore.open(candidate_path),
    )
    if reference.shape != candidate.shape:
        return dict(
            passed=False,
            reason="shape mismatch",
            reference_shape=reference.shape,
            candidate_shape=candidate.shape,
        )
    total, height, width, _ = reference.shape
    count = height * width
    policy = TrajectoryStore.open(pixel_policy_from) if pixel_policy_from is not None else candidate
    if policy.shape != reference.shape:
        raise ValueError("Pixel policy source must have the same frame count and processing grid")
    repeated = np.asarray(policy.metadata.get("repeated_pixel_ids", []), np.int64)
    if np.any(repeated < 0) or np.any(repeated >= count):
        raise ValueError("Invalid repeated pixel IDs in candidate metadata")
    repeated = np.unique(repeated)
    repeated_mask = np.zeros(count, bool)
    repeated_mask[repeated] = True
    # Prefer the exact median used by streamed correction, stored in its manifest.
    scale = policy.metadata.get("scale")
    scale_source = (
        "explicit pixel-policy store normalization scale"
        if pixel_policy_from is not None
        else "candidate stored baseline normalization scale"
    )
    if scale is None:
        first = reference.read_frame("xyz", 0).reshape(-1, 3)
        positive = first[:, 2][np.isfinite(first[:, 2]) & (first[:, 2] > 0)]
        scale = float(np.median(positive)) if positive.size else 1.0
        scale_source = "fallback positive reference-frame corrected depth median"
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Normalization scale must be positive and finite")
    stats = {name: ErrorStats(scale) for name in ("ordinary_xyz", "repeated_xyz", "raw_lift")}
    changed_observations = 0
    changed_dtypes = set()
    for start in range(0, count, chunk_points):
        ids = np.arange(start, min(start + chunk_points, count))
        a = reference.sample(ids)
        b = candidate.sample(ids)
        for field in a:
            if a[field].dtype != b[field].dtype:
                changed_dtypes.add(field)
        ordinary = ~repeated_mask[ids]
        if ordinary.any():
            stats["ordinary_xyz"].add(a["xyz"][:, ordinary], b["xyz"][:, ordinary])
        if (~ordinary).any():
            stats["repeated_xyz"].add(a["xyz"][:, ~ordinary], b["xyz"][:, ~ordinary])
        stats["raw_lift"].add(a["lift"], b["lift"])
        changed_observations += int((a["observed"] != b["observed"]).sum())
    poses = small_array_comparison(reference.array("poses"), candidate.array("poses"))
    intrinsics = small_array_comparison(
        reference.array("intrinsics"), candidate.array("intrinsics")
    )
    images_a, images_b = reference.array("images"), candidate.array("images")
    image_equal = images_a.shape == images_b.shape and images_a.dtype == images_b.dtype
    if image_equal:
        for start in range(0, total, 8):
            image_equal &= np.array_equal(images_a[start : start + 8], images_b[start : start + 8])
    arrays: SharedArrays = dict(
        poses=poses,
        intrinsics=intrinsics,
        images=dict(
            equal=bool(image_equal),
            reference_shape=list(images_a.shape),
            candidate_shape=list(images_b.shape),
        ),
    )
    outputs: ErrorGroups = dict(
        ordinary_xyz=stats["ordinary_xyz"].report(),
        repeated_xyz=stats["repeated_xyz"].report(),
        raw_lift=stats["raw_lift"].report(),
    )
    geometry_ok = all(
        row["reference_dtype"] == row["candidate_dtype"]
        and row["max_absolute_error"] is not None
        and row["max_absolute_error"] <= tolerance
        for row in (poses, intrinsics)
    )
    ordinary = outputs["ordinary_xyz"]
    raw = outputs["raw_lift"]
    passed = (
        ordinary["nonfinite_vectors"] == 0
        and ordinary["max_normalized_vector_error"] is not None
        and ordinary["max_normalized_vector_error"] <= tolerance
        and raw["nonfinite_vectors"] == 0
        and raw["max_normalized_vector_error"] is not None
        and raw["max_normalized_vector_error"] <= tolerance
        and changed_observations == 0
        and not changed_dtypes
        and image_equal
        and geometry_ok
    )
    grouping: PixelGrouping = {}
    if "seed" in policy.metadata:
        grouping["seed"] = int(policy.metadata["seed"])
    if "group" in policy.metadata:
        grouping["group"] = int(policy.metadata["group"])
    if "batch" in policy.metadata:
        grouping["batch"] = int(policy.metadata["batch"])
    manifest = policy.root / "manifest.json"
    pixel_policy: PixelPolicy = dict(
        source=str(pixel_policy_from) if pixel_policy_from is not None else str(candidate_path),
        explicit=pixel_policy_from is not None,
        shape=list(policy.shape),
        manifest_sha256=file_sha256(manifest) if policy.manifest is not None else None,
        grouping=grouping,
    )
    report: TrajectoryComparison = dict(
        passed=passed,
        reference=str(reference_path),
        candidate=str(candidate_path),
        shape=list(reference.shape),
        frames=total,
        pixels=count,
        compared_cells=total * count,
        repeated_pixel_count=len(repeated),
        repeated_pixel_ids=repeated.tolist(),
        normalization_scale=scale,
        normalization_source=scale_source,
        tolerance=tolerance,
        errors=outputs,
        changed_observation_cells=changed_observations,
        shared_arrays=arrays,
        changed_dtypes=sorted(changed_dtypes),
        pixel_policy=pixel_policy,
        policy="All frames and original pixel IDs compared; padded duplicate coordinates are reported separately, not included in ordinary XYZ parity gate. No fitted alignment.",
    )
    if prepared:
        report["prepared_outputs"] = compare_prepared(Path(reference_path), Path(candidate_path))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("reference", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--chunk-points", type=int, default=2048)
    ap.add_argument("--tolerance", type=float, default=1e-4)
    ap.add_argument("--skip-prepared", action="store_true")
    ap.add_argument(
        "--pixel-policy-from",
        type=Path,
        help="use this matching store's repeated pixel IDs and normalization scale, e.g. for two legacy baseline repeats",
    )
    args = ap.parse_args()
    if args.chunk_points < 1 or args.tolerance < 0:
        ap.error("chunk size must be positive and tolerance nonnegative")
    report = compare(
        args.reference,
        args.candidate,
        chunk_points=args.chunk_points,
        tolerance=args.tolerance,
        prepared=not args.skip_prepared,
        pixel_policy_from=args.pixel_policy_from,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
