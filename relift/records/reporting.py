"""Gallery state and normalized report-generator records."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NotRequired, ReadOnly, TypedDict

from .artifacts import ViewerMetadata
from .runtime import PhaseMetrics, StageMetrics


class CatalogEntry(TypedDict):
    source_url: str
    description: str
    local_file: NotRequired[str]
    sha256: NotRequired[str]


class VideoSource(TypedDict, total=False):
    frames: int
    fps: float
    name: NotRequired[str]
    source_url: NotRequired[str]
    sha256: NotRequired[str]
    width: NotRequired[int]
    height: NotRequired[int]
    duration_seconds: NotRequired[float | None]


class InferenceAttempt(TypedDict):
    resolution: int
    corrector_batch: int
    log: str
    started_at: str
    status: NotRequired[str]
    offload: NotRequired[str]
    finished_at: NotRequired[str]
    log_sha256: NotRequired[str]


class GalleryExport(TypedDict, total=False):
    """Read-only display fields also present in older partial gallery exports."""

    dense_trajectories: ReadOnly[int]
    grid: ReadOnly[list[int]]


class GalleryVideo(TypedDict, total=False):
    status: str
    stage: str
    source: VideoSource
    result: GalleryExport
    error: str
    started_at: str
    finished_at: str
    reused: bool
    processing_resolution: int
    inference_attempts: list[InferenceAttempt]


class GalleryState(TypedDict):
    started_at: NotRequired[str]
    videos: dict[str, GalleryVideo]
    updated_at: NotRequired[str]


class DenseInventory(TypedDict):
    dense_bytes: int | None
    dense_allocated_bytes: int | None
    shape: list[int] | None
    files: NotRequired[int]


class TechnicalRecord(TypedDict):
    path: Path
    root: Path
    # This report reads multiple historical schemas and preserves unknown evidence.
    data: dict[str, Any]


class AggregateGroup(TypedDict):
    id: NotRequired[str]
    members: list[TechnicalRecord]
    valid: list[TechnicalRecord]
    values: dict[str, float | None]
    counts: dict[str, int]


class FailureDetail(TypedDict):
    type: str
    detail: str


DemoInventoryRow = TypedDict(
    "DemoInventoryRow",
    {
        "name": str,
        "status": str,
        "source": VideoSource,
        "export": ViewerMetadata,
        "log": NotRequired[str],
        "log_sha256": NotRequired[str],
        "logged_timings": NotRequired[dict[str, float]],
        "sum_of_logged_stage_seconds": NotRequired[float],
        "failure": NotRequired[FailureDetail],
        "tracks.npz_bytes": NotRequired[int],
        "trajectories.html_bytes": NotRequired[int],
    },
)


class SourceVersion(TypedDict):
    code_hashes: dict[str, str] | None
    lockfile_sha256: str | None
    wrapper_sha256: str | None


class Workload(TypedDict):
    scene: str | None
    scenario: str | None
    input_sha256: str | None
    process_res: int | None
    corrector_batch: int | None
    num_frames: int | None
    seed: int | None
    command_settings: NotRequired[dict[str, list[str]] | None]


class WorkloadSignature(TypedDict):
    workload: Workload
    lockfile: str | None
    runtime: dict[str, object] | None
    wrapper: str | None
    unmatched_record: NotRequired[str]


class Distribution(TypedDict):
    values: list[float]
    median: float | None
    min: float | None
    max: float | None
    n: int


class EmptyRecord(TypedDict):
    pass


class BenchmarkRecordReference(TypedDict):
    path: str
    sha256: str
    inventory: DenseInventory | EmptyRecord | None
    stages: dict[str, StageMetrics | PhaseMetrics] | None
    telemetry_interval_seconds: float | None


class BenchmarkChartRow(TypedDict):
    mode: str
    root: str
    source_version: SourceVersion
    n: int
    attempts: int
    statuses: dict[str, int]
    resolved_modes: list[str]
    metrics: dict[str, Distribution]
    records: list[BenchmarkRecordReference]
    label: NotRequired[str]


class WorkloadComparison(TypedDict):
    id: str
    settings: WorkloadSignature
    rows: list[BenchmarkChartRow]


class ArrayMemoryEstimate(TypedDict):
    frames: int
    grid: list[int]
    xyz_float32_mib: float
    raw_float16_mib: float
    xyz_raw_mask_images_mib: float


class DemoPerformanceReport(TypedDict):
    kind: str
    captured_at: str
    python: str
    versions: dict[str, str]
    base_commit: str
    working_tree_hashes: dict[str, str]
    videos: list[DemoInventoryRow]
    theoretical_array_memory: list[ArrayMemoryEstimate]
    timing_limits: str
