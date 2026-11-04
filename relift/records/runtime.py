"""Runtime counters, stage telemetry and storage archive records."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..optimization import InferenceComponents, OptimizationConfig

from .configuration import ConfigurationSelectionRecord, OptimizationRecord


class ObservationCacheMetrics(TypedDict):
    state: Literal["disabled", "budget_declined", "headroom_declined", "active", "closed"]
    budget_bytes: int
    required_bytes: int
    resident_bytes: int
    upload_bytes: int
    upload_calls: int
    gather_calls: int
    pixel_id_upload_bytes: int
    source_dtypes: list[str]
    device: str


class ShardCommitMetrics(TypedDict):
    backend: str
    shards_submitted: int
    shards_committed: int
    max_pending_shards: int
    peak_pending_payload_bytes: int
    pending_payload_bytes: int
    wait_seconds: float
    worker_write_seconds: float


class ShardPayload(TypedDict):
    xyz: NDArray[np.float32]
    lift: NDArray[np.float16]
    observed: NDArray[np.bool_]


class DA3StartupMetrics(TypedDict):
    strategy: str
    cold_parameter_bytes: int
    cold_buffer_bytes: int
    # Size-based transfer accounting, not a profiler or PCIe traffic measurement.
    initial_cold_roundtrip_tensor_bytes: int


class PhaseMetrics(TypedDict):
    seconds: float
    process_rss_highwater_bytes: int
    cuda_allocated_peak_bytes: NotRequired[int]
    cuda_reserved_peak_bytes: NotRequired[int]


class StageMetrics(TypedDict):
    status: Literal["running", "complete", "failed"]
    error: NotRequired[str]
    seconds: NotRequired[float]
    process_rss_highwater_bytes: NotRequired[int]
    cuda_allocated_peak_bytes: NotRequired[int]
    cuda_reserved_peak_bytes: NotRequired[int]
    cold_tensor_bytes_written: NotRequired[int]
    peak_retained_cpu_tensor_bytes: NotRequired[int]
    weight_host_to_device_bytes: NotRequired[int]


class MetricsDocument[S](TypedDict):
    settings: S
    stages: dict[str, StageMetrics]
    wall_seconds_since_metrics_init: NotRequired[float]


class FileReference(TypedDict):
    file: str
    sha256: str


class MappedArchiveManifest(TypedDict):
    source_sha256: str
    members: dict[str, FileReference]


class FrameManifest(TypedDict):
    format: str
    version: int
    source_sha256: str
    requested_frames: int
    shape: tuple[int, ...]
    dtype: str
    color_order: str
    sha256: str


type ShardArrays = dict[str, FileReference]


class ShardRecord(TypedDict):
    start: int
    stop: int
    arrays: ShardArrays


class TrajectoryManifest(TypedDict):
    format: str
    version: int
    state: Literal["writing", "complete"]
    shape: tuple[int, ...] | list[int]
    # User/injected provenance is intentionally extensible; its envelope is fixed.
    metadata: dict[str, Any]
    pixel_order_sha256: str
    repeated_pixel_ids: list[int]
    arrays: dict[str, FileReference]
    shards: list[ShardRecord]
    finite: NotRequired[bool]


class FeatureRecord(TypedDict):
    offset: int
    bytes: int
    shape: list[int]
    dtype: str
    sha256: str


class FeatureManifest(TypedDict):
    format: str
    version: int
    records: list[FeatureRecord]


class DiagnosticSpan(TypedDict):
    calls: int
    seconds: float


class ProfileFunction(TypedDict):
    file: str
    line: int
    function: str
    calls: int
    self_seconds: float
    inclusive_seconds: float


class DiagnosticReport(TypedDict):
    spans: dict[str, DiagnosticSpan]
    caveat: str
    functions: list[ProfileFunction]


class InferenceSettings(TypedDict):
    offload_requested: str
    process_resolution: int
    corrector_batch: int
    ram_budget_gb: float
    vram_budget_gb: float
    scratch_budget_gb: float
    optimizations: OptimizationRecord
    configuration: ConfigurationSelectionRecord
    components: dict[str, str]
    estimated_da3_vram_gb: NotRequired[float]
    estimated_da3_cold_gb: NotRequired[float]
    offload_resolved: NotRequired[str]


class StreamCorrectionOptions(TypedDict, total=False):
    batch: int
    device: str
    depth_file: str
    tracks_file: str
    ram_budget_gb: float
    vram_budget_gb: float
    scratch_budget_gb: float
    depth_cache_mb: float
    hash_writes: bool
    padding_batch: int | None
    gather_backend: str
    optimizations: OptimizationConfig
    components: InferenceComponents
