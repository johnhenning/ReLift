"""Standalone viewer payloads, source maps, native build and profiler records."""

from typing import NotRequired, TypedDict

from .configuration import ConfigurationSelectionRecord
from .runtime import ObservationCacheMetrics, ProfileFunction


class DA3StartupProfile(TypedDict):
    source: str
    source_manifest_sha256: str
    configuration: ConfigurationSelectionRecord
    startup_seconds: float
    cuda_allocated_peak_bytes: int
    cuda_reserved_peak_bytes: int
    functions: list[ProfileFunction]
    caveat: str


class DA3StartupProfileExcerpt(DA3StartupProfile):
    function_count_full: int
    functions_limit: int


class ViewerPayload(TypedDict):
    frames: int
    points: int
    dense_points: int
    fps: float
    finite: bool
    requested_frames: int | None
    grid: list[int]
    process_resolution: int
    corrector_batch: int | None
    raw: str
    corrected: str
    observed: str
    pixels: list[list[int]]
    dynamic: list[int]
    bounds: list[list[float]]


class ViewerMetadata(TypedDict):
    generator: str
    plotly_version: str
    source: str
    source_sha256: str
    frames: int
    storage: str
    input_sha256: str | None
    source_fps: float
    requested_frames: int | None
    dense_trajectories: int
    displayed_trajectories: int
    finite: bool
    duration_seconds: float
    grid: list[int]
    process_resolution: int
    corrector_batch: int | None
    complete_paths_shown: int
    temporal_sampling: str
    display_axes: list[str]
    spatial_selection: str
    axis_quantiles: list[float]
    embedded_plotlyjs: bool
    html_bytes: int


class RenderSummary(TypedDict):
    frames: int
    grid: list[int]
    trajectories: int
    observed_percent: float
    finite: bool
    checkpoint_step: int | None


class SourceExcerpt(TypedDict):
    file: str
    symbol: str
    start: int
    stop: int
    code: str
    sha256: str


class ComponentNode(TypedDict):
    id: str
    title: str
    purpose: str
    inputs: str
    outputs: str
    source: SourceExcerpt


class ComponentGraph(TypedDict):
    nodes: list[ComponentNode]
    edges: list[list[str]]


class NativeBuildManifest(TypedDict):
    command: list[str]
    compiler: str
    python: str
    torch: str
    platform: str
    openssl: str
    sources: dict[str, str]
    binary: str
    binary_sha256: str
    build_seconds: float
    cuda_command: NotRequired[list[str]]
    nvcc: NotRequired[str]
    arch: NotRequired[str]


class SourceSnapshot(TypedDict):
    base_commit: str
    git_revision: NotRequired[str]
    files: dict[str, str]
    python: str
    checkpoints: str


class FileHashes(TypedDict):
    md5: str
    sha256: str
    bytes: int
    mtime_ns: int
    expected_md5: NotRequired[str]
    matches_pinned_checkpoint: NotRequired[bool]
    file: NotRequired[str]


class TimingSample(TypedDict):
    wall_ms: float
    stream_elapsed_ms: float


class OperatorProfile(TypedDict):
    name: str
    calls: int
    self_cpu_us: float
    self_device_us: float
    device_us: float


class TimingDistribution(TypedDict):
    samples: list[TimingSample]
    median_wall_ms: float
    median_stream_elapsed_ms: float
    profile: NotRequired[list[OperatorProfile]]


class RayProfileReport(TypedDict):
    created_at: str
    arguments: dict[str, str | int | float | bool | None]
    torch: str
    gpu: str
    source_hashes: dict[str, str]
    script_sha256: str
    native_build: NativeBuildManifest
    caveat: str
    results: dict[str, TimingDistribution]


class RayAuditTotals(TypedDict):
    points: int
    nearest_mismatches: int
    direction_components_different: int
    max_direction_absolute_error: float
    max_range_absolute_error: float
    range_values_different: int


class NearestMismatch(TypedDict):
    frame: int
    reference_pixel: int
    coordinate: list[float]
    expected_pixel: int
    actual_pixel: int


class MappedRayAudit(TypedDict):
    points: int
    bitwise_differences: dict[str, int]
    passed: bool


class RayAuditReport(TypedDict):
    source: str
    build: NativeBuildManifest
    shape: list[int]
    totals: RayAuditTotals
    first_block_decomposition: dict[str, int]
    nearest_examples: list[NearestMismatch]
    tracker_sha256: str
    script_sha256: str
    mapped_backend: str | None
    mapped: dict[str, MappedRayAudit]


class StartupRecord(TypedDict):
    backend: str
    provision_seconds: float
    factory_seconds: float
    first_call_seconds: float
    subsequent_call_seconds: list[float]
    torch: str
    gpu: str
    created_at: str
    source: str
    ray_source_sha256: str
    script_sha256: str
    native_build: NativeBuildManifest
    command: NotRequired[list[str]]


class ObservationOptions(TypedDict, total=False):
    observation_cache_mb: float
    depth_cache_mb: float
    ray_backend: str


class ObservationConstruction(TypedDict):
    seconds: float
    cache: ObservationCacheMetrics


class ObservationProfileReport(TypedDict):
    created_at: str
    shape: list[int]
    points: int
    results: dict[str, TimingDistribution]
    construction: dict[str, ObservationConstruction]
    bitwise_differences: dict[str, dict[str, int]]
    input_hashes: dict[str, str]
    passed: bool
    final_cache_metrics: dict[str, ObservationCacheMetrics]
    source_hashes: dict[str, str]
    command: list[str]
    script_sha256: str
    torch: str
    gpu: str
    native_build: NativeBuildManifest | None
    caveat: str
