"""Process telemetry, experiment options and persisted benchmark result envelopes."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import NotRequired, Protocol, TypedDict

from .artifacts import NativeBuildManifest
from .configuration import (
    ConfigurationSelectionRecord,
    InferenceConfigurationRecord,
    OptimizationOverrides,
    OptimizationRecord,
)
from .runtime import (
    MetricsDocument,
    ObservationCacheMetrics,
    PhaseMetrics,
    ShardCommitMetrics,
    StageMetrics,
)


class GPUProcess(TypedDict):
    pid: int
    gpu_memory_bytes: int


class GPUSample(TypedDict, total=False):
    gpu_device_used_bytes: int
    gpu_device_total_bytes: int
    gpu_utilization_percent: int
    gpu_memory_utilization_percent: int
    gpu_temperature_celsius: int


class ProcessSnapshot(TypedDict):
    pid: int
    ppid: int
    started_ticks: int
    rss_bytes: int
    minor_faults: int
    major_faults: int
    cpu_seconds: float
    pss_bytes: NotRequired[int]
    anonymous_bytes: NotRequired[int]
    swap_bytes: NotRequired[int]
    pss_unavailable: NotRequired[bool]
    io_unavailable: NotRequired[bool]
    read_bytes: NotRequired[int]
    write_bytes: NotRequired[int]
    rchar: NotRequired[int]
    wchar: NotRequired[int]


class SwapCounters(TypedDict):
    pswpin_bytes: int
    pswpout_bytes: int


class TelemetrySample(GPUSample):
    elapsed_seconds: float
    process_count: int
    process_tree_rss_bytes: int
    process_tree_pss_bytes: int
    process_tree_anonymous_bytes: int
    process_tree_swap_bytes: int
    process_detail_unavailable: bool
    gpu_compute_processes: NotRequired[list[GPUProcess]]
    other_gpu_compute_processes: NotRequired[list[GPUProcess]]


class TelemetrySummary(TypedDict):
    sampled_peak_process_tree_rss_bytes: int | None
    sampled_peak_process_tree_pss_bytes: int | None
    sampled_peak_process_tree_anonymous_bytes: int | None
    sampled_peak_process_tree_swap_bytes: int | None
    sampled_peak_gpu_device_used_bytes: int | None
    process_tree_read_bytes: int
    process_tree_write_bytes: int
    process_tree_rchar: int
    process_tree_wchar: int
    process_tree_minor_faults: int
    process_tree_major_faults: int
    process_tree_cpu_seconds: float
    sample_count: int
    gpu_exclusivity_monitored: bool
    other_gpu_compute_processes: list[GPUProcess]
    gpu_sampling_error: str | None
    sampling_limits: str


class DiskUsage(TypedDict):
    logical_bytes: int
    allocated_bytes: int


class ExecutionMetrics(TypedDict, total=False):
    peak_pinned_bytes: int
    prefetched_host_to_device_bytes: int


class ProviderCounters(TypedDict):
    cold_bytes_written: int
    transfer_calls: int
    weight_transfer_bytes: int
    buffer_transfer_bytes: int
    scratch_files: int


class CompilationPass(TypedDict):
    phase: str
    output: str
    shape: tuple[int, ...]
    finite: bool


class ExperimentOptions(TypedDict, total=False):
    inference_profile: str
    inference_config: str
    configuration: InferenceConfigurationRecord
    optimizations: OptimizationOverrides | OptimizationRecord
    native_build: str
    profile: bool
    prefetch: bool
    batch: int
    padding_batch: int
    depth_cache_mb: float
    hash_writes: bool
    gather_backend: str
    ram_budget_gb: float
    vram_budget_gb: float
    scratch_budget_gb: float
    compile: bool
    passes: int
    device: str
    window: int
    iters: int
    init_weights: bool
    copy_outputs: bool
    reuse: bool
    clips: list[str]
    offload: str
    video: str


class ExperimentResult(TypedDict):
    task: str
    phases: Mapping[str, PhaseMetrics | StageMetrics]
    shape: NotRequired[tuple[int, ...]]
    execution: NotRequired[ExecutionMetrics]
    configuration: NotRequired[ConfigurationSelectionRecord | None]
    shard_commits: NotRequired[ShardCommitMetrics | None]
    observation_cache: NotRequired[ObservationCacheMetrics | None]
    finite: NotRequired[bool]
    source: NotRequired[str]
    started_at: NotRequired[str]
    scope: NotRequired[str]
    artifacts: NotRequired[dict[str, str]]
    state_sha256: NotRequired[str]
    shapes: NotRequired[dict[str, dict[str, list[int]]]]
    input_sha256: NotRequired[dict[str, str]]
    optimizations: NotRequired[OptimizationRecord]
    counters: NotRequired[list[ProviderCounters]]
    totals: NotRequired[dict[str, float]]
    reused: NotRequired[bool]
    runs: NotRequired[list[CompilationPass]]
    compiled: NotRequired[bool]
    compiler_counters: NotRequired[dict[str, dict[str, int]]]


class RuntimeEnvironment(TypedDict):
    python: str
    executable: str
    packages: dict[str, str]
    environment_lockfile_sha256: str


class ExperimentDefinition(TypedDict):
    task: str
    variants: list[ExperimentVariant]
    repeats: NotRequired[int]
    timeout_seconds: NotRequired[float]


class ExperimentVariant(TypedDict):
    name: str
    source: str
    options: NotRequired[ExperimentOptions]


class ExperimentSpec(TypedDict):
    scene: str
    input: str
    prepared_window: str
    process_res: int
    output_root: str
    experiments: dict[str, ExperimentDefinition]


class ExperimentMetrics(TypedDict):
    settings: ExperimentOptions
    stages: Mapping[str, PhaseMetrics | StageMetrics]


class TrialRecord(TypedDict):
    experiment: str
    mode: str
    source: str
    options: ExperimentOptions
    configuration: ConfigurationSelectionRecord | None
    task: str
    diagnostic: bool
    repeat: int
    status: str
    started_at: str
    wall_seconds: float
    command: list[str]
    output: str
    log: str
    returncode: int
    scenario: str
    scene: str
    input_sha256: str
    num_frames: int
    process_res: int
    corrector_batch: int
    seed: int
    code_hashes: dict[str, str]
    # Native build manifests carry compiler/platform fields supplied by the builder.
    native_build: NativeBuildManifest | None
    driver_hashes: dict[str, str]
    lockfile_sha256: str
    wrapper_sha256: str
    environment: RuntimeEnvironment
    telemetry: TelemetrySummary
    telemetry_interval_seconds: float
    sampled_output_peak: DiskUsage
    final_output_size: DiskUsage
    system_swap_deltas: dict[str, int]
    result: ExperimentResult | None
    metrics: ExperimentMetrics | None


class SummaryRow(TypedDict):
    experiment: str
    variant: str
    n: int
    attempts: int
    measurement_role: str
    statuses: dict[str, int]
    diagnostic: bool
    trial_records: list[str]
    options: ExperimentOptions
    source_fingerprint: str
    environment: RuntimeEnvironment | None
    quality: NotRequired[str | None]
    wall_median: NotRequired[float | None]
    wall_min: NotRequired[float | None]
    wall_max: NotRequired[float | None]
    sampled_peak_process_tree_rss_bytes: NotRequired[float | None]
    sampled_peak_process_tree_pss_bytes: NotRequired[float | None]
    sampled_peak_gpu_device_used_bytes: NotRequired[float | None]
    process_tree_read_bytes: NotRequired[float | None]
    process_tree_write_bytes: NotRequired[float | None]
    phase_medians: NotRequired[dict[str, float]]
    process_rss_highwater_bytes: NotRequired[float | None]
    sampled_output_peak_allocated_bytes: NotRequired[float | None]


class PairedMeasurement(TypedDict):
    experiment: str
    repeat: int
    control_seconds: float
    candidate_seconds: float
    candidate_over_control: float
    seconds_saved: float
    control_phases: Mapping[str, PhaseMetrics | StageMetrics]
    candidate_phases: Mapping[str, PhaseMetrics | StageMetrics]


class PhaseRecorder(Protocol):
    @property
    def results(self) -> Mapping[str, PhaseMetrics]: ...
    def stage(self, name: str) -> AbstractContextManager[None]: ...


class TrackerOptions(TypedDict, total=False):
    device: str
    window: int
    iters: int
    init_weights: bool
    copy_outputs: bool


class OffloadTrialRecord(TypedDict):
    scene: str
    mode: str
    repeat: int
    status: str
    returncode: int
    started_at: str
    wall_seconds: float
    command: list[str]
    output: str
    input: str
    input_sha256: str
    process_res: int
    corrector_batch: int
    num_frames: int
    seed: int
    wrapper_sha256: str
    scenario: str
    fixture_copy_seconds_outside_measurement: float
    idle_gpu: GPUSample
    telemetry_interval_seconds: float
    telemetry: TelemetrySummary
    log: str
    baseline_snapshot: str
    system_swap_deltas: NotRequired[SwapCounters]
    metrics: NotRequired[MetricsDocument[dict[str, object]]]
    offload: NotRequired[dict[str, object]]
    code_hashes: NotRequired[dict[str, str]]
    lockfile_sha256: NotRequired[str]
    reused_trial_record: NotRequired[bool]


class OffloadSummary(TypedDict):
    scene: str
    mode: str
    scenario: str
    trials: int
    completed: int
    failures: str
    median_wall_seconds: NotRequired[float | None]
    min_wall_seconds: NotRequired[float | None]
    max_wall_seconds: NotRequired[float | None]
    median_sampled_peak_gpu_device_mib: NotRequired[float | None]
    median_sampled_peak_tree_rss_mib: NotRequired[float | None]
    median_sampled_peak_tree_pss_mib: NotRequired[float | None]
    median_physical_read_mib: NotRequired[float | None]
    median_physical_write_mib: NotRequired[float | None]
    existing_over_candidate_speed_ratio: NotRequired[float | None]
