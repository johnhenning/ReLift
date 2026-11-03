"""Serializable views of the validated component dataclasses."""

from typing import NotRequired, TypedDict


class OptimizationRecord(TypedDict):
    gather: str
    shards: str
    shard_writer: str | None
    commit_backend: str
    rays: str
    da3_weights: str
    da3_features: str
    tracker_init_weights: bool
    tracker_copy_outputs: bool
    hash_writes: bool
    depth_cache_mb: float
    observation_cache_mb: float
    prefetch: bool
    corrector_batch: int
    padding_batch: int | None
    compile_corrector: bool
    reuse_da3: bool


class MemoryRecord(TypedDict):
    offload: str
    ram_gb: float
    vram_gb: float
    scratch_gb: float
    scratch_dir: str | None


class ObservationsRecord(TypedDict):
    gather: str
    cache_mb: float
    depth_cache_mb: float
    prefetch: bool


class RaysRecord(TypedDict):
    backend: str


class CorrectionRecord(TypedDict):
    batch: int
    padding_batch: int | None
    compile: bool


class StorageRecord(TypedDict):
    packing: str
    writer: str | None
    hash_writes: bool
    commits: str


class DA3Record(TypedDict):
    weights: str
    features: str
    reuse: bool


class TrackerRecord(TypedDict):
    init_weights: bool
    copy_outputs: bool


class RuntimeRecord(TypedDict):
    native_build: str | None


class OptimizationOverrides(TypedDict, total=False):
    gather: str
    shards: str
    shard_writer: str | None
    commit_backend: str
    rays: str
    da3_weights: str
    da3_features: str
    tracker_init_weights: bool
    tracker_copy_outputs: bool
    hash_writes: bool
    depth_cache_mb: float
    observation_cache_mb: float
    prefetch: bool
    corrector_batch: int
    padding_batch: int | None
    compile_corrector: bool
    reuse_da3: bool


class InferenceConfigurationRecord(TypedDict):
    memory: MemoryRecord
    observations: ObservationsRecord
    rays: RaysRecord
    correction: CorrectionRecord
    storage: StorageRecord
    da3: DA3Record
    tracker: TrackerRecord
    runtime: RuntimeRecord
    schema_version: int


class ConfigurationSourceRecord(TypedDict):
    path: str
    sha256: str


class ConfigurationSelectionRecord(TypedDict):
    resolved: InferenceConfigurationRecord
    sources: list[ConfigurationSourceRecord]


class WeightsEntry(TypedDict):
    kind: str
    url: NotRequired[str]
    file: NotRequired[str]
    dir: NotRequired[str]
    repo: NotRequired[str]
    revision: NotRequired[str]
    md5: NotRequired[str]
    md5_model_safetensors: NotRequired[str]
    licence: NotRequired[str]
