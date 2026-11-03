"""Typed component configuration and reproducible TOML profile composition."""

from __future__ import annotations

import hashlib
import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, get_args, get_type_hints

from .optimization import OptimizationConfig
from .records.configuration import (
    ConfigurationSelectionRecord,
    ConfigurationSourceRecord,
    CorrectionRecord,
    DA3Record,
    InferenceConfigurationRecord,
    MemoryRecord,
    ObservationsRecord,
    RaysRecord,
    RuntimeRecord,
    StorageRecord,
    TrackerRecord,
)

PROFILES = Path(__file__).resolve().parents[1] / "configs/inference"


@dataclass(frozen=True)
class Section:
    def __post_init__(self) -> None:
        for name, hint in get_type_hints(type(self)).items():
            value = getattr(self, name)
            choices = get_args(hint) or (hint,)
            if type(value) not in choices and not (float in choices and type(value) is int):
                raise TypeError(
                    f"{type(self).__name__}.{name} has invalid type {type(value).__name__}"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class Memory(Section):
    offload: str = "none"
    ram_gb: float = 18
    vram_gb: float = 20
    scratch_gb: float = 64
    scratch_dir: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.offload not in ("none", "streamed", "ram", "disk", "auto"):
            raise ValueError("Invalid memory.offload")
        if min(self.ram_gb, self.vram_gb, self.scratch_gb) <= 0:
            raise ValueError("Memory budgets must be positive")


@dataclass(frozen=True)
class Observations(Section):
    gather: str = "current"
    cache_mb: float = 0
    depth_cache_mb: float = 0
    prefetch: bool = False


@dataclass(frozen=True)
class Rays(Section):
    backend: str = "current"


@dataclass(frozen=True)
class Correction(Section):
    batch: int = 8
    padding_batch: int | None = None
    compile: bool = False


@dataclass(frozen=True)
class Storage(Section):
    packing: str = "current"
    writer: str | None = None
    hash_writes: bool = True
    commits: str = "sync"


@dataclass(frozen=True)
class DA3(Section):
    weights: str = "current"
    features: str = "current"
    reuse: bool = False
    startup: str = "current"


@dataclass(frozen=True)
class Tracker(Section):
    init_weights: bool = False
    copy_outputs: bool = False


@dataclass(frozen=True)
class Runtime(Section):
    native_build: str | None = None


@dataclass(frozen=True)
class InferenceConfig:
    memory: Memory = field(default_factory=Memory)
    observations: Observations = field(default_factory=Observations)
    rays: Rays = field(default_factory=Rays)
    correction: Correction = field(default_factory=Correction)
    storage: Storage = field(default_factory=Storage)
    da3: DA3 = field(default_factory=DA3)
    tracker: Tracker = field(default_factory=Tracker)
    runtime: Runtime = field(default_factory=Runtime)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported inference configuration schema_version")
        optimizations = self.optimizations()
        if self.memory.offload != "auto":
            optimizations.validate_mode(self.memory.offload)

    def optimizations(self) -> OptimizationConfig:
        """Adapt the grouped public configuration to the existing execution contract."""
        return OptimizationConfig(
            gather=self.observations.gather,
            observation_cache_mb=self.observations.cache_mb,
            depth_cache_mb=self.observations.depth_cache_mb,
            prefetch=self.observations.prefetch,
            rays=self.rays.backend,
            corrector_batch=self.correction.batch,
            padding_batch=self.correction.padding_batch,
            compile_corrector=self.correction.compile,
            shards=self.storage.packing,
            shard_writer=self.storage.writer,
            hash_writes=self.storage.hash_writes,
            commit_backend=self.storage.commits,
            da3_weights=self.da3.weights,
            da3_features=self.da3.features,
            da3_startup=self.da3.startup,
            reuse_da3=self.da3.reuse,
            tracker_init_weights=self.tracker.init_weights,
            tracker_copy_outputs=self.tracker.copy_outputs,
        )

    def as_dict(self) -> InferenceConfigurationRecord:
        return InferenceConfigurationRecord(
            memory=MemoryRecord(
                offload=self.memory.offload,
                ram_gb=self.memory.ram_gb,
                vram_gb=self.memory.vram_gb,
                scratch_gb=self.memory.scratch_gb,
                scratch_dir=self.memory.scratch_dir,
            ),
            observations=ObservationsRecord(
                gather=self.observations.gather,
                cache_mb=self.observations.cache_mb,
                depth_cache_mb=self.observations.depth_cache_mb,
                prefetch=self.observations.prefetch,
            ),
            rays=RaysRecord(backend=self.rays.backend),
            correction=CorrectionRecord(
                batch=self.correction.batch,
                padding_batch=self.correction.padding_batch,
                compile=self.correction.compile,
            ),
            storage=StorageRecord(
                packing=self.storage.packing,
                writer=self.storage.writer,
                hash_writes=self.storage.hash_writes,
                commits=self.storage.commits,
            ),
            da3=DA3Record(
                weights=self.da3.weights,
                features=self.da3.features,
                reuse=self.da3.reuse,
                startup=self.da3.startup,
            ),
            tracker=TrackerRecord(
                init_weights=self.tracker.init_weights, copy_outputs=self.tracker.copy_outputs
            ),
            runtime=RuntimeRecord(native_build=self.runtime.native_build),
            schema_version=self.schema_version,
        )

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> InferenceConfig:
        if not isinstance(values, dict):
            raise TypeError("Inference configuration must be a table")
        defaults = cls()
        unknown = values.keys() - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
        sections: dict[str, Any] = {}
        for name, settings in values.items():
            if name == "schema_version":
                sections[name] = settings
                continue
            if not isinstance(settings, dict):
                raise TypeError(f"{name} must be a table")
            section = getattr(defaults, name)
            unknown = settings.keys() - {f.name for f in fields(section)}
            if unknown:
                raise ValueError(f"Unknown {name} fields: {sorted(unknown)}")
            sections[name] = replace(section, **settings)
        return cls(**sections)

    @classmethod
    def from_optimizations(
        cls,
        settings: OptimizationConfig,
        memory: Memory | None = None,
        native_build: str | None = None,
    ) -> InferenceConfig:
        return cls(
            memory=memory or Memory(offload="auto"),
            observations=Observations(
                settings.gather,
                settings.observation_cache_mb,
                settings.depth_cache_mb,
                settings.prefetch,
            ),
            rays=Rays(settings.rays),
            correction=Correction(
                settings.corrector_batch, settings.padding_batch, settings.compile_corrector
            ),
            storage=Storage(
                settings.shards,
                settings.shard_writer,
                settings.hash_writes,
                settings.commit_backend,
            ),
            da3=DA3(
                settings.da3_weights,
                settings.da3_features,
                settings.reuse_da3,
                settings.da3_startup,
            ),
            tracker=Tracker(settings.tracker_init_weights, settings.tracker_copy_outputs),
            runtime=Runtime(native_build),
        )


@dataclass(frozen=True)
class ConfigurationSource:
    path: str
    sha256: str


@dataclass(frozen=True)
class ConfigurationSelection:
    config: InferenceConfig
    sources: tuple[ConfigurationSource, ...] = ()

    def as_dict(self) -> ConfigurationSelectionRecord:
        return ConfigurationSelectionRecord(
            resolved=self.config.as_dict(),
            sources=[ConfigurationSourceRecord(path=s.path, sha256=s.sha256) for s in self.sources],
        )


def merge(base: Mapping[str, object], overlay: Mapping[str, object]) -> dict[str, object]:
    """Merge component tables without mutating either input."""
    result = dict(base)
    for key, value in overlay.items():
        prior = result.get(key)
        result[key] = (
            merge(prior, value) if isinstance(prior, dict) and isinstance(value, dict) else value
        )
    return result


def load_configuration(
    *,
    profile: str | None = None,
    path: str | Path | None = None,
    values: Mapping[str, object] | None = None,
    profiles_dir: Path = PROFILES,
) -> ConfigurationSelection:
    """Resolve one profile/file or inline table; inheritance follows declaring files."""
    if sum(item is not None for item in (profile, path, values)) > 1:
        raise ValueError("Choose one profile, inference config file or inline configuration")
    if values is not None and not isinstance(values, dict):
        raise TypeError("Inline configuration must be a table")
    sources: list[ConfigurationSource] = []

    def read(filename: Path, ancestors: tuple[Path, ...] = ()) -> dict[str, Any]:
        filename = filename.resolve()
        if filename in ancestors:
            raise ValueError(f"Configuration inheritance cycle: {filename}")
        content = filename.read_bytes()
        data = tomllib.loads(content.decode())
        parent = data.pop("extends", None)
        if parent is not None:
            if not isinstance(parent, str) or not parent:
                raise TypeError("extends must name one profile or relative TOML file")
            parent_path = Path(parent)
            if parent_path.suffix == ".toml" or parent_path.is_absolute():
                parent_path = filename.parent / parent_path
            else:
                parent_path = profiles_dir / (parent + ".toml")
            data = merge(read(parent_path, (*ancestors, filename)), data)
        sources.append(ConfigurationSource(str(filename), hashlib.sha256(content).hexdigest()))
        return data

    if profile is not None:
        if not profile or Path(profile).name != profile or profile.endswith(".toml"):
            raise ValueError("Profile must be a name; use --inference-config for file paths")
        values = read(profiles_dir / (profile + ".toml"))
    elif path is not None:
        values = read(Path(path))
    return ConfigurationSelection(InferenceConfig.from_dict(values or {}), tuple(sources))


def experiment_configuration(
    options: Mapping[str, object],
    root: Path,
    profiles_dir: Path = PROFILES,
) -> ConfigurationSelection | None:
    """Read component configuration without changing archived legacy trial options."""
    keys = ("inference_profile", "inference_config", "configuration")
    if not any(key in options for key in keys):
        return None
    if "optimizations" in options:
        raise ValueError("Choose grouped configuration or legacy optimizations")
    path = options.get("inference_config")
    profile = options.get("inference_profile")
    values = options.get("configuration")
    if path is not None and not isinstance(path, (str, Path)):
        raise TypeError("inference_config must be a path")
    if profile is not None and not isinstance(profile, str):
        raise TypeError("inference_profile must be a string")
    if values is not None and not isinstance(values, dict):
        raise TypeError("configuration must be a table")
    return load_configuration(
        profile=profile,
        path=root / path if path is not None else None,
        values=values,
        profiles_dir=profiles_dir,
    )
