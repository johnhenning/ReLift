"""Validated experiment settings and injectable construction, with no global cache."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .configuration import ConfigurationSelection, InferenceConfig
    from .shard_commits import ShardCommitter, ShardSink

import torch

from .records.configuration import OptimizationOverrides, OptimizationRecord


@dataclass(frozen=True)
class OptimizationConfig:
    gather: str = "current"
    shards: str = "current"
    shard_writer: str | None = None
    commit_backend: str = "sync"
    rays: str = "current"
    da3_weights: str = "current"
    da3_features: str = "current"
    da3_startup: str = "current"
    tracker_init_weights: bool = False
    tracker_copy_outputs: bool = False
    hash_writes: bool = True
    depth_cache_mb: float = 0
    observation_cache_mb: float = 0
    prefetch: bool = False
    corrector_batch: int = 8
    padding_batch: int | None = None
    compile_corrector: bool = False
    reuse_da3: bool = False

    def __post_init__(self) -> None:
        for key, choices in dict(
            gather=("current", "python", "cpp"),
            shards=("current", "python", "cpp"),
            shard_writer=(None, "current", "cpp"),
            commit_backend=("sync", "threaded"),
            rays=("current", "python", "cuda", "cuda_timeline"),
            da3_weights=("current", "python", "cpp"),
            da3_features=("current", "python", "cpp"),
            da3_startup=("current", "cpu-first"),
        ).items():
            if getattr(self, key) not in choices:
                raise ValueError(f"{key} must be one of {choices}")
        for key in (
            "tracker_init_weights",
            "tracker_copy_outputs",
            "hash_writes",
            "prefetch",
            "compile_corrector",
            "reuse_da3",
        ):
            if type(getattr(self, key)) is not bool:
                raise TypeError(f"{key} must be a boolean")
        for key in ("depth_cache_mb", "observation_cache_mb"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.observation_cache_mb and self.gather != "current":
            raise ValueError('observation cache replaces CPU gathering; select gather="current"')
        for key in ("corrector_batch", "padding_batch"):
            value = getattr(self, key)
            if key == "padding_batch" and value is None:
                continue
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> OptimizationConfig:
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown optimization settings: {sorted(unknown)}")
        return cls(**cast(OptimizationOverrides, dict(values)))

    @classmethod
    def read(
        cls, path: str | Path | None = None, overrides: tuple[str, ...] | list[str] = ()
    ) -> OptimizationConfig:
        values = json.loads(Path(path).read_text()) if path else {}
        for override in overrides:
            key, separator, value = override.partition("=")
            if not separator:
                raise ValueError("Optimization overrides require key=JSON_VALUE")
            values[key] = json.loads(value)
        return cls.from_dict(values)

    def as_dict(self) -> OptimizationRecord:
        return OptimizationRecord(
            gather=self.gather,
            shards=self.shards,
            shard_writer=self.shard_writer,
            commit_backend=self.commit_backend,
            rays=self.rays,
            da3_weights=self.da3_weights,
            da3_features=self.da3_features,
            da3_startup=self.da3_startup,
            tracker_init_weights=self.tracker_init_weights,
            tracker_copy_outputs=self.tracker_copy_outputs,
            hash_writes=self.hash_writes,
            depth_cache_mb=self.depth_cache_mb,
            observation_cache_mb=self.observation_cache_mb,
            prefetch=self.prefetch,
            corrector_batch=self.corrector_batch,
            padding_batch=self.padding_batch,
            compile_corrector=self.compile_corrector,
            reuse_da3=self.reuse_da3,
        )

    def validate_mode(self, mode: str) -> None:
        if mode not in ("none", "streamed", "ram", "disk"):
            raise ValueError("Resolve the offload mode before validating optimizations")
        if self.da3_weights != "current" and mode not in ("ram", "disk"):
            raise ValueError("Packed DA3 weights require RAM or disk offload")
        if self.da3_features != "current" and mode != "disk":
            raise ValueError("Feature archives require disk offload")
        if self.da3_startup != "current" and mode not in ("ram", "disk"):
            raise ValueError("CPU-first DA3 startup requires RAM or disk offload")
        if mode == "none":
            default = dict(OptimizationConfig().as_dict())
            stream_fields = (
                "gather",
                "shards",
                "commit_backend",
                "rays",
                "hash_writes",
                "depth_cache_mb",
                "observation_cache_mb",
                "prefetch",
                "padding_batch",
                "compile_corrector",
            )
            selected = [key for key in stream_fields if getattr(self, key) != default[key]]
            if self.shard_writer == "cpp":
                selected.append("shard_writer")
            if selected:
                raise ValueError(
                    f"Streaming optimizations require streamed/RAM/disk mode: {selected}"
                )


@dataclass(frozen=True)
class InferenceComponents:
    """Optional overrides implement the same contracts as the default components.

    Give injected implementations stable version/content fingerprints. These are
    included in store provenance; changing a dependency invalidates resume.
    """

    gather_factory: Callable | None = None
    observation_cache_factory: Callable | None = None
    ray_builder: Callable | None = None
    shard_packer: Callable | None = None
    shard_committer_factory: Callable[[ShardSink], ShardCommitter] | None = None
    npy_writer: Callable | None = None
    pager_factory: Callable | None = None
    cold_storage_factory: Callable | None = None
    feature_archive_factory: Callable | None = None
    block_packer: Callable | None = None
    depth_provider_factory: Callable | None = None
    tracker_factory: Callable | None = None
    corrector_loader: Callable | None = None
    native_loader: Callable | None = None
    fingerprints: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        supplied = {
            f.name
            for f in fields(self)
            if f.name != "fingerprints" and getattr(self, f.name) is not None
        }
        named = dict(self.fingerprints)
        if supplied - named.keys():
            raise ValueError(
                f"Injected components require fingerprints: {sorted(supplied - named.keys())}"
            )
        if any(not callable(getattr(self, key)) for key in supplied):
            raise TypeError("Injected components must be callable")

    def describe(self) -> dict[str, str]:
        return dict(self.fingerprints)

    def with_native_build(self, directory: str | Path) -> InferenceComponents:
        from .native_backend import load_cpu

        directory = Path(directory).resolve()
        manifest = json.loads((directory / "build.json").read_text())
        fingerprints = dict(self.fingerprints, native_loader=manifest["binary_sha256"])
        return replace(
            self,
            native_loader=partial(load_cpu, directory),
            fingerprints=tuple(sorted(fingerprints.items())),
        )


class InferenceSession:
    """Own reusable providers explicitly; close on normal exit or failure.

    Reused DA3 models reside on CPU between stages/clips. This avoids keeping its
    always-resident heads on CUDA while tracking/correction runs.
    """

    def __init__(
        self,
        config: OptimizationConfig | InferenceConfig | ConfigurationSelection | None = None,
        components: InferenceComponents | None = None,
        native_build: str | Path | None = None,
    ) -> None:
        from .configuration import ConfigurationSelection, InferenceConfig

        self.configuration: ConfigurationSelection | None = None
        if isinstance(config, InferenceConfig):
            config = ConfigurationSelection(config)
        if isinstance(config, ConfigurationSelection):
            self.configuration = config
            native_build = native_build or config.config.runtime.native_build
            self.config = config.config.optimizations()
        else:
            self.config = config or OptimizationConfig()
        self.components = components or InferenceComponents()
        if native_build:
            self.components = self.components.with_native_build(native_build)
        self.provider = None
        self.provider_key = None

    def depth(self, weights: str | Path, device: str | torch.device, **kwargs: Any) -> Any:
        from .providers.da3 import DepthAnything3Provider

        key = (
            str(Path(weights).resolve()),
            str(device),
            json.dumps({k: v for k, v in kwargs.items() if k != "scratch_dir"}, sort_keys=True),
        )
        if self.provider is not None and self.provider_key != key:
            self.provider.close()
            self.provider = None
        factory = self.components.depth_provider_factory or DepthAnything3Provider
        if self.provider is None:
            provider = factory(
                weights, device, optimizations=self.config, components=self.components, **kwargs
            )
            if self.config.reuse_da3:
                self.provider, self.provider_key = provider, key
        else:
            provider = self.provider
            provider.activate()
        return provider

    def release_depth(self, provider: Any) -> None:
        if self.config.reuse_da3:
            provider.park()
        else:
            provider.close()

    def close(self) -> None:
        if self.provider is not None:
            self.provider.close()
            self.provider = None

    def __enter__(self) -> InferenceSession:
        return self

    def __exit__(self, *error: Any) -> None:
        self.close()
