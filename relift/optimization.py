"""Validated experiment settings and injectable construction, with no global cache."""
from dataclasses import asdict, dataclass, fields, replace
from functools import partial
import hashlib
import json
import math
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class OptimizationConfig:
    gather: str = "current"
    shards: str = "current"
    rays: str = "current"
    da3_weights: str = "current"
    da3_features: str = "current"
    tracker_init_weights: bool = False
    tracker_copy_outputs: bool = False
    hash_writes: bool = True
    depth_cache_mb: float = 0
    prefetch: bool = False
    corrector_batch: int = 8
    padding_batch: int | None = None
    compile_corrector: bool = False
    reuse_da3: bool = False

    def __post_init__(self):
        for key, choices in dict(gather=("current","python","cpp"), shards=("current","python","cpp"),
                                 rays=("current","python","cuda"), da3_weights=("current","python","cpp"),
                                 da3_features=("current","python","cpp")).items():
            if getattr(self,key) not in choices:
                raise ValueError(f"{key} must be one of {choices}")
        for key in ("tracker_init_weights","tracker_copy_outputs","hash_writes","prefetch","compile_corrector","reuse_da3"):
            if type(getattr(self,key)) is not bool:
                raise TypeError(f"{key} must be a boolean")
        if type(self.depth_cache_mb) not in (int,float) or not math.isfinite(self.depth_cache_mb) or self.depth_cache_mb < 0:
            raise ValueError("depth_cache_mb must be finite and nonnegative")
        for key in ("corrector_batch","padding_batch"):
            value = getattr(self,key)
            if key == "padding_batch" and value is None:
                continue
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown optimization settings: {sorted(unknown)}")
        return cls(**values)

    @classmethod
    def read(cls, path=None, overrides=()):
        values = json.loads(Path(path).read_text()) if path else {}
        for override in overrides:
            key, separator, value = override.partition("=")
            if not separator:
                raise ValueError("Optimization overrides require key=JSON_VALUE")
            values[key] = json.loads(value)
        return cls.from_dict(values)

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class InferenceComponents:
    """Optional overrides implement the same contracts as the default components.

    Give injected implementations stable version/content fingerprints. These are
    included in store provenance; changing a dependency invalidates resume.
    """
    gather_factory: Callable | None = None
    ray_builder: Callable | None = None
    shard_packer: Callable | None = None
    npy_writer: Callable | None = None
    pager_factory: Callable | None = None
    depth_provider_factory: Callable | None = None
    tracker_factory: Callable | None = None
    corrector_loader: Callable | None = None
    native_loader: Callable | None = None
    fingerprints: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        supplied = {f.name for f in fields(self) if f.name != "fingerprints" and getattr(self,f.name) is not None}
        named = dict(self.fingerprints)
        if supplied - named.keys():
            raise ValueError(f"Injected components require fingerprints: {sorted(supplied - named.keys())}")
        if any(not callable(getattr(self,key)) for key in supplied):
            raise TypeError("Injected components must be callable")

    def describe(self):
        return dict(self.fingerprints)

    def with_native_build(self, directory):
        from .native_backend import load_cpu
        directory = Path(directory).resolve()
        manifest = json.loads((directory / "build.json").read_text())
        fingerprints = dict(self.fingerprints, native_loader=manifest["binary_sha256"])
        return replace(self, native_loader=partial(load_cpu, directory), fingerprints=tuple(sorted(fingerprints.items())))


class InferenceSession:
    """Own reusable providers explicitly; close on normal exit or failure.

    Reused DA3 models reside on CPU between stages/clips. This avoids keeping its
    always-resident heads on CUDA while tracking/correction runs.
    """
    def __init__(self, config=None, components=None, native_build=None):
        self.config = config or OptimizationConfig()
        self.components = components or InferenceComponents()
        if native_build:
            self.components = self.components.with_native_build(native_build)
        self.provider = None
        self.provider_key = None

    def depth(self, weights, device, **kwargs):
        from .providers.da3 import DepthAnything3Provider
        key = (str(Path(weights).resolve()), str(device), json.dumps({k:v for k,v in kwargs.items() if k != "scratch_dir"}, sort_keys=True))
        if self.provider is not None and self.provider_key != key:
            self.provider.close()
            self.provider = None
        factory = self.components.depth_provider_factory or DepthAnything3Provider
        if self.provider is None:
            provider = factory(weights, device, optimizations=self.config, components=self.components, **kwargs)
            if self.config.reuse_da3:
                self.provider, self.provider_key = provider, key
        else:
            provider = self.provider
            provider.activate()
        return provider

    def release_depth(self, provider):
        if self.config.reuse_da3:
            provider.park()
        else:
            provider.close()

    def close(self):
        if self.provider is not None:
            self.provider.close()
            self.provider = None

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()
