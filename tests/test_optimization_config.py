from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from relift.optimization import InferenceComponents, InferenceSession, OptimizationConfig
from relift.streaming import MappedObservations
from test_streaming import prepared_window
from scripts.optimization_matrix import expand


def test_strict_settings_and_independent_combinations(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(dict(gather="python", hash_writes=False, depth_cache_mb=256)))
    settings = OptimizationConfig.read(path, ["prefetch=true", "corrector_batch=4"])
    assert settings.gather == "python" and settings.prefetch and not settings.hash_writes
    assert settings.corrector_batch == 4
    for values in ({"typo":True}, {"gather":"missing"}, {"prefetch":"true"}, {"corrector_batch":True}, {"depth_cache_mb":float("nan")}):
        with pytest.raises((TypeError,ValueError)):
            OptimizationConfig.from_dict(values)
    matrix = expand({"corrector_batch":8}, {"hash_writes":[True,False],"gather":["current","python"]})
    assert len(matrix) == 4 and len({(row["hash_writes"],row["gather"]) for row in matrix}) == 4
    with pytest.raises(ValueError):
        expand({}, {"gather":[]})


def test_injected_gather_is_used_and_fingerprinted(tmp_path):
    prepared_window(tmp_path)
    calls = []
    def gather(tracks, visibility, ids):
        calls.append(ids.copy())
        rows, cols = np.divmod(ids, tracks.shape[2])
        return tracks[:,rows,cols].astype(np.float32), visibility[:,rows,cols].astype(np.float32)
    components = InferenceComponents(gather_factory=lambda:gather, fingerprints=(("gather_factory","test-v1"),))
    current = MappedObservations(tmp_path, "cpu").rays(np.array([0,3,3,14]))
    actual = MappedObservations(tmp_path, "cpu", components=components).rays(np.array([0,3,3,14]))
    assert len(calls) == 1
    for a,b in zip(current,actual):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    with pytest.raises(ValueError,match="fingerprints"):
        InferenceComponents(gather_factory=lambda:gather)


def test_session_reuses_only_matching_provider_and_closes_on_exception(tmp_path):
    events = []
    class Provider:
        def __init__(self, *args, **kwargs): events.append("create")
        def park(self): events.append("park")
        def activate(self): events.append("activate")
        def close(self): events.append("close")
    components = InferenceComponents(depth_provider_factory=Provider, fingerprints=(("depth_provider_factory","fake-v1"),))
    with pytest.raises(RuntimeError,match="intentional"):
        with InferenceSession(OptimizationConfig(reuse_da3=True), components) as session:
            provider = session.depth(tmp_path,"cpu",scratch_dir=str(tmp_path/"a"))
            session.release_depth(provider)
            assert session.depth(tmp_path,"cpu",scratch_dir=str(tmp_path/"b")) is provider
            raise RuntimeError("intentional")
    assert events == ["create","park","activate","close"]
