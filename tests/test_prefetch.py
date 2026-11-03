"""The experimental worker adapter must restore shared methods on failure."""

from __future__ import annotations

import pytest

from relift import streaming
from relift.prefetch import RayLookahead


def test_lookahead_restores_methods_after_worker_failure() -> None:
    rays, correct = streaming.MappedObservations.rays, streaming.correct
    with pytest.raises(RuntimeError, match="trial failed"):
        with RayLookahead():
            assert streaming.MappedObservations.rays is not rays
            raise RuntimeError("trial failed")
    assert streaming.MappedObservations.rays is rays
    assert streaming.correct is correct
