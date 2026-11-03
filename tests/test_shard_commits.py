"""Background commits stay bounded, propagate failures and preserve verified resume."""

from __future__ import annotations

import gc
import threading
import weakref
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from test_streaming import prepared_window

from relift.model import Corrector
from relift.optimization import InferenceComponents, OptimizationConfig
from relift.shard_commits import BoundedShardCommitter, ShardSink
from relift.storage import TrajectoryStore
from relift.streaming import stream_correct


def payload() -> dict[str, Any]:
    return dict(
        xyz=np.ones((3, 5, 3), np.float32),
        lift=np.ones((3, 5, 3), np.float16),
        observed=np.ones((3, 5), np.bool_),
    )


class BlockingSink:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.starts: list[int] = []
        self.worker_names: list[str] = []

    def write_shard(
        self,
        start: int,
        *,
        xyz: NDArray[np.float32],
        lift: NDArray[np.float16],
        observed: NDArray[np.bool_],
    ) -> None:
        self.started.set()
        if not self.release.wait(5):
            raise TimeoutError("Test failed to release the writer")
        assert xyz[0, 0, 0] == 1 and lift[0, 0, 0] == 1 and observed.all()
        self.starts.append(start)
        self.worker_names.append(threading.current_thread().name)


def test_one_pending_owner_retains_arrays_and_blocks_further_queueing() -> None:
    sink = BlockingSink()
    values = payload()
    reference = weakref.ref(values["xyz"])
    with BoundedShardCommitter(sink, "threaded") as owner:
        owner.submit(0, **values)
        del values
        gc.collect()
        assert sink.started.wait(5)
        assert reference() is not None  # pending future owns the CPU buffers
        assert owner.metrics["pending_payload_bytes"] == 3 * 5 * 19
        done = threading.Event()
        errors: list[BaseException] = []

        def second_submit() -> None:
            try:
                owner.submit(3, **payload())
            except BaseException as error:
                errors.append(error)
            finally:
                done.set()

        producer = threading.Thread(target=second_submit)
        producer.start()
        try:
            assert not done.wait(0.05)
            assert owner.submitted == 1
        finally:
            sink.release.set()
            producer.join(5)
        assert done.is_set() and not errors
        owner.drain()
        assert owner.metrics["shards_committed"] == 2
        assert owner.metrics["pending_payload_bytes"] == 0
        assert owner.metrics["peak_pending_payload_bytes"] == 3 * 5 * 19
        assert sink.starts == [0, 3]
        assert all(name.startswith("relift-shards") for name in sink.worker_names)
    gc.collect()
    assert reference() is None and owner.closed
    with pytest.raises(RuntimeError, match="active"):
        owner.submit(6, **payload())


@pytest.mark.parametrize("backend", ["sync", "threaded"])
def test_failure_surfaces_at_submission_or_drain_and_closes_owner(backend: str) -> None:
    class Broken:
        def write_shard(self, start: int, **arrays: Any) -> None:
            raise OSError("disk full")

    owner = BoundedShardCommitter(Broken(), backend)
    with pytest.raises(OSError, match="disk full"), owner:
        owner.submit(0, **payload())
        owner.drain()
    assert owner.closed
    assert owner.metrics["shards_committed"] == 0
    with pytest.raises(OSError, match="disk full"):
        owner.drain()


def test_cleanup_preserves_primary_error_with_writer_failure_note() -> None:
    class Broken:
        def write_shard(self, start: int, **arrays: Any) -> None:
            raise OSError("write failure")

    owner = BoundedShardCommitter(Broken(), "threaded")
    with pytest.raises(ValueError, match="compute failure") as captured, owner:
        owner.submit(0, **payload())
        raise ValueError("compute failure")
    assert owner.closed and any("write failure" in note for note in captured.value.__notes__)


@pytest.mark.filterwarnings("ignore:CUDA is not available")
def test_injected_failure_resume_exact_outputs_and_checksum_validation(tmp_path: Path) -> None:
    prepared_window(tmp_path)
    torch.manual_seed(7)
    model = Corrector(dim=16, depth=1, heads=2, chunk=4).eval()
    config = OptimizationConfig(corrector_batch=2, commit_backend="threaded")
    failed = False
    writes: list[int] = []

    def factory(store: ShardSink) -> BoundedShardCommitter:
        class Sink:
            def write_shard(self, start: int, **arrays: Any) -> None:
                nonlocal failed
                if writes and not failed:
                    failed = True
                    raise OSError("interrupted")
                store.write_shard(start, **arrays)
                writes.append(start)

        return BoundedShardCommitter(Sink(), "threaded")

    components = InferenceComponents(
        shard_committer_factory=factory, fingerprints=(("shard_committer_factory", "fail-once-v1"),)
    )
    with pytest.raises(OSError, match="interrupted"):
        stream_correct(
            tmp_path,
            tmp_path / "resume",
            model,
            {},
            group=4,
            device="cpu",
            optimizations=config,
            components=components,
        )
    incomplete = TrajectoryStore.open(tmp_path / "resume", allow_incomplete=True)
    assert incomplete.verify(complete=False)
    assert incomplete.manifest is not None and len(incomplete.manifest["shards"]) == 1
    resumed = stream_correct(
        tmp_path,
        tmp_path / "resume",
        model,
        {},
        group=4,
        device="cpu",
        optimizations=config,
        components=components,
    )
    assert writes.count(0) == 1 and resumed.verify()
    baseline = stream_correct(
        tmp_path,
        tmp_path / "sync",
        model,
        {},
        group=4,
        device="cpu",
        optimizations=OptimizationConfig(corrector_batch=2),
    )
    for field in ("xyz", "lift", "observed"):
        for frame in range(5):
            np.testing.assert_array_equal(
                resumed.read_frame(field, frame), baseline.read_frame(field, frame)
            )
    assert resumed.manifest is not None
    shard = resumed.root / resumed.manifest["shards"][0]["arrays"]["xyz"]["file"]
    with shard.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"x")
    with pytest.raises(ValueError, match="checksum"):
        stream_correct(
            tmp_path,
            tmp_path / "resume",
            model,
            {},
            group=4,
            device="cpu",
            optimizations=config,
            components=components,
        )


@pytest.mark.filterwarnings("ignore:CUDA is not available")
def test_pending_workspace_counts_toward_staging_budget(tmp_path: Path) -> None:
    prepared_window(tmp_path)
    model = Corrector(dim=16, depth=1, heads=2, chunk=4).eval()
    # 5 frames * 8 points * 64 workspace bytes; the pending writer reserves another.
    budget = 3000 / 2**30
    stream_correct(
        tmp_path,
        tmp_path / "sync",
        model,
        {},
        group=4,
        device="cpu",
        optimizations=OptimizationConfig(corrector_batch=2),
        ram_budget_gb=budget,
    )
    with pytest.raises(MemoryError, match="staging"):
        stream_correct(
            tmp_path,
            tmp_path / "threaded",
            model,
            {},
            group=4,
            device="cpu",
            optimizations=OptimizationConfig(corrector_batch=2, commit_backend="threaded"),
            ram_budget_gb=budget,
        )
