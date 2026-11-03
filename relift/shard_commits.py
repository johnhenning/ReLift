"""One-owner shard commits with an optional bounded CPU writer thread."""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from types import TracebackType
from typing import Protocol, Self

import numpy as np
from numpy.typing import NDArray


class ShardSink(Protocol):
    def write_shard(
        self,
        start: int,
        *,
        xyz: NDArray[np.float32],
        lift: NDArray[np.float16],
        observed: NDArray[np.bool_],
    ) -> None: ...


class ShardCommitter(Protocol):
    """At most one submitted shard may remain pending; submit transfers ownership.

    The caller must not mutate submitted arrays until drain completes. Writers
    only touch CPU arrays and the sink; no CUDA work or manifest reads run here.
    """

    def submit(
        self,
        start: int,
        *,
        xyz: NDArray[np.float32],
        lift: NDArray[np.float16],
        observed: NDArray[np.bool_],
    ) -> None: ...

    def drain(self) -> None: ...
    @property
    def metrics(self) -> dict[str, str | int | float]: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


class BoundedShardCommitter:
    """Commit synchronously or overlap one owned shard with the next GPU batch.

    Submission waits for the previous commit *after* the next batch has been
    computed. The executor therefore never accumulates an unbounded queue.
    The sink retains its existing validation, hashes, fsync and atomic manifests.
    """

    def __init__(self, sink: ShardSink, backend: str = "sync") -> None:
        if backend not in ("sync", "threaded"):
            raise ValueError("Commit backend must be sync or threaded")
        self.sink, self.backend = sink, backend
        self.executor: ThreadPoolExecutor | None = None
        self.pending: Future[None] | None = None
        self.failure: BaseException | None = None
        self.closed = False
        self.entered = False
        self.submitted = 0
        self.committed = 0
        self.pending_bytes = 0
        self.peak_pending_bytes = 0
        self.wait_seconds = 0.0
        self.write_seconds = 0.0

    def __enter__(self) -> Self:
        if self.entered or self.closed:
            raise RuntimeError("Commit owner cannot be entered twice")
        self.entered = True
        if self.backend == "threaded":
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="relift-shards")
        return self

    def _write(
        self,
        start: int,
        xyz: NDArray[np.float32],
        lift: NDArray[np.float16],
        observed: NDArray[np.bool_],
    ) -> None:
        started = time.perf_counter()
        try:
            self.sink.write_shard(start, xyz=xyz, lift=lift, observed=observed)
            self.committed += 1
        finally:
            self.write_seconds += time.perf_counter() - started

    def submit(
        self,
        start: int,
        *,
        xyz: NDArray[np.float32],
        lift: NDArray[np.float16],
        observed: NDArray[np.bool_],
    ) -> None:
        if not self.entered or self.closed:
            raise RuntimeError("Commit owner is not active")
        self.drain()
        self.submitted += 1
        if self.executor is None:
            try:
                self._write(start, xyz, lift, observed)
            except BaseException as error:
                self.failure = error
                raise
        else:
            self.pending_bytes = xyz.nbytes + lift.nbytes + observed.nbytes
            self.peak_pending_bytes = max(self.peak_pending_bytes, self.pending_bytes)
            self.pending = self.executor.submit(self._write, start, xyz, lift, observed)

    def drain(self) -> None:
        if self.failure is not None:
            raise self.failure
        if self.pending is None:
            return
        started = time.perf_counter()
        try:
            self.pending.result()
        except BaseException as error:
            self.failure = error
            raise
        finally:
            self.pending = None
            self.pending_bytes = 0
            self.wait_seconds += time.perf_counter() - started

    @property
    def metrics(self) -> dict[str, str | int | float]:
        return dict(
            backend=self.backend,
            shards_submitted=self.submitted,
            shards_committed=self.committed,
            max_pending_shards=1 if self.backend == "threaded" else 0,
            peak_pending_payload_bytes=self.peak_pending_bytes,
            pending_payload_bytes=self.pending_bytes,
            wait_seconds=self.wait_seconds,
            worker_write_seconds=self.write_seconds,
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if exc is None:
                self.drain()
            else:
                try:
                    self.drain()
                except BaseException as writer_error:
                    if writer_error is not exc:
                        exc.add_note(f"Background shard commit also failed: {writer_error!r}")
        finally:
            if self.executor is not None:
                self.executor.shutdown(wait=True)
            self.closed = True
