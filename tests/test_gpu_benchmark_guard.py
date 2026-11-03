"""Benchmark exclusivity is checked without starting CUDA or requiring a GPU."""

from __future__ import annotations

import ctypes
from typing import Any, cast

import pytest

from scripts import benchmark_offload
from scripts.benchmark_offload import NvidiaSampler, Telemetry


def test_nvml_process_query_layout_resize_and_errors() -> None:
    calls: list[int] = []

    class Library:
        def nvmlDeviceGetComputeRunningProcesses_v3(
            self, handle: Any, count: Any, entries: Any
        ) -> int:
            size = ctypes.cast(count, ctypes.POINTER(ctypes.c_uint))
            calls.append(size[0])
            if len(calls) == 1:
                size[0] = 40
                return 7
            size[0] = 2
            entries[0].pid, entries[0].usedGpuMemory = 123, 4096
            entries[1].pid, entries[1].usedGpuMemory = 456, 8192
            return 0

    sampler = NvidiaSampler.__new__(NvidiaSampler)
    sampler.library = cast(ctypes.CDLL, Library())
    sampler.handle = ctypes.c_void_p(1)
    sampler.error = None
    assert sampler.compute_processes() == [
        dict(pid=123, gpu_memory_bytes=4096),
        dict(pid=456, gpu_memory_bytes=8192),
    ]
    assert calls == [32, 56]
    sampler.library = None
    with pytest.raises(RuntimeError, match="exclusivity"):
        sampler.compute_processes()


@pytest.mark.parametrize("intruder", [False, True])
def test_telemetry_allows_worker_children_and_flags_other_compute(
    monkeypatch: pytest.MonkeyPatch,
    intruder: bool,
) -> None:
    class GPU:
        error = None

        def sample(self) -> dict[str, int]:
            return {}

        def compute_processes(self) -> list[dict[str, int]]:
            ids = [10, 11, 99] if intruder else [10, 11]
            return [dict(pid=pid, gpu_memory_bytes=1) for pid in ids]

    monkeypatch.setattr(
        benchmark_offload, "process_tree", lambda pid: [dict(pid=11, started_ticks=1)]
    )
    telemetry = Telemetry(GPU(), require_exclusive_gpu=True)
    telemetry.capture(10, 0.1)
    assert [p["pid"] for p in telemetry.interference] == ([99] if intruder else [])
    assert telemetry.summary()["gpu_exclusivity_monitored"]
