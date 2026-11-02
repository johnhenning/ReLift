"""Explicit CUDA stage timing and allocator peaks for offload experiments."""
from contextlib import contextmanager
import json
from pathlib import Path
import resource
import time

import torch


class Metrics:
    def __init__(self, output, settings):
        self.path = Path(output) / "metrics.json"
        self.data = dict(settings=settings, stages={})
        self.started = time.perf_counter()
        self.save()

    def save(self):
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.data, indent=2) + "\n")
        temporary.replace(self.path)

    @contextmanager
    def stage(self, name, cuda=True):
        active = cuda and torch.cuda.is_available()
        if active:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        record = dict(status="running")
        self.data["stages"][name] = record
        self.save()
        try:
            yield record
            if active:
                torch.cuda.synchronize()
            record["status"] = "complete"
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record["seconds"] = time.perf_counter() - start
            record["process_rss_highwater_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            if active:
                record.update(cuda_allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                              cuda_reserved_peak_bytes=torch.cuda.max_memory_reserved())
            self.data["wall_seconds_since_metrics_init"] = time.perf_counter() - self.started
            self.save()
