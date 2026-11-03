"""Experimental one-batch ray lookahead for an isolated inference worker.

The context temporarily adapts the streaming module, so it is deliberately limited
to one inference call per process. It is not enabled by the ordinary demo path.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch


class PrefetchedRays:
    """Bounded injected ray source; no module or class methods are replaced."""
    def __init__(self, reader):
        if reader.device.type != "cuda":
            raise ValueError("Prefetch requires a CUDA reader")
        self.reader, self.pending, self.ids = reader, None, None
        self.stream = torch.cuda.Stream(device=reader.device)
        self.stream.wait_stream(torch.cuda.current_stream(reader.device))
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.peak_pinned_bytes = 0

    def __enter__(self):
        return self

    def schedule(self, ids):
        if self.pending is not None:
            raise RuntimeError("Prefetch exceeded its one-batch bound")
        self.ids = np.array(ids, copy=True)
        self.pending = self.executor.submit(self._prepare, self.ids)

    def _prepare(self, ids):
        hosts = []
        self.reader._pinned_inputs = hosts
        try:
            with torch.cuda.stream(self.stream):
                result = self.reader.rays(ids)
            self.stream.synchronize()
            self.peak_pinned_bytes = max(self.peak_pinned_bytes, sum(t.numel()*t.element_size() for t in hosts))
            return result
        finally:
            self.stream.synchronize()
            self.reader._pinned_inputs = None

    def rays(self, ids):
        if self.pending is None:
            return self.reader.rays(ids)
        if not np.array_equal(ids, self.ids):
            raise RuntimeError("Prefetch selection differs from execution plan")
        result = self.pending.result()
        self.pending = None
        for tensor in result:
            tensor.record_stream(torch.cuda.current_stream(self.reader.device))
        return result

    def __exit__(self, *error):
        try:
            self.executor.shutdown(wait=True)
            self.stream.synchronize()
        finally:
            self.reader._pinned_inputs = None


class RayLookahead:
    def __init__(self, group=1024, batch=8, seed=0):
        self.group, self.batch, self.seed = group, batch, seed
        self.groups, self.index, self.pending, self.stream = [], -1, None, None
        self.peak_pinned_bytes = 0
        self.host_to_device_bytes = 0

    def __enter__(self):
        from . import streaming
        self.module = streaming
        self.original_rays = streaming.MappedObservations.rays
        self.original_correct = streaming.correct
        self.executor = ThreadPoolExecutor(max_workers=1)
        owner = self

        def rays(reader, pixel_ids, frame=None):
            if frame == 0:
                result = owner.original_rays(reader, pixel_ids, frame=frame)
                valid = torch.nonzero(result[2][0] > 0)[:, 0]
                perm, _ = streaming._batch_plan(valid, owner.group, owner.batch, owner.seed, reader.device)
                order = perm.cpu().numpy()
                for start in range(0, len(order), owner.group * owner.batch):
                    ids = order[start:start + owner.group * owner.batch]
                    need = (-len(ids)) % owner.group
                    owner.groups.append(np.concatenate([ids, ids[np.arange(need) % len(ids)]]) if need else ids)
                owner.reader = reader
                owner.stream = torch.cuda.Stream(device=reader.device)
                owner.stream.wait_stream(torch.cuda.current_stream(reader.device))
                return result
            expected = owner.index + 1
            if expected < len(owner.groups) and np.array_equal(pixel_ids, owner.groups[expected]):
                owner.index = expected
                if owner.pending is not None:
                    result = owner.pending.result()
                    owner.pending = None
                    for tensor in result:
                        tensor.record_stream(torch.cuda.current_stream(reader.device))
                    return result
            return owner.original_rays(reader, pixel_ids, frame=frame)

        def correct(*args, **kwargs):
            result = owner.original_correct(*args, **kwargs)
            following = owner.index + 1
            if following < len(owner.groups):
                if owner.pending is not None:
                    raise RuntimeError("Ray lookahead exceeded its one-batch bound")
                owner.pending = owner.executor.submit(owner._prepare, owner.groups[following])
            return result

        streaming.MappedObservations.rays, streaming.correct = rays, correct
        return self

    def _prepare(self, ids):
        hosts = []
        self.reader._pinned_inputs = hosts
        try:
            with torch.cuda.stream(self.stream):
                result = self.original_rays(self.reader, ids)
            self.stream.synchronize()
            size = sum(tensor.numel() * tensor.element_size() for tensor in hosts)
            self.peak_pinned_bytes = max(self.peak_pinned_bytes, size)
            self.host_to_device_bytes += size
            return result
        finally:
            self.stream.synchronize()
            self.reader._pinned_inputs = None

    def __exit__(self, *error):
        try:
            self.executor.shutdown(wait=True)
            if self.stream is not None:
                self.stream.synchronize()
        finally:
            self.module.MappedObservations.rays = self.original_rays
            self.module.correct = self.original_correct
