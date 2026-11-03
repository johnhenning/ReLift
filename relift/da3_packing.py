"""Experimental per-dtype block regions and a temporary mapped feature archive."""

from __future__ import annotations

import hashlib
import json
from types import ModuleType
from typing import Any

import numpy as np
import torch


def pack_tensors(
    tensors: list[torch.Tensor], backend: str = "python", native: ModuleType | None = None
) -> list[tuple[torch.Tensor, list[tuple[int, int, int, tuple[int, ...]]]]]:
    """Return homogeneous packed CPU regions plus original tensor index/offsets."""
    storages = [tensor.untyped_storage().data_ptr() for tensor in tensors if tensor.numel()]
    if len(storages) != len(set(storages)):
        raise ValueError(
            "Packed DA3 blocks require independent tensor storages; aliases are unsupported"
        )
    groups = {}
    for index, tensor in enumerate(tensors):
        if tensor.layout != torch.strided or tensor.is_quantized:
            raise ValueError("Packed blocks require dense unquantized tensors")
        groups.setdefault(tensor.dtype, []).append((index, tensor.detach().cpu().contiguous()))
    result = []
    for dtype, items in groups.items():
        bindings, offset = [], 0
        for index, tensor in items:
            bindings.append((index, offset, tensor.numel(), tuple(tensor.shape)))
            offset += tensor.numel()
        if backend == "cpp":
            if native is None:
                raise ValueError("Native block packing requires an injected native module")
            arrays = [tensor.reshape(-1).view(torch.uint8).numpy() for _, tensor in items]
            data = np.empty(sum(a.nbytes for a in arrays), np.uint8)
            native.pack_bytes(arrays, data)
            packed = torch.from_numpy(data).view(dtype)
        elif backend == "python":
            packed = torch.cat([tensor.reshape(-1) for _, tensor in items])
        else:
            raise ValueError(f"Unknown packed-block backend {backend}")
        result.append((packed, bindings))
    return result


class FeatureArchive:
    """Append-only, private-mapped logical tensors; the parent owns all cleanup."""

    def __init__(self, owner: Any, backend: str = "python", native: Any = None) -> None:
        if owner.mode != "disk":
            raise ValueError("Packed feature archives require disk offload")
        if backend not in ("python", "cpp"):
            raise ValueError("Unknown feature archive backend")
        if backend == "cpp" and native is None:
            raise ValueError("Native feature archives require an injected module")
        self.owner, self.backend, self.native = owner, backend, native
        self.path = owner.root / "features.bin"
        self.index_path = owner.root / "feature-index.json"
        if self.path.exists() or self.index_path.exists():
            raise ValueError(
                "Feature archives require a fresh owner; implicit reuse is unsupported"
            )
        self.records = []

    def put(self, tensor: torch.Tensor) -> torch.Tensor:
        cpu = tensor.detach().cpu().contiguous()
        data = cpu.reshape(-1).view(torch.uint8).numpy()
        size = data.nbytes
        previous = self.path.stat().st_size if self.path.exists() else 0
        alignment = max(cpu.element_size(), 1)
        offset = (previous + alignment - 1) // alignment * alignment
        needed = offset - previous + size
        import shutil

        if (
            self.owner.bytes_written + needed + 4096 > self.owner.disk_limit
            or needed > shutil.disk_usage(self.owner.root).free * 0.9
        ):
            raise MemoryError("Feature archive exceeds scratch budget or free disk capacity")
        try:
            with self.path.open("ab") as stream:
                stream.write(bytes(offset - previous))
                stream.flush()  # native fd writes must follow any buffered padding
                if self.backend == "cpp":
                    assert self.native is not None
                    digest = self.native.write_payload(stream.fileno(), b"", data, True)
                else:
                    view = memoryview(data)
                    for start in range(0, size, 1 << 20):
                        stream.write(view[start : start + (1 << 20)])
                    digest = hashlib.sha256(view).hexdigest()
            record = dict(
                offset=offset,
                bytes=size,
                shape=list(cpu.shape),
                dtype=str(cpu.dtype).removeprefix("torch."),
                sha256=digest,
            )
            self.records.append(record)
            temporary = self.index_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(
                    dict(format="relift.da3-features", version=1, records=self.records), indent=2
                )
                + "\n"
            )
            temporary.replace(self.index_path)
        except BaseException:
            if self.path.exists():
                with self.path.open("r+b") as stream:
                    stream.truncate(previous)
            if self.records and self.records[-1].get("offset") == offset:
                self.records.pop()
            raise
        self.owner.bytes_written += needed + self.index_path.stat().st_size
        return self.map(record)

    def map(self, record: dict[str, Any]) -> torch.Tensor:
        dtype = getattr(torch, record["dtype"], None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError("Invalid feature dtype")
        shape = tuple(record["shape"])
        if any(type(d) is not int or d < 0 for d in shape):
            raise ValueError("Invalid feature shape")
        size = torch.empty((), dtype=dtype).element_size()
        offset, length = record["offset"], record["bytes"]
        if (
            type(offset) is not int
            or type(length) is not int
            or offset < 0
            or length != int(np.prod(shape)) * size
            or offset % size
        ):
            raise ValueError("Invalid feature offset or length")
        if offset + length > self.path.stat().st_size:
            raise ValueError("Truncated feature archive")
        if not length:
            return torch.empty(shape, dtype=dtype)
        mapped = np.memmap(self.path, mode="c", dtype=np.uint8, offset=offset, shape=(length,))
        return torch.from_numpy(mapped).view(dtype).reshape(shape)

    def verify(self) -> bool:
        index = json.loads(self.index_path.read_text())
        if (
            index.get("format") != "relift.da3-features"
            or index.get("version") != 1
            or index.get("records") != self.records
        ):
            raise ValueError("Feature archive index mismatch")
        with self.path.open("rb") as stream:
            for record in self.records:
                self.map(record)
                stream.seek(record["offset"])
                digest = hashlib.sha256()
                remaining = record["bytes"]
                while remaining:
                    block = stream.read(min(remaining, 1 << 20))
                    if not block:
                        raise ValueError("Truncated feature archive")
                    digest.update(block)
                    remaining -= len(block)
                if digest.hexdigest() != record["sha256"]:
                    raise ValueError("Feature archive checksum mismatch")
        return True
