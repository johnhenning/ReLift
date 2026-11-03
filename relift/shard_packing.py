"""Optional shard packing; file durability and manifests stay in storage.py."""
import io

import numpy as np


class ShardPacker:
    def __init__(self, backend, native=None):
        if backend not in ("python","cpp"):
            raise ValueError(f"Unknown shard backend {backend}")
        self.backend, self.native = backend, native
        if backend == "cpp" and native is None:
            from .native_backend import load_cpu
            self.native = load_cpu()

    def __call__(self, name, array):
        if self.backend == "cpp":
            return self.native.pack_finite(array)
        order = "F" if array.flags.f_contiguous and not array.flags.c_contiguous else "C"
        value = np.array(array, copy=True, order=order)
        return value, name == "observed" or bool(np.isfinite(value).all())

    def write(self, stream, array, hash_write):
        if not (array.flags.c_contiguous or array.flags.f_contiguous):
            array = np.ascontiguousarray(array)
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, np.lib.format.header_data_from_array_1_0(array))
        digest = self.native.write_payload(stream.fileno(), header.getvalue(), array, hash_write)
        return digest if hash_write else None
