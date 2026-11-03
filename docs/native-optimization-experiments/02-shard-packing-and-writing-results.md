# Shard packing and writing results

The experiment implements optional NumPy and C++ packing, plus a native NPY payload
writer. Python still generates the NPY header, flushes/fsyncs files, renames them,
commits the manifest and performs final verification. The native writer hashes
exact accepted bytes with OpenSSL and handles short writes/EINTR. Hashing itself
is already native in the control; the experiment targets surrounding passes/copies.

Five balanced fresh-process repeats per arm completed over the full prepared
84-frame swing-tire input. Process medians were **18.225 seconds current**, **18.388
NumPy** and **19.831 C++**. Correction-stage medians were 16.335, 16.460 and 17.875
seconds. GPU peaks remain about 3.882 GiB, sampled tree RSS about 1.53 GiB and
physical writes about 0.388 GiB. The native path does not meet the performance
target and stays disabled by default. Required commit waits were preserved.

All 14 full-output comparisons pass, including repeated pixels, raw lift, masks
and prepared arrays. The CPU suite covers exact NPY bytes for supported dtypes,
C/F order and strided sources, nonfinite rejection, failed fsync, recovery and
checksum rejection. A native write error propagates without publishing a manifest
entry. Twenty-nine targeted CPU checks passed with the built extension.

Select the experiment using `{"shards":"python"}` or `{"shards":"cpp"}` in
the shared optimization config; the C++ selection includes the native payload
writer. `hash_writes` remains independent. Python callers can inject the packer
and NPY writer separately through `InferenceComponents.shard_packer` and
`InferenceComponents.npy_writer` to isolate those subcomponents.

The packer contract is `(field_name, array) -> (packed_array, finite_bool)`. It
preserves values, dtype and the existing NPY C/F order decision. The writer contract
is `(binary_stream, packed_array, hash_write) -> sha256_or_none`; it must neither
close the stream nor publish a manifest. Injected functions require fingerprints.
Native builds require C++17 and OpenSSL development headers/libcrypto. The compiler,
linked OpenSSL version, source and binary hashes are recorded outside timing.

The trial options, build manifest, per-run telemetry and full quality records are
under `outputs/experiments/native-optimizations/shard-writer`. Compact technical
records and figures are included in the review branch. This negative screening
result is not followed by a claim of faster full inference or a default change.
