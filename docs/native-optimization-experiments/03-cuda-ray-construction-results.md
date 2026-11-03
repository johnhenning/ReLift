# Ray construction results

This document records the initial screening. The subsequent [bug audit and corrected
measurements](cuda-ray-audit.md) identify direction arithmetic and operator-validation
bugs. The corrected partial-fusion path passes full-output parity and stays opt-in
because it is slower than its matching control.

The third experiment adds a blocked eager PyTorch ray builder and a registered
custom CUDA operator. Both retain the same 16-frame depth uploads, full timeline,
point groups, dtype and visibility handling. The current ray path remains default.
The CUDA build uses the explicit compiler/architecture manifest and no fast math.

Five balanced repeats per arm completed. Fresh-process medians were **17.411
seconds current**, **16.664 blocked PyTorch** and **19.149 fused CUDA**. Correction
stage medians were 15.346, 14.772 and 17.114 seconds. GPU peaks were about 3.882,
3.901 and 3.882 GiB; physical writes remain about 0.388 GiB. Runtime ranges overlap.
The blocked Python path's modest stage difference falls below the proposed 10%
component target and has no full-pipeline confirmation; it remains optional.

All five blocked-PyTorch outputs are bitwise equal to the first control for ordinary
XYZ, repeated XYZ and raw lift. Every fused-CUDA output fails the strict full-output
gate. The first checked candidate has maximum normalized errors of **0.053855** for
ordinary XYZ, **0.025436** for repeated XYZ and **0.004285** for raw lift. The allowed
maximum is 0.0001. The numerical cause was not isolated during this initial screen;
the linked audit subsequently identifies and corrects it.

Three CUDA edge/stream tests and six gathering/composition checks pass, covering
nearest ties, borders, degenerate image dimensions, NaN/Inf coordinates, float32
validation and execution on a nondefault stream. Those local checks did not
establish full-pipeline equivalence. The failed full-output records remain visible
and red in comparison charts. The CUDA path is not an accepted equivalent speedup.

Select `{"rays":"python"}` or explicitly `{"rays":"cuda"}` to reproduce
the variants. A CUDA selection requires a native build made with `--cuda`; missing
build support fails rather than using another backend. `InferenceComponents.ray_builder`
can inject a callable `(coords, depth_block, intrinsics, rotation)` returning
`(direction, distance)` for one block, preserving point/frame order. Inputs are
float32; output shapes are `(T,N,3)` and `(T,N)`. The built-in operator validates
device/shape/dtype, uses the caller's CUDA stream and has fake-tensor registration.

Trial commands, compiler flags, binary/source hashes and 14 quality comparisons
are under `outputs/experiments/native-optimizations/cuda-rays`. The ordinary and
padded-pixel errors are reported separately; neither is excluded to obtain a pass.
No new ray backend is enabled by default, and isolated savings are not added to
the gathering experiment's results.
