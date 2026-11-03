# CUDA ray bug audit

The failed full-trajectory gate was caused by direction arithmetic, not incorrect
nearest-depth sampling. The audit is reproducible with `scripts/audit_cuda_rays.py`;
its source/build hashes and before/after records are retained with the campaign.

## Findings and changes

| Finding | Evidence | Correction |
|---|---|---|
| Normalization used a different addition order | In the first 16-frame block, 270,336 norms differ for `(x²+y²)+1`; `(x²+1)+y²` matches all of them | Match the pinned three-element ATen reduction order; keep radial distance's separate original expression |
| Rotation used separate multiply/add arithmetic | Replacing rotation with the observed cuBLAS FMA order matches the large real-input case, but fails exact tests for some small shapes/layouts | Keep rotation in the original PyTorch `einsum`, preserving its layout and shape-dependent implementation |
| Direct registered-operator calls bypassed wrapper checks | Invalid dtype/shape or noncontiguous tensors could reach a raw pointer kernel without the convenience wrapper | Validate inside the registered implementation and normalize its raw-kernel inputs there; test direct calls |
| Operator cache identity omitted Python implementation | A changed wrapper could reuse an earlier registered implementation for the same binary | Include both binary and Python implementation bytes in the registered operator identity |
| Very large image bounds were not constrained to exact float32 indices | For width 16,777,220, float32(width−1) rounds to width, which could pass an invalid raw index to the kernel | Reject dimensions above 2²⁴ inside the operator; verify rejection using fake tensors without allocating or executing an unsafe probe |

The norm observation is specific to the pinned PyTorch 2.11 float32 path. Its
[norm kernel](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/ReduceNormKernel.cu)
uses the reduction machinery rather than promising a source-expression addition
order. The rotation accumulation results are measured observations on this setup,
not a universal claim about all cuBLAS kernels. Hardcoding one order was insufficient.

The supported `rays="cuda"` builder now fuses coordinate normalization, nearest
sampling, camera-direction normalization and range, then uses PyTorch for rotation
and final direction sanitization. This is **partial fusion**, not a completely fused
replacement. The older fully fused path remains only in recorded prototype sources
and the internal diagnostic entry point; it is not selected by the configured builder.
Inference callers should use the validated builder/operator, not raw pointer helpers.

## Verification

The original audit scanned every tracker coordinate over all **11,854,080**
frame/reference-pixel pairs of swing-tire. Encoded depth images identified the exact
selected pixel independently of ray length. There were **zero nearest-pixel
mismatches** and **zero range differences**. Direction components differed 9,970,170
times, with a maximum absolute error of 1.7881393432617188e-7.

The corrected scan has **zero differences in directions, ranges or sampled pixels**
over the same inputs. Ten CUDA regression cases pass: image borders/nearest ties,
degenerate dimensions, nonfinite coordinates, nondefault streams, direct-operator
validation, noncontiguous inputs, and exact arithmetic across point counts 1, 7 and
8192 with both rotation layouts, plus rejection of inexact oversized image bounds.
Tests pin imports to the selected source so an
editable installation cannot accidentally test different code.

Five fresh corrected control/candidate pairs then completed. All **nine** subsequent
full-output comparisons against the first control pass. The quality records include
ordinary trajectories, padded repeated trajectories, raw lift, masks, prepared data,
images and camera arrays. The initial failed experiment is preserved separately.

The corrected variant's process median is **19.096 seconds**, versus **18.104** for
its matching current control. It fixes correctness but does not establish a speedup;
the default ray backend remains unchanged. These are correction-only measurements,
not full-pipeline speed claims. Compiler, driver, hardware and source versions remain
part of the evidence; these tests do not prove equivalence for every GPU/library.

The [profiling follow-up](cuda-ray-profile.md) investigates the corrected path's
overhead and tests `cuda_timeline` without changing the audited kernel arithmetic.

## Reproduce

Choose fresh output names. The original failed snapshot and corrected source are
retained locally as `outputs/experiments/native-rays-source` and
`outputs/experiments/native-rays-compatible-source`.

```bash
uv run python scripts/build_native.py \
  --source outputs/experiments/native-rays-compatible-source \
  --out outputs/native/rays-recheck --cuda --cxx g++-13
uv run python scripts/audit_cuda_rays.py \
  --source outputs/experiments/native-rays-compatible-source \
  --build outputs/native/rays-recheck \
  --window outputs/benchmarks/offload-primary/swing-tire/ram/trial-01/output/window \
  --out outputs/ray-audit-recheck.json
uv run python scripts/run_source_tests.py \
  --source outputs/experiments/native-rays-compatible-source \
  --native-build outputs/native/rays-recheck \
  --target tests/gpu/test_cuda_rays.py --target tests/gpu/test_ray_arithmetic.py
```

The trial spec's `cuda-rays-corrected` entry records the paired measurement. Its
outputs and quality records are immutable; change output roots for a new campaign.
