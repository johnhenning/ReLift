# Front-end licences

ReLift vendors no upstream source in Git. `uv sync` installs the `relift-frontends` Python dependency
from `packages/frontends/`. Its build hook downloads the five upstream source archives and CoWTracker's two nested dependencies at the exact
published Git revisions, verifies their SHA256 hashes, and builds a normal wheel containing inference
modules, data files, scoring utilities, and licences. Pins and package mappings live in
`packages/frontends/upstreams.json`; no Git submodules or runtime downloads are required.

AllTracker and CoWTracker have no package metadata, and 4RC's setup omits nested modules and evaluation
helpers. Depth Anything 3 and VGGT-Omega declare NumPy <2 and application/training dependencies that
conflict with ReLift's published NumPy 2 inference environment. The adapter packages the upstream
sources unchanged with the inference dependencies ReLift already used, plus MoviePy for DA3's imports.
It does not install upstream training, app, or optional export environments. Each upstream keeps its
own licence; code and weight licences are not always the same.

The first build needs network access to GitHub; uv caches the resulting wheel. Rebuild after changing
pins with `uv sync --reinstall-package relift-frontends`, updating the revision and checksum together.

| Front end | Role | Code | Weights |
|---|---|---|---|
| [AllTracker](https://github.com/aharley/alltracker) | 2-D tracker | MIT | MIT |
| [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) | geometry | Apache-2.0 | **CC BY-NC-4.0** — the `DA3NESTED-GIANT-LARGE-1.1` checkpoint we pin (DA3's small/base/metric models are Apache-2.0) |
| [4RC](https://github.com/Luo-Yihang/4RC) | geometry | S-Lab License 1.0 — **non-commercial** | S-Lab License 1.0 |
| [CoWTracker](https://github.com/facebookresearch/cowtracker) | 2-D tracker | FAIR Noncommercial Research License | CC BY-NC-4.0 |
| [VGGT-Omega](https://github.com/facebookresearch/vggt-omega) | geometry | FAIR Noncommercial Research License | FAIR Noncommercial Research License — **gated** |

AllTracker is the only front end that is permissive end to end: every geometry model we use ships non-commercial
weights, so the default demo path (Depth Anything 3 + AllTracker) is non-commercial too. Commercial use needs to
be cleared with the respective authors.

Code licences are each repository's `LICENSE`; weight licences are the Hugging Face model cards of the exact
checkpoints `configs/weights.yaml` pins. Checked 2026-10-26.
