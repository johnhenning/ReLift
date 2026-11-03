"""The ported corrector reproduces the research model's outputs.

``tests/fixtures/forward_<gpu>.pt`` holds the released checkpoint's outputs on seed-generated inputs, recorded with
the research-time code on the named GPU type (the inputs are regenerated here from the same seed).  bf16 outputs are
compared bit for bit only on the same GPU type; fp32 within 1e-4 on any CUDA device.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from relift.checkpoint import load

FIX = Path(__file__).resolve().parents[1] / "fixtures"
pytestmark = pytest.mark.gpu


def inputs(seed: int, B: int = 2, T: int = 84, P: int = 128) -> Any:
    rng = np.random.default_rng(seed)
    f = torch.from_numpy(rng.normal(size=(B, T, P, 14)).astype(np.float32) * 0.5)
    a = torch.from_numpy(rng.normal(size=(B, P, 3)).astype(np.float32))
    return f, a


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_forward_matches_recorded_outputs() -> None:
    ckpt = os.environ.get("RELIFT_CHECKPOINT")
    if not ckpt:
        pytest.skip("set RELIFT_CHECKPOINT to the released checkpoint")
    fixtures = sorted(glob.glob(str(FIX / "forward_*.pt")))
    if not fixtures:
        pytest.skip("no recorded fixtures")
    model, _ = load(ckpt)
    gpu = torch.cuda.get_device_name(0)
    for fx in fixtures:
        rec = torch.load(fx, weights_only=False)  # our own fixture: numpy arrays and dicts
        same_gpu = rec["gpu"] == gpu
        for seed, out in rec["cases"].items():
            f, a = inputs(int(seed))
            with torch.no_grad():
                u32 = model(f.cuda(), a.cuda()).cpu()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    u16 = model(f.cuda(), a.cuda()).float().cpu()
            assert torch.allclose(u32, out["fp32"], atol=1e-4), f"{fx} seed {seed} fp32"
            if same_gpu:
                assert torch.equal(u16, out["bf16"]), f"{fx} seed {seed} bf16 on the same GPU type"
            else:
                assert torch.allclose(u16, out["bf16"], atol=0.1), (
                    f"{fx} seed {seed} bf16 across GPU types"
                )
