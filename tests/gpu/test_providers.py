"""Each front end runs on a short clip and produces the documented outputs.

Needs CUDA, RELIFT_CHECKPOINTS_ROOT (weights as laid out by download_weights.py) and RELIFT_TEST_CLIP (a WorldTrack
clip file).  Only the first 10 frames are used, so this takes a minute, not an hour.  Set RELIFT_TEST_PROVIDERS to a
subset (e.g. "da3 alltracker") to run fewer.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from relift.config import REPO
from relift.data.worldtrack import load_clip, load_jpeg_frames_pillow

pytestmark = pytest.mark.gpu
T = 10


def _weights() -> Any:
    root = os.environ.get("RELIFT_CHECKPOINTS_ROOT")
    if not root or not os.environ.get("RELIFT_TEST_CLIP"):
        pytest.skip("set RELIFT_CHECKPOINTS_ROOT and RELIFT_TEST_CLIP")
    w = yaml.safe_load(open(REPO / "configs" / "weights.yaml"))
    return {
        n: os.path.join(root, v.get("dir", ""), v["file"])
        if "file" in v
        else os.path.join(root, v["dir"])
        for n, v in w.items()
    }


def _wanted(name: str) -> None:
    sel = os.environ.get("RELIFT_TEST_PROVIDERS")
    if sel and name not in sel.split():
        pytest.skip(f"{name} not selected")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_geometry_and_trackers() -> None:
    w = _weights()
    clip = load_clip(os.environ["RELIFT_TEST_CLIP"], T)
    frames = np.stack([np.array(im.convert("RGB")) for im in clip["frames"]])
    grid = None
    if not os.environ.get("RELIFT_TEST_PROVIDERS") or "da3" in os.environ["RELIFT_TEST_PROVIDERS"]:
        from relift.providers.da3 import DepthAnything3Provider

        g = DepthAnything3Provider(w["da3"]).run(frames)
        H, W = g["depth"].shape[1:]
        grid = (H, W)
        assert (
            g["depth"].shape == (T, H, W)
            and g["poses"].shape == (T, 4, 4)
            and g["intrinsics"].shape == (T, 3, 3)
        )
        assert (
            max(H, W) == 504 and g["images"].shape == (T, H, W, 3) and g["images"].dtype == np.uint8
        )
        from relift.providers.alltracker import AllTrackerProvider

        tr, vis, conf = AllTrackerProvider(w["alltracker"]).run(g["images"])
        gy, gx = np.mgrid[0:H, 0:W]
        assert (
            tr.shape == (T, H, W, 2)
            and np.abs(tr[0, ..., 0] - gx).max() < 1e-3
            and np.abs(tr[0, ..., 1] - gy).max() < 1e-3
        )
        assert vis.shape == (T, H, W) and 0 <= vis.min() and vis.max() <= 1
    H, W = grid or ((504, 504) if frames.shape[1] == frames.shape[2] else (280, 504))
    for name in ("vggto", "rc4"):
        if (
            os.environ.get("RELIFT_TEST_PROVIDERS")
            and name not in os.environ["RELIFT_TEST_PROVIDERS"]
        ):
            continue
        if name == "vggto":
            from relift.providers.vggt_omega import VGGTOmegaProvider

            g = VGGTOmegaProvider(w["vggt_omega"]).run(frames, H, W)
        else:
            from relift.providers.rc4 import RC4Provider

            g = RC4Provider(w["rc4"]).run(clip, H, W)
            assert g["check_px"] < 0.5 and g["check_rel"] < 0.05
        assert (
            g["depth"].shape == (T, H, W)
            and g["poses"].shape == (T, 4, 4)
            and g["intrinsics"].shape == (T, 3, 3)
        )
        assert np.allclose(g["poses"][0], np.eye(4), atol=1e-5) if name == "rc4" else True
    if (
        not os.environ.get("RELIFT_TEST_PROVIDERS")
        or "cowtracker" in os.environ["RELIFT_TEST_PROVIDERS"]
    ):
        from relift.providers.cowtracker import CoWTrackerProvider, schema_check

        cf = load_jpeg_frames_pillow(os.environ["RELIFT_TEST_CLIP"], T)
        (tr, vis, conf), size, secs, peak = CoWTrackerProvider(w["cowtracker"]).run(
            cf, H, W, 240000
        )
        schema_check(tr, vis, conf, T, H, W)
        assert size in ((336, 672), (448, 448))
