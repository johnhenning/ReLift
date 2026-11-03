"""CUDA checks for actual composition, not only configuration parsing."""
from pathlib import Path
import os

import numpy as np
import pytest
import torch

from relift.model import Corrector
from relift.optimization import InferenceComponents, OptimizationConfig
from relift.streaming import stream_correct

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def window(root):
    root.mkdir()
    T,H,W = 9,5,7
    yy,xx = np.mgrid[:H,:W]
    coords = np.tile(np.stack([xx,yy],-1),(T,1,1,1)).astype(np.float16)
    depth = np.ones((T,H,W),np.float16)*2
    depth[0,0,0] = 0
    vis = np.ones((T,H,W),np.float16)
    vis[2:5,1,2] = .2
    poses = np.tile(np.eye(4,dtype=np.float32),(T,1,1))
    K = np.tile(np.array([[30,0,3],[0,30,2],[0,0,1]],np.float32),(T,1,1))
    np.savez(root/"da3_depth_data.npz",depth=depth,poses=poses,intrinsics=K)
    np.savez(root/"alltracker_tracks.npz",tracks_2d=coords,visibility=vis,confidence=vis)
    np.savez(root/"da3_frames.npz",images=np.full((T,H,W,3),100,np.uint8))
    return H*W


@pytest.mark.parametrize("gather", ["current","python","cpp"])
@pytest.mark.parametrize("prefetch", [False,True])
def test_combinations_preserve_outputs_and_restore_no_globals(tmp_path,gather,prefetch):
    if gather == "cpp" and not os.environ.get("RELIFT_NATIVE_BUILD"):
        pytest.skip("explicit native build required")
    count = window(tmp_path/"window")
    torch.manual_seed(7)
    model = Corrector(dim=16,depth=1,heads=2,chunk=4).eval().cuda()
    reference = stream_correct(tmp_path/"window",tmp_path/"reference",model,{},group=4,batch=2)
    settings = OptimizationConfig(gather=gather,prefetch=prefetch,depth_cache_mb=1,hash_writes=False,corrector_batch=2)
    actual = stream_correct(tmp_path/"window",tmp_path/"candidate",model,{},group=4,optimizations=settings)
    for key,value in reference.sample(np.arange(count)).items():
        np.testing.assert_array_equal(actual.sample(np.arange(count))[key],value)
    # Resume must be safe even with an explicitly selected lookahead source.
    resumed = stream_correct(tmp_path/"window",tmp_path/"candidate",model,{},group=4,optimizations=settings)
    assert resumed.manifest["state"] == "complete"


@pytest.mark.parametrize("writer",["current","cpp"])
@pytest.mark.parametrize("packing",["current","cpp"])
def test_native_components_compose_with_independent_writer(tmp_path,writer,packing):
    if not os.environ.get("RELIFT_NATIVE_BUILD"): pytest.skip("explicit native build required")
    count=window(tmp_path/"window")
    torch.manual_seed(7)
    model=Corrector(dim=16,depth=1,heads=2,chunk=4).eval().cuda()
    reference=stream_correct(tmp_path/"window",tmp_path/"reference",model,{},group=4,batch=2)
    settings=OptimizationConfig(gather="cpp",shards=packing,shard_writer=writer,rays="cuda",prefetch=True,
                                depth_cache_mb=0 if writer == "cpp" else 1,hash_writes=False,corrector_batch=2)
    actual=stream_correct(tmp_path/"window",tmp_path/"candidate",model,{},group=4,optimizations=settings)
    for key,value in reference.sample(np.arange(count)).items():
        np.testing.assert_array_equal(actual.sample(np.arange(count))[key],value)
