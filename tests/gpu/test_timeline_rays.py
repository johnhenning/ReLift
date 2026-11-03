"""Check block boundaries, source layouts, depth caching and frame-only calls."""
import os
import numpy as np
import pytest
import torch

from relift.optimization import InferenceComponents
from relift.streaming import MappedObservations

pytestmark=[pytest.mark.gpu,pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")]


@pytest.mark.parametrize('cache',[0,1])
def test_full_timeline_rays_preserve_exact_values_and_rotation_layout(tmp_path,cache):
    if not os.environ.get('RELIFT_NATIVE_BUILD'): pytest.skip('explicit native build required')
    rng=np.random.default_rng(41)
    T,H,W=35,9,11
    tracks=(rng.random((T,H,W,2))*[W,H]).astype(np.float16)
    depth=(rng.random((T,H,W))+1).astype(np.float16)
    poses=np.tile(np.eye(4,dtype=np.float32),(T,1,1))
    for t in range(T):
        angle=t*.013
        poses[t,:3,:3]=[[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]]
    K=np.tile(np.array([[30.2,0,4.8],[0,29.7,3.2],[0,0,1]],np.float32),(T,1,1))
    np.savez(tmp_path/'da3_depth_data.npz',depth=depth,poses=poses,intrinsics=K)
    np.savez(tmp_path/'alltracker_tracks.npz',tracks_2d=tracks,visibility=rng.random((T,H,W)).astype(np.float16))
    components=InferenceComponents().with_native_build(os.environ['RELIFT_NATIVE_BUILD'])
    reference=MappedObservations(tmp_path,'cuda',depth_cache_mb=cache)
    actual=MappedObservations(tmp_path,'cuda',depth_cache_mb=cache,ray_backend='cuda_timeline',components=components)
    ids=np.array([0,20,20,1,H*W-1,5,19])
    for frame in (None,0,16,T-1):
        expected=reference.rays(ids,frame=frame)
        result=actual.rays(ids,frame=frame)
        for a,b in zip(expected,result):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            assert a.stride()==b.stride()
