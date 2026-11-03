import os

import pytest
import torch
import torch.nn.functional as F

from relift.cuda_rays import make_ray_builder,TimelineRayBuilder
from relift.native_backend import load_cpu

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")]


def reference(coords,depth,K,rotation):
    H,W=depth.shape[1:]
    norm=torch.stack([2*coords[...,0]/max(W-1,1)-1,2*coords[...,1]/max(H-1,1)-1],-1)
    z=F.grid_sample(depth[:,None],norm[:,:,None],mode="nearest",padding_mode="border",align_corners=True)[:,0,:,0]
    x=(coords[...,0]-K[:,0,2,None])/K[:,0,0,None]
    y=(coords[...,1]-K[:,1,2,None])/K[:,1,1,None]
    direction=torch.einsum("tij,tnj->tni",rotation,F.normalize(torch.stack([x,y,torch.ones_like(x)],-1),dim=-1))
    return torch.nan_to_num(direction),torch.nan_to_num(z*torch.sqrt(x*x+y*y+1))


@pytest.mark.parametrize("H,W",[(1,1),(5,7),(280,504)])
@pytest.mark.parametrize("factory",[make_ray_builder,TimelineRayBuilder])
def test_boundaries_ties_nonfinite_and_stream(H,W,factory):
    if not os.environ.get("RELIFT_NATIVE_BUILD"):
        pytest.skip("explicit CUDA build required")
    native=load_cpu()
    if not hasattr(native,"launch_rays"):
        pytest.skip("CPU-only build")
    build=factory(native)
    values=[[-1,-1],[0,0],[.5,.5],[1.5,1.5],[W-1,H-1],[W+3,H+3],
            [float("nan"),0],[float("inf"),2],[-float("inf"),2]]
    coords=torch.tensor([values]*3,dtype=torch.float32,device="cuda")
    depth=torch.arange(3*H*W,device="cuda",dtype=torch.float32).reshape(3,H,W)+1
    K=torch.eye(3,device="cuda").repeat(3,1,1); K[:,0,0]=30; K[:,1,1]=40
    rotation=torch.eye(3,device="cuda").repeat(3,1,1)
    expected=reference(coords,depth,K,rotation)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual=build(coords,depth,K,rotation)
    torch.cuda.current_stream().wait_stream(stream)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)
    with pytest.raises(ValueError):
        build(coords.half(),depth,K,rotation)
    operator=build.camera if isinstance(build,TimelineRayBuilder) else build.operator
    arguments=(depth,K) if isinstance(build,TimelineRayBuilder) else (depth,K,rotation)
    with pytest.raises(ValueError,match="float32"):
        operator(coords.half(),*arguments)
    with pytest.raises(ValueError,match="shapes"):
        operator(coords[:,:,:1],*arguments)
    # Direct operator calls must handle strides safely, rather than bypassing
    # normalization in the convenience wrapper and reading the wrong addresses.
    strided=torch.stack([coords,coords],-1)[...,0]
    if isinstance(build,TimelineRayBuilder):
        camera,distance=build.camera(strided,depth,K)
        direct=torch.nan_to_num(build.rotate(rotation,camera)),distance
    else:
        direct=build.operator(strided,depth,K,rotation)
    for a,b in zip(direct,actual): torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize("factory",[make_ray_builder,TimelineRayBuilder])
def test_direct_operator_rejects_inexact_large_pixel_bounds_without_allocation(factory):
    if not os.environ.get("RELIFT_NATIVE_BUILD"): pytest.skip("explicit CUDA build required")
    native=load_cpu()
    if not hasattr(native,"launch_camera_rays"): pytest.skip("corrected CUDA build required")
    build=factory(native)
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        coords=torch.empty((1,1,2),device="cuda")
        depth=torch.empty((1,1,2**24+4),device="cuda")
        K=torch.empty((1,3,3),device="cuda")
        with pytest.raises(ValueError,match="pixel-index"):
            if isinstance(build,TimelineRayBuilder): build.camera(coords,depth,K)
            else: build.operator(coords,depth,K,K)


def test_timeline_registered_and_eager_launches_match_with_fake_registration():
    if not os.environ.get('RELIFT_NATIVE_BUILD'): pytest.skip('explicit native build required')
    native=load_cpu()
    if not hasattr(native,'launch_camera_rays'): pytest.skip('CUDA build required')
    eager=TimelineRayBuilder(native)
    registered=TimelineRayBuilder(native,registered=True)
    coords=torch.tensor([[[.5,.5],[2.,1.],[float('nan'),0.]]],device='cuda')
    depth=torch.arange(15,device='cuda',dtype=torch.float32).reshape(1,3,5)
    K=torch.eye(3,device='cuda')[None]
    for a,b in zip(eager(coords,depth,K,K),registered(coords,depth,K,K)):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        coords=torch.empty((2,7,2),device='cuda')
        depth=torch.empty((2,9,11),device='cuda')
        K=torch.empty((2,3,3),device='cuda')
        direction,distance=registered.camera(coords,depth,K)
        assert direction.shape==(2,7,3) and distance.shape==(2,7)
