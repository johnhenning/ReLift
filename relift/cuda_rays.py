"""A registered custom operator backed by the explicitly built native module."""
import torch
import torch.nn.functional as F


def torch_ray_builder(coords,depth,K,rotation):
    H,W = depth.shape[1:]
    norm = torch.stack([2*coords[...,0]/max(W-1,1)-1,2*coords[...,1]/max(H-1,1)-1],-1)
    z = F.grid_sample(depth[:,None],norm[:,:,None],mode="nearest",padding_mode="border",align_corners=True)[:,0,:,0]
    x = (coords[...,0]-K[:,0,2,None])/K[:,0,0,None]
    y = (coords[...,1]-K[:,1,2,None])/K[:,1,1,None]
    d = F.normalize(torch.stack([x,y,torch.ones_like(x)],-1),dim=-1)
    return torch.nan_to_num(torch.einsum("tij,tnj->tni",rotation,d)),torch.nan_to_num(z*torch.sqrt(x*x+y*y+1))


def make_ray_builder(native):
    if not hasattr(native,"launch_camera_rays"):
        raise RuntimeError("Selected CUDA rays require a --cuda native build")

    # Registration is local to this factory: no global backend selection is
    # changed. The unique operator name includes the selected binary identity.
    import hashlib
    from pathlib import Path
    identity = hashlib.sha256(Path(native.__file__).read_bytes()+Path(__file__).read_bytes()).hexdigest()[:16]
    name = "relift_native::rays_"+identity
    def validate(coords,depth,K,rotation):
        values=(coords,depth,K,rotation)
        if any(v.device != coords.device or v.dtype != torch.float32 or not v.is_cuda for v in values):
            raise ValueError("CUDA rays require float32 tensors on the same CUDA device")
        if coords.ndim != 3 or coords.shape[-1] != 2 or depth.ndim != 3:
            raise ValueError("CUDA ray input shapes differ")
        T,N,_=coords.shape
        H,W=depth.shape[1:]
        if T<1 or H<1 or W<1 or max(T,N,H,W)>2**31-1 or depth.shape[0]!=T or K.shape!=(T,3,3) or rotation.shape!=(T,3,3):
            raise ValueError("CUDA ray input shapes differ")
        if any(v.requires_grad for v in values):
            raise ValueError("CUDA rays are inference-only")
    try:
        operator = getattr(torch.ops.relift_native,"rays_"+identity)
    except AttributeError:
        @torch.library.custom_op(name, mutates_args=(), device_types="cuda")
        def operator(coords: torch.Tensor, depth: torch.Tensor, K: torch.Tensor, rotation: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
            validate(coords,depth,K,rotation)
            coords,depth,K=(value.contiguous() for value in (coords,depth,K))
            T,N,_ = coords.shape
            H,W = depth.shape[1:]
            direction = torch.empty((T,N,3),dtype=coords.dtype,device=coords.device)
            distance = torch.empty((T,N),dtype=coords.dtype,device=coords.device)
            with torch.cuda.device(coords.device):
                native.launch_camera_rays(coords.data_ptr(),depth.data_ptr(),K.data_ptr(),
                                          direction.data_ptr(),distance.data_ptr(),T,N,H,W,
                                          torch.cuda.current_stream(coords.device).cuda_stream)
                # cuBLAS selects accumulation kernels by shape/layout. Preserve
                # the reference operation instead of hardcoding one FMA order.
                direction=torch.nan_to_num(torch.einsum("tij,tnj->tni",rotation,direction))
            return direction,distance

        @operator.register_fake
        def fake(coords,depth,K,rotation):
            validate(coords,depth,K,rotation)
            return coords.new_empty((*coords.shape[:2],3)),coords.new_empty(coords.shape[:2])

    def build(coords,depth,K,rotation):
        validate(coords,depth,K,rotation)
        return operator(coords,depth,K,rotation)
    build.operator=operator
    return build
