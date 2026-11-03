// Inference-only ray construction, implementing the public nearest/border/
// align_corners=True contract. No fast math or implicit stream synchronization.
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <stdexcept>

__device__ float clean(float value) {
    return isnan(value) ? 0.f : value == INFINITY ? FLT_MAX : value == -INFINITY ? -FLT_MAX : value;
}

__global__ void rays_kernel(const float* coords,const float* depth,const float* K,const float* rotation,
                            float* direction,float* distance,int T,int N,int H,int W,bool rotate) {
    int64_t index = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (index>=int64_t(T)*N) return;
    int t=index/N;
    float u=coords[index*2],v=coords[index*2+1];
    // Tensor/scalar true division in the eager normalization uses a reciprocal.
    float nx=(2.f*u)*(1.f/float(W>1?W-1:1))-1.f;
    float ny=(2.f*v)*(1.f/float(H>1?H-1:1))-1.f;
    float sx=((nx+1.f)*.5f)*float(W-1);
    float sy=((ny+1.f)*.5f)*float(H-1);
    sx=fminf(float(W-1),fmaxf(sx,0.f));
    sy=fminf(float(H-1),fmaxf(sy,0.f));
    int ix=int(nearbyintf(sx)),iy=int(nearbyintf(sy));
    float z=depth[(int64_t(t)*H+iy)*W+ix];
    float xc=(u-K[t*9+2])/K[t*9],yc=(v-K[t*9+5])/K[t*9+4];
    // ATen's three-element norm reduction combines lanes as (x*x + 1) + y*y.
    // The radial-distance expression below deliberately keeps its other order.
    float length=sqrtf((xc*xc+1.f)+yc*yc);
    float denominator=length<1e-12f ? 1e-12f : length;
    float x=xc/denominator,y=yc/denominator,one=1.f/denominator;
    if (!rotate) {
        // Preserve nonfinite values until the reference rotation/nan_to_num.
        direction[index*3]=x; direction[index*3+1]=y; direction[index*3+2]=one;
    } else {
        for (int j=0;j<3;++j) {
            const float* row=rotation+t*9+j*3;
            float acc=row[0]*x;
            acc=__fmaf_rn(row[1],y,acc);
            direction[index*3+j]=clean(__fmaf_rn(row[2],one,acc));
        }
    }
    distance[index]=clean(z*sqrtf((xc*xc+yc*yc)+1.f));
}

void launch_rays(uintptr_t coords,uintptr_t depth,uintptr_t K,uintptr_t rotation,
                 uintptr_t direction,uintptr_t distance,int T,int N,int H,int W,uintptr_t stream) {
    if (T<1 || N<0 || H<1 || W<1) throw std::invalid_argument("invalid ray dimensions");
    int64_t total=int64_t(T)*N;
    if (!total) return;
    rays_kernel<<<(total+255)/256,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<float*>(coords),reinterpret_cast<float*>(depth),reinterpret_cast<float*>(K),
        reinterpret_cast<float*>(rotation),reinterpret_cast<float*>(direction),reinterpret_cast<float*>(distance),T,N,H,W,true);
    auto error=cudaGetLastError();
    if (error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

void launch_camera_rays(uintptr_t coords,uintptr_t depth,uintptr_t K,uintptr_t direction,uintptr_t distance,
                        int T,int N,int H,int W,uintptr_t stream) {
    if (T<1 || N<0 || H<1 || W<1) throw std::invalid_argument("invalid ray dimensions");
    int64_t total=int64_t(T)*N;
    if (!total) return;
    rays_kernel<<<(total+255)/256,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<float*>(coords),reinterpret_cast<float*>(depth),reinterpret_cast<float*>(K),nullptr,
        reinterpret_cast<float*>(direction),reinterpret_cast<float*>(distance),T,N,H,W,false);
    auto error=cudaGetLastError();
    if (error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
