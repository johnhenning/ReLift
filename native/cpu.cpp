// Optional inference data movement helpers. No Python objects are touched while
// the GIL is released; array owners remain alive in the calling frame.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <c10/util/Half.h>
#include <cstring>
#include <stdexcept>
namespace py = pybind11;

static float read_float(const char* p, ssize_t size) {
    if (size == 4) { float v; std::memcpy(&v, p, 4); return v; }
    c10::Half v; std::memcpy(&v, p, 2); return static_cast<float>(v);
}

void gather(py::array tracks, py::array visibility,
            py::array_t<int64_t, py::array::c_style> ids,
            py::array_t<float, py::array::c_style> coords,
            py::array_t<float, py::array::c_style> vis) {
    auto a = tracks.request(), b = visibility.request(), index = ids.request();
    auto c = coords.request(true), v = vis.request(true);
    auto supported = [](const py::array& x) {
        auto name = py::str(x.dtype()).cast<std::string>();
        return name == "float16" || name == "float32";
    };
    if (!supported(tracks) || !supported(visibility) || a.ndim != 4 || b.ndim != 3 ||
        a.shape[3] != 2 || index.ndim != 1 || c.ndim != 3 || v.ndim != 2)
        throw py::value_error("gather requires float16/float32 tracks/visibility and contiguous float32 destinations");
    ssize_t T = a.shape[0], H = a.shape[1], W = a.shape[2], N = index.size;
    if (b.shape != std::vector<ssize_t>{T,H,W} || c.shape != std::vector<ssize_t>{T,N,2} ||
        v.shape != std::vector<ssize_t>{T,N}) throw py::value_error("gather shapes differ");
    auto ix = static_cast<int64_t*>(index.ptr);
    for (ssize_t n=0; n<N; ++n) if (ix[n]<0 || ix[n]>=H*W) throw py::index_error("pixel ID outside grid");
    if (tracks.is(coords) || visibility.is(vis)) throw py::value_error("input/output must not alias");
    auto src = static_cast<const char*>(a.ptr), visibility_src = static_cast<const char*>(b.ptr);
    auto dst = static_cast<float*>(c.ptr), vd = static_cast<float*>(v.ptr);
    py::gil_scoped_release unlock;
    for (ssize_t t=0; t<T; ++t) for (ssize_t n=0; n<N; ++n) {
        auto row = ix[n]/W, col = ix[n]%W;
        auto p = src+t*a.strides[0]+row*a.strides[1]+col*a.strides[2];
        dst[(t*N+n)*2] = read_float(p,a.itemsize);
        dst[(t*N+n)*2+1] = read_float(p+a.strides[3],a.itemsize);
        vd[t*N+n] = read_float(visibility_src+t*b.strides[0]+row*b.strides[1]+col*b.strides[2],b.itemsize);
    }
}

PYBIND11_MODULE(_relift_cpu, module) {
    module.def("gather", &gather, py::arg("tracks").noconvert(), py::arg("visibility").noconvert(),
               py::arg("ids").noconvert(), py::arg("coords").noconvert(), py::arg("vis").noconvert());
}
