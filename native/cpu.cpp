// Optional inference data movement helpers. No Python objects are touched while
// the GIL is released; array owners remain alive in the calling frame.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <c10/util/Half.h>
#include <cstring>
#include <stdexcept>
#include <openssl/evp.h>
#include <unistd.h>
#include <cerrno>
#include <memory>
#include <cmath>
#include <system_error>
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

py::tuple pack_finite(py::array value) {
    auto input = value.request();
    const auto kind = py::str(value.dtype()).cast<std::string>();
    if (kind != "float16" && kind != "float32" && kind != "bool")
        throw py::value_error("packer supports float16, float32 and bool");
    bool fortran = (value.flags() & py::array::f_style) && !(value.flags() & py::array::c_style);
    std::vector<ssize_t> strides(input.ndim);
    ssize_t stride = input.itemsize;
    for (ssize_t i=0; i<input.ndim; ++i) {
        ssize_t dim = fortran ? i : input.ndim-1-i;
        strides[dim] = stride; stride *= input.shape[dim];
    }
    py::array output(value.dtype(), input.shape, strides);
    auto destination = output.request(true);
    bool finite = true;
    {
        py::gil_scoped_release unlock;
        auto src = static_cast<const char*>(input.ptr);
        auto dst = static_cast<char*>(destination.ptr);
        for (ssize_t index=0; index<input.size; ++index) {
            ssize_t remainder=index, offset=0;
            for (ssize_t i=0; i<input.ndim; ++i) {
                ssize_t dim = fortran ? i : input.ndim-1-i;
                offset += (remainder % input.shape[dim])*input.strides[dim];
                remainder /= input.shape[dim];
            }
            std::memcpy(dst+index*input.itemsize, src+offset, input.itemsize);
            if (kind != "bool") finite &= std::isfinite(read_float(src+offset,input.itemsize));
        }
    }
    return py::make_tuple(output, finite);
}

// Header generation, flush/fsync, rename and manifest publication remain Python.
// Hash exactly the bytes accepted by write(2); retry short writes and EINTR.
std::string write_payload(int fd, const std::string& header, py::array value, bool hash) {
    auto input = value.request();
    if (!(value.flags() & (py::array::c_style | py::array::f_style)) || value.dtype().has_fields() || value.dtype().kind() == 'O')
        throw py::value_error("native payload requires a contiguous numeric array");
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (hash && (!context || EVP_DigestInit_ex(context.get(),EVP_sha256(),nullptr)!=1))
        throw std::runtime_error("SHA256 initialization failed");
    unsigned char digest[EVP_MAX_MD_SIZE]; unsigned int length=0;
    {
        py::gil_scoped_release unlock;
        auto write_all = [&](const char* data, size_t size) {
            while (size) {
                auto sent = ::write(fd,data,std::min(size,size_t(1<<20)));
                if (sent<0 && errno==EINTR) continue;
                if (sent<=0) throw std::system_error(sent<0 ? errno : EIO,std::generic_category(),"native NPY write");
                if (hash && EVP_DigestUpdate(context.get(),data,sent)!=1) throw std::runtime_error("SHA256 update failed");
                data+=sent; size-=sent;
            }
        };
        write_all(header.data(),header.size());
        write_all(static_cast<const char*>(input.ptr),input.size*input.itemsize);
        if (hash && EVP_DigestFinal_ex(context.get(),digest,&length)!=1) throw std::runtime_error("SHA256 finalization failed");
    }
    static const char hex[]="0123456789abcdef";
    std::string result;
    for (unsigned int i=0;i<length;++i) { result+=hex[digest[i]>>4]; result+=hex[digest[i]&15]; }
    return result;
}

PYBIND11_MODULE(_relift_cpu, module) {
    module.def("gather", &gather, py::arg("tracks").noconvert(), py::arg("visibility").noconvert(),
               py::arg("ids").noconvert(), py::arg("coords").noconvert(), py::arg("vis").noconvert());
    module.def("pack_finite", &pack_finite, py::arg("array").noconvert());
    module.def("write_payload", &write_payload, py::arg("fd"), py::arg("header"), py::arg("array").noconvert(), py::arg("hash"));
}
