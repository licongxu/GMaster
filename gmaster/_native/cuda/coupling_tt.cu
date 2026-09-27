// Scalar MASTER coupling matrix via the threej_cosmo recurrence, one CUDA kernel.
//
// For each (l1, l2) the summand is a function of lower=min(l1,l2), upper=max(l1,l2)
// and offset, so the upper triangle is enough: thread (l1, l2>=l1) writes both
// M[l1,l2] = S*(2*l2+1) and M[l2,l1] = S*(2*l1+1).  Products are float32 (same as
// GMaster's fp32 coupling path); the offset sum is float64 so a row of up to lmax
// terms does not round in the accumulator.
//
// The JAX scan of padded (16, n, n) chunks materialises ~576 MiB gather temps per
// iteration at lmax 3071.  On a Colab T4 that graph is minutes; this kernel keeps
// only the g / mask_power tables (~48 KiB) and the (n, n) output.
#include <cuda_runtime.h>
#include <cstdint>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

__global__ void coupling_tt_kernel(const float* __restrict__ mask_power,
                                   const float* __restrict__ g,
                                   double* __restrict__ out,
                                   int lmax) {
    const int l2 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int l1 = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int n = lmax + 1;
    if (l1 >= n || l2 >= n || l2 < l1) {
        return;
    }
    const int two_lmax = 2 * lmax;
    double acc = 0.0;
    for (int offs = 0; offs <= l1; ++offs) {
        const int p_total = l2 + offs;
        const int d = l2 - l1;
        const int mp_idx = d + 2 * offs;
        const float num = __ldg(mask_power + (mp_idx < two_lmax ? mp_idx : two_lmax))
            * __ldg(g + (d + offs))
            * __ldg(g + offs)
            * __ldg(g + (l1 - offs));
        const float den = __ldg(g + p_total) * (2.f * (float)p_total + 1.f);
        acc += (double)(num / den);
    }
    out[(size_t)l1 * (size_t)n + (size_t)l2] = acc * (2.0 * (double)l2 + 1.0);
    if (l2 != l1) {
        out[(size_t)l2 * (size_t)n + (size_t)l1] = acc * (2.0 * (double)l1 + 1.0);
    }
}

ffi::Error CouplingTTImpl(cudaStream_t stream,
                          ffi::Buffer<ffi::F32> mask_power,
                          ffi::Buffer<ffi::F32> g,
                          ffi::ResultBuffer<ffi::F64> out,
                          int64_t lmax)
{
    if (lmax < 0) {
        return ffi::Error::InvalidArgument("gm_coupling_tt: lmax < 0");
    }
    const int n = (int)lmax + 1;
    const int glen = (int)g.dimensions()[0];
    const int mlen = (int)mask_power.dimensions()[0];
    if (glen != 2 * (int)lmax + 1 || mlen != glen) {
        return ffi::Error::InvalidArgument("gm_coupling_tt: table length");
    }
    const auto od = out->dimensions();
    if ((int)od[0] != n || (int)od[1] != n) {
        return ffi::Error::InvalidArgument("gm_coupling_tt: output shape");
    }
    dim3 block(32, 8);
    dim3 grid((n + 31) / 32, (n + 7) / 8);
    coupling_tt_kernel<<<grid, block, 0, stream>>>(
        mask_power.typed_data(), g.typed_data(), out->typed_data(), (int)lmax);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error::Internal(cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

#define TT_BIND ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>() \
    .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>() \
    .Ret<ffi::Buffer<ffi::F64>>().Attr<int64_t>("lmax")

XLA_FFI_DEFINE_HANDLER_SYMBOL(gm_coupling_tt, CouplingTTImpl, TT_BIND);
