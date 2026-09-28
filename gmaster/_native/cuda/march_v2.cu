// GMaster v2 latitudinal march: difference-form float32 Wigner-d recurrence, one warp per
// (order m, tile of 32*LPT rings).  Four kernels: spin-0 folded analysis / synthesis and spin-2
// analysis / synthesis, plus XLA FFI handlers so JAX can call them inside jit.
//
// Recurrence (docs/march_v2_maths.md): for the Jacobi row v_n = (c1 x + c0) v_{n-1} - cb v_{n-2},
// northern lanes march (v, D = v_n - v_{n-1}) as
//     C = c1 (x - 1) + (c1 - 1 - cb + c0),   D <- cb D + C v,   v <- v + D,
// southern lanes march the reflected row vt_n = (-1)^n v_n with x -> |x|, c0 -> -c0 (same code,
// table E- instead of E+).  The lane coordinate |x| - 1 and the coefficients c1, E are two-word
// (hi, lo) and C is formed with one dominant FMA rounding plus a small correction, so the phase
// error is the random-walk floor sqrt(n) u rather than the n u drift of a single-word C.
// Every lane carries an int32 binade ex with v normalised every UNR degrees; the emit forms the
// true-magnitude float32 value v * 2^(ex + floor(log2 N(l,m))) by exponent-field assembly.
#include <cuda_runtime.h>
#include <cstdint>
#include "xla/ffi/api/ffi.h"

#ifndef UNR
#define UNR 8
#endif
#ifndef LPT_ANA
#define LPT_ANA 16
#endif
#ifndef LPT_ANA_PAIR
#define LPT_ANA_PAIR 8    // two maps hold twice the rhs in registers
#endif
#ifndef LPT_SYN
#define LPT_SYN 8
#endif
#define HEMI_BLOCK 256   // hemi[] entries per 256 lanes
#define KT 12          // floats per (row, ell) table entry
#define EX_PAD (-10000000)

namespace ffi = xla::ffi;

__device__ __forceinline__ float pow2e(int e) {          // 2^(e-127) for e in [0, 254]; 0 for e = 0
    return __int_as_float(e << 23);
}

// Reduce-scatter NV per-thread values across the warp.  NV = 16: afterwards even threads t hold
// the full sum of value t/2 in acc[0].  NV = 32: thread t holds the sum of value t in acc[0].
template <int NV>
__device__ __forceinline__ void reduce_scatter(float* acc) {
    int n = NV;
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        if (n > 1) {
            const int half = n >> 1;
            const bool hi = (threadIdx.x & off) != 0;
            #pragma unroll
            for (int i = 0; i < NV / 2; ++i) {
                if (i < half) {
                    float send = hi ? acc[i] : acc[i + half];
                    float keep = hi ? acc[i + half] : acc[i];
                    float recv = __shfl_xor_sync(0xffffffffu, send, off);
                    acc[i] = keep + recv;
                }
            }
            n = half;
        } else {
            acc[0] += __shfl_xor_sync(0xffffffffu, acc[0], off);
        }
    }
}

struct Tab { float c1h, c1l, cb, mant, eph, epl, emh, eml; int lexp; };

__device__ __forceinline__ Tab load_tab(const float* __restrict__ t, bool south) {
    const float4 a = *reinterpret_cast<const float4*>(t);
    const float4 b = *reinterpret_cast<const float4*>(t + 4);
    const float4 c = *reinterpret_cast<const float4*>(t + 8);
    Tab r;
    r.c1h = a.x; r.c1l = a.y; r.cb = a.z; r.mant = a.w;
    r.eph = south ? b.z : b.x; r.epl = south ? b.w : b.y;
    r.emh = 0.f; r.eml = 0.f;
    r.lexp = __float_as_int(c.x);
    return r;
}

// one recurrence step for LPT lanes
template <int LPT>
__device__ __forceinline__ void step(const Tab& tb, const float* xh, const float* xl, float* v, float* D) {
    #pragma unroll
    for (int j = 0; j < LPT; ++j) {
        float cbig = fmaf(tb.c1h, xh[j], tb.eph);
        float s = fmaf(tb.c1l, xh[j], tb.epl);
        s = fmaf(tb.c1h, xl[j], s);
        float C = cbig + s;
        D[j] = fmaf(tb.cb, D[j], C * v[j]);
        v[j] = v[j] + D[j];
    }
}

template <int LPT>
__device__ __forceinline__ void normalise(float* v, float* D, int* ex) {
    #pragma unroll
    for (int j = 0; j < LPT; ++j) {
        float big = fmaxf(fabsf(v[j]), fabsf(D[j]));
        int eb = (__float_as_int(big) >> 23) & 0xff;
        eb = max(eb, 1);
        float sc = __int_as_float((254 - eb) << 23);
        v[j] *= sc; D[j] *= sc; ex[j] += eb - 127;
    }
}

// per-block lane scale 2^(ex + lexp_base - 127): the block's values are v * sc * u(ell) with the
// uniform u(ell) = 2^(log2 N(ell) - floor(log2 N(base))) in (2^-53, 2) folded into the tables.
__device__ __forceinline__ float blockscale(int ex, int lexp_base) {
    int e = min(max(ex + lexp_base, 0), 254);
    return pow2e(e);
}

// ----------------------------------------------------------------------------- shared prologue
template <int LPT> struct Lane {
    float v[LPT], D[LPT], xh[LPT], xl[LPT];
    int ex[LPT];
};

// seeds: v_0 = man (binade ex0), v_1 = v_0 p1, p1 = ((a+b+2)/2) x + (a-b)/2 ; with the reflected
// row for the south the state is vt_1 = sigma v_1 and Dh_1 = vt_1 - vt_0 = man (sigma p1 - 1),
// sigma p1 - 1 = ((a+b+2)/2)(|x|-1) + [(a+b)/2 + sigma (a-b)/2].
template <int SPIN, int LPT>
__device__ __forceinline__ void seed(Lane<LPT>& ln, const float* __restrict__ xs_hi, const float* __restrict__ xs_lo,
                                     const float* __restrict__ man, const int* __restrict__ ex0,
                                     int lane0, int row, int npad, int m, bool south) {
    const float a = (float)(m + SPIN), b = fabsf((float)(m - SPIN));
    const float half = 0.5f * (a + b + 2.f);
    const float cst = south ? b : a;          // (a+b)/2 + sigma (a-b)/2
    #pragma unroll
    for (int j = 0; j < LPT; ++j) {
        const int lane = lane0 + j * 32;
        ln.xh[j] = xs_hi[lane]; ln.xl[j] = xs_lo[lane];
        const float mn = man[(size_t)row * npad + lane];
        ln.ex[j] = ex0[(size_t)row * npad + lane];
        ln.v[j] = mn;
        const float p1m1 = fmaf(half, ln.xh[j], fmaf(half, ln.xl[j], cst));
        ln.D[j] = mn * p1m1;
    }
}

__device__ __forceinline__ float warp_max(float x) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, o));
    return x;
}

// ============================================================================ analysis kernels
// NCH = 2 (spin 0: rhs channels [even re, even im, odd re, odd im], parity-selected per degree)
// NCH = 4 (spin 2: rhs channels [dir re, dir im, mir re, mir im], all four every degree)
// NCH = output channels per map (2 for spin 0: the parity-selected pair; 4 for spin 2), NM = maps
// marched together.  The recurrence is a function of (m, ell, theta) alone, so extra maps join the
// emit as extra rhs channels and pay one FMA per (triple, channel) instead of a second march: at
// Nside 2048 spin 0 a field and its mask are two analyses of the same rows, 190 ms of march per
// `NmtField` when launched separately.
template <int SPIN, int NCH, int NM>
__global__ void __launch_bounds__(128)
march_analysis(const float* __restrict__ xs_hi, const float* __restrict__ xs_lo,
               const float* __restrict__ man, const int* __restrict__ ex0,
               const float* __restrict__ tab, const float* __restrict__ rhs,
               const float* __restrict__ mlim, const int* __restrict__ hemi,
               float* __restrict__ out, int m0, int L, int Lp, int Lm, int npad, int ntile)
{
    constexpr int LPT = (NM > 1) ? LPT_ANA_PAIR : LPT_ANA;
    constexpr int TILE = 32 * LPT;
    constexpr int NC = NCH * NM;            // output channels per degree
    constexpr int NVAL = UNR * NC;          // values reduced per unrolled block
    constexpr int RSTRIDE = 32 / NVAL;      // lanes per reduced value after the reduce-scatter
    const int warp = threadIdx.x >> 5, t = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + warp;
    const int m = m0 + row;
    const int tile = blockIdx.y;
    const int lane0 = tile * TILE + t;
    const bool south = hemi[tile * (TILE / HEMI_BLOCK)] != 0;
    float* orow = out + ((size_t)row * ntile + tile) * (size_t)Lm * NC;
    const int M = max(m, SPIN);
    const int head = M - m0;

    float ml = 0.f;
    #pragma unroll
    for (int j = 0; j < LPT; ++j) ml = fmaxf(ml, mlim[lane0 + j * 32]);
    ml = warp_max(ml);
    if (m >= L || (float)m > ml) {
        for (int i = t; i < Lm * NC; i += 32) orow[i] = 0.f;
        return;
    }
    for (int i = t; i < head * NC; i += 32) orow[i] = 0.f;

    Lane<LPT> ln;
    seed<SPIN, LPT>(ln, xs_hi, xs_lo, man, ex0, lane0, row, npad, m, south);
    float r[NM][4][LPT];
    #pragma unroll
    for (int n = 0; n < NM; ++n)
        #pragma unroll
        for (int j = 0; j < LPT; ++j) {
            const float4 v = *reinterpret_cast<const float4*>(
                rhs + (((size_t)row * npad + lane0 + j * 32) * NM + n) * 4);
            r[n][0][j] = v.x; r[n][1][j] = v.y; r[n][2][j] = v.z; r[n][3][j] = v.w;
        }
    const float* trow = tab + (size_t)row * Lp * KT;
    const float oddsign = south ? -1.f : 1.f;      // reflected row carries (-1)^n

    // channel `c` of map `n` at parity `p` reads rhs channel `chan_of(c, p)`
    auto emit_into = [&](float* acc, int off, float val, int j, int parity) {
        #pragma unroll
        for (int n = 0; n < NM; ++n) {
            if (NCH == 2) {
                const int b = parity ? 2 : 0;
                acc[off + 2 * n]     = fmaf(val, r[n][b][j],     acc[off + 2 * n]);
                acc[off + 2 * n + 1] = fmaf(val, r[n][b + 1][j], acc[off + 2 * n + 1]);
            } else {
                #pragma unroll
                for (int c = 0; c < 4; ++c)
                    acc[off + 4 * n + c] = fmaf(val, r[n][c][j], acc[off + 4 * n + c]);
            }
        }
    };

    // degrees M (n = 0) and M + 1 (n = 1)
    {
        float acc[2 * NC];
        #pragma unroll
        for (int i = 0; i < 2 * NC; ++i) acc[i] = 0.f;
        const int le0 = __float_as_int(trow[(size_t)M * KT + 8]);
        #pragma unroll
        for (int j = 0; j < LPT; ++j) {
            const float sc = blockscale(ln.ex[j], le0);
            const float val0 = ln.v[j] * sc;
            const float v1 = ln.v[j] + ln.D[j];
            const float val1 = v1 * sc;
            emit_into(acc, 0, val0, j, 0);
            emit_into(acc, NC, val1, j, 1);
            ln.v[j] = v1;
        }
        #pragma unroll
        for (int i = 0; i < 2 * NC; ++i) {
            float a = acc[i];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
            acc[i] = a;
        }
        if (t == 0) {
            const float mt0 = trow[(size_t)M * KT + 3];
            const float mt1 = trow[(size_t)(M + 1) * KT + 3] * oddsign;
            #pragma unroll
            for (int c = 0; c < NC; ++c) orow[(size_t)head * NC + c] = acc[c] * mt0;
            if (M + 1 < L) {
                #pragma unroll
                for (int c = 0; c < NC; ++c)
                    orow[(size_t)(head + 1) * NC + c] = acc[NC + c] * mt1;
            }
        }
    }
    const int nblk = (L - (M + 2) + UNR - 1) / UNR;
    #pragma unroll 1
    for (int it = 0; it < nblk; ++it) {
        const int base = M + 2 + it * UNR;
        float acc[NVAL];
        #pragma unroll
        for (int i = 0; i < NVAL; ++i) acc[i] = 0.f;
        float sc[LPT];
        {
            const int leb = __float_as_int(trow[(size_t)base * KT + 8]);
            #pragma unroll
            for (int j = 0; j < LPT; ++j) sc[j] = blockscale(ln.ex[j], leb);
        }
        #pragma unroll
        for (int k = 0; k < UNR; ++k) {
            const Tab tb = load_tab(trow + (size_t)(base + k) * KT, south);
            step<LPT>(tb, ln.xh, ln.xl, ln.v, ln.D);
            #pragma unroll
            for (int j = 0; j < LPT; ++j) emit_into(acc, k * NC, ln.v[j] * sc[j], j, k & 1);
        }
        normalise<LPT>(ln.v, ln.D, ln.ex);
        reduce_scatter<NVAL>(acc);
        if ((t % RSTRIDE) == 0) {
            const int q = t / RSTRIDE;
            const int k = q / NC, c = q % NC;
            const int ell = base + k;
            if (ell < L) {
                const float mt = trow[(size_t)ell * KT + 3] * ((k & 1) ? oddsign : 1.f);
                orow[(size_t)(ell - m0) * NC + c] = acc[0] * mt;
            }
        }
    }
}

// ============================================================================ synthesis kernels
// coef (mb, Lp, 4): spin 0 uses [re, im, -, -]; spin 2 [a+ re, a+ im, a- re, a- im].
// out (mb, npad, 4): spin 0 [north re, north im, south re, south im] (rings theta, pi - theta);
// spin 2 [f+(theta) re, im, f-(pi - theta) re, im] with the (-1)^(l+s) mirror sign applied here.
template <int SPIN, int NM>
__global__ void __launch_bounds__(128)
march_synthesis(const float* __restrict__ xs_hi, const float* __restrict__ xs_lo,
                const float* __restrict__ man, const int* __restrict__ ex0,
                const float* __restrict__ tab, const float* __restrict__ coef,
                const float* __restrict__ mlim, const int* __restrict__ hemi,
                float* __restrict__ out, int m0, int L, int Lp, int npad, int ntile)
{
    const int warp = threadIdx.x >> 5, t = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + warp;
    const int m = m0 + row;
    const int tile = blockIdx.y;
    constexpr int LPT = LPT_SYN; constexpr int TILE = 32 * LPT;
    constexpr int NOUT = 4 * NM;   // (theta, pi - theta) x (re, im) per map
    const int lane0 = tile * TILE + t;
    const bool south = hemi[tile * (TILE / HEMI_BLOCK)] != 0;
    const int M = max(m, SPIN);
    float* orow = out + (size_t)row * npad * NOUT;

    float ml = 0.f;
    #pragma unroll
    for (int j = 0; j < LPT; ++j) ml = fmaxf(ml, mlim[lane0 + j * 32]);
    ml = warp_max(ml);
    if (m >= L || (float)m > ml) {
        #pragma unroll
        for (int j = 0; j < LPT; ++j)
            #pragma unroll
            for (int n = 0; n < NM; ++n)
                *reinterpret_cast<float4*>(orow + ((size_t)(lane0 + j * 32) * NM + n) * 4) =
                    make_float4(0.f, 0.f, 0.f, 0.f);
        return;
    }
    Lane<LPT> ln;
    seed<SPIN, LPT>(ln, xs_hi, xs_lo, man, ex0, lane0, row, npad, m, south);
    const float* trow = tab + (size_t)row * Lp * KT;
    const float* crow = coef + (size_t)row * Lp * 4;
    // accumulators by parity of n: [even re, even im, odd re, odd im] for channel 1 (and 2 for spin 2)
    float ae_r[LPT], ae_i[LPT], ao_r[LPT], ao_i[LPT];
    float be_r[LPT], be_i[LPT], bo_r[LPT], bo_i[LPT];
    #pragma unroll
    for (int j = 0; j < LPT; ++j) { ae_r[j] = ae_i[j] = ao_r[j] = ao_i[j] = 0.f; be_r[j] = be_i[j] = bo_r[j] = bo_i[j] = 0.f; }

    // The `b` accumulators carry the second helicity at spin 2 and the second map at spin 0,
    // where they are otherwise idle -- so a paired spin-0 synthesis costs no extra registers.
    auto accum = [&](int k, float val, int j, const float4& c) {
        if (k & 1) {
            ao_r[j] = fmaf(val, c.x, ao_r[j]); ao_i[j] = fmaf(val, c.y, ao_i[j]);
            if (SPIN || NM > 1) { bo_r[j] = fmaf(val, c.z, bo_r[j]); bo_i[j] = fmaf(val, c.w, bo_i[j]); }
        } else {
            ae_r[j] = fmaf(val, c.x, ae_r[j]); ae_i[j] = fmaf(val, c.y, ae_i[j]);
            if (SPIN || NM > 1) { be_r[j] = fmaf(val, c.z, be_r[j]); be_i[j] = fmaf(val, c.w, be_i[j]); }
        }
    };
    {
        const float mt0 = trow[(size_t)M * KT + 3], mt1 = trow[(size_t)(M + 1) * KT + 3];
        const int le0 = __float_as_int(trow[(size_t)M * KT + 8]);
        float4 c0 = *reinterpret_cast<const float4*>(crow + (size_t)M * 4);
        float4 c1 = (M + 1 < L) ? *reinterpret_cast<const float4*>(crow + (size_t)(M + 1) * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
        c0.x *= mt0; c0.y *= mt0; c0.z *= mt0; c0.w *= mt0;
        c1.x *= mt1; c1.y *= mt1; c1.z *= mt1; c1.w *= mt1;
        #pragma unroll
        for (int j = 0; j < LPT; ++j) {
            const float sc = blockscale(ln.ex[j], le0);
            const float val0 = ln.v[j] * sc;
            const float v1 = ln.v[j] + ln.D[j];
            const float val1 = v1 * sc;
            accum(0, val0, j, c0);
            accum(1, val1, j, c1);
            ln.v[j] = v1;
        }
    }
    const int nblk = (L - (M + 2) + UNR - 1) / UNR;
    #pragma unroll 1
    for (int it = 0; it < nblk; ++it) {
        const int base = M + 2 + it * UNR;
        float sc[LPT];
        {
            const int leb = __float_as_int(trow[(size_t)base * KT + 8]);
            #pragma unroll
            for (int j = 0; j < LPT; ++j) sc[j] = blockscale(ln.ex[j], leb);
        }
        #pragma unroll
        for (int k = 0; k < UNR; ++k) {
            const int ell = base + k;
            const Tab tb = load_tab(trow + (size_t)ell * KT, south);
            float4 c = *reinterpret_cast<const float4*>(crow + (size_t)ell * 4);   // coef padded to Lp
            c.x *= tb.mant; c.y *= tb.mant; c.z *= tb.mant; c.w *= tb.mant;
            step<LPT>(tb, ln.xh, ln.xl, ln.v, ln.D);
            #pragma unroll
            for (int j = 0; j < LPT; ++j) {
                const float val = ln.v[j] * sc[j];
                accum(k, val, j, c);
            }
        }
        normalise<LPT>(ln.v, ln.D, ln.ex);
    }
    const float so = south ? -1.f : 1.f;                       // (-1)^n of the reflected row
    const float sm = ((M + SPIN) & 1) ? -1.f : 1.f;            // (-1)^(l+s) at n = 0
    #pragma unroll
    for (int j = 0; j < LPT; ++j) {
        float4 o;
        if (SPIN == 0) {
            o.x = ae_r[j] + ao_r[j]; o.y = ae_i[j] + ao_i[j];   // f(theta)
            o.z = ae_r[j] - ao_r[j]; o.w = ae_i[j] - ao_i[j];   // f(pi - theta): (-1)^(l+m) = (-1)^n
        } else {
            o.x = ae_r[j] + so * ao_r[j]; o.y = ae_i[j] + so * ao_i[j];            // f+(theta)
            o.z = sm * (be_r[j] - so * bo_r[j]); o.w = sm * (be_i[j] - so * bo_i[j]);   // f-(pi - theta)
        }
        *reinterpret_cast<float4*>(orow + ((size_t)(lane0 + j * 32) * NM) * 4) = o;
        if (SPIN == 0 && NM > 1) {
            float4 o2;
            o2.x = be_r[j] + bo_r[j]; o2.y = be_i[j] + bo_i[j];
            o2.z = be_r[j] - bo_r[j]; o2.w = be_i[j] - bo_i[j];
            *reinterpret_cast<float4*>(orow + ((size_t)(lane0 + j * 32) * NM + 1) * 4) = o2;
        }
    }
}

// ======================================================================= ring-Fourier residual
// A HEALPix ring's forward FFT of its inverse FFT is the n_phi-periodic fold of the Hermitian
// spectrum, so the Richardson residual FFT(map - IFFT(F)) never needs either FFT:
//   R[r, k] = n_phi(r) * sum_{m = k mod n_phi(r), |m| < L} Ft_r[m] - Mf[r, k],
//   Ft_r[m] = F[r, m] (m > 0), Re F[r, 0] (m = 0), conj F[r, -m] (m < 0),
// with Mf the map's own ring spectrum (the analysis FFT the first pass already took).  One thread
// per (ring, residue); the fold sums in float64 before the difference is rounded.
// Compensated float32 accumulation (Knuth two-sum): the folds used to sum in float64, which on this
// card's 1/64-rate fp64 pipe made them compute-bound at 19 % of DRAM bandwidth (2.4 ms per Nside 2048
// call against a ~1 ms floor).  The inputs are complex64, so a float32 residual carries the data's own
// absolute error; only the long polar-cap sums (up to L / n_phi terms) need the compensation.
__device__ __forceinline__ void fold_acc(float& s, float& c, float x) {
    const float t = s + x, bp = t - s;
    c += (s - (t - bp)) + (x - bp);
    s = t;
}

__global__ void ring_fold_residual(const float2* __restrict__ F, const float2* __restrict__ Mf,
                                   const int* __restrict__ nphi, float2* __restrict__ R, int L)
{
    const int r = blockIdx.x;
    const int n = nphi[r];
    const float2* f = F + (size_t)r * L;
    const float2* mf = Mf + (size_t)r * L;
    float2* out = R + (size_t)r * L;
    const int nres = min(n, L);
    if (n >= L) {
        // Belt rings (n_phi >= L): each output k has at most two inputs, order k and order
        // -(n - k); straight-line and unrolled so the loads overlap (the loop form below ran
        // latency-bound at a third of DRAM bandwidth, ncu).
        #pragma unroll 4
        for (int k = threadIdx.x; k < L; k += blockDim.x) {
            const float2 v = f[k];
            const int mneg = n - k;
            const float2 w = mneg < L ? f[mneg] : make_float2(0.f, 0.f);
            const float2 q = mf[k];
            const float sy = (k ? v.y : 0.f) - w.y;
            out[k] = make_float2((float)n * (v.x + w.x) - q.x, (float)n * sy - q.y);
        }
        return;
    }
    for (int rho = threadIdx.x; rho < nres; rho += blockDim.x) {
        float sx = 0.f, sy = 0.f, cx = 0.f, cy = 0.f;
        #pragma unroll 8
        for (int m = rho; m < L; m += n) {
            const float2 v = f[m];
            fold_acc(sx, cx, v.x);
            if (m) fold_acc(sy, cy, v.y);
        }
        #pragma unroll 8
        for (int m = n - rho; m < L; m += n) {        // negative orders -m = rho - t n
            const float2 v = f[m];
            fold_acc(sx, cx, v.x);
            fold_acc(sy, cy, -v.y);
        }
        const float fx = (float)n * (sx + cx), fy = (float)n * (sy + cy);
        for (int k = rho; k < L; k += n) {
            const float2 q = mf[k];
            out[k] = make_float2(fx - q.x, fy - q.y);
        }
    }
}

// Complex (non-Hermitian) version for spin fields: centred spectra, column j <-> order j - (L-1),
// m in [-(L-1), L-1].  R[r, k] = n_phi sum_{m = k mod n_phi} F[r, m] - Mf[r, k].
__global__ void ring_fold_residual_c(const float2* __restrict__ F, const float2* __restrict__ Mf,
                                     const int* __restrict__ nphi, float2* __restrict__ R, int L)
{
    const int r = blockIdx.x, n = nphi[r], W = 2 * L - 1, lo = -(L - 1);
    const float2* f = F + (size_t)r * W;
    const float2* mf = Mf + (size_t)r * W;
    float2* out = R + (size_t)r * W;
    const int nres = min(n, W);
    for (int rho = threadIdx.x; rho < nres; rho += blockDim.x) {
        // first order >= lo in residue class rho (mod n)
        const int m0 = lo + (((rho - lo) % n) + n) % n;
        float sx = 0.f, sy = 0.f, cx = 0.f, cy = 0.f;
        for (int m = m0; m <= L - 1; m += n) { const float2 v = f[m - lo]; fold_acc(sx, cx, v.x); fold_acc(sy, cy, v.y); }
        const float fx = (float)n * (sx + cx), fy = (float)n * (sy + cy);
        for (int m = m0; m <= L - 1; m += n) {
            const float2 q = mf[m - lo];
            out[m - lo] = make_float2(fx - q.x, fy - q.y);
        }
    }
}

// ================================================================================ FFI handlers
template <int SPIN, int NCH, int NM>
ffi::Error AnalysisImpl(cudaStream_t stream,
                        ffi::Buffer<ffi::F32> xs_hi, ffi::Buffer<ffi::F32> xs_lo,
                        ffi::Buffer<ffi::F32> man, ffi::Buffer<ffi::S32> ex0,
                        ffi::Buffer<ffi::F32> tab, ffi::Buffer<ffi::F32> rhs,
                        ffi::Buffer<ffi::F32> mlim, ffi::Buffer<ffi::S32> hemi,
                        ffi::ResultBuffer<ffi::F32> out, int64_t m0, int64_t L)
{
    const auto od = out->dimensions();          // (mb, ntile, Lm, NCH)
    const int mb = od[0], ntile = od[1], Lm = od[2];
    const int npad = xs_hi.dimensions()[0];
    const int Lp = tab.dimensions()[1];
    constexpr int LPT = (NM > 1) ? LPT_ANA_PAIR : LPT_ANA;
    if (npad != ntile * 32 * LPT || mb % 4) return ffi::Error::InvalidArgument("gm_march: bad analysis layout");
    if (od[3] != NCH * NM) return ffi::Error::InvalidArgument("gm_march: bad analysis channels");
    dim3 grid(mb / 4, ntile), block(128);
    march_analysis<SPIN, NCH, NM><<<grid, block, 0, stream>>>(
        xs_hi.typed_data(), xs_lo.typed_data(), man.typed_data(), ex0.typed_data(), tab.typed_data(),
        rhs.typed_data(), mlim.typed_data(), hemi.typed_data(), out->typed_data(),
        (int)m0, (int)L, Lp, Lm, npad, ntile);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(err));
    return ffi::Error::Success();
}

template <int SPIN, int NM>
ffi::Error SynthesisImpl(cudaStream_t stream,
                         ffi::Buffer<ffi::F32> xs_hi, ffi::Buffer<ffi::F32> xs_lo,
                         ffi::Buffer<ffi::F32> man, ffi::Buffer<ffi::S32> ex0,
                         ffi::Buffer<ffi::F32> tab, ffi::Buffer<ffi::F32> coef,
                         ffi::Buffer<ffi::F32> mlim, ffi::Buffer<ffi::S32> hemi,
                         ffi::ResultBuffer<ffi::F32> out, int64_t m0, int64_t L)
{
    const auto od = out->dimensions();          // (mb, npad, 4)
    const int mb = od[0], npad = od[1];
    const int Lp = tab.dimensions()[1];
    const int ntile = npad / (32 * LPT_SYN);
    if (npad % (32 * LPT_SYN) || mb % 4) return ffi::Error::InvalidArgument("gm_march: bad synthesis layout");
    if (od[2] != 4 * NM) return ffi::Error::InvalidArgument("gm_march: bad synthesis channels");
    dim3 grid(mb / 4, ntile), block(128);
    march_synthesis<SPIN, NM><<<grid, block, 0, stream>>>(
        xs_hi.typed_data(), xs_lo.typed_data(), man.typed_data(), ex0.typed_data(), tab.typed_data(),
        coef.typed_data(), mlim.typed_data(), hemi.typed_data(), out->typed_data(),
        (int)m0, (int)L, Lp, npad, ntile);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(err));
    return ffi::Error::Success();
}

#define ANA_BIND ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>() \
    .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>() \
    .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>() \
    .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::S32>>().Ret<ffi::Buffer<ffi::F32>>() \
    .Attr<int64_t>("m0").Attr<int64_t>("L")

XLA_FFI_DEFINE_HANDLER_SYMBOL(gm_march_ana_s0, (AnalysisImpl<0, 2, 1>), ANA_BIND);
XLA_FFI_DEFINE_HANDLER_SYMBOL(gm_march_ana_s0_pair, (AnalysisImpl<0, 2, 2>), ANA_BIND);
XLA_FFI_DEFINE_HANDLER_SYMBOL(gm_march_ana_s2, (AnalysisImpl<2, 4, 1>), ANA_BIND);
XLA_FFI_DEFINE_HANDLER_SYMBOL(gm_march_syn_s0, (SynthesisImpl<0, 1>), ANA_BIND);
XLA_FFI_DEFINE_HANDLER_SYMBOL(gm_march_syn_s0_pair, (SynthesisImpl<0, 2>), ANA_BIND);
XLA_FFI_DEFINE_HANDLER_SYMBOL(gm_march_syn_s2, (SynthesisImpl<2, 1>), ANA_BIND);

ffi::Error RingFoldImpl(cudaStream_t stream, ffi::Buffer<ffi::C64> F, ffi::Buffer<ffi::C64> Mf,
                        ffi::Buffer<ffi::S32> nphi, ffi::ResultBuffer<ffi::C64> R)
{
    const auto fd = F.dimensions();              // (nring, L)
    const int nring = fd[0], L = fd[1];
    if (Mf.dimensions()[0] != nring || Mf.dimensions()[1] != L || nphi.dimensions()[0] != nring)
        return ffi::Error::InvalidArgument("gm_ring_fold: shape mismatch");
    ring_fold_residual<<<nring, 256, 0, stream>>>(
        reinterpret_cast<const float2*>(F.typed_data()), reinterpret_cast<const float2*>(Mf.typed_data()),
        nphi.typed_data(), reinterpret_cast<float2*>(R->typed_data()), L);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(err));
    return ffi::Error::Success();
}

ffi::Error RingFoldCImpl(cudaStream_t stream, ffi::Buffer<ffi::C64> F, ffi::Buffer<ffi::C64> Mf,
                         ffi::Buffer<ffi::S32> nphi, ffi::ResultBuffer<ffi::C64> R)
{
    const auto fd = F.dimensions();              // (nring, 2L - 1)
    const int nring = fd[0], L = (fd[1] + 1) / 2;
    ring_fold_residual_c<<<nring, 256, 0, stream>>>(
        reinterpret_cast<const float2*>(F.typed_data()), reinterpret_cast<const float2*>(Mf.typed_data()),
        nphi.typed_data(), reinterpret_cast<float2*>(R->typed_data()), L);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(gm_ring_fold_c, RingFoldCImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::C64>>().Arg<ffi::Buffer<ffi::C64>>().Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::C64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(gm_ring_fold, RingFoldImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::C64>>().Arg<ffi::Buffer<ffi::C64>>().Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::C64>>());
