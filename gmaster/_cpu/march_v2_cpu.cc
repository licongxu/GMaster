// CPU OpenMP port of gmaster/_cuda/march_v2.cu (same difference-form march, same tables).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include "xla/ffi/api/ffi.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef UNR
#define UNR 8
#endif
#ifndef LPT_ANA
#define LPT_ANA 16
#endif
#ifndef LPT_ANA_PAIR
#define LPT_ANA_PAIR 8
#endif
#ifndef LPT_SYN
#define LPT_SYN 8
#endif
#define HEMI_BLOCK 256
#define KT 12
#define EX_PAD (-10000000)

namespace ffi = xla::ffi;

static inline int float_as_int(float x) {
    int i;
    std::memcpy(&i, &x, 4);
    return i;
}
static inline float int_as_float(int i) {
    float x;
    std::memcpy(&x, &i, 4);
    return x;
}
static inline float pow2e(int e) { return int_as_float(e << 23); }
static inline float blockscale(int ex, int lexp_base) {
    int e = std::min(std::max(ex + lexp_base, 0), 254);
    return pow2e(e);
}

struct Tab {
    float c1h, c1l, cb, mant, eph, epl;
    int lexp;
};

static inline Tab load_tab(const float *t, bool south) {
    Tab r;
    r.c1h = t[0];
    r.c1l = t[1];
    r.cb = t[2];
    r.mant = t[3];
    r.eph = south ? t[6] : t[4];
    r.epl = south ? t[7] : t[5];
    r.lexp = float_as_int(t[8]);
    return r;
}

static inline void step_one(const Tab &tb, float xh, float xl, float &v, float &D) {
    float cbig = std::fmaf(tb.c1h, xh, tb.eph);
    float s = std::fmaf(tb.c1l, xh, tb.epl);
    s = std::fmaf(tb.c1h, xl, s);
    float C = cbig + s;
    D = std::fmaf(tb.cb, D, C * v);
    v = v + D;
}

static inline void normalise_one(float &v, float &D, int &ex) {
    float big = std::fmax(std::fabs(v), std::fabs(D));
    int eb = (float_as_int(big) >> 23) & 0xff;
    if (eb < 1)
        eb = 1;
    float sc = int_as_float((254 - eb) << 23);
    v *= sc;
    D *= sc;
    ex += eb - 127;
}

template <int SPIN, int NCH, int NM, int LPT>
static void analysis(const float *xs_hi, const float *xs_lo, const float *man, const int *ex0,
                     const float *tab, const float *rhs, const float *mlim, const int *hemi,
                     float *out, int m0, int L, int Lp, int Lm, int npad, int ntile, int mb) {
    constexpr int TILE = 32 * LPT;
    constexpr int NC = NCH * NM;
#pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int row = 0; row < mb; ++row) {
        for (int tile = 0; tile < ntile; ++tile) {
            const int m = m0 + row;
            float *orow = out + ((size_t)row * ntile + tile) * (size_t)Lm * NC;
            const int lane0 = tile * TILE;
            bool south = hemi[tile * (TILE / HEMI_BLOCK)] != 0;
            float ml = mlim[lane0];
            for (int j = 1; j < TILE; ++j)
                ml = std::fmax(ml, mlim[lane0 + j]);
            if (m >= L || (float)m > ml) {
                std::memset(orow, 0, (size_t)Lm * NC * sizeof(float));
                continue;
            }
            const int M = std::max(m, SPIN);
            const int head = M - m0;
            std::memset(orow, 0, (size_t)head * NC * sizeof(float));
            const float a = (float)(m + SPIN), b = std::fabs((float)(m - SPIN));
            const float half = 0.5f * (a + b + 2.f);
            const float cst = south ? b : a;
            const float oddsign = south ? -1.f : 1.f;
            const float *trow = tab + (size_t)row * Lp * KT;
            for (int j = 0; j < TILE; ++j) {
                const int lane = lane0 + j;
                const float xh = xs_hi[lane], xl = xs_lo[lane];
                const float mn = man[(size_t)row * npad + lane];
                int ex = ex0[(size_t)row * npad + lane];
                float v = mn;
                float D = mn * std::fmaf(half, xh, std::fmaf(half, xl, cst));
                float rch[NM][4];
                for (int n = 0; n < NM; ++n) {
                    const float *p = rhs + (((size_t)row * npad + lane) * NM + n) * 4;
                    rch[n][0] = p[0];
                    rch[n][1] = p[1];
                    rch[n][2] = p[2];
                    rch[n][3] = p[3];
                }
                auto emit = [&](int ell, float val, int parity) {
                    if (ell >= L)
                        return;
                    const float mt = trow[(size_t)ell * KT + 3] * ((parity & 1) ? oddsign : 1.f);
                    float *dst = orow + (size_t)(ell - m0) * NC;
                    for (int n = 0; n < NM; ++n) {
                        if (NCH == 2) {
                            const int b = (parity & 1) ? 2 : 0;
                            dst[2 * n] += val * mt * rch[n][b];
                            dst[2 * n + 1] += val * mt * rch[n][b + 1];
                        } else {
                            for (int c = 0; c < 4; ++c)
                                dst[4 * n + c] += val * mt * rch[n][c];
                        }
                    }
                };
                const int le0 = float_as_int(trow[(size_t)M * KT + 8]);
                const float sc0 = blockscale(ex, le0);
                emit(M, v * sc0, 0);
                v = v + D;
                emit(M + 1, v * sc0, 1);
                const int nblk = (L - (M + 2) + UNR - 1) / UNR;
                for (int it = 0; it < nblk; ++it) {
                    const int base = M + 2 + it * UNR;
                    const int leb = float_as_int(trow[(size_t)base * KT + 8]);
                    const float sc = blockscale(ex, leb);
                    for (int k = 0; k < UNR; ++k) {
                        const Tab tb = load_tab(trow + (size_t)(base + k) * KT, south);
                        step_one(tb, xh, xl, v, D);
                        emit(base + k, v * sc, k);
                    }
                    normalise_one(v, D, ex);
                }
            }
        }
    }
}

template <int SPIN, int NM, int LPT>
static void synthesis(const float *xs_hi, const float *xs_lo, const float *man, const int *ex0,
                      const float *tab, const float *coef, const float *mlim, const int *hemi,
                      float *out, int m0, int L, int Lp, int npad, int mb) {
    constexpr int TILE = 32 * LPT;
    constexpr int NOUT = 4 * NM;
    const int ntile = npad / TILE;
#pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int row = 0; row < mb; ++row) {
        for (int tile = 0; tile < ntile; ++tile) {
            const int m = m0 + row;
            const int lane0 = tile * TILE;
            bool south = hemi[tile * (TILE / HEMI_BLOCK)] != 0;
            float *orow = out + (size_t)row * npad * NOUT;
            float ml = mlim[lane0];
            for (int j = 1; j < TILE; ++j)
                ml = std::fmax(ml, mlim[lane0 + j]);
            if (m >= L || (float)m > ml) {
                for (int j = 0; j < TILE; ++j)
                    std::memset(orow + (size_t)(lane0 + j) * NOUT, 0, NOUT * sizeof(float));
                continue;
            }
            const int M = std::max(m, SPIN);
            const float a = (float)(m + SPIN), b = std::fabs((float)(m - SPIN));
            const float half = 0.5f * (a + b + 2.f);
            const float cst = south ? b : a;
            const float *trow = tab + (size_t)row * Lp * KT;
            const float *crow = coef + (size_t)row * Lp * 4;
            const float so = south ? -1.f : 1.f;
            const float sm = ((M + SPIN) & 1) ? -1.f : 1.f;
            for (int j = 0; j < TILE; ++j) {
                const int lane = lane0 + j;
                const float xh = xs_hi[lane], xl = xs_lo[lane];
                const float mn = man[(size_t)row * npad + lane];
                int ex = ex0[(size_t)row * npad + lane];
                float v = mn;
                float D = mn * std::fmaf(half, xh, std::fmaf(half, xl, cst));
                float ae_r = 0, ae_i = 0, ao_r = 0, ao_i = 0;
                float be_r = 0, be_i = 0, bo_r = 0, bo_i = 0;
                auto accum = [&](int k, float val, float cx, float cy, float cz, float cw) {
                    if (k & 1) {
                        ao_r = std::fmaf(val, cx, ao_r);
                        ao_i = std::fmaf(val, cy, ao_i);
                        if (SPIN || NM > 1) {
                            bo_r = std::fmaf(val, cz, bo_r);
                            bo_i = std::fmaf(val, cw, bo_i);
                        }
                    } else {
                        ae_r = std::fmaf(val, cx, ae_r);
                        ae_i = std::fmaf(val, cy, ae_i);
                        if (SPIN || NM > 1) {
                            be_r = std::fmaf(val, cz, be_r);
                            be_i = std::fmaf(val, cw, be_i);
                        }
                    }
                };
                const float mt0 = trow[(size_t)M * KT + 3];
                const float mt1 = trow[(size_t)(M + 1) * KT + 3];
                const int le0 = float_as_int(trow[(size_t)M * KT + 8]);
                const float *c0 = crow + (size_t)M * 4;
                float c1x = 0, c1y = 0, c1z = 0, c1w = 0;
                if (M + 1 < L) {
                    const float *c1 = crow + (size_t)(M + 1) * 4;
                    c1x = c1[0] * mt1;
                    c1y = c1[1] * mt1;
                    c1z = c1[2] * mt1;
                    c1w = c1[3] * mt1;
                }
                const float sc0 = blockscale(ex, le0);
                accum(0, v * sc0, c0[0] * mt0, c0[1] * mt0, c0[2] * mt0, c0[3] * mt0);
                v = v + D;
                accum(1, v * sc0, c1x, c1y, c1z, c1w);
                const int nblk = (L - (M + 2) + UNR - 1) / UNR;
                for (int it = 0; it < nblk; ++it) {
                    const int base = M + 2 + it * UNR;
                    const int leb = float_as_int(trow[(size_t)base * KT + 8]);
                    const float sc = blockscale(ex, leb);
                    for (int k = 0; k < UNR; ++k) {
                        const int ell = base + k;
                        const Tab tb = load_tab(trow + (size_t)ell * KT, south);
                        const float *c = crow + (size_t)ell * 4;
                        step_one(tb, xh, xl, v, D);
                        accum(k, v * sc, c[0] * tb.mant, c[1] * tb.mant, c[2] * tb.mant, c[3] * tb.mant);
                    }
                    normalise_one(v, D, ex);
                }
                float *o = orow + (size_t)lane * NOUT;
                if (SPIN == 0) {
                    o[0] = ae_r + ao_r;
                    o[1] = ae_i + ao_i;
                    o[2] = ae_r - ao_r;
                    o[3] = ae_i - ao_i;
                    if (NM > 1) {
                        o[4] = be_r + bo_r;
                        o[5] = be_i + bo_i;
                        o[6] = be_r - bo_r;
                        o[7] = be_i - bo_i;
                    }
                } else {
                    o[0] = ae_r + so * ao_r;
                    o[1] = ae_i + so * ao_i;
                    o[2] = sm * (be_r - so * bo_r);
                    o[3] = sm * (be_i - so * bo_i);
                }
            }
        }
    }
}

template <int SPIN, int NCH, int NM>
ffi::Error AnalysisImpl(ffi::Buffer<ffi::F32> xs_hi, ffi::Buffer<ffi::F32> xs_lo,
                        ffi::Buffer<ffi::F32> man, ffi::Buffer<ffi::S32> ex0,
                        ffi::Buffer<ffi::F32> tab, ffi::Buffer<ffi::F32> rhs,
                        ffi::Buffer<ffi::F32> mlim, ffi::Buffer<ffi::S32> hemi,
                        ffi::ResultBuffer<ffi::F32> out, int64_t m0, int64_t L) {
    const auto od = out->dimensions();
    const int mb = od[0], ntile = od[1], Lm = od[2];
    const int npad = xs_hi.dimensions()[0];
    const int Lp = tab.dimensions()[1];
    constexpr int LPT = (NM > 1) ? LPT_ANA_PAIR : LPT_ANA;
    if (npad != ntile * 32 * LPT || mb % 4)
        return ffi::Error::InvalidArgument("gm_march cpu: bad analysis layout");
    if (od[3] != NCH * NM)
        return ffi::Error::InvalidArgument("gm_march cpu: bad analysis channels");
    analysis<SPIN, NCH, NM, LPT>(
        xs_hi.typed_data(), xs_lo.typed_data(), man.typed_data(), ex0.typed_data(), tab.typed_data(),
        rhs.typed_data(), mlim.typed_data(), hemi.typed_data(), out->typed_data(), (int)m0, (int)L,
        Lp, Lm, npad, ntile, mb);
    return ffi::Error::Success();
}

template <int SPIN, int NM>
ffi::Error SynthesisImpl(ffi::Buffer<ffi::F32> xs_hi, ffi::Buffer<ffi::F32> xs_lo,
                         ffi::Buffer<ffi::F32> man, ffi::Buffer<ffi::S32> ex0,
                         ffi::Buffer<ffi::F32> tab, ffi::Buffer<ffi::F32> coef,
                         ffi::Buffer<ffi::F32> mlim, ffi::Buffer<ffi::S32> hemi,
                         ffi::ResultBuffer<ffi::F32> out, int64_t m0, int64_t L) {
    const auto od = out->dimensions();
    const int mb = od[0], npad = od[1];
    const int Lp = tab.dimensions()[1];
    if (npad % (32 * LPT_SYN) || mb % 4)
        return ffi::Error::InvalidArgument("gm_march cpu: bad synthesis layout");
    if (od[2] != 4 * NM)
        return ffi::Error::InvalidArgument("gm_march cpu: bad synthesis channels");
    synthesis<SPIN, NM, LPT_SYN>(
        xs_hi.typed_data(), xs_lo.typed_data(), man.typed_data(), ex0.typed_data(), tab.typed_data(),
        coef.typed_data(), mlim.typed_data(), hemi.typed_data(), out->typed_data(), (int)m0, (int)L,
        Lp, npad, mb);
    return ffi::Error::Success();
}

#define ANA_BIND                                                                                 \
    ffi::Ffi::Bind()                                                                             \
        .Arg<ffi::Buffer<ffi::F32>>()                                                            \
        .Arg<ffi::Buffer<ffi::F32>>()                                                            \
        .Arg<ffi::Buffer<ffi::F32>>()                                                            \
        .Arg<ffi::Buffer<ffi::S32>>()                                                            \
        .Arg<ffi::Buffer<ffi::F32>>()                                                            \
        .Arg<ffi::Buffer<ffi::F32>>()                                                            \
        .Arg<ffi::Buffer<ffi::F32>>()                                                            \
        .Arg<ffi::Buffer<ffi::S32>>()                                                            \
        .Ret<ffi::Buffer<ffi::F32>>()                                                            \
        .Attr<int64_t>("m0")                                                                     \
        .Attr<int64_t>("L")

XLA_FFI_DEFINE_HANDLER_SYMBOL(gm_march_ana_s0, (AnalysisImpl<0, 2, 1>), ANA_BIND);
XLA_FFI_DEFINE_HANDLER_SYMBOL(gm_march_ana_s0_pair, (AnalysisImpl<0, 2, 2>), ANA_BIND);
XLA_FFI_DEFINE_HANDLER_SYMBOL(gm_march_ana_s2, (AnalysisImpl<2, 4, 1>), ANA_BIND);
XLA_FFI_DEFINE_HANDLER_SYMBOL(gm_march_syn_s0, (SynthesisImpl<0, 1>), ANA_BIND);
XLA_FFI_DEFINE_HANDLER_SYMBOL(gm_march_syn_s0_pair, (SynthesisImpl<0, 2>), ANA_BIND);
XLA_FFI_DEFINE_HANDLER_SYMBOL(gm_march_syn_s2, (SynthesisImpl<2, 1>), ANA_BIND);
