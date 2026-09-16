/* AVX-512 fp32 (v, D) difference-form Jacobi march on the CPU: spin-0 folded analysis.
 *
 * A port of `march_analysis<0, 2, 1>` in gmaster/_cuda/march_v2.cu to ducc0's loop order:
 * OpenMP over the order m, 16 northern rings per instruction, the degree ell innermost in blocks
 * of UNR = 8.  Same seed (mantissa + int32 binade), same two-word x and coefficients, same
 * per-block emit scale 2^(ex + lexp_base - 127) with u(ell) folded into a per-m table, same
 * normalisation every UNR degrees.  The southern hemisphere is folded into the right-hand side
 * (G + G' on even ell - m, G - G' on odd), as `_fold_rhs` does on the GPU.
 *
 * Unlike the GPU route nothing is cached across calls: the per-m coefficient row and the per-lane
 * seeds are computed inside the call (as ducc0 computes its own), so the wall time is the whole
 * latitudinal stage.
 *
 * Layouts (float32 unless stated; npad is a multiple of 16; lanes >= nnorth are padding):
 *   xh, xl     (npad)          |cos theta| - 1 as (hi, lo)
 *   lg         (npad) float64  log2 sin(theta/2) + log2 cos(theta/2)
 *   mlim       (npad)          ducc0's polar cutoff (-1 on padding); rows with m > max(mlim) skip
 *   rhs        (L, 4, npad)    [(G+G') re, (G+G') im, (G-G') re, (G-G') im]
 *   norm       (L)             sqrt((2 ell + 1) / 4 pi)
 *   out        (L, L, 2)       row m, degree ell: (-1)^m norm(ell) u(ell) sum_lanes val * rhs
 */
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <complex.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define V 16
#define UNR 8
#define KT 8
#define EX_PAD (-10000000)

static inline __m512 emit_scale(__m512i ex, __m512i lexp) {
    __m512i e = _mm512_add_epi32(ex, lexp);
    e = _mm512_min_epi32(_mm512_max_epi32(e, _mm512_setzero_si512()), _mm512_set1_epi32(254));
    return _mm512_castsi512_ps(_mm512_slli_epi32(e, 23));
}

static inline void normalise(__m512 *v, __m512 *D, __m512i *ex) {
    const __m512 mag = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
    __m512 big = _mm512_max_ps(_mm512_and_ps(*v, mag), _mm512_and_ps(*D, mag));
    __m512i eb = _mm512_and_si512(_mm512_srli_epi32(_mm512_castps_si512(big), 23),
                                  _mm512_set1_epi32(0xff));
    eb = _mm512_max_epi32(eb, _mm512_set1_epi32(1));
    __m512 sc = _mm512_castsi512_ps(
        _mm512_slli_epi32(_mm512_sub_epi32(_mm512_set1_epi32(254), eb), 23));
    *v = _mm512_mul_ps(*v, sc);
    *D = _mm512_mul_ps(*D, sc);
    *ex = _mm512_add_epi32(*ex, _mm512_sub_epi32(eb, _mm512_set1_epi32(127)));
}

static inline float hsum(__m512 a) { return _mm512_reduce_add_ps(a); }

/* Per-m coefficient row: tab[(ell - m) * KT + {0..6}] = c1h, c1l, cb, u, eh, el, lexp for
   ell in [m, m + nrow).  `lg2n` is scratch of nrow doubles. */
static void table_row(int m, int L, int nrow, double *lg2n, float *tab) {
    const double ln2 = 0.6931471805599453;
    lg2n[0] = 0.5 * (lgamma(2.0 * m + 1.0) - 2.0 * lgamma(m + 1.0)) / ln2;
    for (int i = 1; i < nrow; ++i) {
        const int ell = m + i;
        if (ell <= L - 1) {
            const double e = (double)ell;
            lg2n[i] = lg2n[i - 1] + 0.5 * log2(((e + m) * (e - m)) / (e * e));
        } else {
            lg2n[i] = lg2n[i - 1];
        }
    }
    for (int i = 0; i < nrow; ++i) {
        const double n = (double)i, dm = (double)m;
        double c1 = 1.0, cb = 1.0, G = 0.0;
        if (i >= 2) {
            const double S = 2.0 * n + 2.0 * dm;
            const double den = 2.0 * n * (n + 2.0 * dm) * (S - 2.0);
            c1 = (S - 1.0) * S * (S - 2.0) / den;
            cb = 2.0 * (n + dm - 1.0) * (n + dm - 1.0) * S / den;
            G = c1 - 1.0 - cb;
        }
        int base_i = (i < 2) ? 0 : 2 + UNR * ((i - 2) / UNR);
        if (m + base_i > L - 1)
            base_i = L - 1 - m;
        const double lex = floor(lg2n[base_i]);
        float *t = tab + (size_t)i * KT;
        t[0] = (float)c1;
        t[1] = (float)(c1 - (double)t[0]);
        t[2] = (float)cb;
        t[3] = exp2f((float)(lg2n[i] - lex));
        t[4] = (float)G;
        t[5] = (float)(G - (double)t[4]);
        ((int32_t *)t)[6] = (int32_t)lex + 127;
        t[7] = 0.f;
    }
}

void dform_rings_spin0(int npad, int L, const float *xh, const float *xl, const double *lg,
                       const float *mlim, const float *rhs, const float *norm, float *out,
                       int nthreads) {
#ifdef _OPENMP
    if (nthreads > 0)
        omp_set_num_threads(nthreads);
#endif
    float mlmax = -1.f;
    for (int r = 0; r < npad; ++r)
        if (mlim[r] > mlmax)
            mlmax = mlim[r];
    const int nv = npad / V;

#pragma omp parallel
    {
        __m512 *v = (__m512 *)_mm_malloc((size_t)npad * sizeof(float), 64);
        __m512 *D = (__m512 *)_mm_malloc((size_t)npad * sizeof(float), 64);
        __m512i *ex = (__m512i *)_mm_malloc((size_t)npad * sizeof(int32_t), 64);
        float *man = (float *)_mm_malloc((size_t)npad * sizeof(float), 64);
        int32_t *ex0 = (int32_t *)_mm_malloc((size_t)npad * sizeof(int32_t), 64);
        double *lg2n = (double *)_mm_malloc((size_t)(L + UNR) * sizeof(double), 64);
        float *trow = (float *)_mm_malloc((size_t)(L + UNR) * KT * sizeof(float), 64);
#pragma omp for schedule(dynamic, 1)
        for (int m = 0; m < L; ++m) {
            float *orow = out + (size_t)m * L * 2;
            memset(orow, 0, (size_t)L * 2 * sizeof(float));
            if ((float)m > mlmax)
                continue;
            const int nrow = L - m + UNR;
            table_row(m, L, nrow, lg2n, trow);
            const float *rp_re = rhs + ((size_t)m * 4 + 0) * npad;
            const float *rp_im = rhs + ((size_t)m * 4 + 1) * npad;
            const float *rm_re = rhs + ((size_t)m * 4 + 2) * npad;
            const float *rm_im = rhs + ((size_t)m * 4 + 3) * npad;
            const float rs = (m & 1) ? -1.f : 1.f;

            /* first vector whose rings reach this order (mlim rises towards the equator) */
            int r0 = 0;
            while (r0 < npad && mlim[r0] < (float)m)
                ++r0;
            const int v0 = r0 / V;

            /* seeds: v_0 = man 2^ex0 = (sin th/2 cos th/2)^m */
            for (int r = v0 * V; r < npad; ++r) {
                if (mlim[r] < 0.f) {
                    man[r] = 0.f;
                    ex0[r] = EX_PAD;
                } else {
                    const double f = (double)m * lg[r];
                    const double e = floor(f);
                    man[r] = exp2f((float)(f - e));
                    ex0[r] = (int32_t)e;
                }
            }

            /* degrees m, m + 1 */
            const __m512 half = _mm512_set1_ps((float)(m + 1));
            const __m512 cst = _mm512_set1_ps((float)m);
            const __m512i le0 = _mm512_set1_epi32(((const int32_t *)trow)[6]);
            __m512 a0r = _mm512_setzero_ps(), a0i = _mm512_setzero_ps();
            __m512 a1r = _mm512_setzero_ps(), a1i = _mm512_setzero_ps();
            for (int q = v0; q < nv; ++q) {
                const int r = q * V;
                const __m512 h = _mm512_loadu_ps(xh + r), l = _mm512_loadu_ps(xl + r);
                const __m512 mn = _mm512_loadu_ps(man + r);
                const __m512i e0 = _mm512_loadu_si512((const void *)(ex0 + r));
                const __m512 p1m1 = _mm512_fmadd_ps(half, h, _mm512_fmadd_ps(half, l, cst));
                const __m512 vv = mn;
                const __m512 vD = _mm512_mul_ps(mn, p1m1);
                const __m512 sc = emit_scale(e0, le0);
                const __m512 val0 = _mm512_mul_ps(vv, sc);
                const __m512 v1 = _mm512_add_ps(vv, vD);
                const __m512 val1 = _mm512_mul_ps(v1, sc);
                a0r = _mm512_fmadd_ps(val0, _mm512_loadu_ps(rp_re + r), a0r);
                a0i = _mm512_fmadd_ps(val0, _mm512_loadu_ps(rp_im + r), a0i);
                a1r = _mm512_fmadd_ps(val1, _mm512_loadu_ps(rm_re + r), a1r);
                a1i = _mm512_fmadd_ps(val1, _mm512_loadu_ps(rm_im + r), a1i);
                v[q] = v1;
                D[q] = vD;
                ex[q] = e0;
            }
            {
                const float u0 = trow[3] * rs * norm[m];
                orow[2 * m] = hsum(a0r) * u0;
                orow[2 * m + 1] = hsum(a0i) * u0;
            }
            if (m + 1 < L) {
                const float u1 = trow[KT + 3] * rs * norm[m + 1];
                orow[2 * (m + 1)] = hsum(a1r) * u1;
                orow[2 * (m + 1) + 1] = hsum(a1i) * u1;
            }

            /* blocks of UNR degrees from m + 2 */
            const int nblk = (L - (m + 2) + UNR - 1) / UNR;
            for (int it = 0; it < nblk; ++it) {
                const int base = m + 2 + it * UNR;
                const float *tb = trow + (size_t)(base - m) * KT;
                const __m512i leb = _mm512_set1_epi32(((const int32_t *)tb)[6]);
                __m512 ar[UNR], ai[UNR];
                for (int k = 0; k < UNR; ++k) {
                    ar[k] = _mm512_setzero_ps();
                    ai[k] = _mm512_setzero_ps();
                }
                for (int q = v0; q < nv; ++q) {
                    const int r = q * V;
                    const __m512 h = _mm512_loadu_ps(xh + r), l = _mm512_loadu_ps(xl + r);
                    const __m512 pre = _mm512_loadu_ps(rp_re + r), pim = _mm512_loadu_ps(rp_im + r);
                    const __m512 mre = _mm512_loadu_ps(rm_re + r), mim = _mm512_loadu_ps(rm_im + r);
                    __m512 vv = v[q], vD = D[q];
                    __m512i vex = ex[q];
                    const __m512 sc = emit_scale(vex, leb);
#pragma GCC unroll 8
                    for (int k = 0; k < UNR; ++k) {
                        const float *t = tb + (size_t)k * KT;
                        const __m512 c1h = _mm512_set1_ps(t[0]);
                        const __m512 c1l = _mm512_set1_ps(t[1]);
                        const __m512 cb = _mm512_set1_ps(t[2]);
                        const __m512 eh = _mm512_set1_ps(t[4]);
                        const __m512 el = _mm512_set1_ps(t[5]);
                        const __m512 cbig = _mm512_fmadd_ps(c1h, h, eh);
                        __m512 s = _mm512_fmadd_ps(c1l, h, el);
                        s = _mm512_fmadd_ps(c1h, l, s);
                        const __m512 C = _mm512_add_ps(cbig, s);
                        vD = _mm512_fmadd_ps(cb, vD, _mm512_mul_ps(C, vv));
                        vv = _mm512_add_ps(vv, vD);
                        const __m512 val = _mm512_mul_ps(vv, sc);
                        if (k & 1) {
                            ar[k] = _mm512_fmadd_ps(val, mre, ar[k]);
                            ai[k] = _mm512_fmadd_ps(val, mim, ai[k]);
                        } else {
                            ar[k] = _mm512_fmadd_ps(val, pre, ar[k]);
                            ai[k] = _mm512_fmadd_ps(val, pim, ai[k]);
                        }
                    }
                    normalise(&vv, &vD, &vex);
                    v[q] = vv;
                    D[q] = vD;
                    ex[q] = vex;
                }
                for (int k = 0; k < UNR; ++k) {
                    const int ell = base + k;
                    if (ell < L) {
                        const float u = tb[(size_t)k * KT + 3] * rs * norm[ell];
                        orow[2 * ell] = hsum(ar[k]) * u;
                        orow[2 * ell + 1] = hsum(ai[k]) * u;
                    }
                }
            }
        }
        _mm_free(v);
        _mm_free(D);
        _mm_free(ex);
        _mm_free(man);
        _mm_free(ex0);
        _mm_free(lg2n);
        _mm_free(trow);
    }
}

/* ducc0 `map2leg` output (nring, L) complex128 (weights and ring phases already applied) ->
   the folded right-hand side (L, 4, npad) float32 of the kernel.  Blocked over 16 orders so
   every ring's read is two cache lines and the 64 destination rows stay in L2. */
void fold_leg(const double _Complex *leg, int nring, int L, int npad, float *rhs, int nthreads) {
#ifdef _OPENMP
    if (nthreads > 0)
        omp_set_num_threads(nthreads);
#endif
    const int north = (nring + 1) / 2;
    const int nmb = (L + 15) / 16;
#pragma omp parallel for schedule(dynamic, 1)
    for (int mb = 0; mb < nmb; ++mb) {
        const int m0 = mb * 16;
        const int mn = (m0 + 16 <= L) ? 16 : L - m0;
        for (int j = 0; j < north; ++j) {
            const int p = nring - 1 - j;
            const double _Complex *gn = leg + (size_t)j * L + m0;
            const double _Complex *gp = leg + (size_t)p * L + m0;
            for (int k = 0; k < mn; ++k) {
                const double _Complex a = gn[k];
                const double _Complex b = (p == j) ? 0.0 : gp[k];
                float *row = rhs + (size_t)(m0 + k) * 4 * npad;
                row[0 * npad + j] = (float)creal(a + b);
                row[1 * npad + j] = (float)cimag(a + b);
                row[2 * npad + j] = (float)creal(a - b);
                row[3 * npad + j] = (float)cimag(a - b);
            }
        }
        for (int k = 0; k < mn; ++k) {
            float *row = rhs + (size_t)(m0 + k) * 4 * npad;
            for (int c = 0; c < 4; ++c)
                for (int j = north; j < npad; ++j)
                    row[c * npad + j] = 0.f;
        }
    }
}
