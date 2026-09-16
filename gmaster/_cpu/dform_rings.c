/* Difference-form Jacobi march in ducc0 loop order: OpenMP over m, SIMD over rings.
   Spin 0 analysis. Same (v, D) recurrence as march_v2, not the GPU tile layout. */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static void jacobi_coeffs(int m, int ell, float *c1, float *c0, float *cb) {
    int n = ell - m;
    double S = 2.0 * n + 2.0 * m;
    double den = 2.0 * n * (n + 2.0 * m) * (S - 2.0);
    *c1 = (float)((S - 1.0) * S * (S - 2.0) / den);
    *c0 = 0.f;
    *cb = (float)(2.0 * (n + m - 1) * (n + m - 1) * S / den);
}

void dform_rings_spin0(int nr, int L, const float *x, const uint8_t *south,
                       const float *rhs_re, const float *rhs_im,
                       float *alm_re, float *alm_im, int nthreads) {
#ifdef _OPENMP
    if (nthreads > 0)
        omp_set_num_threads(nthreads);
#endif
#pragma omp parallel
    {
        float *v = (float *)malloc((size_t)nr * sizeof(float));
        float *D = (float *)malloc((size_t)nr * sizeof(float));
        int *ex = (int *)malloc((size_t)nr * sizeof(int));
#pragma omp for schedule(dynamic, 1)
        for (int m = 0; m < L; ++m) {
            const float half = (float)(m + 1);
            const float *re = rhs_re + (size_t)m * nr;
            const float *im = rhs_im + (size_t)m * nr;
            float *ar = alm_re + (size_t)m * L;
            float *ai = alm_im + (size_t)m * L;
            memset(ar, 0, (size_t)m * sizeof(float));
            memset(ai, 0, (size_t)m * sizeof(float));
            for (int r = 0; r < nr; ++r) {
                float cst = south[r] ? 0.f : (float)m;
                v[r] = 1.f;
                D[r] = half * x[r] + cst;
                v[r] += D[r];
                ex[r] = 0;
            }
            double acc_re = 0, acc_im = 0;
            for (int r = 0; r < nr; ++r) {
                float val = ldexpf(v[r], ex[r]);
                acc_re += (double)val * re[r];
                acc_im += (double)val * im[r];
            }
            ar[m] = (float)acc_re;
            ai[m] = (float)acc_im;
            if (m + 1 < L) {
                acc_re = acc_im = 0;
                for (int r = 0; r < nr; ++r) {
                    float val = ldexpf(v[r], ex[r]);
                    acc_re += (double)val * re[r];
                    acc_im += (double)val * im[r];
                }
                ar[m + 1] = (float)acc_re;
                ai[m + 1] = (float)acc_im;
            }
            for (int ell = m + 2; ell < L; ++ell) {
                float c1, c0, cb;
                jacobi_coeffs(m, ell, &c1, &c0, &cb);
                float Ep = c1 - 1.f - cb + c0;
                float Em = c1 - 1.f - cb - c0;
                acc_re = acc_im = 0;
                for (int r = 0; r < nr; ++r) {
                    float C = c1 * x[r] + (south[r] ? Em : Ep);
                    D[r] = cb * D[r] + C * v[r];
                    v[r] += D[r];
                    float val = ldexpf(v[r], ex[r]);
                    acc_re += (double)val * re[r];
                    acc_im += (double)val * im[r];
                }
                ar[ell] = (float)acc_re;
                ai[ell] = (float)acc_im;
                if (((ell - m) & 7) == 0) {
                    for (int r = 0; r < nr; ++r) {
                        float big = fmaxf(fabsf(v[r]), fabsf(D[r]));
                        int bits;
                        memcpy(&bits, &big, 4);
                        int eb = (bits >> 23) & 0xff;
                        if (eb < 1)
                            eb = 1;
                        int scb = (254 - eb) << 23;
                        float sc;
                        memcpy(&sc, &scb, 4);
                        v[r] *= sc;
                        D[r] *= sc;
                        ex[r] += eb - 127;
                    }
                }
            }
        }
        free(v);
        free(D);
        free(ex);
    }
}
