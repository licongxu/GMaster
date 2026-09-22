// GMaster divide-and-conquer spin-0 latitudinal transform: apply kernels and XLA FFI handlers.
//
// For each order m and parity p the orthonormal Legendre functions phi_k(y) = psi_{m+p+2k}(sqrt y)
// of y = cos^2 theta satisfy a symmetric three-term recurrence with Jacobi matrix T = V Y V^T.
// Synthesis f(y_r) = sum_k c_k phi_k(y_r) is applied as (Christoffel-Darboux)
//     f(y_r) = E_{n-1} phi_n(y_r) sum_j V[n-1, j] (V^T c)_j / (y_r - y_j),
// with V^T applied by the Cuppen divide-and-conquer tree of T: dense 16 x 16 leaf blocks, then one
// Cauchy-like merge per level (direct below DIRECT_MAX kept poles, a 1-D FMM above), and the last
// step a 1-D FMM from the Gauss nodes y_j to the rings.  Every step is O(n) or O(n log n) per
// (m, p), so the transform is O(L^2 log L) instead of the march's O(L^2 N_ring).  Analysis is the
// exact adjoint (same data, transposed kernels, reverse order).  Plan: gmaster/_cpu/dc_plan.cpp.
#include <cuda_runtime.h>
#include <cstdint>
#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;
#define P 12

struct Node { int base, nk, nd, poff, doff, pad0, pad1, pad2; };

__device__ __forceinline__ int org_of(int j, float tau) { return tau > 0.f ? j : j + 1; }

// lam_j - d_i = (d_o - d_i) + tau_j, o = org(j): exactly tau_j when i == o (the difference is 0)
// and ~1e-15 absolute otherwise, with no branch.
__device__ __forceinline__ float lam_minus_d(const float* dh, const float* dl, int j, float tau, int i) {
    int o = org_of(j, tau);
    return ((dh[o] - dh[i]) + (dl[o] - dl[i])) + tau;
}
__device__ __forceinline__ float rcp(float x) { return __fdividef(1.0f, x); }   // MUFU.RCP

// ------------------------------------------------------------------ direct, one block per node
__global__ void merge_direct(
    const Node* __restrict__ nodes, int nnode, int dir,
    const float* __restrict__ gdh, const float* __restrict__ gdl, const float* __restrict__ gtau,
    const float* __restrict__ gz, const float* __restrict__ gc,
    const short* __restrict__ ggidx, const short* __restrict__ gslot,
    const short* __restrict__ gdsrc, const short* __restrict__ gddst,
    float2* __restrict__ w)
{
    extern __shared__ float sm[];
    int nb = blockIdx.x;
    if (nb >= nnode) return;
    Node nd = nodes[nb];
    int nk = nd.nk;
    float* dh = sm; float* dl = sm + nk; float* tau = sm + 2 * nk; float2* q = (float2*)(sm + 3 * nk + (nk & 1));
    const float* pdh = gdh + nd.poff; const float* pdl = gdl + nd.poff; const float* pt = gtau + nd.poff;
    for (int i = threadIdx.x; i < nk; i += blockDim.x) {
        dh[i] = pdh[i]; dl[i] = pdl[i]; tau[i] = pt[i];
        if (dir == 0) { float2 v = w[nd.base + ggidx[nd.poff + i]]; float zz = gz[nd.poff + i]; q[i] = make_float2(zz * v.x, zz * v.y); }
        else          { float2 v = w[nd.base + gslot[nd.poff + i]]; float cc = gc[nd.poff + i]; q[i] = make_float2(cc * v.x, cc * v.y); }
    }
    // deflated values: at most blockDim.x per node in this kernel (checked on the host)
    float2 dv = make_float2(0.f, 0.f);
    int t = threadIdx.x;
    if (t < nd.nd) dv = w[nd.base + (dir == 0 ? gdsrc[nd.doff + t] : gddst[nd.doff + t])];
    __syncthreads();
    for (int k = threadIdx.x; k < nk; k += blockDim.x) {
        float2 acc = make_float2(0.f, 0.f);
        if (dir == 0) {                // target root k, sources poles i
            float tk = tau[k];
            int o = org_of(k, tk);
            float oh = dh[o], ol = dl[o];   // lam_k - d_i = ((oh - dh[i]) + (ol - dl[i])) + tau_k; tau last keeps i == o exact
            for (int i = 0; i < nk; ++i) {
                float inv = rcp(((oh - dh[i]) + (ol - dl[i])) + tk);
                acc.x += q[i].x * inv; acc.y += q[i].y * inv;
            }
            float cc = -gc[nd.poff + k];
            w[nd.base + gslot[nd.poff + k]] = make_float2(cc * acc.x, cc * acc.y);
        } else {                       // target pole k, sources roots j:  1/(d_k - lam_j)
            float kh = dh[k], kl = dl[k];
            for (int j = 0; j < nk; ++j) {
                float tj = tau[j];
                int o = org_of(j, tj);
                float inv = rcp((kh - dh[o]) + ((kl - dl[o]) - tj));   // d_k - lam_j
                acc.x += q[j].x * inv; acc.y += q[j].y * inv;
            }
            float zz = gz[nd.poff + k];
            w[nd.base + ggidx[nd.poff + k]] = make_float2(zz * acc.x, zz * acc.y);
        }
    }
    if (t < nd.nd) w[nd.base + (dir == 0 ? gddst[nd.doff + t] : gdsrc[nd.doff + t])] = dv;
}

// ------------------------------------------------------------------ FMM, one block per node
// Leaves are 8 poles + 8 roots (interlaced), box b at level l covers leaves [b 2^l, (b+1) 2^l).
// Scratch per node (offset sbase, in boxes): geo lo/hi, moments, locals; qs (staged strengths).
__constant__ float BIN[P * P] = {1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f,9.0f,10.0f,11.0f,12.0f,1.0f,3.0f,6.0f,10.0f,15.0f,21.0f,28.0f,36.0f,45.0f,55.0f,66.0f,78.0f,1.0f,4.0f,10.0f,20.0f,35.0f,56.0f,84.0f,120.0f,165.0f,220.0f,286.0f,364.0f,1.0f,5.0f,15.0f,35.0f,70.0f,126.0f,210.0f,330.0f,495.0f,715.0f,1001.0f,1365.0f,1.0f,6.0f,21.0f,56.0f,126.0f,252.0f,462.0f,792.0f,1287.0f,2002.0f,3003.0f,4368.0f,1.0f,7.0f,28.0f,84.0f,210.0f,462.0f,924.0f,1716.0f,3003.0f,5005.0f,8008.0f,12376.0f,1.0f,8.0f,36.0f,120.0f,330.0f,792.0f,1716.0f,3432.0f,6435.0f,11440.0f,19448.0f,31824.0f,1.0f,9.0f,45.0f,165.0f,495.0f,1287.0f,3003.0f,6435.0f,12870.0f,24310.0f,43758.0f,75582.0f,1.0f,10.0f,55.0f,220.0f,715.0f,2002.0f,5005.0f,11440.0f,24310.0f,48620.0f,92378.0f,167960.0f,1.0f,11.0f,66.0f,286.0f,1001.0f,3003.0f,8008.0f,19448.0f,43758.0f,92378.0f,184756.0f,352716.0f,1.0f,12.0f,78.0f,364.0f,1365.0f,4368.0f,12376.0f,31824.0f,75582.0f,167960.0f,352716.0f,705432.0f};      // BIN[j*P + k] = C(j+k, j)
__constant__ float CKJ[P * P] = {1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,2.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,3.0f,3.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,4.0f,6.0f,4.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,5.0f,10.0f,10.0f,5.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,6.0f,15.0f,20.0f,15.0f,6.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,7.0f,21.0f,35.0f,35.0f,21.0f,7.0f,1.0f,0.0f,0.0f,0.0f,0.0f,1.0f,8.0f,28.0f,56.0f,70.0f,56.0f,28.0f,8.0f,1.0f,0.0f,0.0f,0.0f,1.0f,9.0f,36.0f,84.0f,126.0f,126.0f,84.0f,36.0f,9.0f,1.0f,0.0f,0.0f,1.0f,10.0f,45.0f,120.0f,210.0f,252.0f,210.0f,120.0f,45.0f,10.0f,1.0f,0.0f,1.0f,11.0f,55.0f,165.0f,330.0f,462.0f,462.0f,330.0f,165.0f,55.0f,11.0f,1.0f};      // CKJ[k*P + j] = C(k, j)

__device__ __forceinline__ void pos_src(int dir, const float* pdh, const float* pdl, const float* pt, int i, float& h, float& l) {
    if (dir == 0) { h = pdh[i]; l = pdl[i]; }
    else { int o = org_of(i, pt[i]); h = pdh[o] + pt[i]; l = pdl[o] + ((pdh[o] - h) + pt[i]); }
}
__device__ __forceinline__ void pos_tgt(int dir, const float* pdh, const float* pdl, const float* pt, int i, float& h, float& l) {
    pos_src(1 - dir, pdh, pdl, pt, i, h, l);
}


// Upward (M2M), interaction (M2L) and downward (L2L) passes of the 1-D Cauchy FMM for one
// problem whose leaf boxes already hold geometry and moments.  Box b at level l covers leaves
// [b 2^l, (b+1) 2^l); lof[l] is the first box of level l.
__device__ void fmm_passes(int nlev, const int* lof, int bo, float* __restrict__ geo,
                           float2* __restrict__ mom, float2* __restrict__ loc) {
    const int tid = threadIdx.x, nth = blockDim.x;
    // M2M: thread per parent box.  M_k = sum_j C(k,j) be^(k-j) (al^j Mc_j) over both children.
    for (int l = 1; l < nlev; ++l) {
        int nbx = lof[l + 1] - lof[l], cb0 = lof[l - 1], pb0 = lof[l], nc = lof[l] - lof[l - 1];
        for (int b = tid; b < nbx; b += nth) {
            int c1 = 2 * b, c2 = min(2 * b + 1, nc - 1);
            float lo = fminf(geo[2 * (bo + cb0 + c1)], geo[2 * (bo + cb0 + c2)]);
            float hi = fmaxf(geo[2 * (bo + cb0 + c1) + 1], geo[2 * (bo + cb0 + c2) + 1]);
            int g = bo + pb0 + b;
            geo[2 * g] = lo; geo[2 * g + 1] = hi;
            float c = 0.5f * (lo + hi), r = fmaxf(0.5f * (hi - lo), 1e-6f * fabsf(c) + 1e-30f), ir = 1.0f / r;
            float2 Mp[P];
            #pragma unroll
            for (int k = 0; k < P; ++k) Mp[k] = make_float2(0.f, 0.f);
            for (int ch = c1; ch <= c2; ++ch) {
                int gc_ = bo + cb0 + ch;
                float clo = geo[2 * gc_], chi = geo[2 * gc_ + 1];
                float cc = 0.5f * (clo + chi), rc = fmaxf(0.5f * (chi - clo), 1e-6f * fabsf(cc) + 1e-30f);
                float al = rc * ir, be = (cc - c) * ir;
                float2 A[P]; float bp[P];
                float ap = 1.f; bp[0] = 1.f;
                #pragma unroll
                for (int j = 0; j < P; ++j) { float2 v = mom[gc_ * P + j]; A[j] = make_float2(v.x * ap, v.y * ap); ap *= al; if (j) bp[j] = bp[j - 1] * be; }
                #pragma unroll
                for (int k = 0; k < P; ++k) {
                    #pragma unroll
                    for (int j = 0; j <= k; ++j) {
                        float f = CKJ[k * P + j] * bp[k - j];
                        Mp[k].x += f * A[j].x; Mp[k].y += f * A[j].y;
                    }
                }
            }
            #pragma unroll
            for (int k = 0; k < P; ++k) mom[g * P + k] = Mp[k];
        }
        __syncthreads();
    }
    // M2L: thread per box.  L_j += (bb^j / D) sum_k C(j+k, j) (a^k M_k) per interaction-list box.
    for (int l = 0; l < nlev; ++l) {
        int nbx = lof[l + 1] - lof[l], b0 = lof[l];
        for (int b = tid; b < nbx; b += nth) {
            int g = bo + b0 + b;
            float lo = geo[2 * g], hi = geo[2 * g + 1];
            float C = 0.5f * (lo + hi), R = fmaxf(0.5f * (hi - lo), 1e-6f * fabsf(C) + 1e-30f);
            float2 Lacc[P];
            #pragma unroll
            for (int j = 0; j < P; ++j) Lacc[j] = make_float2(0.f, 0.f);
            int pb = b >> 1;
            int c0 = (l == nlev - 1) ? 0 : 2 * (pb - 1), c1 = (l == nlev - 1) ? nbx - 1 : 2 * (pb + 1) + 1;
            for (int cand = max(c0, 0); cand <= min(c1, nbx - 1); ++cand) {
                if (cand >= b - 1 && cand <= b + 1) continue;
                int gs = bo + b0 + cand;
                float slo = geo[2 * gs], shi = geo[2 * gs + 1];
                float c = 0.5f * (slo + shi), r = fmaxf(0.5f * (shi - slo), 1e-6f * fabsf(c) + 1e-30f);
                float iD = 1.0f / (C - c), a = r * iD, bb = -R * iD;
                float2 A[P];
                float ak = 1.f;
                #pragma unroll
                for (int k = 0; k < P; ++k) { float2 v = mom[gs * P + k]; A[k] = make_float2(v.x * ak, v.y * ak); ak *= a; }
                float bj = iD;
                #pragma unroll
                for (int j = 0; j < P; ++j) {
                    float2 s2 = make_float2(0.f, 0.f);
                    #pragma unroll
                    for (int k = 0; k < P; ++k) { float f = BIN[j * P + k]; s2.x += f * A[k].x; s2.y += f * A[k].y; }
                    Lacc[j].x += bj * s2.x; Lacc[j].y += bj * s2.y;
                    bj *= bb;
                }
            }
            #pragma unroll
            for (int j = 0; j < P; ++j) loc[g * P + j] = Lacc[j];
        }
        __syncthreads();
    }
    // L2L: thread per child box.  L_i += al^i sum_{j>=i} C(j,i) be^(j-i) Lp_j
    for (int l = nlev - 2; l >= 0; --l) {
        int nbx = lof[l + 1] - lof[l], b0 = lof[l], pb0 = lof[l + 1];
        for (int b = tid; b < nbx; b += nth) {
            int g = bo + b0 + b, gp = bo + pb0 + (b >> 1);
            float lo = geo[2 * g], hi = geo[2 * g + 1];
            float C1 = 0.5f * (lo + hi), R1 = fmaxf(0.5f * (hi - lo), 1e-6f * fabsf(C1) + 1e-30f);
            float plo = geo[2 * gp], phi = geo[2 * gp + 1];
            float C = 0.5f * (plo + phi), R = fmaxf(0.5f * (phi - plo), 1e-6f * fabsf(C) + 1e-30f), iR = 1.0f / R;
            float al = R1 * iR, be = (C1 - C) * iR;
            float2 Lp[P]; float bp[P]; bp[0] = 1.f;
            #pragma unroll
            for (int j = 0; j < P; ++j) { Lp[j] = loc[gp * P + j]; if (j) bp[j] = bp[j - 1] * be; }
            float ap = 1.f;
            #pragma unroll
            for (int i = 0; i < P; ++i) {
                float2 acc = make_float2(0.f, 0.f);
                #pragma unroll
                for (int j = i; j < P; ++j) { float f = CKJ[j * P + i] * bp[j - i]; acc.x += f * Lp[j].x; acc.y += f * Lp[j].y; }
                float2 cur = loc[g * P + i];
                loc[g * P + i] = make_float2(cur.x + ap * acc.x, cur.y + ap * acc.y);
                ap *= al;
            }
        }
        __syncthreads();
    }
}

__global__ void merge_fmm(
    const Node* __restrict__ nodes, int nnode, int dir,
    const float* __restrict__ gdh, const float* __restrict__ gdl, const float* __restrict__ gtau,
    const float* __restrict__ gz, const float* __restrict__ gc,
    const short* __restrict__ ggidx, const short* __restrict__ gslot,
    const short* __restrict__ gdsrc, const short* __restrict__ gddst,
    float2* __restrict__ w,
    const int* __restrict__ sbase, float* __restrict__ geo, float2* __restrict__ mom, float2* __restrict__ loc,
    float2* __restrict__ qs, float2* __restrict__ dstage)
{
    int nb = blockIdx.x;
    if (nb >= nnode) return;
    Node nd = nodes[nb];
    int nk = nd.nk, tid = threadIdx.x, nth = blockDim.x;
    const float* pdh = gdh + nd.poff; const float* pdl = gdl + nd.poff; const float* pt = gtau + nd.poff;
    float2* q = qs + nd.poff;
    float2* dst = dstage + nd.doff;
    // stage strengths and deflated values (all reads of w happen here)
    for (int i = tid; i < nk; i += nth) {
        if (dir == 0) { float2 v = w[nd.base + ggidx[nd.poff + i]]; float zz = gz[nd.poff + i]; q[i] = make_float2(zz * v.x, zz * v.y); }
        else          { float2 v = w[nd.base + gslot[nd.poff + i]]; float cc = gc[nd.poff + i]; q[i] = make_float2(cc * v.x, cc * v.y); }
    }
    for (int i = tid; i < nd.nd; i += nth) dst[i] = w[nd.base + (dir == 0 ? gdsrc[nd.doff + i] : gddst[nd.doff + i])];
    int nleaf = (nk + 7) / 8;
    int nlev = 1, sz = nleaf;
    int lof[20]; lof[0] = 0;
    while (true) { lof[nlev] = lof[nlev - 1] + sz; if (sz <= 3) break; sz = (sz + 1) / 2; ++nlev; }
    int bo = sbase[nb];
    __syncthreads();
    // leaves: geometry + P2M
    for (int b = tid; b < nleaf; b += nth) {
        int i0 = 8 * b, i1 = min(8 * b + 8, nk);
        float sh0, sl0, sh1, sl1, th0, tl0, th1, tl1;
        pos_src(dir, pdh, pdl, pt, i0, sh0, sl0); pos_src(dir, pdh, pdl, pt, i1 - 1, sh1, sl1);
        pos_tgt(dir, pdh, pdl, pt, i0, th0, tl0); pos_tgt(dir, pdh, pdl, pt, i1 - 1, th1, tl1);
        float lo = fminf(sh0, th0), hi = fmaxf(sh1, th1);
        int g = bo + b;
        geo[2 * g] = lo; geo[2 * g + 1] = hi;
        float c = 0.5f * (lo + hi), r = fmaxf(0.5f * (hi - lo), 1e-6f * fabsf(c) + 1e-30f), ir = 1.0f / r;
        float2 M[P];
        #pragma unroll
        for (int k = 0; k < P; ++k) M[k] = make_float2(0.f, 0.f);
        for (int s = i0; s < i1; ++s) {
            float h, l; pos_src(dir, pdh, pdl, pt, s, h, l);
            float xi = ((h - c) + l) * ir;
            float2 qq = q[s];
            float pw = 1.f;
            #pragma unroll
            for (int k = 0; k < P; ++k) { M[k].x += qq.x * pw; M[k].y += qq.y * pw; pw *= xi; }
        }
        #pragma unroll
        for (int k = 0; k < P; ++k) mom[g * P + k] = M[k];
    }
    __syncthreads();
    fmm_passes(nlev, lof, bo, geo, mom, loc);
    // targets: L2P + P2P, then scaled scatter
    for (int k = tid; k < nk; k += nth) {
        int b = k >> 3;
        int g = bo + b;
        float lo = geo[2 * g], hi = geo[2 * g + 1];
        float C = 0.5f * (lo + hi), R = fmaxf(0.5f * (hi - lo), 1e-6f * fabsf(C) + 1e-30f), iR = 1.0f / R;
        float th, tl; pos_tgt(dir, pdh, pdl, pt, k, th, tl);
        float eta = ((th - C) + tl) * iR;
        float2 acc = make_float2(0.f, 0.f);
        #pragma unroll
        for (int j = P - 1; j >= 0; --j) { float2 v = loc[g * P + j]; acc.x = acc.x * eta + v.x; acc.y = acc.y * eta + v.y; }
        int s0 = max(8 * (b - 1), 0), s1 = min(8 * (b + 2), nk);
        if (dir == 0) {
            float tk = pt[k];
            for (int s = s0; s < s1; ++s) {
                float inv = rcp(lam_minus_d(pdh, pdl, k, tk, s));
                acc.x += q[s].x * inv; acc.y += q[s].y * inv;
            }
            float cc = -gc[nd.poff + k];
            w[nd.base + gslot[nd.poff + k]] = make_float2(cc * acc.x, cc * acc.y);
        } else {
            for (int s = s0; s < s1; ++s) {
                float inv = rcp(-lam_minus_d(pdh, pdl, s, pt[s], k));
                acc.x += q[s].x * inv; acc.y += q[s].y * inv;
            }
            float zz = gz[nd.poff + k];
            w[nd.base + ggidx[nd.poff + k]] = make_float2(zz * acc.x, zz * acc.y);
        }
    }
    for (int i = tid; i < nd.nd; i += nth) w[nd.base + (dir == 0 ? gddst[nd.doff + i] : gdsrc[nd.doff + i])] = dst[i];
}

// ------------------------------------------------------------------ leaves (warp per leaf, in place)
__global__ void leaves(const int* __restrict__ off, const int* __restrict__ sz,
                                  const float* __restrict__ Q, float2* __restrict__ w, int nleaf, int dir) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= nleaf) return;
    int o = off[warp], s = sz[warp];
    const float* q = Q + 256 * warp;
    float2 x = lane < s ? w[o + lane] : make_float2(0.f, 0.f);
    float2 acc = make_float2(0.f, 0.f);
    for (int k = 0; k < s; ++k) {
        float2 xk = make_float2(__shfl_sync(0xffffffff, x.x, k), __shfl_sync(0xffffffff, x.y, k));
        float v = lane < s ? (dir == 0 ? q[16 * k + lane] : q[16 * lane + k]) : 0.f;
        acc.x += v * xk.x; acc.y += v * xk.y;
    }
    if (lane < s) w[o + lane] = acc;
}

// ------------------------------------------------------------------ Christoffel-Darboux step
// Per problem (m, p): nodes y_j (the root Gauss nodes, ascending, double-float) and the first nt
// rings in ascending y (the rest sit in the forbidden region and are skipped).  Leaves are
// contiguous ranges of the merged order, given by ln0 (node ranges) and lr0 (ring ranges).
// dir 0 (synthesis): ring[p, R-1-t] = scale[p, R-1-t] * sum_j vlast_j w[so+j] / (y_t - y_j)
// dir 1 (analysis):  w[so+j] = vlast_j * sum_t scale ring / (y_t - y_j)            (adjoint)
struct CDProb { int so, n, lo, nleaf, bo, nt, pad0, pad1; };

__global__ void cd_fmm(
    const CDProb* __restrict__ probs, int nprob, int dir,
    const float* __restrict__ yh, const float* __restrict__ yl, const float* __restrict__ vlast,
    const float* __restrict__ ryh, const float* __restrict__ ryl, int R,
    const int* __restrict__ ln0, const int* __restrict__ lr0, const float* __restrict__ scale,
    float2* __restrict__ w, float2* __restrict__ ring,
    float* __restrict__ geo, float2* __restrict__ mom, float2* __restrict__ loc, float2* __restrict__ qs)
{
    int pb = blockIdx.x;
    if (pb >= nprob) return;
    CDProb pr = probs[pb];
    int tid = threadIdx.x, nth = blockDim.x;
    const int* L0 = ln0 + pr.lo + pb;          // nleaf + 1 entries each
    const int* R0 = lr0 + pr.lo + pb;
    const float* sc = scale + (size_t)pb * R;
    float2* rg = ring + (size_t)pb * R;
    const float* nh = yh + pr.so; const float* nl = yl + pr.so;
    // source strengths
    int ns = dir == 0 ? pr.n : pr.nt;
    float2* q = qs + (dir == 0 ? pr.so : (size_t)pb * R);
    for (int i = tid; i < ns; i += nth) {
        if (dir == 0) { float2 v = w[pr.so + i]; float f = vlast[pr.so + i]; q[i] = make_float2(f * v.x, f * v.y); }
        else { int r = R - 1 - i; float2 v = rg[r]; float f = sc[r]; q[i] = make_float2(f * v.x, f * v.y); }
    }
    if (dir == 0) for (int t = pr.nt + tid; t < R; t += nth) rg[R - 1 - t] = make_float2(0.f, 0.f);
    int nleaf = pr.nleaf, nlev = 1, sz = nleaf, lof[24]; lof[0] = 0;
    while (true) { lof[nlev] = lof[nlev - 1] + sz; if (sz <= 3) break; sz = (sz + 1) / 2; ++nlev; }
    int bo = pr.bo;
    __syncthreads();
    // leaves: hull of both point sets, P2M of this direction's sources
    for (int b = tid; b < nleaf; b += nth) {
        int n0 = L0[b], n1 = L0[b + 1], r0 = R0[b], r1 = R0[b + 1];
        float lo = 1e30f, hi = -1e30f;
        if (n1 > n0) { lo = fminf(lo, nh[n0]); hi = fmaxf(hi, nh[n1 - 1]); }
        if (r1 > r0) { lo = fminf(lo, ryh[r0]); hi = fmaxf(hi, ryh[r1 - 1]); }
        int g = bo + b;
        geo[2 * g] = lo; geo[2 * g + 1] = hi;
        float c = 0.5f * (lo + hi), r = fmaxf(0.5f * (hi - lo), 1e-6f * fabsf(c) + 1e-30f), ir = 1.0f / r;
        float2 M[P];
        #pragma unroll
        for (int k = 0; k < P; ++k) M[k] = make_float2(0.f, 0.f);
        int s0 = dir == 0 ? n0 : r0, s1 = dir == 0 ? n1 : r1;
        for (int s = s0; s < s1; ++s) {
            float h = dir == 0 ? nh[s] : ryh[s], l = dir == 0 ? nl[s] : ryl[s];
            float xi = ((h - c) + l) * ir;
            float2 qq = q[s];
            float pw = 1.f;
            #pragma unroll
            for (int k = 0; k < P; ++k) { M[k].x += qq.x * pw; M[k].y += qq.y * pw; pw *= xi; }
        }
        #pragma unroll
        for (int k = 0; k < P; ++k) mom[g * P + k] = M[k];
    }
    __syncthreads();
    fmm_passes(nlev, lof, bo, geo, mom, loc);
    // targets
    int ntg = dir == 0 ? pr.nt : pr.n;
    const int* T0 = dir == 0 ? R0 : L0;
    const int* S0 = dir == 0 ? L0 : R0;
    for (int t = tid; t < ntg; t += nth) {
        int lo_b = 0, hi_b = nleaf;
        while (hi_b - lo_b > 1) { int mid = (lo_b + hi_b) >> 1; if (T0[mid] <= t) lo_b = mid; else hi_b = mid; }
        int b = lo_b, g = bo + b;
        float lo = geo[2 * g], hi = geo[2 * g + 1];
        float C = 0.5f * (lo + hi), Rr = fmaxf(0.5f * (hi - lo), 1e-6f * fabsf(C) + 1e-30f), iR = 1.0f / Rr;
        float th = dir == 0 ? ryh[t] : nh[t], tl = dir == 0 ? ryl[t] : nl[t];
        float eta = ((th - C) + tl) * iR;
        float2 acc = make_float2(0.f, 0.f);
        #pragma unroll
        for (int j = P - 1; j >= 0; --j) { float2 v = loc[g * P + j]; acc.x = acc.x * eta + v.x; acc.y = acc.y * eta + v.y; }
        int s0 = S0[max(b - 1, 0)], s1 = S0[min(b + 1, nleaf - 1) + 1];
        for (int s = s0; s < s1; ++s) {
            float sh_ = dir == 0 ? nh[s] : ryh[s], sl_ = dir == 0 ? nl[s] : ryl[s];
            float inv = rcp((th - sh_) + (tl - sl_));
            acc.x += q[s].x * inv; acc.y += q[s].y * inv;
        }
        if (dir == 0) { int r = R - 1 - t; float f = sc[r]; rg[r] = make_float2(f * acc.x, f * acc.y); }
        else { float f = -vlast[pr.so + t]; w[pr.so + t] = make_float2(f * acc.x, f * acc.y); }
    }
}


// ================================================================================ FFI handlers
static constexpr int DIRECT_THREADS = 128;
static constexpr int FMM_THREADS = 256;

static void run_levels(cudaStream_t s, ffi::Span<const int64_t> lev, int dir,
                       const int32_t* nodes, const int32_t* sbase, const float* dh, const float* dl,
                       const float* tau, const float* z, const float* c,
                       const short* gidx, const short* slot, const short* dsrc, const short* ddst,
                       float2* w, float* geo, float2* mom, float2* loc, float2* qs, float2* dstage) {
    const int nlev = lev.size() / 5;
    for (int q = 0; q < nlev; ++q) {
        const int i = dir == 0 ? q : nlev - 1 - q;
        const int kind = lev[5 * i], nn = lev[5 * i + 1], node0 = lev[5 * i + 2], maxk = lev[5 * i + 3], sb0 = lev[5 * i + 4];
        const Node* nd = reinterpret_cast<const Node*>(nodes) + node0;
        if (kind == 0) {
            merge_direct<<<nn, DIRECT_THREADS, 4 * (5 * maxk + 2), s>>>(nd, nn, dir, dh, dl, tau, z, c, gidx, slot, dsrc, ddst, w);
        } else {
            merge_fmm<<<nn, FMM_THREADS, 0, s>>>(nd, nn, dir, dh, dl, tau, z, c, gidx, slot, dsrc, ddst, w,
                                                  sbase + sb0, geo, mom, loc, qs, dstage);
        }
    }
}

#define PLAN_ARGS \
    ffi::Buffer<ffi::S32> leaf_off, ffi::Buffer<ffi::S32> leaf_sz, ffi::Buffer<ffi::F32> leaf_Q, \
    ffi::Buffer<ffi::S32> nodes, ffi::Buffer<ffi::S32> sbase, ffi::Buffer<ffi::F32> dh, ffi::Buffer<ffi::F32> dl, \
    ffi::Buffer<ffi::F32> tau, ffi::Buffer<ffi::F32> z, ffi::Buffer<ffi::F32> c, \
    ffi::Buffer<ffi::S16> gidx, ffi::Buffer<ffi::S16> slot, ffi::Buffer<ffi::S16> dsrc, ffi::Buffer<ffi::S16> ddst, \
    ffi::Buffer<ffi::S32> cd_desc, ffi::Buffer<ffi::S32> cd_ln, ffi::Buffer<ffi::S32> cd_lr, \
    ffi::Buffer<ffi::F32> cd_nh, ffi::Buffer<ffi::F32> cd_nl, ffi::Buffer<ffi::F32> vlast, ffi::Buffer<ffi::F32> scale, \
    ffi::Buffer<ffi::F32> ring_h, ffi::Buffer<ffi::F32> ring_l

#define PLAN_BIND \
    .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::F32>>() \
    .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>() \
    .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>() \
    .Arg<ffi::Buffer<ffi::S16>>().Arg<ffi::Buffer<ffi::S16>>().Arg<ffi::Buffer<ffi::S16>>().Arg<ffi::Buffer<ffi::S16>>() \
    .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>() \
    .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>() \
    .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()

#define SCRATCH_ARGS \
    ffi::ResultBuffer<ffi::F32> geo, ffi::ResultBuffer<ffi::C64> mom, ffi::ResultBuffer<ffi::C64> loc, \
    ffi::ResultBuffer<ffi::C64> qs, ffi::ResultBuffer<ffi::C64> dstage

#define SCRATCH_BIND \
    .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::C64>>().Ret<ffi::Buffer<ffi::C64>>() \
    .Ret<ffi::Buffer<ffi::C64>>().Ret<ffi::Buffer<ffi::C64>>()

#define F2(b) reinterpret_cast<float2*>((b)->typed_data())
#define CF2(b) reinterpret_cast<const float2*>((b).typed_data())

static ffi::Error apply(cudaStream_t s, int dir, const float2* in, float2* w, float2* ring_out,
                        const float2* ring_in, int nprob, int R, ffi::Span<const int64_t> lev, PLAN_ARGS,
                        float* g, float2* mo, float2* lo, float2* q, float2* dst) {
    const int nleaf = leaf_off.element_count();
    if (dir == 0) {
        cudaMemcpyAsync(w, in, sizeof(float2) * vlast.element_count(), cudaMemcpyDeviceToDevice, s);
        leaves<<<(nleaf * 32 + 127) / 128, 128, 0, s>>>(leaf_off.typed_data(), leaf_sz.typed_data(), leaf_Q.typed_data(), w, nleaf, 0);
    } else {
        cd_fmm<<<nprob, 256, 0, s>>>(reinterpret_cast<const CDProb*>(cd_desc.typed_data()), nprob, 1,
            cd_nh.typed_data(), cd_nl.typed_data(), vlast.typed_data(), ring_h.typed_data(), ring_l.typed_data(), R,
            cd_ln.typed_data(), cd_lr.typed_data(), scale.typed_data(), w, const_cast<float2*>(ring_in), g, mo, lo, q);
    }
    run_levels(s, lev, dir, nodes.typed_data(), sbase.typed_data(), dh.typed_data(), dl.typed_data(),
               tau.typed_data(), z.typed_data(), c.typed_data(), gidx.typed_data(), slot.typed_data(),
               dsrc.typed_data(), ddst.typed_data(), w, g, mo, lo, q, dst);
    if (dir == 0) {
        cd_fmm<<<nprob, 256, 0, s>>>(reinterpret_cast<const CDProb*>(cd_desc.typed_data()), nprob, 0,
            cd_nh.typed_data(), cd_nl.typed_data(), vlast.typed_data(), ring_h.typed_data(), ring_l.typed_data(), R,
            cd_ln.typed_data(), cd_lr.typed_data(), scale.typed_data(), w, ring_out, g, mo, lo, q);
    } else {
        leaves<<<(nleaf * 32 + 127) / 128, 128, 0, s>>>(leaf_off.typed_data(), leaf_sz.typed_data(), leaf_Q.typed_data(), w, nleaf, 1);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(err));
    return ffi::Error::Success();
}

// synthesis: coef (ntot) -> ring (nprob, R); work (ntot) is the tree's in-place vector
ffi::Error SynthImpl(cudaStream_t s, ffi::Buffer<ffi::C64> coef, PLAN_ARGS,
                     ffi::ResultBuffer<ffi::C64> ring, ffi::ResultBuffer<ffi::C64> work, SCRATCH_ARGS,
                     ffi::Span<const int64_t> lev) {
    const auto rd = ring->dimensions();
    return apply(s, 0, CF2(coef), F2(work), F2(ring), nullptr, rd[0], rd[1], lev,
                 leaf_off, leaf_sz, leaf_Q, nodes, sbase, dh, dl, tau, z, c, gidx, slot, dsrc, ddst,
                 cd_desc, cd_ln, cd_lr, cd_nh, cd_nl, vlast, scale, ring_h, ring_l,
                 geo->typed_data(), F2(mom), F2(loc), F2(qs), F2(dstage));
}

// analysis (adjoint): ring (nprob, R) -> coef (ntot)
ffi::Error AnaImpl(cudaStream_t s, ffi::Buffer<ffi::C64> ring, PLAN_ARGS,
                   ffi::ResultBuffer<ffi::C64> coef, SCRATCH_ARGS, ffi::Span<const int64_t> lev) {
    const auto rd = ring.dimensions();
    return apply(s, 1, nullptr, F2(coef), nullptr, CF2(ring), rd[0], rd[1], lev,
                 leaf_off, leaf_sz, leaf_Q, nodes, sbase, dh, dl, tau, z, c, gidx, slot, dsrc, ddst,
                 cd_desc, cd_ln, cd_lr, cd_nh, cd_nl, vlast, scale, ring_h, ring_l,
                 geo->typed_data(), F2(mom), F2(loc), F2(qs), F2(dstage));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(gm_dc_synth, SynthImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<ffi::Buffer<ffi::C64>>() PLAN_BIND
        .Ret<ffi::Buffer<ffi::C64>>().Ret<ffi::Buffer<ffi::C64>>() SCRATCH_BIND
        .Attr<ffi::Span<const int64_t>>("levels"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(gm_dc_ana, AnaImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<ffi::Buffer<ffi::C64>>() PLAN_BIND
        .Ret<ffi::Buffer<ffi::C64>>() SCRATCH_BIND
        .Attr<ffi::Span<const int64_t>>("levels"));
