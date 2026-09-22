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
//
// Every kernel carries NR right-hand sides (interleaved: value (i, r) at [i * NR + r]).  The plan
// geometry -- reciprocals of pole-root gaps, FMM translation factors, index maps -- is shared, so
// a field and its mask, which the MASTER pipeline always transforms in lockstep, cost one
// traversal instead of two.
#include <cuda_runtime.h>
#include <cstdint>
#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;
#define P 12
#ifndef CDF
#define CDF 4          // CD leaves: this many of the plan's 16-point merged leaves per FMM leaf
#endif
#ifndef FL
#define FL 32          // points of each set per FMM leaf in the merge kernels (8/16/32/64 swept at Nside 2048: 32 best)
#endif

struct Node { int base, nk, nd, poff, doff, pad0, pad1, pad2; };

__device__ __forceinline__ int org_of(int j, float tau) { return tau > 0.f ? j : j + 1; }

// lam_j - d_i = ((d_o - d_i) + tau_j), o = org(j): exactly tau_j when i == o, with no branch.
__device__ __forceinline__ float lam_minus_d(const float* dh, const float* dl, int j, float tau, int i) {
    int o = org_of(j, tau);
    return ((dh[o] - dh[i]) + (dl[o] - dl[i])) + tau;
}
__device__ __forceinline__ float rcp(float x) { return __fdividef(1.0f, x); }   // MUFU.RCP
__device__ __forceinline__ void fma2(float2& a, float f, float2 b) { a.x = fmaf(f, b.x, a.x); a.y = fmaf(f, b.y, a.y); }
__device__ __forceinline__ float2 mul2(float f, float2 b) { return make_float2(f * b.x, f * b.y); }
__device__ __forceinline__ float box_r(float lo, float hi) { return fmaxf(0.5f * (hi - lo), 1e-6f * fabsf(0.5f * (lo + hi)) + 1e-30f); }

__constant__ float BIN[P * P] = {1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f,9.0f,10.0f,11.0f,12.0f,1.0f,3.0f,6.0f,10.0f,15.0f,21.0f,28.0f,36.0f,45.0f,55.0f,66.0f,78.0f,1.0f,4.0f,10.0f,20.0f,35.0f,56.0f,84.0f,120.0f,165.0f,220.0f,286.0f,364.0f,1.0f,5.0f,15.0f,35.0f,70.0f,126.0f,210.0f,330.0f,495.0f,715.0f,1001.0f,1365.0f,1.0f,6.0f,21.0f,56.0f,126.0f,252.0f,462.0f,792.0f,1287.0f,2002.0f,3003.0f,4368.0f,1.0f,7.0f,28.0f,84.0f,210.0f,462.0f,924.0f,1716.0f,3003.0f,5005.0f,8008.0f,12376.0f,1.0f,8.0f,36.0f,120.0f,330.0f,792.0f,1716.0f,3432.0f,6435.0f,11440.0f,19448.0f,31824.0f,1.0f,9.0f,45.0f,165.0f,495.0f,1287.0f,3003.0f,6435.0f,12870.0f,24310.0f,43758.0f,75582.0f,1.0f,10.0f,55.0f,220.0f,715.0f,2002.0f,5005.0f,11440.0f,24310.0f,48620.0f,92378.0f,167960.0f,1.0f,11.0f,66.0f,286.0f,1001.0f,3003.0f,8008.0f,19448.0f,43758.0f,92378.0f,184756.0f,352716.0f,1.0f,12.0f,78.0f,364.0f,1365.0f,4368.0f,12376.0f,31824.0f,75582.0f,167960.0f,352716.0f,705432.0f};      // BIN[j*P + k] = C(j+k, j)
__constant__ float CKJ[P * P] = {1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,2.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,3.0f,3.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,4.0f,6.0f,4.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,5.0f,10.0f,10.0f,5.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,6.0f,15.0f,20.0f,15.0f,6.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,7.0f,21.0f,35.0f,35.0f,21.0f,7.0f,1.0f,0.0f,0.0f,0.0f,0.0f,1.0f,8.0f,28.0f,56.0f,70.0f,56.0f,28.0f,8.0f,1.0f,0.0f,0.0f,0.0f,1.0f,9.0f,36.0f,84.0f,126.0f,126.0f,84.0f,36.0f,9.0f,1.0f,0.0f,0.0f,1.0f,10.0f,45.0f,120.0f,210.0f,252.0f,210.0f,120.0f,45.0f,10.0f,1.0f,0.0f,1.0f,11.0f,55.0f,165.0f,330.0f,462.0f,462.0f,330.0f,165.0f,55.0f,11.0f,1.0f};      // CKJ[k*P + j] = C(k, j)

// ------------------------------------------------------------------ direct merge, one block per node
// dir 0: w[base+slot_k] = -c_k sum_i (z_i w[base+gidx_i]) / (lam_k - d_i)
// dir 1: w[base+gidx_k] =  z_k sum_j (c_j w[base+slot_j]) / (d_k - lam_j)
template <int NR>
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
    const int nk = nd.nk;
    float* dh = sm; float* dl = sm + nk; float* tau = sm + 2 * nk;
    float2* q = reinterpret_cast<float2*>(sm + 3 * nk + (nk & 1));
    const float* pdh = gdh + nd.poff; const float* pdl = gdl + nd.poff; const float* pt = gtau + nd.poff;
    for (int i = threadIdx.x; i < nk; i += blockDim.x) {
        dh[i] = pdh[i]; dl[i] = pdl[i]; tau[i] = pt[i];
        const int src = dir == 0 ? ggidx[nd.poff + i] : gslot[nd.poff + i];
        const float f = dir == 0 ? gz[nd.poff + i] : gc[nd.poff + i];
        #pragma unroll
        for (int r = 0; r < NR; ++r) q[i * NR + r] = mul2(f, w[(nd.base + src) * NR + r]);
    }
    float2 dv[NR];
    const int t = threadIdx.x;
    if (t < nd.nd) {
        const int src = dir == 0 ? gdsrc[nd.doff + t] : gddst[nd.doff + t];
        #pragma unroll
        for (int r = 0; r < NR; ++r) dv[r] = w[(nd.base + src) * NR + r];
    }
    __syncthreads();
    for (int k = threadIdx.x; k < nk; k += blockDim.x) {
        float2 acc[NR];
        #pragma unroll
        for (int r = 0; r < NR; ++r) acc[r] = make_float2(0.f, 0.f);
        if (dir == 0) {                // target root k, sources poles i
            const float tk = tau[k];
            const int o = org_of(k, tk);
            const float oh = dh[o], ol = dl[o];
            for (int i = 0; i < nk; ++i) {
                const float inv = rcp(((oh - dh[i]) + (ol - dl[i])) + tk);
                #pragma unroll
                for (int r = 0; r < NR; ++r) fma2(acc[r], inv, q[i * NR + r]);
            }
            const float cc = -gc[nd.poff + k];
            const int dst = nd.base + gslot[nd.poff + k];
            #pragma unroll
            for (int r = 0; r < NR; ++r) w[dst * NR + r] = mul2(cc, acc[r]);
        } else {                       // target pole k, sources roots j
            const float kh = dh[k], kl = dl[k];
            for (int j = 0; j < nk; ++j) {
                const float tj = tau[j];
                const int o = org_of(j, tj);
                const float inv = rcp((kh - dh[o]) + ((kl - dl[o]) - tj));
                #pragma unroll
                for (int r = 0; r < NR; ++r) fma2(acc[r], inv, q[j * NR + r]);
            }
            const float zz = gz[nd.poff + k];
            const int dst = nd.base + ggidx[nd.poff + k];
            #pragma unroll
            for (int r = 0; r < NR; ++r) w[dst * NR + r] = mul2(zz, acc[r]);
        }
    }
    if (t < nd.nd) {
        const int dst = dir == 0 ? gddst[nd.doff + t] : gdsrc[nd.doff + t];
        #pragma unroll
        for (int r = 0; r < NR; ++r) w[(nd.base + dst) * NR + r] = dv[r];
    }
}

// ------------------------------------------------------------------ 1-D FMM passes (block per problem)
// Box b at level l covers leaves [b 2^l, (b+1) 2^l); lof[l] is the first box of level l.  Moments
// and locals are P terms in the box's normalised coordinate, stored for NR right-hand sides as
// mom[(g * P + k) * NR + r].  The expansion passes run one right-hand side at a time: carrying
// NR sets of P complex terms per thread spilled (159 registers and a stack frame at NR = 2, 2.7x
// the NR = 1 time), while the geometry they would share is a few flops per box.
__device__ void fmm_passes_one(int nlev, const int* lof, int bo, float* __restrict__ geo,
                               float2* __restrict__ mom, float2* __restrict__ loc, int NR, int r,
                               bool write_geo) {
    const int tid = threadIdx.x, nth = blockDim.x;
    // M2M: M_k = sum_j C(k,j) be^(k-j) (al^j Mc_j) over both children
    for (int l = 1; l < nlev; ++l) {
        const int nbx = lof[l + 1] - lof[l], cb0 = lof[l - 1], pb0 = lof[l], nc = lof[l] - lof[l - 1];
        for (int b = tid; b < nbx; b += nth) {
            const int c1 = 2 * b, c2 = min(2 * b + 1, nc - 1);
            const float lo = fminf(geo[2 * (bo + cb0 + c1)], geo[2 * (bo + cb0 + c2)]);
            const float hi = fmaxf(geo[2 * (bo + cb0 + c1) + 1], geo[2 * (bo + cb0 + c2) + 1]);
            const int g = bo + pb0 + b;
            if (write_geo) { geo[2 * g] = lo; geo[2 * g + 1] = hi; }
            const float c = 0.5f * (lo + hi), ir = 1.0f / box_r(lo, hi);
            float2 Mp[P];
            #pragma unroll
            for (int k = 0; k < P; ++k) Mp[k] = make_float2(0.f, 0.f);
            for (int ch = c1; ch <= c2; ++ch) {
                const int gc_ = bo + cb0 + ch;
                const float clo = geo[2 * gc_], chi = geo[2 * gc_ + 1];
                const float al = box_r(clo, chi) * ir, be = (0.5f * (clo + chi) - c) * ir;
                float2 A[P]; float bp[P];
                float ap = 1.f; bp[0] = 1.f;
                #pragma unroll
                for (int j = 0; j < P; ++j) { A[j] = mul2(ap, mom[(gc_ * P + j) * NR + r]); ap *= al; if (j) bp[j] = bp[j - 1] * be; }
                #pragma unroll
                for (int k = 0; k < P; ++k)
                    #pragma unroll
                    for (int j = 0; j <= k; ++j) fma2(Mp[k], CKJ[k * P + j] * bp[k - j], A[j]);
            }
            #pragma unroll
            for (int k = 0; k < P; ++k) mom[(g * P + k) * NR + r] = Mp[k];
        }
        __syncthreads();
    }
    // M2L: L_j += (bb^j / D) sum_k C(j+k, j) (a^k M_k) per interaction-list box
    for (int l = 0; l < nlev; ++l) {
        const int nbx = lof[l + 1] - lof[l], b0 = lof[l];
        for (int b = tid; b < nbx; b += nth) {
            const int g = bo + b0 + b;
            const float lo = geo[2 * g], hi = geo[2 * g + 1];
            const float C = 0.5f * (lo + hi), R = box_r(lo, hi);
            float2 La[P];
            #pragma unroll
            for (int j = 0; j < P; ++j) La[j] = make_float2(0.f, 0.f);
            const int pb = b >> 1;
            const int c0 = (l == nlev - 1) ? 0 : 2 * (pb - 1), c1 = (l == nlev - 1) ? nbx - 1 : 2 * (pb + 1) + 1;
            for (int cand = max(c0, 0); cand <= min(c1, nbx - 1); ++cand) {
                if (cand >= b - 1 && cand <= b + 1) continue;
                const int gs = bo + b0 + cand;
                const float slo = geo[2 * gs], shi = geo[2 * gs + 1];
                const float iD = 1.0f / (C - 0.5f * (slo + shi)), a = box_r(slo, shi) * iD, bb = -R * iD;
                float2 A[P];
                float ak = 1.f;
                #pragma unroll
                for (int k = 0; k < P; ++k) { A[k] = mul2(ak, mom[(gs * P + k) * NR + r]); ak *= a; }
                float bj = iD;
                #pragma unroll
                for (int j = 0; j < P; ++j) {
                    float2 s2 = make_float2(0.f, 0.f);
                    #pragma unroll
                    for (int k = 0; k < P; ++k) fma2(s2, BIN[j * P + k], A[k]);
                    fma2(La[j], bj, s2);
                    bj *= bb;
                }
            }
            #pragma unroll
            for (int j = 0; j < P; ++j) loc[(g * P + j) * NR + r] = La[j];
        }
        __syncthreads();
    }
    // L2L: L_i += al^i sum_{j>=i} C(j,i) be^(j-i) Lp_j
    for (int l = nlev - 2; l >= 0; --l) {
        const int nbx = lof[l + 1] - lof[l], b0 = lof[l], pb0 = lof[l + 1];
        for (int b = tid; b < nbx; b += nth) {
            const int g = bo + b0 + b, gp = bo + pb0 + (b >> 1);
            const float lo = geo[2 * g], hi = geo[2 * g + 1];
            const float plo = geo[2 * gp], phi = geo[2 * gp + 1];
            const float iR = 1.0f / box_r(plo, phi);
            const float al = box_r(lo, hi) * iR, be = (0.5f * (lo + hi) - 0.5f * (plo + phi)) * iR;
            float2 Lp[P]; float bp[P]; bp[0] = 1.f;
            #pragma unroll
            for (int j = 0; j < P; ++j) { Lp[j] = loc[(gp * P + j) * NR + r]; if (j) bp[j] = bp[j - 1] * be; }
            float ap = 1.f;
            #pragma unroll
            for (int i = 0; i < P; ++i) {
                float2 acc = make_float2(0.f, 0.f);
                #pragma unroll
                for (int j = i; j < P; ++j) fma2(acc, CKJ[j * P + i] * bp[j - i], Lp[j]);
                fma2(loc[(g * P + i) * NR + r], ap, acc);
                ap *= al;
            }
        }
        __syncthreads();
    }
}

template <int NR>
__device__ void fmm_passes(int nlev, const int* lof, int bo, float* __restrict__ geo,
                           float2* __restrict__ mom, float2* __restrict__ loc) {
    for (int r = 0; r < NR; ++r) fmm_passes_one(nlev, lof, bo, geo, mom, loc, NR, r, r == 0);
}

// P2M of one right-hand side
__device__ __forceinline__ void p2m_one(float c, float ir, const float* h, const float* l, const float2* q,
                                        int s0, int s1, float2* mom_g, int NR, int r) {
    float2 M[P];
    #pragma unroll
    for (int k = 0; k < P; ++k) M[k] = make_float2(0.f, 0.f);
    for (int s = s0; s < s1; ++s) {
        const float xi = ((h[s] - c) + l[s]) * ir;
        const float2 qq = q[s * NR + r];
        float pw = 1.f;
        #pragma unroll
        for (int k = 0; k < P; ++k) { fma2(M[k], pw, qq); pw *= xi; }
    }
    #pragma unroll
    for (int k = 0; k < P; ++k) mom_g[k * NR + r] = M[k];
}

template <int NR>
__device__ __forceinline__ void p2m(float c, float ir, const float* h, const float* l, const float2* q,
                                    int s0, int s1, float2* mom_g) {
    for (int r = 0; r < NR; ++r) p2m_one(c, ir, h, l, q, s0, s1, mom_g, NR, r);
}

template <int NR>
__device__ __forceinline__ void l2p(float eta, const float2* loc_g, float2* acc) {
    #pragma unroll
    for (int r = 0; r < NR; ++r) {
        float2 a = make_float2(0.f, 0.f);
        #pragma unroll
        for (int j = P - 1; j >= 0; --j) {
            const float2 v = loc_g[j * NR + r];
            a = make_float2(fmaf(a.x, eta, v.x), fmaf(a.y, eta, v.y));
        }
        acc[r] = a;
    }
}

// ------------------------------------------------------------------ FMM merge, one block per node
// Leaves are FL poles + FL roots (interlaced).  Root positions are formed in double-float as
// d_org + tau; poles are stored double-float.
template <int NR>
__global__ void merge_fmm(
    const Node* __restrict__ nodes, int nnode, int dir,
    const float* __restrict__ gdh, const float* __restrict__ gdl, const float* __restrict__ gtau,
    const float* __restrict__ gz, const float* __restrict__ gc,
    const short* __restrict__ ggidx, const short* __restrict__ gslot,
    const short* __restrict__ gdsrc, const short* __restrict__ gddst,
    float2* __restrict__ w,
    const int* __restrict__ sbase, float* __restrict__ geo, float2* __restrict__ mom, float2* __restrict__ loc,
    float2* __restrict__ qs, float2* __restrict__ dstage, float* __restrict__ rpos)
{
    const int nb = blockIdx.x;
    if (nb >= nnode) return;
    const Node nd = nodes[nb];
    const int nk = nd.nk, tid = threadIdx.x, nth = blockDim.x;
    const float* pdh = gdh + nd.poff; const float* pdl = gdl + nd.poff; const float* pt = gtau + nd.poff;
    float2* q = qs + (size_t)nd.poff * NR;
    float2* dst = dstage + (size_t)nd.doff * NR;
    float* rh = rpos + 2 * (size_t)nd.poff;          // root positions (hi, lo) for this node
    float* rl = rh + nk;
    for (int i = tid; i < nk; i += nth) {
        const int src = dir == 0 ? ggidx[nd.poff + i] : gslot[nd.poff + i];
        const float f = dir == 0 ? gz[nd.poff + i] : gc[nd.poff + i];
        #pragma unroll
        for (int r = 0; r < NR; ++r) q[i * NR + r] = mul2(f, w[(nd.base + src) * NR + r]);
        const int o = org_of(i, pt[i]);
        const float h = pdh[o] + pt[i];
        rh[i] = h; rl[i] = pdl[o] + ((pdh[o] - h) + pt[i]);
    }
    for (int i = tid; i < nd.nd; i += nth) {
        const int src = dir == 0 ? gdsrc[nd.doff + i] : gddst[nd.doff + i];
        #pragma unroll
        for (int r = 0; r < NR; ++r) dst[i * NR + r] = w[(nd.base + src) * NR + r];
    }
    const int nleaf = (nk + FL - 1) / FL;
    int nlev = 1, sz = nleaf;
    int lof[20]; lof[0] = 0;
    while (true) { lof[nlev] = lof[nlev - 1] + sz; if (sz <= 3) break; sz = (sz + 1) / 2; ++nlev; }
    const int bo = sbase[nb];
    // sources: poles (dir 0) or roots (dir 1); targets the other set
    const float* sh = dir == 0 ? pdh : rh; const float* sl = dir == 0 ? pdl : rl;
    const float* th = dir == 0 ? rh : pdh; const float* tl = dir == 0 ? rl : pdl;
    __syncthreads();
    for (int b = tid; b < nleaf; b += nth) {
        const int i0 = FL * b, i1 = min(FL * b + FL, nk);
        const float lo = fminf(sh[i0], th[i0]), hi = fmaxf(sh[i1 - 1], th[i1 - 1]);
        const int g = bo + b;
        geo[2 * g] = lo; geo[2 * g + 1] = hi;
        p2m<NR>(0.5f * (lo + hi), 1.0f / box_r(lo, hi), sh, sl, q, i0, i1, mom + (size_t)g * P * NR);
    }
    __syncthreads();
    fmm_passes<NR>(nlev, lof, bo, geo, mom, loc);
    for (int k = tid; k < nk; k += nth) {
        const int b = k / FL, g = bo + b;
        const float lo = geo[2 * g], hi = geo[2 * g + 1];
        const float eta = ((th[k] - 0.5f * (lo + hi)) + tl[k]) / box_r(lo, hi);
        float2 acc[NR];
        l2p<NR>(eta, loc + (size_t)g * P * NR, acc);
        const int s0 = max(FL * (b - 1), 0), s1 = min(FL * (b + 2), nk);
        if (dir == 0) {
            const float tk = pt[k];
            for (int s = s0; s < s1; ++s) {
                const float inv = rcp(lam_minus_d(pdh, pdl, k, tk, s));
                #pragma unroll
                for (int r = 0; r < NR; ++r) fma2(acc[r], inv, q[s * NR + r]);
            }
            const float cc = -gc[nd.poff + k];
            const int d = nd.base + gslot[nd.poff + k];
            #pragma unroll
            for (int r = 0; r < NR; ++r) w[d * NR + r] = mul2(cc, acc[r]);
        } else {
            for (int s = s0; s < s1; ++s) {
                const float inv = rcp(-lam_minus_d(pdh, pdl, s, pt[s], k));
                #pragma unroll
                for (int r = 0; r < NR; ++r) fma2(acc[r], inv, q[s * NR + r]);
            }
            const float zz = gz[nd.poff + k];
            const int d = nd.base + ggidx[nd.poff + k];
            #pragma unroll
            for (int r = 0; r < NR; ++r) w[d * NR + r] = mul2(zz, acc[r]);
        }
    }
    for (int i = tid; i < nd.nd; i += nth) {
        const int d = dir == 0 ? gddst[nd.doff + i] : gdsrc[nd.doff + i];
        #pragma unroll
        for (int r = 0; r < NR; ++r) w[(nd.base + d) * NR + r] = dst[i * NR + r];
    }
}

// ------------------------------------------------------------------ leaves (warp per 16 x 16 block)
template <int NR>
__global__ void leaves(const int* __restrict__ off, const int* __restrict__ sz,
                       const float* __restrict__ Q, float2* __restrict__ w, int nleaf, int dir) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= nleaf) return;
    const int o = off[warp], s = sz[warp];
    const float* q = Q + 256 * warp;
    float2 x[NR], acc[NR];
    #pragma unroll
    for (int r = 0; r < NR; ++r) {
        x[r] = lane < s ? w[(o + lane) * NR + r] : make_float2(0.f, 0.f);
        acc[r] = make_float2(0.f, 0.f);
    }
    for (int k = 0; k < s; ++k) {
        const float v = lane < s ? (dir == 0 ? q[16 * k + lane] : q[16 * lane + k]) : 0.f;
        #pragma unroll
        for (int r = 0; r < NR; ++r)
            fma2(acc[r], v, make_float2(__shfl_sync(0xffffffffu, x[r].x, k), __shfl_sync(0xffffffffu, x[r].y, k)));
    }
    if (lane < s)
        #pragma unroll
        for (int r = 0; r < NR; ++r) w[(o + lane) * NR + r] = acc[r];
}

// ------------------------------------------------------------------ Christoffel-Darboux step
// Per problem (m, p): nodes y_j (the root Gauss nodes, ascending, double-float) and the first nt
// rings in ascending y (the rest are in the forbidden region and skipped).  Leaves are contiguous
// ranges of the merged order: ln0 (node ranges) and lr0 (ring ranges), nleaf + 1 entries each.
// dir 0: ring[p, R-1-t] = scale[p, R-1-t] * sum_j vlast_j w[so+j] / (y_t - y_j)
// dir 1: w[so+j] = -vlast_j * sum_t scale ring / (y_j - y_t)                   (adjoint)
struct CDProb { int so, n, lo, nleaf, bo, nt, pad0, pad1; };

template <int NR>
__global__ void cd_fmm(
    const CDProb* __restrict__ probs, int nprob, int dir,
    const float* __restrict__ yh, const float* __restrict__ yl, const float* __restrict__ vlast,
    const float* __restrict__ ryh, const float* __restrict__ ryl, int R,
    const int* __restrict__ ln0, const int* __restrict__ lr0, const float* __restrict__ scale,
    float2* __restrict__ w, float2* __restrict__ ring,
    float* __restrict__ geo, float2* __restrict__ mom, float2* __restrict__ loc, float2* __restrict__ qs)
{
    const int pb = blockIdx.x;
    if (pb >= nprob) return;
    const CDProb pr = probs[pb];
    const int tid = threadIdx.x, nth = blockDim.x;
    const int* L0p = ln0 + pr.lo + pb;          // the plan's leaf cuts (pr.nleaf + 1 entries)
    const int* R0p = lr0 + pr.lo + pb;
    const float* sc = scale + (size_t)pb * R;
    float2* rg = ring + (size_t)pb * R * NR;
    const float* nh = yh + pr.so; const float* nl = yl + pr.so;
    const int ns = dir == 0 ? pr.n : pr.nt;
    float2* q = qs + (dir == 0 ? (size_t)pr.so : (size_t)pb * R) * NR;
    for (int i = tid; i < ns; i += nth) {
        if (dir == 0) {
            const float f = vlast[pr.so + i];
            #pragma unroll
            for (int r = 0; r < NR; ++r) q[i * NR + r] = mul2(f, w[(pr.so + i) * NR + r]);
        } else {
            const int rr = R - 1 - i; const float f = sc[rr];
            #pragma unroll
            for (int r = 0; r < NR; ++r) q[i * NR + r] = mul2(f, rg[rr * NR + r]);
        }
    }
    if (dir == 0)
        for (int t = pr.nt + tid; t < R; t += nth)
            #pragma unroll
            for (int r = 0; r < NR; ++r) rg[(R - 1 - t) * NR + r] = make_float2(0.f, 0.f);
    // coarse leaves: cut b is the plan's cut min(CDF b, nleaf); staged in shared memory
    const int nleaf = (pr.nleaf + CDF - 1) / CDF;
    __shared__ int L0[1024], R0[1024];
    for (int b = tid; b <= nleaf; b += nth) { const int c = min(CDF * b, pr.nleaf); L0[b] = L0p[c]; R0[b] = R0p[c]; }
    int nlev = 1, sz = nleaf, lof[24]; lof[0] = 0;
    while (true) { lof[nlev] = lof[nlev - 1] + sz; if (sz <= 3) break; sz = (sz + 1) / 2; ++nlev; }
    const int bo = pr.bo;
    const float* sh = dir == 0 ? nh : ryh; const float* sl = dir == 0 ? nl : ryl;
    const float* th = dir == 0 ? ryh : nh; const float* tl = dir == 0 ? ryl : nl;
    const int* S0 = dir == 0 ? L0 : R0;
    const int* T0 = dir == 0 ? R0 : L0;
    __syncthreads();
    for (int b = tid; b < nleaf; b += nth) {
        const int n0 = L0[b], n1 = L0[b + 1], r0 = R0[b], r1 = R0[b + 1];
        float lo = 1e30f, hi = -1e30f;
        if (n1 > n0) { lo = fminf(lo, nh[n0]); hi = fmaxf(hi, nh[n1 - 1]); }
        if (r1 > r0) { lo = fminf(lo, ryh[r0]); hi = fmaxf(hi, ryh[r1 - 1]); }
        const int g = bo + b;
        geo[2 * g] = lo; geo[2 * g + 1] = hi;
        p2m<NR>(0.5f * (lo + hi), 1.0f / box_r(lo, hi), sh, sl, q, S0[b], S0[b + 1], mom + (size_t)g * P * NR);
    }
    __syncthreads();
    fmm_passes<NR>(nlev, lof, bo, geo, mom, loc);
    const int ntg = dir == 0 ? pr.nt : pr.n;
    for (int t = tid; t < ntg; t += nth) {
        int lo_b = 0, hi_b = nleaf;
        while (hi_b - lo_b > 1) { const int mid = (lo_b + hi_b) >> 1; if (T0[mid] <= t) lo_b = mid; else hi_b = mid; }
        const int b = lo_b, g = bo + b;
        const float lo = geo[2 * g], hi = geo[2 * g + 1];
        const float eta = ((th[t] - 0.5f * (lo + hi)) + tl[t]) / box_r(lo, hi);
        float2 acc[NR];
        l2p<NR>(eta, loc + (size_t)g * P * NR, acc);
        const int s0 = S0[max(b - 1, 0)], s1 = S0[min(b + 1, nleaf - 1) + 1];
        for (int s = s0; s < s1; ++s) {
            const float inv = rcp((th[t] - sh[s]) + (tl[t] - sl[s]));
            #pragma unroll
            for (int r = 0; r < NR; ++r) fma2(acc[r], inv, q[s * NR + r]);
        }
        if (dir == 0) {
            const int rr = R - 1 - t; const float f = sc[rr];
            #pragma unroll
            for (int r = 0; r < NR; ++r) rg[rr * NR + r] = mul2(f, acc[r]);
        } else {
            const float f = -vlast[pr.so + t];
            #pragma unroll
            for (int r = 0; r < NR; ++r) w[(pr.so + t) * NR + r] = mul2(f, acc[r]);
        }
    }
}

// ================================================================================ FFI handlers
static constexpr int DIRECT_THREADS = 128;
static constexpr int FMM_THREADS = 256;

#define PLAN_ARGS \
    ffi::Buffer<ffi::S32> leaf_off, ffi::Buffer<ffi::S32> leaf_sz, ffi::Buffer<ffi::F32> leaf_Q, \
    ffi::Buffer<ffi::S32> nodes, ffi::Buffer<ffi::S32> sbase, ffi::Buffer<ffi::F32> dh, ffi::Buffer<ffi::F32> dl, \
    ffi::Buffer<ffi::F32> tau, ffi::Buffer<ffi::F32> z, ffi::Buffer<ffi::F32> c, \
    ffi::Buffer<ffi::S16> gidx, ffi::Buffer<ffi::S16> slot, ffi::Buffer<ffi::S16> dsrc, ffi::Buffer<ffi::S16> ddst, \
    ffi::Buffer<ffi::S32> cd_desc, ffi::Buffer<ffi::S32> cd_ln, ffi::Buffer<ffi::S32> cd_lr, \
    ffi::Buffer<ffi::F32> cd_nh, ffi::Buffer<ffi::F32> cd_nl, ffi::Buffer<ffi::F32> vlast, ffi::Buffer<ffi::F32> scale, \
    ffi::Buffer<ffi::F32> ring_h, ffi::Buffer<ffi::F32> ring_l

#define PLAN_PASS leaf_off, leaf_sz, leaf_Q, nodes, sbase, dh, dl, tau, z, c, gidx, slot, dsrc, ddst, \
    cd_desc, cd_ln, cd_lr, cd_nh, cd_nl, vlast, scale, ring_h, ring_l

#define PLAN_BIND \
    .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::F32>>() \
    .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>() \
    .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>() \
    .Arg<ffi::Buffer<ffi::S16>>().Arg<ffi::Buffer<ffi::S16>>().Arg<ffi::Buffer<ffi::S16>>().Arg<ffi::Buffer<ffi::S16>>() \
    .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>() \
    .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>() \
    .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()

// scratch: geo (2 nbox), mom / loc (P nbox NR), qs (nq NR), dstage (ndefl NR), rpos (2 nkept)
#define SCRATCH_ARGS \
    ffi::ResultBuffer<ffi::F32> geo, ffi::ResultBuffer<ffi::C64> mom, ffi::ResultBuffer<ffi::C64> loc, \
    ffi::ResultBuffer<ffi::C64> qs, ffi::ResultBuffer<ffi::C64> dstage, ffi::ResultBuffer<ffi::F32> rpos

#define SCRATCH_BIND \
    .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::C64>>().Ret<ffi::Buffer<ffi::C64>>() \
    .Ret<ffi::Buffer<ffi::C64>>().Ret<ffi::Buffer<ffi::C64>>().Ret<ffi::Buffer<ffi::F32>>()

#define F2(b) reinterpret_cast<float2*>((b)->typed_data())
#define CF2(b) reinterpret_cast<const float2*>((b).typed_data())

template <int NR>
static ffi::Error apply(cudaStream_t s, int dir, const float2* in, float2* w, float2* ring_out,
                        const float2* ring_in, int nprob, int R, ffi::Span<const int64_t> lev, PLAN_ARGS,
                        float* g, float2* mo, float2* lo, float2* q, float2* dst, float* rp) {
    const int nleaf = leaf_off.element_count();
    const int lgrid = (nleaf * 32 + 127) / 128;
    const CDProb* cdp = reinterpret_cast<const CDProb*>(cd_desc.typed_data());
    if (dir == 0) {
        cudaMemcpyAsync(w, in, sizeof(float2) * NR * vlast.element_count(), cudaMemcpyDeviceToDevice, s);
        leaves<NR><<<lgrid, 128, 0, s>>>(leaf_off.typed_data(), leaf_sz.typed_data(), leaf_Q.typed_data(), w, nleaf, 0);
    } else {
        cd_fmm<NR><<<nprob, 256, 0, s>>>(cdp, nprob, 1, cd_nh.typed_data(), cd_nl.typed_data(), vlast.typed_data(),
            ring_h.typed_data(), ring_l.typed_data(), R, cd_ln.typed_data(), cd_lr.typed_data(), scale.typed_data(),
            w, const_cast<float2*>(ring_in), g, mo, lo, q);
    }
    const int nlev = lev.size() / 5;
    for (int qq = 0; qq < nlev; ++qq) {
        const int i = dir == 0 ? qq : nlev - 1 - qq;
        const int kind = lev[5 * i], nn = lev[5 * i + 1], node0 = lev[5 * i + 2], maxk = lev[5 * i + 3], sb0 = lev[5 * i + 4];
        const Node* nd = reinterpret_cast<const Node*>(nodes.typed_data()) + node0;
        if (kind == 0) {
            const int smem = 4 * (3 * maxk + 1) + 8 * NR * maxk;
            merge_direct<NR><<<nn, DIRECT_THREADS, smem, s>>>(nd, nn, dir, dh.typed_data(), dl.typed_data(),
                tau.typed_data(), z.typed_data(), c.typed_data(), gidx.typed_data(), slot.typed_data(),
                dsrc.typed_data(), ddst.typed_data(), w);
        } else {
            merge_fmm<NR><<<nn, FMM_THREADS, 0, s>>>(nd, nn, dir, dh.typed_data(), dl.typed_data(),
                tau.typed_data(), z.typed_data(), c.typed_data(), gidx.typed_data(), slot.typed_data(),
                dsrc.typed_data(), ddst.typed_data(), w, sbase.typed_data() + sb0, g, mo, lo, q, dst, rp);
        }
    }
    if (dir == 0) {
        cd_fmm<NR><<<nprob, 256, 0, s>>>(cdp, nprob, 0, cd_nh.typed_data(), cd_nl.typed_data(), vlast.typed_data(),
            ring_h.typed_data(), ring_l.typed_data(), R, cd_ln.typed_data(), cd_lr.typed_data(), scale.typed_data(),
            w, ring_out, g, mo, lo, q);
    } else {
        leaves<NR><<<lgrid, 128, 0, s>>>(leaf_off.typed_data(), leaf_sz.typed_data(), leaf_Q.typed_data(), w, nleaf, 1);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(err));
    return ffi::Error::Success();
}

// synthesis: coef (ntot, NR) -> ring (nprob, R, NR); work (ntot, NR) is the tree's in-place vector
template <int NR>
ffi::Error SynthImpl(cudaStream_t s, ffi::Buffer<ffi::C64> coef, PLAN_ARGS,
                     ffi::ResultBuffer<ffi::C64> ring, ffi::ResultBuffer<ffi::C64> work, SCRATCH_ARGS,
                     ffi::Span<const int64_t> lev) {
    const auto rd = ring->dimensions();
    return apply<NR>(s, 0, CF2(coef), F2(work), F2(ring), nullptr, rd[0], rd[1], lev, PLAN_PASS,
                     geo->typed_data(), F2(mom), F2(loc), F2(qs), F2(dstage), rpos->typed_data());
}

// analysis (adjoint): ring (nprob, R, NR) -> coef (ntot, NR)
template <int NR>
ffi::Error AnaImpl(cudaStream_t s, ffi::Buffer<ffi::C64> ring, PLAN_ARGS,
                   ffi::ResultBuffer<ffi::C64> coef, SCRATCH_ARGS, ffi::Span<const int64_t> lev) {
    const auto rd = ring.dimensions();
    return apply<NR>(s, 1, nullptr, F2(coef), nullptr, CF2(ring), rd[0], rd[1], lev, PLAN_PASS,
                     geo->typed_data(), F2(mom), F2(loc), F2(qs), F2(dstage), rpos->typed_data());
}

#define SYNTH_BIND ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<ffi::Buffer<ffi::C64>>() PLAN_BIND \
        .Ret<ffi::Buffer<ffi::C64>>().Ret<ffi::Buffer<ffi::C64>>() SCRATCH_BIND .Attr<ffi::Span<const int64_t>>("levels")
#define ANA_BIND_DC ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<ffi::Buffer<ffi::C64>>() PLAN_BIND \
        .Ret<ffi::Buffer<ffi::C64>>() SCRATCH_BIND .Attr<ffi::Span<const int64_t>>("levels")

XLA_FFI_DEFINE_HANDLER_SYMBOL(gm_dc_synth, SynthImpl<1>, SYNTH_BIND);
XLA_FFI_DEFINE_HANDLER_SYMBOL(gm_dc_synth2, SynthImpl<2>, SYNTH_BIND);
XLA_FFI_DEFINE_HANDLER_SYMBOL(gm_dc_ana, AnaImpl<1>, ANA_BIND_DC);
XLA_FFI_DEFINE_HANDLER_SYMBOL(gm_dc_ana2, AnaImpl<2>, ANA_BIND_DC);
