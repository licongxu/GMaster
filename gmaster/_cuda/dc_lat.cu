// GMaster divide-and-conquer spin-0 latitudinal transform: apply kernels and XLA FFI handlers.
//
// For each order m and parity p the orthonormal Legendre functions phi_k(y) = psi_{m+p+2k}(sqrt y)
// of y = cos^2 theta satisfy a symmetric three-term recurrence with Jacobi matrix T = V Y V^T.
// Synthesis f(y_r) = sum_k c_k phi_k(y_r) is applied as (Christoffel-Darboux)
//     f(y_r) = E_{n-1} phi_n(y_r) sum_j V[n-1, j] (V^T c)_j / (y_r - y_j),
// with V^T applied by the Cuppen divide-and-conquer tree of T: 16 x 16 leaf blocks (rebuilt on the fly), then one
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
#ifndef CD_MINB
#define CD_MINB 3      // cd_fmm blocks per SM the register allocation must allow (94 regs -> 2; 3: 107.6 -> 105.0 ms per field at 2048, 4 spills: 133 ms)
#endif
#ifndef FL
#define FL 32          // points of each set per FMM leaf in the merge kernels (8/16/32/64 swept at Nside 2048: 32 best)
#endif

struct Node { int base, nk, nd, poff, doff, goff, ng, pad; };   // goff/ng: this node's exact-gap records

__device__ __forceinline__ int org_of(int j, float tau) { return tau > 0.f ? j : j + 1; }

// d_o - d_i.  Adjacent poles use the stored gap: at the upper merge levels a pole of the left child
// and one of the right can nearly coincide (gaps far below 1e-7 near y = 1 at small m), where the
// double-float difference is only good to ~4e-15 absolute -- 8.8e-6 in the m = 0 analysis at
// Nside 2048 (LAPACK removes such pairs by Givens deflation instead).
__device__ __forceinline__ float pole_diff(const float* dh, const float* dl, const float* gp, int o, int i) {
    if (i == o - 1) return gp[i];
    if (i == o + 1) return -gp[o];
    return (dh[o] - dh[i]) + (dl[o] - dl[i]);
}
// lam_j - d_i = (d_o - d_i) + tau_j, o = org(j): exactly tau_j when i == o.
__device__ __forceinline__ float lam_minus_d(const float* dh, const float* dl, const float* gp, int j, float tau, int i) {
    return pole_diff(dh, dl, gp, org_of(j, tau), i) + tau;
}
__device__ __forceinline__ float rcp(float x) { return __fdividef(1.0f, x); }   // MUFU.RCP
// Hot-loop reciprocal of a double-float pole difference, clamped: the <= 3 terms nearest each
// target (origin pole and its neighbours) are then replaced by their exact values (tau, gap + tau)
// after the loop, which cancels the loop's term to rounding; the clamp only bounds what a
// near-coincident pair, or a root within 1e-17 of its pole, can inject into the running sum.
__device__ __forceinline__ float rcp_c(float x) { return fminf(fmaxf(rcp(x), -1e17f), 1e17f); }
__device__ __forceinline__ void fma2(float2& a, float f, float2 b) { a.x = fmaf(f, b.x, a.x); a.y = fmaf(f, b.y, a.y); }
__device__ __forceinline__ float2 mul2(float f, float2 b) { return make_float2(f * b.x, f * b.y); }
// Boxes are (centre c, radius r): c is any float inside the box and r bounds |(h - c) + l| over its
// double-float points, measured exactly.  (A box measured from float high parts needed a radius floor,
// 1e-6 |c|, that inflated the boxes where the Gauss nodes crowd towards y = 1 -- spacings ~1e-7 at
// small m -- and cost the separation the index-based interaction lists rely on: 1.6e-4 at Nside 4096.)
// s + c accumulates x with the rounding error of every addition carried in c (Knuth two-sum)
__device__ __forceinline__ void two_sum_acc(float& s, float& c, float x) {
    const float t = s + x, bp = t - s;
    c += (s - (t - bp)) + (x - bp);
    s = t;
}
__device__ __forceinline__ float up(float r) { return fmaxf(r * 1.0000010f, 1e-30f); }

// Adjacent gaps d_{i+1} - d_i: the double-float difference of the stored poles (good to ~3.6e-15
// absolute), except where the plan recorded the gap exactly (below 1e-6).  The dense gap array this
// replaces was 4 B per kept pole per level (2.3 GiB at Nside 4096).
template <bool WARP>
__device__ __forceinline__ void stage_gaps(const Node& nd, const float* dh, const float* dl,
                                           const int* __restrict__ gpr_i, const float* __restrict__ gpr_f,
                                           float* gp, int tid, int nth) {
    const int nk = nd.nk;
    for (int i = tid; i < nk; i += nth) gp[i] = i + 1 < nk ? (dh[i + 1] - dh[i]) + (dl[i + 1] - dl[i]) : 0.f;
    if (WARP) __syncwarp(); else __syncthreads();
    for (int r = tid; r < nd.ng; r += nth) gp[gpr_i[nd.goff + r]] = gpr_f[nd.goff + r];
}

__constant__ float BIN[P * P] = {1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f,9.0f,10.0f,11.0f,12.0f,1.0f,3.0f,6.0f,10.0f,15.0f,21.0f,28.0f,36.0f,45.0f,55.0f,66.0f,78.0f,1.0f,4.0f,10.0f,20.0f,35.0f,56.0f,84.0f,120.0f,165.0f,220.0f,286.0f,364.0f,1.0f,5.0f,15.0f,35.0f,70.0f,126.0f,210.0f,330.0f,495.0f,715.0f,1001.0f,1365.0f,1.0f,6.0f,21.0f,56.0f,126.0f,252.0f,462.0f,792.0f,1287.0f,2002.0f,3003.0f,4368.0f,1.0f,7.0f,28.0f,84.0f,210.0f,462.0f,924.0f,1716.0f,3003.0f,5005.0f,8008.0f,12376.0f,1.0f,8.0f,36.0f,120.0f,330.0f,792.0f,1716.0f,3432.0f,6435.0f,11440.0f,19448.0f,31824.0f,1.0f,9.0f,45.0f,165.0f,495.0f,1287.0f,3003.0f,6435.0f,12870.0f,24310.0f,43758.0f,75582.0f,1.0f,10.0f,55.0f,220.0f,715.0f,2002.0f,5005.0f,11440.0f,24310.0f,48620.0f,92378.0f,167960.0f,1.0f,11.0f,66.0f,286.0f,1001.0f,3003.0f,8008.0f,19448.0f,43758.0f,92378.0f,184756.0f,352716.0f,1.0f,12.0f,78.0f,364.0f,1365.0f,4368.0f,12376.0f,31824.0f,75582.0f,167960.0f,352716.0f,705432.0f};      // BIN[j*P + k] = C(j+k, j)
__constant__ float CKJ[P * P] = {1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,2.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,3.0f,3.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,4.0f,6.0f,4.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,5.0f,10.0f,10.0f,5.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,6.0f,15.0f,20.0f,15.0f,6.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,7.0f,21.0f,35.0f,35.0f,21.0f,7.0f,1.0f,0.0f,0.0f,0.0f,0.0f,1.0f,8.0f,28.0f,56.0f,70.0f,56.0f,28.0f,8.0f,1.0f,0.0f,0.0f,0.0f,1.0f,9.0f,36.0f,84.0f,126.0f,126.0f,84.0f,36.0f,9.0f,1.0f,0.0f,0.0f,1.0f,10.0f,45.0f,120.0f,210.0f,252.0f,210.0f,120.0f,45.0f,10.0f,1.0f,0.0f,1.0f,11.0f,55.0f,165.0f,330.0f,462.0f,462.0f,330.0f,165.0f,55.0f,11.0f,1.0f};      // CKJ[k*P + j] = C(k, j)

// ------------------------------------------------------------------ direct merge, one block per node
// dir 0: w[base+slot_k] = -c_k sum_i (z_i w[base+gidx_i]) / (lam_k - d_i)
// dir 1: w[base+gidx_k] =  z_k sum_j (c_j w[base+slot_j]) / (d_k - lam_j)
template <int NR>
__global__ void merge_direct(
    const Node* __restrict__ nodes, int nnode, int dir,
    const float* __restrict__ gdh, const float* __restrict__ gdl, const int* __restrict__ gpr_i,
    const float* __restrict__ gpr_f, const float* __restrict__ gtau, const float* __restrict__ gz, const float* __restrict__ gc,
    const short* __restrict__ ggidx, const short* __restrict__ gslot,
    const short* __restrict__ gdsrc, const short* __restrict__ gddst,
    float2* __restrict__ w)
{
    extern __shared__ float sm[];
    int nb = blockIdx.x;
    if (nb >= nnode) return;
    Node nd = nodes[nb];
    const int nk = nd.nk;
    float* dh = sm; float* dl = sm + nk; float* tau = sm + 2 * nk; float* gp = sm + 3 * nk;
    float2* q = reinterpret_cast<float2*>(sm + 4 * nk + 2);
    const float* pdh = gdh + nd.poff; const float* pdl = gdl + nd.poff; const float* pt = gtau + nd.poff;
    for (int i = threadIdx.x; i < nk; i += blockDim.x) {
        dh[i] = pdh[i]; dl[i] = pdl[i]; tau[i] = pt[i];
        const int src = dir == 0 ? ggidx[nd.poff + i] : gslot[nd.poff + i];
        const float f = dir == 0 ? gz[nd.poff + i] : gc[nd.poff + i];
        #pragma unroll
        for (int r = 0; r < NR; ++r) q[i * NR + r] = mul2(f, w[(nd.base + src) * NR + r]);
    }
    __syncthreads();
    stage_gaps<false>(nd, dh, dl, gpr_i, gpr_f, gp, threadIdx.x, blockDim.x);
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
                const float inv = rcp_c(((oh - dh[i]) + (ol - dl[i])) + tk);
                #pragma unroll
                for (int r = 0; r < NR; ++r) fma2(acc[r], inv, q[i * NR + r]);
            }
            #pragma unroll
            for (int e = -1; e <= 1; ++e) {                    // poles o-1, o, o+1: exact gap / 0
                const int i = o + e;
                if (i < 0 || i >= nk) continue;
                const float ex = e < 0 ? gp[i] : (e > 0 ? -gp[o] : 0.f);
                const float corr = rcp(ex + tk) - rcp_c(((oh - dh[i]) + (ol - dl[i])) + tk);
                #pragma unroll
                for (int r = 0; r < NR; ++r) fma2(acc[r], corr, q[i * NR + r]);
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
                const float inv = rcp_c((kh - dh[o]) + ((kl - dl[o]) - tj));
                #pragma unroll
                for (int r = 0; r < NR; ++r) fma2(acc[r], inv, q[j * NR + r]);
            }
            for (int j = max(k - 2, 0); j < min(k + 2, nk); ++j) {   // roots whose origin is k-1, k, k+1
                const float tj = tau[j];
                const int o = org_of(j, tj);
                if (o < k - 1 || o > k + 1) continue;
                const float ex = o == k + 1 ? -gp[k] : (o == k - 1 ? gp[o] : 0.f);   // d_k - d_o
                const float corr = rcp(ex - tj) - rcp_c((kh - dh[o]) + ((kl - dl[o]) - tj));
                #pragma unroll
                for (int r = 0; r < NR; ++r) fma2(acc[r], corr, q[j * NR + r]);
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
template <bool WARP>
__device__ __forceinline__ void team_sync() { if (WARP) __syncwarp(); else __syncthreads(); }

template <bool WARP>
__device__ void fmm_passes_one(int nlev, const int* lof, int bo, float* __restrict__ geo,
                               float2* __restrict__ mom, float2* __restrict__ loc, int NR, int r,
                               bool write_geo) {
    const int tid = WARP ? (threadIdx.x & 31) : threadIdx.x, nth = WARP ? 32 : blockDim.x;
    // M2M: M_k = sum_j C(k,j) be^(k-j) (al^j Mc_j) over both children
    for (int l = 1; l < nlev; ++l) {
        const int nbx = lof[l + 1] - lof[l], cb0 = lof[l - 1], pb0 = lof[l], nc = lof[l] - lof[l - 1];
        for (int b = tid; b < nbx; b += nth) {
            const int c1 = 2 * b, c2 = min(2 * b + 1, nc - 1);
            const float ca = geo[2 * (bo + cb0 + c1)], ra = geo[2 * (bo + cb0 + c1) + 1];
            const float cb = geo[2 * (bo + cb0 + c2)], rb = geo[2 * (bo + cb0 + c2) + 1];
            const float c = 0.5f * (ca + cb);             // children are close: c - ca is exact
            const float rp = up(fmaxf(fabsf(ca - c) + ra, fabsf(cb - c) + rb));
            const int g = bo + pb0 + b;
            if (write_geo) { geo[2 * g] = c; geo[2 * g + 1] = rp; }
            const float ir = 1.0f / rp;
            float2 Mp[P];
            #pragma unroll
            for (int k = 0; k < P; ++k) Mp[k] = make_float2(0.f, 0.f);
            for (int ch = c1; ch <= c2; ++ch) {
                const int gc_ = bo + cb0 + ch;
                const float al = geo[2 * gc_ + 1] * ir, be = (geo[2 * gc_] - c) * ir;
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
        team_sync<WARP>();
    }
    // M2L: L_j += (bb^j / D) sum_k C(j+k, j) (a^k M_k) per interaction-list box
    for (int l = 0; l < nlev; ++l) {
        const int nbx = lof[l + 1] - lof[l], b0 = lof[l];
        for (int b = tid; b < nbx; b += nth) {
            const int g = bo + b0 + b;
            const float C = geo[2 * g], R = geo[2 * g + 1];
            float2 La[P];
            #pragma unroll
            for (int j = 0; j < P; ++j) La[j] = make_float2(0.f, 0.f);
            const int pb = b >> 1;
            const int c0 = (l == nlev - 1) ? 0 : 2 * (pb - 1), c1 = (l == nlev - 1) ? nbx - 1 : 2 * (pb + 1) + 1;
            for (int cand = max(c0, 0); cand <= min(c1, nbx - 1); ++cand) {
                if (cand >= b - 1 && cand <= b + 1) continue;
                const int gs = bo + b0 + cand;
                const float iD = 1.0f / (C - geo[2 * gs]), a = geo[2 * gs + 1] * iD, bb = -R * iD;
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
        team_sync<WARP>();
    }
    // L2L: L_i += al^i sum_{j>=i} C(j,i) be^(j-i) Lp_j
    for (int l = nlev - 2; l >= 0; --l) {
        const int nbx = lof[l + 1] - lof[l], b0 = lof[l], pb0 = lof[l + 1];
        for (int b = tid; b < nbx; b += nth) {
            const int g = bo + b0 + b, gp = bo + pb0 + (b >> 1);
            const float iR = 1.0f / geo[2 * gp + 1];
            const float al = geo[2 * g + 1] * iR, be = (geo[2 * g] - geo[2 * gp]) * iR;
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
        team_sync<WARP>();
    }
}

template <int NR, bool WARP = false>
__device__ void fmm_passes(int nlev, const int* lof, int bo, float* __restrict__ geo,
                           float2* __restrict__ mom, float2* __restrict__ loc) {
    for (int r = 0; r < NR; ++r) fmm_passes_one<WARP>(nlev, lof, bo, geo, mom, loc, NR, r, r == 0);
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
template <int NR, bool WARP>
__global__ void merge_fmm(
    const Node* __restrict__ nodes, int nnode, int dir,
    const float* __restrict__ gdh, const float* __restrict__ gdl, const int* __restrict__ gpr_i,
    const float* __restrict__ gpr_f, const float* __restrict__ gtau, const float* __restrict__ gz, const float* __restrict__ gc,
    const short* __restrict__ ggidx, const short* __restrict__ gslot,
    const short* __restrict__ gdsrc, const short* __restrict__ gddst,
    float2* __restrict__ w,
    const int* __restrict__ sbase, float* __restrict__ geo, float2* __restrict__ mom, float2* __restrict__ loc,
    float2* __restrict__ qs, float2* __restrict__ dstage, float* __restrict__ rpos, int poff0, int doff0)
{
    // WARP: one warp per node (mid-size nodes, where a whole block would idle between barriers)
    const int nb = WARP ? blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5) : blockIdx.x;
    if (nb >= nnode) return;
    const Node nd = nodes[nb];
    const int nk = nd.nk, tid = WARP ? (threadIdx.x & 31) : threadIdx.x, nth = WARP ? 32 : blockDim.x;
    const float* pdh = gdh + nd.poff; const float* pdl = gdl + nd.poff; const float* pt = gtau + nd.poff;
    // scratch is level-local: levels run one after another, so it holds one level at a time
    float2* q = qs + (size_t)(nd.poff - poff0) * NR;
    float2* dst = dstage + (size_t)(nd.doff - doff0) * NR;
    float* rh = rpos + 3 * (size_t)(nd.poff - poff0);  // root positions (hi, lo) and gaps for this node
    float* rl = rh + nk;
    float* pgp = rh + 2 * nk;
    stage_gaps<WARP>(nd, pdh, pdl, gpr_i, gpr_f, pgp, tid, nth);
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
    team_sync<WARP>();
    for (int b = tid; b < nleaf; b += nth) {
        const int i0 = FL * b, i1 = min(FL * b + FL, nk);
        const float c = 0.5f * (fminf(sh[i0], th[i0]) + fmaxf(sh[i1 - 1], th[i1 - 1]));
        float r = 0.f;
        for (int i = i0; i < i1; ++i) r = fmaxf(r, fmaxf(fabsf((sh[i] - c) + sl[i]), fabsf((th[i] - c) + tl[i])));
        r = up(r);
        const int g = bo + b;
        geo[2 * g] = c; geo[2 * g + 1] = r;
        p2m<NR>(c, 1.0f / r, sh, sl, q, i0, i1, mom + (size_t)g * P * NR);
    }
    team_sync<WARP>();
    fmm_passes<NR, WARP>(nlev, lof, bo, geo, mom, loc);
    for (int k = tid; k < nk; k += nth) {
        const int b = k / FL, g = bo + b;
        const float eta = ((th[k] - geo[2 * g]) + tl[k]) / geo[2 * g + 1];
        float2 acc[NR];
        l2p<NR>(eta, loc + (size_t)g * P * NR, acc);
        const int s0 = max(FL * (b - 1), 0), s1 = min(FL * (b + 2), nk);
        if (dir == 0) {
            const float tk = pt[k];
            const int o = org_of(k, tk);
            const float oh = pdh[o], ol = pdl[o];
            for (int s = s0; s < s1; ++s) {
                const float inv = rcp_c(((oh - pdh[s]) + (ol - pdl[s])) + tk);
                #pragma unroll
                for (int r = 0; r < NR; ++r) fma2(acc[r], inv, q[s * NR + r]);
            }
            #pragma unroll
            for (int e = -1; e <= 1; ++e) {
                const int i = o + e;
                if (i < s0 || i >= s1) continue;
                const float ex = e < 0 ? pgp[i] : (e > 0 ? -pgp[o] : 0.f);
                const float corr = rcp(ex + tk) - rcp_c(((oh - pdh[i]) + (ol - pdl[i])) + tk);
                #pragma unroll
                for (int r = 0; r < NR; ++r) fma2(acc[r], corr, q[i * NR + r]);
            }
            const float cc = -gc[nd.poff + k];
            const int d = nd.base + gslot[nd.poff + k];
            #pragma unroll
            for (int r = 0; r < NR; ++r) w[d * NR + r] = mul2(cc, acc[r]);
        } else {
            const float kh = pdh[k], kl = pdl[k];
            for (int s = s0; s < s1; ++s) {
                const float ts = pt[s]; const int o = org_of(s, ts);
                const float inv = rcp_c((kh - pdh[o]) + ((kl - pdl[o]) - ts));
                #pragma unroll
                for (int r = 0; r < NR; ++r) fma2(acc[r], inv, q[s * NR + r]);
            }
            for (int s = max(k - 2, s0); s < min(k + 2, s1); ++s) {
                const float ts = pt[s]; const int o = org_of(s, ts);
                if (o < k - 1 || o > k + 1) continue;
                const float ex = o == k + 1 ? -pgp[k] : (o == k - 1 ? pgp[o] : 0.f);
                const float corr = rcp(ex - ts) - rcp_c((kh - pdh[o]) + ((kl - pdl[o]) - ts));
                #pragma unroll
                for (int r = 0; r < NR; ++r) fma2(acc[r], corr, q[s * NR + r]);
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

// ------------------------------------------------------------------ leaves (warp per <= 16 x 16 block)
// A leaf is the diagonal block [k0, k0 + s) of order (m, p)'s Jacobi matrix after the tree's tears
// (each internal boundary k subtracts E_{k-1} from both neighbours' diagonals), so its entries are
// closed-form.  Its eigenvectors are not stored: lane j rebuilds eigenvector j from the stored
// eigenvalue by a twisted factorisation (forward and backward LDL^T of T - lambda, started from
// the twist index of smallest |gamma|), normalised with first component positive -- the plan's
// convention.  Against fp64 eigh that is 2-4e-6 on unit vectors; the dense fp32 block it replaces
// was 64 bytes per coefficient (1.6 GiB at Nside 2048).
__device__ __forceinline__ float acoef(int l, int m) {
    return l > m ? sqrtf((float)(l - m) * (float)(l + m) / (4.f * (float)l * (float)l - 1.f)) : 0.f;
}

// One leaf per half-warp (s <= 16), 8 per 128-thread block; the rebuilt block is staged in
// shared memory so both directions read it row- or column-wise.
__device__ __forceinline__ float aspin(int l, int m, int s) {     // Wigner-d recurrence, x-form
    return l > max(m, s) ? sqrtf((float)(l - m) * (float)(l + m) * (float)(l - s) * (float)(l + s)) /
                               ((float)l * sqrtf(4.f * (float)l * (float)l - 1.f)) : 0.f;
}

template <int NR>
__global__ void __launch_bounds__(128) leaves(const int* __restrict__ off, const int* __restrict__ sz,
                       const int* __restrict__ mk, const float* __restrict__ lam, float2* __restrict__ w,
                       int nleaf, int dir, int spin) {
    __shared__ float Qs[8][16][17];
    const int hw = threadIdx.x >> 4, lane = threadIdx.x & 15;
    const int leaf = blockIdx.x * 8 + hw;
    const bool live = leaf < nleaf;
    const int o = live ? off[leaf] : 0, s = live ? sz[leaf] : 0;
    float v[16];
    if (live) {
        const int m = mk[4 * leaf], p = mk[4 * leaf + 1], k0 = mk[4 * leaf + 2], n = mk[4 * leaf + 3];
        float D[16], E[16];
        if (spin == 0) {                                  // y-form, parity p
            #pragma unroll
            for (int k = 0; k < 16; ++k) {
                const int l = m + p + 2 * (k0 + k);
                const float a1 = acoef(l + 1, m), a0 = acoef(l, m);
                D[k] = a1 * a1 + a0 * a0;
                E[k] = a1 * acoef(l + 2, m);
            }
            if (k0 > 0) { const int l = m + p + 2 * (k0 - 1); D[0] -= acoef(l + 1, m) * acoef(l + 2, m); }
        } else {                                          // x-form, l = max(m, s) + k
            const int M = max(m, spin);
            #pragma unroll
            for (int k = 0; k < 16; ++k) {
                const int l = M + k0 + k;
                D[k] = -(float)m * (float)spin / ((float)l * (float)(l + 1));
                E[k] = aspin(l + 1, m, spin);
            }
            if (k0 > 0) D[0] -= aspin(M + k0, m, spin);
        }
        #pragma unroll
        for (int k = 0; k < 16; ++k) if (k == s - 1 && k0 + s < n) D[k] -= E[k];
        const float lj = lane < s ? lam[o + lane] : 0.f;
        float dp[16], dm[16];
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            const float sh = D[k] - lj;
            dp[k] = k == 0 ? sh : sh - E[k - 1] * E[k - 1] * rcp(dp[k - 1]);
            if (dp[k] == 0.f) dp[k] = 1e-30f;
        }
        #pragma unroll
        for (int k = 15; k >= 0; --k) {
            const float sh = D[k] - lj;
            dm[k] = (k >= s - 1) ? sh : sh - E[k] * E[k] * rcp(dm[k + 1]);
            if (dm[k] == 0.f) dm[k] = 1e-30f;
        }
        int r = 0; float best = 3.4e38f;
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            const float g = fabsf(dp[k] + dm[k] - (D[k] - lj));
            if (k < s && g < best) { best = g; r = k; }
        }
        #pragma unroll
        for (int k = 15; k >= 0; --k) v[k] = (k == r) ? 1.f : ((k < r) ? -E[k] * v[min(k + 1, 15)] * rcp(dp[k]) : 0.f);
        #pragma unroll
        for (int k = 0; k < 16; ++k) if (k > r && k < s) v[k] = -E[k - 1] * v[k - 1] * rcp(dm[k]);
        float nrm = 0.f;
        #pragma unroll
        for (int k = 0; k < 16; ++k) nrm += k < s ? v[k] * v[k] : 0.f;
        const float scl = copysignf(rsqrtf(nrm), v[0]);
        #pragma unroll
        for (int k = 0; k < 16; ++k) { v[k] = (k < s && lane < s) ? v[k] * scl : 0.f; Qs[hw][k][lane] = v[k]; }
    }
    __syncthreads();
    if (!live) return;
    float2 x[NR];
    #pragma unroll
    for (int rr = 0; rr < NR; ++rr) x[rr] = lane < s ? w[(o + lane) * NR + rr] : make_float2(0.f, 0.f);
    float2 acc[NR];
    #pragma unroll
    for (int rr = 0; rr < NR; ++rr) acc[rr] = make_float2(0.f, 0.f);
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        // dir 0: out_j = sum_k Q[k, j] x_k;  dir 1: out_k = sum_j Q[k, j] x_j  (lane = output index)
        const float q = dir == 0 ? v[k] : Qs[hw][lane][k];
        #pragma unroll
        for (int rr = 0; rr < NR; ++rr)
            fma2(acc[rr], q, make_float2(__shfl_sync(0xffffffffu, x[rr].x, k, 16), __shfl_sync(0xffffffffu, x[rr].y, k, 16)));
    }
    if (lane < s)
        #pragma unroll
        for (int rr = 0; rr < NR; ++rr) w[(o + lane) * NR + rr] = acc[rr];
}

// ------------------------------------------------------------------ Christoffel-Darboux step
// Per problem (m, p): nodes y_j (the root Gauss nodes, ascending, double-float) and the first nt
// rings in ascending y (the rest are in the forbidden region and skipped).  Leaves are contiguous
// ranges of the merged order: ln0 (node ranges) and lr0 (ring ranges), nleaf + 1 entries each.
// dir 0: ring[p, R-1-t] = scale[p, R-1-t] * sum_j vlast_j w[so+j] / (y_t - y_j)
// dir 1: w[so+j] = -vlast_j * sum_t scale ring / (y_j - y_t)                   (adjoint)
struct CDProb { int so, n, lo, nleaf, bo, nt, t0, pad; };   // active rings: [t0, t0 + nt) ascending

template <int NR>
__global__ void __launch_bounds__(256, CD_MINB) cd_fmm(
    const CDProb* __restrict__ probs, int nprob, int dir,
    const float* __restrict__ yh, const float* __restrict__ yl, const float* __restrict__ vlast,
    const float* __restrict__ ryh, const float* __restrict__ ryl, int R,
    const int* __restrict__ ln0, const int* __restrict__ lr0, const float* __restrict__ scale,
    const float* __restrict__ der,
    float2* __restrict__ w, float2* __restrict__ ring,
    float* __restrict__ geo, float2* __restrict__ mom, float2* __restrict__ loc)
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
    // source strengths are formed where they are read (no staged copy): q_s = vlast w (dir 0) or
    // scale * ring (dir 1), both indexed in the ascending order of this direction's sources
    auto qv = [&](int s, int r) -> float2 {
        if (dir == 0) return mul2(vlast[pr.so + s], w[(pr.so + s) * NR + r]);
        const int rr = R - 1 - (pr.t0 + s);
        return mul2(sc[rr], rg[rr * NR + r]);
    };
    if (dir == 0)
        for (int t = tid; t < R; t += nth) if (t < pr.t0 || t >= pr.t0 + pr.nt)
            #pragma unroll
            for (int r = 0; r < NR; ++r) rg[(R - 1 - t) * NR + r] = make_float2(0.f, 0.f);
    // coarse leaves: cut b is the plan's cut min(CDF b, nleaf); staged in shared memory
    const int nleaf = (pr.nleaf + CDF - 1) / CDF;
    __shared__ int L0[1024], R0[1024];
    for (int b = tid; b <= nleaf; b += nth) { const int c = min(CDF * b, pr.nleaf); L0[b] = L0p[c]; R0[b] = R0p[c]; }
    int nlev = 1, sz = nleaf, lof[24]; lof[0] = 0;
    while (true) { lof[nlev] = lof[nlev - 1] + sz; if (sz <= 3) break; sz = (sz + 1) / 2; ++nlev; }
    const int bo = pr.bo;
    const float* rh_ = ryh + pr.t0; const float* rl_ = ryl + pr.t0;     // active rings, ascending
    const float* sh = dir == 0 ? nh : rh_; const float* sl = dir == 0 ? nl : rl_;
    const float* th = dir == 0 ? rh_ : nh; const float* tl = dir == 0 ? rl_ : nl;
    const int* S0 = dir == 0 ? L0 : R0;
    const int* T0 = dir == 0 ? R0 : L0;
    __syncthreads();
    for (int b = tid; b < nleaf; b += nth) {
        const int n0 = L0[b], n1 = L0[b + 1], r0 = R0[b], r1 = R0[b + 1];
        float lo = 1e30f, hi = -1e30f;
        if (n1 > n0) { lo = fminf(lo, nh[n0]); hi = fmaxf(hi, nh[n1 - 1]); }
        if (r1 > r0) { lo = fminf(lo, rh_[r0]); hi = fmaxf(hi, rh_[r1 - 1]); }
        const float c = 0.5f * (lo + hi);
        float rad = 0.f;
        for (int i = n0; i < n1; ++i) rad = fmaxf(rad, fabsf((nh[i] - c) + nl[i]));
        for (int i = r0; i < r1; ++i) rad = fmaxf(rad, fabsf((rh_[i] - c) + rl_[i]));
        rad = up(rad);
        const int g = bo + b;
        geo[2 * g] = c; geo[2 * g + 1] = rad;
        const float ir = 1.0f / rad;
        for (int r = 0; r < NR; ++r) {
            float2 M[P];
            #pragma unroll
            for (int k = 0; k < P; ++k) M[k] = make_float2(0.f, 0.f);
            for (int s = S0[b]; s < S0[b + 1]; ++s) {
                const float xi = ((sh[s] - c) + sl[s]) * ir;
                const float2 qq = qv(s, r);
                float pw = 1.f;
                #pragma unroll
                for (int k = 0; k < P; ++k) { fma2(M[k], pw, qq); pw *= xi; }
            }
            #pragma unroll
            for (int k = 0; k < P; ++k) mom[((size_t)g * P + k) * NR + r] = M[k];
        }
    }
    __syncthreads();
    fmm_passes<NR>(nlev, lof, bo, geo, mom, loc);
    const int ntg = dir == 0 ? pr.nt : pr.n;
    for (int t = tid; t < ntg; t += nth) {
        int lo_b = 0, hi_b = nleaf;
        while (hi_b - lo_b > 1) { const int mid = (lo_b + hi_b) >> 1; if (T0[mid] <= t) lo_b = mid; else hi_b = mid; }
        const int b = lo_b, g = bo + b;
        const float eta = ((th[t] - geo[2 * g]) + tl[t]) / geo[2 * g + 1];
        float2 acc[NR];
        l2p<NR>(eta, loc + (size_t)g * P * NR, acc);
        // Near field with compensated (two-sum) accumulation: at small m the Gauss nodes and the
        // dense polar rings interleave where phi_n' ~ n^2, so the analysis direction sums terms
        // ~1e7 that cancel to O(1) (m = 0 analysis was 1.3e-5 at Nside 2048 with a plain sum).
        // Near field, clamped reciprocal: node-ring pairs closer than 1e-9 (where the double-float
        // difference loses digits, or the term is 0/0 at a coincidence) are replaced by exact terms
        // afterwards (`cd_fix`, from the plan's list).
        const int s0 = S0[max(b - 1, 0)], s1 = S0[min(b + 1, nleaf - 1) + 1];
        for (int s = s0; s < s1; ++s) {
            const float inv = rcp_c((th[t] - sh[s]) + (tl[t] - sl[s]));
            #pragma unroll
            for (int r = 0; r < NR; ++r) fma2(acc[r], inv, qv(s, r));
        }
        if (dir == 0) {
            const int rr = R - 1 - (pr.t0 + t); const float f = sc[rr];
            #pragma unroll
            for (int r = 0; r < NR; ++r) rg[rr * NR + r] = mul2(f, acc[r]);
        } else {
            const float f = -vlast[pr.so + t];
            #pragma unroll
            for (int r = 0; r < NR; ++r) w[(pr.so + t) * NR + r] = mul2(f, acc[r]);
        }
    }
}

// Exact terms for the plan's near node-ring pairs (see cd_fmm's near field).  Record q: problem pb,
// node j, ascending ring index ta, exact y_r - y_j (0: coincident -> limit E phi_n'(y_j) = der_j).
template <int NR>
__global__ void cd_fix(const int* __restrict__ cdx_i, const float* __restrict__ cdx_f, int nrec, int dir,
                       const CDProb* __restrict__ probs, const float* __restrict__ nh, const float* __restrict__ nl,
                       const float* __restrict__ vlast, const float* __restrict__ ryh, const float* __restrict__ ryl,
                       int R, const float* __restrict__ scale, const float* __restrict__ der,
                       float2* __restrict__ w, float2* __restrict__ ring)
{
    const int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= nrec) return;
    const int pb = cdx_i[3 * qi], j = cdx_i[3 * qi + 1], ta = cdx_i[3 * qi + 2];
    const float ex = cdx_f[qi];
    const int so = probs[pb].so, rr = R - 1 - ta;
    const float sc = scale[(size_t)pb * R + rr];
    const float dfd = (ryh[ta] - nh[so + j]) + (ryl[ta] - nl[so + j]);   // y_r - y_j as cd_fmm formed it
    float2* rg = ring + (size_t)pb * R * NR;
    if (dir == 0) {                     // ring rr holds sc * sum_j q_j * rcp_c(y_r - y_j)
        const float bad = sc * rcp_c(dfd);
        const float good = ex == 0.f ? der[so + j] : sc / ex;
        const float vj = vlast[so + j];
        for (int r = 0; r < NR; ++r) {
            const float2 qj = mul2(vj, w[(so + j) * NR + r]);
            atomicAdd(&rg[rr * NR + r].x, (good - bad) * qj.x);
            atomicAdd(&rg[rr * NR + r].y, (good - bad) * qj.y);
        }
    } else {                            // node j holds -vlast_j * sum_r sc_r g_r * rcp_c(y_j - y_r)
        const float bad = sc * rcp_c(-dfd);
        const float good = ex == 0.f ? -der[so + j] : -sc / ex;
        const float f = -vlast[so + j];
        for (int r = 0; r < NR; ++r) {
            const float2 g = rg[rr * NR + r];
            atomicAdd(&w[(so + j) * NR + r].x, f * (good - bad) * g.x);
            atomicAdd(&w[(so + j) * NR + r].y, f * (good - bad) * g.y);
        }
    }
}

// ================================================================================ FFI handlers
static constexpr int DIRECT_THREADS = 128;
static constexpr int FMM_THREADS = 256;
#ifndef WARP_MAX_K
#define WARP_MAX_K 1024   // FMM merges with at most this many kept poles run one warp per node
#endif

#define PLAN_ARGS \
    ffi::Buffer<ffi::S32> leaf_off, ffi::Buffer<ffi::S32> leaf_sz, ffi::Buffer<ffi::S32> leaf_mk, ffi::Buffer<ffi::F32> leaf_lam, \
    ffi::Buffer<ffi::S32> nodes, ffi::Buffer<ffi::S32> sbase, ffi::Buffer<ffi::F32> dh, ffi::Buffer<ffi::F32> dl, \
    ffi::Buffer<ffi::S32> gpr_i, ffi::Buffer<ffi::F32> gpr_f, \
    ffi::Buffer<ffi::F32> tau, ffi::Buffer<ffi::F32> z, ffi::Buffer<ffi::F32> c, \
    ffi::Buffer<ffi::S16> gidx, ffi::Buffer<ffi::S16> slot, ffi::Buffer<ffi::S16> dsrc, ffi::Buffer<ffi::S16> ddst, \
    ffi::Buffer<ffi::S32> cd_desc, ffi::Buffer<ffi::S32> cd_ln, ffi::Buffer<ffi::S32> cd_lr, \
    ffi::Buffer<ffi::F32> cd_nh, ffi::Buffer<ffi::F32> cd_nl, ffi::Buffer<ffi::F32> vlast, ffi::Buffer<ffi::F32> scale, \
    ffi::Buffer<ffi::F32> ring_h, ffi::Buffer<ffi::F32> ring_l, ffi::Buffer<ffi::F32> cd_der, \
    ffi::Buffer<ffi::S32> cdx_i, ffi::Buffer<ffi::F32> cdx_f

#define PLAN_PASS leaf_off, leaf_sz, leaf_mk, leaf_lam, nodes, sbase, dh, dl, gpr_i, gpr_f, tau, z, c, gidx, slot, dsrc, ddst, \
    cd_desc, cd_ln, cd_lr, cd_nh, cd_nl, vlast, scale, ring_h, ring_l, cd_der, cdx_i, cdx_f

#define PLAN_BIND \
    .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::F32>>() \
    .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>() \
    .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::F32>>() \
    .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>() \
    .Arg<ffi::Buffer<ffi::S16>>().Arg<ffi::Buffer<ffi::S16>>().Arg<ffi::Buffer<ffi::S16>>().Arg<ffi::Buffer<ffi::S16>>() \
    .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>() \
    .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>() \
    .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>() \
    .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::F32>>()

// scratch (all sized for the largest single tree level, or for the CD step if larger):
// geo (2 nbox), mom / loc (P nbox NR), qs (nkept NR), dstage (ndefl NR), rpos (3 nkept)
#define SCRATCH_ARGS \
    ffi::ResultBuffer<ffi::F32> geo, ffi::ResultBuffer<ffi::C64> mom, ffi::ResultBuffer<ffi::C64> loc, \
    ffi::ResultBuffer<ffi::C64> qs, ffi::ResultBuffer<ffi::C64> dstage, ffi::ResultBuffer<ffi::F32> rpos

#define SCRATCH_BIND \
    .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::C64>>().Ret<ffi::Buffer<ffi::C64>>() \
    .Ret<ffi::Buffer<ffi::C64>>().Ret<ffi::Buffer<ffi::C64>>().Ret<ffi::Buffer<ffi::F32>>()

#define F2(b) reinterpret_cast<float2*>((b)->typed_data())
#define CF2(b) reinterpret_cast<const float2*>((b).typed_data())

template <int NR>
// stages: bit 0 runs the tree (leaves + merges), bit 1 the Christoffel-Darboux step.  CD-only calls
// map node-space values to rings and back, tree-only analysis maps node space to coefficients; the
// refinement loop runs in node space, where V^T V = I cancels the trees between iterations.
static ffi::Error apply(cudaStream_t s, int dir, int spin, int stages, const float2* in, float2* w, float2* ring_out,
                        const float2* ring_in, int nprob, int R, ffi::Span<const int64_t> lev, PLAN_ARGS,
                        float* g, float2* mo, float2* lo, float2* q, float2* dst, float* rp) {
    const int nleaf = leaf_off.element_count();
    const int lgrid = (nleaf + 7) / 8;
    const CDProb* cdp = reinterpret_cast<const CDProb*>(cd_desc.typed_data());
    const bool tree = stages & 1, cd = stages & 2;
    const size_t wbytes = sizeof(float2) * NR * vlast.element_count();
    // The tree works in place on w; a CD-only synthesis reads its input where it is (the node-space
    // loop makes three per iteration pair).
    float2* src = (dir == 0 && !tree) ? const_cast<float2*>(in) : w;
    if (dir == 0) {
        if (tree) cudaMemcpyAsync(w, in, wbytes, cudaMemcpyDeviceToDevice, s);
        if (tree) leaves<NR><<<lgrid, 128, 0, s>>>(leaf_off.typed_data(), leaf_sz.typed_data(), leaf_mk.typed_data(), leaf_lam.typed_data(), w, nleaf, 0, spin);
    } else if (!cd) {
        cudaMemcpyAsync(w, ring_in, wbytes, cudaMemcpyDeviceToDevice, s);   // node-space input
    } else {
        cd_fmm<NR><<<nprob, 256, 0, s>>>(cdp, nprob, 1, cd_nh.typed_data(), cd_nl.typed_data(), vlast.typed_data(),
            ring_h.typed_data(), ring_l.typed_data(), R, cd_ln.typed_data(), cd_lr.typed_data(), scale.typed_data(), cd_der.typed_data(),
            w, const_cast<float2*>(ring_in), g, mo, lo);
        {
            const int nrec = cdx_f.element_count();
            if (nrec) cd_fix<NR><<<(nrec + 127) / 128, 128, 0, s>>>(cdx_i.typed_data(), cdx_f.typed_data(), nrec, 1, cdp,
                cd_nh.typed_data(), cd_nl.typed_data(), vlast.typed_data(), ring_h.typed_data(), ring_l.typed_data(), R,
                scale.typed_data(), cd_der.typed_data(), w, const_cast<float2*>(ring_in));
        }
    }
    // levels: (kind, nnode, node0, maxk, sb0, poff0, doff0) per tree level
    const int nlev = tree ? (int)(lev.size() / 7) : 0;
    for (int qq = 0; qq < nlev; ++qq) {
        const int i = dir == 0 ? qq : nlev - 1 - qq;
        const int kind = lev[7 * i], nn = lev[7 * i + 1], node0 = lev[7 * i + 2], maxk = lev[7 * i + 3], sb0 = lev[7 * i + 4];
        const int poff0 = lev[7 * i + 5], doff0 = lev[7 * i + 6];
        const Node* nd = reinterpret_cast<const Node*>(nodes.typed_data()) + node0;
        if (kind == 0) {
            const int smem = 4 * (4 * maxk + 2) + 8 * NR * maxk;
            merge_direct<NR><<<nn, DIRECT_THREADS, smem, s>>>(nd, nn, dir, dh.typed_data(), dl.typed_data(),
                gpr_i.typed_data(), gpr_f.typed_data(), tau.typed_data(), z.typed_data(), c.typed_data(), gidx.typed_data(), slot.typed_data(),
                dsrc.typed_data(), ddst.typed_data(), w);
        } else {
            if (maxk <= WARP_MAX_K)
                merge_fmm<NR, true><<<(nn + 7) / 8, 256, 0, s>>>(nd, nn, dir, dh.typed_data(), dl.typed_data(),
                    gpr_i.typed_data(), gpr_f.typed_data(), tau.typed_data(), z.typed_data(), c.typed_data(), gidx.typed_data(), slot.typed_data(),
                    dsrc.typed_data(), ddst.typed_data(), w, sbase.typed_data() + sb0, g, mo, lo, q, dst, rp, poff0, doff0);
            else
                merge_fmm<NR, false><<<nn, FMM_THREADS, 0, s>>>(nd, nn, dir, dh.typed_data(), dl.typed_data(),
                    gpr_i.typed_data(), gpr_f.typed_data(), tau.typed_data(), z.typed_data(), c.typed_data(), gidx.typed_data(), slot.typed_data(),
                    dsrc.typed_data(), ddst.typed_data(), w, sbase.typed_data() + sb0, g, mo, lo, q, dst, rp, poff0, doff0);
        }
    }
    if (dir == 0 && cd) {
        cd_fmm<NR><<<nprob, 256, 0, s>>>(cdp, nprob, 0, cd_nh.typed_data(), cd_nl.typed_data(), vlast.typed_data(),
            ring_h.typed_data(), ring_l.typed_data(), R, cd_ln.typed_data(), cd_lr.typed_data(), scale.typed_data(), cd_der.typed_data(),
            src, ring_out, g, mo, lo);
        {
            const int nrec = cdx_f.element_count();
            if (nrec) cd_fix<NR><<<(nrec + 127) / 128, 128, 0, s>>>(cdx_i.typed_data(), cdx_f.typed_data(), nrec, 0, cdp,
                cd_nh.typed_data(), cd_nl.typed_data(), vlast.typed_data(), ring_h.typed_data(), ring_l.typed_data(), R,
                scale.typed_data(), cd_der.typed_data(), src, ring_out);
        }
    } else if (dir == 1 && tree) {
        leaves<NR><<<lgrid, 128, 0, s>>>(leaf_off.typed_data(), leaf_sz.typed_data(), leaf_mk.typed_data(), leaf_lam.typed_data(), w, nleaf, 1, spin);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(err));
    return ffi::Error::Success();
}

// synthesis: coef (ntot, NR) -> ring (nprob, R, NR); work (ntot, NR) is the tree's in-place vector
template <int NR>
ffi::Error SynthImpl(cudaStream_t s, ffi::Buffer<ffi::C64> coef, PLAN_ARGS,
                     ffi::ResultBuffer<ffi::C64> ring, ffi::ResultBuffer<ffi::C64> work, SCRATCH_ARGS,
                     ffi::Span<const int64_t> lev, int64_t spin, int64_t stages) {
    const auto rd = ring->dimensions();
    return apply<NR>(s, 0, (int)spin, (int)stages, CF2(coef), F2(work), F2(ring), nullptr, rd[0], rd[1], lev, PLAN_PASS,
                     geo->typed_data(), F2(mom), F2(loc), F2(qs), F2(dstage), rpos->typed_data());
}

// analysis (adjoint): ring (nprob, R, NR) -> coef (ntot, NR)
template <int NR>
ffi::Error AnaImpl(cudaStream_t s, ffi::Buffer<ffi::C64> ring, PLAN_ARGS,
                   ffi::ResultBuffer<ffi::C64> coef, SCRATCH_ARGS, ffi::Span<const int64_t> lev, int64_t spin,
                   int64_t stages) {
    const auto rd = ring.dimensions();
    return apply<NR>(s, 1, (int)spin, (int)stages, nullptr, F2(coef), nullptr, CF2(ring), rd[0], rd[1], lev, PLAN_PASS,
                     geo->typed_data(), F2(mom), F2(loc), F2(qs), F2(dstage), rpos->typed_data());
}

#define SYNTH_BIND ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<ffi::Buffer<ffi::C64>>() PLAN_BIND \
        .Ret<ffi::Buffer<ffi::C64>>().Ret<ffi::Buffer<ffi::C64>>() SCRATCH_BIND .Attr<ffi::Span<const int64_t>>("levels").Attr<int64_t>("spin").Attr<int64_t>("stages")
#define ANA_BIND_DC ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<ffi::Buffer<ffi::C64>>() PLAN_BIND \
        .Ret<ffi::Buffer<ffi::C64>>() SCRATCH_BIND .Attr<ffi::Span<const int64_t>>("levels").Attr<int64_t>("spin").Attr<int64_t>("stages")

XLA_FFI_DEFINE_HANDLER_SYMBOL(gm_dc_synth, SynthImpl<1>, SYNTH_BIND);
XLA_FFI_DEFINE_HANDLER_SYMBOL(gm_dc_synth2, SynthImpl<2>, SYNTH_BIND);
XLA_FFI_DEFINE_HANDLER_SYMBOL(gm_dc_ana, AnaImpl<1>, ANA_BIND_DC);
XLA_FFI_DEFINE_HANDLER_SYMBOL(gm_dc_ana2, AnaImpl<2>, ANA_BIND_DC);
