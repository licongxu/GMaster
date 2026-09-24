// Plan builder for the divide-and-conquer spin-0 latitudinal transform (fp64, OpenMP).
// See gmaster/_dc_lat.py for the algorithm and gmaster/_cuda/dc_lat.cu for the apply.
//
// One problem = (order m, parity p): the orthonormal functions phi_k(y) = psi_{m+p+2k}(sqrt y)
// obey y phi_k = E_k phi_{k+1} + D_k phi_k + E_{k-1} phi_{k-1} (Jacobi matrix T, size n).  The
// builder produces, per problem, the Cuppen tree the GPU applies (leaf eigenvector blocks and,
// per merge node, poles d, root offsets tau, Gu-Eisenstat z, column norms c and the kept /
// deflated index maps) plus the Christoffel-Darboux data (polished nodes y_j = zeros of phi_n,
// the last row of V, E_{n-1}, and E_{n-1} phi_n(y_r) at every northern ring).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

constexpr int LEAF = 16;
constexpr double DEFLATE = 1e-12;
// Adjacent-pole gaps below this are stored exactly (as sparse records); the kernels recompute the
// others as the double-float difference of the stored poles, good to ~3.6e-15 absolute.
constexpr double GAP_TINY = 1e-6;

inline double a_coef(int l, int m) {
    return l > m ? std::sqrt(double(l) * l - double(m) * m) / std::sqrt(4.0 * l * l - 1.0) : 0.0;
}

// EISPACK tql2-style implicit QL for a small symmetric tridiagonal; eigenvalues ascending,
// eigenvectors in columns of Z (row-major n x n, Z[k*n + j] = component k of vector j).
void tql2(int n, std::vector<double>& d, std::vector<double> e, std::vector<double>& Z) {
    Z.assign(n * n, 0.0);
    for (int i = 0; i < n; ++i) Z[i * n + i] = 1.0;
    e.push_back(0.0);
    for (int l = 0; l < n; ++l) {
        int iter = 0, mm;
        do {
            for (mm = l; mm < n - 1; ++mm) {
                double dd = std::fabs(d[mm]) + std::fabs(d[mm + 1]);
                if (std::fabs(e[mm]) <= 1e-17 * dd) break;
            }
            if (mm != l) {
                if (iter++ == 60) break;
                double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
                double r = std::hypot(g, 1.0);
                g = d[mm] - d[l] + e[l] / (g + (g >= 0 ? std::fabs(r) : -std::fabs(r)));
                double s = 1.0, c = 1.0, p = 0.0;
                int i;
                for (i = mm - 1; i >= l; --i) {
                    double f = s * e[i], b = c * e[i];
                    e[i + 1] = (r = std::hypot(f, g));
                    if (r == 0.0) { d[i + 1] -= p; e[mm] = 0.0; break; }
                    s = f / r; c = g / r;
                    g = d[i + 1] - p;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    d[i + 1] = g + (p = s * r);
                    g = c * r - b;
                    for (int k = 0; k < n; ++k) {
                        f = Z[k * n + i + 1];
                        Z[k * n + i + 1] = s * Z[k * n + i] + c * f;
                        Z[k * n + i] = c * Z[k * n + i] - s * f;
                    }
                }
                if (r == 0.0 && i >= l) continue;
                d[l] -= p; e[l] = g; e[mm] = 0.0;
            }
        } while (mm != l);
    }
    // sort ascending
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return d[a] < d[b]; });
    std::vector<double> d2(n), Z2(n * n);
    for (int j = 0; j < n; ++j) {
        d2[j] = d[idx[j]];
        for (int k = 0; k < n; ++k) Z2[k * n + j] = Z[k * n + idx[j]];
    }
    // sign convention: first component positive (nonzero for an unreduced tridiagonal), which is
    // what the apply kernel reproduces when it rebuilds a leaf's eigenvectors on the fly
    for (int j = 0; j < n; ++j)
        if (Z2[j] < 0) for (int k = 0; k < n; ++k) Z2[k * n + j] = -Z2[k * n + j];
    d = d2; Z = Z2;
}

struct Merge {
    int height, off, n;
    std::vector<double> d;            // kept poles, ascending
    std::vector<float> tau, z;          // z: Gu-Eisenstat z times the children's column norms
    std::vector<int16_t> kept, slot_kept, defl, slot_defl;
};
struct Leaf { int off, s; float lam[16]; };

struct Node {
    int n, height;
    std::vector<double> lam, first, last;
    // Column-norm factors not yet applied to this node's output slots.  A merge's column norms c_j
    // scale its roots on output (synthesis) and input (analysis); the next level up scales the same
    // values by its z_i.  So each c_j is folded into the parent's z_i for that edge (or carried
    // through the parent's deflations, or into vlast at the top), and no c is stored.
    std::vector<double> pending;
};

// roots of 1 + rho sum z_i^2 / (d_i - lam), d ascending, rho > 0.  Root j is in
// (d_j, d_{j+1}) (last: (d_{n-1}, d_{n-1} + rho |z|^2)).  Returned as org index and offset.
void secular(const std::vector<double>& d, const std::vector<double>& z, double rho,
             std::vector<int>& org, std::vector<double>& tau) {
    const int n = d.size();
    org.resize(n); tau.resize(n);
    double zz = 0; for (double v : z) zz += v * v;
    std::vector<double> z2(n);
    for (int i = 0; i < n; ++i) z2[i] = z[i] * z[i];
    std::vector<double> dd(n);
    for (int j = 0; j < n; ++j) {
        const bool last = (j == n - 1);
        const double right = last ? d[j] + rho * zz : d[j + 1];
        // pick the origin: evaluate f at the bracket midpoint
        const double mid = 0.5 * (d[j] + right);
        double fm = 1.0;
        for (int i = 0; i < n; ++i) fm += rho * z2[i] / (d[i] - mid);
        int o = (last || fm >= 0.0) ? j : j + 1;
        org[j] = o;
        for (int i = 0; i < n; ++i) dd[i] = d[i] - d[o];
        double lo = (o == j) ? 0.0 : d[j] - d[o];
        double hi = (o == j) ? 0.5 * (right - d[j]) : 0.0;
        if (last) hi = right - d[j];
        double t = 0.5 * (lo + hi);
        for (int it = 0; it < 60; ++it) {
            double f = 1.0, psi1 = 0.0, phi1 = 0.0;
            for (int i = 0; i < n; ++i) {
                const double r = dd[i] - t, q = rho * z2[i] / r;
                f += q;
                if (i <= j) psi1 += q / r; else phi1 += q / r;
            }
            if (f > 0) hi = t; else lo = t;
            // two-pole rational model through f, psi', phi' at t (fixed-weight method)
            double tn;
            const double dl = dd[j] - t;
            if (last) {
                const double b1 = psi1 * dl * dl;
                const double a = f - b1 / dl;
                tn = (a != 0.0) ? dd[j] + b1 / a : 0.5 * (lo + hi);   // a + b1/(dd_j - s) = 0
            } else {
                const double dr = dd[j + 1] - t;
                const double b1 = psi1 * dl * dl, b2 = phi1 * dr * dr;
                const double a = f - b1 / dl - b2 / dr;
                // a (P - s)(Q - s) + b1 (Q - s) + b2 (P - s) = 0,  P = dd_j, Q = dd_{j+1}
                const double P = dd[j], Q = dd[j + 1];
                const double A = a, B = -(a * (P + Q) + b1 + b2), C = a * P * Q + b1 * Q + b2 * P;
                double s1, s2;
                if (std::fabs(A) < 1e-300) { s1 = s2 = -C / B; }
                else {
                    const double disc = std::max(B * B - 4 * A * C, 0.0), sq = std::sqrt(disc);
                    const double qq = -0.5 * (B + (B >= 0 ? sq : -sq));
                    s1 = qq / A; s2 = (qq != 0.0) ? C / qq : s1;
                }
                tn = (s1 > lo && s1 < hi) ? s1 : ((s2 > lo && s2 < hi) ? s2 : 0.5 * (lo + hi));
            }
            if (!(tn > lo && tn < hi)) tn = 0.5 * (lo + hi);
            const bool done = std::fabs(tn - t) <= 4e-16 * std::max(std::fabs(t), 1e-300) || hi - lo <= 4e-16 * std::max(std::fabs(t), 1e-300);
            t = tn;
            if (done) break;
        }
        tau[j] = t;
    }
}

Node build(const std::vector<double>& D, const std::vector<double>& E, int lo, int hi,
           int base, std::vector<Leaf>& leaves, std::vector<Merge>& merges) {
    Node node;
    node.n = hi - lo;
    if (node.n <= LEAF) {
        std::vector<double> d(D.begin() + lo, D.begin() + hi), e(E.begin() + lo, E.begin() + hi - 1), Z;
        tql2(node.n, d, e, Z);
        Leaf lf; lf.off = base + lo; lf.s = node.n;
        for (int j = 0; j < 16; ++j) lf.lam[j] = j < node.n ? (float)d[j] : 0.f;
        leaves.push_back(lf);
        node.height = 0; node.lam = d;
        node.first.resize(node.n); node.last.resize(node.n);
        for (int j = 0; j < node.n; ++j) { node.first[j] = Z[j]; node.last[j] = Z[(node.n - 1) * node.n + j]; }
        node.pending.assign(node.n, 1.0);
        return node;
    }
    const int k = lo + node.n / 2;
    const double rho = E[k - 1];
    std::vector<double> D1(D);
    D1[k - 1] -= rho; D1[k] -= rho;
    Node L = build(D1, E, lo, k, base, leaves, merges);
    Node R = build(D1, E, k, hi, base, leaves, merges);
    node.height = 1 + std::max(L.height, R.height);
    const int n = node.n;
    std::vector<double> dall(n), zall(n), fr(n, 0.0), lr(n, 0.0);
    for (int i = 0; i < L.n; ++i) { dall[i] = L.lam[i]; zall[i] = L.last[i]; fr[i] = L.first[i]; }
    for (int i = 0; i < R.n; ++i) { dall[L.n + i] = R.lam[i]; zall[L.n + i] = R.first[i]; lr[L.n + i] = R.last[i]; }
    std::vector<int> keptv, deflv;
    for (int i = 0; i < n; ++i) (std::fabs(zall[i]) > DEFLATE ? keptv : deflv).push_back(i);
    std::stable_sort(keptv.begin(), keptv.end(), [&](int a, int b) { return dall[a] < dall[b]; });
    std::stable_sort(deflv.begin(), deflv.end(), [&](int a, int b) { return dall[a] < dall[b]; });
    const int nk = keptv.size(), nd = deflv.size();
    std::vector<double> dk(nk), zk(nk);
    for (int i = 0; i < nk; ++i) { dk[i] = dall[keptv[i]]; zk[i] = zall[keptv[i]]; }
    std::vector<int> org; std::vector<double> tau;
    if (nk) secular(dk, zk, rho, org, tau);
    // lam_j - d_i = (d_org(j) - d_i) + tau_j
    auto lmd = [&](int j, int i) { return (dk[org[j]] - dk[i]) + tau[j]; };
    std::vector<double> zh(nk), cn(nk);
    for (int i = 0; i < nk; ++i) {
        double prod = lmd(i, i) / rho;           // (lam_i - d_i) / rho
        for (int j = 0; j < nk; ++j) if (j != i) prod *= lmd(j, i) / (dk[j] - dk[i]);
        zh[i] = std::copysign(std::sqrt(std::fabs(prod)), zk[i]);
    }
    for (int j = 0; j < nk; ++j) {
        double s = 0;
        for (int i = 0; i < nk; ++i) { const double u = zh[i] / lmd(j, i); s += u * u; }
        cn[j] = 1.0 / std::sqrt(s);
    }
    // sorted eigenvalue list: kept roots and deflated poles
    std::vector<double> lamk(nk);
    for (int j = 0; j < nk; ++j) lamk[j] = dk[org[j]] + tau[j];
    std::vector<int16_t> slot_k(nk), slot_d(nd);
    node.lam.resize(n);
    {
        int a = 0, b = 0, s = 0;
        while (a < nk || b < nd) {
            if (b >= nd || (a < nk && lamk[a] <= dall[deflv[b]])) { slot_k[a] = s; node.lam[s++] = lamk[a++]; }
            else { slot_d[b] = s; node.lam[s++] = dall[deflv[b++]]; }
        }
    }
    auto applyUt = [&](const std::vector<double>& x, std::vector<double>& out) {
        out.assign(n, 0.0);
        for (int j = 0; j < nk; ++j) {
            double s = 0;
            for (int i = 0; i < nk; ++i) s += zh[i] * x[keptv[i]] / (-lmd(j, i));
            out[slot_k[j]] = cn[j] * s;
        }
        for (int b = 0; b < nd; ++b) out[slot_d[b]] = x[deflv[b]];
    };
    applyUt(fr, node.first);
    applyUt(lr, node.last);
    Merge mg;
    mg.height = node.height; mg.off = lo; mg.n = n;
    mg.d = dk; mg.tau.resize(nk); mg.z.resize(nk);
    mg.kept.resize(nk); mg.defl.resize(nd);
    // fold the children's pending column norms into this node's z (kept poles) or pass them through
    // its deflations; this node's own c becomes pending on its root slots
    std::vector<double> pend_in(n);
    for (int i = 0; i < L.n; ++i) pend_in[i] = L.pending[i];
    for (int i = 0; i < R.n; ++i) pend_in[L.n + i] = R.pending[i];
    node.pending.assign(n, 1.0);
    for (int j = 0; j < nk; ++j) node.pending[slot_k[j]] = cn[j];
    for (int b = 0; b < nd; ++b) node.pending[slot_d[b]] = pend_in[deflv[b]];
    for (int i = 0; i < nk; ++i) { mg.tau[i] = (float)tau[i]; mg.z[i] = (float)(zh[i] * pend_in[keptv[i]]); mg.kept[i] = (int16_t)keptv[i]; }
    for (int b = 0; b < nd; ++b) mg.defl[b] = (int16_t)deflv[b];
    mg.slot_kept = slot_k; mg.slot_defl = slot_d;
    merges.push_back(std::move(mg));
    return node;
}

// phi_n(y) and phi_n'(y) of the unnormalised polynomial part by the y-recurrence (scaled).
void poly_eval(const std::vector<double>& D, const std::vector<double>& E, int n, double y, double& f, double& fp) {
    double p0 = 1.0, p1 = (y - D[0]) / E[0], q0 = 0.0, q1 = 1.0 / E[0];
    for (int k = 1; k < n; ++k) {
        const double p2 = ((y - D[k]) * p1 - E[k - 1] * p0) / E[k];
        const double q2 = ((y - D[k]) * q1 + p1 - E[k - 1] * q0) / E[k];
        p0 = p1; p1 = p2; q0 = q1; q1 = q2;
        const double big = std::max(std::fabs(p1), std::fabs(q1));
        if (big > 1e200) { p0 *= 1e-200; p1 *= 1e-200; q0 *= 1e-200; q1 *= 1e-200; }
    }
    f = p1; fp = q1;
}

}  // namespace

struct Problem {
    int m, p, n;
    std::vector<Leaf> leaves;
    std::vector<Merge> merges;
    std::vector<double> y;
    std::vector<float> vlast, scale, der;   // der_j = E phi_n'(y_j): the CD term's limit at a node
    std::vector<float> croot;                // the top node's pending column norms, per output slot
    double E;
};

// Flat device layout (see _dc_lat.py for the field meanings).
struct Packed {
    std::vector<int32_t> leaf_off, leaf_sz, leaf_mk;   // leaf_mk: (m, p, k0, n) per leaf
    std::vector<float> leaf_lam;                       // leaf eigenvalues, one per coefficient
    // levels
    std::vector<int32_t> lev_kind, lev_nnode, lev_node0, lev_maxk, lev_sb0;
    std::vector<int32_t> nodes;                 // 8 per node: base nk nd poff doff 0 0 0
    std::vector<int32_t> sbase;                 // fmm nodes: first scratch box
    std::vector<float> dh, dl, tau, z, croot;   // croot: per coefficient, the top node's pending column norms
    std::vector<int32_t> gpr_i;                 // tiny adjacent gaps: local index i (gap = d[i+1] - d[i])
    std::vector<float> gpr_f;
    std::vector<int16_t> gidx, dsrc, ddst;   // the kept roots' slots follow from ddst (slot_of)
    int64_t nbox = 0;
    // Christoffel-Darboux
    std::vector<int32_t> cd_desc, cd_ln, cd_lr;
    std::vector<float> cd_nh, cd_nl, vlast, scale, ring_h, ring_l, cd_der;
    // near node-ring pairs (|y_r - y_j| < CD_NEAR): (problem, node, ascending ring index) and the
    // exact difference y_r - y_j (0 marks a coincidence, which takes the limit E phi_n'(y_j))
    std::vector<int32_t> cdx_i;
    std::vector<float> cdx_f;
    int64_t cd_nbox = 0, ntot = 0, nprob = 0, R = 0;
};

static Packed* g_pack = nullptr;

static void split(double v, float& h, float& l) { h = (float)v; l = (float)(v - (double)h); }

static int nbox_of(int npts, int per_leaf) {
    int nleaf = std::max(1, (npts + per_leaf - 1) / per_leaf), nb = 0, s = nleaf;
    while (true) { nb += s; if (s <= 3) break; s = (s + 1) / 2; }
    return nb;
}

extern "C" {

// Build and pack every (m, p) problem for bandlimit L on the northern rings xr[0..R) (cos theta,
// pole to equator).  direct_max: kept-size cut between the direct and FMM merge kernels.
// skip: CD rings whose |E phi_n| is below skip * max are left out (forbidden region).
// spin 0: xr are the northern rings (pole to equator); problems (m, p) in y = x^2, two parities.
// spin s > 0: xr are all rings (pole to pole); one problem per m in x over the normalised Wigner
// functions u_l = sqrt((2l+1)/2) d^l_{m,-s}, l = max(m, s) ..., whose Jacobi matrix has
// diagonal -m s / (l (l+1)) and off-diagonal a_{l+1}, a_l = sqrt((l^2-m^2)(l^2-s^2)) / (l sqrt(4l^2-1)).
static inline double a_spin(int l, int m, int s) {
    return l > std::max(m, s) ? std::sqrt((double(l) * l - double(m) * m) * (double(l) * l - double(s) * s))
                                    / (l * std::sqrt(4.0 * l * l - 1.0)) : 0.0;
}

int64_t dc_plan(int L, const double* xr, int R, int nthreads, int direct_max, int direct_threads,
                double skip, int spin) {
    std::vector<std::pair<int, int>> pairs;
    if (spin == 0) { for (int m = 0; m < L; ++m) for (int p = 0; p < 2; ++p) if (m + p < L) pairs.push_back({m, p}); }
    else { for (int m = 0; m < L; ++m) if (std::max(m, spin) < L) pairs.push_back({m, 0}); }
    std::vector<Problem> probs(pairs.size());
    #pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
    for (size_t t = 0; t < pairs.size(); ++t) {
        const int m = pairs[t].first, p = pairs[t].second;
        Problem& pr = probs[t];
        pr.m = m; pr.p = p;
        const int M = std::max(m, spin);
        int n = 0;
        if (spin == 0) { for (int l = m + p; l < L; l += 2) ++n; } else n = L - M;
        pr.n = n;
        std::vector<double> D(n), E(n);
        for (int k = 0; k < n; ++k) {
            if (spin == 0) {
                const int l = m + p + 2 * k;
                const double a1 = a_coef(l + 1, m), a0 = a_coef(l, m);
                D[k] = a1 * a1 + a0 * a0;
                E[k] = a_coef(l + 1, m) * a_coef(l + 2, m);
            } else {
                const int l = M + k;
                D[k] = -double(m) * spin / (double(l) * (l + 1));
                E[k] = a_spin(l + 1, m, spin);
            }
        }
        Node root = build(D, E, 0, n, 0, pr.leaves, pr.merges);
        pr.y = root.lam;
        for (int j = 0; j < n; ++j) {
            double yj = pr.y[j];
            for (int it = 0; it < 3; ++it) {
                double f, fp; poly_eval(D, E, n, yj, f, fp);
                if (fp == 0.0) break;
                const double step = f / fp;
                yj -= step;
                if (std::fabs(step) <= 1e-17 * std::fabs(yj)) break;
            }
            const double lo = j ? 0.5 * (pr.y[j - 1] + pr.y[j]) : -1.0;
            const double hi = j + 1 < n ? 0.5 * (pr.y[j] + pr.y[j + 1]) : 2.0;
            if (yj > lo && yj < hi) pr.y[j] = yj;
        }
        pr.vlast.resize(n);
        for (int j = 0; j < n; ++j) pr.vlast[j] = (float)root.last[j];
        pr.croot.resize(n);
        for (int j = 0; j < n; ++j) pr.croot[j] = (float)root.pending[j];
        pr.E = E[n - 1];
        // E phi_n'(y_j) = phi_0(y_j) / (V[0,j] V[n-1,j]) (Christoffel-Darboux at a node; invariant
        // under the column's sign), in log form since phi_0 and V[0,j] underflow together at large m
        pr.der.resize(n);
        for (int j = 0; j < n; ++j) {
            const double yj = pr.y[j];
            double lg0;                                         // log2 |phi_0(y_j)|
            double sg0 = 1.0;
            if (spin == 0) {
                const double xx = std::sqrt(std::max(yj, 0.0)), sn = std::sqrt(std::max(1.0 - yj, 0.0));
                lg0 = 0.5 * std::log2((2.0 * m + 1) / 2.0);
                for (int k = 1; k <= m; ++k) lg0 += 0.5 * std::log2((2.0 * k - 1) / (2.0 * k));
                lg0 += sn > 0 ? m * std::log2(sn) : -1e9;
                if (p == 1) lg0 += std::log2(std::sqrt(2.0 * m + 3.0) * std::max(xx, 1e-300));
            } else {
                const int aa = m + spin, bb = std::abs(m - spin), MM = std::max(m, spin);
                lg0 = 0.5 * std::log2((2.0 * MM + 1) / 2.0)
                    + 0.5 * (std::lgamma(aa + bb + 1.0) - std::lgamma(aa + 1.0) - std::lgamma(bb + 1.0)) / std::log(2.0)
                    + 0.5 * aa * std::log2(std::max(0.5 * (1.0 - yj), 1e-300))
                    + 0.5 * bb * std::log2(std::max(0.5 * (1.0 + yj), 1e-300));
            }
            const double f = root.first[j], l = root.last[j];
            if (f == 0.0 || l == 0.0) { pr.der[j] = 0.f; continue; }
            const double lg = lg0 - std::log2(std::fabs(f)) - std::log2(std::fabs(l));
            pr.der[j] = (float)(sg0 * ((f * l) < 0 ? -1.0 : 1.0) * std::exp2(std::max(std::min(lg, 120.0), -140.0)));
        }
        pr.scale.resize(R);
        std::vector<double> cur(R), prev(R, 0.0), ex(R), xs(xr, xr + R);
        if (spin == 0) {
            const int lnext = m + p + 2 * n;
            double lg2 = 0.5 * std::log2((2.0 * m + 1) / 2.0);
            for (int k = 1; k <= m; ++k) lg2 += 0.5 * std::log2((2.0 * k - 1) / (2.0 * k));
            for (int r = 0; r < R; ++r) {
                const double s = std::sqrt(std::max(1.0 - xs[r] * xs[r], 0.0));
                const double e2 = lg2 + (s > 0 ? m * std::log2(s) : -1e9);
                ex[r] = std::floor(e2); cur[r] = std::exp2(e2 - ex[r]);
            }
            for (int l = m + 1; l <= lnext; ++l) {
                const double a = std::sqrt((4.0 * l * l - 1) / (double(l) * l - double(m) * m));
                const double b = l - 1 > m ? std::sqrt(((l - 1.0) * (l - 1.0) - double(m) * m) / (4.0 * (l - 1.0) * (l - 1.0) - 1)) : 0.0;
                for (int r = 0; r < R; ++r) {
                    const double nx = a * (xs[r] * cur[r] - b * prev[r]);
                    prev[r] = cur[r]; cur[r] = nx;
                }
                if ((l & 15) == 0)
                    for (int r = 0; r < R; ++r)
                        if (std::fabs(cur[r]) > 0x1p200) { prev[r] *= 0x1p-200; cur[r] *= 0x1p-200; ex[r] += 200; }
            }
        } else {
            // u_M = sqrt((2M+1)/2) sqrt((a+b)! / (a! b!)) sin(theta/2)^a cos(theta/2)^b, a = m+s, b = |m-s|
            const int aa = m + spin, bb = std::abs(m - spin);
            const double lg2 = 0.5 * std::log2((2.0 * M + 1) / 2.0)
                + 0.5 * (std::lgamma(aa + bb + 1.0) - std::lgamma(aa + 1.0) - std::lgamma(bb + 1.0)) / std::log(2.0);
            for (int r = 0; r < R; ++r) {
                const double sh = std::max(0.5 * (1.0 - xs[r]), 1e-300), ch = std::max(0.5 * (1.0 + xs[r]), 1e-300);
                const double e2 = lg2 + 0.5 * aa * std::log2(sh) + 0.5 * bb * std::log2(ch);
                ex[r] = std::floor(e2); cur[r] = std::exp2(e2 - ex[r]);
            }
            for (int l = M; l < L; ++l) {          // u_{l+1} = ((x - b_l) u_l - a_l u_{l-1}) / a_{l+1}
                const double bl = -double(m) * spin / (double(l) * (l + 1)), al = a_spin(l, m, spin), an = a_spin(l + 1, m, spin);
                for (int r = 0; r < R; ++r) {
                    const double nx = ((xs[r] - bl) * cur[r] - al * prev[r]) / an;
                    prev[r] = cur[r]; cur[r] = nx;
                }
                if ((l & 15) == 0)
                    for (int r = 0; r < R; ++r)
                        if (std::fabs(cur[r]) > 0x1p200) { prev[r] *= 0x1p-200; cur[r] *= 0x1p-200; ex[r] += 200; }
            }
        }
        for (int r = 0; r < R; ++r)
            pr.scale[r] = (float)(pr.E * cur[r] * std::exp2(std::max(std::min(ex[r], 1000.0), -1100.0)));
    }
    // ---------------------------------------------------------------- pack
    delete g_pack;
    g_pack = new Packed();
    Packed& P = *g_pack;
    P.nprob = probs.size(); P.R = R;
    std::vector<int64_t> off(probs.size() + 1, 0);
    for (size_t t = 0; t < probs.size(); ++t) off[t + 1] = off[t] + probs[t].n;
    P.ntot = off.back();
    P.leaf_lam.assign(P.ntot, 0.f);
    for (size_t t = 0; t < probs.size(); ++t)
        for (const Leaf& lf : probs[t].leaves) {
            P.leaf_off.push_back((int32_t)(off[t] + lf.off)); P.leaf_sz.push_back(lf.s);
            const int32_t mk[4] = {probs[t].m, probs[t].p, lf.off, probs[t].n};
            P.leaf_mk.insert(P.leaf_mk.end(), mk, mk + 4);
            for (int j = 0; j < lf.s; ++j) P.leaf_lam[off[t] + lf.off + j] = lf.lam[j];
        }
    int H = 0;
    for (const auto& pr : probs) for (const auto& mg : pr.merges) H = std::max(H, mg.height);
    int64_t poff = 0, doff = 0;
    for (int h = 1; h <= H; ++h) {
        for (int kind = 0; kind < 2; ++kind) {           // 0 direct, 1 fmm
            const int node0 = P.nodes.size() / 8, sb0 = P.sbase.size();
            int nn = 0, maxk = 0;
            for (size_t t = 0; t < probs.size(); ++t)
                for (const auto& mg : probs[t].merges) {
                    if (mg.height != h) continue;
                    const int nk = mg.d.size(), nd = mg.defl.size();
                    const bool direct = nk <= direct_max && nd <= direct_threads;
                    if (direct != (kind == 0)) continue;
                    const int32_t goff = P.gpr_i.size();
                    for (int i = 0; i + 1 < nk; ++i) {
                        const double g = mg.d[i + 1] - mg.d[i];
                        if (g < GAP_TINY) { P.gpr_i.push_back(i); P.gpr_f.push_back((float)g); }
                    }
                    const int32_t rec[8] = {(int32_t)(off[t] + mg.off), nk, nd, (int32_t)poff, (int32_t)doff,
                                            goff, (int32_t)P.gpr_i.size() - goff, 0};
                    P.nodes.insert(P.nodes.end(), rec, rec + 8);
                    for (int i = 0; i < nk; ++i) {
                        float a, b; split(mg.d[i], a, b); P.dh.push_back(a); P.dl.push_back(b);
                    }
                    P.tau.insert(P.tau.end(), mg.tau.begin(), mg.tau.end());
                    P.z.insert(P.z.end(), mg.z.begin(), mg.z.end());
                    P.gidx.insert(P.gidx.end(), mg.kept.begin(), mg.kept.end());
                    P.dsrc.insert(P.dsrc.end(), mg.defl.begin(), mg.defl.end());
                    P.ddst.insert(P.ddst.end(), mg.slot_defl.begin(), mg.slot_defl.end());
                    if (kind == 1) { P.sbase.push_back((int32_t)P.nbox); P.nbox += nbox_of(nk, 8); }
                    poff += nk; doff += nd; ++nn; maxk = std::max(maxk, nk);
                }
            if (nn) {
                P.lev_kind.push_back(kind); P.lev_nnode.push_back(nn); P.lev_node0.push_back(node0);
                P.lev_maxk.push_back(maxk); P.lev_sb0.push_back(sb0);
            }
        }
    }
    // Christoffel-Darboux: rings ascending in the problem variable (y = x^2 from the equator for
    // spin 0; x from the south pole otherwise -- both are the given ring order reversed)
    std::vector<double> yr(R);
    for (int r = 0; r < R; ++r) yr[r] = spin == 0 ? xr[R - 1 - r] * xr[R - 1 - r] : xr[R - 1 - r];
    for (int r = 0; r < R; ++r) { float a, b; split(yr[r], a, b); P.ring_h.push_back(a); P.ring_l.push_back(b); }
    int64_t lo = 0;
    for (size_t t = 0; t < probs.size(); ++t) {
        const Problem& pr = probs[t];
        float smax = 0; for (float v : pr.scale) smax = std::max(smax, std::fabs(v));
        // active rings: the interval [t0, t0 + nt) of the ascending order outside the forbidden region(s)
        int t0 = R, t1 = 0;
        for (int r = 0; r < R; ++r) if (std::fabs(pr.scale[R - 1 - r]) > skip * smax) { t0 = std::min(t0, r); t1 = r + 1; }
        if (t1 <= t0) { t0 = 0; t1 = 0; }
        const int nt = t1 - t0;
        const int tot = pr.n + nt, nleaf = std::max(1, (tot + 15) / 16);
        // merged order of nodes (ascending) and the active ring coordinates (ascending)
        int a = 0, b = 0;
        std::vector<int32_t> ln(nleaf + 1), lr(nleaf + 1);
        for (int q = 0; q <= nleaf; ++q) {
            const int cut = std::min(q * 16, tot);
            while (a + b < cut) { if (b >= nt || (a < pr.n && pr.y[a] <= yr[t0 + b])) ++a; else ++b; }
            ln[q] = a; lr[q] = b;
        }
        const int32_t rec[8] = {(int32_t)off[t], pr.n, (int32_t)lo, nleaf, (int32_t)P.cd_nbox, nt, t0, 0};
        {   // near pairs: the CD kernel's double-float difference is ~4e-17 absolute, so a pair closer
            // than ~1e-10 loses digits (one node sat 8.9e-15 from a ring at Nside 4096 spin 2, m = 179)
            constexpr double CD_NEAR = 1e-9;
            int j = 0;
            for (int b2 = 0; b2 < nt; ++b2) {
                const double yrv = yr[t0 + b2];
                while (j + 1 < pr.n && pr.y[j + 1] <= yrv) ++j;
                for (int jj = std::max(j - 1, 0); jj <= std::min(j + 1, pr.n - 1); ++jj) {
                    const double diff = yrv - pr.y[jj];
                    if (std::fabs(diff) < CD_NEAR) {
                        P.cdx_i.push_back((int32_t)t); P.cdx_i.push_back(jj); P.cdx_i.push_back(t0 + b2);
                        P.cdx_f.push_back(std::fabs(diff) < 1e-15 ? 0.f : (float)diff);
                    }
                }
            }
        }
        P.cd_desc.insert(P.cd_desc.end(), rec, rec + 8);
        P.cd_ln.insert(P.cd_ln.end(), ln.begin(), ln.end());
        P.cd_lr.insert(P.cd_lr.end(), lr.begin(), lr.end());
        lo += nleaf; P.cd_nbox += nbox_of(tot, 16);
        for (int j = 0; j < pr.n; ++j) { float h, l; split(pr.y[j], h, l); P.cd_nh.push_back(h); P.cd_nl.push_back(l); }
        P.vlast.insert(P.vlast.end(), pr.vlast.begin(), pr.vlast.end());
        P.croot.insert(P.croot.end(), pr.croot.begin(), pr.croot.end());
        P.cd_der.insert(P.cd_der.end(), pr.der.begin(), pr.der.end());
        P.scale.insert(P.scale.end(), pr.scale.begin(), pr.scale.end());
    }
    return P.ntot;
}

// Sizes and copies of the packed arrays, by name.
#define DC_FIELDS(X) \
    X(leaf_off) X(leaf_sz) X(leaf_mk) X(leaf_lam) X(lev_kind) X(lev_nnode) X(lev_node0) X(lev_maxk) X(lev_sb0) \
    X(nodes) X(sbase) X(dh) X(dl) X(gpr_i) X(gpr_f) X(tau) X(z) X(croot) X(gidx) X(dsrc) X(ddst) \
    X(cd_desc) X(cd_ln) X(cd_lr) X(cd_nh) X(cd_nl) X(vlast) X(scale) X(ring_h) X(ring_l) X(cd_der) X(cdx_i) X(cdx_f)

int64_t dc_size(const char* name) {
#define X(f) if (!std::strcmp(name, #f)) return (int64_t)g_pack->f.size();
    DC_FIELDS(X)
#undef X
    if (!std::strcmp(name, "nbox")) return g_pack->nbox;
    if (!std::strcmp(name, "cd_nbox")) return g_pack->cd_nbox;
    return -1;
}

void dc_copy(const char* name, void* dst) {
#define X(f) if (!std::strcmp(name, #f)) { std::memcpy(dst, g_pack->f.data(), g_pack->f.size() * sizeof(g_pack->f[0])); return; }
    DC_FIELDS(X)
#undef X
}

void dc_release() { delete g_pack; g_pack = nullptr; }

}  // extern "C"
