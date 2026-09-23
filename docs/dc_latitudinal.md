# A sub-cubic latitudinal transform for GMaster: divide-and-conquer eigenbasis + 1-D FMM

Session 38 (22-23 September 2026, branch `exp/cpu-dform`).  Code: `gmaster/_dc_lat.py`,
`gmaster/_cuda/dc_lat.cu`, `gmaster/_cpu/dc_plan.cpp`.  Probes and raw logs: `.qwen/tmp/s38/`.
All timings: RTX PRO 6000 Blackwell (GPU0 of this box; GPU1 was occupied by another user's job),
warm medians; NaMaster / ducc0 / SHTns-CPU on the 96-core EPYC host.

## 1. What changed and why it is sub-cubic

The shipped v2 march evaluates the associated Legendre functions by recurrence at every ring for
every `(ell, m)`: `O(L^2 N_ring) = O(L^3)` work per transform (it is a very good `O(L^3)` kernel,
~45 % of the card's fp32 issue rate).  The new route never forms `lambda_lm(x_r)`.

For order `m` and parity `p` the orthonormal functions `phi_k(y) = psi_{m+p+2k}(sqrt y)` of
`y = cos^2 theta` satisfy `y phi_k = E_k phi_{k+1} + D_k phi_k + E_{k-1} phi_{k-1}`; with
`T = V Y V^T` its Jacobi matrix (`Y` = zeros `y_j` of `phi_n`), Christoffel-Darboux gives, at any ring,

    f(y_r) = sum_k c_k phi_k(y_r) = E_{n-1} phi_n(y_r) sum_j V[n-1, j] (V^T c)_j / (y_r - y_j).

* `V^T c` is applied through Cuppen's divide-and-conquer tree of `T` in Gu & Eisenstat's stable form:
  16x16 leaves, then per level one rank-one-update merge whose matrix is Cauchy-like,
  `U[i,j] = z_i c_j / (d_i - lam_j)`.  Each merge is a 1-D Cauchy sum -- direct below 256 poles, a
  1-D FMM (P = 12, 32-point leaves) above.
* The last sum over `j` is a 1-D FMM from the Gauss nodes to the rings.
* Every step is `O(n)` or `O(n log n)` for an `n`-term order, so a transform is `O(L^2 log L)`.
  Analysis is the exact adjoint (same plan, transposed kernels, reversed order).

Measured scaling of one transform: doubling `L` costs x4.3 here and x5.8 for the march (L 3072 -> 6144).
Scaling plot (datapoints from `.qwen/tmp/s38/scaling.log` and `.qwen/tmp/s38/s2_scale.log`, guides pinned at Nside 2048): `docs/dc_lat_scaling.png`.
Regenerate: `python .qwen/tmp/s38/plot_scaling.py`.

## 1b. Spin 2

For spin `s` the latitudinal functions are Wigner `d^l_{m,-s}(theta) = (sin theta/2)^a (cos theta/2)^b
P_n^{(a,b)}(x)`, `a = m + s`, `b = |m - s|`: orthonormal Jacobi functions in `x = cos theta`.  With
`u_l = sqrt((2l+1)/2) d^l_{m,-s}`,

    x u_l = a_{l+1} u_{l+1} + b_l u_l + a_l u_{l-1},   b_l = -m s / (l (l+1)),
    a_l = sqrt((l^2 - m^2)(l^2 - s^2)) / (l sqrt(4 l^2 - 1))            (checked against sympy to 1e-16).

The diagonal is non-zero, so there is no parity split: one problem per `m`, of size `L - max(m, s)`,
over all rings, with Gauss-Jacobi nodes in `x`.  Everything downstream -- the Cuppen tree, the
secular solves, Christoffel-Darboux -- is generic in the tridiagonal.  Negative orders reuse the
same plan: `d^l_{-m,-s}(theta) = (-1)^(l+s) d^l_{m,-s}(pi - theta)`, i.e. the ring-reversed data with a
sign, carried as a second right-hand side of the same traversal.

## 1c. Refinement in node space (the trees cancel)

The transforms factor as `S = C V^T diag(fac)` and `A = diag(fac) V C^T`, with `C` the
Christoffel-Darboux node-to-ring map and `V` the orthogonal eigenvector matrix the tree applies.  For
the normalised transforms of either spin `fac^2 = 1/(2 pi)` (spin `s`: `fac_bare^2 N_l^2 =
(2/(2l+1)) ((2l+1)/(4 pi))`; the mirror sign squares to 1), so in `w = V^T (alm / fac)` the MASTER
refinement

    alm <- alm - A (S alm - map)     becomes     w <- w - C^T (C w / (2 pi) - map),

and `alm = fac V w` at the end.  An iteration costs two CD steps and a ring fold; the tree -- 70 % of
a transform -- runs once instead of `2 n_iter + 1` times.  With the ring-Fourier fold (and its
complex, non-Hermitian version for spin fields) the loop never leaves (node, ring-Fourier) space.

## 1d. Robustness: near-coincidences

* Adjacent poles of the two children can nearly coincide; the plan stores each adjacent gap exactly
  and the kernels replace the <= 3 terms nearest each target by exact values after a clamped,
  branch-free hot loop.
* A Gauss node can sit arbitrarily close to a HEALPix ring (exactly on one for the size-1 problem;
  8.9e-15 away at Nside 4096 spin 2, m = 179).  The plan lists every node-ring pair closer than 1e-9
  with its fp64 difference (or the limit `E phi_n'(y_j) = phi_0(y_j) / (V[0,j] V[n-1,j])` when they
  coincide), and a small kernel substitutes those terms exactly.
* The D&C engine is never traced inside another jitted program (its plan would become constants);
  its callers compose their programs eagerly.

## 2. Engineering that made it fit (memory and accuracy)

* **Plan** (geometry only; host-built once in C++/OpenMP with 8 threads, cached on disk):
  per merge level `dh, dl` (double-float poles), `tau` (root offset from its nearest pole),
  Gu-Eisenstat `z`, column norms `c`, int16 index maps, and the adjacent-pole gaps; CD nodes,
  last row of `V`, `E phi_n(y_r)`.  Build time 28 s / 198 s / 1402 s at Nside 1024 / 2048 / 4096.
* **Leaves are not stored**: a half-warp rebuilds each 16x16 eigenvector block from 16 stored
  eigenvalues by a twisted factorisation of the torn tridiagonal (entries closed-form in `m, ell`).
  64 -> 4 bytes per coefficient (1.5 GiB at Nside 2048).
* **Scratch is level-local** (levels run in sequence) and the CD step reads its strengths in place:
  pipeline peak 23.6 -> 14.3 GiB at Nside 2048 before the gaps below.
* **Two right-hand sides per traversal** (field and mask, which MASTER refines in lockstep).
* **Exact adjacent-pole gaps**.  Near `y = 1` at small `m` a pole of the left child and one of the
  right can nearly coincide; their double-float difference is only good to ~4e-15 absolute
  (m = 0 analysis was 1.3e-5 at Nside 2048).  The plan stores each gap in fp32; the hot loops use
  the double-float difference with a clamped reciprocal and each target replaces its <= 3 nearest
  terms by exact values afterwards (branch-free; the per-iteration select cost +20-28 %).
  Givens ("type-2") deflation with a loose tolerance was tried and is wrong here (the secular
  root between the two poles is closer than the rotation's perturbation: errors 1e-3).
* **Ring-Fourier fold** (independent of the latitudinal engine): `FFT(IFFT(F))` on a HEALPix ring is
  the `n_phi`-periodic fold of `F`, so the `n_iter` refinement never runs a ring FFT again
  (`_march_v2.ring_fold_residual`, 3.3 vs 11.5 ms at Nside 2048).
* **Scalar coupling matrix**: half-node parity-block GEMMs, cached `P_l` table, fp32 knob honoured
  (Nside 2048: 1097 -> 7.6 ms).

## 3. Results

Board of 23 September 2026 (`.qwen/tmp/s38/suite2.log`, `suite3.log`; commit a2f9b8a; router
`GMASTER_DC_MIN_L=3072`, i.e. the D&C engine from Nside 1024 up, march below).  NaMaster / ducc0 on
96 host threads, timed live in the same run.  GPU0, RTX PRO 6000 Blackwell.

### Complexity: one latitudinal pass (`docs/dc_lat_complexity.png`)

Measured log-log slopes in Nside, 2048 -> 4096 (synthesis / analysis):

| engine | spin 0 | spin 2 |
|---|---|---|
| D&C + FMM | 2.05 / 2.06 | 2.09 / 2.07 |
| v2 march | 4.00 / 3.51 | 3.30 / 3.23 |

`O(N^2 log N)` predicts 2.13 over this interval, `O(N^3)` 3.  Below Nside 512 both engines sit on a
0.1-1 ms launch floor.  Nside 4096 pass times: spin 0 81 / 102 ms (march 485 / 422), spin 2
141 / 148 ms (march 585 / 620).

### MASTER pipeline end to end (`benchmarks/benchmark_pipeline.py`, n_iter = 3, dl = 30)

| Nside | spin 0: NaMaster -> GMaster | ratio | spin 2: NaMaster -> GMaster | ratio | C_l rel (s0 / s2) |
|---|---|---|---|---|---|
| 64 | 10 -> 2 ms | 5.3x | 14 -> 3 ms | 4.4x | 1e-7 / 2e-7 |
| 128 | 20 -> 2 ms | 10.8x | 40 -> 3 ms | 11.9x | 6e-7 / 4e-7 |
| 256 | 64 -> 4 ms | 16.9x | 173 -> 9 ms | 20.3x | 6e-7 / 8e-7 |
| 512 | 345 -> 21 ms | 16.4x | 714 -> 33 ms | 21.6x | 1.1e-6 / 2.4e-6 |
| 1024 | 1804 -> 38 ms | 47.9x | 3316 -> 96 ms | 34.7x | 3.2e-6 / 3.2e-6 |
| 2048 | 10776 -> 139 ms | 77.3x | 18797 -> 462 ms | 40.7x | 1.2e-5 / 1.2e-5 |
| 4096 | (not rerun) -> 592 ms | -- | OOM (see below) | -- | -- |

Nside 2048 spin 2 stages: field 209 ms, coupling 192 ms, mask 64 ms -- the coupling (spin-2
coupling matrix, untouched this session) is now as large as the field.  GPU peak at 2048: 16.0 GiB
(spin 0), 32.3 GiB (spin 2).  **Open:** Nside 4096 spin 2 runs out of the card's memory when both
the spin-0 plan (18.7 GiB, for the mask) and the spin-2 plan (20.7 GiB) are resident beside the
coupling GEMM; the 4096 spin-0 pipeline's 55.2 GiB peak is also above the march route's.

### Isolated map2alm / alm2map (`benchmarks/benchmark_sht.py`, vs NaMaster ducc0, 96 threads)

map2alm speed-up (GMaster time) / alm2map speed-up:

| Nside | spin 0, n_iter 0 | spin 2, n_iter 0 | spin 0, n_iter 3 | spin 2, n_iter 3 |
|---|---|---|---|---|
| 64 | 3.2x / 2.7x | 4.0x / 3.7x | 7.4x / 2.4x | 6.2x / 2.6x |
| 256 | 6.0x / 4.7x | 7.6x / 3.7x | 8.4x / 4.6x | 7.9x / 4.3x |
| 512 | 7.6x / 5.6x | 11.0x / 7.5x | 9.8x / 5.2x | 12.2x / 7.1x |
| 1024 | 8.4x / 8.2x | 6.2x / 9.1x | 24.5x (18 ms) / 7.8x | 16.1x (55 ms) / 9.1x |
| 2048 | 10.9x / 9.9x | 8.2x / 11.9x | 40.1x (59 ms) / 10.0x | 22.7x (208 ms) / 12.0x |
| 4096 | 15.6x / 16.6x | 12.2x / 18.2x | 62.4x (231 ms) / 19.1x | 35.8x (815 ms) / 18.9x |

Map-level rel. error vs ducc0: <= 1.4e-5 at every size (1.9e-5 for 4096 spin 2, n_iter 3).  The
n_iter = 3 analysis gains most because the node-space refinement (section 1c) runs the tree once.

### Earlier spin-0 pipeline snapshots (for the record)
Session start: 82 ms (1024), 472 ms (2048), 7458 ms (4096, report).  Before node space: 57 / 228 /
1237 ms.  At 4096 every GMaster route, the exact fp64 one included, is 1.56e-4 from NaMaster (a
uniform ratio that fp64 coupling and fp64 rings do not move -- pre-existing, not transform error);
the D&C engine is 1.2e-6 from GMaster's exact route, the shipped fp32 march 7.9e-6.

### One latitudinal pass (field + mask pair), Nside 2048: D&C vs the march's pair kernels
synthesis 24.9 vs 44.6 ms, analysis 31.2 vs 47.0 ms.  Per order against fp64 direct sums, every
checked order is <= 1.2e-5 (the march reaches 4.7e-5).

### Isolated transforms against other codes (spin 0, lmax = 3 Nside - 1), ms, synthesis / analysis

| Nside | GMaster march (HEALPix) | ducc0 via NaMaster 96c (HEALPix) | SHTns CPU 96t (Gauss grid) | SHTns GPU fp64 | SHTns GPU fp32 | s2fft GPU fp64 (HEALPix) |
|---|---|---|---|---|---|---|
| 256 | 0.4 / 0.4 | 1.8 / 2.3 | 0.66 / 0.81 | 0.46 / 0.39 | 0.32 / 0.27 | 114 / 110 |
| 512 | 1.8 / 1.8 | 11.9 / 15.4 | 4.1 / 3.2 | 2.8 / 2.5 | 1.9 / 1.7 | 649 / 719 |
| 1024 | 6.9 / 7.2 | 53.7 / 58.1 | 16.1 / 16.2 | 17.5 / 17.4 | 13.3 / 12.7 | -- |
| 2048 | 36.5 / 41.9 | 281.6 / 290.8 | 87.5 / 88.6 | 129.5 / 132.5 | 99.9 / 98.1 | -- |
| 4096 | 301.9 / 266.5 | 1874 / 1850 | 571 / 574 | 2138 / 1032 | 767 / 770 | -- |

SHTns 3.7.5 was built from source here (`--enable-openmp --enable-cuda=120`, CUDA 13.0) and
timed with device-resident buffers (`benchmarks/independent/bench_shtns.py`); its accuracy against
ducc0 on the same grid was 1e-12 (fp64) and 7e-6..1e-4 (fp32).  s2fft (`method="jax"`, on-the-fly
recursion; its CUDA extension is not built here) is `benchmarks/independent/bench_s2fft.py`;
its fp32 mode gave 9e-2 errors and is not reported.

**Non-claims.**  SHTns runs on its own Gauss-Legendre grid, which has ~25 % fewer rings than
HEALPix at the same `lmax`, so its column is a different (cheaper) operator, not a HEALPix
transform.  cunuSHT was not timed.  The D&C engine is gated to `L >= 3072`; below it the march is
at parity or faster, and the table above is the march.

## 4. Where the remaining time is (Nside 2048, one paired synthesis + analysis, nsys)

merge FMM 23.9 ms, direct merges 12.1 ms, CD FMM 11.3 ms, leaves 3.7 ms, XLA packing ~3.4 ms.  The
D&C route does ~7x fewer flops than the march but runs at ~11 % of fp32 peak (the FMM passes are
latency-bound: one block per node, a `__syncthreads` per tree level); that, not the algorithm, is
the headroom.
