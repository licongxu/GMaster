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

### MASTER pipeline, spin 0 (`benchmarks/benchmark_pipeline._run_pipeline`, n_iter = 3, dl = 30)

| Nside | NaMaster 96c | GMaster, session start | GMaster now | ratio now | GPU peak now / march |
|---|---|---|---|---|---|
| 512 | 314 ms | 22.5 ms (report) | 15.8 ms (march route; D&C gated off) | 19.9x | 0.9 GiB |
| 1024 | 1638 ms | 82 ms | 64.1 ms | 25.6x | 3.6 / 3.4 GiB |
| 2048 | 10034 ms | 472 ms | 278.6 ms | 36.0x | 14.8 / 13.5 GiB |
| 4096 | 71583 ms (report) | 7458 ms (report) | 1450.9 ms | 49.3x | 58.9 / 58.0 GiB |

Decoupled bandpowers vs NaMaster: 3.6e-6 (1024), 1.15e-5 (2048).  At 4096 every GMaster route --
including the exact fp64 one with the march off and fp64 rings -- sits 1.56e-4 from NaMaster, a
uniform ratio (median -1.53e-4) that fp64 coupling and fp64 rings do not move; it is a pre-existing
GMaster/NaMaster difference, not transform error.  Against GMaster's exact fp64 route the new
engine is 1.2e-6 and the shipped fp32 march 7.9e-6.

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
