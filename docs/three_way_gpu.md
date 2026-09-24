# GPU transforms: GMaster D&C vs GMaster v2 march vs SHTns (Nside 64 - 4096)

24 September 2026, branch `exp/cpu-dform` (commit 921b86b and later), RTX PRO 6000 Blackwell (GPU0),
`XLA_PYTHON_CLIENT_PREALLOCATE=false`.  One process per (engine, Nside, spin) point, median of 5
warm calls.  Raw lines: `benchmarks/independent/three_way_gpu0.log`; script
`benchmarks/independent/bench_three_way.py`; tables / figure `three_way_report.py` ->
`docs/three_way_gpu.png`.

**What is compared.**
* *GMaster D&C* -- forced at every size (`GMASTER_DC_MIN_L=0`; the shipped router only uses it from
  Nside 1024 up) -- and *GMaster v2 march* (`GMASTER_DC=0`): the same HEALPix operators, lmax =
  3 Nside - 1: `alm2map`, `map2alm` with n_iter = 0 and with n_iter = 3 (NaMaster's default).
  Errors are max |d| / max |ref| against NaMaster (ducc0) on the identical operator.
* *SHTns 3.7.5* on its own Gauss-Legendre grid (lmax + 1 rings, exact quadrature, so no iteration
  and ~25 % fewer rings than HEALPix): a different, cheaper operator, not a HEALPix transform.  Its
  error is against ducc0 on that GL grid.  **SHTns has no spin-2 transform**; the spin-2 column is
  its spin-1 vector (spheroidal / toroidal) GPU transform, fp64 (its Python layer exposes no fp32
  vector path), with a round-trip error -- the closest analogue, not a like-for-like number.
* Memory: device-memory growth over the run (cudaMemGetInfo, includes pool retention), and XLA's
  accounted peak in brackets.  SHTns: fp64 runs (the fp32 run shared a process with fp64, so its
  own figure is not separable).


### spin 0: time (ms) synthesis / analysis n_iter=0 / analysis n_iter=3

| Nside | GMaster D&C | GMaster v2 march | SHTns GPU fp32 (GL) syn / ana |
|---|---|---|---|
| 64 | 0.26 / 0.27 / 0.41 | 0.28 / 0.25 / 0.52 | 0.04 / 0.03 (fp64 0.06 / 0.04) |
| 128 | 0.34 / 0.31 / 0.73 | 0.44 / 0.29 / 0.79 | 0.08 / 0.06 (fp64 0.11 / 0.08) |
| 256 | 0.52 / 0.62 / 1.31 | 0.52 / 0.48 / 1.79 | 0.32 / 0.27 (fp64 0.46 / 0.39) |
| 512 | 1.77 / 1.97 / 4.11 | 1.87 / 1.93 / 10.00 | 1.93 / 1.72 (fp64 2.76 / 2.46) |
| 1024 | 5.85 / 6.67 / 15.35 | 6.90 / 7.34 / 42.62 | 13.31 / 12.71 (fp64 17.57 / 17.48) |
| 2048 | 23.67 / 29.24 / 57.40 | 36.90 / 42.32 / 250.75 | 99.92 / 98.14 (fp64 129.57 / 132.48) |
| 4096 | 122.61 / 145.07 / 250.23 | 510.97 / 453.28 / 3110.11 | 767.16 / 767.03 (fp64 990.32 / 1028.14) |

### spin 0: memory (GiB, device growth; XLA peak in brackets) and accuracy

| Nside | D&C mem | march mem | SHTns mem | D&C err syn / ana0 / ana3 | march err syn / ana0 / ana3 | SHTns err |
|---|---|---|---|---|---|---|
| 64 | 0.06 [0.05] | 0.09 [0.09] | 0.01 (fp64; fp32 n/a) | 1.5e-06 / 2.1e-06 / 1.5e-06 | 1.5e-06 / 2.1e-06 / 1.5e-06 | fp32 1.8e-06, fp64 2.1e-13 |
| 128 | 0.12 [0.07] | 0.12 [0.07] | 0.03 (fp64; fp32 n/a) | 2.5e-06 / 3.7e-06 / 2.0e-06 | 2.5e-06 / 3.7e-06 / 3.0e-06 | fp32 4.5e-06, fp64 7.0e-13 |
| 256 | 0.35 [0.17] | 0.32 [0.15] | 0.08 (fp64; fp32 n/a) | 4.7e-06 / 7.4e-06 / 2.3e-06 | 4.7e-06 / 7.4e-06 / 6.7e-06 | fp32 6.8e-06, fp64 1.4e-12 |
| 512 | 0.72 [0.56] | 0.75 [0.54] | 0.30 (fp64; fp32 n/a) | 2.1e-06 / 2.4e-06 / 2.5e-06 | 1.2e-05 / 1.3e-05 / 1.3e-05 | fp32 1.4e-05, fp64 8.5e-12 |
| 1024 | 2.73 [2.34] | 2.66 [2.14] | 1.29 (fp64; fp32 n/a) | 3.6e-06 / 3.0e-06 / 3.1e-06 | 2.4e-05 / 2.8e-05 / 2.6e-05 | fp32 2.6e-05, fp64 2.2e-11 |
| 2048 | 11.09 [9.77] | 10.21 [8.52] | 5.14 (fp64; fp32 n/a) | 4.8e-06 / 6.0e-06 / 6.1e-06 | 4.6e-05 / 5.4e-05 / 5.5e-05 | fp32 5.4e-05, fp64 4.9e-10 |
| 4096 | 40.96 [37.52] | 40.55 [25.03] | 20.55 (fp64; fp32 n/a) | 5.8e-06 / 1.0e-05 / 1.0e-05 | 8.5e-05 / 1.3e-04 / 1.2e-04 | fp32 9.6e-05, fp64 1.3e-09 |

### spin 2: time (ms) synthesis / analysis n_iter=0 / analysis n_iter=3

| Nside | GMaster D&C | GMaster v2 march | SHTns GPU fp64 spin-1 vector (GL) syn / ana |
|---|---|---|---|
| 64 | 0.45 / 0.74 / 1.59 | 0.33 / 0.30 / 0.77 | 0.13 / 0.08 |
| 128 | 0.65 / 0.86 / 1.93 | 0.43 / 0.35 / 1.54 | 0.22 / 0.16 |
| 256 | 1.23 / 1.64 / 2.81 | 0.84 / 0.67 / 4.38 | 0.94 / 0.79 |
| 512 | 2.72 / 3.63 / 7.31 | 2.74 / 2.35 / 15.90 | 5.55 / 4.95 |
| 1024 | 12.23 / 14.46 / 27.38 | 13.14 / 13.46 / 90.32 | 35.22 / 35.08 |
| 2048 | 62.40 / 71.61 / 117.75 | 90.35 / 91.75 / 629.94 | 259.27 / 265.20 |
| 4096 | 253.46 / 294.35 / 468.97 | 695.29 / 706.69 / 4966.75 | 1989.39 / 2066.43 |

### spin 2: memory (GiB, device growth; XLA peak in brackets) and accuracy

| Nside | D&C mem | march mem | SHTns mem | D&C err syn / ana0 / ana3 | march err syn / ana0 / ana3 | SHTns err |
|---|---|---|---|---|---|---|
| 64 | 0.02 [0.01] | 0.09 [0.09] | 0.01 | 7.4e-07 / 8.3e-07 / 7.7e-07 | 1.6e-06 / 1.7e-06 / 1.5e-06 | round trip 4.5e-13 |
| 128 | 0.08 [0.06] | 0.18 [0.12] | 0.04 | 7.9e-07 / 9.9e-07 / 9.5e-07 | 2.7e-06 / 4.1e-06 / 3.5e-06 | round trip 3.3e-12 |
| 256 | 0.30 [0.23] | 0.38 [0.21] | 0.12 | 2.6e-06 / 2.7e-06 / 2.2e-06 | 5.8e-06 / 6.9e-06 / 5.7e-06 | round trip 5.5e-12 |
| 512 | 1.31 [0.94] | 1.42 [0.77] | 0.41 | 4.5e-06 / 4.0e-06 / 3.4e-06 | 1.2e-05 / 1.5e-05 / 1.3e-05 | round trip 5.6e-11 |
| 1024 | 4.60 [3.87] | 5.39 [2.99] | 1.72 | 5.2e-06 / 4.2e-06 / 3.8e-06 | 2.3e-05 / 2.7e-05 / 2.8e-05 | round trip 1.8e-10 |
| 2048 | 17.71 [14.35] | 24.35 [12.26] | 6.84 | 6.1e-06 / 6.8e-06 / 5.4e-06 | 4.7e-05 / 5.4e-05 / 5.2e-05 | round trip 3.6e-09 |
| 4096 | 71.90 [59.88] | 64.75 [43.86] | 27.30 | 1.3e-05 / 3.2e-05 / 1.6e-05 | 8.4e-05 / 1.2e-04 / 1.1e-04 | round trip 7.5e-09 |


## Reading

* **Speed.**  From Nside 1024 up the D&C engine is the fastest of the three on every operation.
  At Nside 4096 spin 0: synthesis 123 ms vs 511 ms (march) and 767 ms (SHTns fp32, cheaper grid);
  map2alm with n_iter = 3 250 ms vs 3110 ms (march, 12.4x).  Spin 2, 4096: synthesis 253 vs 695 ms,
  map2alm n_iter = 3 469 vs 4967 ms (10.6x); SHTns' spin-1 vector synthesis is 1989 ms.  The n_iter = 3
  gap is the node-space refinement (the tree runs once instead of 2 n_iter + 1 times).
  Below Nside 512 the march is as fast or faster (spin 2 at 64-256 clearly so), which is why the
  shipped router keeps it there; SHTns' fp32 path is fastest at Nside <= 256.
* **Accuracy.**  D&C 2e-6 .. 3e-5 against ducc0 at every size; the march grows to 1.2e-4 at 4096
  (the D&C route is ~10x closer from 1024 up).  SHTns fp32 is at the march's level (9.6e-5 at
  4096); its fp64 path is ~1e-9.
* **Memory -- the D&C engine's cost.**  Its plan is O(L^2 log L) (the march's tables are O(L^2)):
  XLA peak at 4096 is 37.5 vs 25.0 GiB (spin 0) and 59.9 vs 43.9 GiB (spin 2); device growth
  41.0 vs 40.6 GiB and 71.9 vs 64.7 GiB.  Up to Nside 2048 the two routes are within ~1-2 GiB
  (spin 2 at 2048: 14.4 vs 12.3 GiB peak).  SHTns fp64 needs 20.6 GiB (spin 0) / 27.3 GiB
  (spin-1 vector) at 4096 on its smaller grid.
