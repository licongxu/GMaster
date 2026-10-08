# SHTns on the HEALPix grid, fp32 recurrence, vs GMaster and NaMaster

SHTns (v3.7-33, `/scratch/scratch-lxu/shtns_build/shtns-git`) has no HEALPix grid.
`shtns_healpix.patch` adds a Legendre-only GPU mode to it:

* `cushtns_set_latitudes(cfg, cos_theta, weights)` replaces the plan's colatitudes and
  ring weights.  HEALPix is passed as `4 nside` latitudes with the equator listed
  twice at half weight, so the grid stays even and symmetric.
* `cu_SH_to_fourier_float` / `cu_fourier_to_SH_float` run only SHTns's Legendre kernels.
* `SHTNS_GPU_NO_FFT=1` skips the cuFFT plan and the init self-test, both of which need
  SHTns's own uniform-ring FFT.  It also disables SHTns's fp32 "separate mean" trick in
  the m = 0 analysis.  That trick assumes the quadrature annihilates `Y_l0` for l > 0,
  which holds on Gauss nodes and not on HEALPix (it left 4e-4 errors in m = 0).

`shtns_healpix.py` does the HEALPix longitude step on the GPU with cupy.  The
`2 nside + 1` belt rings go through one batched FFT, and the polar-cap rings through
one batched Bluestein FFT.  Modes m ≥ n_ring alias onto the ring, and each ring's
phase offset is applied.  The recurrence is SHTns's own:

* `SHTNS_GPU_REC_PREC=1` forces SHTns's standard three-term recurrence in fp32.  SHTns
  turns off its Ishioka variant in that mode.
* `SHTNS_GPU_REC_PREC=2` forces it to fp64.
* With the plain `SHT_FP32` flag, SHTns picks fp64 itself for lmax > 128 on this card.

Data, FFTs and sums are fp32 in all modes.  Only spin 0 is implemented.

Build: apply the patch, then
`CUDA_PATH=/usr/local/cuda-13.0 python setup.py build_ext --inplace`.
Runtime: cupy-cuda12x needs `nvidia/cufft/lib` and `nvidia/cuda_nvrtc/lib` (CUDA 12)
on `LD_LIBRARY_PATH`.

## Runs (2026-10-08, GPU0 RTX PRO 6000 Blackwell, idle; GPU1 was held by another job)

All runs use HEALPix, lmax = 3 nside - 1 and spin 0.

### 1. ACT DR6 PA4 f150 masked temperature map, n_iter = 3 (`bench_pcl.py`)

The reference is NaMaster 2.x in fp64 (ducc0 on the CPU, 8 threads).  "fp32 rec" is
SHTns with `SHTNS_GPU_REC_PREC=1`, "march" is GMaster's fp32 (v, D) march, and "fp64 rec"
is the SHTns control.  σ_CV = C_l sqrt(2 / ((2l+1) f_sky)).  The bandpowers are binned
at Δℓ = 30.

| Nside | code | alm rms rel. err | Cl median rel. err | max ΔCl/σ_CV | max ΔBP/σ_CV | map2alm n_iter=3 |
|---|---|---|---|---|---|---|
| 256  | SHTns fp32 rec | 1.2e-5 | 1.5e-6 | 1.7e-4 | 5.8e-4 | 1.54 ms |
| 256  | GMaster march  | 3.3e-6 | 2.9e-7 | 2.4e-5 | 4.3e-5 | 1.62 ms |
| 1024 | SHTns fp32 rec | 4.8e-5 | 1.1e-5 | 2.0e-3 | 7.3e-3 | 31.6 ms |
| 1024 | GMaster march  | 1.4e-5 | 3.3e-6 | 2.2e-4 | 9.1e-4 | 41.6 ms |
| 2048 | SHTns fp32 rec | 9.6e-5 | 8.5e-6 | 4.1e-3 | 6.7e-3 | 184 ms |
| 2048 | GMaster march  | 2.6e-5 | 9.8e-6 | 6.8e-4 | 3.3e-3 | 246 ms |
| 4096 | SHTns fp32 rec | 6.3e-4 (max 5.6e-3) | 1.0e-5 | 1.2e-2 | 1.6e-2 | 1.22 s |
| 4096 | GMaster march  | 5.2e-5 (max 5.1e-5) | 8.2e-6 | 1.6e-3 | 7.9e-3 | 2.80 s |
| 4096 | SHTns fp64 rec | 2.2e-6 | 1.1e-7 | 1.7e-5 | 5.4e-5 | 7.49 s |

### 2. White-spectrum stress test (`bench_white.py`)

Every (l, m) has unit variance, so low ℓ cannot hide the recurrence error.  The reference
is ducc0 in fp64.  Synthesis errors are relative to the map rms.  The analysis is a single
pass (n_iter = 0) compared with ducc0's own single pass.

| Nside | SHTns fp32 rec: syn rms / **syn max** / ana rms | GMaster march | SHTns fp64 rec |
|---|---|---|---|
| 256  | 2.5e-5 / 3.3e-3 / 2.6e-5 | 3.3e-6 / 2.6e-5 / 2.7e-6 | 7.8e-7 / 4.4e-5 / 1.1e-6 |
| 512  | 5.4e-5 / 1.7e-2 / 4.8e-5 | 6.3e-6 / 6.3e-5 / 5.1e-6 | 1.1e-6 / 9.0e-5 / 2.0e-6 |
| 1024 | 1.1e-4 / 5.0e-2 / 1.1e-4 | 1.2e-5 / 1.3e-4 / 9.9e-6 | 1.5e-6 / 1.7e-4 / 3.7e-6 |
| 2048 | 2.4e-4 / 2.2e-1 / 2.2e-4 | 2.4e-5 / 2.6e-4 / 2.0e-5 | 2.1e-6 / 3.9e-4 / 6.9e-6 |
| 4096 | 6.0e-4 / **2.8**  / 5.4e-4 | 4.8e-5 / 5.0e-4 / 3.9e-5 | 2.9e-6 / 8.1e-4 / 1.3e-5 |

The plain fp32 recurrence has an rms error about 10× GMaster's at every Nside, and its
maximum error blows up at the poles.  At Nside 1024 the worst rings are within 0.05° of
a pole (5e-2), against 4.8e-4 for |latitude| < 60°.  That is the near-double
characteristic root of the three-term recurrence (`gmaster/_sht/march_v2.py` docstring).
The (v, D) difference form removes it: GMaster's max/rms ratio stays about 10 at every
Nside.  The fp64-recurrence control's max error (8e-4 at 4096) is the fp32 ring-FFT
stage of `shtns_healpix.py`, not the recurrence.

### Timing notes

* Times are on the same GPU, warm, median of 5.
* SHTns-HEALPix with its fp32 recurrence is 1.3× faster than GMaster's march at
  Nside 1024–2048 and 2.3× faster at 4096 (map2alm, n_iter = 3).  It is at par at 256.
* The cupy ring-FFT stage is 20–60 % of one SHTns analysis, so the Legendre kernel
  alone is faster still.
* With an fp64 recurrence, SHTns is 4–7× slower than with fp32 on this card, which
  has 1/64-rate fp64.
