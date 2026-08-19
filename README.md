# GMaster

GPU implementation of [NaMaster](https://github.com/LSSTDESC/NaMaster):
pseudo-\(C_\ell\) (MASTER) power spectra of masked spin fields.

CPU reference: `/home/lxu/scratch/agent_dev/auto_research_agent/NaMaster`
(LSSTDESC/NaMaster, `pymaster` 3.0).

## Status

GMaster is under active development. Its exported computational API now covers
NaMaster 3.0's public functions and classes. Implemented and checked directly
against NaMaster are curved- and flat-sky binning; HEALPix and CAR
scalar/arbitrary-spin transforms; standard and pure-E/B fields;
contaminant-template deprojection; anisotropic spin masks; coupled
pseudo-spectra; arbitrary-spin, joint-TEB, and Toeplitz MASTER workspaces; beam
convolution; MASTER/FKP normalization; decoupling; bandpower windows;
deprojection bias; full-estimator calls; and NaMaster-compatible FITS I/O.
Arbitrary-position catalog, clustering, and momentum fields include shot-noise
and deprojection terms. Curved- and flat-sky Gaussian covariance workspaces,
covariance evaluation, iNKA spectra, mask apodization, and correlated
spherical/flat Gaussian simulations are also implemented. NaMaster's own
spin-0/spin-2 and isotropic-weight restrictions on Gaussian covariance remain.
Numerical kernels are JAX device arrays and run on CPU or GPU. Set
`JAX_ENABLE_X64=1` for scientific calculations.

Curved general-spin and pure-E/B coupling matrices use exact Gauss-Legendre
quadrature, stable fixed-index Wigner-d/Jacobi recurrences, and accelerator
matrix multiplication. This replaces the former quartic Racah fallback with a
cubic algorithm. The scalar TT kernel retains its optimized recurrence. The
Toeplitz scalar path evaluates only entries retained by NaMaster's approximation
(the exact low multipoles, protected diagonal band, diagonal, and reference
column), then reconstructs the rest. It therefore avoids computing a full exact
matrix that would immediately be overwritten. The
flat-sky coupling and covariance kernels use circular FFT convolutions instead
of NaMaster's nested Fourier-mode loops. Direct CAR and arbitrary-position
catalog transforms still scale as `O(npix*lmax^2)` and are the main remaining
targets for large-map optimization.

An initial warmed scalar flat-workspace benchmark on an NVIDIA GPU gave a
13.8x speedup over NaMaster's CPU implementation at 512x512 pixels (0.177 s
versus 2.45 s). Reproduce it with
`python -m benchmarks.benchmark_flat_workspace --sizes 512`; timings depend on
the available CPU, GPU, and thread settings.

The warmed flat-covariance workspace benchmark at 512x512 pixels is currently
208x faster on the same NVIDIA GPU/CPU host (0.0192 s versus 4.00 s), while
matching NaMaster to floating-point precision. Reproduce it with
`XLA_PYTHON_CLIENT_PREALLOCATE=false python -m benchmarks.benchmark_flat_covariance --sizes 512`.

At `lmax=2047`, the warmed curved-sky benchmark measures 15.49x for an
arbitrary-spin coupling matrix (0.0533 s versus 0.825 s) and 4.14x for pure
0x2 coefficients (0.437 s versus 1.81 s). The maximum relative differences in
that run were approximately `3.1e-8` and `7.1e-9`. Reproduce it with
`XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORM_NAME=gpu python -m benchmarks.benchmark_curved_coupling --lmax 2047`.

For a scalar kernel at `lmax=4095` with `l_toeplitz=600`, `l_exact=100`, and
`dl_band=50`, selective evaluation takes 0.0576 s after compilation versus
1.709 s for ordinary exact NaMaster on the same host: 29.7x faster. Its output
matches NaMaster's Toeplitz result to a maximum absolute difference of
`8.7e-14`. Reproduce this with:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORM_NAME=gpu \
  python -m benchmarks.benchmark_scalar_toeplitz --lmax 2047 4095
```

## Nside 4096 memory

At `Nside=4096`, one float64 HEALPix map is 1.50 GiB and one packed complex128
alm array at the default `lmax=12287` is 1.125 GiB. Jacobi refinements are
dispatched as repeated single-iteration XLA programs, rather than unrolled in
one graph. A quadratic extrapolation of XLA's compiled buffers from `Nside=16`
estimates 38.0 GiB peak for scalar refinement and 51.9 GiB for spin-2 on the
current S2FFT backend. Inspect the estimate on the target accelerator with:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m benchmarks.benchmark_nside4096_memory
```

Use `lite=True` for fields when input maps and templates do not need to remain
resident after their alms are computed. Device-resident JAX masks are accepted
without a GPU-to-host-to-GPU round trip. The memory figures above are planning
estimates, not a substitute for hardware telemetry during a full-resolution
run. The current on-the-fly S2FFT transform is correctness-tested against
NaMaster, but an end-to-end `Nside=4096` runtime speedup over DUCC has not yet
been demonstrated; coupling-matrix acceleration is the currently measured GPU
speedup.

Run the parity suite on CPU with:

```bash
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python -m pytest -q
```

The current suite contains 108 tests, including direct API/numerical comparisons
against NaMaster. GPU tests use the same suite with a CUDA JAX platform.
