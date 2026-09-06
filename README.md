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

## Fused and multi-GPU spherical harmonic transforms

Scalar HEALPix transforms on NVIDIA GPUs use a fused Pallas/Triton kernel at
`L>=128`. One persistent program evaluates each associated-Legendre recurrence,
instead of launching three small kernels per harmonic order. Exact north-south
parity halves the latitude domain, powers-of-two rescaling protects forbidden
regions without changing float64 values, and HEALPix quadrature and ring phases
are fused into the recurrence. This avoids materializing the complex
`ntheta x L` phase matrix. The custom JAX transpose uses the same fused kernels,
so map and alm gradients remain available.

HEALPix analysis now splits the on-the-fly Wigner recurrence by harmonic band,
and synthesis splits it by latitude. The two independent kernels are dispatched
concurrently. Packed Healpy alms are expanded with an algebraic gather rather
than a GPU scatter; at `L=384` this unpack kernel fell from 0.925 s to 0.00161 s.
Spin E/B alms are fused directly into one complex harmonic grid so full E, B,
and combined grids are not simultaneously materialized.

The default `jax` calculator selects the fused scalar kernel on NVIDIA GPUs,
shards it across two GPUs at `L>=2048` when available, and continues to select
two GPUs for generic spin transforms at `L>=768`. Traced/autodifferentiated
scalar calls stay on one GPU because the explicit cross-device staging is not a
JAX primitive. Selection can be controlled explicitly:

```python
import gmaster as nmt

nmt.set_sht_calculator("jax")         # automatic, the default
nmt.set_sht_calculator("jax-single")  # force one fused device
nmt.set_sht_calculator("jax-mgpu")    # request two devices
nmt.set_sht_calculator("jax-generic") # force generic S2FFT reference path
```

The precomputed transform tables (the scalar Legendre band and the polarised
Wigner-d triangles) are dispatched by a fit test, so their storage precision
selects the route as well as the bytes:

```python
nmt.set_table_precision("fp64")  # default
nmt.set_table_precision("fp32")  # half the table bytes
```

The recurrence that generates the table values and every contraction over them
stay in float64 either way, so the only cost is the table's own representation
error: `6.5e-9` relative on the decoupled `Cl` at `Nside=512` spin 0 and
`6.3e-8` at `Nside=1024`, against `1.7e-12` with float64 tables. Halving the
bytes is what lets `Nside=1024` build its band at all and take the
memory-bound contraction instead of the recurrence kernel.

On two RTX PRO 6000 Blackwell GPUs, warmed scalar analysis at `Nside=512`
improves from 0.7057 s on one GPU to 0.3255 s on two (2.17x), while synthesis
improves from 0.6517 s to 0.3750 s (1.74x). At `Nside=1024`, analysis improves
from 7.934 s to 3.406 s (2.33x). Single- and two-GPU analysis agree to better
than `4.6e-16` maximum absolute error in that run. Reproduce the complete
single-GPU, two-GPU, and NaMaster comparison with:

```bash
CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m benchmarks.benchmark_sht --nside 512
```

The fused kernel changes the single-GPU result substantially. At `Nside=512`,
warmed analysis takes 0.0470 s and synthesis 0.0528 s, versus 0.8097 s and
0.7494 s for the generic single-GPU path: 17.2x and 14.2x faster. Maximum
differences from NaMaster in that run were `1.93e-14` for alms (relative
`2.51e-12`) and `3.20e-8` for a white unit-variance synthesis map (relative
`1.02e-11`). NaMaster's CPU DUCC transform still took 0.0207 s and 0.0136 s,
respectively. The fused GPU SHT is therefore a large acceleration over the
previous GPU algorithm, but it does **not** yet beat DUCC. The demonstrated
NaMaster speedups apply to the MASTER coupling, flat-workspace, and covariance
kernels above.

## Batched chirp-Z ring FFT and coefficient tables

HEALPix ring FFTs use one uniform batched chirp-Z transform per direction for
all rings simultaneously, replacing s2fft's per-ring-size JAX path that unrolls
hundreds of tiny polar FFT groups. Chirp angles use exact integer modular
reduction, so results match the reference transforms to floating-point
round-off (`<=5e-13` at Nside 512). At Nside 512 the forward ring FFT drops
from 20.2 ms to 4.8 ms and the inverse from 31.1 ms to 10.7 ms; Nside-4096
compilation drops from 182 s to 51 s because the polar no longer unrolls.

The fused latitudinal kernel keeps NaMaster-exact normalized-recurrence
arithmetic but loads its `c1/c2` coefficients from tables computed once by
parallel vector operations, removing square roots and divisions from the
sequential degree loop, and streams inputs, outputs, and accumulators through
transposed contiguous layouts. Combined effect at Nside 512: warmed analysis
45.0 -> 25.3 ms and synthesis 51.7 -> 29.4 ms against NaMaster's 14.2 / 11.5 ms,
with unchanged parity (alms max abs `1.94e-14`). The full Nside-4096
constant-map analysis now takes 10.84 s warm on one GPU (was 12.25 s) with
`a00=sqrt(4*pi)` exact to 1e-15.

An unnormalized-recurrence variant with deferred K normalization was built and
rejected: forward-l application of the DLMF recurrence is numerically unstable
for m > 0 (measured relative error 5.8e-5); the normalized recurrence is kept.


At `Nside=1024`, two fused GPUs improve analysis from 0.2532 s to 0.2092 s
(1.21x) and synthesis from 0.4556 s to 0.3168 s (1.44x). At `Nside=512`, PCIe
staging outweighs the analysis gain, which is why automatic sharding starts at
`L=2048` rather than whenever a second GPU exists.

## Nside 4096 memory

At `Nside=4096`, one float64 HEALPix map is 1.50 GiB and one packed complex128
alm array at the default `lmax=12287` is 1.125 GiB. Jacobi refinements are
dispatched as repeated single-iteration XLA programs, rather than unrolled in
one graph. A quadratic extrapolation of the single-device compiled graphs from
`Nside=16` estimates 36.3 GiB peak for scalar refinement and 43.9 GiB for
spin-2. The two-GPU staged recurrence estimate is about 19 GiB on each device
for scalar and at most 24 GiB on either device for spin-2, before caller-retained
buffers, FFT workspaces, and allocator headroom. Inspect both estimates with:

```bash
CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m benchmarks.benchmark_nside4096_memory
```

A full scalar `Nside=4096`, `lmax=12287`, `n_iter=0` constant-sky analysis now
runs on one GPU. First compilation plus execution took 182.0 s and the warmed
transform took 12.248 s, 18.2x faster than the former 222.4 s generic two-GPU
path. The fused two-GPU path takes 7.059 s, a further 1.74x improvement and a
31.5x total GPU-algorithm speedup. It returned
`a00=3.544907701811031+1.2e-16j`, matching `sqrt(4*pi)`.
Comparison of all 75.5 million packed coefficients with NaMaster gave maximum
absolute difference `4.78e-13` (relative `1.35e-13`). NaMaster took 1.764 s, so
the exact fused transforms remain 6.95x (one GPU) and 4.03x (two GPUs) slower
than DUCC.

Actual device telemetry peaked at 36,652 MiB including the 236 MiB baseline,
or about 35.6 GiB incremental. This removes the former 32 GiB allocation on a
second GPU. Use a device with at least 48 GiB free for allocator and FFT
workspace headroom. XLA allocator pools can exceed the principal-array inventory
reported by `benchmark_nside4096_memory`.

For the fused two-GPU run, the primary peak was unchanged and the secondary
needed about 9.9 GiB beyond its existing allocation, versus roughly 32 GiB for
the old generic replica.

Use `lite=True` for fields when input maps and templates do not need to remain
resident after their alms are computed. Device-resident JAX masks are accepted
without a GPU-to-host-to-GPU round trip. The memory estimates remain planning
aids rather than substitutes for telemetry, especially for spin-2 and
Jacobi-refined runs that have not yet been executed at full resolution. The fused scalar and streamed spin-2 SHT transforms are correctness-tested against
NaMaster. Measured end to end on one GPU against `pymaster` (same process, clocks
forced up, `NmtField(n_iter=3)` + coupling matrix + decoupled cell), the full MASTER
pipeline is **1.57x** at `Nside=512` spin 0 with float64 tables and **2.20x** with
`set_table_precision("fp32")`; at `Nside=1024` spin 0 the band only fits in float32 and
delivers **1.99x** (900.6 ms vs 1791.1 ms).

For **spin 2**, GMaster features a streaming fused on-the-fly recurrence architecture
(`_spin_streamed.py`) that evaluates Wigner-$d$ recurrences directly in GPU registers
with periodic base-2 exponent rescaling and guarded tile padding. This eliminates
all DRAM table materialization (which would require 73.5 GiB at $N_{\rm side}=1024$,
$>300$ GiB at 2048, and $>2.4$ TiB at 4096). Combined with monolithic latitudinal
synthesis and wide-window analysis, GMaster's single-GPU spin-2 SHT now outperforms
192-thread CPU DUCC / NaMaster across all resolutions up to $N_{\rm side}=2048$:
- **$N_{\rm side}=256$**: 2.0 ms GPU vs 19.8 ms CPU (**9.9x faster**)
- **$N_{\rm side}=512$**: 9.6 ms GPU vs 50.1 ms CPU (**5.2x faster**)
- **$N_{\rm side}=1024$**: 126.3 ms GPU vs 183.4 ms CPU (**1.45x faster**)
- **$N_{\rm side}=2048$**: 784.3 ms GPU vs 851.8 ms CPU (**1.09x faster**)
- **$N_{\rm side}=4096$**: 5.25 s on 1 GPU (was 33.5 s, **6.4x GPU speedup**), with only 3.2 GB peak VRAM.

Run the parity suite on CPU with:

```bash
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python -m pytest -q
```

The current CPU suite contains 111 passing tests and 5 hardware-dependent skips,
including direct API/numerical comparisons against NaMaster. Dedicated NVIDIA
tests cover fused scalar parity and gradients; two-GPU tests cover scalar and
spin-2 analysis, synthesis, and Jacobi refinement.
