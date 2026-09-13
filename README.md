# GMaster

GPU implementation of [NaMaster](https://github.com/LSSTDESC/NaMaster):
pseudo-\(C_\ell\) (MASTER) power spectra of masked spin fields.

CPU reference: `/home/lxu/scratch/agent_dev/auto_research_agent/NaMaster`
(LSSTDESC/NaMaster, `pymaster` 3.0).

## Status

The one-GPU HEALPix MASTER comparison through `Nside=4096` is closed (HANDOFF
addendum 33). GMaster TOTAL is strictly below NaMaster TOTAL on every cell the
reference can run (1.9–4.2× at 64–512, 2.1–2.9× at 1024–4096, both spins).
`Nside=4096` spin 2 is GMaster-only: NaMaster segfaults; GMaster finishes in
50.6 s with finite decoupled bandpowers. Isolated SHT on the shipped march
beats DUCC at 1024–4096 in both spins, including the polarised 4096 pass
(1.38× / 1.74×).

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
Jacobi-refined runs that have not yet been executed at full resolution. The
fused scalar and generic spin S2FFT transforms are correctness-tested against
NaMaster. Measured end to end on one GPU against `pymaster` (same process, clocks
forced up, `NmtField(n_iter=3)` + coupling matrix + decoupled cell), the full MASTER
pipeline is **1.57x** at `Nside=512` spin 0 with float64 tables and **2.20x** with
`set_table_precision("fp32")`; at `Nside=1024` spin 0 the band only fits in float32 and
delivers **1.99x** (900.6 ms vs 1791.1 ms). Spin 2 takes 425.8 ms against 657.1 ms at
`Nside=512` (**1.54x**; the same 426 ms measured against a 708 ms reference run is 1.68x — the
CPU reference varies ~8% run to run) and still loses badly at `Nside=1024`, where the precomputed Wigner-d
layout that would replace the generic transform is 73.5 GiB in float32 and the build
cannot assemble it inside the pool.

Run the parity suite on CPU with:

```bash
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python -m pytest -q
```

The same suite runs at the float32 table precision:

```bash
python -m pytest -q tests --gm-precision=fp32 --gm-ring-precision=fp64
```

`--gm-precision` pins the whole session to `set_table_precision("fp32")` — re-asserting it
before every test, so a module that restores the default cannot silently move the rest
of the run back to float64. `--gm-ring-precision` then controls the azimuthal transforms
independently, which matters because `set_table_precision` used to drag them down with it:
the float32 chirp tables cast the *pixels* too, so a float32 session analyzed the map
itself in float32. Held to `complex128` the whole suite passes — **193 passed, 3 skipped** — the same counts as
the default float64 run.
Measured against it on the whole board with the shipped fp64 tables (`--ring-precision fp32`),
letting the rings fall to `complex64` costs 2-6 % of wall (698 → 656 ms at `Nside=1024` spin 0,
164 → 160 ms at 512 spin 0, 8694 → 8457 ms at 2048 spin 2), leaves spin-2 accuracy untouched
(`rel dCl` 3.69e-06 → 3.45e-06 at 1024, 1.09e-05 both ways at 2048) and puts spin 0 at ~1e-07; it
is therefore not the default. What it buys is memory, and it buys it at both ends: peak host RSS
falls 27.0 → 18.9 GB at `Nside=1024` spin 2 and 91.1 → 58.8 GB at 2048 spin 2, the device peak
3.6 → 2.2 GiB and 13.5 → 8.2 GiB there (HANDOFF addendum 29).

The knob used to be **forward-only**, and the recorded diagnosis was wrong. With fp32 rings the
gradient of the scalar transform raised
`lax.mul requires arguments to have the same dtypes, got complex64, complex128` in the polar chirp-Z
and the chirp-Z was blamed. The producer is the latitudinal stage: the kernel accumulates and stores
in float64 whatever element type it is handed and its transpose is the float64 synthesis kernel, so
a `complex64` operand got a `complex128` cotangent back and the first op downstream that multiplied
the two failed to lower. Both scalar adjoints in `gmaster/_sht_pallas.py` now cast the cotangent to
the operand's dtype, which is the boundary the note asked for and costs the forward pass nothing;
the float32-ring gradient then differs from the float64-ring gradient by the azimuthal stage's own
rounding (3.50e-07 analysis, 1.74e-07 synthesis) and nothing the adjoint adds, verified against a
central difference to 1.77e-11 with exact rings. The gradient parity test passes under
`--gm-ring-precision fp32` on the suite's existing two-parts-per-million floor, and the whole
suite now passes with `complex64` rings too — **193 passed, 3 skipped**, the same counts as the
default run. When quoting that
test, remember its reference is ring-precision-blind: `ring_dtype()` is read only by the four
ring-table builders, and `sht_calculator="jax-generic"` bypasses all of them, so a ring-relaxed
parity result is always a float32-ringed transform against a float64 reference (the generic alms
and gradients are bit-identical between the two ring precisions).

The pipeline benchmark takes the same flag, and this is what it scores against NaMaster in
one process (spin 0 / spin 2): **2.2x / 3.8x** at `Nside=64`, **1.8x / 3.9x** at 128,
**3.2x / 6.6x** at 256, **3.7x / 4.7x** at 512, **2.8x / 3.0x** at 1024 and
**2.0x / 2.1x** at 2048. At `Nside=32` both codes finish in 2-4 ms; GMaster takes 2 ms in
both spins (**1.8x** spin 2, spin 0 tied at the benchmark's resolution floor, where the
reference itself varies 2-5 ms run to run). The shipped fp64 default scores 1.5x / 1.4x at
32, 1.7x / 3.2x at 64, 1.8x / 3.2x at 128 and 2.5x / 3.1x at 256. At 2048 the
tables are refused in either precision and the route is worth 1.00-1.01x over the default.

The scalar refinement loop runs as one XLA program up to `lmax = 767` (`_PALLAS_TRACED_MAX_L`), which
is `Nside=256`; the Legendre band is built at top level before that route is chosen, because a traced
program may read the band but must never build it. That is worth 13.74 → 12.44 ms (9.4 %) at
`Nside=256` spin 0 with bit-identical alms and costs ~10 GB of peak host RSS during the compile;
`Nside=512` is 17 % *slower* as one program and is left op-by-op (HANDOFF addendum 26).

The **polarised** refinement loop now has the same treatment, up to `L_work = 1536`
(`_SPIN_SLAB_TRACED_MAX_L`, `Nside=512`). `map2alm` at spin 2 ran its `1+2*n_iter` Jacobi passes as
separate XLA programs — seven at `n_iter=3` — and one program over the whole sequence is
bit-identical (`max|d| = 0.000e+00` at every size tried) while removing the six boundaries between
them: 36.949 → 34.479 ms at `Nside=256` and 239.320 → 225.025 ms at 512 with the default float64
tables, 8.966 → 7.900 and 60.955 → 52.372 ms with `set_table_precision("fp32")`. On the harness the
spin-2 cell goes 56 → 52 ms at 256 (3.0x → **3.2x**) and 340 → 333 ms at 512, `dCl` and device
`bytes_in_use` unchanged (HANDOFF addendum 30). Note this is *not* contradicted by the analysis
fusion gate above: that gate prices one slab riding one boundary (+11 % at 512), this prices six
boundaries disappearing (−6 % at the same size and precision). The route is declined under an outer
trace, so a gradient still sees the per-pass boundaries, and above `Nside=512` it does not exist —
`_spin_slabs` returns no slab pair at `L_work=3072`, which is why the sizes where GMaster is
weakest are the ones this lever cannot reach.

A scalar field and its mask are now analysed by *one* program. `lmax_mask` defaults to `lmax` and
`n_iter_mask` to `n_iter` in both codes, so a field built the way a pipeline builds one carries two
independent spin-0 transforms of identical geometry over the same resident Legendre band. One
latitudinal reduction with four accumulators (real and imaginary parts of both maps) serves the pair,
so the band is read once instead of twice; per-map work is duplicated, only the slab read is shared.
Measured in one process with the fusion forced to decline as the control arm, spin-0 `TOTAL` goes
3.757 → 3.061 ms at `Nside=128`, 12.969 → 9.212 ms (**1.41x**) at 256, 68.235 → 50.262 ms (1.36x) at
512 and 469.817 → 377.517 ms (1.24x) at 1024, with the alms **bit-identical** on both halves and the
decoupled cell unmoved. Paired synthesis is refused wherever the re-layout is strided (`Nside=1024`
measured 4.19 *worse* there), so that size pairs the analysis only. Spin 2 is untouched.
`Nside>=2048` has no band either, but the route that serves it regenerates its Legendre row inside
the kernel, and that regeneration — not the bytes — is what a pass costs; two maps join one emit
channel block and share the recurrence, which costs **0.752** of two separate calls at `Nside=2048`
(435.0 ms against 578.4 ms). On the harness that takes the `Nside=2048` spin-0 cell from 2.4x to
**2.7x** (4558 → 3961 ms) with its own `rel dCl` going 1.40e-06 → 1.35e-06, so unlike the band
pairing this one is not bit-identical: widening the emit block reassociates the theta sum, and the
paired route's error against the fp64 band is the march's own error unchanged (`2.03e-04` either
way). The route is declined, never raised, for `lite`
fields, template/catalog/flat/anisotropic fields, a differing `lmax_mask` or `n_iter_mask`, or a spin-2
field whose mask is spin 0. It is also declined on memory: the fold doubles the one object a paired
program cannot split, and at `Nside=4096` that raised `RESOURCE_EXHAUSTED` (8.15 GiB short inside the
71.2 GiB pool) where the same geometry as two separate calls completes with a 16.3 GiB peak, so
`fold_pair_fits` demands the footprint from the geometry and refuses above a quarter of the device pool
— pass at 512/1024/2048, refuse at 3072/4096 here — leaving the two-call path, which is the arm that
scores **2.2x** at 4096 (`73538 → 33456 ms`, `rel dCl` 1.42e-06). Consequence for the benchmark: the
`mask` column is now ~0 at spin 0
because the work moved into `field`, which is the expected reading, not a vanished stage. The spin-0
`TOTAL`s in the two paragraphs above and below predate this; with both precision switches on, the
fused board is **7.5x / 6.7x / 6.8x / 4.8x** at `Nside=128/256/512/1024` and **2.7x** at 2048
(HANDOFF addenda 27 and 28).

```bash
python benchmarks/benchmark_pipeline.py --nside 512 --spins 0,2 \
  --precision fp32 --ring-precision fp64
```

The coupling matrix has a precision switch of its own, because both of its builders are width-bound:
the polarised quadrature's contraction is 115.9 GFLOP at `lmax=3071` and runs at 1.76 TFLOP/s in
float64, **94 % of this card's 1.88 TFLOP/s fp64 ceiling**, while the scalar build is five table
lookups and a few elementwise ops over ~n³/3 elements with no `dot` in it at all.
`GMASTER_COUPLING_PRECISION=fp32` (or `nmt.set_coupling_precision("fp32")`, read back with
`nmt.coupling_precision()`) is worth 34.3x on the contraction for a matrix error of 2.1e-06 and 9.4x on
the scalar build for 1.9e-07 (its accumulator stays float64). On the board, TOTAL against NaMaster
(spin 0 / spin 2): **3.7x / 7.4x → 5.0x / 8.8x** at `Nside=256`, **4.0x / 5.3x → 5.7x / 6.8x** at 512,
**2.9x / 3.2x → 4.1x / 4.1x** at 1024 and **2.0x / 2.2x → 2.5x / 2.8x** at 2048, with the decoupled Cls
essentially unmoved (`rel dCl` 1.34e-07 → 2.04e-07 spin 0 and 3.22e-06 → 3.21e-06 spin 2 at 1024). Below
`Nside=256` the coupling stage is sub-millisecond and the switch changes nothing. It is opt-in and
defaults to float64, since the suite holds `get_coupling_matrix()` to `atol=2e-14` against float64
tables. At the top of the board it is worth most: at `Nside=4096` spin 0 (`lmax=12287`) the stage goes
12764 → 3189 ms and the cell from **1.7x to 2.3x**, for a `rel dCl` of 1.20e-06 → 1.24e-06 — inside the
0.9 % run-to-run drift of the reference's own column. The 8.8 GiB device peak on that row belongs to
*float32 rings*, not to the coupling switch: the same geometry measures 16.3 GiB with `complex128`
rings and coupling `fp32`, and 8.8 GiB once the rings drop to `complex64`, with host RSS 49.3 GB either
way (HANDOFF addendum 29 §3b). Its `TOTAL` is then 90 % the two transform stages, and 2.50x is what a
free coupling, free cell and free decoupling would buy there — stages that run at 1.0x against `ducc0`,
and which the paired march that would share them is refused by the device pool at this size.

The benchmark also reports a `mask` stage: a field's mask `a_lm` are computed lazily (as in NaMaster),
by a second spin-0 analysis at the **same** `lmax` and the same `n_iter` (`lmax_mask` defaults to
`minfo.get_lmax()` in both codes), so the work lands in `TOTAL` but in no stage column unless it is
measured on a fresh field. It is worth naming — at `Nside=2048` spin 0 it is 1855 ms of a 4210 ms
pipeline, i.e. as much again as the science transform, and with it the columns add up to the total. The
estimator therefore runs 14 latitudinal passes per call; per pass GMaster is **2.17x** (analysis) and
**1.86x** (synthesis) faster than the ducc0 C code `pymaster` calls where the Legendre band fits
(`Nside=1024` spin 0) and 0.97-1.39x on the table-free routes. The full per-pass ladder, both spins,
1024→4096 is in HANDOFF addendum 25 §3, and its shape matters when quoting the top of the board: the
synthesis pass wins at every size (1.14-1.44x), the analysis pass wins at 1024, ties at 2048, and falls
below `ducc0` at 4096 (0.86x spin 2, 0.88x spin 0), because the Wigner-d march is cubic in `Nside` where
`ducc0` pays 6.3x per doubling. The `Nside=4096` cell is therefore **2.3x on the strength of its algebra
stages** (`coupling` 42058 → 3189 ms, `coupled_cell` 530 → 2 ms), not of its kernels. With both transform
groups counted, the large `Nside` cells already score 87-94 % of what they could if the coupling matrix,
the coupled cell and the decoupling were free (2.74x at `Nside=2048` spin 0, 3.17x at spin 2, 2.50x at
`Nside=4096` spin 0).

While float32 tables are live, `numpy.testing.assert_allclose` is held to a floor of
**2e-6 of the compared quantity** (its own `max|desired|`, not a fixed absolute), which is
about the size of the float32 table's representation error and is stated relative because a
decoupled `Cl` and a ring sum differ by twelve orders of magnitude. The default float64 run
installs nothing: every bar stays exactly as its test file writes it.

The current CPU suite contains 111 passing tests and 5 hardware-dependent skips,
including direct API/numerical comparisons against NaMaster. Dedicated NVIDIA
tests cover fused scalar parity and gradients; two-GPU tests cover scalar and
spin-2 analysis, synthesis, and Jacobi refinement.
