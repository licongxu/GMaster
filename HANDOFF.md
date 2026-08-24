# GMaster development handoff

## Objective and references

GMaster is a NaMaster-compatible Python/JAX implementation. Correctness against
`/home/lxu/scratch/agent_dev/auto_research_agent/NaMaster` is the first
constraint; GPU speed and `Nside=4096` memory are the next constraints. Treat the
NaMaster, S2FFT, and threej_cosmo source trees listed in `AGENTS.md` as read-only.
Read the relevant papers and derivations in `ref_paper/` and `ref_derivation/`
before changing MASTER or SHT mathematics.

Always activate:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
```

Do not stage the user-owned untracked `.claude/`, `AGENTS.md`, `CLAUDE.md`,
`ref_paper/`, or `ref_derivation/` paths.

## Completed milestones

Commit `0421dc6` adds the exact fused scalar HEALPix transform. The important
implementation is `gmaster/_sht_pallas.py`; routing and HEALPix FFT integration
are in `gmaster/utils.py`.

The fused NVIDIA path:

- runs one persistent Pallas/Triton program per harmonic order;
- evaluates normalized associated-Legendre recurrences in float64;
- uses exact base-2 rescaling every 16 degrees to survive forbidden regions;
- uses north/south parity to halve the latitude domain;
- fuses quadrature weights and HEALPix ring phases into the recurrence;
- provides custom VJPs that match the generic JAX gradients;
- caps latitude tiles at 1024 lanes for high-resolution register safety;
- uses the upstream S2FFT CUDA ring FFT when that optional extension was built
  with CUDA, otherwise the ordinary JAX ring FFT.

The uncommitted follow-up shards the fused recurrence by harmonic order. The
split balances `sum(L-m)` work rather than column count. It packs analysis
coefficients without rebuilding a full matrix and stages only the high-order
columns on the second GPU. Backend meanings are now:

- `jax`: automatic fused scalar path, two fused GPUs at `L>=2048`;
- `jax-single`: force one fused GPU;
- `jax-mgpu`: request two fused GPUs (generic two-GPU path for spins);
- `jax-generic`: force the generic S2FFT reference path.

Cross-device staging currently passes through host memory because direct JAX
device-to-device copies returned zero data on this machine. Traced/autodiff calls
therefore remain on one GPU; `_use_multi_gpu_pallas` detects JAX tracers.

## Verified performance and accuracy

Hardware: two RTX PRO 6000 Blackwell GPUs. Physical GPU 0 also hosted VLLM with
about 64.8 GiB allocated, so runs used `CUDA_VISIBLE_DEVICES=1,0`.

### Session update (ring-FFT rewrite and recurrence tables)

Two structural changes landed on top of the fused-kernel work:

1. **Batched chirp-Z ring FFT** (`gmaster/utils.py`, `_forward_ring_fft` /
   `_inverse_ring_fft`). Replaces s2fft's per-ring-size JAX path (which unrolls
   ~nside tiny polar FFT groups) with ONE uniform batched chirp-Z transform over
   all rings per direction. Chirp angles are reduced exactly with integer
   modular arithmetic (`q^2 mod 2*nside`) so no precision is lost to large
   arguments. Verified against `healpix_ffts.healpix_fft/healpix_ifft` to
   `<=5e-13` at Nside 512 on identical grids. Measured at Nside 512:
   forward 20.2 ms -> 4.8 ms, inverse 31.1 ms -> 10.7 ms.
2. **Normalized-recurrence coefficient tables** (`gmaster/_sht_pallas.py`,
   `_normalized_coefficients_numpy`). The stable normalized recurrence keeps
   its exact arithmetic, but `c1/c2` are now computed once as parallel vector
   ops outside the kernel and loaded per degree, removing two square roots and
   a division from the sequential chain. Analysis/synthesis inputs, outputs,
   and accumulators use transposed layouts for contiguous streaming.
   An unnormalized-recurrence variant (DLMF 14.10.7 with deferred K
   normalization) was implemented and REJECTED: it is numerically unstable in
   forward-l order because P_l^m is the minimal solution for m > 0; measured
   relative error 5.8e-5 versus the 3e-11 gate.

Measured end-to-end (warmed, this session):

| Case | Before | After | NaMaster | Accuracy |
|---|---:|---:|---:|---|
| Nside 512 analysis | 45.0 ms | 25.3 ms | 14.2 ms | max abs `1.94e-14`, rel `2.5e-12` |
| Nside 512 synthesis | 51.7 ms | 29.4 ms | 11.5 ms | max abs `3.24e-8`, rel `1.03e-11` |
| Nside 1024 analysis | ~200 ms | 177 ms | ~55 ms | max abs `2.06e-14` |
| Nside 4096 analysis, 1 GPU | 12.248 s | 10.843 s | ~1.76 s | `a00=sqrt(4pi)` to 1e-15 |
| Nside 4096 compile | 182 s | 51 s | - | - |

Multi-GPU staging still loses below L~2048 because cross-device copies pass
through host memory; unchanged this session. `num_warps=2` remains optimal;
block_size 512 is slightly better than 1024 for analysis (~20%).

Profiling note: `ncu` is installed but GPU performance-counter permission is
missing (`ERR_NVGPUCTRPERM`). Granting counters would enable stall-reason
analysis of the fused kernel.

### Previous baseline

Hardware: two RTX PRO 6000 Blackwell GPUs. Physical GPU 0 also hosted VLLM with
about 64.8 GiB allocated, so runs used `CUDA_VISIBLE_DEVICES=1,0`.

| Case | GMaster | Comparison | Accuracy |
|---|---:|---:|---:|
| Nside 512 analysis, fused 1 GPU | 0.0470 s | 17.2x faster than generic GPU; 2.27x slower than NaMaster | max abs `1.93e-14` |
| Nside 512 synthesis, fused 1 GPU | 0.0528 s | 14.2x faster than generic GPU; 3.88x slower than NaMaster | max abs `3.20e-8`, rel `1.02e-11` |
| Nside 1024 analysis, fused 1/2 GPU | 0.2532 / 0.2092 s | 1.21x multi-GPU gain | zero single/multi difference |
| Nside 1024 synthesis, fused 1/2 GPU | 0.4556 / 0.3168 s | 1.44x multi-GPU gain | single/multi max `4.55e-13` |
| Nside 4096 analysis, fused 1 GPU | 12.248 s | 18.2x faster than former 222.4 s GPU path; 6.95x slower than NaMaster | max abs `4.78e-13`, rel `1.35e-13` |
| Nside 4096 analysis, fused 2 GPU | 7.059 s | 31.5x faster than former GPU path; 4.03x slower than NaMaster | same all-coefficient error |

The full `Nside=4096`, `lmax=12287`, constant-map run compared all 75.5 million
packed coefficients. It returned
`a00=3.544907701811031+1.19e-16j`. NaMaster took 1.754-1.764 s.

Single-GPU device telemetry peaked at 36,652 MiB including a 236 MiB baseline,
or about 35.6 GiB incremental. The two-GPU run kept that primary peak and added
about 9.9 GiB beyond the secondary GPU's existing allocation. The old generic
path needed roughly 35 GiB plus 32 GiB. Principal fused array inventories are
10.87 GiB for analysis and 15.37 GiB for synthesis; XLA pools and FFT workspaces
explain the higher observed allocation.

## Verification state

The last checks all passed:

```text
CPU full suite:       111 passed, 5 skipped, 2 expected warnings
single-GPU SHT/utils: 14 passed, 3 skipped
two-GPU SHT tests:    3 passed, 11 deselected
```

Commands:

```bash
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python -m pytest -q
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m pytest -q tests/test_sht.py tests/test_utils.py
CUDA_VISIBLE_DEVICES=1,0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m pytest -q tests/test_sht.py -k multi_gpu
```

`benchmarks/benchmark_sht.py` compares automatic, fused single, generic, and
explicit multi-GPU backends. `benchmarks/benchmark_nside4096_memory.py` reports
both generic estimates and fused principal arrays.

## Important failed experiments and limits

- Float32/mixed-precision SHT was rejected: at Nside 512 it remained slower than
  NaMaster and produced about 4.4% relative error.
- A 4096-lane Pallas tile spilled and exceeded five minutes at Nside 4096.
  A 2048-lane tile also cliffs by `L=6144`. Keep the 1024 cap unless profiling
  proves a hardware-specific alternative safe. The isolated Nside 4096
  1024-lane analysis recurrence takes 11.95 s.
- Renormalizing every 32 degrees was slightly slower than every 16.
- The current S2FFT wheel has a non-CUDA extension. A separately built ABI3 CUDA
  extension made ring FFT compilation much shorter and runtime 1.1-1.5x faster,
  but it is not distributable from this repo and small-Nside CUFFT behavior needs
  upstream resolution.
- Exact scalar SHT is still slower than CPU DUCC. Do not claim a NaMaster SHT
  speedup. Existing genuine NaMaster speedups are in MASTER/covariance kernels
  (for example 29.7x selective scalar coupling and 208x flat covariance).

## Session 2 status: spin kernels built (guarded); the road to destroying NaMaster

### What works now
- Scalar fused SHT: CZT ring FFT + table-driven normalized recurrence.
  Nside 512: analysis 25.3 ms / synthesis 29.4 ms (NaMaster 14.2 / 11.5).
  Nside 4096 single-GPU warm analysis 10.84 s (NaMaster 1.76 s); compile 51 s.
- Exact TT coupling at lmax=3071: GMaster 0.72 s vs NaMaster 0.84-1.19 s
  (~parity). The large historical speedups apply to arbitrary-spin,
  pure-E/B, Toeplitz and flat-covariance kernels, not plain exact TT.
- End-to-end scalar pipelines currently LOSE ~2-2.6x at Nside 1024-2048:
  field construction (n_iter=3 => 4 analyses + 3 syntheses) dominates and
  each fused transform is still 1.8-3x slower than CPU DUCC.

### Spin-weighted kernels: built, correct at small L, blocked at scale
Fused spin synthesis/analysis + two-helicity E/B estimator implemented and
wired; matches NaMaster coupled cells to 2e-14 at L=48 but diverges beyond
L~96. Root causes, both nailed down experimentally:
1. Closed-form Wigner-d seeds cancel catastrophically for |m|->l (terms
   e^{+6000} cancelling below fp64 range): direct sums cannot work at any
   precision. Exponent-tracked evaluation fixes NaN/inf semantics but not
   relative precision of representable values.
2. s2fft's generic forward carries extra operator structure in the
   |m| < |spin| columns that a naive single-ladder adjoint does not
   reproduce (verified by delta-probing columns independently).
Resolution path (next session): port the Turok-Bucher sideways m-recursion
(with retrospective base-10 renormalization) as an in-kernel seed generator,
validated entry-by-entry against `turok_jax.compute_slice`. A literal numpy
port exists in session notes; its padded-index bookkeeping still diverges.
Alternative: Trapani-Navaza three-times recursion. Until then the dispatch
guard keeps every spin transform on the generic path (mainline stays exact).

### Raw-SHT gap analysis (why DUCC still wins per-transform)
- GPU FP64 peak is 1.89 TFLOP/s (FP32: 238). The fused recurrence runs at
  ~0.56 TFLOP/s effective: ~3.4x stall headroom exists but ncu lacks
  permission (ERR_NVGPUCTRPERM) so stall mix is unmeasured.
- Op accounting per lane-degree: ~3 FMA recurrence + 1 scale-mul + 4
  accumulation + ~2 reduction-amortized + RMW stores. Double-single fp32
  emulation of the recurrence products projects only ~1.4x (accumulation
  and reductions must stay fp64 for the 3e-12 gates).
- Highest-value levers, in order: (a) atomic_add stores to drop RMW reads;
  (b) two interleaved chains per program for latency hiding; (c) DS-fp32
  core; (d) counter permission for targeted stall fixes. Composite
  projection: 2-3x, reaching DUCC parity-to-better; pipeline wins grow
  from there since coupling is already ahead and multi-GPU sharding halves
  the residual deficit at L >= 2048.

### VALIDATED NEXT BIG ROCK: blocked parallel-prefix recurrence

The fused recurrence is LATENCY-bound on its L-step dependency chain
(measured: quartering the order count cuts runtime only ~30%), which caps
any constant-factor tuning. A blocked two-phase scheme was derived and
VALIDATED EXACTLY (numpy prototype, diff=0 vs sequential):

  Phase A (parallel over degree-blocks, grid = orders x blocks):
    each block runs the recurrence from CANONICAL seeds (1,0),(0,1),
    emitting data-weighted partial sums CA/CB and the block's composed
    2x2 transfer matrix M_j (per theta-lane).
  Phase B (serial scan over ~sqrt(L) blocks, vectorized over lanes):
    carry the physical per-lane seeds through M_j and combine partials:
    alm = sum_j u_j*CA_j + v_j*CB_j (+ seed-degree terms).

Sequential depth drops L -> G + L/G (minimized at G~sqrt(L)); per-degree
RMW stores and reduction trees disappear. Scratch requirement for the
per-lane matrices: orders x blocks x lanes x 4 doubles ~ 4.8 GB at
Nside 2048 (fits) but ~19 GB at 4096 => ship at <=2048 first; for 4096
either tile lanes through Phase B with recomputation or move M to fp32
pairs. Expected transform win: 3-6x (latency no longer floors runtime),
taking scalar transforms decisively past DUCC and making pipelines win
at every Nside. Synthesis uses the identical scheme transposed.

## Recommended next work

1. Commit the current ring-FFT rewrite and recurrence-table changes after
   reviewing `git diff --check` and the test results above (all suites green:
   CPU 111 passed, GPU 14 passed, two-GPU 3 passed).
2. Close the remaining ~1.8x SHT gap with double-single fp32 emulation of the
   recurrence hot loop. The GPU's FP64 peak is 1.89 TFLOP/s versus 238 TFLOP/s
   FP32, so even 10x-emulation overhead wins. Design: keep seeds, phase
   rotations, `_scaled_value` reconstruction, and lane reductions in fp64
   (cheap); convert only the recurrence state `(qm1, qm2)` and coefficient
   products to Dekker two-product/two-sum fp32 pairs (~40 fp32 ops per degree
   replacing 6 fp64). Validate every stage against the fp64 kernel; watch for
   Triton FMA contraction breaking error-free transforms (empirically test
   before trusting Dekker splits).
3. Grant GPU performance-counter permission (`ERR_NVGPUCTRPERM`) so `ncu` can
   attribute stalls in the fused kernel; measured throughput is ~30% of the
   FP64 peak with unknown stall mix.
4. Generalize the persistent recurrence to spin-weighted harmonics so spin-2
   fields do not fall back to generic S2FFT. Derive and test signs/conventions
   against NaMaster before benchmarking.
5. Execute full-band Nside-4096 synthesis and spin-2 memory telemetry; only
   scalar analysis has been run end to end at that size.
