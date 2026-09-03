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

### DS-fp32 experiment (this session): two-product exact, integration blocked

Dekker/Veltkamp double-single arithmetic was verified MACHINE-EXACT inside
a Pallas/Triton kernel on this GPU/compiler (max rel 7.9e-15 <= 2^-47), so
compilation does not break error-free transforms. However the full DS
analysis kernel integration produced wrong values (seed-degree entries off,
NaN rows for high orders) and a chunked-output refactor simultaneously
regressed the previously-green fp64 kernel (nside-64 alm err 5.0). Both were
REVERTED to the last green commit. Root cause candidates: Pallas out_shape
buffers are UNINITIALIZED (untouched cells leak pool garbage/NaN patterns),
and the load-modify-store accumulation pattern reads that memory; the
original design's input_output_aliases+jnp.zeros pairing apparently relies
on subtle XLA zeroing behavior that broke under refactoring.

Resume recipe: (1) restore DS kernel from session history on top of the
green base; (2) keep the original aliases/zeros pattern EXACTLY, changing
only the degree-step products to DS; (3) validate at nside 8/16 with
per-degree error profiles before any structural refactor; (4) an in-kernel
debug channel (store intermediates to a debug array) substitutes for the
missing GPU counter permissions and CPU interpreter.

### Session 3 status: block_size tuning confirmed; remaining levers identified

- Tile-parallel analysis (split latitude tiles across the grid, 3D partials
  buffer `(num_tiles, m_count, L)` + final `jnp.sum` over the tile axis,
  replacing the per-degree global RMW) was TRIED and REVERTED: it changed
  analysis by <1% (Nside 512: 27.3->26.5ms; Nside 1024: 193->189ms) but adds a
  3D partials buffer that is small at 512 (0.33GB/dir) but ~50GB/dir at Nside
  4096 — prohibitive. The per-degree RMW was NOT the bottleneck; the 3D buffer
  + reduction pass cost ~= the RMW it removed. Do NOT re-attempt unless the
  RMW is first proven to be the stall via ncu.
- `block_size` default lowered from `min(1024, 2*nside)` to `min(512, 2*nside)`
  in `_pallas_block_size`. Validated: GPU SHT+utils tests 14 passed / 3 skipped.
  Effect: synthesis ~1.5-1.6x faster at Nside >= 1024 (171ms vs 260ms at
  L=2047; 1354ms vs 2191ms at Nside 2048). Analysis is block-size-insensitive
  (degree-loop work dominates). Block 2048 cliffs hard on synthesis.
- `num_warps` sweep at Nside 512 confirmed 2 is optimal: analysis 29.3ms /
  synthesis 32.8ms at warps=2; 31.6/32.5 at 4; 40.7/33.4 at 8.
- Component decomposition at Nside 512, L=1023, block_size=512:
  - chirp-Z forward ring FFT ~3.4 ms, inverse ~4.9 ms
  - full map2alm (n_iter=0) ~15.4 ms  => latitudinal ~12 ms
  - full alm2map  (n_iter=0) ~15.7 ms  => latitudinal ~10.7 ms
- End-to-end benchmark (lmax = 3*Nside-1):
  - Nside 512:  NaMaster 15.0/11.9 ms, GMaster 27.3/30.5 ms (0.55x / 0.39x)
  - Nside 1024: NaMaster 54.5/51.9 ms, GMaster 193/203 ms (0.28x / 0.25x)
  Gap widens with Nside; the latitudinal recurrence kernel is the bottleneck.
- Analysis tile-parallelism: the analysis kernel currently keeps the latitude-
  tile loop INSIDE the program so the per-degree RMW into `out[m, ell]` is
  serialized. Splitting tiles across programs requires a partials buffer
  `out_partial[tile, m, ell]` + a final reduction pass, or it becomes a
  data race. Memory: ~9 x m_count x L x 16B ~ 0.23 GB at Nside 512, ~4 GB at
  Nside 2048 (fits), ~17 GB at 4096 (does not fit, needs tiling/recompute).
- 2-m-chain interleave: each program handles two adjacent m's to raise in-
  program ILP from 8 to 16. Register pressure is the risk: at block_size 512
  with num_warps=2 each thread already holds 8 lanes x 8 carried state doubles
  = 128 doubles = 1024 B; doubling the state would spill. Needs to pair with
  a smaller block_size (256) and num_warps=4/8. Untested.

## Recommended next work

1. Commit the `block_size=512` change (only validated SHT win this session;
   ring-FFT and recurrence-table changes were already committed).
2. Two-m-chain interleave for synthesis first (no in-loop reduction / RMW,
   pure register accumulation, single end store), paired with
   `block_size=256` and `num_warps=4` to control register pressure. Validate
   against current kernel output (~1e-11 rel) at Nside 8/16/64, then bench.
   Then mirror into analysis (RMW per degree step preserved, one program owns
   two m's so no cross-program race).
3. Analysis tile-parallelism via partials buffer + reduction pass (grid
   becomes `(m_count, num_tiles)`); ship for Nside <= 2048 where partials
   fit (~4 GB at 2048); add lane tiling or recompute for 4096.
4. Ring-FFT: the chirp-Z pads every ring to `next_pow2(L + 4*nside)` (4096
   at Nside 512) and does 3 batched FFTs of that length. Rings come in
   length groups (4k north/south, 4*nside equator, equator band is wide).
   A per-length-group batched `jnp.fft` (or cuFFT) of the actual ring length
   would cut the transform length ~2x for the equator and much more for
   polar rings. Validate against `_forward_ring_fft`/`_inverse_ring_fft`
   to <=5e-13 before replacing.
5. Grant GPU performance-counter permission (`ERR_NVGPUCTRPERM`) so `ncu`
   can attribute stalls in the fused kernel; measured throughput is ~30% of
   the FP64 peak with unknown stall mix.
6. Generalize the persistent recurrence to spin-weighted harmonics so spin-2
   fields do not fall back to generic S2FFT. Derive and test signs/conventions
   against NaMaster before benchmarking.
7. Execute full-band Nside-4096 synthesis and spin-2 memory telemetry; only
   scalar analysis has been run end to end at that size.

### Session 3 analysis: why the SHT gap to DUCC persists

- The chirp-Z ring FFT is **necessary**, not overkill: the chirp rate is
  per-ring (`two_nphi = 2 * actual_ring_length`), so a single plain FFT of the
  zero-padded `4*nside` width would compute DFTs at the wrong frequency
  spacing for the short polar rings. Per-length-group batched FFTs could save
  on polar rings but the equatorial band (2*nside-1 rings) still forces the
  full-size transform, so the realistic win is small (<1ms at Nside 512).
- "Quartering L cuts analysis time only ~30%" shows the analysis kernel is
  NOT recurrence-latency-bound. The dominant cost is the cross-lane
  `jnp.sum` reduction at every degree step (~2 reductions over block_size
  lanes per degree, ~70% of the FLOPs). This is a GPU-vs-AVX512 mismatch in
  reduction throughput, not something the 2-m interleave (which targets the
  ~30% latency slice) will meaningfully fix.
- Synthesis has NO in-loop reduction (pure register accumulation + 4 scalar
  broadcast loads per degree); at ~38% of NaMaster it is near the throughput
  limit for this FP64 access pattern.
- Practical conclusion: single-GPU scalar SHT will stay ~2-4x behind DUCC at
  Nside<=1024. The realistic paths to "massively surpass" NaMaster are
  (a) multi-GPU sharding at L>=2048 (halves SHT time), (b) the coupling/
  covariance kernels that already lead by 20-200x, and (c) end-to-end
  pipeline amortization. ncu stall attribution (item 5) is the prerequisite
  for any further single-kernel rewrite.
- Multi-GPU measured at Nside 1024 and 2048 (eager, jax-mgpu, 2 GPUs):
  - Nside 1024: analysis 170ms (0.33x NMT 55.9ms), synthesis 205ms (0.25x NMT
    50.3ms). Single-GPU was 190/203ms, so multi-GPU saves ~10% on analysis.
  - Nside 2048: analysis 1096ms (0.25x NMT 275ms), synthesis 1499ms (0.18x
    NMT 271ms). Single-GPU was 1407/1354ms, so multi-GPU helps analysis
    ~28% but HURTS synthesis (host-staging overhead dominates the shorter
    kernel).
  - Correctness confirmed at both sizes (max |dalm| ~1e-14, |dmap| ~1e-10).
  - The cross-device staging path goes through host memory (known JAX
    limitation on this machine), which caps the multi-GPU win. Fixing that
    would require device-to-device P2P or a restructured split that avoids
    staging the full high-order column block.

### Session 4 status: DFP32 (double-fp32) analysis kernel, wired behind `jax-dfp32`

- `gmaster/_sht_dfp32.py` (new): Pallas analysis (map->alm) kernel with a
  72-bit (3-limb f32) recurrence state and a 48-bit data path. All
  hot-loop arithmetic is f32 FFMA (PTX `fma.rn.f32`); only seed/scale setup
  and the per-theta block reduction into the f64 output use f64. A 72-bit
  state is REQUIRED: a 48-bit state drifts ~1e-7 over ~190 recurrence steps
  at m=0 while the 3-limb state stays ~1e-13. `num_warps=2` is the fast
  config (num_warps=8 is the slow one).
- The power-of-two scale `fac` is f64, applied only at the f64 reduction
  step (passed per-call into `add_coefficient`; the loop-carried copy is
  rescaled by the every-step renorm and must not be captured stale). An
  f32 `fac` silently underflows below 2^-149; `scale_exponent ~
  m*log2(sin theta) + log2|diagonal|` reaches ~-250 at m ~ L, but this is
  NOT the dominant DFP32 error source (the error barely changed when fac
  became f64) — it only removes a silent-zeroing hazard at extreme m.
- Wiring: `set_sht_calculator("jax-dfp32")` enables it (default stays
  "jax", i.e. DFP32 OFF). Analysis-only: forward map2alm uses the DFP32
  kernel; iterative-refinement residual synthesis stays fp64. Dispatch goes
  through a new helper `_fused_forward_sht` in utils.py. CAUTION: never
  name that helper `_forward_latitudinal` — a generic-S2FFT function with
  the same name already exists at module scope; shadowing it silently broke
  the `jax-generic` path (caught by the GPU test suite).
- Test: `tests/test_sht.py::test_dfp32_scalar_analysis_matches_namaster`
  (nside 64, lmax 159, n_iter 0/1, atol 3e-11 vs fused fp64 and NaMaster,
  plus synthesis-direction check). Passing; full GPU SHT suite 12 passed /
  3 skipped; CPU suite 111 passed / 6 skipped.

Precision envelope (vs fp64 Pallas, random unit-variance map):
- nside 64 kernel level: L=128/160 -> ~4e-14; L=192 -> 3.6e-12;
  L=224 -> 8.8e-8. Sharp cliff between L=192 and 224: recurrence f32
  roundoff accumulation exceeds the 3e-11 gate beyond L ~ 192-200.
- Nside 1024, lmax 1023 end-to-end: max|d| ~ 2.3e-3, max rel ~ 1.8 on
  |val|>1e-3 cells. DFP32 does NOT retain the 3e-11 gate at large L. The
  error is concentrated at high |m| (m > ~0.85 L) where the q recurrence
  cancels catastrophically — the same limit that keeps the spin kernels on
  the generic path. Low-m cells stay ~1e-14 (m=0 column vs NaMaster:
  5e-14 at Nside 512).

Performance (warmed, CUDA_VISIBLE_DEVICES=1, single GPU):
- SHT kernel only: Nside 512, L=512: fp64 4.4 ms -> DFP32 3.9 ms (1.15x).
- Full map2alm, n_iter=0: Nside 512/lmax 511: NaMaster 5.4 ms, fp64 9.2 ms,
  DFP32 8.3 ms. Nside 1024/lmax 1023: NaMaster 23.8 ms, fp64 42.2 ms,
  DFP32 36.9 ms. DFP32 is 1.15x end-to-end and still 0.65x of NaMaster.
- Verdict: DFP32 delivers ~1.15x e2e but does NOT "massively beat
  NaMaster". The analysis SHT is reduction-bound: the f64 cross-lane
  `jnp.sum` at every degree step is ~70% of the kernel cost, so cutting
  compute precision only helps the ~30% compute slice. NaMaster's flat alm
  storage is m-major (flat index `m(m+1)/2 + ell`; l-major is wrong).
- Next levers (in order): (1) shrink the per-degree reduction — pack the
  re/im sums into one complex reduction or restructure the theta sum;
  (2) multi-GPU sharding at L>=2048; (3) pipeline amortization. The
  blocked parallel-prefix recurrence (Session 3 "big rock") remains
  blocked by the fp64 conditioning wall at m~L/2 for synthesis; analysis
  does not block that way but its theta sum prevents the same trick.
  **SUPERSEDED by Session 5 below — the reduction story is wrong.**

## Session 5 (2026-09-01): the kernel is SPILL-bound and fp64 SIMT is the wall

Scripts (scratch, untracked, `.qwen/tmp/`): `diag_analysis.py` (block/warp
sweep), `attr_analysis.py` (cost attribution by variant), `alias_probe.py`
(are aliased rings needed?), `fma_ceiling.py` (fp64 vector FMA ceiling),
`gemm_sht_probe.py` (precomputed-matrix + split-fp32 GEMM).
Timing protocol: warm once, then `out = fn(); jax.block_until_ready(out)` —
`block_until_ready(None)` is a no-op and produced fake 0.09 ms numbers for most
of this session. GPU 1 idles at 180 MHz and needs ~0.5 s to reach 2.87 GHz, so
always time an in-process reference config in the same run and quote ratios.

Reference point: fused analysis kernel, Nside 512 / L=1536, 25.3-26.5 ms at
~0.6 TFLOP/s effective.

Cost attribution (one change per variant, same grid):
| variant | change | delta |
|---|---|---|
| arith | +31% fp64 ALU in the recurrence | +16% |
| nostore | drop the per-degree RMW store | -1% |
| nonorm | drop the every-16-degree renormalisation | -9% |
| extra | +4 loop-carried fp64 FMAs per lane-degree (+31% ALU) | **+161%** |

`extra` is the tell: 31% more arithmetic costing 161% more time is register
pressure crossing a spill threshold, not throughput. It confirms the block-size
sweep was a dead end (25-30 ms plateau over block 256-1024, warps 2-4) and that
the old "reduction-bound, ~70% of cost" belief is wrong: cutting cross-lane
reductions 32x gave only 2.1x, and halving them again (block 512->1024) gave
3.4%. RMW stores and coefficient-table loads are free. It also re-explains two
older results: DFP32's 1.15x (its 3-limb state spills, so extra limbs cost what
the cheaper ALU saves) and the plain-fp32 22x (halved register footprint, not
just cheaper ALU).

Measured ceilings on GPU 1 (RTX PRO 6000 Blackwell; sustained 2.87 GHz / 207 W
under load — the clocks DO ramp, idle 180 MHz is not the operating point):
| probe | TFLOP/s |
|---|---|
| fp64 SIMT vector FMA, 8 independent register-resident chains (Pallas) | 0.85 |
| fp64 matmul 4096^3 (cublas) | 1.87 |
| fp32 matmul 4096^3 (tensor cores) | **213** |

The analysis kernel therefore runs at ~70% of the fp64 **SIMT** ceiling:
register-budget redesign is worth maybe 1.3-1.5x, never "massive". fp32 tensor
cores are 250x the fp64 SIMT rate and 114x the fp64 matmul rate, so every
"massive" idea must terminate in a batched GEMM on tensor cores with fp64
accuracy reconstructed by splitting (Ozaki / 3xTF32 style), not by limb-wise
SIMT arithmetic like DFP32.

Two exactness results that close off shortcuts:
1. Aliased (ring, order) pairs with `nphi <= m` cannot be zeroed. They are
   18.4% of samples at nside 64 / L=192; zeroing them gives 2.1e-4 vs NaMaster
   while the all-rings code matches at 1.45e-14. GMaster's s2fft weight
   convention needs every ring for every m — unlike ducc0, which restricts the
   ring set per order. Never "optimise" by skipping aliased rings.
2. Exact masked-ring skipping is only valid for `n_iter=0`. With Richardson
   refinement it changes the fixed point from `(A S)^-1 A m` to
   `(A M S M)^-1 A m`. Needs an `n_iter==0` guard plus a decision on the
   iterative path.

Measurement gaps left open: no register/spill counts (`ERR_NVGPUCTRPERM` blocks
ncu; JAX Pallas writes no Triton cache, so `TRITON_CACHE_DIR` produced
nothing). Next best is a `plt.CompilerParams(maxnreg=..., num_stages=...)`
sweep — if runtime is flat in `maxnreg` past some value the spill story is
confirmed and the fix is fewer live loop-carried values, not more registers.

### Session 5b: the precision ladder, and four tensor-core routes already killed

Full ladder measured on GPU 1 (4096³ in JAX, sustained 2.87 GHz; scripts
`.qwen/tmp/fp64_emu.py`, `.qwen/tmp/gemm_sht_probe.py`):

| path | rate | note |
|---|---|---|
| fp64 SIMT vector FMA | 0.85 TFLOP/s | where the fused kernel lives (~70% of this) |
| fp64 matmul (cublas) | 1.87 TFLOP/s | |
| fp64 matmul, `CUBLAS_EMULATE_DOUBLE_PRECISION=1` | 2.11 TFLOP/s | only 1.13x, `eager` and `performant` identical |
| fp32 matmul | 213 TFLOP/s | 250x the fp64 SIMT rate |
| bf16 / fp16 -> fp32 matmul | 327 / 311 TFLOP/s | |
| **int8 -> int32 matmul** | **418 TOP/s** | `preferred_element_type=int32`, exact accumulation |

The environment has cuBLAS 13.4 (`nvidia/cu13/lib/libcublasLt.so.13`), which
does contain `COMPUTE_64F_EMULATED_FIXEDPOINT` and the `CUBLAS_EMULATE_*`
knobs — but turning them on buys 13%. Do not plan around NVIDIA's own fp64
emulation; it is not selected on this card.

Four ways to reach tensor-core speed at fp64 accuracy, tested, and why each
fails:
1. **Ozaki/split-fp32 with fp32 accumulation is worthless.** `(Ahi+Alo)(Xhi+Xlo)`
   with three GEMMs returned the *identical* error to plain fp32 (2.7e-4 relative
   on a Legendre-shaped transform with median cancellation sum|t|/|sum| = 51).
   The fp32 accumulator is the limiter, not the operand representation.
2. **XLA will not widen the accumulator.** `jnp.dot`/`einsum` with f32 inputs and
   `preferred_element_type=float64` raises
   `INTERNAL: Unexpected GEMM dtype: f32 f32 f64`. K-chunking to bound the fp32
   accumulate error lands around 1e-7 relative — four orders off the 3e-11 gate.
3. **cuBLAS fp64 emulation: no-op** (above).
4. **Naive fp32/bf16 kernels: unshippable** (0.17% error + recurrence overflow,
   Session 4).

The one route that survives on paper is **int8 fixed-point (Ozaki scheme III)
with int32 accumulation** — int32 accumulate is exact, which is precisely the
property fp32 accumulation lacked. ~7 bf16 limbs or ~9 int8 limbs per fp64
operand, 20-30 limb-products: 14-21 TFLOP/s theoretical, 3-6 TFLOP/s realised
would be 4-7x over fp64 SIMT, and unlike DFP32 it runs on tensor cores instead
of SIMT (so no spill wall). Requires a Triton `tl.dot(int8, int8,
out_dtype=int32)` Pallas kernel plus per-block exponent recombination. Untested.

### Session 5c: kill the recurrence, then batch the transforms

The other half of the win is algorithmic and independent of precision. The
per-degree Legendre recurrence is *why* the SHT cannot be a GEMM today. Storing
`Pbar_l^m(mu_j)` as an explicit matrix makes theta a batched matmul and removes
the recurrence (roughly 2/3 of the kernel's work): per-field transform work
drops from ~20 GFLOP to ~4.8 GFLOP at Nside 512 / L=1536.

Caveats that decide whether it pays:
- As a matrix-*vector* product it is bandwidth-bound: the full matrix is
  ~9.7 GB fp32 (19 GB fp64) at Nside 512 / L=1536, ~77 GB fp32 at Nside 1024.
  Chunk over m-bands (e.g. 128 orders -> ~6.4 GB at nside 1024).
- CORRECTION (verified at `gmaster/utils.py:1135-1152`): Richardson iterations
  are **strictly serial** — `alm -= A(S(alm) - maps)` each step, so the
  4 analyses + 3 syntheses of `n_iter=3` CANNOT be batched into one GEMM.
  The earlier "batch over Richardson iterations" note here was wrong. Available
  RHS width B comes only from genuinely independent transforms: independent
  fields/masks in a workspace, and B=2 for free on the spin path (Q and U go
  through one call). A scalar `map2alm` is B=1, which is the case that matters.
  Amortise the one-time matrix build over the whole workspace, not per call.
- SHTns and s2fft both expose exactly this "matrix vs on-the-fly recurrence"
  switch, and s2fft's `precompute` mode is known to lose accuracy. Gate any
  matrix-based path against NaMaster, not against the current kernel.

### Session 5d: what the new references actually give us

Two research subagents read `ref_paper/cunuSHT.pdf`,
`ref_paper/fastpartialsht.pdf` and the `ref_paper/cuHPX/` tree. Their claims are
evidence, not gospel — the file:line anchors below are worth re-checking before
acting. Headline: **neither new paper removes the Legendre recurrence, so neither
is a massive win for GMaster's full-sky ring SHT. cuHPX already implements the
Session 5c idea and shows exactly where it hurts.**

**cunuSHT (arXiv:2406.14542).** DFS factorisation `Y = N F D S` on the doubled
sphere; needs a ~(2L)^2 grid (2Nr+2 samples in theta) plus a 1.25 nuFFT
upsampling factor. The `S` factor is still an iso-latitude rSHT with on-the-fly
`P_lm(theta)` recurrence (it literally calls SHTns), still O(lmax^3). The paper's
own profile has `S` at only ~20% of runtime because the nuFFT/FFT/doubling
operators dominate on GPU — and DUCC's *CPU* `S` is >2x faster than the GPU `S`.
Their speed-ups (up to 5x single / 3x double precision, GPU overtaking CPU at
lmax~300-400, up to lmax~9000 on an A100 80GB) are for *arbitrary-point* nuSHT,
not for a ring SHT. Verdict: adopting the DFS structure for full-sky HEALPix
would ADD operations. Skip unless we ever need non-iso-latitude pixelisations.

**fastpartialsht (arXiv:2603.17166).** Partial-sky Fourier-sphere method:
band-limited 2D Fourier series on the doubled sphere, Dirichlet-kernel
reconstruction from theta samples, patch -> cap with `theta_max = theta*(1+eps_apo)`,
then ring transforms with new ring locations and weights ("minor tweaks to
ducc0.sht") — again the recurrence stays. Ring count drops from lmax+1 to
Lmax+1 with `Lmax = lmax (theta*/pi)(1+eps_apo)^2`. Measured 3.5-9.0x (up to
39-43x) on 1500 / 980 / 57 deg^2 footprints; **full sky degenerates to the
standard full-sky Fourier-sphere method and loses.** Error is approximate,
controlled by tolerance eps with `alpha = |ln eps|/pi`, `Delta l = 2 alpha`,
`eps_apo = sqrt(2|ln eps| / (theta* lmax))`. Useful corollary (derived from their
eq. 2.15, worth re-deriving before use): cost grows only like `sqrt(ln 1/eps)`,
so pushing eps from 1e-7 to 1e-11 costs just 1.25x in `eps_apo` — an approximate
method CAN be driven to our 3e-11-ish budget cheaply. **The transferable piece is
polar optimisation** (truncating the m-range per ring to `m <= lmax sin theta +
margin`), which they measure at 1.5-3x op-count on native HEALPix rings without
changing the transform structure. Tension to resolve first: Session 5 finding #1
(aliased rings) shows GMaster's weight convention currently needs every ring for
every m, so per-ring m-truncation is exactly the kind of shortcut that already
failed once. Check it against the ring-weight convention before implementing.

**cuHPX (JAX + custom CUDA HEALPix SHT).** This is the closest existing
implementation of Session 5c and the best source of engineering detail:
- Precomputes the Legendre matrix `pct` (mmax x lmax x ntheta) and does theta as
  `einsum` on cuBLAS — recurrence gone from the hot loop. Memory is the wall, so
  it streams m-bands from pinned host memory with a double-buffered,
  event-synchronized pipeline (`cuhpx/hpx_sht.py:430-478`, `nchunk=12` at :506,
  selected when `pct` is not on device at :603; device-resident only when
  `min(nside, lmax, mmax) <= 2**9`, guard at :611). Read this before writing our
  own chunker.
- Batching is first-class: `healpix_rfft_batch` (`src/harmonic_transform/
  hpx_fft.cpp:21-93`) and `einsum("...kmn,mlk->...lmn")` where the trailing dim is
  the flattened batch — confirms the "batch fields x iterations so the matrix is
  read once" arithmetic-intensity argument.
- Bluestein on ragged HEALPix rings uses ONE uniform plan
  (`cufftPlan1d(padding=8*nside, Z2Z, ntheta*n)`) with the chirp kernel FFT
  cached and reused (`initializeYpad`, `src/harmonic_transform/hpx_fft.h:99-105`),
  so a transform is 2 cuFFT calls + pre/post chirp kernels. GMaster pads to
  `next_pow2(L + 4*nside)`; bucketing the pad per ring group instead of one
  global pad is the visible improvement here.
- Precision is dtype plumbing only (f32/f64) — no TF32/fp16/bf16 anywhere, and
  README/tests carry no measured numbers (benchmarks read nside from stdin), so
  there is no published accuracy/speed envelope to lean on.
- CUDA-graph discipline worth copying into JAX: no allocation inside the
  captured region, workspace `zero_()` in place, run on the caller's stream
  (`hpx_sht.py:636-637`), pre-cast `pct` to both dtypes to keep `.to()` out of
  the capture (`hpx_sht.py:625-631`). Note the *remap* path is NOT graph-safe
  (raw `cudaMalloc`/`cudaFree` per call, `src/data_remapping/
  hpx_remapping_cuda.cu:590-610`) — do not trust the README's blanket "CUDA
  graphs" claim.
- Ring weights come from the standard healpy FITS file and are made
  north/south symmetric, `w = 4*pi/npix * (1 + w[northring-1])`
  (`cuhpx/sht_tools.py:33-58`) — useful cross-check for GMaster's own ring-weight
  convention.

### Session 5e: the numbers from the completed deep reads (these change the verdicts)

The two research agents finished full reads (cunuSHT pp.1-14, fastpartialsht
pp.1-10, cuHPX source audit). Additions that matter, in order of impact:

- **cunuSHT cannot pass our gate at all.** With the nuFFT up-sampling factor 1.25
  used in every benchmark, its own double-precision accuracy *floor* is **1e-10**
  (footnote 6, p.5). Our gate is `atol=3e-11`. Single-precision nuFFT only reaches
  `eps_eff > 1e-4`, and getting to 1e-6 requires the double nuFFT (~2x slower).
  Its `S` factor is SHTns on a Clenshaw-Curtis grid with an on-the-fly `P_lm`
  recurrence, `O(lmax^3)`, and the paper states plainly (footnote 1, p.1) that the
  `O(lmax^2 log^2 lmax)` matrix-based rSHT "requires high memory usage and
  significant pre-computations. A general purpose implementation that is
  competitive with the `O(lmax^3)` counterpart has yet to be published." That is
  exactly the project Session 5c proposes — nobody has published a good one.
  Also: analysis is *type 1*, which they expect to be worse on GPU because many
  threads write the same locations concurrently; nuFFT **plan creation is the most
  time-consuming operation**; and all published speed-ups (1-5x single / 1-3x
  double, GPU overtaking at lmax ~300-400) are against a 32-core CPU on the
  *non-uniform* problem, never against a healpix ring SHT. Doubled-sphere grid is
  ~(2L)^2 complex128 ~ 64 L^2 bytes (~1 GB at L=4000), x1.25^2 for the nuFFT grid.
- **fastpartialsht's speed-ups are CPU-vs-CPU.** Table I benchmarks
  `ducc0.sht 0.40.0` on 10 realizations x 4 cores; there is **no GPU number in the
  paper**. Synthesis general/adjoint: SPT-3G main 1500 deg^2 (theta*=35 deg,
  lmax=4000, spin 0) -> 3.5 / 3.3; 980 deg^2 (theta*=20 deg) -> 7.7 / 7.12 (9.0 /
  8.8 at lmax=6000); EDFS 57 deg^2 (theta*=7 deg) -> 23 / 21 and 39 / 43 at
  lmax=10000. `lmax+1` rings x `2 lmax+1` points is stated as necessary and
  sufficient. Worked example of the band-limit shrink: theta*=35 deg, lmax=4000,
  eps=1e-7 -> `Lmax ~ 960` (my arithmetic, not in the paper). Gain scales as
  `1/f_sky` and the authors note the measured gain is *below* their polar-optimisation
  forecast "because of our optimistic treatment of polar optimization". They also
  exclude the one-off per-pixel coordinate reassignment from timings (footnote 6),
  and state the cost is "vastly dominated by the Legendre transform ... plausibly
  close to optimal" - i.e. ring count *is* the cost, which is the same wall we hit.
- **cuHPX is a structure reference, not an accuracy reference.** Its own
  empirically-fixed tolerances (`tests/conftest.py:85-97`) are fp64 sht `(1e-8,
  1e-8)`, isht `(1e-5, 1e-5)`, Bluestein `(1e-5, 1e-5)` / `(1e-4, 1e-4)` annotated
  *"algorithm-limited (~1e-6 for SHT, ~1e-4 for iSHT) not precision-limited"*,
  roundtrip `rtol 0.01 / atol 0.05`. So cuHPX is 3 orders of magnitude off our
  3e-11 gate and its chirp-Z path is *algorithmically* limited — worth remembering
  that GMaster's own chirp-Z ring FFT matches NaMaster at 1.45e-14, i.e. ours is
  materially better than the reference implementation in this tree. Zero hits
  repo-wide for tf32/bfloat/float16/reduced_precision and no `--use_fast_math`.
  Confirmed pipeline: phi = Bluestein chirp-Z for every ring padded to one uniform
  `8*nside` (`hpx_fft.cpp:39`, `cufftPlan1d(..., CUFFT_Z2Z, ntheta*n)` at
  `hpx_fft.h:82`) with the chirp FFT cached (`hpx_fft.h:99-105`); theta = one
  `torch.einsum("...kmn,mlk->...lmn", x, pct_weights)` (`hpx_sht.py:510`) with
  quadrature weights folded into the matrix at build time (`hpx_sht.py:204,628`);
  matrix built **on the host** in fp64 by `legpoly` (`sht_tools.py:191-256`),
  ~4.8 GB at lmax=mmax=3*256 — which is *why* it needs the pinned-memory chunker
  (`hpx_sht.py:430-478`, `nchunk=12` at :506, gate at :611). GEMM op count equals
  the recurrence's (~2*lmax*mmax*ntheta, with ~2x wasted on the `l<m` triangle);
  the win is cuBLAS + batch amortisation, not fewer FLOPs.
  No masks, no deprojection, no mode-coupling, no pseudo-C_ell anywhere in it.

Net effect on the plan: the paper-derived options are now *excluded by their own
accuracy numbers*, which leaves Session 5c (precomputed matrix + batched GEMM,
exact fp64, chunked a la cuHPX) and Session 5b (int8 fixed-point Ozaki) as the only
live routes, and makes the "nobody has published a competitive matrix-based rSHT"
footnote the risk statement for 5c.

### Recommended order of attack (for the next session)

1. Batched precomputed-matrix theta transform (Session 5c) with m-band chunking
   modelled on cuHPX, RHS batched over Richardson iterations x E/B. Exact fp64
   first — target 3-5x e2e, gated at 3e-11 vs NaMaster.
2. int8 fixed-point Ozaki GEMM (Session 5b) as a drop-in accelerator for the
   matrix in step 1 once it exists. Triton `tl.dot` accepts int8 with an int32
   accumulator (verified against the installed Triton 3.6 docstring), so it is
   expressible in Pallas. Potential further 4-7x, high risk.
3. Cheap, orthogonal wins: Richardson is 4 analyses + 3 syntheses per field —
   check whether Chebyshev/Krylov acceleration cuts the iteration count at the
   same fixed point, and confirm NaMaster's own `n_iter` defaults for `master()`
   vs `NmtField`/`purify` (still unverified).
4. Do NOT re-litigate: SIMT register tuning, reduction packing, aliased-ring
   skipping, cuBLAS fp64 emulation, DFP32 limb widening, DFS/cunuSHT for full sky.

## Session 5f (2026-09-01, later): the matrix transform measured for real

`.qwen/tmp/matrix_theta_bench.py`, Nside 512 / L=1536, folded theta grid
`nt = 2*nside = 1024`, rows `l>=m` = 1.18 M, RHS carried as real [Re|Im]
columns so cuBLAS reads A once instead of running separate Re and Im GEMMs.
Baseline to beat: 25.3 ms (fused Pallas analysis kernel, one analysis call).

| layout | B=1 | B=2 | B=4 | A stream |
|---|---|---|---|---|
| fp64, chunked `l>=m` (9.9 GB) | 47.7 ms (**0.53x**) | 23.9 ms (1.06x) | 11.9 ms (2.12x) | **207 GB/s** |
| fp32, chunked `l>=m` (4.9 GB) | 4.45 ms (**5.7x**) | 2.36 ms (10.7x) | 1.15 ms (**22x**) | **1.11 TB/s** |
| fp64, naive full-`l` padding, cuHPX layout (19.3 GB) | 82.6 ms (0.31x) | 41.4 ms | 20.6 ms | 234 GB/s |

Three conclusions, all load-bearing:

1. **The naive cuHPX layout costs exactly the predicted 2x** (19.3 GB vs 9.9 GB,
   82.6 vs 47.7 ms). m-banded `l>=m` chunking is worth 2x on its own and we
   should do it regardless of precision.
2. **fp64 storage through cuBLAS is a dead end.** Runtime is flat at 47.7 ms for
   B=1..4 — purely A-read-bound — but cuBLAS's fp64 narrow-N kernels only stream
   at 207 GB/s, 11% of the card's ~1.8 TB/s. So an exact-fp64 *cuBLAS* matrix
   transform is slower than the recurrence kernel we already have.
3. **fp32 storage streams at 1.11 TB/s and would be 5.7x (B=1) / 22x (B=4).**
   The speed is there; only accuracy is missing (fp32 storage + fp32 accumulate
   measured 2.7e-4 relative in Session 5b, and XLA will not give fp32-in /
   fp64-accumulate).

So the whole problem reduces to: **contract a streamed matrix in fp64 at ~1 TB/s.**
That is a memory-system question, not an ALU question — 1.18 M rows x 1024 fp64
contractions against 1-2 RHS columns is 2.4 GFLOP of work (2.8 ms at the 0.85
TFLOP/s fp64 SIMT rate) inside a 9.9 GB read (~9 ms at 1.1 TB/s). A custom Pallas
streaming kernel that keeps fp64 accumulation and reads A coalesced should land
near max(read, compute) ~ 9-10 ms = **~2.6x, exact fp64**, and if the matrix is
stored as fp32 limbs (2 limbs = 9.9 GB, same bytes as one fp64) the same kernel
could recover ~53 bits with fp64 accumulate at no extra bandwidth — that is the
variant worth building.

`fp64_stream_probe.py` settles what the card can and cannot do with a 9.7 GB
fp64 matrix (`ROWS = 1536*1537/2 = 1180416`, `NT = 1024`):

| probe | time | A-stream rate |
|---|---|---|
| pure fp64 read (row-sum) | 6.03 ms | **1603 GB/s** |
| cuBLAS fp64 dense, K=1024, N=2 | 41.5 ms | 233 GB/s (0.12 TFLOP/s) |
| cuBLAS fp64 dense, N=4 / N=16 | 41.4 / 41.2 ms | 233 / 235 GB/s |
| cuBLAS fp64 dense, N=64 / N=256 | 82.4 / 327 ms | 117 / 30 GB/s (1.88 TFLOP/s, compute-bound) |
| cuBLAS fp64 batched N=2, b=16 / 32 / 128 | 41.5 / 41.6 / 41.5 ms | ~233 GB/s |

So the 233 GB/s is **not** a batching artifact and **not** a memory limit: the
card reads fp64 at 1.6 TB/s, but cuBLAS's fp64 narrow-N kernels are
ALU/algorithm-capped at ~0.12 TFLOP/s and flat from N=2 to N=16 and across every
batch size tried. Conclusion: the precomputed-matrix theta transform must be a
**hand-written Pallas streaming kernel**, and its budget is
`max(6.0 ms read, 2.4-4.8 GFLOP at 0.85 TFLOP/s)` = **~6-9 ms vs 25.3 ms =
2.8-4.2x, in exact fp64**, before any precision tricks. Lowering the matrix to
~40 mantissa bits (fp32 + bf16 correction = 6.1 GB instead of 9.7 GB) drops the
read floor to 3.8 ms; that mantissa dial is the cheapest remaining knob and it is
directly gateable against NaMaster.

## Session 5g (2026-09-01, later): the contraction does not need cuBLAS — 4.2x, exact fp64

`.qwen/tmp/fused_contract_probe.py`. The trick is to ask XLA for a fused
multiply-reduce instead of a GEMM, i.e. `jnp.sum(A * X[:, None, :], axis=-1)` /
`jnp.einsum("bkj,bj->bk", A, X)` rather than `jnp.matmul`. Measured on the same
9.7 GB fp64 matrix (batched view `(32, 36888, 1024)`, `nt=1024`):

| form | time | A-stream | vs 25.3 ms kernel |
|---|---|---|---|
| shared RHS across rows `sum(A*x, -1)` | 19.43 ms | 498 GB/s | 1.30x |
| **per-m RHS `sum(A*X[:,None,:], -1)`** | **6.02 ms** | **1605 GB/s** | **4.20x** |
| per-m RHS `einsum("bkj,bj->bk")` | 6.03 ms | 1604 GB/s | 4.20x |
| per-m RHS x 2 cols (a complex field's Re+Im) | 7.02 ms | 1378 GB/s | **3.6x for the whole field** |

Numerics: the batched contraction agrees with `einsum` to 0.0 (identical lowering)
and with cuBLAS `A @ x` to 3.4e-16 — fp64 SIMT accumulation, no TF32, no widened
accumulator needed. **No custom kernel was required**: XLA lowers this shape to a
streaming fused kernel and hits 1605 GB/s, beating cuBLAS's 233 GB/s by 6.9x on
identical data.

Why the right shape wins: the SHT contraction is `alm[m,l] = sum_j A[m,l,j] X[m,j]`
— the RHS varies with m, and that is exactly the `bkj,bj->bk` form XLA tiles well.
The `shared-RHS` variant is 3.2x slower, so do not "optimise" by hoisting the RHS.

**This is the shippable route.** Replace `_analysis_kernel`'s theta stage with a
precomputed `A[m, l>=m, j]` contracted via broadcast-reduce: ~4.2x on the analysis
theta transform (3.6x for a full complex field including both Re and Im) in exact
fp64, gated only by the existing NaMaster tests. What is still to be done:

1. Build `A` from the *same* recurrence as `_sht_pallas._analysis_kernel` so the
   convention (normalization, `weights`, `phase`, the north/south parity fold at
   `_sht_pallas.py:129-131`) matches by construction; fold `w_j * phase_j` into the
   rows and feed `X_plus = f_N + f_S` / `X_minus = f_N - f_S` as two contractions
   over the same rows (no extra bytes). One-time build is ~14 GFLOP.
2. Keep the m-banded `l>=m` layout — it is worth a further 2x versus cuHPX-style
   full-`l` padding and it is already assumed by the 9.7 GB figure above.
3. Memory: 9.7 GB at Nside 512/L=1536 is fine, but `L^2/2 * 2*nside * 8 B` is
   ~77 GB at Nside 1024 — m-band chunking (device-resident per band, built or
   streamed once per workspace) is mandatory there, not optional.
4. Then re-measure e2e `map2alm` / full MASTER pipeline vs NaMaster; the SHT theta
   stage is only part of the pipeline and its share has never been measured.

## Session 5h (2026-09-02): matrix convention validated; the *layout* is the whole game

**The convention is nailed.** `.qwen/tmp/matrix_convention_probe.py` builds
`values[m, ell, j]` from the *same* normalized recurrence as
`_sht_pallas._analysis_kernel` (same seed `log2|diag| + m·log2(sin)`, same
exp2-rescale-every-16 schedule, same `weight · exp(i m phase)` applied to north
and south rings, same `parity = 1 - 2·((ell+m)&1)` fold, equator ring counted
once) and reproduces the fused Pallas kernel bit-for-bit to fp64 rounding:

| config | max abs diff vs kernel | relative |
|---|---|---|
| nside 64, L=192 | 2.665e-15 | 5.1e-16 |
| nside 256, L=768 | 7.139e-17 | 5.2e-15 |

Zero leakage into `ell < m`. This is ~4 orders of magnitude inside the
`atol=3e-11` gate in `tests/test_sht.py`, and even inside the `atol=3e-13`
main-path gate. **Numerics are no longer a risk for this route** — only speed,
memory, and build cost are.

Three exactness simplifications that made the build tractable (worth keeping in
mind for any future variant):
* the exp2 rescale every 16 degrees is **value-preserving**, so a matrix built
  without tracking the exponent at all is still the same number;
* the `ell = m+1` term equals the general recurrence with `P_{m-1} = 0`, so one
  `where(ell == m, seed, general)` covers the whole triangle;
* the parity fold distributes over the `j`-reduction
  (`alm = A_north + parity · A_south`), so it can be applied *after* the
  contraction instead of inside it.

**Negative result — do NOT use a padded `(m, ell, j)` cube.**
`.qwen/tmp/matrix_vs_kernel.py`, nside 256/L=768, four RHS columns, one jit'd
`sum(v[:,:,:,None] * r[:,None,:,:], axis=2)`:

| | time | stream |
|---|---|---|
| fused kernel | 4.08 ms | — |
| padded 4-col cube contraction | 6.11 ms | **395 GB/s** |
| one-time build | 24.27 ms | — |

**0.67x — slower than the kernel.** The padded cube carries 2x dead bytes and
XLA only streams it at 395 GB/s, not the 1605 GB/s measured in Session 5g. At
nside 512/L=1536 the same code OOMs: the build materialises an 18 GB cube and
the autotuner profiles 5 configs concurrently.

**Positive result — the m-banded m-block slab layout works.**
`.qwen/tmp/band_probe.py` stores only `ell >= m`, split into m-blocks of `mb`,
each slab `(n_ells, mb, north)` with `j` **contiguous** (the natural `lax.scan`
output order — so the build never needs a transpose copy), and reduces each slab
over its last axis with a per-m RHS, i.e. the exact `bkj,bj->bk` form that hit
1605 GB/s in Session 5g. Timed honestly: the matrix path takes raw `ftm` and
does the `weight · exp(i m phase)` rotation and the north/south fold *inside*
the timed region, exactly as the kernel does.

nside 256, L=768, `mb=64`, band 1.31 GB (padded would be 2.42 GB):

| form | time | vs kernel | stream |
|---|---|---|---|
| fused kernel | 4.09 ms | 1.00x | — |
| 4-col single pass (fold after reduce) | 2.97 ms | **1.38x** | 441 GB/s |
| 2-col — the parity-split production budget | 1.65 ms | **2.47x** | 792 GB/s |
| accuracy vs kernel | 7.1e-17 (rel 5.2e-15) | | |

Why 2-col is the production form and not 4-col: parity of `ell + m` equals
parity of `ell - m`, so if the band is *stored* split by parity of `k = ell - m`
(the scan naturally emits `k` in order), each half is contracted with **one**
complex RHS = 2 real columns, and the two halves together still read the band
exactly once. That halves the flops per matrix element for free. The 4-col
single pass above is therefore a pessimistic lower bound, not the design.

### What the binding constraint now is

1. **Stream rate is only half of peak** (792 GB/s vs 1605 GB/s). Reduction
   length `north = 2·nside` is short (512 at nside 256) and each slab's output is
   small, so occupancy/tail effects dominate at this size. Bigger nside should
   stream better; if not, the fix is XLA-independent (choose `mb`, or hand the
   contraction to a Pallas kernel that keeps the 1605 GB/s schedule).
2. **The build now dominates and must be cached.** Warm rebuild is 36 ms against
   a 4 ms theta stage. That is ~7x off the build's own flop budget (~1.6 GFLOP
   → ~2 ms), because the `lax.scan` carries 4 arrays and rewrites `qm2/qm1/
   exponent/factor` every step. Two mitigations, both cheap and both required:
   * the matrix is a **pure function of `(nside, L)`** — cache it device-
     resident on the workspace and on disk (`np.save` per m-block); a survey run
     uses one `(nside, L)` for every field, mask, and bin;
   * the analysis stage is called **many times per pipeline**: `n_iter=3` means
     4 theta calls per field, and `compute_coupling_matrix` runs `O(nlb)`
     apodized analyses. So a 36 ms build amortises over tens of 4 ms calls.
3. **NaMaster's `n_iter` defaults are confirmed 3 / 3** (masked and unmasked) at
   `NaMaster/pymaster/utils.py:66-67`, identical to `gmaster/utils.py:26-27`.
   This closes an open question and re-confirms that masked-ring skipping stays
   dead (exact only at `n_iter=0`), while simultaneously *helping* the matrix
   amortisation argument above.
4. **The trick does not obviously transfer to synthesis.** The `alm2map` kernel
   contracts over `ell` with an RHS that is shared across the other axis
   (`out[b,j] = sum_k A[b,j,k] x[b,k]`), which is the *shared-RHS* variant
   measured at 498 GB/s in Session 5g — 3.2x worse than the per-m form. Worse,
   synthesis wants `ell`-contiguous storage while analysis wants `j`-contiguous,
   so a matrix pipeline needs two layouts or eats a stride. Treat the matrix
   route as an **analysis-side** win until synthesis is measured separately.

## Session 5i (2026-09-02): the e2e measurement inverts the priority ordering

`benchmarks/benchmark_pipeline.py --nside 256 --repeats 2 --spins 0,2` (first
time this has ever been run and recorded — Stage 4 of Session 5g's list):

```
nside=256 devices=['cuda:0']
spin=0: TOTAL 59->91ms (0.6x) | field 21->36ms (1x)  coupling 17->16ms (1x)
        coupled_cell 1->0ms (16x)  decouple 0->0ms (0x) | max|dCl|=8.08e-18 rel=4.85e-13
spin=2: TOTAL 162->1338ms (0.1x) | field 48->1272ms (0x)  coupling 85->41ms (2x)
        coupled_cell 3->0ms (48x)  decouple 0->0ms (0x) | max|dCl|=4.59e-13 rel=2.74e-08
```
(GMaster timings listed second; correctness is fine — decoupled spectra agree to
4.9e-13 relative for spin 0.)

**GMaster is currently 1.6x SLOWER than NaMaster end-to-end at nside 256 for
spin 0 and 8x slower for spin 2.** Everything measured so far (the theta kernel,
the matrix route, the fp64 roofline) optimises a stage that is not the
bottleneck. The bottleneck is the **`NmtField` construction stage**, and within
spin 2 it is catastrophic: 1272 ms vs NaMaster's 48 ms.

Why, and what it means:

* `.qwen/tmp/spin_stage_probe.py`, nside 256, spin 0 (Pallas path):
  `map2alm(n_iter=0)` 5.05 ms = ring FFT **0.81 ms + theta 4.05 ms** →
  **the theta stage is 80.2% of one analysis**; `alm2map` 5.99 ms;
  `map2alm(n_iter=3)` 35.92 ms (exactly the pipeline's `field` number).
  So for spin 0 the matrix route's 2.0x on theta really does convert into
  ~1.6x on the analysis stage — worth having, but analysis is only part of the
  1338 ms total.
* **Spin ≠ 0 never touches the fused Pallas kernels at all.**
  `utils.py:597` inside `_use_pallas_sht`:
  ```python
  if spin != 0:
      return False
  ```
  with the comment that the fused spin-weighted kernels are experimental because
  "their closed-form Wigner-d seeds lose relative precision through catastrophic
  cancellation once |m| approaches l", pending a renormalised sideways recursion
  (Turok–Bucher class). So polarisation runs `utils.py:_map2alm_once` →
  `_forward_s2fft(complex map, spin=2, reality=False)` → s2fft's *generic*
  latitudinal transform, which is what costs ~1.2 s.
* Spin 2 uses **one** complex transform (`maps[0] + i·maps[1]`, `reality=False`)
  and recovers both ±m from it (`_map2alm_once`, `utils.py:963-974`) — that part
  is already efficient; the cost is the generic theta stage itself.

Full stage split from `.qwen/tmp/spin_stage_probe.py` (warm medians, GPU 1):

| nside | spin | `map2alm` n_iter=0 | ring FFT | **theta stage** | `map2alm` n_iter=3 | `alm2map` |
|---|---|---|---|---|---|---|
| 256 | 0 (Pallas) | 5.02 ms | 0.82 ms | **4.04 ms (80.5%)** | 35.86 ms | 5.95 ms |
| 256 | 2 (generic) | 190.29 ms | 6.30 ms | **185.23 ms (97.3%)** | 1268.66 ms | 190.79 ms |
| 512 | 0 (Pallas) | 27.29 ms | 3.51 ms | **23.59 ms (86.5%)** | 198.03 ms | 30.90 ms |
| 512 | 2 (generic) | 1329.26 ms | 20.31 ms | **2287.05 ms (>100%)** | 14273.03 ms | 1972.22 ms |

The spin-2 theta stage is **30–100x the cost of the spin-0 fused kernel at the
same nside** (185 ms vs 4.0 ms at 256; 2287 ms vs 23.6 ms at 512) and is ~97% of
a spin-2 analysis. The `>100%` entry is an artefact of timing `_forward_latitudinal`
in isolation (its own jit loses an optimisation the composed call gets, plus only
1–2 samples fit in the 2 s window when one call is >1 s); treat 1.3–2.3 s as the
nside-512 range for that stage. It is `pallas=False` in every spin-2 row — the
fused kernels are simply not in the loop for polarisation.

### Estimated payoff of the matrix route on spin 2 (not yet measured)

Spin 2 with `reality=False` needs `d^ell_{m,2}(theta_j)` for all
`m in [-(L-1), L-1]`, `ell >= max(|m|, 2)`, over the full `ntheta = 4·nside-1`
ring set (no north/south parity fold like spin 0 has), i.e.

```
bytes ~= (2L) * (L/2) * ntheta * 8  =  38.6 GB at nside 512 / L=1536
```

Streamed at the *measured* banded rate (844 GB/s) that is **~46 ms**, and even at
the pessimistic padded-cube rate (395 GB/s) **~98 ms** — against 1329–2287 ms
today. So **~15–30x on the stage that is 97% of a polarised analysis**, which is
the "massively faster" the goal asks for, in exact fp64, with no unstable
recurrence at runtime. Memory is the price: 38.6 GB resident at nside 512 is fine
on this 96 GB card, but nside 1024 (~310 GB) forces per-m-band streaming with the
matrix re-read from host/disk rather than rebuilt.

Two things to verify before believing the estimate: (a) that a *spin-weighted*
matrix built once with a stable normalised recursion matches s2fft's generic
spin-2 stage to ~1e-14 (the spin-0 analogue matched the kernel to 5e-15), and
(b) that the spin-2 RHS has the per-m form — it does, since `m` indexes the ring
FFT columns exactly as in spin 0, so the same `bkj,bj->bk` schedule applies.

### nside 512 confirms the diagnosis and gives the exact target

`benchmarks/benchmark_pipeline.py --nside 512 --repeats 1 --spins 0,2`:

```
spin=0: TOTAL 338->501ms (0.7x) | field 105->198ms (1x)  coupling 106->100ms (1x)
        coupled_cell 4->0ms (66x)  decouple 0->0ms (0x) | rel=1.70e-12
spin=2: TOTAL 744->14648ms (0.1x) | field 237->14299ms (0x)  coupling 388->124ms (3x)
        coupled_cell 21->0ms (155x)  decouple 6->1ms (9x) | rel=3.56e-07
```

Every stage **except** `field` is already a win at nside 512 (coupled cell 155x,
decoupling 9x, coupling 3x). For spin 2 the `field` stage alone is 14299 ms of
the 14648 ms total (98%) — a **60x deficit against NaMaster concentrated in one
place**. The arithmetic of the fix is therefore explicit and checkable:

| spin-2 `field` stage | GMaster total | vs NaMaster 744 ms |
|---|---|---|
| today (generic theta ~1.3–2.3 s/analysis) | 14648 ms | 0.05x |
| theta → ~100 ms (pessimistic matrix rate) | ~2.4 s | 0.3x |
| theta → ~46 ms (measured banded rate) | **~0.8 s** | **~1.0x** |
| theta → ~46 ms **with the matrix cached on disk** | below 0.8 s | **>1x** |

So a validated spin-2 matrix theta stage is not just a speedup, it is *the*
change that flips GMaster from 0.1x to parity-or-better end-to-end for
polarisation, because the other four stages already win.

Note for whoever gates this: the spin-2 decoupled-cell discrepancy **grows with
nside** — `rel=2.74e-08` at 256 and `rel=3.56e-07` at 512 (spin 0 is
`4.85e-13` / `1.70e-12`). That is pre-existing and unrelated to this session's
measurements, but a spin-2 kernel swap must be gated against NaMaster on an
absolute tolerance that accounts for it, not against the current generic path
(which would hide it).

## Session 5j (2026-09-02): the s2fft `precomps` lever — measured, then deliberately NOT shipped

`_forward_latitudinal`/`_inverse_latitudinal` pass `precomps=None`, and s2fft
then regenerates its Price–McEwen recursion coefficients **inside the primitive's
lowering**, i.e. on every transform call, with no caching anywhere in s2fft.
Measured with `.qwen/tmp/precomps_probe.py` (precomps built once, passed as
explicit jit arguments):

| nside | precomps | build (one-time) | theta stage as-is | with hoisted precomps | saving |
|---|---|---|---|---|---|
| 256 (L_work 768) | 38 MB, 5 arrays | 337 ms | 185.22 ms | 161.64 ms | **1.15x, −23.6 ms/call** |
| 512 (L_work 1536) | 151 MB, 5 arrays | 421 ms | 1310.73 ms | 1230.88 ms | **1.06x, −79.9 ms/call** |

Numerics vs the current path: `max|d|` 2.1e-15 / 2.5e-15, `rel` 2.8e-14 / 4.7e-14 —
safe, but **not bit-identical**, so it must still be gated against NaMaster.
The five arrays are exactly as documented at
`s2fft/recursions/price_mcewen.py:181-255` (verified shapes at nside 512:
`lrenorm (2,2047,1536)`, `vsign (3071,1536)`, `cpi/cp2 (1537,1536)`,
`indices (2047,1536)`). They are **recursion coefficients, not a Wigner-d table**
— the recurrence itself is untouched, which is why the win is 1.06–1.15x and not
the 30–100x gap to the fused spin-0 kernel. At nside 512 that is only ~4% of the
spin-2 `field` stage (7 theta calls in `map2alm(n_iter=3)`).

**I implemented it, the test suite caught a JAX trap, and I reverted it.** The
naive patch (`@lru_cache` helper returning `generate_precomputes_jax(...)`,
referenced from `_forward_latitudinal`/`_inverse_latitudinal`) fails **5 of the
15 tests** in `tests/test_sht.py` with `jax.errors.UnexpectedTracerError`:

```
The function being traced when the value leaked was _inverse_s2fft ...
The leaked intermediate value was created on line ...:451 (_sht_precomps)
```

Mechanism: the first call to the cached helper happens **while `_forward_s2fft` /
`_inverse_s2fft` are being traced**, so `lru_cache` stores *tracers* in global
state and every later call reuses a leaked tracer. The standalone probe measured
cleanly precisely because it passed precomps as **explicit arguments to a jit**
instead of closing over them. `gmaster/utils.py` was reverted to its exact
pre-session diffstat (38 insertions / 12 deletions) and the suite re-run to
confirm the baseline; nothing of mine is left in `gmaster/`.

To ship it correctly (if ever wanted — it is a ~4% pipeline win, so it is *not*
the priority): precomps must be **created outside any trace and threaded in as
arguments** — `map2alm`/`alm2map` (untraced) → `_map2alm_core` → `_map2alm_once`
(jitted) → `_forward_s2fft` → `_forward_latitudinal`, and the same chain for the
inverse. Three traps on that path: (1) `_inverse_latitudinal_device` splits the
beta grid in half (`utils.py:845-857`), so precomps must be keyed per truncated
grid or left off there; (2) the multi-GPU forward uses two different
`(L, L_lower)` keys; (3) hoisted arrays are pinned to the device they are built
on, so a cross-device transform would add a per-call copy — and s2fft's transpose
rule for this primitive **drops precomps**, so `jax.grad` regenerates them anyway
(gradient tests must still pass).

### Corrected belief: s2fft has two different "precompute" modes

HANDOFF line ~540 previously asserted "s2fft's `precompute` mode is known to lose
accuracy". Verified from source, that conflates two unrelated mechanisms:

* `generate_precomputes_jax` (`s2fft/recursions/price_mcewen.py:119-134`) —
  **O(L²) recursion coefficients**, documented as "well worth the acceleration",
  with **no accuracy caveat**; s2fft's own test asserts 1e-12 agreement with the
  on-the-fly path.
* `precompute_transforms` (full Wigner-d kernel, `construct.py:29/177/345/484`) —
  documented **O(L³) memory, "infeasible for large bandlimits"**, and its forward
  needs `iterative_refinement` (`spherical.py:265-266, 305`). *That* is the mode
  with the accuracy/memory problems, and it also silently disables `reality` for
  spin ≠ 0.

So "s2fft precomputes are inaccurate" is **not** a reason to avoid the coefficient
tables. It *is* a reason to be careful about the kernel route at L ≥ 512 — which
matters because GMaster's own plan (a precomputed Legendre/Wigner-d matrix) is
that second category, and must therefore be validated against NaMaster rather
than assumed, exactly as the spin-0 matrix was validated here to 5e-15.




### Reordered plan for "massively faster, accuracy preserved"

1. **Bring the matrix theta stage to spin 2.** This is now the highest-value
   application of the same technique, not an afterthought: the validated
   precomputed-matrix contraction has *no recurrence and no unstable seed* — the
   exact defect blocking `_use_pallas_sht(spin=2)`. Build
   `d^ell_{m,±2}(theta_j)` once in fp64 with a stable normalised recursion, verify
   against the generic path (the same verification harness already validates the
   spin-0 matrix to 5e-15), then contract with the multiply-reduce form. Against
   a ~1.2 s generic stage, a matrix contraction of the same ~10 GB band is a
   plausible **order-of-magnitude** win on the stage that dominates today's
   polarised pipeline — whereas the identical trick on spin 0 is only 2.0x of 80%.
2. Keep the spin-0 matrix route (2.0x on 80% of the analysis stage) and cache the
   matrix per `(nside, L)` on disk; both spins then share one mechanism.
3. Also re-measure spin-0 `coupling` (17→16 ms, only 1x) — NaMaster's coupling
   matrix construction is 30 `NmtWorkspace`-scale solves and GMaster barely wins
   there today.
4. Fix the loose end this benchmark exposed: spin-2 decoupled cells agree only to
   `rel=2.7e-08` (vs `4.9e-13` for spin 0). Pre-existing, not caused by anything
   in this session, but it should be understood before any spin-2 kernel swap is
   gated on test parity.

## Session 5k (2026-09-02): the fused spin-2 kernel is not "slightly inaccurate" — it is disqualified

Plan item 1 has a tempting shortcut: `_map2alm_once_pallas_spin` already exists
(`_spin_forward_latitudinal`, grid `(orders, cdiv(ntheta, block))` — far better
parallelism than the scalar kernel's `(m_count,)`), and the only reason
`_use_pallas_sht` refuses `spin != 0` is the comment at `utils.py:591-596`:

> Fused spin-weighted kernels remain experimental: their closed-form Wigner-d
> seeds lose relative precision through catastrophic cancellation once |m|
> approaches l.

That wording implies a *localised, marginal* defect, and flipping the switch
would be a one-line win on the stage that is 97% of a polarised analysis.
`.qwen/tmp/spin2_fused_probe2.py` tests it properly instead of trusting it: it
monkeypatches `_use_pallas_sht` to return True and runs the public
`map2alm(maps, spin=2, ...)` against `pymaster` with `n_iter=3`, on maps built by
`alm2map` from random spin-2 alms.

| config | generic vs pymaster | fused vs pymaster | fused / generic time |
|---|---|---|---|
| nside 32, lmax 95  | 2.547e-12 | **2.335e+01** | 29.1 ms vs **370.8 ms (0.08x)** |
| nside 64, lmax 191 | 1.528e-11 | **3.392e+01** | 74.8 ms vs **1367.1 ms (0.05x)** |

Two independent disqualifications, neither of them the one the comment describes:

1. **The output is wrong by ~5 orders of magnitude, uniformly in m.** Error by
   m-bucket at nside 64: `m0:2.4e1 m24:2.6e1 m48:2.7e1 m72:2.3e1 m96:3.3e1
   m120:3.4e1 m144:2.9e1 m168:2.9e1`, against `max|ref alm| = 5.4`. This is not
   cancellation near the bandlimit edge — it is a convention/normalisation/grid
   error somewhere in `_spin_forward_latitudinal` (or in the
   `signal_plus/minus · window` prep around it, `utils.py:1084-1100`). The error
   is flat in m, so nothing is gained by "fixing the seed only for large |m|".
2. **Even if it were correct, it would be 12-20x SLOWER than the generic path it
   would replace.** The two-helicity form does two full `L x 2L-1` latitudinal
   passes (`spin=+s` and `spin=-s`) over a `_forward_ring_fft_full` grid, whereas
   production does one complex transform and recovers the second helicity from
   `(-1)^m conj(.)` (`utils.py:969-974`). Parallelism cannot pay for 2x the work
   on a grid that is also bigger; the generic path's single complex transform is
   simply the cheaper algorithm.

**Consequences, in order of importance:**

* **Do not "just enable" the fused spin kernels.** The route from the code
  comment to a quick win does not exist. Anyone picking this up should delete
  `_map2alm_once_pallas_spin` + `_spin_forward_latitudinal` or leave them clearly
  marked dead, rather than treating them as 90%-done.
* **The generic spin-2 path is not the soft target it looked like.** It already
  beats a purpose-built parallel kernel. Plan item 1 (a precomputed
  `d^ell_{m,+-2}` matrix) is therefore the *only* known route to the polarisation
  win, and it has to beat the generic path's single-complex-transform work count,
  not the Pallas kernel's. Re-derive the matrix byte/flop budget against
  `_forward_s2fft` with `reality=False` before writing any kernel.
* The generic path's own agreement with NaMaster degrades with bandlimit
  (`2.5e-12` at nside 32, `1.5e-11` at nside 64 for this input), which is a second
  reason plan item 4 exists: a future spin-2 matrix stage verified against the
  generic path inherits that error, so the gate must be NaMaster, not "matches
  the current path".

## Session 5l (2026-09-02): precomputed-Legendre theta stage shipped as `jax-matrix`

`gmaster/_theta_matrix.py` is live behind `set_sht_calculator("jax-matrix")`
(`utils.py`: allowed-calculator tuple, `_fused_forward_sht` dispatch,
`_MATRIX_BAND_BUDGET = 32 GiB` fit gate). It replaces the in-kernel regeneration of
`d^l_{m,0}(theta_j)` with a cached m-banded matrix, moving the theta stage from
recurrence-bound to memory-bound. Values come from the same normalised 3-term
recurrence, seed, rescale schedule and equator rule as
`_sht_pallas._analysis_kernel`, so this is a reorganisation, not an approximation.

Theta stage against the fused kernel (`.qwen/tmp/matrix_module_test.py`, GPU
warmed, both sides timed twice in one process):

| nside | kernel | matrix | speedup | rel vs kernel |
|---|---|---|---|---|
| 64 | 0.30 ms | 0.15 ms | 1.96x | 1.9e-15 |
| 128 | 0.87 ms | 0.34 ms | 2.58x | 3.1e-15 |
| 256 | 4.03 ms | 1.51 ms | 2.66x | 5.2e-15 |

Zero leakage into `ell < m` in all three. The parity split is what got it there:
carrying the `(-1)^(ell+m)` north/south fold as four real RHS columns measured only
1.34x at nside 256 — arithmetic, not bytes, was the limit. Splitting rows by parity
of `i = ell - m0` turns the fold into a per-column sign, so each half contracts with
one complex RHS: half the flops at identical bytes. `BLOCK = 64` being even is
load-bearing for `(-1)^(m-m0) == (-1)^m`.

Whole pipeline with the calculator selected through the public switch (nothing
monkeypatched) and `pymaster` as reference (`.qwen/tmp/matrix_pipeline.py`,
`.qwen/tmp/matrix_512.py`):

| config | `jax` | `jax-matrix` | speedup | rel vs NaMaster |
|---|---|---|---|---|
| nside 256 spin 0 | 90.0 ms | 70.1 ms | 1.28x | 4.84e-13 (both) |
| nside 512 spin 0 | 501.4 ms | 411.0 ms | 1.22x | 1.70e-12 (both) |

NaMaster runs the nside-256 spin-0 pipeline in 59.6 ms, so this closes the gap from
0.66x to 0.85x. Band cost is 1.3 GiB at 256 and 10.1 GiB at 512 (measured
`bytes_in_use`, matches `band_bytes`); at 1024 it would be ~77 GiB and the gate
falls back to the kernel. Tests: `115 passed, 3 skipped` (was 114), including the new
`test_matrix_theta_stage_matches_fused_kernel`; the suite also passes with
`jax-matrix` forced for every test.

**Two measurement traps cost a full detour and must be pinned down.**

1. *XLA scatter is pathological.* The first `_transform` wrote blocks into the
   `(ell, m)` output with `out.at[rows[:, None], cols[None, :]].set()` and measured
   150-1900 ms — 500x slower than the kernel it replaces, while numerically exact.
   `.qwen/tmp/bisect_matrix.py` separated reduction from assembly at nside 128
   (kernel 0.87 ms): chained 2-D scatter **519.9 ms**, one flattened scatter
   **588 ms**, strided slice `at[m0+off::2]` **8.44 ms**, contiguous
   `dynamic_update_slice` **0.366 ms**, `jnp.pad` + `jnp.concatenate` **0.354 ms**,
   reduction alone **0.243 ms**. Assembly was the entire cost. Banded blocks are
   rectangles at `(m0, m0)`, so they must be assembled with pad-and-concatenate —
   never a scatter, never a strided slice. This applies to any future matrix stage,
   including the spin-2 one.
2. *Cold-GPU clock ramp.* Timing the Pallas kernel as the first GPU work in a
   process inflated it ~50x (221.90 ms vs 4.03 ms steady). That single artifact
   explains the mutually contradictory times across earlier probes. Harnesses must
   `burn()` the GPU and time the incumbent both before and after the contender.

Also: `compute_coupled_cell` memoises onto the field, so timing it against a warm
field measures a dict lookup (0.1 ms). Every stage split in this session rebuilds
its inputs.

**Trace safety.** `jax.ensure_compile_time_eval()` does *not* make the band builder
concrete under `jax.grad`/`jax.linear_transpose` —
`test_fused_scalar_transform_gradients_match_generic_jax` caught it. `_BAND_CACHE`
is therefore an explicit dict that stores a geometry only once its slabs are real
device buffers (tested with `hasattr(slab, "block_until_ready")`) and returns `None`
otherwise, which makes `_fused_forward_sht` fall back to the kernel. Gradients still
flow; they just do not get the speedup.

**Where the remaining time is.** `NmtField.__init__` runs `map2alm` itself
(`field.py:197`), so the SHT lives in the *field* stage and `compute_coupled_cell`
only multiplies cached alms. Any stage table that rebuilds fields inside each timed
closure therefore counts the SHT two or three times — the first version of this
table summed to 6357 ms at nside 256, which was an artifact. Attributing the SHT to
the stage that actually pays for it:

| stage (nside 256, unique work) | spin 0 | spin 2 |
|---|---|---|
| field, incl. the SHT | 35.7 ms | **1248.0 ms** |
| coupled cell (alm products on cached alms) | ~3 ms | ~3 ms |
| coupling matrix | ~90 ms | ~113 ms |
| whole pipeline, `jax` | 90.1 ms | 1332.9 ms |
| whole pipeline, `jax-matrix` | **70.6 ms** | 1324.8 ms (1.01x, noise) |
| whole pipeline, NaMaster | 58.4 ms | 166.2 ms |

(`.qwen/tmp/matrix_pipeline.py`, `.qwen/tmp/matrix_pipeline_spin2.py`,
`.qwen/tmp/matrix_512.py`.) The spin-2 accuracy floor against NaMaster is
`2.682e-08` under both calculators — the pre-existing generic-path gap, unchanged by
this work.

`jax-matrix` moved the spin-0 field stage 35.7 → 25.5 ms (1.40x) and the spin-2
field stage 1248.0 → 1248.0 ms (**exactly 1.00x**), because `_use_pallas_sht` refuses
`spin != 0` and the polarised path never reaches `_fused_forward_sht`. The spin-2 SHT
is ~91% of the polarised pipeline on its own, so ~96% of the distance to NaMaster
sits in the generic s2fft spin-2 latitudinal transform (`_forward_latitudinal` →
`ftm_to_flm`). Together with Session 5k (the purpose-built fused spin kernel is
disqualified and 12-20x slower than generic), the only credible route to "massively
faster" is plan item 1: an `m' = ±2` column transform for spin 2 — two extra
Wigner-d columns, ideally reusing this module's banded layout and pad-and-concatenate
assembly — which would price a spin-2 SHT at roughly 2x scalar instead of ~35x.

Do not try to answer this by wrapping `map2alm`/`_forward_s2fft` in timing
decorators: the calls nest, the decorators double-count each other, forcing
synchronisation destroys the async pipeline being measured, and the probe OOMed on
CUDA-graph instantiation ("22 alive graphs"). Read `field.py` and take the
difference of whole-pipeline measurements instead.

Reduced precision is the other lever and is still blocked: the Pallas kernels are
fp64-hard-wired through their `plt` reference dtypes (`ValueError: Invalid dtype for
swap. Ref dtype: float64. Value dtype: float32`), and cuBLAS 13.0.2.14 on this host
exposes only FP32 BF16x9 emulation, no FP64 emulation. Achieved streaming is already
865-1600 GB/s against a 1603 GB/s measured peak, so within exact fp64 this stage is
close to its ceiling.

---

## Session 6 (2026-09-03) — four landed wins; scoreboard by Nside

Everything below is measured on GPU1 with `CUDA_VISIBLE_DEVICES=1`, `repeats=5`,
`total` computed as `field + coupling + coupled_cell + decouple_cell` (never the
wall-clock `pipeline` column), and `rel` = max relative deviation from NaMaster's
coupled spectrum. **No change this session moved any `rel` value** — spin 0 stays at
5.9e-14 (n64) / 1.03e-13 (n128) / 4.76e-13 (n256), spin 2 at 1.07e-10 / 4.90e-09 /
2.74e-08. Test suite is `115 passed, 3 skipped` at every commit.

### Landed

1. **`c485d64` — scalar synthesis reads the cached Legendre band.** The spin-0
   `inverse_latitudinal` was recomputing the `D`-band from the three-term recurrence
   on every call while the analysis side already used `_band_cache`. Added
   `_theta_matrix._synth_band` (same rows, re-laid out `(m_local, j, ell_row)` for the
   synthesis contraction) + `_SYNTH_CACHE`, and `_fused_inverse_sht` now dispatches to
   it under the same `_prefer_theta_band(nside, L, 0)` and `2·band_bytes ≤
   _MATRIX_BAND_BUDGET` gates as analysis. Stage **1.67–1.87×** at Nside 128/256/512.
2. **`c502c26` — skip the zero head of the polarised slab.** Row `L-1+m` of the
   `_spin_slice` slab holds `d^ell_{m,-spin}`, which is *identically zero for `ell <
   |m|`*: just under half of a buffer that is streamed twice per polarised transform.
   `_windows(L)` returns `(r0, r1, lo)` per 64-row block and both
   `forward_latitudinal`/`inverse_latitudinal` contract only `slab[r0:r1, lo:]`
   (analysis, THETA_CONTIG) / `slab[r0:r1, :, lo:]` (synthesis, ELL_CONTIG), writing
   `out.at[lo:, …]`. Skipped rows keep their exact zeros, so the result is
   **bit-identical** (`rel` 0.0 / 1e-17). Stage **1.79× analysis / 1.75× synthesis**,
   n256 spin-2 TOTAL **165 → 108 ms**. Block 64 is optimal: 16 skips more bytes but
   pays 4× the kernels (1.34×/1.20×), 128 gives 1.70×/1.71× at n256 but loses at
   n128; a variant that also skips the `ell > 2L-2-|m|` tail was consistently worse.
3. **`70e7f07` — traced refinement loop, gated.** Running the whole `n_iter`
   refine-loop as one traced program removes the per-op host dispatch that dominates
   small-N (`field` at n64 spends ~8 ms of *host* time per field: 395
   `apply_primitive` calls and 115 `rewriting_take`/`_gather` gathers from the
   triangle↔square layout conversion). Traced is 1.8× faster at n64, 1.05× at n128,
   and **1.4× slower at n256** (one program must hold every intermediate), so
   `_PALLAS_TRACED_MAX_L = 256` selects traced below it and eager above.
   n64 spin 0 **12 → 7 ms**, n64 spin 2 **13 → 12 ms**.
4. **`a3ddbc9` — offset-blocked scalar MASTER matrix.** A coupling term vanishes
   unless `offset ≤ min(l1, l2)`, so offset block `[o0, o0+C)` only touches the
   sub-matrix `[o0:, o0:]`: ~n³/3 element evaluations instead of n³. Chunk 16 measured
   best (3.0–3.6× vs 2.0–3.3× for 8/32/64). Coupling **16 → 6 ms** at n256,
   4.4 → 1.5 ms at n128, 2.1 → 0.75 ms at n64; n256 spin-0 TOTAL **61 → 47 ms**.
   The spin-2 matrix builder already used matmuls and needed nothing (16 ms vs
   NaMaster's 83 ms).

### Scoreboard (TOTAL ms, GPU1, repeats=5)

| Nside | spin | NaMaster | GMaster before | GMaster now | NaMaster / now |
|---|---|---|---|---|---|
| 32 | 0 | 4.5 | 3.7 | 4.0 | 1.1× |
| 32 | 2 | 3.7 | 7.3 | 7.1 | **0.5×** |
| 64 | 0 | 7.5 | 7.7 | 6.8 | 1.1× |
| 64 | 2 | 14.4 | 12.7 | 13.2 | 1.1× |
| 128 | 0 | 17.9 | 21.1 | 16.1 | 1.1× |
| 128 | 2 | 41.5 | 27.5 | 26.2 | **1.6×** |
| 256 | 0 | 61.2 | 59.5 | 47.2 | **1.3×** |
| 256 | 2 | 162.7 | 114.6 | 108.6 | **1.5×** |
| 512 | 0 | ~300 | 196 | (see below) | — |
| 512 | 2 | — | 8500 | (see below) | — |

Nside 32 is host-bound and its ±0.5 ms noise exceeds the difference; both codes are
"one launch budget" there. Nside 32 spin 2 is the one cell where GMaster is genuinely
slower than NaMaster: NaMaster's spin-2 cost at tiny Nside is dominated by small
matrix work it does in optimized BLAS, while GMaster's pipeline has ~3× the kernel
count and its spin-2 `field` stage cannot amortise it.

### Two measurement traps that cost hours (now in memory)

- **Async leaves.** `benchmarks/benchmark_pipeline.py::_block` walks the result with
  `jax.tree.map(..., is_leaf=lambda x: hasattr(x, "block_until_ready"))`. `NmtField`
  and `NmtWorkspace` are plain Python objects, so a jitted `field` stage returns
  *without* any device array in it and is timed as host enqueue only — that is why
  "field 21 → 4 ms" appeared when the loop was traced while the TOTAL got worse.
  Trust a stage only if its return value contains arrays, and always cross-check the
  TOTAL.
- **Constant-folded probes.** A probe `@jax.jit` that closes over module-level device
  arrays makes them HLO constants, and XLA folds the whole call away (it reported
  8.6 TB/s on a 1.2 GiB read). Always pass big tables as explicit jit arguments.

### Refuted / dead ends (do not re-derive)

- **π−θ symmetry to halve the polarised slab.** `d^l_{m,-spin}(π−θ) = (-1)^l
  d^l_{-m,-spin}(θ)` looks like a 2× memory win and is in every table-of-Wigner-d
  handbook, but it is **false for this slice** — tested directly against the built
  table (`_m_slice` row `L-1+m`): residual 1.334 of the table max for spin 2, 1.998
  for spin 1 (a true identity would be ~1e-15). `.qwen/tmp/slab_symmetry.py` has the
  three control comparisons.
- **Real-pair packing** (`_pack_pair`) to halve the slab footprint: same total bytes
  as the complex form because the pair covers both signs of `m`; measured no win in a
  prior session and re-derived here — `2 × (rows, L, θ) real = 2 × (rows, 2L-1, θ)`
  real.
- **Per-window slab rebuild** to bound memory at n512: the slab takes ~15 s to build
  against a ~0.2 s transform, so rebuilding it per window costs more time than the
  windowing saves.
- **Splitting the band's complex reduce** into two real reduces: 1543 vs 1412 GB/s in
  a matched synthetic test (and the two forms differ by 1.6e-15 because of summation
  order). XLA already fuses the 4-D product; do not pay an accuracy change for 9%.
- **Always-traced core loops**, m-window blocks of 16 and 128, and a window variant
  that skips both ell ends: all measured worse than what ships.

### Where the time still goes (n256, spin 2, after all four)

`field 19` · `map2alm 53` · `alm2map 36` · `coupling 16` · `coupled_cell 3` ·
`decouple 2` · `synthesize 6`. Both transforms are now pure bandwidth: the spin-2
latitudinal contraction streams the windowed slab at ~735 GB/s against the ~1578 GB/s
real-read rate, and the shortfall is the **complex-RHS tax** — every table element is
multiplied by a complex weight, and XLA has no way to avoid two real reductions per
element (see memory `complex-rhs-tax.md`: every XLA-level reformulation tried,
including einsum variants, `lax.dot_general`, splitting, and batched forms, measured
at the same 1.00–1.05×). The scalar band contraction runs at 938 GB/s of table bytes
at n512 and is partly per-block launch-bound (24 block reduces + a pad/concatenate
assembly chain per transform).

### The n512 spin-2 wall

The `_spin_slice` slab pair is `2 × (2L-1) × ntheta × L × 16 B` = **154 GiB** at
Nside 512 against a 71.2 GiB JAX pool (0.75 × 96 GiB RTX PRO 6000). Even a *single*
layout (77 GiB) does not fit, so spin 2 at n512 falls through to the generic s2fft
path (measured 8.5 s, i.e. ~0.03× NaMaster). The m-window skip does not change the
*allocation*, only the bytes read. Options, none free: (a) split the slab along `m`
with a per-window transform — rebuild cost exceeds the win (above); (b) contract the
Wigner-d recurrence on the fly inside a Pallas kernel — the real fix, and the only
route past the complex-RHS tax at any Nside; (c) keep n512 spin 2 on the generic path
and say so. Spin 0 at n512 is unaffected (band budget fits) and remains the fastest
cell in the table.


