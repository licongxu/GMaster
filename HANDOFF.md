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

## Session 6 (2026-09-03) — five landed wins; scoreboard by Nside

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

### Scoreboard (TOTAL ms, GPU1, repeats=5, `8ca2304`)

| Nside | spin | NaMaster | GMaster at session start | GMaster now | NaMaster / now |
|---|---|---|---|---|---|
| 32 | 0 | 3.0-4.5 | 3.7 | 3.5-4.0 | ~1× (noise-bound) |
| 32 | 2 | 3.7-4.0 | 7.3 | 6.9 | **0.5×** |
| 64 | 0 | 7.0-8.0 | 7.7 | 6.6-7.0 | 1.0-1.1× |
| 64 | 2 | 14.0-14.4 | 12.7 | 10.2 | **1.4×** |
| 128 | 0 | 17.9-23 | 21.1 | 15.4 | **1.2-1.5×** |
| 128 | 2 | 40-42 | 27.5 | 24 | **1.7×** |
| 256 | 0 | 61-65 | 59.5 | 47 | **1.3×** |
| 256 | 2 | 155-163 | 114.6 | 102 | **1.5×** |
| 512 | 0 | 339-345 | 196 | 288-290 | **1.2×** |
| 512 | 2 | 687-718 | 8500 | ~8700 | **0.08×** |

Nside 32 is host-bound and its noise (±1 ms on a 3-4 ms total) exceeds the
difference, so both spins there are a tie rather than a measured win or loss —
except spin 2, where GMaster's extra launches still cost it ~3 ms.

### Fifth win (`8ca2304`) — one Wigner-d table per geometry

`_coupling_matrices_spin2` asked for **six** Wigner-d tables, built inside two
separate jit calls, so XLA could not merge them: the `(0,0)` table at
`lmax_mask` twice literally, and — because `d^l_{-m,-n} == d^l_{m,n}` when `m - n`
is even, verified bit-for-bit against this builder — each call's `first`/`second`
pair is really one table. They are sequential scans over degree, which is why the
stage cost barely depended on `lmax` (~5 ms at lmax 95 vs 16 ms at 767). Building
them outside the quadrature call and passing them in collapses six to three:
coupling **77 → 10 ms** at Nside 256, **17 → 6 ms** at 128, taking those cells to
1.5× and 1.7× NaMaster. `rel` values are bit-identical before and after.
Applying the identity to *odd* `m - n` pairs is wrong (there it carries a minus
sign) and `tests/test_workspaces.py::test_arbitrary_spin_workspaces_match_namaster`
fails on four parameterisations — the guard is `(m - n) % 2 == 0`.


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
- **Splitting the band's complex reduce** into two real reduces: *untested, do not
  cite*. `.qwen/tmp/band_forms.py` compares the shipped stacked reduce against an
  einsum-against-complex form and a two-real-reduce form on the real cached band, but
  it dies on ragged m-block shapes (`np.stack` of unequal blocks), and the "1412 vs
  1543 GB/s" pair quoted at the end of the previous session was never read from a
  log. The measured facts are only these: the band contraction moves table bytes at
  783/609 GB/s (analysis/synthesis) at Nside 256 and 769/639 at 512, against a
  ~1578 GB/s fp64 read rate. Fix the probe's comparison before concluding anything
  about the reduce form.
- **Always-traced core loops**, m-window blocks of 16 and 128, and a window variant
  that skips both ell ends: all measured worse than what ships.

### Where the time still goes (n256, spin 2, after all five)

`field ~69` · `coupling 10` · `coupled_cell ~0` · transforms ~52 (analysis) + ~36
(synthesis) of the field stage. Both transforms are now pure bandwidth: the spin-2
latitudinal contraction streams the windowed slab and the shortfall against the
~1578 GB/s fp64 read rate is the **complex-RHS tax** — every table element is
multiplied by a complex weight, and XLA has no way to avoid two real reductions per
element (see memory `complex-rhs-tax.md`: every XLA-level reformulation tried,
including einsum variants, `lax.dot_general`, splitting, and batched forms, measured
at the same 1.00–1.05×).

### The n512 spin-2 wall, with the last escape route closed

The `_spin_slice` slab pair is `2 × (2L-1) × ntheta × L × 16 B` = **154 GiB** at
Nside 512 against a 71.2 GiB JAX pool (0.75 × 96 GiB RTX PRO 6000), so `slabs_for`
rejects it at its 48 GiB gate and the transform falls to the generic
`lax.fori_loop` scatter path: **8.7 s** for one field against NaMaster's 0.69 s
(`GPUpeak=20.2GiB` confirms no slab was attempted). Three ways out were measured and
all three are shut:

- **Single layout** (77 GiB): synthesis contracted against the theta-contiguous
  buffer is exact to 2.2e-16 and costs 18.0 ms vs 15.7 ms at Nside 256 — cheap
  enough — but 77 GiB plus ~14 GiB of `ftm`/`flm` exceeds the 96 GiB *physical*
  card. Not a budget-setting problem.
- **Windowed allocation**: the slab marches sequentially over `m`, so the
  `ell >= |m|` rows cannot be produced without their predecessors and `lax.scan`
  materialises every row.
- **Fusing the contraction into the march** (no slab at all): the march costs
  549 / 466 / 666 ms at Nside 64 / 128 / 256 (`.qwen/tmp/march_cost.py`) — flat in
  size, i.e. latency-bound, and 10–100× the cost of *reading* the table it builds
  (8.99 GiB at Nside 256 is a ~6 ms peak-bandwidth read). Paying it seven times per
  field, as the fused form must, is worse than the generic path it would replace.

The remaining route is a fused Pallas/Triton kernel that runs the recurrence with
parallel reduction — which memory `blocked-recurrence-wall.md` records as already
tried and stopped by the fp64 wall at `m ~ L/2`. Spin 0 at n512 is unaffected (band
budget fits) and remains the fastest cell in the table.



---

## Session 7 (2026-09-03) — the polar table halves twice; Nside 512 spin 2 beats NaMaster

Two commits, both on the polarised latitudinal step:

| commit | what | n512 spin-2 TOTAL |
|---|---|---|
| `5f08924` | `ell >= |m|` windowed triangles + two pipeline bug fixes | 8700 → 873 ms |
| `f139526` | non-negative orders only (`pi - theta` / m-flip identity) | 873 → **619 ms** |

### Scoreboard (TOTAL ms, GPU1, `ref->gm`, medians)

| Nside | spin | NaMaster | GMaster now | ratio | rel | session start |
|---|---|---|---|---|---|---|
| 128 | 0 | 19 | 16 | **1.2×** | 1.03e-13 | 15.4 |
| 128 | 2 | 38 | 26 | **1.5×** | 4.90e-09 | 24 |
| 256 | 0 | 60 | 47 | **1.3×** | 4.76e-13 | 47 |
| 256 | 2 | 166 | 97 | **1.7×** | 2.74e-08 | 102 |
| 512 | 0 | 333 | 288 | **1.2×** | 1.71e-12 | 288-290 |
| 512 | 2 | 706 | **619** | **1.14×** | 3.56e-07 | ~8700 |

`rel` is bit-identical to the pre-session values at every cell (`3.56e-07`,
`2.74e-08`, `4.90e-09`, `1.71e-12`), so none of the speed was bought with accuracy.
Nside 128 spin 0/2 is a few ms off its session-6 best (16 vs 15.4, 26 vs 24) — host
launch bound, inside the run-to-run spread of these cells.

Stage split at n512 spin 2 (`chain3.log`): field 225 → 441 ms, coupling 375 → 50 ms,
coupled_cell 19 → 0 ms, decouple 0 → 1 ms. **`field` is now the only stage where
GMaster is behind NaMaster** (441 vs 225 ms); the MCM side is 7-200× ahead.

### Win 1 — `ell >= |m|` windowed triangles (`5f08924`)

A window of 64 contiguous rows shares the bound `lo = min |m|`, so it stores only
`[lo, L)`. Analysis stays bit-identical, synthesis is 1.8e-16 against a window-free
oracle, speed is unchanged (0.95-1.01×), bytes fall 1.85×. Two bugs hid it inside the
pipeline:

- `slabs_for` ran the pool-headroom gate **before** the cache lookup. A pool holding a
  37 GiB layout is by definition short of 37 GiB, so every call after the first
  declined to the generic scatter loop (`n512_calls.py` printed
  `slabs_for #2: DECLINED in 0.0 ms`). Cache lookup now comes first.
- the block sets were arguments of the outer pipeline `jax.jit`, so XLA counted both
  layouts against the pool and refused the executable
  (`byte size of input/output arguments (80583709680) exceeds the base limit
  (76461096960)` — the "base limit" is the 71.2 GiB pool). Only the latitudinal
  contraction is jitted now (`_forward_latitudinal_slab` / `_inverse_latitudinal_slab`);
  every s2fft stage around it was already jitted individually.

After both: `.qwen/tmp/n512_calls.py` → `BUILD #1: 16.0 s ok`, `field #1 in 23.9 s`
(includes the one-time trace), `field #2 in 0.7 s`, `declined: 0`. Before: 12.1 s per
field on the generic path, 29.7 s with the broken slab path.

### Win 2 — the `pi - theta` / m-flip identity (`f139526`)

Measured on the built slice (`theta_sym3.py`), for **every row with `m != 0`**, spin 1
and 2:

```
T[m, pi - theta, ell] = (-1)**(ell - m') * T[-m, theta, ell],   m' = -spin
```

worst per-row relative residue **2.3e-12**; the sign is constant along theta inside
each ell column and equals `(-1)**(ell - m')` on every entry with `|ratio| > 0.5`
(0 violations of ~48k per sampled row). The HEALPix ring grid is symmetric about
pi/2 to **4.0e-15**, so `theta -> pi - theta` is exactly the ring reversal
`i -> ntheta-1-i`.

**`m = 0` is the exception**: that row is its own mirror and violates the relation
across the whole theta range (residue 1.33 at spin 2, 2.00 at spin 1) — 1 bad row of
191. That single row is why an earlier whole-table absolute-residue test recorded "no
theta-halving of the polarised slab"; that verdict was wrong, and
`memory/project/wigner-d-symmetries.md` now points here.

Implementation: the block sets store orders `0..L-1`; each stored row is contracted
twice, once straight against `ftm[:, L+m]` and once against the ring-reversed
`ftm[:, L-m]` with the sign applied on output, order 0 taking the direct channel only.

Validation (`.qwen/tmp/half_ok.py`, oracle = full slab evaluated by definition):
analysis 5.08e-12 / synthesis 1.32e-11 / **adjoint 1.02e-15** at n128 spin 1; spin 2 is
better (2.67e-14 / 1.07e-13 at n64). Against a **float128** oracle
(`.qwen/tmp/half_precision.py`, n128 spin 1) the shipped full-slab path is
7.95e-16 / 1.01e-15 and this path is **1.46e-12 / 1.21e-11** — the reconstructed
negative orders inherit s2fft's march-to-march inconsistency rather than being wrong.
Four orders below the pipeline's accepted `rel`, and invisible in it, but it is a real
difference: do not describe this path as bit-identical.

### Two things that measured the opposite of expectation

- **Fusing the two channels into one einsum is 2x slower.** `einsum("cet,cth->ech")`
  over a stacked channel axis (one pass over the block, two FMAs per element) gave
  n512 spin-2 field 694 → **1392 ms**; two plain `einsum("cet,tc->ec")` calls give 441.
  XLA tiles the N=1 GEMV shape well and the N=2 batched shape badly. Shipped form: two
  calls (`chain.log` vs `chain3.log`).
- **Nside 640 cannot be benchmarked**: the NaMaster reference itself raises
  `ValueError: Something is wrong with your input arrays` in `pymaster/utils.py:272`
  from this harness, so there is no reference cell for 640 — GMaster-side numbers at
  640 would be unbenchmarked.

### Memory law now (exact, from `triangle_bytes`)

| Nside | one layout | pair | fits (56 GiB budget, 71.2 GiB pool) |
|---|---|---|---|
| 256 | 2.44 GiB | 4.87 | both layouts |
| 384 | 8.01 | 16.02 | both |
| 512 | 18.74 | 37.48 | **both** (was: one 37.46, pair impossible) |
| 640 | 36.31 | 72.63 | one layout (was: declined at 72.59) |
| 768 | 62.42 | 124.83 | declined — 62.4 + 16 GiB build reserve > pool |
| 1024 | 146.96 | 293.93 | **impossible on one card** |
| 2048 | 1163.9 | 2327.7 | impossible |
| 4096 | 9263.4 | 18526.9 | impossible |

The table is cubic in Nside and the halving extends the reachable range by only
`2^(1/3) = 1.26x`. **Nside 1024 and above cannot be made memory-fittable by any
further shrinking of a table that is stored at all** — the slice is
`(2L-1) x ntheta x L` and even one row per `(m, theta, ell)` triple is ~147 GiB at
1024. What would actually be needed, in the order I would try it:

1. **768 on one card**: admit the single 62.42 GiB layout by lowering `_BUILD_RESERVE`
   and re-chunking the build (the 6 GiB chunk target exists because smaller chunks
   fragment the pool — see the `_BUILD_CHUNK_TARGET` comment). Untested; synthesis
   would pay the 2.1x strided-axis penalty.
2. **1024 without a table**: a fused Wigner-d kernel that recomputes the slice per
   block (no resident table at any Nside). `memory/fused-spin-kernel-dead.md` records
   the Pallas attempt as disqualified on accuracy, and
   `memory/blocked-recurrence-wall.md` the blocked-recurrence restart problem — the
   gap is a *correct* fused kernel, not another tuning pass.
3. **1024+ with sharding**: the m-windows are independent, so the block sets shard
   across cards cleanly; two 96 GiB cards would put 1024 (147 GiB pair-split) within
   reach. Multi-GPU is already wired for the MCM (`_forward_latitudinal_device`).

Suite state at both commits: **115 passed, 3 skipped**; `flake8 --max-line-length=90`
clean on `gmaster/_spin_slice.py`.

## Session 8 (2026-09-03) — one contraction over four real channels; the large-Nside wall measured

One commit: `3ff71bc` — *perf(sht): contract each Wigner-d block once over four real
channels*. Session 7 left the polar latitudinal step doing **two** complex `einsum`
calls per block (direct + ring-reversed mirror). Both read the *same* block, so the
block was streamed from HBM twice at 717-745 GB/s. The fix folds the two complex
contractions and the real/imaginary split into **one** multiply-and-reduce over four
*real* right-hand sides:

```python
rhs = jnp.stack([direct.real.T, direct.imag.T, mirror.real.T, mirror.imag.T], -1)
acc = jnp.sum(block[..., None] * rhs[:, None, :, :], axis=2)   # (m, ell, 4)
```

Isolated n512 spin 2: analysis **51.9 → 28.9 ms (1.80×)**, synthesis **57.4 → 34.0 ms
(1.69×)**, `rel = 0.00e+00` — bit-identical to the shipped two-`einsum` form, and
bit-identical against a float128 oracle at n128. The suite is unchanged at
**115 passed, 3 skipped**, and `max|dCl|`/`rel` are the same digits at every cell.

### Scoreboard (TOTAL ms, GPU1, `ref->gm`, medians; `real_split_chain.log`)

| Nside | spin | NaMaster | GMaster now | ratio | was (session start) | rel (unchanged) |
|---|---|---|---|---|---|---|
| 128 | 0 | 20 | 13 | **1.5×** | 16 | 1.03e-13 |
| 128 | 2 | 41 | 23 | **1.7×** | 26 | 4.90e-09 |
| 256 | 0 | 59 | 47 | **1.2×** | 47 | 4.76e-13 |
| 256 | 2 | 161 | 94 | **1.7×** | 97 | 2.74e-08 |
| 512 | 0 | 340 | 289 | **1.2×** | 288 | 1.71e-12 |
| 512 | 2 | 713 | **536** | **1.33×** | 619 (1.14×) | 3.56e-07 |

n512 spin-2 `field` stage: 441 → **360 ms**. It is still the only stage where
GMaster trails NaMaster (360 vs 224); coupling/coupled_cell/decouple are 8-240× ahead.

**The `field` stage measured piece by piece (`field_split.log`, n512 spin 2).** An
earlier reading of this session claimed the two contractions only accounted for
217.6 ms of the 360 ms and that ~142 ms was somewhere else. **That was wrong** — it
used the best-case isolated contraction rates. Measured in situ:

| piece | per call | calls | total |
|---|---|---|---|
| analysis: ring transform + quadrature + phase shifts | 10.45 ms | 4 | 41.8 ms |
| analysis: **contraction** | 35.82 ms | 4 | 143.3 ms |
| analysis: finish (`sqrt((2l+1)/4pi)`, spin zeroing) | 0.09 ms | 4 | 0.4 ms |
| synthesis: prepare | 4.69 ms | 3 | 14.1 ms |
| synthesis: **contraction** | 51.45 ms | 3 | 154.4 ms |
| synthesis: inverse ring transform | 7.32 ms | 3 | 22.0 ms |
| **sum** | | | **375.8 ms** vs 360 ms benchmarked |

So `field` *is* the transforms (94 %); the ring/FFT/factor work is 23 ms (6 %), and
there is no hidden pool there. Two side findings from the same probes:

- **The `(m, theta, ell)` synthesis layout is still worth its memory.** Matched-clock
  A/B in one process (`layout_ab2.log`): synthesis **56.06 ms** on ell-contiguous
  against **83.99 ms** on theta-contiguous = **1.50x**, i.e. ~84 ms off a `field` stage
  for 18.74 GiB, so `want_ell` stays. `forward_latitudinal` accepts only the
  theta-contiguous layout (ell-contiguous raises a broadcast shape error), so a
  one-layout world is not available without writing a second forward form.
- **Probe variance on this operation is large and it is the cold-clock trap again**: the
  same forward contraction measured **28.91 ms** (session-8 probe, warm), **35.82 ms**
  in situ, **42.47 ms** in a cold single-purpose probe. Only end-to-end medians are
  quotable; the "85 % of the fp64 FMA floor" figure above is the *best observed* case,
  and in situ the contraction sits nearer 58 % of that floor — which is the one place
  left on this card where n512 spin 2 could still get faster.
- **The in-situ gap is not clock ramp.** `--repeats 7` under sustained load gives
  `field 216->358ms`, `TOTAL 536 ms` (`n512_rep7.log`) against `field 360`, `TOTAL 536`
  at repeats 3 — identical. The pipeline's 358 ms reproduces the sum of the separately
  measured pieces (375.8 ms), so the ~130-160 ms distance to the fp64 floor is a stable
  property of the shipped schedule (candidate: XLA tiling when the blocks arrive as jit
  arguments rather than constants — see the next bullet for what is already excluded),
  not measurement noise. That is the biggest remaining win below 1024: field 358 →
  ~220 ms would make n512 spin 2 ~1.9-2.0x overall.
- **Two explanations for that gap are already ruled out.** It is *not* fp32 inflation in
  the optimistic probe: `reduce_forms2.py` sets `jax.config.update("jax_enable_x64",
  True)` before importing `jnp` (line 19), so 28.91 ms was genuine fp64. It is *not*
  residency either: that probe called `S.slabs_for(...)`, i.e. the same pair-layout path
  the pipeline uses (the `18.74 GiB` in its header is one layout's byte count, not what
  was resident). Resolving the difference needs an HLO / profiler look at the two
  schedules, not another timing probe.

### Why `sum(block[..., None] * real_rhs)` beats `einsum` here

`einsum`/`matmul`/`dot_general` all lower to a tiled GEMM; with an `N = 1`-ish
complex operand XLA picks a tile shape that reads the shared block twice (once per
complex channel) and lands at **717-745 GB/s**. A real-valued RHS makes the same
expression a *fused reduction* — a different lowering — so the block is read once and
multiplied by 4 channels. Reference points on this card: pure fp64 row-read
**1603 GB/s**, bare `sum(block * rhs)` over the shipped n512 layout **1484 GB/s**,
shipped complex-RHS `einsum` **717-745 GB/s**.

The spin-0 path in `_theta_matrix.py` already used this real-channel form, which is
why scalar was never the deficit; only the polarised path was paying the tax.

### Where the fp64 ceiling is, and the gap to it

For the shipped n512 layout the 4-channel contraction is 1.0e10 FMA; at the measured
fp64 SIMT ceiling of **0.82 TFLOP/s (0.41 T FMA/s)** the floor is **24.5 ms** per
transform. The best isolated measurement reaches 28.9 ms (**85 % of the floor**), but
in situ the same calls run at 35.8 ms (analysis) and 51.5 ms (synthesis), i.e.
**~58 % of the floor**, and that gap — 126 ms of the 360 ms `field` stage — is the one
remaining place n512 spin 2 could get faster on this card. Whatever closes it must not
trade accuracy; going *past* the floor needs tensor cores / fp32-emulation (this card
has no fp64 tensor cores, dense fp64 GEMM tops out at 1.88 TFLOP/s) or fewer FLOPs.

### Things that measured the opposite of expectation

- **"The complex-RHS tax can only be removed by a custom kernel."** Wrong, and the
  standing memory said so. It was removed with `jnp.sum` on a real RHS.
- **Complex 2-channel fusion** (one pass, 2 complex channels instead of 4 real):
  **1.01×** — no read sharing without a real RHS.
- **Pad-and-concatenate epilogue** instead of the `at[].set` scatter chain: **worse**
  (35.7 vs 28.9 ms forward, 47.0 vs 34.0 ms inverse). The scatter chain is not the
  bottleneck.

### The large-Nside wall, measured rather than assumed

- **640 and 768 have no reference cell at all.** `NaMaster/pymaster/utils.py:266-272`
  (`NmtMapInfo.__init__`) doubles `nside` from 2 until `12*nside*nside == npix` and
  raises past 65536, so **pymaster accepts only power-of-two Nsides**. The session-7
  plan item "admit the single 62.42 GiB layout at 768 and benchmark it" is moot: the
  reference itself dies with `ValueError: Something is wrong with your input arrays`
  before GMaster is ever called (`n768_try.log`). Nothing to beat at 640/768.
- **Nside 1024, both spins, one card (`n1024_try.log`):**

  | spin | NaMaster | GMaster | ratio | field `ref->gm` | rel |
  |---|---|---|---|---|---|
  | 0 | 1760 | 2962 | **0.60×** | 511 → 1366 ms | 2.20e-12 |
  | 2 | 3294 | 87658 | **0.04×** | 899 → 85799 ms | 2.97e-06 |

  Accuracy holds at both spins; the loss is purely the transform path. At 1024 the
  polar table is **146.96 GiB** for one layout (294 GiB for the pair) against a
  **71.2 GiB** pool, so `slabs_for` declines it and the polarised path drops to the
  generic s2fft scatter loop — that loop, not the Wigner-d contraction, is what costs
  85.8 s. Spin 0 falls back to the fused Pallas kernel (`_MATRIX_BAND_BUDGET` is
  32 GiB vs a 77 GiB band) and lands at 0.6×.
- **PCIe streaming is not a workaround:** host RAM is 376 GiB (307 available), but a
  147 GiB table moves at ~16-25 GB/s → ~2.7 s *per transform*, 7 transforms per
  `NmtField(n_iter=3)`. NaMaster's whole field stage is 0.9 s.

### `_M_BLOCK` (the m-window width) is not a lever — swept

`_M_BLOCK = 64` has never been swept for the polar path: HANDOFF line 841 chose `mb=64`
for the **spin-0 band**, and the polar path inherited the constant. Swept it per Nside in
one process (matched clocks; every variant bit-identical to the `mb=64` control to
≤9.5e-17), timing 4 analysis + 3 synthesis contractions of one `field` stage:

| Nside | mb=32 | mb=64 | mb=128 | mb=256 |
|---|---|---|---|---|
| 256 | **52.2 ms** | 58.8 | 58.9 | 64.4 |
| 384 | 162.4 | **160.6** | 162.5 | 175.4 |
| 512 | 349.0 | **334.0 / 336.6** | build declined | build declined |

Narrower wins 11 % at 256, ties at 384, and **loses at 512**; wider is flat-to-worse and
costs bytes (mb=256 is +23 % bytes at n256, and `mb ≥ 128` at n512 exhausts the pool
during the build even in a fresh process). Nothing ships: a size-dependent width would
buy ~6 ms on the n256 cell while hurting the n512 cell by more. Probes
`.qwen/tmp/mblock_sweep.py`, logs `mblock_sweep.log`, `mblock_small.log`, `mblock_512.log`.

### Tiling the reduction axis is flat too (and it costs bit-identity)

Session 8's `reduce_forms.log` has `theta-chunked reduce 256/512` recorded as **FAIL**
(a shape bug), so the idea was never actually measured. Measured now
(`.qwen/tmp/tile_reduce.py`, n256, shipped call as the in-process control, ratio vs
shipped for 4 analysis + 3 synthesis):

| tile | analysis | synthesis | 4f+3i |
|---|---|---|---|
| — (shipped, one `jnp.sum`) | 8.52 ms | 8.45 ms | 59.4 ms |
| 64 | 9.14 (0.93×) | 9.08 (0.93×) | 63.8 (0.93×) |
| 128 | 9.06 (0.94×) | 7.45 (1.13×) | 58.6 (1.01×) |
| 256 | 9.89 (0.86×) | 7.19 (1.18×) | 61.1 (0.97×) |
| 512 | 7.39 (1.15×) | 8.91 (0.95×) | 56.3 (1.06×) |

No tile size helps both directions; the best aggregate (512, 1.06×) is inside run-to-run
spread, and every tiled variant differs from shipped by ~2e-16 because partial sums
reassociate — i.e. it would cost the bit-identity the shipped form currently has, for
nothing. **XLA's schedule for the shipped form cannot be beaten by reshaping or tiling
knobs**; closing the remaining distance to the fp64 floor needs a hand-written kernel.

### Streaming the table: pitched, measured, disqualified

The obvious idea for 1024 is that the bytes never all need to be resident: analysis is
a sum over `theta`, synthesis is disjoint in `theta`, and m-windows are independent, so
build one chunk, contract it into the output accumulator, drop it, next chunk. The
window arithmetic does fit (`triangle_bytes`, worst `m0 = 0` window of 64 orders):

| Nside | windows | one layout total | worst window, one layout | worst window, pair |
|---|---|---|---|---|
| 512 | 24 | 18.74 GiB | 1.50 GiB | 3.00 GiB |
| 1024 | 48 | 146.96 GiB | **6.00 GiB** | **12.00 GiB** |
| 2048 | 96 | 1163.86 GiB | 24.00 GiB | 47.99 GiB |
| 4096 | 192 | 9263.43 GiB | 95.99 GiB | 191.99 GiB (32-order windows → 96 GiB) |

**It dies on the march, not on the arithmetic.** `_build` recurses over `m`, so a chunk
of the table can only be produced by marching, and the march is nowhere near
bandwidth (`march_cost3.log`, one 6 GiB build chunk):

| Nside | chunk | rings/chunk | march per chunk | **effective rate** | full-table march |
|---|---|---|---|---|---|
| 512 | 6.0 GiB | 171 | 144.9 ms | **41 GB/s** | **1.74 s** |
| 1024 | 6.0 GiB | 43 | 171.8 ms | **35 GB/s** | **16.49 s** |

Against the 1603 GB/s row-read rate that is a **39x penalty**, and it is paid *per
march*. A resident table pays it once for the whole `field` stage; a stream pays it
seven times, because the refinement loop needs all alms before it can synthesise, so
each of the 7 transforms re-marches. At n1024 that is **7 x 16.5 s = 115 s** — worse
than the 85.8 s generic scatter loop we are trying to replace (which is doing the same
re-march, and is the reason it costs what it costs). The transpose that the build does
per window is not the problem: 0.2-0.5 ms per window-chunk, 0.1 s (n512) / 0.9 s
(n1024) all-inclusive.

The same numbers say why the resident table is the *only* shape that can win: at n512
the full march (1.74 s) is 8.6x the cost of all seven contractions (7 x 28.9 ms =
202 ms), so the build must stay cached and outside the timed region.

### What can still move 1024

1. **Two cards.** Windows are independent, so the layout shards; 1024 needs 73.5 GiB
   per card (half of 147 GiB) against a 71.2 GiB pool — 3% short, so a raised
   `gpu_memory_fraction` plus a free second card is the whole requirement. If the
   contraction then runs at its fp64 floor (FLOPs are cubic: 8 x 24.5 ms = 196 ms per
   transform, 7 transforms = 1.37 s) plus the 361 ms of coupling already measured at
   1024, the spin-2 field lands at **~1.7 s against NaMaster's 3.29 s ≈ 1.9x**. Not
   available now: GPU0 is another tenant at ~96.9 GiB.
2. **Tensor cores.** `fp64-roofline-wall.md` measures the ladder: no fp64 tensor cores
   on this card, dense fp64 GEMM tops out at 1.88 TFLOP/s, so Ozaki/DFP32-style
   fp32-pair emulation is the only remaining *massive* lever, and it must hold fp64
   accuracy to be usable at all (`dfp32-analysis-kernel.md` records the precision cliff
   the current DFP32 analysis kernel sits on).
3. **Fewer FLOPs by algorithm, not layout.** The table is already at its Z2-orbit
   minimum and the 4-channel contraction is at 85% of the fp64 FMA floor, so the only
   sub-linear route is a chirp-Z / factorised-`d` scheme of the libsharp family. That is
   a rewrite of the latitudinal step, not a tuning pass, and `blocked-recurrence-wall.md`
   and `fused-spin-kernel-dead.md` record two attempts at neighbouring ground that died
   on fp64 conditioning.

**Do not re-pitch a window or theta stream, a cleverer layout, or PCIe streaming as the
route to beating NaMaster above 512.** Capacity is the wall, the march is 39x too slow
to dodge it, and the arithmetic is already at the fp64 floor.

## Session 9 (2026-09-03) — the hand-written kernel measured 3x slower, and the tensor-core lever closed by arithmetic

Session 8 ended with one lever left open: *"closing the remaining distance to the fp64
floor needs a hand-written kernel."* It has now been written, validated and measured.
It loses. So does every alternative arithmetic path, and this session ends with the
first **quantitative** closure of the massive-speedup question rather than an
appeal to effort.

### The Pallas contraction kernel: correct, and 3x slower than XLA

`.qwen/tmp/cp3.py` (log `cp3b.log`) implements the windowed 4-channel contraction
directly — one block element loaded once, four register accumulators, tiled over the
reduce axis — and validates it against the shipped `jnp.sum` form in the same process.
Sweep: `x_block` 16..128, `y_block` 32..128, `num_warps` 4/8, `num_stages` 2..4.

| shape | plain fp64 sum | shipped 4-channel sum | best Pallas kernel |
|---|---|---|---|
| (64, 1536, 2047), 1535 MiB | 1.02 ms / 1572 GB/s | **2.30 ms / 699 GB/s** | 6.74 ms / 239 GB/s = **0.34×**, rel 6.6e-16 |
| (64, 1536, 343), 257 MiB | 0.21 ms / 1293 GB/s | 0.41 ms / 664 GB/s | 1.17 ms / 230 GB/s = **0.35×**, rel 4.0e-16 |

Every configuration lands in the 180-240 GB/s band. The first version kept a
`(x_block, y_block, 4)` product tile (64 KB of registers per program) and spilled;
splitting it into four independent `(x_block,)` accumulators fixed the spill and moved
the rate by nothing. Triton's fp64 reduction codegen on this card is simply ~3x behind
XLA's. **The kernel lever is closed: XLA's schedule is the best fp64 form reachable
here, from either side.**

Probe trap found on the way, worth remembering: a `pl.pallas_call` that is not jitted
re-traces on every timed call. A trivial read-scale-write kernel measured 8 GB/s
untraced and 1182 GB/s traced — the first number is host time, not kernel time.

### Alignment is not the limiter, and narrow-N cublas is dead again

The reduce axis is `ntheta = 4·Nside − 1 = 2047` at Nside 512 — odd, so with fp64 only
the first row of the table is 128 B aligned. Padding the reduce length costs 0.05 % of
bytes and buys (`align_probe.log`, same contraction, same bytes):

| reduce length | row offset | plain sum | 4-channel sum | `matmul` (N=4) |
|---|---|---|---|---|
| 2047 | 120 B | 1521 GB/s | 671.7 GB/s | 222.6 GB/s |
| 2048 | 0 B | 1549 GB/s | 679.3 GB/s | 219.6 GB/s |
| 2052 | 32 B | 1564 GB/s | 702.1 GB/s | 219.7 GB/s |
| 2064 | 0 B | 1570 GB/s | 704.2 GB/s | 222.9 GB/s |

Alignment is worth ≤4.8 % — real but tiny, and it would need the table builder changed
to hold it. cublas `matmul` sits at 222 GB/s at every N (agrees with the 4-channel sum
to 3.5e-15), repeating the narrow-N finding in `fp64-roofline-wall.md`.

### Three structural variants, each bit-identical, none of them faster

The shipped forward chains 48 functional `at[...].set(...)` updates over an
(L, 2L−1) complex array and takes the negative-order half with a negative-stride
gather per window. Both were rebuilt from the placement algebra (`out[ell, off±m]`,
`out[th, L±m]`, `m = 0` having no negative column) and measured against the shipped
function in-process at n512:

| variant | forward | inverse | rel |
|---|---|---|---|
| shipped | 42.58 ms | 58.23 ms | — |
| concatenate + single transpose, no scatter at all | 55.56 (**0.77×**) | 61.90 (**0.94×**) | 1.05e-15 |
| mirror arrays reversed and signed once per call, every window slice contiguous | 41.85 (**1.00×**) | 55.96 (**1.00×**) | **0.00e+00** |
| RHS materialised in its own jit (6.65 / 4.84 ms alone), contraction in a second | 41.93 (**1.00×**) | 55.90 (**1.00×**) | **0.00e+00** |

Two of them reproduce the shipped output **bit for bit**, which also confirms the
placement algebra in `forward_latitudinal`/`inverse_latitudinal` is exactly as derived.
The scatter chain is already fused into the producers; the gathers and the producer
fusion cost nothing.

For reference, summing the raw contraction over all 24 windows with the RHS already in
hand gives 29.99 ms (analysis shapes) and 31.01 ms (synthesis shapes) — **1.03×**, so
the direction asymmetry seen in situ is not the block shape either
(`dir_split.log`). What is left between that and the in-situ 42/56 ms is XLA's schedule
for the fused producer-plus-reduce, and every tool tried on it (kernel, tiling, layout,
window width, scatter removal, materialisation barrier) now measures flat or worse.

### The massive lever, closed by counting passes instead of by taste

`narrow_tc.log` and `narrow_n4.log` measure the same M=98304, K=2047 operand under
tensor-core matmuls at the N our contraction actually has (4 channels):

| dtype | N=4 | N=8 | N=16 | N=32 | A+B bandwidth |
|---|---|---|---|---|---|
| bf16 | 0.310 ms | 0.300 | 0.301 | 0.301 | ~1.34 TB/s |
| fp16 | 0.298 ms | 0.295 | 0.300 | 0.304 | ~1.35 TB/s |
| fp32 | 0.551 ms | 0.546 | 0.550 | 0.551 | ~1.46 TB/s |
| int8 | 0.408 ms | 0.500 | 0.829 | 1.644 | 493 → 122 GB/s |

bf16/fp16/fp32 time is **flat in N**: at N=4 the tensor cores are not the constraint,
the operand stream is, and it streams at 1.35 TB/s. int8 is compute-bound at ~7.8 T
flop/s and *slower* per byte than fp64. That turns the emulation question into a
pass count with a hard break-even:

- shipped fp64 4-channel contraction of that block: **2.30 ms**;
- one bf16 pass over it: **0.31 ms** → **break-even = 7.4 limb products**.

Limb counts required for fp64-class accuracy, at 8 bits (bf16) or 7 bits (int8) per
limb: ~1e-13 needs 44 bits ⇒ 10 bf16 products (4 limbs) or 28 int8 products — **all
beyond break-even, so all slower than plain fp64**. The only splits that fit inside
7.4 passes are 2-3 limbs (16-24 bits), which represent the table to 1.5e-5 and 6e-8
respectively: the first is 2 orders worse than the shipped agreement with NaMaster
(3.56e-07 at n512), the second reaches it only by consuming the whole budget and still
lands at 1.24× at best. **A limb scheme at N=4 cannot both beat fp64 and stay accurate
on this card. "Massively faster" is arithmetic here, not an engineering gap.**

### What that leaves, stated honestly

At Nside ≤ 512 both spins beat NaMaster (1.2-1.7× through 512, n512 spin 2 at 1.33×)
with the contraction at 83-85 % of the fp64 SIMT FMA ceiling, and no reachable form —
XLA, Triton, cublas, or tensor-core emulation — improves on it without spending
accuracy the objective does not allow. Above 512 the wall is capacity: one layout is
146.96 GiB at 1024 against a 71.2 GiB pool, the march needed to dodge it is 39× off
bandwidth, PCIe is 60× off it, and 640/768 have no NaMaster reference to beat at all.
The two things that could still move the scoreboard are outside this box: a second
free ≥96 GiB card (windows shard, 73.5 GiB per card, ~1.9× projected at 1024), or a
chirp-Z/factorised-`d` latitudinal rewrite that reduces the FLOP count itself.

## Session 10 (2026-09-03) — the channel axis of the reduce was the defect all along

Session 9 closed the polar contraction as "at 83-85 % of the fp64 FMA floor, six forms
tried, all flat or worse". It had tried six forms of the *right-hand side* and of the
scatter epilogue. It never changed the shape of the **reduce** itself, and that is where
the loss was. Everywhere in the theta path the contraction was written as

```python
jnp.sum(block[..., None] * rhs[:, None, :, :], axis=2)   # channels AFTER the reduced axis
```

Leaving the output channels in a dimension behind the axis being reduced makes XLA
materialise the whole product instead of folding the multiply into the reduction, so the
stage pays a write plus a read of its own product on top of the table read. The fix is
one `lax.reduce` over a tuple of accumulators — the table is read once and nothing wide
is ever materialised:

```python
re, im = lax.reduce((slab * chan[..., 0], slab * chan[..., 1]), (0.0, 0.0),
                    lambda a, b: (a[0] + b[0], a[1] + b[1]), (2,))
```

### The three wins, each measured inside the shipped program

| site | shipped | one-pass | gain | rel | log |
|---|---|---|---|---|---|
| `_theta_matrix._transform` (scalar analysis) | 12.16 ms / 823 GB/s | 8.11 ms / 1238 GB/s | **1.50×** | 2.5e-16 | `lat_var.log` |
| `_theta_matrix._inverse` (scalar synthesis) | 22.55 ms / 446 GB/s | 16.78 ms / 600 GB/s | **1.34×** | 3.1e-16 | `synth_ab.log` |
| `_spin_slice.forward_latitudinal` + `inverse_latitudinal` (4 channels) | field 64.9 ms | field 47.7 ms @ n256 | **1.36×** | 5.4e-16 | `spin_ab256.log` |

All three are interleaved-rounds medians with an honest `jax.random.normal` bandwidth
control beside every line (1389-1459 GB/s). Landed as `_contract_theta` /
`_contract_ell` (`8880401`) and `_spin_slice._reduce_channels`.

### Scoreboard (GPU1, `ref->gm` ms, clocks provably up)

`.qwen/tmp/score_after.log` is the scalar half alone (`8880401`); `.qwen/tmp/score_spin.log`
is both halves shipped (`c80263b`). `field` is the stage both fixes act on.

| cell | session-9 baseline | scalar half | both halves | `field` trajectory |
|---|---|---|---|---|
| n128 spin 0 | 1.25× (19→15) | 1.41× (21→15) | 1.18× (21→18) | 7→6→9 ms (±3 ms host noise) |
| n128 spin 2 | 2.26× (50→22) | 2.13× (46→21) | 1.87× (43→23) | 14→13→12 ms |
| n256 spin 0 | 1.27× (61→48) | **1.53×** (65→43) | **1.54×** (64→42) | 0.95× → **1.30×/1.18×** (20→17 ms) |
| n256 spin 2 | 1.83× (172→94) | 1.87× (175→94) | **2.26×** (168→74) | 0.68× → 0.67× → **0.92×** (66→48 ms) |
| n512 spin 0 | 1.20× (351→291) | **1.54×** (343→222) | **1.56×** (347→222) | 0.92× → **1.24×/1.28×** (118→92 ms) |
| n512 spin 2 | 1.35× (722→537) | 1.45× (730→504) | **1.75×** (742→424) | 0.64× → 0.67× → **0.85×** (239→281 ms) |

The n128 rows move by 3 ms between processes on a 15-22 ms total, which is host-side
compile/launch noise, not a regression: the two spin-0 n128 runs bracket 15/18 ms with
identical code and both controls at ~1400 GB/s. Everything from n256 up is stable to
±1 ms across runs.

`rel` vs NaMaster is byte-identical to the session-8/9 values at every cell
(1.03e-13 / 4.90e-09 / 4.76e-13 / 2.74e-08 / 1.71e-12 / 3.56e-07) and the suite is
**115 passed, 3 skipped** on both commits.

**Session-9 conclusions this revises:** "no reachable form — XLA, Triton, cublas, or
tensor-core emulation — improves on [the polar contraction]" (end of session 9) and the
memory note "the remaining 15 % is not recoverable — measured six ways". Six ways were
tried against the RHS producers and the scatter epilogue; the reduce's own channel axis
was never tried and was worth 1.36×. Session 9's *measurements* all stand — the listed
forms really are flat or worse — only the universality of the conclusion was wrong.

### The trap that hid it: an A/B of a patched function needs the jit cache dropped

`utils._forward_latitudinal_slab` is itself `jax.jit` and resolves
`_spin_slice.forward_latitudinal` from module globals at *trace* time, so swapping the
module attribute leaves its cache key untouched — the second form times the first form's
compiled HLO and the A/B reports 1.00×. `.qwen/tmp/spin_ab.py` calls
`_forward_latitudinal_slab.clear_cache()` / `_inverse_latitudinal_slab.clear_cache()` on
every swap and prints a trace counter (2 traces per form, which is also the proof the
patch is on the path).

### Probe hygiene, three more ways to be wrong

- **Unit bug that manufactured a fake defect:** `band_rate.py::measure` returned
  `np.median(ts) * 1e3` where `ts` was already ms, printing "the band runs at 1.6 GB/s"
  for what is really 3.50 ms / 1577 GB/s. An entire detour into a nonexistent
  memory-path pathology. Convert to ms exactly once, at the print.
- **Ranking is program-dependent:** a per-block `jax.jit` says
  `einsum("emj,mjc->emc")` is 4× *faster* than the broadcast product (882 vs 205 GB/s);
  inside the shipped whole-band program the same einsum is 4× *slower* (203 vs 817),
  and patching it in scores 0.5× end to end (`lat_pipe.log`). Isolated probes generate
  hypotheses; only in-situ A/Bs rank.
- **The stock harness parks the GPU:** `benchmarks/benchmark_pipeline.py` runs the CPU
  reference first and GMaster second, and this card idles at 180 MHz needing ~0.5 s of
  continuous load. The error is one-sided against GMaster (same code: `field` 51 ms in a
  fresh process, 364 ms inside a sweep). `.qwen/tmp/scoreboard.py` wraps the same
  `_run_pipeline` with a spin-up and a bandwidth control before/after each GMaster run —
  use it for any number that will be quoted.

### What the win does *not* buy

The capacity wall is unchanged: the scalar band is 77 GiB at Nside 1024 against a
71.2 GiB pool, and the polar table is 146.96 GiB per layout, so neither is storable
there regardless of how fast the contraction is. Faster contraction changes the *value*
of solving the capacity problem, not the capacity itself.

Measured live at this HEAD (`.qwen/tmp/score_one_1024s0.log`, fresh process, repeats 1):
**n1024 spin 0 = 0.61× (1806 → 2976 ms), `field` 512 → 1372 ms (0.37×), rel 2.20e-12** —
indistinguishable from session 8's 0.60× / 511 → 1366 ms, as expected: at 1024 the band
is declined (77 GiB > 32 GiB budget) and the fused Pallas kernel runs, which neither
commit touches. No regression, no gain. The arithmetic for what a resident band *would*
buy there: 7 passes over 77 GiB at the now-measured 1238 GB/s ≈ 455 ms of `field`
against NaMaster's 512 ms — so even solving the capacity problem only reaches ~1.1× at
1024, not a massive win.

(Careful with the 600 GB/s figure for the synthesis band quoted above: it comes from
dividing the *whole* `alm2map` stage — FFTs, mask algebra, residual included — by the
band's byte count, so it is a floor on the contraction's rate, not a measurement of it.
The clean contraction numbers in this session's table are 1238 GB/s analysis and the
16.78 ms synthesis stage.)

### The number that governs every large-Nside plan: tables are produced 54x slower than they are read

`.qwen/tmp/slab_build_rate.py` measures how fast `_spin_slice.slabs_for` can *make* a
Wigner-d slab set (clocks spun up, bandwidth control printed beside every line):

| Nside | pair bytes | cold build (incl. compile) | warm rebuild (cached compiles) |
|---|---|---|---|
| 128 | 0.65 GiB | 1463 ms (0.5 GB/s) | 29 ms → **24.1 GB/s** |
| 256 | 4.87 GiB | 3917 ms (1.3 GB/s) | 228 ms → **22.9 GB/s** |
| 512 | 37.48 GiB | 10834 ms (3.7 GB/s) | (probe bug, see below) |

The warm production rate is ~23 GB/s and size-independent, while this session measured
the *consumption* rate of the same bytes at 1238 GB/s. **A produced table therefore has
to be read ~54 times to pay for itself.** A `NmtField(..., n_iter=3)` reads it 7 times.
That single ratio is why:

- **n1024 spin 2 cannot be rescued by streaming.** Producing the 146.96 GiB analysis
  layout takes ~6.4 s at 23 GB/s; build-a-chunk/contract/free would pay that per
  transform (7x), i.e. ~45 s, against the 85.8 s the generic scatter loop costs today —
  the same order, which is exactly what the observed 85.8 s is (7 x ~12 s of production).
- **The scalar band at 1024 cannot be streamed either.** Its build rate is ~90 GB/s
  (session 8: 97-166 ms for 9.38 GiB), so a chunk costs ~0.86 s to build against 65 ms
  to contract — build dominates by 13x. Full residency is the only form that wins, and
  77 GiB does not fit a 71.2 GiB pool.
- **The march itself is ~40x off the arithmetic floor** (23 GB/s / 8 B = 2.9 G elem/s at
  a few FMAs each ≈ 10 GFLOP/s against 0.41 T FMA/s), so it is structure/latency-bound.
  Making the march faster is a legitimate target, but even 10x only reaches ~1 GFLOP-
  class parity, and the operator's O(L^2 * ntheta) flop count is what caps n1024 anyway.

Probe bug worth noting: the script's `rebuild` leg calls `SS.clear_cache()` and then
`slabs_for` again while the first pair is still bound, so at n512 the cache decline
kicks in and `pair` comes back `None`. Delete the pair (and let the pool settle) before
timing a rebuild.

**Consequence for the standing objective.** Up to Nside 512 both spins now beat NaMaster
by 1.5-2.3x and the dominant stages are at the fp64 SIMT floor. At Nside >= 1024 the
binding constraints are capacity (77 GiB band / 147 GiB polar layout vs a 71.2 GiB pool),
production rate (23-90 GB/s vs 1238 GB/s consumption) and — for anything "massive" — the
operator's O(L^2 * ntheta) flop count, which is the same order NaMaster spends. Beating
NaMaster by a large factor there needs a different operator (chirp-Z / factorised-Wigner-d
class), or a second large card to shard the tables; it is not reachable by scheduling the
existing one faster.

### Two hypotheses tested and closed in the same session

- **The recurring `bfc_allocator ran out of memory ... 1.19 GiB` warning at n512 spin 2
  is not costing time.** It appears in every multi-cell run, and the obvious theory was
  that the spin-0 cells' scalar band + synth copy (18.7 GiB) were still resident while
  the polar slab pair was being built, forcing eviction traffic inside the `field`
  stage. `.qwen/tmp/score_one.py` runs a single `(nside, spin)` cell in a fresh process
  that cached nothing: **n512 spin 2 gives 1.72× / field 279 ms fresh vs 1.75× / field
  281 ms in the multi-cell run.** No effect — do not add a `_theta_matrix.release()`
  call to the polar path for performance reasons; the warning is benign.
- **The polar contraction is now provably at the fp64 SIMT floor, so the `field` stage
  cannot go faster in fp64.** After `c80263b` the stage decomposes as 4 analysis + 3
  synthesis passes; scaling session 9's in-situ times by the measured 1.36× gives
  ~30.7 ms analysis and ~41 ms synthesis, i.e. ~252 ms of the measured 279 ms is the
  contraction. One analysis pass reads the 18.74 GiB layout = 2.5e9 elements × 4 FMAs in
  30.7 ms = **335 G FMA/s against the card's measured 410 G FMA/s fp64 SIMT ceiling
  (82 %)**. Session 9's floor arithmetic was right; only its claim that nothing could
  reach it was wrong. Getting below needs fewer FMAs per element (the `pi - theta`
  identity is already spent, the other symmetries measured unavailable) or faster
  arithmetic, and both tensor-core and limb-emulation routes are closed by
  `fp64-roofline-wall.md` (N=4 is operand-stream-bound; 7.4-pass break-even vs ≥10
  products for fp64-class accuracy).
- **The last "losing stage" was a measurement artefact too.** `decouple` reads
  0.11-0.30x for spin 0 on the scoreboard, which looks like the one stage GMaster
  genuinely loses. `.qwen/tmp/decouple_probe.py` times the same call in one process,
  clocks spun up before every rep, 40 reps, next to its pieces:

  | cell | NaMaster | GMaster `decouple_cell` | ratio | pure `jnp.linalg.solve` | precomputed-inverse matvec |
  |---|---|---|---|---|---|
  | n256 spin 0 (MCM 25×25) | 1033 µs | 991 µs | **1.04×** | 324 µs | 187 µs |
  | n512 spin 0 (MCM 51×51) | 1380 µs | 1045 µs | **1.32×** | 418 µs | 271 µs |

  `rel` is identical for both solve paths (4.76e-13 / 1.71e-12), so an explicit inverse
  would be admissible — but it only buys ~150 µs while the wrapper (validation +
  `bins.bin_cell` + reshape) spends ~670 µs of the 991 µs. If this stage is ever
  revisited the target is the wrapper (fuse `bin_cell` and the solve into one jitted
  function), not `solve` → `inv`; at ~1 ms inside a 42-222 ms total it was left alone.

**General lesson on stage ratios:** anything the scoreboard prints as `0->0ms` is being
measured at a scale where the harness's ordering (CPU reference first, GPU parked for
GMaster's turn, no inter-rep keep-alive) dominates the number. Do not read ratios off
sub-millisecond stage figures — re-measure them interleaved with the clocks held up.

(Probe caveat: `decouple_probe.py` rebuilds the binned RHS as `bins.bin_cell(cl).T.reshape(-1)`,
which matches `decouple_cell` only for `ncls == 1`. At n512 spin 2 (ncls=4) its `A`/`B`
rows therefore come out at rel 0.98 — a probe bug, not a library defect; the library call
itself is correct there (rel 3.56e-07, 1.08x vs NaMaster). The spin-0 rows are sound:
their `A`/`B` relative errors match the library's to the printed digit.)

### Where that leaves the objective

Measured honestly, every stage at every Nside up to 512 in both spins is ≥ 1× against
NaMaster, with `field` at 0.85-1.30× and the MCM stages at 1.2-200×. The residual spin-2
`field` deficit is arithmetic (82 % of the fp64 SIMT ceiling), and Nside ≥ 1024 is bound
by capacity and by table-production rate, both measured above. The remaining large levers
are therefore not scheduling: a second ≥96 GiB card to shard the tables, or a
lower-FLOP latitudinal operator (chirp-Z / factorised-`d` class).

## Session 11 (2026-09-04) — Table storage precision, and the memory failures that ship with it

**Shipped `nmt.set_table_precision("fp64" | "fp32")`.** It changes only the *device
storage width* of the precomputed transform tables — the Legendre band
(`_theta_matrix`) and the polar Wigner-d layouts (`_spin_slice`). The recurrence that
generates the values stays float64 and every contraction still accumulates in float64
(`_contract_theta` upcasts the field channels, not the slabs), so the price is the
table's own representation error: ~1e-7 relative on the coupling matrix, measured. It
is opt-in and never inferred; `"fp64"` remains the default and every earlier number in
this file was taken with it. `tests/test_table_precision.py` pins both widths.

What it buys is **engagement**, not just bytes: the large geometries are dispatched by
a fit test, so a geometry that declined the tables and took the recurrence-bound fused
kernel can take the memory-bound contraction instead. Band sizes: Nside 512 9.4 GiB
(fp64) / 4.7 (fp32); Nside 1024 73.5 / 36.8 GiB. `_MATRIX_BAND_BUDGET` went 32 → 40
GiB, which is exactly what lets the float32 Nside 1024 band land on this box's default
71.2 GiB pool.

Measured at Nside 512 (`.qwen/tmp/score_n512_fixed.log`, clock-controlled, both spins
against pymaster in the same process):

| cell | NaMaster | GMaster | speedup | rel |
|---|---|---|---|---|
| spin-0 fp64 | 347.1 ms | 220.6 ms | **1.57x** | 1.7e-12 |
| spin-0 fp32 | 347.1 ms | 157.6 ms | **2.20x** | 6.5e-09 |

**Re-measured at the end of Session 12 on the shipped code** (`.qwen/tmp/score_n512_final.log`, all
four cells in one process, budget 40 GiB, default 71.2 GiB pool): spin-0 fp64 348.0 / 222.2 =
**1.57x** (rel 1.7e-12); spin-0 fp32 348.0 / 157.2 = **2.21x** (rel 6.5e-09); spin-2 fp64
657.1 / 425.8 = **1.54x** (rel 3.6e-07, polar pair resident at 18.7 + 18.7 GiB, control 1380/1342).
The GMaster milliseconds are unchanged from the earlier measurement (220.6 / 157.6 / 422.1): the
spin-2 ratio moved only because `pymaster` needed 657 ms here instead of 707.8 ms there, and the
reference varies ~8% run to run under load. Quote the GMaster time alongside any ratio.
spin-2 fp32 in the same run was 657.1 / 411.1 = **1.60x** (rel 3.6e-07), i.e. float32 tables are
*marginally faster* here rather than the 1.39x regression recorded earlier (523.4 ms in
`.qwen/tmp/score_spin2_polarfix.log`) — the polar contraction is no longer the loser under
narrower storage that it used to be.

**Three memory failure modes found and fixed, all of them silent before:**

1. **A cached XLA executable keeps the buffers it produced alive.** A table built as
   many separate jits therefore costs roughly twice its bytes, and the OOM lands on an
   allocation that looks far too small for the table. Both builders now call
   `jax.clear_caches()` every `_BUILD_CLEAR_EVERY = 8` chunks, and `_synth_band` clears
   before its final concatenate.
2. **`_synth_band` asked for a second, re-laid-out copy for synthesis and died instead
   of declining.** It now tests `2 * nbytes + 16 GiB > pool_bytes()` up front and takes
   the strided reduction on the analysis layout when that fails. Measured bit-identical
   (rel 0.0), so a synthesis-only caller pays the strided reduction rather than losing
   the band.
3. **Refusals were not remembered.** `_theta_matrix._BAND_FAILED` and
   `_spin_slice._BUILD_FAILED` are now negative caches: a geometry that failed is asked
   about once per process, then stays declined without a second OOM-and-retry (which is
   what made a failed run tens of minutes rather than minutes). Cleared by
   `release()` / `clear_cache()` / `set_table_precision()`.

**Do not try to widen the pool.** `XLA_PYTHON_CLIENT_MEM_FRACTION` is a trap at these
sizes — the CUDA context and cuBLAS workspaces live *outside* the pool, so a bigger
pool starves them and the table build fails in a pool that has more free bytes. Same
process shape, budget 20, Nside 1024 float32:

| pool | GMaster | speedup | band |
|---|---|---|---|
| default 71.2 GiB | **898.2 ms** | **2.00x** | built, resident 38.0 GiB, peak 43.3 |
| 0.95 → 90.2 GiB | 1793.4 ms | 1.00x | `bands cached: 0, failed: [(1024, 3072, 64, float32)]`, peak 16.9 |
| 0.98 → 93.0 GiB | 1794.1 ms | 0.02x (other cells) | same refusal, peak 14.0 |

The headline of the session, then, is **Nside 1024 spin-0 fp32 = 1.99–2.00x against
NaMaster CPU with no environment variable at all**: `.qwen/tmp/score_n1024_fp32_defaultpool.log`
is a fresh process with only `CUDA_VISIBLE_DEVICES=1`, `nmt.set_table_precision("fp32")`,
NaMaster 1791.1 ms / GMaster 900.6 ms, rel 6.3e-08, control 1354/1429 GB/s, one band
cached, `synth layouts: {}` (the second copy declined, so the strided reduction runs),
warm-up including the build 389.4 s.

## Session 12 (2026-09-04) — Nside 1024 spin-2: one route, 73.5 GiB, and the pool cannot assemble it

**Attribution first (`.qwen/tmp/stage_n1024_spin2.log`).** Of GMaster's end-to-end time
at Nside 1024 spin-2, 99.6% is the field stage: 964.4 ms (NaMaster) vs 85 504.9 ms
(0.01x), while the coupling matrix is 4.96x and the coupled cell 220.5x faster than the
reference. There is nothing else to optimize — the deficit *is* the latitudinal spin-2
transform running without a table, i.e. s2fft's generic scatter loop.

The only route to a fast spin-2 latitudinal step in this codebase is the precomputed
Wigner-d layout, and at Nside 1024 that layout is `(mmax+1, ntheta, L)` = **73.48 GiB in
float32**. The gate is `tri > _SLAB_BUDGET`, then `tri + _BUILD_RESERVE <= live pool
headroom`. Every way of moving those numbers was tried and measured; every one ends in
a single `RESOURCE_EXHAUSTED` of ~2.7 GiB — exactly one joined window block — with the
peak at 86–87 GiB of a 90.2 GiB pool and `largest_free_block_bytes: 0`:

- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` and `0.98`; `POLAR_RESERVE_GIB=16, 10, 6`.
- `TF_GPU_ALLOCATOR=cuda_malloc_async` — 0.04x, same refusal (so it is not fragmentation).
- `PREALLOCATE=false` — same refusal.

What *was* shipped is correct on its own merits and makes the build strictly cheaper:
`_spin_slice._build` marched the Wigner-d recurrence over **both** m-halves and returned
`(2L-1, ntheta, L)`, of which the `L-1` negative-order rows were never read —
`_build_triangles` slices `[:, :, L-1:]` and the pi-theta identity
(`T[m, pi-theta, l] = (-1)**(l-spin) T[-m, theta, l]`, m=0 excepted) is why only m >= 0
is kept at all. It now returns `(L, ntheta, L)`: half the march work and a chunk table
of 3.02 GiB instead of 6.05 at this size. The float32 cast is fused into the transpose
(one copy per window instead of two). `tests/test_sht.py`: 13 passed, 3 skipped.

**Verdict.** The assembled layout would fit a 90 GiB pool; assembling it does not,
because the per-chunk pieces and the join output have to exist at the same time and the
build peak lands ~13 GiB above the layout. Nside 1024 spin-2 needs either a fused kernel
that passes the accuracy test (the existing one loses precision through catastrophic
cancellation as |m| -> l, which is why `spin != 0` returns False in `_use_pallas_sht`)
or a layout that is structurally smaller. Budget, reserve, allocator and prealloc knobs
are exhausted — do not spend another session on them.

Measured spin-2 state on this code: **Nside 512 fp64 1.68x**
(`.qwen/tmp/final_score_512_folded.log`), **Nside 512 fp32 1.39x**
(`.qwen/tmp/score_spin2_polarfix.log`), Nside 1024 0.04x with no resident table. The
current scoreboard at Nside 1024 is therefore spin-0 **1.99–2.00x** (fp32) and spin-2
**not beating NaMaster**; there is no measured Nside 1024 spin-2 win, and there is no
measured Nside 2048 win of any kind (both table sizes decline there, and `pymaster`
itself needs 40 min for 50 ms of work at Nside 4096).
measured Nside 2048 win of any kind (both table sizes decline there, and `pymaster`
itself needs 40 min for 50 ms of work at Nside 4096).

**The one route that is still open for Nside 1024 spin-2, with the arithmetic done.**
Every in-process way to make room has failed because *building* the layout costs ~13 GiB
more than *holding* it. So build it out-of-process and load it:

1. A separate process marches the triangles chunk by chunk and writes each finished
   window block as raw float32 to a cache file (`.gmaster-cache/<nside>-<L>-<spin>-<dtype>.bin`).
   Peak there is one chunk (3.02 GiB) plus one window block — the same shape that already
   fits at Nside 512 — and `/scratch` has 745 GB free against the 73.5 GiB file.
2. The serving process memory-maps (or `read`s) the file into **one** contiguous device
   buffer and installs it in `_spin_slice._CACHE` directly, bypassing `_build_triangles`
   entirely. A single 73.5 GiB host->device copy needs no join and no cuBLAS workspace, so
   the `XLA_PYTHON_CLIENT_MEM_FRACTION` breakage that kills the *build* does not apply to
   a *loader*; the pool at 0.95 is 90.2 GiB and the layout is 73.48 GiB.
3. Expected payoff, scaled from the measured Nside 512 spin-2 field stage (281 ms for
   seven polar passes, `.qwen/tmp/final_score_512_folded.log`) by the cubic in Nside:
   ~2.25 s against NaMaster's 3.41 s total, i.e. **~1.4x**, with the coupling matrix and
   coupled cell already 5x and 220x ahead. That is the honest ceiling of this route — the
   fp32 polar contraction is *not* faster than fp64 (n512 spin-2 went 1.68x -> 1.39x with
   fp32 tables), so do not expect the 2x that the scalar band got.
Costs to weigh before starting: ~3-4 min one-off build, ~40 s load per process, a 73.5 GiB
file per geometry, and a cache-invalidation key (nside, L, spin, dtype *and* the revision
of `_spin_slice._build`).
file per geometry, and a cache-invalidation key (nside, L, spin, dtype *and* the revision
of `_spin_slice._build`).

**Verification of everything shipped in Sessions 11-12** (GPU1, `CUDA_VISIBLE_DEVICES=1`):
full suite **124 passed, 3 skipped in 304.46 s** (`.qwen/tmp/suite_final.log`) — the three
skips are the two-GPU tests at `tests/test_sht.py:301,326`, and the run includes the six new
tests in `tests/test_table_precision.py`, including `test_float32_tables_keep_the_coupling_matrix`
at spin 0 and spin 2. No test-count change from the pre-session baseline.

**One comment-level fabrication was found and removed while reviewing the diff for the
commit message**: the `_BUILD_RESERVE` comment claimed the 73.5 GiB layout "builds in a
90.2 GiB pool and runs the pipeline in 2.2 s". No log contains such a cell — `grep` over
every `.qwen/tmp/*.log` row for Nside 1024 spin 2 returns only 0.02-0.04x
(87 420-87 532 ms, and one 142 456 ms). The comment now states the measured refusals. If a
future session sees a claim about a fast Nside 1024 spin-2 pipeline, it is wrong until a log
row proves it.

## Session 13 (2026-09-04) — the isolated SHT against the DUCC C code: the table route is at the bandwidth floor, the chirp-Z was transforming a mirror image

**The question this session existed to answer.** Every previous scoreboard is the *whole*
MASTER pipeline, which mixes the transform with the coupling matrix and the deprojection and
therefore cannot answer "is our SHT faster than the ducc0 C code NaMaster calls".
`.qwen/tmp/sht_vs_ducc.py` times one `map2alm` and one `alm2map` on each side — ducc driven
through NaMaster's own helper (`pymaster.utils._map2alm_ducc0` / `_alm2map_ducc0` with
`_ducc_kwargs` = `theta/nphi/phi0/ringstart/lmax/mmax/mstart/nthreads: 0`, ducc0 0.39.1 on all
192 cores), GMaster through `utils.map2alm(..., n_iter=0)` / `utils.alm2map`, `L = 3·nside−1`,
clocks spun up before every round and a `jnp.sum` bandwidth control printed on every line
(1330-1440 GB/s in these runs).

**Stage split (`.qwen/tmp/sht_stage_split.py`) is the finding that redirected the session.**
Timing the latitudinal contraction alone against a pure `jnp.sum` over the same table slab:

| | nside 512 | nside 1024 |
|---|---|---|
| band read floor (`sum` over one slab) | 1488 GB/s | 1488 GB/s |
| theta analysis | 3.67 ms (1369 GB/s) | 26.18 ms (**1506 GB/s**) |
| theta synthesis | 4.19 ms (1200 GB/s) | 28.60 ms (1379 GB/s) |
| everything else (azimuthal ring CZT + ring↔pixel scatter) | 3.5 / 7.4 ms | 15.1 / 27.9 ms |

The precomputed-Legendre contraction is **already streaming at the card's floor** — 1506 GB/s
against a 1488 GB/s `sum`. There is nothing left to win in the table route, which retires the
"effective bandwidth of the whole call is only 939 GB/s, close it for 1.4x" reading from
Session 11-12: the missing time is in the *azimuthal* stage, not in the table.

**Win 1 — the synthesis chirp-Z was transforming an array it already knew.**
`_inverse_ring_fft` mirrored the positive-m block into a centred `2L`-wide window and
chirp-Z-transformed the whole thing, so its convolution bound was `N ≥ 2L−1+width` = 8192 at
Nside 512 and **16384** at 1024, twice the analysis side's 4096/8192. The mirror is exact by
construction, so with `P(p) = Σ_{m<L} F_m e^{2πipm/nphi}`,

    Σ_m full[m] e^{2πipm/nphi} = 2 P(p) − F_0        (real part)

which needs only the bound `L + width − 1` — 4096 at 512, 8192 at 1024. `utils._inverse_ring_fft_herm`
implements that and also drops `wrap_phase` entirely (the centred form's bookkeeping for the
`L`-sample index shift disappears because the new form extracts the window at `L−1+p` instead
of `2L−1+p`). Measured **2.07x on the stage at Nside 512** and identical to the old form to
**1.3e-15** (`tests/test_sht.py::test_ring_synthesis_from_positive_half_matches_centred_window`,
Nside 32/64); the transform size halves at every size (1024→512, 2048→1024, 4096→2048,
8192→4096). `_inverse_ring_fft` is kept as the equivalence baseline — nothing in the shipped
path calls it.

**Win 2 — the analysis side built a Hermitian window and immediately threw half of it away.**
`_map2alm_once_pallas` called the filling ring FFT and sliced `[:, L:]` straight back off it.
`utils._forward_ring_fft_positive` returns that block directly and `_forward_ring_fft` is now
that call plus the fill. Stage speedup is small (**1.02 / 1.03 / 1.05x** at 256 / 512 / 1024,
`.qwen/tmp/ring_forward_ab.log`, sliced-vs-block difference exactly 0.0) and the pipeline does
not move — it is worth landing for the 100 MB (Nside 512) / 400 MB (Nside 1024) of complex
buffer it stops materialising, not for the clock.

**Isolated SHT scoreboard, fp32 tables, both ring fixes in.** GPU1, `CUDA_VISIBLE_DEVICES=1`,
one Nside per process, min over 3-7 reps. The ducc columns are the **45 clean baseline runs**
of `.qwen/tmp/ducc_only.py` (5 processes × 9 reps per size, **no JAX and no CUDA context on the
box**): best-of-45 first, then the median-of-process-medians in brackets.

| nside | dir | ducc best (median) ms | GMaster ms | ratio vs best | ratio vs median |
|---|---|---|---|---|---|
| 256 | map2alm | 2.12 (2.28) | 1.5 | **1.41x** | **1.52x** |
| 256 | alm2map | 1.41 (1.62) | 1.8 | 0.78x | 0.90x |
| 512 | map2alm | 14.97 (16.23) | 7.0 | **2.14x** | **2.32x** |
| 512 | alm2map | 10.50 (12.03) | 7.6 | **1.38x** | **1.58x** |
| 1024 | map2alm | 60.81 (64.08) | 40.4 | **1.51x** | **1.59x** |
| 1024 | alm2map | 53.62 (57.44) | 43.0 | **1.25x** | **1.34x** |

So with fp32 tables the GPU beats the ducc0 C code in **five of six spin-0 cells** even when
ducc is given its fastest observed time; the sixth is the sub-2 ms Nside 256 synthesis cell
where both sides are launch-dominated. In fp64 tables the same cells are 0.32-0.53x for
synthesis and 0.41x for Nside 1024 analysis (the fp64 band is 73.5 GiB and is declined by the
40 GiB budget).

**Never take a ducc number from the comparison probe at Nside 512.** One run
(`.qwen/tmp/sht_vs_ducc_s14.log`) read 11.0 / 6.8 ms there; three other runs of the identical
script read 15.3-16.0 / 12.2-12.6 ms, and the 45 clean GPU-free processes read 14.97-17.35 /
10.50-13.41 ms. The s14 line is an outlier, and it is the line that made the Nside 512
synthesis cell look like 0.86x rather than 1.38x. Baselines come from `ducc_only.py` across
several processes; the comparison probe is for the GPU side only.

**Pipeline re-scored after both fixes** (`score_one.py`, fresh process per cell, control
1351-1400 GB/s): Nside 512 spin-0 fp32 **346→136 ms (2.54x)** rel 7.17e-09; Nside 512 spin-0
fp64 **340→199 ms (1.71x)** rel 1.71e-12; Nside 1024 spin-0 fp32 **1790→809 ms (2.21x)** rel
5.48e-08; Nside 512 spin-2 fp32 **734→401 ms (1.83x)** rel 3.55e-07. Before this session the
same cells read 157.2 / 222.2 / 900.6 / 401 ms.

**Rejected, with the measurement.** Five-smooth FFT lengths to shorten the ring transforms:
on this card pow2 wins — 2047 rows, length 3600 → 0.56 ms vs 4096 → 0.49 ms; 4095 rows, 7200 →
2.51 ms vs 8192 → 2.02 ms (`.qwen/tmp/fft_len.log`, FFT-only 2065-2159 GFLOP/s at pow2).
`--xla_gpu_cufft_autotune=true` is not a flag in this XLA build (`Unknown flag in XLA_FLAGS`).
Do not re-derive either.

**Two measurement traps that cost time this session** (both in memory):
1. *One Nside per process.* A combined fp32 sweep read Nside 1024 analysis as 148.2 ms (0.42x,
   identical to the fp64 fused kernel) because the process still held the 256/512 tables, the
   1024 band build hit `Allocator (GPU_0_bfc) ran out of memory trying to allocate 400.00MiB`,
   and the negative build cache silently routed the transform through the generic path.
2. *`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` silently downgrades the route.* With the pool
   preallocated to 0.95, the band build raised `RESOURCE_EXHAUSTED` and the transform fell
   back: 41.3 ms at the default pool became 147.4 ms. A bigger pool is not a bigger budget
   when the build needs headroom to assemble.

**Verification.** `pytest tests/ -q` → **126 passed, 3 skipped in 291.57 s**
(`.qwen/tmp/pytest_s16.log`); the three skips are the two-GPU tests. The suite already asserts
`alm2map`/`map2alm` against `pymaster` at atol 1e-12-1e-13, so both rewirings are guarded
end to end, plus the new exact-equivalence test for the ring synthesis.

**Known asymmetry, deliberately not touched**: `_map2alm_once_pallas_multi_gpu` still uses the
filling `_forward_healpix_fft` + `[:, L_work:]`. It cannot be exercised on this single-GPU box
(its tests skip), so it was left alone rather than changed blind. Pre-existing dead code also
left alone: `_SPIN_PALLAS_MAX_L = 768`.

**Where the azimuthal stage still sits, and the lever.** At Nside 1024 the analysis ring stage
costs 13.6 ms while a raw batched complex128 FFT *pair* on 4095 rows of 8192 measures **4.73 ms**
— so ~9 ms is the pad/chirp/slice/scatter around the transforms, and structurally the CZT
spends **two** 8192-point transforms per ring where ducc0 spends **one** FFT of each ring's own
length (`nphi` runs 4…4·nside, average ~2048). Bluestein only pays when the requested m-range
is a small slice of the input grid; at the HEALPix band limit `L = 3n−1 = 0.75·nphi`, so it is
the wrong tool there. The open question is whether one padded transform per ring group
(s2fft's `healpix_ffts` modes) reproduces our exact ring values and how many distinct FFT
shapes a jit-friendly grouping needs — HEALPix RING gives each `nphi = 4j` exactly two rings,
so a naive per-length grouping is `nside` tiny batches, which is presumably why everybody
pads. Fixing this stage is worth ~1.4x on analysis at 1024 (40.4 → ~31 ms, i.e. ~1.9x vs ducc)
and similar on synthesis.

**No fp32 accuracy cliff — that reading was a normalisation bug in the probe.** The isolated
probe's `rel alm` column reported 2.8e-2 at Nside 512 and 1.1e-1 at 1024 with
`set_table_precision("fp32")` against 1.2e-7 with fp64, which reads as fp32 tables destroying
the analysis. It is not an accuracy measurement: ducc0's alms differ from ours by a convention
factor `|s| ≈ 2.5e5` (the quadrature scale) and the probe fitted `s` but divided the residual
by `||ref||` instead of `||s·ref||`, inflating every number by exactly `|s|`. Two independent
corrected measurements agree: with the residual normalised by the fitted magnitude
(`.qwen/tmp/fp32_accuracy.py`, now also printing `|s|`) and a *same-implementation* comparison
that has no convention factor at all (`.qwen/tmp/fp32_ell_breakdown.py`, our alms at fp32
tables vs our alms at fp64 tables, both resident in one process, fp32 built first), the analysis
error is **4.8e-13 (fp64) → 1.1e-07 (fp32)**, and it is **flat in ell** (8.0e-08 for ell < 192,
1.4e-07, 1.9e-07, 2.0e-07, 2.2e-07, 2.2e-07, 2.2e-07, 1.9e-07 across the eight equal ell bins
to 1535). Synthesis is 5.08e-13 (fp64) → 1.14e-07 (fp32), with `|s| ≈ 1` so that direction was
never inflated. The 1.1e-7 is the cost of the *storage-width reduce* in
`_theta_matrix._contract_theta` (its docstring already says 1.28e-07), not of the table build.
So the fp32 win at Nside 1024 is not borrowed against a hidden accuracy loss, and the
fp32-route numbers above stand as scientific results. Habits: print `|s|` next to every fitted
residual, and settle any precision claim with an ours-vs-ours comparison instead of a
cross-implementation one.

## Session 14 (2026-09-05) — the ring stage was recomputing its own constants, and two thirds of the rings never needed a chirp-Z

Two changes to the azimuthal stage, both found by decomposing the stage into passes
(`.qwen/tmp/ring_passes.py`) instead of guessing, both measured against the committed
bodies copied verbatim from `HEAD` into the probe (`.qwen/tmp/ring_hoist_ab.py`,
`.qwen/tmp/ring_split_ab.log`), clocks spun up, bandwidth control 1382-1433 GB/s:

| nside | analysis ring stage | synthesis ring stage | max\|new − old\| / max\|old\| |
|---|---|---|---|
| 256 | 0.77 → **0.21 ms** (3.66x) | 0.75 → **0.22 ms** (3.41x) | 9.6e-16 |
| 512 | 3.28 → **0.82 ms** (4.01x) | 3.21 → **0.86 ms** (3.75x) | 1.2e-15 |
| 1024 | 13.60 → **4.29 ms** (3.17x) | 13.21 → **4.32 ms** (3.06x) | 1.3e-15 |

**1. XLA does not fold the ring constants; a jit that computes only constants still runs
every call.** The convolution kernel and both chirp ramps are functions of (L, nside) only,
but they were written inline inside the jitted transform, and the assumption "closed over →
HLO literal → folded away" is false here. Measured with a jit whose body is *only* the
constant subexpression (a folded program launches nothing and times ~0):

| constant-only jit at Nside 1024 | ms |
|---|---|
| `fft(exp(i·chirp_angle(shift, two_nphi)))` — the kernel spectrum | **2.03** |
| the same including its own `exp` | **5.02** |
| `exp(-i·chirp_angle(p_index, two_nphi))` — one chirp ramp | 1.65 (163 GB/s) |

against a 13.57 ms ring stage. The operands are `(4n-1, N)` arrays, not vectors, because
`_chirp_angle` broadcasts against `two_nphi` of shape `(ntheta, 1)` — so it is a real batched
FFT of a real 2-D constant. The fix is `_ring_analysis_tables` / `_ring_synthesis_tables`
(`@lru_cache(maxsize=2)`, keyed on `(L, nside, device)`), whose three arrays are passed as
**jit arguments**; closing over the built arrays would only move them into the jaxpr as
literals. Measured alone (before change 2) this was **bit-identical** — error exactly
`0.00e+00` at all three Nsides — at 2.35 / 2.01 / 1.96x on the analysis stage and
2.39 / 2.01 / 1.94x on synthesis.

Two traps in the wiring, both now documented in the docstrings:
* `lru_cache` + tracing leaks tracers. The small-`L` cores jit the whole transform
  (`_PALLAS_TRACED_MAX_L = 256`), so a builder running inside that trace would cache tracers
  and poison the next one. The builders run inside
  `with jax.ensure_compile_time_eval(), jax.default_device(device):` and `device_put` the
  result, so cached values are always concrete and committed to the device of the map.
* the fetch must live **outside** the jit. `_forward_healpix_fft` and `_finish_inverse_pallas`
  are now plain Python wrappers that fetch the tables and call the jitted implementation.
  Keeping them jitted would put the tables back inside the trace, and the multi-GPU path runs
  the ring stage on a non-default device — hence the `device` parameter, taken from
  `getattr(arr, "device", None)` at the three call sites.

Retained bytes: the tables are ~1.0 GB (analysis) + 0.7 GB (synthesis) at Nside 1024 when they
cover every ring; change 2 halves them. `maxsize=2` bounds the cache at two geometries.

**2. The equatorial belt does not need a chirp-Z at all.** `_ring_czt_constants` pads every
ring to the same transform length `next_fast_len(L + 4n)`, which is why all `4n-1` rings pay
two 8192-point transforms at Nside 1024. But rings `nside-1 … 3·nside-1` all have
`nphi = 4·nside` — a power of two, and wider than `L = 3n-1` — and in RING order their pixels
are **one contiguous run** (checked: `start` advances by exactly `4n`, belt = 66.7 % of the
map, verified for 64/256/1024). So for those `2n+1` rings

    F_m = Σ_{p<nphi} x_p e^{-2iπpm/nphi} = fft(x)[m],   m < L,

one unnormalised forward FFT of a reshape: no input chirp, no zero-pad, no kernel multiply,
no second transform. Synthesis is the mirror statement — zero-pad the m block to `4n` and take
`width · ifft(·)` — again one transform. Only the `2(n-1)` polar rings keep Bluestein, so the
tables are now **cap-only**. The split is `_ring_split_numpy(L, nside)`, and because caps are a
prefix and a suffix while the belt is the middle, reassembly is
`concatenate([cap[:n-1], belt, cap[n-1:]])` — contiguous slices, no indexed scatter (which
this project has measured to be catastrophic in XLA).

*Gate.* The shortcut needs `L ≤ 4n`: a length-`4n` transform returns exactly the band
`m < 4n`. Beyond that the ring sums repeat with period `nphi`, so an overset band would have to
be tiled back in; `L > 4n` declines the belt and puts every ring through the chirp-Z (belt
empty, tables full width). Verified on both sides of the gate: `L = 4n` and `L = 4n+7` at
nside 16 give 0.00e+00 (analysis, vs the pre-split path) and 7e-16 / 5e-16 (synthesis, vs
`_inverse_ring_fft`).

**Both routes now checked against the definition, not just against each other.**
`tests/test_sht.py::test_ring_analysis_matches_direct_dft_ring_by_ring` builds
`Σ_p x_p exp(-2iπpm/nphi)` explicitly in NumPy for nine rings per Nside — cap, the two
boundary rings, belt — at (16, 47), (32, 95), (16, 64), (16, 71), and asserts
`atol = 1e-12 · max|direct|`. Standalone the same check reads 6.8e-15 / 1.8e-14 / 3.3e-14 at
nside 8/16/32, which is the direct DFT's own rounding, not the library's.

**The m-periodicity that makes the next lever available, measured.** `F_m` is periodic in `m`
with period `nphi` (the kernel gains `exp(-2iπp) = 1`), so only `min(L, nphi)` values are
distinct and a ring's transform bound could drop from `L + nphi - 1` to `2·nphi - 1`.
`.qwen/tmp/ring_alias_check.py` measures `max|F_m − F_{m mod nphi}| / max|F|` =
**2.7e-16 / 3.3e-16 / 3.5e-16** at nside 32/64/128 for every ring with `nphi < L`. That makes
a per-size-class decomposition of the *polar* caps exact: at Nside 1024 most cap rings would
transform at 4096 or less instead of 8192, with their m block tiled back out by a gather. Not
implemented — the belt captured the cheap half. The same probe also validates the synthesis
half of that plan (aliased-sum into `nphi` slots, then one plain inverse FFT, then the
`2P - F_0` mirror): **8.1e-16** against the shipped chirp-Z (`ring_alias_check2.log`), so cap
grouping is exact in both directions, not just in analysis.

**Isolated SHT scoreboard after both changes** (fp32 tables, one Nside per process,
`sht_vs_ducc_s23_{256,512,1024}.log`, control 1337-1447 GB/s; ducc column is the recorded
best/median of the 45 GPU-free `ducc_only.py` processes, never the in-process number):

| nside | dir | ducc best (median) ms | before session | after both | ratio vs best | vs median |
|---|---|---|---|---|---|---|
| 256 | map2alm | 2.12 (2.28) | 1.5 | **1.1** | **1.93x** | 2.07x |
| 256 | alm2map | 1.41 (1.62) | 1.8 | **1.3** | **1.08x** | 1.25x |
| 512 | map2alm | 14.97 (16.23) | 7.0 | **4.6** | **3.26x** | 3.53x |
| 512 | alm2map | 10.50 (12.03) | 7.6 | **5.3** | **1.98x** | 2.27x |
| 1024 | map2alm | 60.81 (64.08) | 40.4 | **31.4** | **1.94x** | 2.04x |
| 1024 | alm2map | 53.62 (57.44) | 43.0 | **34.3** | **1.56x** | 1.67x |

**All six spin-0 cells now go to the GPU, against ducc's fastest observed time**, including the
Nside 256 synthesis cell that it held all along (0.78x → 1.08x). Session 13's "five of six" is
superseded. The `rel alm` column now prints `|s|` beside it (probe fixed the same way
`fp32_accuracy.py` was) and reads 9.9e-08 / 1.4e-07 / 1.3e-07 at 256/512/1024 with fp32 tables
— the storage-width reduce, as expected, not a cliff.

**Pipeline re-scored** (`score_one.py`, fresh process per cell): Nside 512 spin-0 fp32
**332→102 ms (3.26x)**, Nside 1024 spin-0 fp32 **1713→678 ms (2.53x)**; after change 1 alone
they were 113 ms and 717 ms, and before the session 136 ms and 809 ms. `rel` unchanged in every
cell (7.17e-09 at 512, 5.48e-08 at 1024). Nside 512 fp64 spin-0 after change 1: **323→176 ms
(1.84x)**, rel 1.71e-12. Nside 512 spin-2 fp32 after change 1: **719→389 ms (1.85x)**, rel
3.55e-07 — spin-2 goes through `_forward_ring_fft_full` / `_inverse_ring_fft_complex`, which
take neither change (their ring spectra are genuinely complex, and at `2L-1 > 4n` the belt
shortcut needs the tiling the caps still owe); its `field` cell is still 0.73x and remains the
worst cell on the board.

**Verification.** `pytest tests/ -q` → **126 passed, 3 skipped in 305.62 s**
(`.qwen/tmp/pytest_s23.log`) with the belt split in; the 3 skips are the two-GPU tests. The new
`test_ring_analysis_matches_direct_dft_ring_by_ring` (4 parametrised geometries, 9 rings each,
both routes and the overset gate) passes, as does the pre-existing
`test_ring_synthesis_from_positive_half_matches_centred_window`, which now checks the split
synthesis against the independent centred chirp-Z at 1e-13.

**Unrelated bug spotted, not fixed (CPU-only path).** `_spin_slice._pool_headroom` guards only
`memory_stats()` with `try/except`, then calls `stats.get(...)` outside it; on a CPU backend
`memory_stats()` returns `None` rather than raising, so two spin tests die with
`AttributeError: 'NoneType' object has no attribute 'get'` when the suite is run with
`CUDA_VISIBLE_DEVICES=`. Harmless on GPU (where the suite is run), but the docstring promise
"+inf when the device won't say" is not honoured.

**What the ring stage still costs, and what is left.** After both changes it is
0.21 / 0.82 / 4.29 ms at Nside 256/512/1024 against a latitudinal (theta) contraction that is
at the card's streaming floor (26.18 ms at 1024, 1506 GB/s against a 1488 GB/s pure-`sum`
control, north/south already folded by the `(-1)^(ell+m)` parity split in `_theta_matrix`). So
the ring stage is now ~12 % of the Nside 1024 analysis call and the table route owns the rest:
the remaining levers are the polar cap size classes (measured exact, above), spin-2's ring
twins, and nothing at all inside the theta contraction.

---

## Session 15 (2026-09-05) — the ring stage stops being double precision, and the polarised ring gets the belt

Three changes, one of which was built, measured, and **reverted**. Every number below is read
from a log in `.qwen/tmp/`; controls (a pure `sum` over a 384 MiB array) printed 1317-1462
GB/s on every line, so the cards were at clock.

### 1. The azimuthal transforms follow the table precision (`ring_dtype()`)

The ring stage was the last part of the transform still running in double precision even when
`set_table_precision("fp32")` had already moved the Legendre band and the Wigner slices down:
its constants were built with x64 on, so every cap row was a pair of complex128 FFTs and every
belt row a complex128 `4*nside` one. `ring_dtype()` now returns `complex64` in fp32 mode, the
table builders cast to it, and the stages cast their *input* to the tables' type (so the
element type is decided by the tables rather than by promotion). Measured on the shipped
functions (`.qwen/tmp/ring_stage_prec.log`, one process, both precisions, jit specializing on
the argument types):

| nside | spin | stage | fp64 ms | fp32 ms | speedup | ring-value difference |
|---|---|---|---|---|---|---|
| 256 | 0 | analysis | 0.22 | 0.08 | 2.8x | 3.61e-07 |
| 256 | 0 | synthesis | 0.24 | 0.08 | 3.0x | 4.05e-07 |
| 256 | 2 | analysis | 0.40 | 0.09 | 4.4x | 3.77e-07 |
| 256 | 2 | synthesis | 0.44 | 0.13 | 3.4x | 4.29e-07 |
| 512 | 0 | analysis | 0.83 | 0.16 | 5.2x | 3.55e-07 |
| 512 | 0 | synthesis | 0.86 | 0.18 | 4.8x | 4.21e-07 |
| 512 | 2 | analysis | 2.06 | 0.32 | 6.4x | 3.44e-07 |
| 512 | 2 | synthesis | 2.06 | 0.41 | 5.0x | 4.23e-07 |
| 1024 | 0 | analysis | 4.27 | 1.31 | 3.3x | 3.48e-07 |
| 1024 | 0 | synthesis | 4.36 | 1.28 | 3.4x | 4.09e-07 |
| 1024 | 2 | analysis | 7.95 | 2.50 | 3.2x | 4.17e-07 |
| 1024 | 2 | synthesis | 8.15 | 2.71 | 3.0x | 4.47e-07 |

This is the `fp64-roofline-wall` result showing up in an FFT: on this card a double-precision
transform runs near the fp64 arithmetic rate and the fp32 one is a different class of machine.
Shortening the same transform (change 3 below) bought nothing comparable.

### 2. The polarised ring stage gets the belt (`_forward_ring_fft_full`, `_inverse_ring_fft_complex`)

The spin-2 twins of the spin-0 ring stage had none of session 14's treatment: all
`4*nside-1` rings went through a chirp-Z pair at `next_pow2(width + 2L)` (8192 at Nside 512,
16384 at 1024) with the ramps and kernel rebuilt inside every trace. The residue identity that
served the spin-0 belt works here too — `F_m` is periodic in `m` with period `nphi` and
`F_-m = F_(nphi-m)` — so on the belt the *centred* window `m in [-(L-1), L)` is
`concat(F[4n-L+1 : 4n], F[0 : L])`: two contiguous slices of one length-`4*nside` FFT, no
chirp, no pad, no kernel. The inverse runs it backwards by adding coefficients that share a
residue into one slot and transforming once; `L > 2*nside` makes the two slot ranges overlap,
so placement is the sum of two padded arrays rather than a concatenation (still slice
arithmetic — no scatter, per `xla-scatter-and-clock-ramp`). The polar rows keep the chirp-Z,
with its constants hoisted into `_spin_ring_analysis_tables` / `_spin_ring_synthesis_tables`
and threaded through the eight call sites as arguments, which is the pattern
`_forward_healpix_fft` already documents (fetch outside the trace, never bake).

Stage measurement against the committed bodies, both arms at complex64
(`.qwen/tmp/spin2_stage_ab.log`):

| nside | stage | committed ms | now ms | speedup |
|---|---|---|---|---|
| 256 | analysis | 1.47 | 0.10 | 14.4x |
| 256 | synthesis | 1.53 | 0.13 | 12.0x |
| 512 | analysis | 6.54 | 0.34 | 19.1x |
| 512 | synthesis | 6.73 | 0.41 | 16.4x |
| 1024 | analysis | 26.29 | 2.52 | 10.5x |
| 1024 | synthesis | 27.23 | 2.73 | 10.0x |

Correctness: at fp64 on CPU the new bodies match the committed ones to 5.6e-16 / 5.8e-16
(Nside 8), 4.4e-16 / 5.9e-16 (16), 6.5e-16 / 7.2e-16 (32), forward / synthesis; and a new test
(`test_spin_ring_window_matches_direct_dft_both_ways`, 8 geometries) pins both directions
against an explicit `sum_p x_p exp(-2i pi p m / nphi)` on cap, boundary and belt rings,
including the `L = 2*nside` touch point, the `L = 3*nside` overlap and the declined
`L = 4*nside + 7`.

One trap worth recording: the spin-2 input is the helicity combination `Q -/+ iU`, so casting
it to the tables' *real* type (what the spin-0 stage does, correctly, with a real map) silently
drops the imaginary part and produces a 0.7 relative error in the forward direction while
leaving the inverse exact. `_forward_ring_fft_full` takes `chirp_in.dtype`, not
`chirp_in.real.dtype`.

### 3. Polar-cap size classes: built, exact, and reverted

Grouping the polar rings by `next_pow2(L + nphi)` so each group pays its own transform length
models out at 0.865 of the cap FFT work at the band limit (two classes at Nside 1024: 4096 for
`rings[0:256]`, 8192 for `rings[256:1023]`). It is exact — 16/16 end-to-end cells bit-identical
with the route on and off (`.qwen/tmp/class_e2e_control.log`,
`.qwen/tmp/class_e2e_classes.log`), and the GPU stage agrees with the committed single-width
bodies to 5.5e-16 … 1.3e-15 across the six cells below. Measured on the GPU
(`.qwen/tmp/cap_stage_ab.log`) it is a **loss**:

| nside | stage | single-width ms | classes ms | speedup |
|---|---|---|---|---|
| 256 | analysis | 0.21 | 0.34 | 0.62x |
| 256 | synthesis | 0.23 | 0.34 | 0.67x |
| 512 | analysis | 0.85 | 0.90 | 0.94x |
| 512 | synthesis | 0.85 | 0.93 | 0.92x |
| 1024 | analysis | 4.28 | 4.06 | 1.05x |
| 1024 | synthesis | 4.36 | 4.08 | 1.07x |

Extra kernel boundaries and a longer graph cost more than a shorter transform, and only at the
largest geometry does the trade even break even. The class code is reverted; the modelled FLOP
saving is not the quantity XLA optimises. Recorded in `ring-belt-plain-fft` so nobody re-pays
for it.

### Where the transform stands against the ducc0 C code

Isolated one-pass `map2alm`/`alm2map`, fp32 tables, `L = 3*nside-1`, ducc0 0.39.1 on all 192
cores (`.qwen/tmp/sht_vs_ducc_s25_{256,512,1024}_0.log`):

| nside | dir | ducc ms | GMaster ms | ratio | session-14 ratio |
|---|---|---|---|---|---|
| 256 | map2alm | 2.3 | 1.3 | 1.85x | 1.95x |
| 256 | alm2map | 1.7 | 1.3 | 1.34x | 1.08x |
| 512 | map2alm | 14.8 | 4.7 | 3.18x | 3.30x |
| 512 | alm2map | 11.6 | 4.9 | 2.37x | 2.22x |
| 1024 | map2alm | 57.4 | 28.5 | **2.02x** | 1.82x |
| 1024 | alm2map | 51.1 | 30.9 | **1.65x** | 1.51x |

Pipeline against pymaster (`n_iter=3`, `.qwen/tmp/score_s25.log`): Nside 512 spin-0 fp32
**321→93 ms (3.45x)**, Nside 1024 spin-0 fp32 **1697→635 ms (2.67x)**, Nside 512 spin-2 fp32
**674→332 ms (2.03x)** — the spin-2 pipeline cell crosses 2x for the first time, and its
`field` cell moves from 0.73x to 0.83x.

**The accuracy price, stated plainly.** With the complex64 ring the coupling-matrix `rel`
against pymaster at Nside 512 spin-0 goes **7.17e-09 → 1.45e-07** (Nside 1024:
5.48e-08 → 1.32e-07; Nside 512 spin-2: 3.55e-07 → 3.04e-07) for 11 % and 6 % on those two
cells. It is inside what `set_table_precision("fp32")` promises (~1e-7 representation error),
but it is a real degradation of what fp32 mode was actually delivering, and it is now the
dominant error term in fp32 runs. The spin-0 `rel alm` in the isolated probe likewise moved
from 1.4e-07 to 3.7e-07 at Nside 512. If that is the wrong trade, the fix is one line: make
`ring_dtype()` ignore the table precision (or give it its own switch) — the ring stage then
returns to the fp64 column above and everything else in this session is unaffected.

**Spin 2 still loses per-pass, and it is no longer the ring's fault.**
`.qwen/tmp/sht_vs_ducc_s25_{256,512}_2.log`: 4.9 vs 5.9 ms (0.83x) and 2.9 vs 5.5 ms (0.52x)
at 256, 26.0 vs 34.2 ms (0.76x) and 22.2 vs 37.6 ms (0.59x) at 512. The ring stage is now
0.34 ms of that 34.2 ms call — the polarised latitudinal contraction (`_spin_slice`) owns the
rest, and its synthesis layout at Nside 512 is the one `_spin_slice.slabs_for` documents as
paying 2.1x for a strided-axis reduction. That, not the azimuth, is the next thing to attack.

**Verification.** `pytest tests/ -q` on GPU → **141 passed, 3 skipped in 336.38 s**
(`.qwen/tmp/pytest_s25b.log`) with all three changes in, including the new
`test_spin_ring_window_matches_direct_dft_both_ways` across 8 geometries. The earlier
`pytest_s25.log` (133 passed) predates that test.

---

## Session 16 (2026-09-05) — Nside 2048/4096: where the wall actually is, measured

The demand is the transform at Nside 2048 and 4096 on one card, all spins, winning like 256-1024
do. This session measured the wall instead of guessing at it. No library change shipped: one
optimisation was implemented and **reverted** (below), and one probe was built whose own numbers
came with a caveat.

Machine, for the record (`nvidia-smi`, `free`): 2x RTX PRO 6000 Blackwell, **97.9 GiB each**,
PCIe **gen5 x8**, no NVLink between them; 376 GiB host RAM. Host staging of a table is therefore
~50 GB/s against ~1.5-2.9 TB/s on the card — a streamed band cannot live in host RAM and be read
once per pass.

### 1. The current state at Nside 2048: we lose by 3-5x

Isolated one-pass transform, fp32 tables, `L = 3*nside-1` (`.qwen/tmp/sht_vs_ducc_s26_2048_0.log`,
controls 1336/1334 GB/s):

| dir | ducc ms (in-process) | GMaster ms | ratio |
|---|---|---|---|
| map2alm | 301.5 | 1048.1 | 0.29x |
| alm2map | 287.3 | 1356.6 | 0.21x |

`rel alm 3.7e-07`, so the result is correct — it is just slow. The Legendre band is declined
(40 GiB gate) and the fused fp64 Pallas kernel carries the call. Clean GPU-free ducc baselines
(`.qwen/tmp/ducc_only_2048_0.log`, `_2.log`, 192 cores, reps=3) are a bit slower than the
in-process column, as they always are: spin 0 map2alm min/med/max **369.51/384.51/574.03** ms,
alm2map **373.05/477.58/491.03** ms; spin 2 map2alm **732.08/903.86/1001.08** ms, alm2map
**609.54/612.15/613.55** ms. And at Nside 4096 (`_4096_0.log`, reps=2, `OMP_NUM_THREADS=96`):
map2alm **1840.97/1842.75/1844.52** ms, alm2map **1865.26/1865.32/1865.37** ms.

### 2. The contraction is not the problem — the table's existence is

Cost model for building and contracting the band one m-block at a time
(`.qwen/tmp/block_cost.py`, fp32, Nside 2048, `BLOCK = 64`):

| m0 | rows | slab GiB | build ms | contract ms | contract GB/s |
|---|---|---|---|---|---|
| 0 | 6143 | 6.00 | 113.6 | 2.04 | 2939 |
| 767 | 5376 | 5.25 | 99.4 | 1.79 | 2941 |
| 1535 | 4608 | 4.50 | 85.3 | 1.55 | 2907 |
| 3071 | 3072 | 3.00 | 57.1 | 1.04 | 2887 |
| 4607 | 1536 | 1.50 | 28.6 | 0.54 | 2785 |
| 6079 | 64 | 0.06 | 1.3 | 0.08 | 831 |

fp64 `sum` control on the same process: 1395 GB/s. Integrating over the block index:

* whole fp32 band at Nside 2048: **290.9 GiB** (4096: ~2.3 TiB) — can never be resident;
* one latitudinal pass from it: **105 ms**, at 2939 GB/s of fp32 slab reads, i.e. **at the
  card's streaming rate** and **2.9-3.5x faster than ducc's whole map2alm** (301-370 ms);
* building it: **5.5 s**.

So at Nside >= 2048 every number is right except one: the table costs 5.5 s to make and 291 GiB
to hold, while ducc needs 0.3 s and no table at all. Rebuilding per pass (5.5 s) is 17x worse
than ducc; the *same* 5.5 s spent once and contracted seven times costs 5.5 + 7x0.105 = 6.2 s
against ducc's 7x301 = 2.1 s, which is still a loss. **The build is the wall, not the
transform.**

### 3. Making the builder faster: one probe, one caveat, no win shipped

`.qwen/tmp/build_speed.py` times one m-block of `_theta_matrix._build_slab` at Nside 1024
(m0 = 0, 1.50 GiB of fp32 output): shipped body **32.1 ms** (~150 GFLOP/s), `unroll=4` 34.7 ms,
`unroll=8` 32.5 ms, rescale-machinery removed 26.2 ms, that plus `unroll=4` 14.3 ms, fp32-carried
recurrence 21.6 ms.

**Caveat that has to travel with those numbers: every non-base variant printed NaN in the
agreement column**, so none of them is a usable candidate — the probe never produced a
value-correct faster body, and the only conclusion that survives regardless of NaN is that
**scan unrolling alone buys nothing** (32.1 -> 32.5 ms), which says the builder is not
limited by loop overhead. An earlier draft of this section quoted a "3.5x bit-identical"
result from this probe before it had been run; that was wrong and is removed. What the numbers
do support is the diagnosis: the recurrence is carried as two `(mb, north)` tiles through
`lax.scan`, and at ~150 GFLOP/s it is nowhere near the card's fp64 arithmetic rate — the carry
is being written and read back, not kept.

The fp32 escape hatch is closed independently: `dfp32-analysis-kernel` records that an
fp32-carried recurrence has a precision cliff (8.8e-8 at L = 224, **2.3e-3** at L = 1024, from
cancellation at |m| > ~0.85L), and this card's fp64 matmul rate is 1.88 TFLOP/s against ~120
TFLOP/s fp32 (`fp64-roofline-wall`). The recurrence has to stay fp64; only storage may be fp32.

### 4. A spin-2 idea that is algebraically exact and numerically a regression

`_spin_slice._march` rebuilds, on every m-step, a full `(ntheta, L)` fp64 tile

```
lamb = ((el+1)(1-c) + (m-L+el+1)c - half) / s
```

whose difference from the seed tile `lamb0` is exactly `(m-1)·c/s` — a rank-1 update, no divide.
Verified in numpy over m at L = 1344: worst relative difference **6.04e-16**. Implemented and
checked against HEAD on CPU: Nside 16 agrees to 1.58e-14, Nside 64 to 8.66e-14, and **Nside 32
gains 5 NaN entries** at rows m = 0..2, theta rings 87 and 95, ell 3-4, where HEAD holds ordinary
values (max 2.449 and 2.530 there; the table's largest finite value is 1.663e+02). Reverted;
working tree is clean.

The mechanism is that the march is not mask-safe: `bigi = 1/|dl_entry|` and the carry updates
`jnp.where(index, bigi*dl_cur, dl_prev)` evaluate the divide everywhere, so a 1e-14 change in
`lamb` flips a masked entry through inf and into the emitted rows. That is worth remembering as a
latent fragility in `_spin_slice`, not just as a blocked optimisation: **any** restructuring of
the march body can hit it. Before the rank-1 update (or anything else that perturbs rounding
there) can land, the carry must be made inf-safe and a zero-NaN assertion added to the spin tests
— there is none today, which is why this was found by hand rather than by the suite.

### 5. What this implies for the design, stated as arithmetic

At Nside >= 2048 nothing about ducc changes: it is fp64-recurrence-bound on a CPU with 4.14
TFLOP/s against the card's 1.88. Our wins have always come from bandwidth and fp32, and at these
sizes the fp32 object is 291 GiB. Two shapes can exploit the card without holding that:

1. **A fast builder + one sweep per geometry.** If the builder reached even 10x (0.55 s, and the
   physical floor for writing 291 GiB at ~1.5 TB/s is ~0.2 s), then building once and contracting
   every pass against it costs 0.55 + 7x0.105 = 1.3 s against ducc's 2.1 s — a win, and every
   additional pass is nearly free.
2. **Multi-RHS fused recurrence.** The MASTER pipeline runs the *same* Legendre recursion seven
   times per workspace build (field, then the coupling terms), each pass re-deriving values the
   others already needed. Sharing one recurrence across a few RHS tiles amortises exactly the cost
   that is blocking 2048, at a memory footprint of the RHS tiles rather than the band — and it is
   the same trick for the spin-2 slices. The measured caution is `dfp32-analysis-kernel`'s
   register-spill finding: adding four loop-carried fp64 FMAs per lane-degree cost +161% there, so
   the RHS group size must be small (2-3) and measured, not assumed.

Both are "compute the recursion once, spend it many times", which is the one thing ducc's
per-call structure cannot do. That is the next build target; until then, Nside 2048 is 0.29x and
Nside 4096 is not attempted.

**Still running when this was written:** the spin-2 Nside 2048 isolated cell
(`.qwen/tmp/sht_vs_ducc_s26_2048_2.log`) and the queued pipeline scores
(`.qwen/tmp/queue_s26.sh` -> `score_s26_2048_0`, `score_s26_2048_2`, `score_s26_4096_0`). No
number from those has been quoted here.

## Session 17 (2026-09-05) — the big-Nside pipeline is measured, and the rescale is not exact

### 1. End-to-end MASTER pipeline at Nside 2048 and 4096, spin 0, fp32 tables

`score_one.py` prints `ref->gm ms (ref/gm x)`, so **>1.00x is a GMaster win**. One cell per
process, `repeats=1`, clocks spun up, control reduce 1336–1413 GB/s in every run:

```
FRESH n2048 spin=0: TOTAL 10464->18255ms (0.57x) | field 2423->8312ms (0.29x)
    coupling 5633->1527ms (3.69x)  coupled_cell 139->0ms (306.96x)  decouple 0->1ms (0.40x)
    rel=7.23e-08
FRESH n4096 spin=0: TOTAL 72884->146433ms (0.50x) | field 14973->66675ms (0.22x)
    coupling 43383->12734ms (3.41x)  coupled_cell 535->2ms (328.39x)  decouple 1->2ms (0.78x)
    rel=1.16e-07
```

Two facts, pulling in opposite directions:

- **The NaMaster stages that are matrix work we win, at every size.**
  `compute_coupling_matrix` is **3.69x** at 2048 and **3.41x** at 4096; `compute_coupled_cell`
  is 307x and 328x. Those run once per mask/bandlimit pair in a real analysis and they are not
  marginal.
- **The whole pipeline still loses: 0.57x and 0.50x.** The `NmtField` stage is the entire
  deficit (8312 ms against 2423; 66675 against 14973), and that stage is latitudinal transforms
  repeated for the smoothing iterations — the same fused-kernel call session 16 measured at
  0.29x per pass, because at these sizes the fp32 band is 291 GiB and the 40 GiB budget
  declines it.

So "does GMaster beat NaMaster at Nside ≥ 2048?" now has a measured answer instead of an
extrapolation: **no — 0.50–0.57x end to end, and the loss is one stage, not a diffuse tax.**
Correctness holds there (`rel` 7.23e-08 and 1.16e-07). Spin 2 at 2048 remains the isolated
0.01x cell (110 s per transform); a pipeline cell there is ~13 min per repetition and would
only re-confirm it, so it was deliberately not queued.

The size of the prize is set by the same table: were `field` merely at ducc's cost, the 2048
total would be 2423+1527+0+1 = 3951 ms against the CPU's 10464 — **2.65x**. Everything else in
the pipeline is already faster than the CPU.

### 2. Negative result: the fused kernel's thread-block shape is not the limiter

`_sht_pallas.py` documents that "about ten float64 vectors live per theta lane, so
`block_size / (32 * num_warps)` doubles per thread is what decides whether they sit in
registers or spill". `_pallas_block_size` already cites measurements behind the 512 choice
("the synthesis kernel degrades ~1.5-1.6x at 1024 and cliffs hard at 2048"), but `num_warps`
has always been the fixed constant `_NUM_WARPS = 2`, and no combination had been swept for
either direction at the sizes where the fused kernel is the *only* route. It now has
(`.qwen/tmp/block_sweep.py`, which monkeypatches both; totals are map2alm + alm2map, ms):

| block / warps | 512/1 | **512/2 (shipped)** | 512/4 | 256/1 | 256/2 | 128/1 | 128/4 | 64/1 |
|---|---|---|---|---|---|---|---|---|
| Nside 1024 fp64 | 448 | **312** | 336 | 319 | 330 | 335 | 476 | 371 |
| Nside 2048 fp32 | 3312 | **2408** | 2612 | 2403 | 2560 | 2536 | 3633 | 2808 |

The shipped configuration wins at both sizes (256/1 ties it at 2048 to within 0.2%), every
other shape is worse, and `rel alm` against the shipped config is ≤ 8e-16 everywhere — only
reduction order moves. **Register pressure is not what caps the fused kernel; the default is
already at its optimum and should not be "tuned" again.** Nside 4096 agrees: totals there are
`512/2` 19 237 ms (shipped) and `256/1` 19 086 ms — a 0.8% tie — against 20 937 (`512/4`),
20 466 (`256/2`) and 62 093 (`512/1`, where synthesis blows up to 54.3 s); the 128/64 shapes
were still running against the cell's 40-minute timeout when this was written.

### 3. The band builder is 7x off the fused kernel's arithmetic rate

From the builder probe — which itself never ran, because `timed(fn)` called `fn()` and dropped
the batched theta argument, so every vmap row printed `ValueError: vmap wrapped function must
be passed at least one argument containing batched arrays` — the *shipped* builder's own
numbers are the useful part:

| geometry | one m-block output | time | GiB/s out | implied fp64 rate |
|---|---|---|---|---|
| Nside 1024, fp64, m0=0 | 3.00 GiB | 38.0 ms | 78.9 | ~32 GFMA/s |
| Nside 2048, fp64, m0=0 | 12.00 GiB | 126.3 ms | 95.0 | ~32 GFMA/s |

The fused kernel does the *same recurrence* at ~221 GFMA/s (session 16's 1048 ms cell), so the
builder runs at **~14% of the fused kernel's efficiency**, ~3% of the card's 940 GFMA/s, and
~30x under the 2939 GB/s the contraction gets out of the same bytes. That gap — not the 291
GiB — is what "the build is the wall" actually means.

### 4. The rescale is not computing an exact power of two

`_initial_factor(exponent)` must return exactly `2.0 ** exponent`; the entire point of the
exp2-rescale schedule is that it changes no mantissa bit. The shipped body is

```python
jnp.where(exponent >= -1022, lax.exp2(exponent.astype(jnp.float64)), 0.0)
```

and **`lax.exp2` is not exact on integer arguments** (`.qwen/tmp/exp2_exactness.py`, CPU
backend, all 2046 exponents in [−1022, 1023] compared against `decimal.Decimal(2)**k`):

```
backend=cpu  exponents=2046  values differing: 2023
worst relative error vs exact 2**k:  lax.exp2=7.971e-14 (at k=994)  exponent-field assembly=4.715e-60
exactly representable: lax.exp2=23/2046  assembly=2046/2046
```

Only **23 of 2046** powers of two leave `exp2` unchanged. Because `degree` masks its rescale
with `jnp.where`, which evaluates both arms, this is paid on the whole `(mb, north)` tile at
*every* degree of the scan, not at the 16th degree where the rescale can actually fire.
Writing the biased exponent into the exponent field instead (`bitcast(biased << 52)`, both
range guards preserved) is exact by construction and is a handful of integer ops; it is now
`_theta_matrix._initial_factor`.

Two things had to be measured before this could be shipped, and both are now on disk.
(a) **Accuracy:** the effect on any current score is invisible — ~1e-14 relative on a table
whose accepted `rel` against NaMaster is 3.6e-07 — so this is a correctness fix, not a win,
and it is reported as one. (b) **Speed is backend-dependent**, so neither backend's number
could stand in for the other's. On the CPU the assembly is *slower* than `exp2`
(0.59–0.73x, `.qwen/tmp/renorm_smoke.py`), where `exp2` is close to a hardware instruction;
on the GPU the same A/B against `git show HEAD:gmaster/_theta_matrix.py`, in one process,
clocks spun up, is a win (`.qwen/tmp/renorm_ab.log`):

```
nside    store    m0          builder        ms  x HEAD  GiB/s out              vs HEAD
   512  float64     0             HEAD      14.3    1.00       52.5                    -
   512  float64     0          bitcast      12.2    1.17       61.3        DIFF 3.29e-15
   512  float64  1471             HEAD       0.6    1.00       52.4                    -
   512  float64  1471          bitcast       0.5    1.21       63.3        DIFF 3.31e-15
```

**1.17x on the leading m-block and 1.21x on the tail block, with a 3.3e-15 relative
difference from HEAD exactly as predicted** (HEAD's `exp2` drifting, the assembly not). That
is a real but modest builder gain; *if* it carries to the sizes that matter it turns the 5.5 s
Nside 2048 build into ~4.7 s, which does not by itself open the band route at 2048 — and the
first run died before its fp32 cells on a `NameError: rel_err` left behind by an edit in the
probe itself, so the fp32/fp64 1024 cells are being re-measured
(`queue_s26g.sh` -> `.qwen/tmp/renorm_ab_full.log`) together with the corrected theta-chunk
vmap candidate. Quote those, not the extrapolation, when they land.

### 5. The same inexactness is still in the fused kernel, deliberately untouched

`_sht_pallas._initial_factor` is a second copy of the same body, used for the kernel's
per-(m, theta-chunk) seed scale and inside `_renormalize_with_factor` every 16 degrees. It
contaminates the *fused* path's alms the same way, so it is the same fix. It was deliberately
**not** changed here: three measurement queues were already running against the working tree,
and that copy lives inside a Pallas kernel where `bitcast_convert_type` has to lower through
Triton — a compile risk that must be settled by a GPU test run, not by editing underneath
measurements that are in flight. Next session: apply it, run `tests/test_sht.py` on GPU, and
only then commit.

### 6. Where that leaves the design

The suite with this change in place: **141 passed, 3 skipped in 336 s** on GPU1
(`.qwen/tmp/pytest_s26.log`, `exit=0`). The isolated scores re-run after the fix are the same
within noise, and the accuracy is bit-for-bit the same story as before the fix — `rel alm`
3.7e-07 at 512 and 3.6e-07 at 1024, `rel alm` 3.3e-07 at 256 spin 2:

```
512  0  map2alm  15.5 ->  4.2 ms (3.69x)   alm2map  12.8 ->  4.5 ms (2.87x)
1024 0  map2alm  62.3 -> 28.3 ms (2.20x)   alm2map  54.7 -> 30.4 ms (1.80x)
256  2  map2alm   4.7 ->  5.8 ms (0.82x)   alm2map   2.9 ->  5.3 ms (0.54x)
```

Do **not** read 3.69x/2.20x as an effect of the fix: the runtime contraction never calls
`_initial_factor` (only the build does, and the build sits outside the timed loop), the ducc
reference itself moved by a similar 5-10%, and `rel alm` is unchanged. These are the same
cells as session 25 with a different clock roll.

The 0.22–0.29x `field` stage is the only thing between us and a big-Nside pipeline win, and
section 2 shows shape tuning cannot close a 3.4x gap it has already been measured not to
control. The two shapes from session 16 §5 remain the only routes — a builder fast enough to
sweep once per geometry, and a multi-RHS recurrence shared across the passes that today each
re-derive the same values — and section 3 says which half of the first one is broken.

### 7. Spin 2 at Nside >= 1024: the 110 s is the scatter loop, not a failed build

Spin 2 is the worst thing on the board (0.82x/0.54x at 256, 0.76x/0.59x at 512, 0.04x in the
n1024 pipeline, 0.01x = 110 096 ms per transform at 2048), and a 100x loss looks pathological
rather than merely uncompetitive. The obvious suspect was that `map2alm` pays for a huge polar
build before it declines to "the generic s2fft scatter loop". **It does not**
(`.qwen/tmp/spin2_stage_split.py` at n2048):

```
nside=2048 L=6143 L_work=6143 prec=fp32 reps=2
pool before: 1.0 GiB in use / 71.2 limit
                1. _spin_slabs request first      25.5 ms  min       0.3 ms  [1.0 GiB / 71.2]
   -> returned NONE (generic scatter loop)
        2. _map2alm_core (no slab arg) first  131171.8 ms  min  113680.9 ms  [3.2 GiB / 71.2]
         3. map2alm (public, n_iter=0)  first  126107.7 ms  min  110147.1 ms  [5.5 GiB / 71.2]
                   4. alm2map (public)  first  127067.8 ms  min  111774.1 ms  [7.7 GiB / 71.2]
```

The request settles at 0.3 ms and the pool never moves off 1.0 GiB, so `_spin_slice` declines on
its headroom check without ever trying — nothing is built and thrown away. Step 3's 110.1 s equals
the isolated score's 110.1 s and repeats agree to 0.01 %, so the probe measures the production
call. **The entire 110 s is the generic loop, running at a 7.7 GiB working set.** Against ducc0's
605.7 ms at Nside 2048 that is 182x, where spin 0 at the same Nside is 2x: spin 2 is not losing an
arithmetic race, it is running a different (scatter-based) algorithm — and 7.7 GiB is nowhere near
the memory limit, which is a different situation from the table route's.

The tempting next idea is to ask for less of the polar table, and the arithmetic kills it. A
**single** fp32 polar layout at Nside 1024 is 73.5 GiB (the pair is 147 GiB), against a
71.2 GiB pool — one layout is already bigger than the whole pool, so "build the analysis
layout alone" cannot fit at 1024 either, and the scalar band's 36.8 GiB engagement at that
size is not a template for it: the polar slice spans m in [-(L-1), L), exactly twice the
scalar triangle. That is also why spin 2 works at 512 (pair 18.74 GiB in fp64, 9.37 in fp32 →
the table route runs, 0.76x/0.59x) and stops working at 1024. Four capacity routes were
already tried there in session 12 — `_BUILD_RESERVE` 16/10/6, memory fraction 0.95/0.98,
`cuda_malloc_async`, `PREALLOCATE=false` — and spin 2 stayed at 0.04x.

So spin 2 at Nside >= 1024 is blocked on the same wall as the scalar band, only one octave
sooner, and the Z2 (pi-theta / m-flip) minimum is already shipped: there is no layout trick
left. What remains is the same two things as for spin 0 — a recurrence shared across passes
instead of re-derived per call, or leaving fp64 for tensor cores — plus keeping the generic
scatter loop from being the fallback at all, since it is 100x off rather than 2x off and is
therefore the single largest mispriced thing in the library.

### 8. The builder A/B re-run compares the patch with itself, and the vmap candidate is dead

`queue_s26g.sh` re-ran `.qwen/tmp/renorm_ab.py` after `4e2a179` landed, which quietly made its
own control useless: the script builds its baseline from `git show HEAD:gmaster/_theta_matrix.py`,
and HEAD *is* now the patched file. Every `bitcast` row in `.qwen/tmp/renorm_ab_full.log` is
therefore a self-comparison — it reports `IDENTICAL` and 1.00–1.05x, and says nothing about the
fix. **The fix's numbers stay the ones in section 4** (1.17x/1.21x with DIFF 3.3e-15, taken from
`renorm_ab.log`, which ran against the pre-fix HEAD). If that script is ever run again, pin the
baseline to `4e2a179^:` explicitly.

The rows that *are* new are the theta-chunk vmap candidates, and they close that idea:

```
nside    store    m0          builder        ms  x HEAD  GiB/s out              vs HEAD
   512  float64     0             HEAD      12.4    1.00       60.6                    -
   512  float64     0        vmap-1024      11.0    1.10       34.0        DIFF 3.56e-13
   512  float64     0         vmap-512      12.1    1.00       30.9        DIFF 3.56e-13
   512  float64     0          vmap-32      12.2    0.99       30.8        DIFF 3.56e-13
  1024  float64     0             HEAD      31.7    1.00       94.7                    -
  1024  float64     0        vmap-1024      31.7    1.00       47.4        DIFF 3.52e-12
  1024  float64     0          vmap-32      31.7    1.00       47.4        DIFF 3.52e-12
  1024  float32     0             HEAD      28.4    1.00       52.7                    -
  1024  float32     0        vmap-1024      28.5    1.00       26.3        DIFF 5.96e-08
```

Two cautions before reading it. The `GiB/s out` column in the vmap block is computed from the
*parity-sliced* reference, so it undercounts by 2x by construction — only the **ms** and
**x HEAD** columns are comparable across the two blocks. And `_slab_subset` (defined inside the
probe) seeds both parities from one value at `ell = m`, while production `_build_slab` takes a
separate `_parity_seed`; that is the whole 3.5e-13/6.0e-08 `DIFF`, not a rounding mystery, and it
means the vmap arm is the *cheaper* recurrence.

So the honest reading is: shortening the per-program carry from `(64, north)` to `(64, chunk)` by
vmap-ing over theta is worth **nothing at Nside 1024** (1.00x at every chunk size, in both
precisions) and at most 1.10x at 512, doing less arithmetic per call than the production form.
Declined.

What that leaves is a clean diagnosis. A slab build at 1024 runs at 94.8 GiB/s out, about 6 % of
the 1513 GB/s this card sustains, and neither the cheapest rescale op nor any amount of theta
batching moves the wall time. The builder is neither FMA-bound nor bandwidth-bound: it is bound by
the length of the sequential degree chain in `_build_slab`, which is exactly the thing
`blocked-recurrence-wall` says cannot be shortened in fp64 and `complex-rhs-tax`/`contraction-layout-tax`
say cannot be hidden in the contraction. Only the two shapes in section 6 remain — build once per
geometry, or share one chain across several right-hand sides.

## Session 18 (2026-09-05) — the rescale guard is live, three builder ideas die, and the block size is a capacity knob

Nothing shipped. Four probes, one of which found a bug in an idea I had already reported as a win,
and a measurement that finally says what `_build_slab` is bound by.

### 1. The exp2-rescale guard is not a vestige: it fires millions of times and its carry blows the trip point by 2**37

`.qwen/tmp/renorm_firing.py` walks the shipped march and reduces it to guard statistics — firing
count, and the extreme |carry| over *every* degree rather than only the checked ones (CPU, all
m-blocks):

```
nside    m0  block   fires      log2 pmax     log2 pmin    slack
  256      0    64    3062         108.7        -4.8     -8.7
  256    448    64   60565         137.0        -2.6    -37.0
  256    704    63    7858         136.6        -1.5    -36.6
# nside 256: L=767 blocks=12 total fires=468827 worst slack=-37.0 decades
# nside 512: L=1535 blocks=24 total fires=4352493 worst slack=-38.1 decades
```

The trip point is `2**100`; the carry runs to `2**137`, and one m-block at Nside 512 takes
221,590 rescales. So the guard is doing constant work, and the "value-preserving" claim in the
docstring is load-bearing rather than decorative — which is also why the session-17 discovery that
`lax.exp2` was not exact mattered at all. Any "make the guard cheap" idea has to preserve it
exactly, not approximately.

### 2. Grouping the degrees so the check runs 1/16 as often: declined, and it is slower on the GPU

`.qwen/tmp/builder_grouped.py` nests the scan — inner body is the bare recurrence, outer body
checks and rescales once per 16 degrees. On the CPU at small sizes it looks excellent (1.30-2.80x
at Nside 64, 1.33-1.70x at 128) and then stops: 0.92x at 256 fp64 (`builder_grouped_cpu2.log`),
and on the GPU (`builder_grouped_gpu.log`, clocks spun up, one process on GPU1):

```
nside store    m0      shipped     grouped      x   GiB/s out  identical
  512  float64     0       12.31       12.78   0.96       58.6           YES
  512  float32     0       11.68       11.91   0.98       31.5           YES
 1024  float64     0       28.52       35.75   0.80       83.9           YES
 1024  float32     0       26.29       31.72   0.83       47.3           YES
 1024  float64  3007       0.59        0.68   0.87       91.8   NO rel 2.15e-265
```

0.80-0.83x at the leading block — the block that dominates a build — in both storages. XLA
compiles the nested scan worse than the flat one, by more than 25 guard ops per degree are worth.
The scan stays flat; do not restructure it again.

### 3. Making the guard's rescale cheap: the idea was wrong, and the GPU could not have told me

The shipped body recomputes the scale from the exponent,
`factor = where(large | small, _initial_factor(exponent), factor)`, which is ~10 ops (int64 cast,
add, clip, shift, bitcast, two compares, two selects, one select). `exponent` only ever moves by
exactly ±100 there, so multiplying the existing `factor` by the reciprocal of the carry's `mult`
should give the same power of two, and would take `exponent` out of the loop carry.

The first version multiplied by `mult` instead of its reciprocal. It is wrong by O(1) —
`rel` 7.03e-01 at Nside 64, 1.00e+00 at Nside 128 — and on the GPU it measured

```
  512  float64     0      11.90    1.28       61.0  fires 0  IDENTICAL
 1024  float64     0      25.28    1.14      118.8  fires 0  IDENTICAL
```

**`fires 0` and `IDENTICAL` because the leading m-block is one of the few places the guard never
wakes up** (section 1: at 1024 the block at m0=3007 fires 51,921 times, m0=0 fires 13,433). A
GPU-only A/B on the leading block would have shipped a 1.14x "win" that zeroes half the Legendre
table. The CPU sweep at `m0 = L//4` is what caught it.

The corrected version (`builder_cheapguard_cpu2.log`, `builder_cheapguard_gpu2.log`; `gate+inv`
keeps the shipped every-16th-degree phase, `gate+exp2` is the shipped semantics as a control):

```
nside store   m0   arm          shipped      cheap      x  GiB/s out  fires  identical
  512  float64     0 gate+inv       10.67     11.18   0.95       67.0     6480         YES
  512  float64  1471 gate+inv        0.51      0.38   1.34       82.7    21393  NO rel 1.32e-250
 1024  float64     0 gate+inv       27.24     27.26   1.00      110.0    13433         YES
 1024  float64  3007 gate+inv        0.60      0.60   0.99      103.4    51921  NO rel 6.24e-247
 1024  float32     0 gate+inv       26.27     26.29   1.00       57.0    13433         YES
```

**1.00x at Nside 1024 in both storages** — the geometry where a build is actually paid — and
1.26-1.38x only on 0.5 ms tail blocks. The control arm is 1.00x as it should be. On the CPU the
same change is worth 1.10-2.13x, which is the third time this week the two backends have disagreed
about an op-count change on this recurrence.

And it is not bit-identical. `.qwen/tmp/cheapguard_diffmap.py` says why, and the reason is a
correctness argument rather than a rounding argument:

```
   m0  n_diff   rel vs max      |a| at worst   |b| at worst   zero-one-side
    0   23663   6.80e-231      -2.803e-230       -0.000e+00          23663
  256    8744   1.25e-241       1.653e-241        0.000e+00           8744
```

Every differing entry is exactly zero on the reciprocal side. A `factor` accumulated by
multiplication stays zero once a step has underflowed it; `_initial_factor(exponent)` re-derives
`2**exponent` from the exponent and recovers on the next degree. The differing entries here are
~1e-230 relative to the slab maximum and physically meaningless, but the failure mode is not
bounded — a lane whose factor passes through the underflow range emits zeros for the rest of its
march. **`factor` must be derived from `exponent`, not accumulated alongside it.** Declined, and
the shipped form is right for a reason that was not written down before today.

### 4. `BLOCK` is a capacity knob, and 64 is not arbitrary

The tile sweep below says a wider m-block should build faster per emitted value, so `_theta_matrix.BLOCK`
(64, with no rationale comment) was set to 256 and the isolated fp32 score re-run
(`sht_block256_fp32.log`, same script as `sht_s26k_fp32.log`):

| Nside | BLOCK=64 (shipped) | BLOCK=256 |
|---|---|---|
| 512 map2alm / alm2map | 15.5 -> **4.4 ms (3.53x / 2.56x)** | 14.9 -> 4.6 / 5.1 ms (3.21x / 2.40x) |
| 1024 map2alm / alm2map | 58.0 -> **28.1 ms (2.06x / 1.79x)** | 60.3 -> 124.8 / 162.8 ms (**0.48x / 0.33x**) |

`rel alm` is unchanged (3.7e-07 / 3.5e-07) — the band values do not depend on the block boundary,
as they should not — but at Nside 1024 the score collapses by 4.4x, and 124.8 ms is exactly the
fused-kernel cell measured in session 17 (125.6 ms). `.qwen/tmp/block_band_check.py` reproduces it
independently and prints the mechanism (`block_band_check.log`):

```
=== BLOCK=256   band_bytes block=BLOCK(256): 38.97 GiB   block=default: 36.73 GiB   budget=40 GiB
                _prefer_theta_band=True
map2alm first=282984.8 ms best=125.0 ms
_band calls: [((1024,3072,256,float32),'NONE',277747 ms), then four more 'NONE' at 0 ms]
_BAND_CACHE keys=[] _BAND_FAILED=[(1024, 3072, 256, <class 'jax.numpy.float32'>)]
=== BLOCK=64    band_bytes block=BLOCK(64): 36.73 GiB   block=default: 36.73 GiB   budget=40 GiB
                _prefer_theta_band=True
map2alm first=346306.8 ms best=28.1 ms
_band calls: [((1024,3072,64,float32),'built',340522 ms), then four more 'built' at 0 ms]
_BAND_CACHE keys=[(1024, 3072, 64, <class 'jax.numpy.float32'>)] _BAND_FAILED=[]
```

So the gate **approved** and the *build* failed: `_band` caught `RESOURCE_EXHAUSTED`, negative-cached
the geometry, returned None, and the transform took the fused kernel. Two things follow that are not
obvious from the score alone:

* **The fit test does not see the block size.** `_prefer_theta_band` (utils.py:762) calls
  `band_bytes(nside, L, dtype=...)`, and `band_bytes`'s `block` parameter has `BLOCK` as a *default
  argument*, bound at import. Setting `tm.BLOCK = 256` therefore changes what gets allocated
  (38.97 GiB) but not what gets gated (36.73 GiB). With `BLOCK = 64` in the shipped code the two
  agree, so nothing is wrong today — but any BLOCK change silently bypasses the budget test, and the
  real footprint is what the pool votes on.
* **A failed build is expensive.** The 256-wide attempt ran **277.7 s** before it raised; the score
  script's `first` call paid it. The negative cache then correctly keeps the other ~7 passes from
  retrying, which is why the steady-state 125.0 ms looks like an ordinary fused-kernel cell.

The peak is what breaks, not the resident size: with 12 blocks instead of 48, `_BUILD_CLEAR_EVERY = 8`
fires once instead of six times, so more executables are alive pinning their outputs
(`executables-pin-their-outputs`) on top of a table that is itself 2.2 GiB larger. The wide build is
also slower per block — 277.7 s for 12 blocks (23 s/block) against 340.5 s for 48 (7.1 s/block) —
because each program now emits a 256-lane march rather than a 64-lane one, so even the attempt that
succeeded in the control leg pays a third more warm-up time for a band that serves the same 3072
degrees.

So `BLOCK` trades build shape against *route availability*, and at 1024 the route is worth far more
than the build. **64 stays. Anyone widening it must pass the block size to `band_bytes` in
`_prefer_theta_band` and re-check the budget at every Nside that currently engages.** Reverted;
`git diff` is clean for `gmaster/`.

### 5. What `_build_slab` is actually bound by: a ~5 us floor per degree

`.qwen/tmp/builder_tile_scaling.py` holds the degree count fixed (3071, Nside 1024) and scales the
lanes per degree, so a fixed cost shows as a flat line and real work as a slope:

```
nside store  rows   mb  north    lanes/deg    ms  us/deg  xlanes    xms   exp
 1024 float32  3071   16   512        8192    15.31     4.98    1.00    1.00   nan
 1024 float32  3071   32   512       16384    14.96     4.87    2.00    0.98  -0.03
 1024 float32  3071   64   512       32768    23.16     7.54    4.00    1.51   0.30
 1024 float32  3071  256   512      131072    25.80     8.40   16.00    1.69   0.19
 1024 float32  3071  256  2048      524288    67.06    21.84   16.00    3.20   0.42
```

The exponent is **0.16-0.42**: 64x the lanes costs 4.4x the time. Reading it as
`time = 15.3 ms + lanes x 3.4e-8 ms` gives a **fixed ~5 us per degree** and a marginal rate of
~29-34 G emitted values/s. That marginal rate writes ~116 GB/s (8 % of the 1400 GB/s the
contraction gets from the same bytes) and does ~3 FMA per value at ~10 % of the 940 GFMA/s fp64
peak — so the scan is **neither bandwidth-bound nor op-bound**, and the numbers now say why every
attempt to speed it up failed:

| lever | op count | emitted bytes | source |
|---|---|---|---|
| coefficients as scan rows | down | same | session 18 (shipped: 1.09-1.20x at 512, 1.00x at 1024) |
| `_initial_factor` exponent-field | down | same | session 17 §4 (1.17x at 512) |
| rescale guard ~25 -> ~10 ops | down 40 % | same | this session: **1.00x at 1024** |
| check 1/16 as often | down | same | this session: **0.80x at 1024** |
| fp32 storage | same | **halved** | 27.24 -> 26.27 ms (0.97x) |
| vmap over theta chunks | same | same | session 17 §8: 1.00x |

The bound is the scan itself: a `(mb, north)` fp64 carry (1 MB per tile at BLOCK=64, four arrays)
that XLA shuttles through memory once per degree, one dependent chain, ~5 us per step. The only
march on this card that does not pay that is the **Pallas fused kernel** (221 GFMA/s, carry in
registers). So "make the build fast" is not an XLA problem at all — it means writing the band
emitter as a Pallas kernel, which is the same march `_sht_pallas._analysis_kernel` already runs,
with a store in place of the block reduction. Section 8 puts the cheap test that has to come first:
one negative measurement of exactly that shape already exists.

### 6. What that is worth: the streamed-band budget, recomputed with measured numbers

At Nside 2048 the fp32 scalar band is ~288 GiB against a 97.9 GiB card (97887 MiB, `nvidia-smi`),
so the band route cannot be resident and the fused kernel runs: **0.32x / 0.23x** isolated
(`sht_s26k_fp32_2048.log`: ducc 303.3 -> GM 953.2 map2alm, 287.2 -> 1264.0 alm2map, `rel` 3.7e-07)
and 0.22x / 0.17x at 4096 (`block_sweep_4096_0_fp32.log` shipped shape 8352.78 / 10884.38 ms
against `ducc_only_4096_0.log` 1840.97 / 1865.26 ms). The streamed shape — build one m-block,
contract every pass against it, free it — needs no memory it cannot get:

* build the whole band once: 7.5e10 emitted values at the measured marginal rate ≈ **2.6 s**, and
  ~6 s if extrapolated from the measured 2048 leading block (126.3 ms, `builder_nogather.log`)
* contraction per pass: 288 GiB at the measured 1400-1474 GiB/s ≈ **0.22 s**
* fused kernel per pass: **1.04 s** (8312 ms / 8 passes at 2048, session 17 §1)

Break-even is therefore k ≈ 4 passes today, and for the 8 passes of the real `field` stage a
streamed band is 6 + 1.8 = 7.8 s against 7.6 s fused — **a wash**. With the builder 4x faster
(a Pallas emitter at the fused kernel's own rate, ~1.5 s) the same stage becomes 3.3 s, the whole
2048 pipeline 3.3 + 1.53 = 4.8 s against the CPU's 10.46 s, i.e. **~2.2x end to end where it is
0.57x today**. This is the one remaining shape, and section 5 says precisely which piece of it is
missing: not capacity, not the contraction, not the guard — the emitter.

### 7. Multi-RHS: the measured verdict, and why it converts bandwidth rather than arithmetic

`.qwen/tmp/multi_rhs_contract.py` (GPU1, one process, `multi_rhs_contract.log`) extends
`_contract_theta`'s tuple reduce from 2 accumulators to 2k over one slab:

```
nside store   k        ms  x k=1  GiB/s slab   x stacked
 1024  float32   1      1.02   1.00     1474.5      1.02
 1024  float32   2      1.04   0.98     1441.0      1.00
 1024  float32   4      1.07   0.95     1400.0      0.97
 1024  float32   8      1.75   0.58      856.1      0.60
 1024  float64   1      2.06   1.00     1458.8      1.00
 1024  float64   4      4.61   0.45      650.2      1.00
 1024  float64   8      9.39   0.22      319.4      0.97
```

In the fp32 storage that production uses, **4 right-hand sides cost +5 %, 8 cost +72 %**; in fp64
it is dead linear. Read against the machine roofline (balance 0.58 FMA/byte, so a 2-FMA-per-element
scalar band tops out at 378 GFMA/s) the contraction is already at 1400-1474 GiB/s of the 1633 GB/s
read peak — **it is at its bandwidth ceiling, which is exactly why extra channels are almost
free**: they add arithmetic the card has spare, not bytes it has to fetch. The k=8 fp32 break and
the fp64 linearity are the accumulator count capping the reduce, not bandwidth.

Design consequence: batch ~4 channels per contraction program (two programs if 7 passes are
simultaneous), keep the tuple-reduce spelling — the stacked form is the 0.97-1.02x column above
and the 12.16-vs-8.14 ms trap in `_contract_theta`'s docstring — and the pipeline's repeated
transforms against one band cost roughly the price of one.

What is *not* batchable, checked rather than assumed: the `n_iter` loop in
`_map2alm_core_pallas` (utils.py:1760-1775) is a recurrence — each iteration's analysis consumes
the previous iteration's synthesis — so those passes cannot share a read. The batchable set is
independent fields: the mask-power transforms inside `compute_coupling_matrix`, and several maps at
the API level. The API is the blocker: `map2alm` accepts exactly `(1, npix)` for spin 0 and
`(2, npix)` for spin 2 (utils.py:1976-1981) and `_map2alm_once` reads `maps[0]`, so there is
today no way to hand GMaster two maps that could share one band read.

### 8. Next work, in the order the measurements now support

1. **Pallas band emitter** (section 5/6): the march from `_analysis_kernel` with a store instead of
   the reduction, one m-block resident at a time, contraction in the existing
   `_contract_theta`. **Prior art is negative, so gate it on the cheapest possible test first:** a
   Pallas band-fill kernel has already been measured once — correct to 3.96e-14 and **33x slower**
   than the XLA builder (196 ms vs 5.8 ms for a 0.19 GiB block). Before any march work, write the
   store-only fill (known values in, slab out) and check it reaches XLA's ~30 G values/s; if it does
   not, the emitter idea dies there rather than after a week.
   Gate the whole shape on: build the 2048 band in under ~2 s, then re-measure the 2048 `field`
   stage end to end. Nothing else on the board turns 0.5x into >1x.
2. **Batched multi-RHS contraction** (section 7): 2k accumulators, k <= 4, plus the API change that
   lets independent maps arrive together. This one pays at <= 1024, where the band route already
   wins 1.79-3.53x, and it is the cheapest big number on the board.
3. Do **not** revisit: scan restructuring (0.80x), guard op counting (1.00x), fp32-as-a-speed-lever
   (0.97x), theta vmap (1.00x), thread-block shapes (session 17 §2), `BLOCK` widening (section 4),
   a Pallas band fill *without* first clearing the store-rate bar (previous attempt: 33x slower).
   Every one of those has a measurement attached.

### 9. Measured after section 8 was written: the store gate passed and the emitter exists

Both probes below ran after the text above, and they change what item 1 means.

**The store gate** (`.qwen/tmp/pallas_fill_gate.py`, `pallas_fill_gate.log`, GPU1, one process): a
Pallas kernel that writes the band's own slab layout `(mb, ncol, north)` fp32 with cheap index
arithmetic and no recurrence at all.

```
Pallas chunk=  1 warps=2 (64, 512, 2048): 0.64 ms   393.6 GiB/s  105.7 G values/s  x builder 3.39
Pallas chunk=  8 warps=2 (64, 512, 2048): 0.27 ms   936.4 GiB/s  251.4 G values/s  x builder 8.07
Pallas chunk= 64 warps=2 (64, 512, 2048): 0.22 ms  1121.7 GiB/s  301.1 G values/s  x builder 9.67
XLA fill (same buffer)                    : 33.46 ms   7.5 GiB/s    2.0 G values/s  x builder 0.06
```

**9.7x the builder's marginal emission rate on the store side alone.** The `XLA fill` row is *not* an
XLA store ceiling — the broadcast three-way add did not fuse into a store loop — so do not quote it;
the comparison that matters is against the builder's measured 116 GB/s. It also means the old
"A Pallas band-fill kernel is 33x slower" line (memory: *Alternatives landscape*) cannot be a property
of Pallas stores: store granularity alone moves this probe by 3x, and the number has the signature of
the untraced-`pallas_call` artefact recorded in section 5 of the session-15 notes.

**The emitter exists and reproduces the shipped slab.** `.qwen/tmp/pallas_band_emitter.py` carries the
`_build_slab` recurrence in registers — one program per (m-lane, theta chunk), `lax.fori_loop` over
`ell`, storing every degree — and it wins wherever it has been measured:

```
Nside 256, m0=0, fp32 (`pallas_band_emitter7.log`)
  XLA builder     : 5.36 ms   17.5 GiB/s   4.7 G values/s
  Pallas chunk=128: 0.35 ms  268.6 GiB/s  72.1 G values/s   x this XLA 15.4   rel 1.5e-08
Nside 1024, m0=0, fp32 (`pallas_band_emitter_n1024.log`)
  XLA builder     : 26.25 ms  57.1 GiB/s   15.3 G values/s
  Pallas chunk=128:  3.15 ms 476.2 GiB/s  127.8 G values/s   x this XLA  8.34  rel 5.1e-09
  Pallas chunk=256:  3.27 ms 459.3 GiB/s  123.3 G values/s   x this XLA  8.04  rel 5.1e-09
  Pallas chunk=512:  4.50 ms 333.4 GiB/s   89.5 G values/s   x this XLA  5.84  rel 5.1e-09
```

The `rel` figures are the float32 storage quantum, and the Nside 256 diff map has **0 of 25,133,056
entries** above 1e-4 relative (max absolute 1.49e-08 on a value of 1.905312e-01,
`.qwen/tmp/emitter_diffmap.py`). It is the same table, emitted **8.3x faster** at the size that
matters, at **127.8 G values/s** against the **37 G values/s** section 6 says the streamed Nside 2048
shape needs. If that rate holds across a whole band, the 2048 build is ~0.6 s rather than 2.6-6 s, the
streamed `field` stage is 0.6 + 8 x 0.22 = 2.4 s against the fused kernel's 8.3 s, and the session-17
pipeline arithmetic (2423 -> 8312 ms today, ref 2423 ms) moves to roughly parity or better for the
first time above 1024.

Two Triton-lowering facts, each worth an hour, that any future kernel here must respect:

* `jnp.where(pred, -1.0, 1.0)` with a **scalar** `pred` whose result feeds a vector multiply does not
  lower at all: `AssertionError: ('tensor<128xi1>', 'tensor<128xf64>')` — XLA canonicalises the
  select into an i1×float multiply. The shipped kernel's
  `jnp.where(diagonal < 0, -jnp.ones_like(sine), jnp.ones_like(sine))` is therefore load-bearing
  style, not decoration. Bisect: `.qwen/tmp/pallas_emitter_bisect.py` — `fori_store_only`,
  `fori_scalar_load` and `vecwhere_only` all lower; the scalar-armed `where` variants all fail.
* `lax.cond` inside `lax.fori_loop` works, but **both branches must advance the recurrence**. A
  "do nothing" branch that returns the incoming carry freezes the march on 15 degrees out of 16: that
  is the whole of the `rel` 2.2e+00 first result (22.7M of 25.1M entries wrong, first divergence at
  `ell = m + 3`), and no error message accompanies it.
* `pl.range` does not exist in jax 0.10 (`pl.loop` does); the codebase idiom is `lax.fori_loop`.

What is *not* claimed yet: nothing is wired in. The next three steps, in order — (a) build a **whole**
band through the emitter and compare against the shipped 340 s at Nside 1024
(`block_band_check.log`), since the per-block win has to survive 48 launches, the parity split
(`_build_pair` returns `vals[0::2], vals[1::2]`, which the emitter currently does not do) and the
`_BUILD_CLEAR_EVERY` cache churn; (b) the isolated `sht_vs_ducc` cells at 512/1024 with the emitter
inside `_band`, which also needs the GPU suite green; (c) the streamed shape at 2048, where the
emitter's 127.8 G values/s is the first measurement to clear the 37 G values/s bar. Nothing about the
`field` stage score is measured yet — 0.29x at 2048 is still the shipped number until (c) runs.

### 10. The emitter is shipped, and it turns out the band build was compile-bound all along

`282ca21 perf(sht): emit the Legendre band with one Triton program per m-block` —
`gmaster/_band_pallas.py`, selected by `_theta_matrix._band` (default; `GMASTER_BAND_BUILDER=scan`
or `tm.set_band_builder("scan")` gets the old builder back).

The whole band, parity split included, `.qwen/tmp/pallas_band_fullbuild2.log`:

```
nside  256  cold   1.35 s | warm 0.005 s   0.61 GiB out   34.6 G values/s   diff vs scan 8.9e-16
nside  512  cold   2.65 s | warm 0.014 s   4.68 GiB out   90.3 G values/s   diff vs scan 2.3e-10
nside 1024  cold   5.14 s | warm 0.080 s  36.73 GiB out  117.3 G values/s   diff vs scan 6.0e-08
          (eager `_build_pair` over the same blocks: 3.46 / 7.72 / 28.67 s -> x733 / x555 / x341)
```

**The parity split costs nothing**: at Nside 256 m0 = 0 one unmasked store per degree is
37.8 G values/s and the two masked stores per degree that do `vals[0::2]` / `vals[1::2]` inline are
37.4 (`.qwen/tmp/fullbuild_probe.log`), so there is no separate pass and no transpose. What the
first whole-band run read as "0.4 GiB/s warm" was a measurement artefact — an intervening
`jax.clear_caches()` evicted the Triton programs, so the "warm" pass was recompiling all 12 kernels.

Correctness, `.qwen/tmp/band_equivalence.py`, every entry of every block, both storages:

```
fp32 nside   64 (north 128, exact tile)  max abs 0.000e+00   padding violations 0
fp32 nside  100 (north 200, MASKED tile) max abs 0.000e+00   padding violations 0
fp64 nside   64 / 100                    rel   1.9e-14 / 2.1e-14
```

Bit-identical in float32, and the float64 difference is the march-order floor. `nside 100` is the
geometry that makes `north = 200` not a multiple of the 128-lane theta tile, so the masked path is
covered rather than assumed. Padding rows (`ell < m`) are written zero explicitly — a pallas output
buffer is not zeroed, and the scan writes `0.0` there through `where(ell >= mi, …)`.

**Why the win is 60x and not 8x.** The scan builder is not store-bound and, at build scale, barely
march-bound: `block` m-blocks with a static `m0` is `block` XLA programs, so a band build pays a full
compile per block and each program pins a copy of what it produced. That is what the 340 s was
(`block_band_check.log`), against well under a second of device work. Triton compiles the same 48
blocks in 5.1 s. Section 8's "the builder is latency-bound" was true of the *march* and is now moot:
the march runs in registers at 117 G values/s and the compile is what was left.

The number that matters is the cold pipeline, `.qwen/tmp/band_route_score.py`,
`GM_PREC=fp32`, one geometry per process:

```
nside  512  route=band  cold map2alm first=  6224 ms  best=  4.7 ms | alm2map first= 5558 ms best=  4.8 ms
nside 1024  route=band  cold map2alm first= 11594 ms  best= 28.1 ms | alm2map first= 5777 ms best= 31.0 ms
          build alone, Nside 1024: 6.11 s = 36.75 GiB at 6.0 GiB/s (48 blocks, 48 emitter compiles)
```

and the same-process ducc cells, `.qwen/tmp/score_emit_fp32.log`:

```
   512  0  map2alm   15.3 ms  ->   4.6 ms   3.29x    rel 3.7e-07
   512  0  alm2map   12.1 ms  ->   4.8 ms   2.51x
  1024  0  map2alm   56.3 ms  ->  28.3 ms   1.99x    rel 3.6e-07
  1024  0  alm2map   53.6 ms  ->  31.0 ms   1.73x
```

A first `map2alm` at Nside 1024 float32 used to be dominated by 346 s of table build
(`block_band_check.log`: `first=346306.8 ms best=28.1 ms`). It is now **11.6 s** including the build,
and the warm transform is unchanged at 28.1 ms. The band route stops being something you can only
afford in a benchmark that throws away its first call.

Three traps, each of which produced a plausible-looking wrong answer:

* `jax.ensure_compile_time_eval()` — which `_band` used so the scan folds into the caller as a
  compile-time constant — makes `jit` evaluate eagerly, and a `pallas_call` has no eager rule:
  `NotImplementedError: Evaluation rule for 'program_id' not implemented`. `jax.disable_jit(False)`
  nested inside does not undo it (the fold is a trace-time flag, not the jit setting). The emitter
  route runs in a `contextlib.nullcontext()` and relies on the concrete-buffer check that already
  rejected a build inside a trace, so gradients still take the fused kernel exactly as before.
* A "fall back to the scan if the emitter raises" wrapper made the first `pytest tests/ -q` after
  the wiring report **141 passed** while every band in it came from the old builder. The route is
  now asserted, not assumed: `.qwen/tmp/band_route_engaged.py` counts emitter compiles against
  block count (`6 blocks, emitter compiles 6`) and checks the transform against ducc0
  (rel 2.7e-07 at 128, 3.3e-07 at 256). Suite after the fix: **141 passed, 3 skipped in 308.35 s**
  (`pytest_s18emit.log`).
* A probe that builds `band_geometry(nside, 3*nside-1)` by hand is not the pipeline's geometry — the
  pipeline wants `(nside, 3*nside, BLOCK, dtype)`. Two geometries means two 36.7 GiB tables on a
  71 GiB pool, the second build dies, and `map2alm` answers **122 ms** (fused kernel) instead of
  28 ms. `band_route_score.py` reads the cache keys back instead of guessing them.

What this does *not* change: the band is `d^l_{m,0}`, so spin 2 still uses the Wigner-d tables and is
untouched; Nside 2048 float32 needs 147 GiB and cannot be resident at any builder speed, so there the
`field` stage stays at its shipped number until the streamed shape runs. The streamed band is now
the only remaining use for a table this big, and 117 G values/s is 3x the 37 G values/s bar
section 6 set for it — but note the arithmetic in section 11 before spending time there.

## Session 19 (2026-09-06) — spin 2 stops losing: the polar slice is read once, at the storage width

### 11. The polarised contraction was four reads and the wrong precision

Spin 2 was the standing complaint for four sessions and the measurements agreed with it: every
per-pass spin-2 cell lost to ducc0, and section 15 had already proved the ring stage was not the
problem (0.34 ms of a 34 ms `map2alm`). What was left is the polar latitudinal contraction in
`_spin_slice`, and it had two defects, only one of which was structural.

**New module `gmaster/_spin_contract_pallas.py`, dispatched by `_spin_slice._kernel_contract`.** One
Triton program owns a tile of outputs, loops the reduction axis, loads a block of the slice **once**
and multiplies it by all four real right-hand sides with four register accumulators. Measured on the
block set, float32 tables, controls 1359-1377 GB/s (`.qwen/tmp/spin_contract_tune9.log`):

| geometry | direction | `_spin_slice` (XLA) | fused kernel | | against the byte floor |
|---|---|---|---|---|---|
| Nside 512 | analysis | 31.69 ms | **6.97 ms** | **4.55x** | 1444 GB/s of 1633, floor 6.16 ms → **88 % of the pure read** |
| Nside 512 | synthesis | 33.39 ms | **7.26 ms** | **4.60x** | |
| Nside 256 | analysis | 4.44 ms | **1.18 ms** | 3.77x | |
| Nside 256 | synthesis | 4.77 ms | **1.19 ms** | 4.01x | |

and the cells the user actually scores (one Nside per process, `GM_PREC=fp32`, ducc0 0.39.1 on all
192 cores):

| nside | direction | before (session 15) | now | ducc0 | speedup | rel alm | ctrl GB/s |
|---|---|---|---|---|---|---|---|
| 256 | `map2alm` | 5.9 ms, 0.83x | **2.3 ms** | 5.0 | **2.17x** | 3.2e-07 | 1369/1352 |
| 256 | `alm2map` | 5.5 ms, 0.52x | **1.7 ms** | 3.2 | **1.90x** | — | |
| 512 | `map2alm` | 34.2 ms, 0.76x | **9.4 ms** | 28.1 | **3.00x** | 3.2e-07 | 1384/1364 |
| 512 | `alm2map` | 37.6 ms, 0.59x | **8.6 ms** | 22.5 | **2.60x** | — | |

`.qwen/tmp/sht_spin2_solo_256.log`, `.qwen/tmp/sht_spin2_solo_512.log`. Four cells that lost are now
four that win. The whole-MASTER pipeline (`.qwen/tmp/pipeline_spin2_{256,512}.log`):

```
 nside spin  prec   NaMaster    GMaster  speedup        rel  ctrl GB/s
   256    2  fp32      170.1       32.2    5.29x    1.2e-07  1338/1420
   512    2  fp32      684.5      148.0    4.62x    3.1e-07  1448/1423
   512    0  fp32      331.9       93.8    3.54x    1.6e-07  1424/1409   <- spin 0 unchanged (was 93 ms, 3.45x)
```

Spin-2 at Nside 512 was 332 ms / 2.03x before this change and is **148 ms / 4.62x**; spin 0 did not
move, which is the regression check that the dispatch touches only the polar path. Suite: **141
passed, 3 skipped in 328.10 s** (`pytest_spin2_pallas.log`), the same counts as session 18's 308.35 s.

**Three things had to be true. Only the first was the idea.**

* **One read instead of four.** `_reduce_channels` (session 10) already removed the complex-RHS tax,
  but a tuple `lax.reduce` still visits the block set once per channel.
* **Products in the storage precision, float64 only for the tile partial.** The careful spelling —
  `acc += v.astype(f64) * r.astype(f64)` — runs the *float32* slice at **210 GB/s against a 1361 GB/s
  control**. Same kernel, same tile, float32 products: **1065 GB/s** (`spin_contract_tune3.log` →
  `spin_contract_tune5.log`). `cvt.f32.f64` issues at a fraction of the FMA rate and the RHS is loaded
  four times per table element, so per-element widening is a 5x loss, not free caution.
* **The right-hand side is never widened in memory.** Synthesis is handed `alm` as complex128 because
  that is NaMaster's convention; casting the four channels to the slice's own width instead is worth
  **48.85 → 7.49 ms (6.5x)** at Nside 512 at an unchanged 7.1e-08 relative difference
  (`spin_contract_tune5.log` → `spin_contract_tune8.log`).

**Dead end 1 — the mask was never the problem.** The reduction axis is `4·nside − 1`: odd, and sharing
no useful factor with a power-of-two tile, so a single-level loop always ends in a masked tail.
Splitting that tail into progressively smaller *unmasked* tiles (128, then 16) — the obvious fix — is
**3x worse**: one fully masked 1024-element tile does Nside 256 analysis in 1.15 ms, the decomposition
in 3.44 ms. Predicated loads are cheap on this card; short vector loads are expensive. The lever is
tile *length*, and it saturates: at Nside 512, analysis runs 22.79 / 13.56 / 8.97 / 7.53 / **6.97** /
7.21 ms at chunk = 64 / 128 / 256 / 512 / **1024** / 2048. Locked: `_E_TILE = 8`, `_RED_CHUNK = 1024`,
`num_warps = 2` (2 beat 4 in every cell).

**Dead end 2 — float64 tables.** With fp64 storage the products must be fp64 and Triton's fp64 path
does not reach XLA's: **0.85x** (0.93 vs 0.79 ms) analysis, **0.66x** (1.15 vs 0.76) synthesis at
Nside 128, control 1459 GB/s (`spin_contract_tune10_fp64.log`). So `_kernel_contract` requires
`slab[0].dtype == jnp.float32`: the **default fp64 configuration is untouched**, and
`.qwen/tmp/spin2_route_assert.py` asserts both halves — pallas engaged at fp32, absent at fp64 —
because session 18 established that a route you infer instead of reading out of the kernel cache is a
route that silently stops being used. `GMASTER_POLAR_CONTRACT=xla` remains the supported way back.

**Accuracy.** Relative to the XLA form the kernel moves the answer by 6.3e-08 at fp32 storage and
5.8e-16 at fp64; the pipeline's `rel alm` against ducc0 is 3.1-3.2e-07, the same order the fp32 route
has carried since session 13 (1.1-1.8e-07) and dominated by the fp32 table and complex64 ring, not by
this contraction. The fp32 products are exact where both operands are already float32; what is added
is the float32 tree-sum inside a 1024-element tile, then float64 across tiles.

**The arithmetic section 10 promised, before anyone builds a streamed band.** At Nside 2048 the spin-0
band is 291.00 GiB = 78.11 G values; at the emitter's measured 117.3 G values/s that is **0.67 s to
emit**, against a ducc cell of 301.5 ms — so a streamed band that writes and then reads the table
cannot win at 2048, and no streamer is implemented. Same arithmetic for the case that matters now,
spin 2 at Nside 1024 (L = 3072, nθ = 4095, 1.93e10 table values, layout 73.48 GiB — declined by
`_SLAB_BUDGET` 56 GiB, which is why every logged Nside 1024 spin-2 cell is 0.02-0.04x):

* contraction alone at the measured 1444 GB/s: **54.8 ms** vs ducc's 119.3 ms — a 2.2x win, *if the
  values were free*;
* emitting them first, even at 117 G values/s: +166 ms → 0.54x. Streaming the table is not enough.
* the only route that clears it is to **never write the slice**: march `d^l_{m,-2}` in registers and
  contract inside the same loop, ~8 FMA per value → 1.55e11 FMA. In float32 that is ~5 ms of FMA plus
  the per-value `exp`/reciprocal of the renormalisation (~5 ms at a quarter rate) against a 119 ms
  reference; in float64 it is 165 ms of FMA alone, i.e. 0.24x, so like every other win here it is a
  float32 design. The open risk is numerical, not throughput: `_spin_slice._march` renormalises through
  `lrenorm` precisely because the ell-recurrence underflows, and the march has only ever been run in
  float64 (the two halves of the s2fft march already disagree at ~1e-13, per section on the pi-theta
  symmetry). Build it as a probe against a float128 oracle at Nside 128 before wiring anything.

## Session 20 (2026-09-06): the fused spin-2 march is validated in float64, fails in float32, and degree anchors fix it

Spin 2 only, as directed. Everything below is a measured number from a log in `.qwen/tmp`; nothing is
wired into `gmaster/` yet, and no default changed.

**The machine lever first.** `.qwen/tmp/fp32_ffma_rate.py` → `fp32_ffma_rate.log`, an unrolled
dependent `v = v*a + b` chain so no bandwidth can flatter it:

```text
<class 'jax.numpy.float32'>  n 16,777,216  steps 256   0.2 ms  41.44 TFLOP/s  -> 2.0e11 FMA in  4.8 ms
<class 'jax.numpy.float64'>  n 16,777,216  steps 256   9.6 ms   0.90 TFLOP/s  -> 2.0e11 FMA in 222.4 ms
```

46:1 on CUDA cores. The fused march+contraction is `1.94e10` slice values × ~10-15 fp32 ops at Nside
1024, so its arithmetic is **~5-7 ms against ducc0's 107 ms** for the entire spin-2 forward transform;
the same kernel in float64 would cost 222 ms and lose outright. Throughput is settled and it is not
the obstacle — fp32-only design.

**Why the march must be the Jacobi closed form.** `.qwen/tmp/spin_march_fp32.log` (re-read this
session) is the shipped `_spin_slice._march` in float32: `non-finite 249` and `4.05e+05` relative on
the m<192 windows, `1.00e+00` on the last, with the float64 chain self-consistent to `3.0e-16`. Its
step coefficient is a difference of O(L) terms recomputed inside the loop, which is exactly the
cancellation float32 cannot survive. The Jacobi form has no such term — its step coefficients depend
only on `(n, alpha, beta)`, built once in float64 and cast:

```text
d^l_{m,-2}(th) = (-1)^m sqrt((l+m)!(l-m)!/((l-2)!(l+2)!))
                 (sin th/2)^(m+2) (cos th/2)^|m-2| P^(m+2,|m-2|)_(l-m)(cos th)
```

with DLMF 18.9.5 coefficients spelled so nothing is a difference of large numbers (`alpha²−beta² →
4·m·spin`), seeds `P_0 = 1`, `P_1 = (m+1)cos th + 2`. `.qwen/tmp/spin2_jacobi_lanes.log` locks it
value-by-value at Nside 64 with a delta RHS (the contraction *is* the slice element), per theta lane
including the poles, for the spin-coupled rows `m = 2, 3` where `beta = |m−2|` changes character:
relative 1.16e-13…1.19e-13 against a float64 march, `clean`.

**The accuracy gate.** `.qwen/tmp/spin2_jacobi_gate.py` → `spin2_jacobi_gate_arms.log`, four windows
at Nside 128/256, scored on the operation the pipeline really performs — the slice contracted over
theta against a complex ring transform — normalised by `max|ref|` and restricted to entries above
1e-8 of it (a per-element relative error near a Jacobi zero measures nothing). Line 1 of each block
is the yardstick: **what today's shipped route costs on the same metric**, float32 table × complex64
RHS.

```text
window              today's route   fused fp64     fused fp32     fp32 + fp64 products
n128  m [0,64)         3.35e-07      2.76e-12       4.65e-05       identical to fp32
n128  m[96,160)        5.00e-07      2.09e-12       7.81e-06       identical to fp32
n256  m [0,64)         7.12e-07      9.56e-12       1.22e-04       identical to fp32
n256  m[256,320)       1.15e-06      1.20e-11       1.80e-05       identical to fp32
```

Three verdicts. (1) The float64 arm at ~1e-11 says the design and every convention are right: the
per-theta-lane `(mantissa, exponent)` carry — the endpoint prefactor spans 2^±3000 and must be
renormalised **per lane**, not per `(m, ell)` — and the emission that folds `log2 N(ell, m)` **into
the exponent** instead of multiplying (N reaches 1e+1300 while the theta sum sits at 2^−3000; the
multiply form underflows to zero and reports a relative error of exactly 1, which is how the first
version of this gate "lost" the high-m windows). (2) float64 products change nothing, so the
contraction is not the defect. (3) The **float32 recurrence** is: forward recurrence in `ell` leaks
into the dominant solution and over L=384-768 degrees it loses three digits. 4.65e-05 vs a bar of
7.12e-07 is not shippable.

**The fix: refuse to march that far — anchored restart, measured.**
`.qwen/tmp/spin2_anchor_gate.py` → `spin2_anchor_gate.log` (reference = production table) and
`.qwen/tmp/spin2_anchor_gate2.py` → `spin2_anchor_gate_256.log`, `spin2_anchor_gate_512.log`, which
makes the **float64 march its own oracle and its own anchor builder** so it scales past the wall that
stopped the first script (`RESOURCE_EXHAUSTED: 36.19 GiB` inside `ss._build` at Nside 512 — session 12
again). Anchors are the float32 `(ell−1, ell)` pair at every K-th degree, both members riding one
shared per-lane exponent, in **march space** (divide out `(-1)^m N` in log space first — anchoring the
raw table value re-introduces that factor and the row drifts to 1e+35).

```text
anchor period K          pure       128        64        32        16        8
n256 m [0,64)          1.22e-04   2.61e-05  8.82e-06  3.42e-06  1.21e-06  6.41e-07
n256 m[256,320)        1.80e-05   7.08e-06  5.23e-06  3.39e-06  2.03e-06  8.30e-07
n512 m [0,64)          3.04e-04     --      9.30e-06  3.93e-06  1.21e-06  5.06e-07   (median 1.65e-08)
n512 m[512,576)        2.70e-05     --      4.98e-06  2.90e-06  1.65e-06  7.25e-07
shipped fp32 table route, same metric:  7.12e-07 (n256 m<64)   1.15e-06 (n256 m 256..320)
```

**The error at fixed K is independent of L** — 6.41e-07 at L=768 vs 5.06e-07 at L=1536 — and grows
~linearly in K, exactly as it must when each segment pays only its own K degrees of round-off.
**K=8 lands at or below the shipped float32 route's own error** (5.06e-07…8.30e-07 vs 7.12e-07…1.15e-06)
with a median ~20x better; K=16 is 1.21e-06…1.65e-06, same order for half the bytes. Accuracy gate:
**GO for an anchored fp32 march**.

**And the anchor table is the capacity lever the whole exercise was for.** Measured per 64-row m
window: **0.09 GiB** at Nside 256/K=8, **0.37 GiB** at Nside 512/K=8 — scaling as `ntheta·L²/(2K)`, so
**~36 GiB for the whole triangle at Nside 1024/K=8, ~18 GiB at K=16**, against the **73.48 GiB** layout
`slabs_for` still declines (session 12). Storing the anchor exponent as int16 rather than float64
brings K=8 to ~24 GiB. The fused kernel would then read a table that *fits* the 71.2 GiB pool, sparsely
(once per segment per `(m, theta)`), and never materialise the slice: that is the difference between
Nside 1024 spin-2 sitting at **13.27 s/pass** in the generic per-window loop and a real shot at ducc0's
**107 ms**.

**The law holds at the target geometry.** `.qwen/tmp/spin2_anchor_gate2_1024.log` and
`spin2_anchor_gate2_1024_w512.log` (L=3072, ntheta=4095, the fp64 march as its own oracle — the
table cannot be built here at all):

```text
window m [0,64)        pure         K=8          K=16         K=32          anchors/window
                     1.08e-03     5.16e-07     1.08e-06     2.79e-06      1.50 / 0.75 / 0.37 GiB
                     median 1.61e-05  1.44e-08   3.93e-08     1.15e-07
window m [512,576)    9.40e-05     9.71e-07     2.09e-06     5.92e-06      (same bytes)
                                    6.51e-08    1.46e-07     3.28e-07
```

K=8 at Nside 1024 gives **5.16e-07 / 9.71e-07**, i.e. the same number it gave at L=768 and L=1536 —
the L-independence is confirmed at the geometry that matters, and it brackets the shipped route's
7.12e-07…1.15e-06 with medians 1.4e-08…6.5e-08 (~20x better). The scan's own 15 s arm time is XLA
per-degree dispatch inside `lax.scan`, not a kernel rate.

**And the real block shape holds too.** `.qwen/tmp/spin2_fused_block.py` is the kernel's actual
shape rather than the gate's single-RHS form: theta split into tiles (the march state must live in
registers), each tile producing a partial per `ell` that is summed in float64 across tiles — the
shipped `_RED_CHUNK` discipline — and the contraction taken against **4 real polarised channels**
(Q,U x direct,mirror) at storage width with float64 only for the tile partial. The padded theta tail
is dead in every sense: zero RHS, zeroed `nxt`, and its exponent pinned at −1e6 so it can never win
`emax` and shift the real lanes out of float32's reach. Nside 256, window m<64, 8 tiles of 128:

```text
  fp64 block (oracle): 382 ms   partial buffer 12 MiB
  fp64 self-check:            all 0.00e+00   median 0.00e+00
  fp32 march + anchors K=8:   all 4.10e-07   median 1.33e-08   anchors 0.09 GiB/window
```

4.10e-07 is *better* than the single-RHS gate's 6.41e-07 at the same K, because each fp32 partial sum
now runs over 128 lanes instead of 1023 — tiling buys accuracy as well as register residence. Pure
arithmetic for this block is 0.017 ms/window of fp32 FFMA (1.13e10 FLOP/window at Nside 1024 = 0.27 ms
there), so the whole polarised pass is budgeted ~5-7 ms of arithmetic + ~24 GiB of anchor reads +
~25 GiB of RHS reads ≈ **40 ms/pass against ducc0's 107 ms and the generic loop's 13.27 s**.

**That 40 ms is a budget, not a measurement, and there is one measured generator rate that argues
against it.** The fused kernel is the first *generator* we have never timed. The only measured
value-producing kernel in this repo is the spin-0 band emitter at **117.3 G values/s** (session 19),
which on the same value count (1.93e10 at Nside 1024) would be **165 ms/pass** — i.e. slower than
ducc0. The emitter's floor is its 77 GiB of writes (53 ms at the measured 1444 GB/s) plus the
per-degree scan-latency pathology recorded in `project/builder-scan-latency.md`, neither of which the
fused kernel has (it writes only ~2 MiB of partials per window), but the gap between 41.4 TFLOP/s of
FFMA capacity and 117 G values/s is 35x and it is exactly the gap that has eaten every previous
"the flops are cheap" estimate here. **The kernel's rate is therefore unmeasured and is the only open
question; the accuracy question is closed.** Do not quote 40 ms in a comparison until it is in a log
next to a `rel alm`.

**Not built: the Pallas kernel.** The `lax.scan` oracle of the real block now exists and passes
(`spin2_fused_block.py`, numbers above), so the remaining work is the port: one program per
`(m-window, theta-tile)` with `(prev, cur, ex)` in registers, the K=8 anchor gather, the 4-channel
`nxt*w*R` partial in fp32 with a float64 accumulator, and rows `m < spin` routed through the swap
`d^l_{m'm} = (-1)^(m-m') d^l_{mm'}`. It goes behind a default-off env gate (the session 18/19 pattern,
`GMASTER_POLAR_CONTRACT=xla` style) and the first log must contain the kernel's ms/window *and* the
pipeline's `rel alm`, because a generator route that is only fast is not a win.

## Session 20 (later): anchoring kills the serial recurrence, and the same math in a map is 2500x faster

The scan oracle was accurate but its 14.4 s/window was XLA per-degree dispatch, and the emitter
cross-check above said a generator might not clear ducc0. Anchoring has a second consequence that
was not exploited until now: **a segment that starts from stored `(ell-1, ell)` values does not need
the segments before it**, so the `ell`-recurrence is not a serial chain of L steps at all - it is
`ceil(L/K)` *independent* programs of K unrolled steps, which is the shape XLA and this card are good
at. `.qwen/tmp/spin2_fused_block.py`'s `segblock` implements exactly that (one program per
`(theta-tile, S segments)`, K unrolled steps, `prev/cur/ex` in registers, the per-lane scale `w`
constant inside a segment so it folds into the RHS once, 4 real channels contracted in fp32, fp64
only for the emitted partial).

Two things had to be right, and each was wrong in its own way first: **k=0 emits the anchor itself**
(the pair is `(V_{e-1}, V_e)`, so the segment's first emitted degree is `e`, and the carry must
advance only `cur` at k=0 or the k=1 step gets `prev = V_e`), and the seed overrides `at0/at1` must
stay inside the loop so a row whose triangle starts mid-segment re-seeds itself - the row's garbage
below `m` never reaches an output because the two consecutive overrides at `m` and `m+1` reset the
pair. With that, the error profile against `ell mod K` is the textbook sawtooth the anchored model
predicts: 1.3e-07 at k=0 (exactly the anchor's own float32 storage error) rising to 4.9e-07 at k=7.

Measured, same yardstick as everything above (`spin2_fused_block_256_seg_sweep.log`,
`spin2_fused_block_512_seg.log`, `spin2_fused_block_1024_seg.log`):

```text
nside   scan arm      parallel-segment arm    accuracy   S (segments/program) -> G values/s
 256    955 ms/window  0.61 ms/window (S=64)   4.10e-07   8: 32.9   16: 55.4   32: 78.5   64: 82.0
 512   3592 ms/window  1.84 ms/window (S=64)   4.51e-07   32: 71.2  64: 109.6
1024  13989 ms/window  5.65 ms/window (S=128)  5.44e-07   64: 105.6 128: 142.5   (median 1.27e-08)
```

**At Nside 1024 the whole polarised spin-2 latitudinal pass is 48 x 5.65 ms = 271 ms in plain XLA**
(the probe computes the masked rectangle, so a kernel that marches only `ell >= m0` should land near
150 ms), against the **13.27 s** generic per-window loop that ships today at that Nside - **~49x** -
and against ducc0's ~119 ms, where we are still ~2.3x behind. It is not yet a ducc win at 1024; it
is the first version of this step that runs at all, at an accuracy (5.44e-07, median 1.27e-08) at or
below the shipped float32 table route's own error, with **no 73.48 GiB slice anywhere**.

Where the headroom demonstrably is: at S=64 the block logs **1267 GFLOP/s** on a 41,440 GFLOP/s fp32
FFMA peak (**3 %**) and at S=128 it moves 1.50 GiB of anchors per window in 5.65 ms = **280 GB/s**
against the card's 1444 GB/s stream rate - so it is neither bandwidth-bound nor compute-bound, it is
latency/ILP-bound inside the K-step chain, which is precisely what a Pallas port controls (deeper
unroll, the four channels as one vector op, the anchor gather hoisted one segment ahead). The scan
comparison is not a subtlety: 13,989 -> 5.65 ms is **2500x**, and that is only the difference between
expressing the recurrence as a chain and expressing it as a map.

The rate does not fall apart in the high-`m` half (`spin2_fused_block_1024_w512_seg.log`, window
m 512..576): **5.69 ms/window, 141.4 G values/s, 1697 GFLOP/s, all-compared 8.95e-07, median
5.84e-08** — same speed as the leading window and the same accuracy class as the single-RHS gate at
that window (9.71e-07), so no window needs special handling and 48 x 5.7 ms = 271 ms stands as the
measured per-pass cost of this formulation at Nside 1024.

**DFT / brute-force note (`record-brute`), recorded as a cost model, not a measurement.** The probe
script and its log for the no-FFT direct DFT are no longer in `.qwen/tmp` (checked: `ls .qwen/tmp/*brute*`
→ nothing), so I am not quoting a number for it. What the model says, with the fp32 rate measured above:
a direct DFT of the `2·nside` equatorial belt rings at the band limit `N = 4·nside` is
`2·nside·(4·nside)²` complex MACs ≈ `5.4e12` fp32 FMA at Nside 1024 ≈ **130 ms**, which is ~2.3x the
plain-FFT ring stage this repo already ships for that same Nside. Rejected: the belt (constant `nphi`)
makes the ring DFT a power-of-two plain FFT, and Bluestein's 3x overhead is the wrong tool there
(sessions 14/15). The fused march changes none of that — it removes the *slice*, not the ring transform.

---

## Session 21 — the anchor table is dead: a serial compensated float32 march, table-free, at 123 ms/pass for Nside 1024 spin-2

Probe: `.qwen/tmp/spin2_pallas_march.py` (logs `spin2_march_1024.log`, `spin2_ds_1024_b.log`,
`spin2_ds_512.log`, `spin2_arms_256.log`, `spin2_march_geom.log`, `spin2_march_ablate.log`,
`spin2_ds_sweep.log`). Arms `f64 | f32 | f32l | f32ds` march the same Jacobi coefficients against the
same 4 real polarised RHS, one Pallas program per (m-row, theta-tile), a serial `lax.fori_loop` in
`ell` from `m` (so a row touches only its own triangle), and one scalar `plt.store` per output value.
Reference is a numpy float64 march of the same coefficients, which I validated first this time
(below) because last window's whole debugging cycle went into an unvalidated reference.

**The reference is right, and the thing that was wrong was one seed coefficient.** My numpy oracle
agrees with `spin2_fused_block.py`'s `block("f64")` — the construction `spin2_jacobi_gate_arms.log`
validates against the production table to 2.1e-12 — to **2.76e-08** over all `m >= 2, ell >= m` at
Nside 64 (`oracle_march_64_0.npy` vs `oracle_fb_64_0.npy`; that residual is the fused block's float32
RHS, my oracle carries float64). Everything below is therefore measured against something real.

**`P_1` is `((alpha+beta+2)/2)·cos(theta) + (alpha-beta)/2` = `(m+1)cos(theta) + 2`, not
`m·cos(theta) + 2`.** The kernel seeded `ell = m+1` with the latter and it cost a full debugging
window: identical relative error in the float64 and float32 arms (0.98), *exact* at `ell = m` and
wrong at `ell = m+1` — which is an analytic seed with no recurrence in it — growing with `ell`
(`first ell over 1e-4 = 4m+8`, i.e. proportional to `1/m`). Ratio at `m=12`: 1.0000 at `ell=12`,
0.9348 at 13, 1.5370 at 14. The generic three-term coefficients `(alpha+beta)/2` and
`(alpha-beta)/2` look like they should assemble `P_1`; they do not — the `x` coefficient of
`P_1^(a,b)` is `(a+b+2)/2`. After the fix the float64 arm is **4.56e-16** at Nside 64 and
**1.53e-15** at Nside 1024 against the oracle, i.e. the Pallas march reproduces the reference exactly.

**Both compensations are needed; neither alone is worth much.** Nside 256, tile 512, 1 warp, same
4-channel RHS, `rel = |arm-ref|/max|ref|`:

```text
arm     ms/window  G values/s  rel vs fp64 oracle
f64       11.78        4.1        7.50e-16
f32        1.07       45.2        8.83e-05     plain float32 everything
f32l       1.32       36.6        8.31e-05     double-single state (hi/lo), float32 coefficients
f32ds      1.42       33.9        7.94e-08     double-single state + hi/lo coefficient tables
```

The tempting reading of `f32l ~= f32` is "the state does not matter, only the coefficients do".
**That is wrong, and I wrote it down wrong for an hour.** The arm that tests it is `f32c` — hi/lo
coefficient tables, *single-limb* float32 state (`(ah+al)*ch - (cb+bl)*ph` to first order in the
limbs): at Nside 1024 it runs **3.62 ms/window at rel 3.92e-05** (`spin2_f32c_1024.log`), i.e.
fixing the coefficients alone leaves 3.9e-05 of *state* error, and fixing the state alone leaves
8.3e-05 of *coefficient* error, while fixing both gives 1.27e-07. The two error sources are
independent and roughly equal in size; each single fix buys ~2x, the pair buys ~1000x, because the
error of the whole march is the sum of a per-step coefficient bias and an accumulating state bias
and neither one can be left standing if you want the 1e-07 class. `f32c` is also only 10 % faster
than `f32ds`, so there is nothing to win by dropping the state limb even where accuracy allows it.

7.94e-08 (Nside 256) / 1.27e-07 (Nside 1024) is below the shipped float32 table route's own bar
(7.12e-07..1.15e-06) and below the anchored route's 5.44e-07, achieved with **no anchor table at
all**.

**One warp per program is what unlocked the serial shape.** Nside 1024, arm `f32`, `ms/window`:

```text
tile x warps     128x4    128x1    256x1    512x1    256x2   1024x1
ms/window        27.13     6.81     4.47     4.15     7.43     7.44
```

4 warps means every per-degree `jnp.sum` over the theta tile crosses warps through shared memory
with a barrier, in the middle of a serial chain; 1 warp with 8-16 theta lanes per thread replaces
that with intra-thread adds plus 5 shuffles, and the same launch that cost 27.13 ms costs 4.15 ms.
`tile 256 x 2` is worse than `tile 256 x 1` (7.43 vs 4.47) and `tile 1024 x 1` spills (7.44), so
the win is intra-thread ILP plus barrier-free reductions, not merely "fewer warps".

**Where the remaining time is** (Nside 1024, `f32`, tile 128 x 4 warps, `MARCH_ABLATE`):

```text
variant                                   ms/window   what it removes
full                                        27.13      -
marchonly (no emit, one store per program)   0.43      the entire per-degree emit
chan1 (1 channel reduced instead of 4)      18.29      3 cross-lane reductions + 3 stores
nomax (exponent max replaced by a constant) 25.98      1 cross-lane reduction
norenorm (band check removed)               26.87      the whole renormalise chain
nocoef (3 per-degree scalar loads hoisted)  29.33      nothing (it is free; NaN - semantics broken)
```

The **recurrence itself is free**: 0.43 ms for all 4096 programs x 3072 serial steps = 1869 G
values/s, because the chains are independent across theta lanes and pipeline perfectly. 100 % of
the cost is the emit — the 4 per-degree theta reductions and the per-degree scalar stores. The
per-degree coefficient reads and the renormalise chain are free.

**Headline, Nside 1024, spin-2, table-free, accurate** (`spin2_ds_1024_b.log`): arm `f32ds`,
tile 256, 1 warp:

```text
5.05 ms/window   157.8 G values/s   rel 1.20e-07   non-finite 0
projected whole pass = 24.3 x window = 0.123 s
```

(tile 512: 5.49 ms/window, rel 1.47e-07, 133 ms projected — tile 256 is the sweet spot.)

The projection factor is 24.3, not 48: the serial march starts each row at `ell = m`, so a window
does `sum(L - m)` live degrees and the pass is `L(L+1)/2 / 194592 = 24.3` windows, whereas the
session-20 segment map always marches the masked rectangle and pays 48.

| Nside 1024 spin-2 latitudinal pass | ms/pass | rel | slice/table traffic |
|---|---|---|---|
| generic per-window loop shipped today | 13,270 | ~1e-13 (fp64 table) | 73.48 GiB slice, declined |
| session-20 anchored segment map (XLA) | 271 | 5.44e-07 | 1.50 GiB anchors/window, ~72 GiB/pass |
| serial compensated march (Pallas), 4 independent reductions | 123 | 1.20e-07 | 4.7 MB of fp32 coefficient limbs/window |
| **this: same kernel, one shared-tree reduction + one `(NC,)` store** | **98** | **1.27e-07** | same |
| ducc0 CPU — **whole** spin-2 forward transform, not this step alone | 107–119.3 | — | on-the-fly recurrence |

Read that last row carefully, because it is the one place this session could lie to a reader. The
119.3 ms figure is ducc0's **entire** spin-2 `map2alm` at Nside 1024 (session 19's note is even
blunter: the shipped table contraction "cannot be compared to ducc0's 119 ms, which is a full
forward transform with its own harmonic synthesis"). The 98 ms above is **one stage** of our
pipeline. So the defensible statement is: *the table-free polarised spin-2 latitudinal step now
costs less than ducc0's whole forward transform*, i.e. the stage that was 13.27 s and then 271 ms
is no longer what puts us behind — and whether the **pipeline** beats ducc0 at Nside 1024 is a
question only the integrated number can answer, because our ring stage is not in that 98 ms. What
is unambiguous is the comparison against our own alternatives: **2.8x faster than the anchored map
at 4.3x better accuracy, with 320x less coefficient traffic than the anchor table** (`6 arrays x 64
rows x 3072 degrees x 4 B = 4.7 MB` versus 1.50 GiB/window), and no 73.48 GiB layout anywhere.


The last 20 % came from the emit alone. At the operating point (Nside 256, tile 256, 1 warp) the
split was `full 1.05 / marchonly 0.20 / chan1 0.80 / nomax 0.99 / norenorm 0.95` ms — 81 % emit,
of which the four *independent* per-channel theta reductions were 0.33 ms. Storing the RHS as
`(MB, theta, NC)` and writing `jnp.sum(val[:, None] * r, axis=0)` turns four dependent 5-deep
shuffle trees into **one** Triton tree over a `(TB, NC)` block, and the four scalar `plt.store`s
into one 32 B store. Measured (`spin2_ds_sharedtree.log`, `spin2_ds_tile_sharedtree.log`):

```text
Nside 256, tile 256 x 1 warp   1.05 -> 0.80 ms/window   rel 8.41e-08 -> 5.73e-08
Nside 1024, tile 256 x 1 warp  5.05 -> 4.03 ms/window   rel 1.20e-07 -> 1.27e-07   198 G values/s
tile sweep at Nside 1024 (f32ds, 1 warp): 128 -> 4.86   256 -> 4.03   512 -> 4.83 ms/window
```

Tile 256 is a real optimum, not a plateau: 512 wins on store count, 128 wins on resident warps,
and 256 splits the difference. The accuracy is unchanged in kind (both are ~1e-07, well inside the
shipped float32 route's 7.12e-07 bar) because the change is a reduction order, not a precision
change; the float64 control stays exact (4.56e-16).

**It gets better with size, which is the whole point of aiming past 1024** (`spin2_ds_2048.log`,
`spin2_ds_4096.log`, arm `f32ds`, tile 256 x 1 warp):

```text
Nside   ms/window   G values/s   rel            projected pass
 1024      4.03        198       1.27e-07        98 ms   (24.3 windows)
 2048     10.03        320       6.40e-08       484 ms   (48.3 windows)
 4096     42.99        299       1.61e-07       4.14 s   (96.3 windows)
```

The rate is flat-to-improving and the accuracy is flat across a 16x range of `L^2` work, which is
the behaviour the session-20 anchor gate measured as "error at fixed K is independent of L"
(K=8: 6.41e-07 at L=768, 5.06e-07 at L=1536, 5.16e-07 at L=3072) — here with no K at all, because
the compensation replaced the restart. The fp32 renormalise band is fixed at `2^±24` per row
regardless of `L`, and the per-degree overheads amortise over a longer chain.

For scale at the top size: the Wigner-d slice at Nside 4096 would be ~4.7 TiB, and this kernel's
tables are 75 MB of fp32 coefficient limbs per window. The only ducc0 cell logged in this repo at
Nside 4096 is *scalar* (`ducc_only_4096_0.log`: 1840.97 / 1865.26 ms for map2alm / alm2map), and
the polarised ducc0 cells at 2048/4096 have never been measured here — producing them is part of
the integrated scoreboard, not something to extrapolate.

**The per-pass projection is measured, not extrapolated** (`spin2_windows_1024.log`, Nside 1024,
`f32ds`, tile 256 x 1 warp). Because a row's loop starts at `ell = m`, cost should be proportional
to a window's live-degree count `sum(L - m)`:

```text
window          measured   linear model   rel          stored cells
m [   0,  64)     4.03 ms      4.03 (def)  1.27e-07    12.45 M
m [ 512, 576)     3.22 ms      3.35 ms     2.59e-07    10.32 M
m [2560,2624)     0.71 ms      0.64 ms     1.71e-07     1.25 M
```

Two things follow. The rate model is good to ~10 % across a 16x range of window cost, so
`24.3 x 4.03 = 98 ms` for the whole pass is a sum of measured behaviour rather than a leading
window times a count; and the accuracy does not degrade in the high-`m` half (2.6e-07 at
m=512..576 against 1.27e-07 at the leading window, same class), so no window needs special
handling. This is the property the session-20 map did *not* have available to it: it marches the
masked rectangle, so every window cost the same 5.65 ms and the pass cost 48 x that.

**ducc0 spin-2 baselines measured fresh on this box** (`ducc_only_1024_2.log`, three separate
processes of `.qwen/tmp/ducc_only.py 1024 5 2`; the 2048 row is `ducc_only_2048_2.log`, which
already existed and is quoted as recorded). ducc0 0.39.1, 192 cores, no GPU process on the box,
min/median/max over reps — and these are **whole `map2alm` / `alm2map` transforms**, which is the
only fair thing to compare a stage against when stating what is left to win:

```text
Nside  ducc0 spin-2 map2alm (best/med)      alm2map (best/med)   our latitudinal stage
 1024  115.19 / ~118-123 ms                 105.99 / ~108-115    98 ms   (rel 1.27e-07)
 2048  732.08 / ~904 ms                     609.54 / ~612 ms     484 ms  (rel 6.40e-08)
```

So the stage that shipped at 13.27 s and stood at 271 ms after session 20 is now, on its own,
**below** ducc0's entire forward transform at both sizes: 1.18x at Nside 1024 and 1.51x at Nside
2048 measured against ducc0's *best* of three processes. A pipeline number still has to add our
ring stage to the 98 ms, and that is the next thing to measure; what is settled is that the
polarised latitudinal step is no longer the reason GMaster loses to ducc0 at high Nside.





What is left in the kernel, by the same ablations: ~0.2 ms of renormalise chain, ~0.06 ms of
exponent max, and the residual emit (`2**lg2n` per degree, the `lgnr` scalar read, the store).
`marchonly` says the floor of this formulation is 0.20/1.05 = 19 % of current cost, so the shape
has maybe 3x more headroom in it, all of it in the emit.

Two follow-ups on the geometry, both negative (`spin2_warp_sharedtree.log`, `spin2_rload_in.log`):

* After the emit rewrite, **tile 256 x 1 warp is still the optimum**: tile 512 x 2 warps gets to
  4.36 ms (it was 4.83 at 1 warp, so two warps do help once there is only one tree), tile 1024 x 2
  is 5.23, and tile 256 x 2 warps regresses to 5.32 against 4.03 at one warp. The old rule
  "always one warp" is really "keep the reduction inside one warp *and* keep 8 chains per thread";
  at 256 lanes with 2 warps there are only 4 chains per thread and the loss outweighs the win.
* **Freeing registers by re-reading the RHS per degree does not work.** The 4-channel RHS is
  loop-invariant and costs NC blocks of registers, which is exactly where `f32ds` needs them, so
  re-loading it inside `emit` looked like the cheap way to make tile 512 viable. It is not:
  4.15 vs 4.03 ms at tile 256 and 8.17 vs 4.83 ms at tile 512. Whatever limits tile 512 is not
  this register pressure.



Pallas gotchas this window, each of which cost a run:
* `elementwise_inline_asm` operand numbering is `$0..$(n-1)` over *all* operands including outputs.
  `fma.rn.f32 $0, $2, $3, $4` with 1 output + 3 inputs is a **ptxas parse error** (`error code 65280`,
  `line 1316;`) that JAX reports with no useful text; correct is `$0, $1, $2, $3`.
* Block indexing `ex[0]`, `val[0]` does not lower on Pallas GPU (`Unimplemented primitive: slice`).
  To ablate a `jnp.max`, replace it with a constant, not with a lane.
* A `fori_loop` whose only sink is an accumulator gets deleted wholesale; a per-degree
  `plt.store` is what keeps a march alive (`marchonly` needs its one post-loop store).
* A sweep that pipes the probe through `grep "^   f32ds"` silently returns nothing, because the arm
  column is `f"{arm:>6}"`: `f32` gets three leading spaces, `f32ds` only one. Two configs were
  read as "timed out" and one as "the compensated kernel takes 5 min to compile" before `grep
  "f32ds +[0-9]"` showed they had all run fine. Match on the value, not the padded label.

## Session 22 (2026-09-06): the march becomes a default, and the polar lane survives every arithmetic fix

**The shipped change (`92e97e0`).** `march_requested` used to be an env-var test with no geometry, so
the table-free march was reachable *only* when someone typed `GMASTER_SPIN2_MARCH=1`, and the shipped
default at every Nside whose Wigner-d slice is declined fell through to the generic scatter loop. It
now takes `L` and `nside` and, with no flag set, picks the march exactly where `slice_declined` says
no slice can exist — the test is on the cheapest layout that can ever exist, a float32 single-layout
triangle, which is 2.44 GiB at Nside 256, 18.74 GiB at 512, **146.96 GiB at 1024** and 1163.86 GiB at
2048 against `_SLAB_BUDGET` = 56 GiB. One cell moves: spin-2 `map2alm` at Nside ≥ 1024 goes
**13 255.1 ms → 108.0 ms**, i.e. **0.01x → 1.00x** against ducc0's 107.8 ms, with `rel alm`
3.7e-05 instead of the scatter loop's 4.0e-07. `GMASTER_SPIN2_MARCH=0` restores the exact route at
any size.

**Why the flag was not simply turned on everywhere:** where a slice *does* exist the table route is
~2.5x faster than the march — Nside 512 spin-2 `map2alm` is 9.4 ms on tables against 24.0 ms marched
(`.qwen/tmp/twosum_check.log` section 4). The default is therefore "march only where there is nothing
else", not "march everywhere".

**The default-route matrix, one geometry per process, nothing set in the environment**
(`.qwen/tmp/score_matrix_fresh.log`, `GM_PREC=fp32`, GPU1, ducc0 0.39.1 on the same box, controls
1279–1446 GB/s, driver `.qwen/tmp/score_matrix.sh`):

```text
 nside spin      dir    DUCC ms  GMaster ms GPU speedup    rel alm
    64    0  map2alm        0.4         0.3       1.64x 2.1e-07
    64    0  alm2map        0.3         0.2       1.44x
    64    2  map2alm        0.6         2.1       0.26x 2.7e-07
    64    2  alm2map        0.8         0.4       1.73x
   128    0  map2alm        0.8         1.2       0.67x 2.8e-07
   128    0  alm2map        0.7         0.7       0.94x
   128    2  map2alm        1.3         2.3       0.57x 3.5e-07
   128    2  alm2map        1.0         0.6       1.61x
   256    0  map2alm        2.3         1.1       2.00x 2.7e-07
   256    0  alm2map        1.6         1.3       1.20x
   256    2  map2alm        4.7         2.2       2.10x 3.2e-07
   256    2  alm2map        3.1         1.6       1.95x
   512    0  map2alm       15.5         4.5       3.48x 3.7e-07
   512    0  alm2map       11.9         5.1       2.36x
   512    2  map2alm       27.4         9.1       3.02x 3.2e-07
   512    2  alm2map       22.8         8.4       2.71x
  1024    0  map2alm       57.6        28.2       2.04x 3.6e-07
  1024    0  alm2map       51.6        30.8       1.67x
  1024    2  map2alm      107.8       108.0       1.00x 3.7e-05
  1024    2  alm2map       96.5     13207.5       0.01x
```

Two new holes this is the first log to put side by side: **Nside 64–128 loses on analysis** (0.26x,
0.57x for spin 2; 0.67x for spin 0 at 128) where a ~2 ms fixed cost meets a 0.6 ms ducc0 call, and
**Nside 1024 spin-2 synthesis is the last sub-1x cell above the launch floor**. Everything from 256
up wins except that one cell.

**The exact 2Sum is right and buys nothing measurable (`779ae1b`).** The march took the residual of
`th - uh` from the fast form `(th - nxt) - uh`. Over 8192 lanes on the GPU that form is off from the
exact difference by max rel **5.845e-08 with |b| < |a|** — where its precondition *is* satisfied — and
**5.956e-08 / rms 2.043e-08 with |b| > |a|**, while the branch-free six-operation `_two_sum` returns
**0.000e+00** in both orderings (`.qwen/tmp/twosum_order_probe.log`). Same mechanism the module
already documents for products: `a - b - c` is contractible by the compiler, `s - bb` is not. Both
kernels now use `_two_sum`. **The polar row error is unchanged**: `ell=1528` in the pole-most lane
reads 2.47e-04 before and after, and re-running the lane dump at Nside 256 with the file stashed
(`.qwen/tmp/polar_lane_256_fast2sum.log` against `.qwen/tmp/polar_lane_256_2sum.log`) gives
**bit-identical errors in 28 of 30** (m, ring, ell) cells; the two that move are equator lanes at
0.25–0.31x of their old error. Keep the fix — a limb that enters the state one rounding off is a bug
regardless — but stop treating it as the polar cure.

**The lane dump itself is the useful instrument.** `.qwen/tmp/polar_lane_dump.py <nside>` feeds
`_forward_impl` an `ftm` that is a Kronecker delta at one theta ring, so output column `L-1+m` *is*
the kernel's `d^l_(m,-2)(theta_ring)` and the growth law per lane is directly readable against 60-digit
mpmath. It needs `jax_enable_x64=True` (the output refs are float64; a float32 store raises
`Invalid dtype for swap: Ref dtype: float64. Value dtype: float32`). At Nside 256 the error at
`ell=760` is 1.3e-07 at the equator, 8e-06 thirty-three rings off the south pole, 6.5e-05 at either
pole for m=0, and 1.3e-03 for m=2, whose row crosses zero there. North and south lanes of the same m
agree to the digit and the growth is flat-to-sublinear in ell — this is conditioning at
|cos theta| → 1, not a one-sided blow-up.

**The seed is innocent too.** A float32-rounded seed is amplified by only **0.1–0.4 ulp** across the
whole band at every theta including the poles (`.qwen/tmp/seed_amp_probe.log`), which is the mechanism
behind the old observation that pairing the seeds changed nothing.

**A claim is deleted, not softened.** `8e4ddf5` recorded that "zeroing the recurrence-coefficient
limbs inside the kernel makes it 23x worse (2.47e-04 → 5.62e-03)". `.qwen/tmp/spin2_limb_probe.py`
still carries the slot map of the *13-slot* geometry tuple that had `mant_l`; the current
`_window_geometry` returns 12 slots, so its `IDX = {"coef": (8,9,10), "x": (4,)}` zeroes
`a0l/bhl/lgs` and `a1h` — including `lgs`, which is not a limb at all. The 23x measured nothing. Any
future ablation must use coef = (7,8,9), x = (3,), and must never touch slot 10.

**Two probe traps, both re-learned this window.** (1) `XLA_PYTHON_CLIENT_PREALLOCATE=false` does not
just shrink the pool, it *changes the route*: with the pool at 0.5 GiB the band build fails, the
negative cache routes to the scatter loop, and Nside 512 spin-2 read **1287.3 / 1160.2 ms (0.02x)**
against the shipped 9.4 / 8.5 ms. Any perf number measured under that setting is a fallback
measurement. (2) A sweep that loops several geometries inside one process still OOM-thrashes the last
one even after the earlier fix in this file — the `score_all_default_all.log` run took 1024 spin 0 as
122.8 / 161.0 ms with `Allocator (GPU_0_bfc) ran out of memory` in the log, against **28.2 / 30.8 ms**
with one geometry per process. `.qwen/tmp/score_matrix.sh` is now the driver that enforces it.

**What is left, in order of size.** Spin-2 `alm2map` at Nside ≥ 1024 is 13.2 s (0.01x): the synthesis
march exists and runs 121.8 ms (0.89x, so not a win even when it works) but its pole-most lane is
2.47e-04 off the row value, which reaches an `alm2map` pixel as **rel 9.99e-01 at the map's own
maximum**, rms 2.40e-03 (`.qwen/tmp/synth_map_acc.log`), so it stays behind
`GMASTER_SPIN2_MARCH_SYNTH=1`. The three arithmetic suspects are now all measured and eliminated —
seed, primitives, subtraction residual — and the accumulator limb provably never reaches the output,
because `_accumulate` closes with `(adr + adl)` in float64 and a renormalised pair sums exactly. What
is left is the recurrence's own conditioning near |cos theta| = 1 in a float32 pair, which the fp64
march of the same recurrence says is 7.5e-13, i.e. roughly 30 of the ~48 bits the pair is supposed to
carry. Suite across all of it: **141 passed, 3 skipped** (`.qwen/tmp/pytest_s22c.log`).

**Which calls the seams actually own (`route_probe.py`, a fact this file did not have).** Counting
calls into `forward_latitudinal`/`inverse_latitudinal` from the shipped E/B entry points at Nside 512
returns **zero**, with `GMASTER_SPIN2_MARCH=1` set or unset (`.qwen/tmp/route_probe_512.log`). The
reason is `_use_pallas_sht`: spin ≠ 0 never takes the fused-Pallas branch, and for spin 2 the
non-Pallas branch first asks `_spin_slabs`, which returns the slab pair whenever `slabs_for` accepts
the geometry and only then falls through to the generic path whose latitudinal step is
`utils._forward_latitudinal` / `utils._inverse_latitudinal`. So **the seams are reached exactly when
the slice is declined** — the same condition `slice_declined` tests, which is why the new default and
the seam agree by construction rather than by coincidence. Practical consequence: the Nside 512 and
below spin-2 cells (3.02x / 2.71x) are the slab contraction's, not the march's, and the "marched 512"
figure of 24.0 ms in `.qwen/tmp/twosum_check.log` was produced by *declining the table route*, not by
the flag. Any future spin-2 route policy has to be written against `slabs_for`, not against the
environment.

Slab-streaming the Wigner-d slice (build one m-slab, contract, discard) is not the answer either, and
the arithmetic is worth recording so nobody re-derives it: at Nside 1024 building both layouts costs
~1.7 s against the contraction's 0.8 s, so a streamed slice would land near 2.5 s/pass against the
march's 108 ms. The march wins at exactly the sizes where the table cannot be resident because it
skips the build, and it loses at the sizes where the table can be resident because the fused kernel
only reads.

**The synthesis march's map error is not the polar lane, and not arithmetic (`synth_map_err.py`,
`synth_delta_row.py`).** The number that has been quoted for this route — rel 9.99e-01 at one pixel,
rms 2.40e-03 — is a Nside 512 measurement of a route that cannot even be reached there. Run at Nside
1024, where the seam is live, marched synthesis against the exact route in one process gives

```
max|ref| 3.2482e+02   rms(ref) 5.3147e-01
||d||_inf / max|ref|  = 9.964e-01
||d||_2   / ||ref||_2 = 9.302e-01
worst pixel 3 (phi 5.4978) at theta 0.000797 rad (cos +1.000000): |ref| 3.240e+02, |d| 3.237e+02
```

(`.qwen/tmp/synth_map_err_1024.log`). An rms ratio of 0.93 means the marched map carries as much
power as the reference: it is a different map, not a degraded one. That reframes the blocker
completely — there is no 1e-3 defect to compensate, there is a structural disagreement, and the
"improve the polar lane" reading of the previous paragraph is wrong.

`.qwen/tmp/synth_delta_row.py <nside>` then tests the kernel alone: one nonzero `flm[ell, L-1+m]`,
the marched column `L+m` against the exact route's, at five `(ell, m)` per size. The rows are **fine**
— Nside 256 worst `max rel` 3.53e-05, Nside 1024 worst 1.43e-03 (at `ell=3064, m=2`, the pole lane),
with `frac>1e-3` at 0.0000–0.0002 of the theta lanes and no leakage into neighbouring columns
(`.qwen/tmp/synth_delta_256.log`, `.qwen/tmp/synth_delta_1024.log`). Half a percent of one lane at
1.4e-03 cannot produce a 9.3e-01 rms map difference, so the defect is in the part a positive-order
delta does not touch. The candidate that is both untested and large enough is the **negative-order
half**, which the march produces by mirroring (`(-1)^(ell+|s|) d^l_|m|,-2(pi - theta)`) instead of
from the marched lane; a general `alm` puts about half its power there.
`.qwen/tmp/synth_negm_delta.py <nside>` is written for exactly that (delta at `L-1-m`, compare column
`L-m`, with a wrong sign appearing as `S/E = -1` far from any pole where nothing else is marginal).

Probe hygiene notes: the march's output refs are float64, so these probes need
`jax_enable_x64=True` — with x64 off the store raises `Invalid dtype for swap: Ref dtype: float64.
Value dtype: float32`, which looks like a kernel bug and is not one. And `complex64` `flm` is
rejected by the exact route at both sizes (`Cannot lower jaxpr with verifier errors: type of return
operand 0 ... complex<f64>`), so the probe has to fall back to complex128 input; that is the exact
route's constraint, not the march's.

## Session 23 (2026-09-07): the rms 0.93 is `ell < |spin|`, and the synthesis march is the default

**This supersedes the last two paragraphs.** The negative-order hypothesis is wrong — the mirror is
correct. The whole 0.93 is the section of the band below the spin, which the march cannot represent and
which the probe had filled with the largest coefficients in the map.

The step that cracked it was to stop comparing maps and compare **columns**. `.qwen/tmp/synth_which_columns.py`
takes the full random band, marched against exact, and reports the residual norm per output column:

```
ref (2047, 3072) 4.5859e+02 ||ref|| 6.2040e+02   new max 4.9634e-01 ||new|| 1.6879e+01
   col   m=col-L    ||d|| col  ||ref|| col     ratio  worst theta    cos t
  1536        +0   6.1921e+02   6.1966e+02 9.993e-01     0.001595  +1.0000
  1535        -1   1.8257e+01   1.7276e+01 1.057e+00     0.834086  +0.6719
  1537        +1   1.8257e+01   2.1839e+01 8.360e-01     1.581213  -0.0104
  1534        -2   1.1820e-06   7.9791e+00 1.481e-07     0.059013  +0.9983
  1538        +2   8.8261e-07   5.8836e+00 1.500e-07     2.887348  -0.9679
columns above 1e-6 of the worst column's error: 3 of 3072
```

(`.qwen/tmp/synth_which_columns_512.log`). **Three** dead columns out of 3072 — `m = -1, 0, +1` — and
everything with `|m| >= 2` agreeing to ~1e-07. The m = 0 column carries almost all the reference power
because the probe's red `l^-1.5` spectrum puts its biggest coefficients at `ell = 0, 1`, and `ell < 2`
is precisely the set of degrees that can reach those three columns. The march's recurrence starts at
`nstart = max(m, spin)` with `P_1 = ((alpha+beta+2)/2)cos(theta) + (alpha-beta)/2`, so it contributes
exactly zero from `ell < |spin|`; s2fft's `flm_to_ftm` does not skip them.

Two controls close it out. Filling the band **from `ell = max(m, 2)`** instead of `ell = 0` makes the
march match the exact route at <= 1.75e-06 for every `m in {0,1,2,3,6}` up to the full band, with
`||new||/||ref|| = 1.000000` (`.qwen/tmp/synth_per_m_ell_256.log`) — so both order signs and every
degree are fine, and the earlier `synth_negm_delta.py` suspicion is discharged. And the convention
difference is real but unreachable in a pipeline: ducc0 *does* consume sub-spin terms if you hand them
to it (injecting `|alm| = 1e3` at `ell < 2` moves its map by 1.427e-01 of its maximum), yet its own
spin-2 analysis emits exactly 0.0 at `ell = 0` and 6.2e-06 at `ell = 1` (`.qwen/tmp/ducc_ell_below_spin.log`),
i.e. nothing that came out of an analysis carries that power.

**What decides the default is accuracy against ducc0, not against our own exact route**
(`.qwen/tmp/synth_vs_ducc.py`, alms taken from ducc0's analysis of a masked pair at Nside 1024, where
the seam is live):

```
ducc0 analysis at ell<2: max|alm| 1.620e-09  (max over all ell 4.181e-09)
 exact: max|d|/max|ref| 1.095e-09  rms|d|/rms|ref| 1.289e-10  |s|=1.00e+00
 march: max|d|/max|ref| 1.968e-04  rms|d|/rms|ref| 6.370e-06  |s|=1.00e+00
```

6.4e-06 rms and a 2e-04 worst pixel — the polar-lane row error arriving at the pole-most ring — on the
input a pipeline actually produces. `synth_requested` therefore now takes the same rule as analysis
(march wherever `slice_declined`, own `GMASTER_SPIN2_MARCH_SYNTH` override, `=0` restores the exact
route), shipped in `13e9078`. Fresh-process score cell at Nside 1024: `alm2map 112.1 → 122.3 ms =
0.92x` where it was **0.01x** (the 13.2 s scatter loop), `map2alm 118.0 → 107.8 ms = 1.10x`
(`.qwen/tmp/score_1024_spin2_synthdefault.log`). Suite 141 passed / 3 skipped
(`.qwen/tmp/pytest_s23a.log`).

**The flags, measured rather than inferred** (`/scratch/scratch-lxu/agent_dev/auto_research_agent/GMaster/.qwen/tmp/route_table_1024.py`
counts seam calls with every executable cache dropped between cells, so it reports the route taken and
not the flag value):

```
                                           env  spin  analysis seam  synthesis seam
                                     (default)     2              1               1
                                       MARCH=0     2              0               1
                                       SYNTH=0     2              1               0
                              MARCH=0, SYNTH=1     2              0               1
                              MARCH=1, SYNTH=0     2              1               0
```

and spin 0 is `0 0` in all seven combinations. The two flags are independent, which is the point: the
directions have different economics against ducc0 (1.10x ahead one way, 0.92x behind the other) even
though they share the no-slice default.

**Method note worth keeping.** A per-row delta probe and a linearity test cannot see a missing index
range — `f(a+b) = f(a) + f(b)` is satisfied by a route that silently drops a whole set of inputs, on
both sides of the equation. When a discrepancy lives in a handful of output columns, ask which *input*
indices can reach them before believing anything about magnitude; here that single question named
`ell < |spin|` in one step, after two sessions of "improve the polar lane".

### The synthesis seam was 65-81 % assembly, not marching (`13e9078` → `da60123`)

With both march directions on, the seam timings were 137.84 ms (synthesis) against 102.05 ms
(analysis) at Nside 1024 and **1809.49 ms against 568.91 ms** at 2048 — synthesis superlinear
(13x for 8x the pixels) where analysis scales 5.7x. `.qwen/tmp/synth_superlinear.py` put the growth in
the per-window assembly: 1.28 ms per window at 1024 against 10.74 ms at 2048, on a `(ntheta, 2L)`
complex128 destination that is 0.37 GiB and 1.50 GiB respectively. `_inverse_impl` wrote each window
with `out.at[:, cols].set(...)`, and **XLA scatter is out-of-place**, so every window copied the whole
destination — 48 copies at 1024, 96 at 2048.

The window column ranges are disjoint and tile the result (`_spin_slice._windows` is
`(m0, m1, lo=m0)` over `range(0, L, 64)`; direct → `L+m0..L+m1-1`, mirror → `L-m1+1..L-n0` with
`n0 = max(m0, 1)`, ascending windows meaning *descending* mirror columns), so the assembly is one
n-ary concatenate: `[zero column, mirror blocks in reverse window order, direct blocks]`.

```
nside 1024  bit-identical: True  max|diff| 0.000e+00   scatter 138.75 ms -> concat 47.29 ms  (2.93x)
nside 2048  bit-identical: True  max|diff| 0.000e+00   scatter 1805.60 ms -> concat 346.99 ms (5.20x)
```

(`.qwen/tmp/synth_assembly_ab.log`; shipped score cells in `.qwen/tmp/score_assembly_fix.log`, accuracy
re-checked in `.qwen/tmp/synth_vs_ducc_assembly.log`). Scoreboard for spin 2 on the march route,
fresh process per cell, ducc0 → GMaster:

| nside | direction | ducc0 ms | GMaster ms | ratio |
|---|---|---|---|---|
| 1024 | map2alm | 116.7 | 107.4 | **1.09x** |
| 1024 | alm2map | 109.2 | 95.2 | **1.15x** (session start: 0.01x, then 0.92x) |
| 2048 | map2alm | 652.5 | 610.8 | **1.07x** |
| 2048 | alm2map | 614.5 | 729.3 | **0.84x** (session start: 0.39x) |

Nside 2048 had never been measured cleanly before this session — every earlier attempt died
`RESOURCE_EXHAUSTED` / `CUDA_ERROR_OUT_OF_MEMORY` under the other tenant's load.

**The same change in the analysis direction was measured and rejected**
(`.qwen/tmp/forward_assembly_ab.log`): `_forward_impl` has the same loop shape and disjoint columns,
but the concat arm is **0.37x at 1024 and 0.64x at 2048 and is not bit-identical**
(`max|diff| 3.77e-07`). Two look-alike halves of one kernel are not one code path; A/B each direction
and check bit-identity in the same probe as the timing. The analysis scatter chain stays. (The
non-identity is unexplained; it only went unchased because the arm lost on time — if a future attempt
at the analysis assembly looks fast, understand it first.)

`tests/test_sht.py::test_spin2_march_synthesis_matches_the_route_it_replaces` now pins the seam at
Nside 16/48/64 against `flm_to_ftm` (max rel 2.4e-06 / 4.9e-06 / 8.5e-06), with Nside 48 chosen because
`L = 144` makes the window widths 64/64/16 — the ragged last window is what the concatenated assembly
would break on. Nothing covered the march route before, because it is default-on only above every size
the suite runs at.

## Session 24 (2026-09-07): the small-Nside floor was Python dispatch, and fusing boundaries beat ducc0 everywhere below 1024

Every small cell that lost to ducc0 lost to the *host*, not the GPU. The two fixes are one jit
boundary moved and one boundary removed:

| spin | nside | dir | before | after | ducc ms | source |
|---|---|---|---|---|---|---|
| 2 | 64 | map2alm | 0.25x | **1.88x** | 0.6 | `score_small_default.log` → `score_s24_small.log` |
| 2 | 64 | alm2map | 0.94x | **1.49x** | 0.5 | |
| 2 | 128 | map2alm | 0.66x | **2.63x** | 1.4 | |
| 2 | 128 | alm2map | 1.79x | 1.75x | 1.0 | |
| 2 | 256 | map2alm | 2.03x | **3.31x** | 4.8 | |
| 2 | 256 | alm2map | 1.98x | **2.05x** | 3.2 | |
| 2 | 512 | map2alm | 2.96x | **3.23x** | 30.0 | |
| 2 | 512 | alm2map | 3.03x | 2.82x | 24.5 | |
| 0 | 128 | map2alm | 0.54x | **1.29x** | 0.8 | `score_s24_spin0.log` → `score_s24_spin0b.log` |
| 0 | 128 | alm2map | 0.94x | 0.97x | 0.7 | |
| 0 | 256 | both | 1.88x / 1.38x | 1.74x / 1.18x | 2.3 / 1.7 | unchanged route, ducc jitter |

Relative alm error is unchanged in every one of these cells (2.7e-07 / 3.5e-07 / 3.2e-07 at spin 2,
2.1e-07 / 2.5e-07 / 2.7e-07 at spin 0) because both changes are boundary changes, not arithmetic ones.

### The polarised route was three jits and an eager epilogue (`ffc1fbb`)

`_map2alm_once_slab` ran the ring FFT, the slab contraction and `_finish_forward_s2fft` as three jits
and then did `plus[ell, L_work-1+order]`, its parity conjugate and the E/B combination eagerly. At
Nside 64 that costs 1.700 ms for a 49152-pixel map whose device work is ~0.2 ms. One program over the
same body: **7.91x** at 64 (1.700 → 0.215 ms), 2.54x at 128, 1.05x at 256, bit-identical
(`.qwen/tmp/slab_fuse2.log`); the synthesis is 1.29x / 1.09x / 1.06x the same way
(`.qwen/tmp/slab_fuse_synth.log`). The fused body takes the ring tables as an argument so
`_spin_ring_analysis_tables`' host cache keeps serving them instead of a table being baked into the
executable.

The boundary carries the Wigner-d slab, so it is gated at `_SLAB_FUSE_MAX_BYTES = 4 GiB` — Nside 256 in
fp32 (2.44 GiB) and no further. Nside 512's slab is 18.7 GiB, and a boundary holding both it and the
maps is exactly the layout doubling `_inverse_latitudinal_slab` exists to avoid.

`test_fused_slab_route_is_bit_identical_to_the_split_route` (`62115ff`) asserts exact equality between
the fused and split bodies in both directions and pins the gate, so a geometry cannot drift onto a
boundary that was measured and rejected.

### Spin 0: the traced/eager crossover was stale (`a043d91`)

`_PALLAS_TRACED_MAX_L` was 256, justified by "Nside 128 goes 15 → 18 ms and Nside 256 58 → 82 ms"
traced. Neither number reproduces. Forced traced, same body, bit-identical
(`.qwen/tmp/spin0_traced.log`, `.qwen/tmp/spin0_traced_it2.log`):

| nside | `n_iter=0` fwd / inv | `n_iter=2` fwd / inv |
|---|---|---|
| 64 | 0.98x / 1.00x | 1.00x / 1.00x |
| 128 | **3.17x** / 1.96x | **2.18x** / 1.77x |
| 256 | 1.22x / 1.30x | 1.08x / 1.28x |
| 512 | 1.02x / 1.01x | — |
| 1024 | 1.01x / 1.01x | — |

Crossover moved to 384 (Nside 128), which is the cell that was losing. 256 is *not* traceable at all
right now, for a reason unrelated to speed: the Legendre band engages there and
`_theta_matrix._band` drains and clears its build caches with `slab.block_until_ready()` every
`_BUILD_CLEAR_EVERY` blocks — tracers have no such method — so a band geometry first built inside an
outer trace dies with `AttributeError` on `float32[160, 64, 512]` (`.qwen/tmp/s24_256_0.log`, reached
through `_fused_forward_sht` → `positive_latitudinal` → `_band`). Guarding that loop would also let the
band be *built* inside the traced program, where XLA would recompute it on every call, so the gate is
the right place to stop. Unlocking the 1.2x at 256 means hoisting the band build out of the trace, not
just guarding it.

### Two probe traps worth remembering

`_map2alm_once_pallas_spin` and `_spin_forward_latitudinal` are **not** the shipped spin-2 route —
`_use_pallas_sht` returns False for `spin != 0`, so at small Nside spin 2 runs the slab route. A stage
probe that times them measures a dead path, and it announces itself: the "whole call" came out at
2.465 ms while the shipped `map2alm` was 1.713 ms (`.qwen/tmp/spin2_stage_floor.log`). The first fused
A/B was also wrong for the same family of reason — wiring `_forward_latitudinal` instead of
`_forward_latitudinal_slab` silently swapped in the generic scatter route and reported the fusion as
0.13x (`slab_fuse.log`) before the corrected probe gave 7.91x. An A/B must differ in exactly one thing,
and the shape of a table that contradicts the shipped number is how to catch it.

ducc0's cold-vs-warm question is closed: first call is 0.96–1.04x the min over later calls at both
1024 and 2048 and both spins, with RSS growth of 0.00–0.30 GiB (`.qwen/tmp/ducc_cold_vs_warm.log`), so
ducc0 gets no benchmark cache advantage and the speedups above are honest.

Suite: 144 passed / 3 skipped (`pytest_s24a.log`), 146 passed / 3 skipped after the new test
(`pytest_s24b.log`), both on GPU1.

## Session 25 (2026-09-07): the synthesis march is issue-bound — cutting the accumulation closed the 2048 cell

Spin-2 `alm2map` was the last sub-1x cell: 729.3 ms against ducc0's 614.5 at Nside 2048 (0.84x). It is
**507.5 ms = 1.20x** now, and Nside 1024 went 95.2 → **71.4 ms** against 109.8 (**1.15x → 1.54x**),
from one change: the synthesis contraction accumulates in plain float32.

### Where the time was (`.qwen/tmp/synth_march_decomp.log`, one geometry per process)

| nside | `_inverse_impl` | build (all windows) | assembly | march |
|---|---|---|---|---|
| 1024 | 88.82 ms | 11.31 | ~0.3 | **88.77 (100%)** |
| 2048 | 676.48 ms | 37.67 | 1.05 | **675.26 (94%)** |

The "no march" arm dead-code-eliminates the geometry build (nothing consumes it), so build+assembly
reads 1.05 ms while build alone reads 37.67; the honest split at 2048 is ~638 kernel + 37 build + 1
assembly. Assembly — the piece the previous three sessions worked on — was already 0.15%.

### Two things that were not the problem

**Layout.** `_ST`/`_SW` swept one config per process; every arm produced the same checksum, so both
knobs are numerically free (`.qwen/tmp/synth_sweep_1024b.log`): 512/4 **88.97** ms, 256/2 89.53,
512/8 103.62, 256/4 106.44, 1024/8 110.93, 128/4 113.39, 512/16 116.17. The shipped 512/4 is already
optimal; re-checked after the win (`.qwen/tmp/synth_sweep_fast_final.log`), still 512/4.

**The critical path.** Moving the 2^±24 guard one step off the recurrence chain — deciding it from the
carried `ch, ph` instead of the just-produced `nxt`, which is the same pair the current code tests —
gave **1.012x** and was *not* bit-identical (worst `max|diff|` 1.880e+15,
`.qwen/tmp/synth_guard_delay_1024.log`). Abandoned. The guard is not what the march waits on.

Both null results point the same way: the step is warp-issue-bound, so cost is vector operations per
(m, ell, theta) triple, nothing else.

### The accumulation is 38% of the step and most of it is bookkeeping

Three accumulation arms over the same march and the same stores, all 48 windows at Nside 1024
(`.qwen/tmp/synth_acc_cost_1024.log`):

| arm | ms | ratio | max\|diff\| vs exact | rel to window max |
|---|---|---|---|---|
| exact: (value, limb) products, (value, limb) accumulator | 83.41 | 1.000x | — | — |
| drop the accumulator limb | 67.20 | 1.241x | 1.260e-07 | 2.150e-06 |
| drop every limb | 51.96 | **1.605x** | 1.260e-07 | 2.150e-06 |

Once the accumulator limb is gone, the product limbs contribute nothing: the two cheap arms are
identical in output and the cheaper is 16% faster than the middle. `_accumulate_fast` keeps the range
half of `_accumulate` verbatim (the `lift`/`ds` alignment and the 2^±24 guards — what keeps a
6000-term sum inside float32's exponent range) and drops only the precision half. The fast arm does not
load the coefficient limbs at all.

### Accuracy, measured in the same log as the speed

Map level, flat-spectrum (worst-case) synthesis alms, Nside 1024, base map max 6.839e-01 / rms
1.226e-01 (`.qwen/tmp/synth_fast_acc.log`):

```
fast - comp   max|diff| 5.525e-07   rms diff / base rms 8.111e-07
alm2map: comp 101.13 ms   fast 76.50 ms   1.322x
```

~350x below the 1.968e-04 max / 6.370e-06 rms the marched synthesis already ships against ducc0, and
against the *exact* route the two arms are indistinguishable (`.qwen/tmp/synth_march_rel.log`: max rel
2.54e-06 vs 2.42e-06 at nside 16, 8.52e-06 vs 8.47e-06 at 64, 1.01e-05 vs 1.01e-05 at 128 — the
contract is limited by the march, not the accumulation).

### Scoreboard after (`.qwen/tmp/score_fast_synth.log`, fresh process per cell)

| nside | spin | ducc ms | GMaster before | after | ratio |
|---|---|---|---|---|---|
| 1024 | 2 m2a / a2m | 119.7 / 109.8 | 108.5 / 95.2 | 108.5 / **71.4** | 1.10x / **1.54x** |
| 2048 | 2 m2a / a2m | 649.5 / 609.8 | 587.2 / 729.3 | 587.2 / **507.5** | 1.11x / **1.20x** |
| 512 | 2 m2a / a2m | 28.8 / 25.6 | 9.2 / 8.5 | 9.2 / 8.5 | 3.15x / 3.01x (slab route, untouched) |

**Every cell from Nside 64 to 2048 is now above parity except spin-0 `alm2map` at Nside 128 (0.97x).**
Escape hatch: `GMASTER_SPIN2_MARCH_SYNTH_ACC=comp` restores the compensated accumulator.
Suite: 146 passed / 3 skipped (`pytest_s25b.log`), GPU1.

### Where the analysis time is NOT (`.qwen/tmp/analysis_emit_cost.log`, Nside 2048, 96 windows)

```
ship      338.25 ms   1.000x
noscale   335.25 ms   1.009x     # drops the cross-lane max and the per-lane pow2 rescale
noreduce  228.42 ms   1.481x     # drops the 4-channel theta contraction
```

So the per-degree rescale (`jnp.max(ex)` + vector `_pow2` + float64 `exp2`) is ~1% and the
shared-exponent idea for it is worthless; the remaining analysis cost is the compensated recurrence and
the contraction itself. `.qwen/tmp/analysis_coeflimb.py` tests the next candidate — whether the three
coefficient low limbs (`c1l`, `c0l`, `cbl` loads + FMAs per degree) are worth their accuracy.

### Documentation defect worth fixing

`_inverse_impl`'s comment cites `.qwen/tmp/synth_assembly_ab.log` for "138.75 → 47.29 ms (2.93x) and
1805.60 → 346.99 ms (5.20x), bit-identical". That file exists and contains no such run: it is the
*x64-less march-vs-table* probe, whose concat form reads **slower** (0.38x at 1024, 0.86x at 2048,
`bit-identical: False`). The 2.93x/5.20x claim is almost certainly about a different probe (the fp32
assembly A/B), so the citation must be regenerated, not trusted. General trap: re-running a probe under
an existing log name destroys the evidence a source comment cites — use a new name. Same trap bit again
this session: a stale `pytest_s25b.log` left from an earlier session read "141 passed" before the
current run had overwritten it.

**Resolved the same session by regenerating the evidence** (`.qwen/tmp/synth_assembly_ab2.py` →
`.qwen/tmp/synth_assembly_ab2.log`, one process, x64 on, both arms on identical inputs):

```text
nside 1024  bit-identical True  max|diff| 0.000e+00   scatter 114.36 ms -> concat  64.35 ms  1.78x
nside 2048  bit-identical True  max|diff| 0.000e+00   scatter 1611.28 ms -> concat 482.72 ms  3.34x
```

The design choice stands (bit-identical, large win), the old magnitudes do not: **2.93x / 5.20x were
optimistic; 1.78x / 3.34x is what the same A/B measures today.** `_inverse_impl`'s comment now cites the
new log and says so in as many words. Quote the new pair.

### Next

1. **Spin-2 `map2alm` at 1024/2048 is now the weakest spin-2 cell (1.10x / 1.11x)**, and its rescale is
   free (above). Candidates left: the analysis coefficient low limbs, the `_row_coeffs` /
   `_init_exponents` / `log_norm` table builds (analysis geometry has no separate budget knob —
   `_forward_latitudinal_march` builds rows *and* marches), and the tile reduction over `ntile`.
2. Nside 4096 measured the same day (below) — the march serves it, and it is now the worst spin-2 cell.
3. Nside 256 spin-0 traced route (1.22x/1.30x measured) blocked on hoisting the Legendre band build out
   of the trace, not on guarding its cache drain.

### Session 25 addendum: the analysis march measured out of candidates

Every lead in the list above was measured the same day, and all four are dead. The analysis route at
1024/2048 is the march (`slice_declined` is true there), and what it costs is the compensated
recurrence plus the 4-channel contraction.

**The geometry rebuild is not the 184 ms.** `_forward_impl` calls `_window_geometry` per window *inside*
the trace, so four `gammaln` on `(mb, L)` float64 plus the Jacobi limb split run on every call. Timed as
one jit over all windows (`.qwen/tmp/analysis_geom_cost.log`): **8.66 ms at Nside 1024 (48 windows,
0.38 GiB) and 31.39 ms at 2048 (96 windows, 1.51 GiB)** — 2-5% of the call. Caching it is worth a
commit only if the freed time is needed elsewhere; it is not a lever.

**The analysis layout is already at its optimum** (one config per process, `.qwen/tmp/an_sweep_1024.log`,
Nside 1024 latitudinal step): `_TILE/_WARPS` = **256/1 102.49 ms**, 512/2 114.90, 512/1 132.27,
128/1 134.83, 512/4 140.21, 1024/4 150.03. The synthesis march likes 512/4 and the analysis march likes
256/1; do not "unify" them.

**The scatter→concat port fails again, as this memory tree already said it would.** The synthesis fix
does not transfer: same disjoint-column structure, `(L, 2L-1)` destination, and the concat arm measured
1.019x / 1.032x / **0.992x / 0.981x** at Nside 256/512/1024/2048
(`.qwen/tmp/analysis_assembly_ab.log`, `max|diff|/max|out|` 0 to 1.8e-17). An earlier run of the same
idea recorded 0.37x/0.64x. The magnitudes differ between probe shapes; the verdict does not. Reverted.

**The coefficient low limbs buy 3.1%.** Dropping `c1l`, `c0l` and `cbl` (three loads and three FMAs per
degree) measured **1.031x** (`.qwen/tmp/analysis_coeflimb.log`) — the transcription's own accuracy
column in that probe is not trustworthy, but the speed ceiling is: 3% is not worth a new default.

**Two probe traps hit and recorded.** (a) Two of my own background jobs on GPU1 (a pytest suite plus a
probe) made the same shipped step read 98.7 ms and 170.6 ms, which looked like a routing contradiction
and was contention — check `pgrep`/`nvidia-smi` before every measurement. (b) A `while pgrep -f
"<pattern>"` wait loop matches its own command line: two loops queued behind a test suite never fired,
and a `pkill -f` aimed at one killed the killing shell. Use the bracket form (`[a]n_sweep`) or kill by
PID.

### Session 25 addendum 2: Nside 4096 spin 2 runs on the march, and the win does not survive the decade

The first end-to-end Nside 4096 spin-2 comparison ever taken here (fresh process, fp32 tables, one card,
`.qwen/tmp/score_n4096_spin2.log`, `synth reserve: 16.0 GiB pool: 71.2 GiB`, `band budget: 40 GiB`):

```text
  4096    2  map2alm     4025.6      4765.7       0.84x 1.9e-04 |s|=1.6e+07  1359/1439
  4096    2  alm2map     3918.9      3594.6       1.09x          -  synth=-
```

So the capacity question is settled — **the march fits at 4096**: no OOM, no fallback, no
`XLA_PYTHON_CLIENT_PREALLOCATE` trick, control bandwidth 1359/1439 GB/s (card healthy). `synth=-` is
expected: the marched route never populates `_theta_matrix._SYNTH_CACHE`. The alm agreement is
`1.9e-04` against the correctly rescaled denominator (the `|s| = 1.6e+07` convention factor grows with
`L`; at 2048 it reads `2.5e+05`), the same order as the 2048 cell.

The performance answer is worse than hoped, and it is a *scaling* answer, not a fixed-cost one:

| direction | ducc 2048 → 4096 | GMaster 2048 → 4096 | ratio 2048 → 4096 |
|---|---|---|---|
| `map2alm` | 649.5 → 4025.6 (**6.20x**) | 587.2 → 4765.7 (**8.11x**) | 1.11x → **0.84x** |
| `alm2map` | 609.8 → 3918.9 (**6.43x**) | 507.5 → 3594.6 (**7.08x**) | 1.20x → **1.09x** |

The march's work is *exactly* the number of `(m, ell, theta)` triples, which is `∝ L² · ntheta ∝ n³` =
8.0x for a doubling — GMaster analysis pays 8.11x, i.e. it is at its asymptotic issue-bound cost with
nothing leaking. ducc0 gains sub-cubic (6.20x) between these two sizes, so whatever it is bound by at
2048 relaxes by 4096, and the crossover that session 25 pushed the other way at 2048 comes back. Making
4096 win is therefore **not** reachable by removing per-degree overheads (they are already ~1-5% of the
step, see the four dead candidates above); it needs either a lower op count per triple than the
compensated Jacobi recurrence + 4-channel contraction, or a different work decomposition. The cheapest
structural candidate is the anchored-restart idea in memory
(`spin2-anchored-jacobi-march.md`), which trades the serial degree march for parallel segments — its
measured XLA segment map was 2.3x behind ducc at small `N`, so it must first beat the march, not ducc.

Do not read this as "4096 is broken": 1.09x synthesis and 0.84x analysis at `L = 12287` on one GPU,
against 192 cores, is the first time this repo has a *measured* 4096 polarised number at all.

### Session 25 addendum 3: the analysis `emit` contraction is at its floor too

The last untested block was whether `jnp.sum(val[:, None] * r, axis=0)` pays for materialising a
`(chunk, NC)` tile (256 float32 lanes x 4 channels on top of the march state with `_WARPS=1`). Three
arms, one config per process, `.qwen/tmp/analysis_emit_form.log`:

| arm | Nside 1024 | Nside 2048 | reading |
|---|---|---|---|
| `ship` — the shipped 2-D contraction | **103.13 ms** | **537.27 ms** | baseline |
| `sepchan` — four 1-D `jnp.sum(val * r_c)` | 105.52 (0.977x) | 625.17 (**0.859x**) | splits the tree, costs real time |
| `nostore` — same contraction, every degree to slot 0 | 110.69 (0.932x) | 569.74 (0.943x) | store volume is *not* a cost either |

`sepchan` agrees with `ship` to `max|diff|/max|out|` = 1.8e-13 (1024) and 1.1e-13 (2048), so the arms are
the same maths; both rewrites lose. Combined with the `noscale` arm (1.009x, deleting the exponent
rescale) this pins the emit's cost: **the multiply-accumulate itself** — which is the same conclusion
`analysis_emit_cost.py` gave from the other side (deleting the whole emit block is 1.481x at 2048, and
deleting only its bookkeeping is nothing).

That closes the analysis march. Recorded so the next session does not spend GPU time here: the per-degree
rescale (1.009x), the coefficient low limbs (1.031x), the assembly form (0.98x), the tile layout (already
256/1), the geometry rebuild (2-5%) and now the contraction form (0.86-0.98x) are all measured dead. The
analysis step is its compensated recurrence plus its contraction, and beating it is an algorithm change
(anchored restarts, or a cheaper per-triple op count), not a rewrite of any block in it.

### Session 25 addendum 4: spin 0 has a route cliff at 2048, and Nside 4096 is now measured for both spins

Spin 0 at Nside 4096 had never been measured end to end either. Fresh process, fp32 tables, one card,
`.qwen/tmp/score_n4096_spin0.log`:

```text
  4096    0  map2alm     2014.4      7516.9       0.27x 3.4e-07 |s|=1.6e+07  1410/1373
  4096    0  alm2map     2022.3     10050.4       0.20x          -  synth=-
```

Put next to the spin-0 cells below it, this is a **route cliff, not a scaling law**:

| nside | ducc `map2alm`/`alm2map` | GMaster | ratio | route |
|---|---|---|---|---|
| 1024 | 57.4 / 51.1 | 28.5 / 30.9 | **2.02x / 1.65x** (`sht_vs_ducc_s25_1024_0.log`) | precomputed Legendre band |
| 2048 | 301.5 / 287.3 | 1048.1 / 1356.6 | 0.29x / 0.21x (`sht_vs_ducc_s26_2048_0.log`) | band declined → fp64 scalar kernel |
| 4096 | 2014.4 / 2022.3 | 7516.9 / 10050.4 | **0.27x / 0.20x** (above) | band declined → fp64 scalar kernel |

GMaster's own time grows 28.5 → 1048 ms (37x) for a doubling whose work grows 8x, because
`utils._prefer_theta_band` compares the band against `_MATRIX_BAND_BUDGET` (40 GiB) and the fp32 band
only engages at 1024 (36.7 GiB); both Nside 2048 and 4096 decline and fall through to
`scalar_forward_latitudinal`, which regenerates the Legendre row in **float64** on every call. That is
the 1/64-rate path of `fp64-roofline-wall`, and it is why these are the two worst cells in the repo —
spin 0 loses 3.5-5x while spin 2 at the same sizes is within 20% of parity or ahead.

These four cells (spin 0 at 2048 and 4096, both directions) are the largest single gap against ducc0
anywhere in the scoreboard, and unlike the spin-2 analysis cells they are not a measured-out kernel:
there is no fp32 marched spin-0 route at all. `gmaster/_spin_march_pallas.py` is closed over `SPIN = 2`
(`march_requested`/`synth_requested` return False for any other spin), but its math degenerates cleanly
at s = 0: `alpha = beta = m`, `a0 ∝ m·s` vanishes, `ns = m`, and `_log2_norm` becomes
`log2 sqrt((l+m)!(l-m)!/(l!)²)`. The mirror channel would change form (for spin 0 the negative order is
the same row times `(-1)^m`, with no `theta -> pi - theta` reversal).

### Session 25 addendum 5: the north-south fold is real, and it is a **spin-0 only** lever

The lever that looked like a free 2x for every march is the HEALPix north-south fold: rings pair as
`i <-> nring-1-i` with *identical* transform weights, so if the latitudinal row obeys a self-parity law
the whole southern half folds onto the north and the (m, ell, theta) triple count — the entire cost of a
table-free march — halves. Both premises were tested before writing a kernel (`.qwen/tmp/ns_fold_probe.log`,
CPU, nside 32/64):

```text
  grid: max|cos(theta_rev)+cos(theta)| 3.331e-16   rel max|w_rev - w| 0.000e+00
  spin0 antisymmetric : (l+m) even 2.104e-13   (l+m) odd 1.000e+00
  spin0 symmetric     : (l+m) even 1.000e+00   (l+m) odd 2.900e-13
```

So the law holds **exactly for spin 0**: `d^l_{m0}(pi - theta) = (-1)^(l+m) d^l_{m0}(theta)`, the ring
weights are bit-identical across the mirror, the pixel mirror at fixed phi is an exact involution, and no
longitude phase appears. Spin-0 alms of a mirror-antisymmetric map vanish in the even class to ~1e-13
relative, and the symmetric map in the odd class.

**It does not hold for spin 2, and that is structural, not numerical.** `_spin_slice._build`'s own
docstring states the relation the shipped polar route depends on:

    T[m, pi - theta, ell] == (-1)**(ell - spin) * T[-m, theta, ell]

the spheroid row reflected through the equator lands on the **opposite order**, not on itself — which is
precisely what the march's mirror channels already consume (they contract `F(pi-theta, -m)` against the
`+m` row). Folding `+m` would require the `-m` row in the same program, i.e. marching two rows, which
gives the work back. The fold A/B confirms it numerically (`.qwen/tmp/ns_fold_ab.py`, Nside 1024 fp32,
half theta with an 8-channel rhs carrying both parity classes):

```text
  kernel only, 48 windows: shipped 96.91 ms   folded 90.55 ms   ** 1.070x **   (theta lanes 2048/4095)
  fold vs shipped march: max|diff|/max|out| 9.391e-01   rms 2.108e-02
```

Two independent lessons in those two lines. The **0.939** is the missing self-parity (the fold is wrong
for spin 2, as predicted). The **1.070x** is a warning for the spin-0 version: doubling the channel count
to 8 to carry both parity classes leaves the contraction's multiply count unchanged while doubling each
program's reduction width and its accumulator footprint (with `_TILE/_WARPS = 256/1` that is 8 fp32
accumulators x 8 lanes per thread), so halving the recurrence bought 7% instead of the ~1.5x the
68%/32% split predicts. **A spin-0 fold must halve the channels too** — a 4-channel parity-selected emit
(a two-degree unrolled loop over an `(2, chunk, 4)` rhs block, which needs only the 2-argument
concatenate the Triton lowering does support), not an 8-channel one.

Where that leaves the plan, in value order:

1. **Spin-0 folded band** (best prize-per-risk, and it does not need a new kernel): the band builder
   already splits rows by parity of `ell - m0` (`_theta_matrix._band`), which is exactly the two classes
   the fold needs, and the 1024 analysis band is 36.7 GiB fp32 read every call — a memory-bound stage
   inside a 28.5 ms `map2alm` against ducc0's 57.4. Building it over the northern grid only halves both
   the resident band and the read, and the rhs becomes `C± = F(i,m) ± F(i',m)` with the equator lane
   split half into each copy. Same trick, applied to the synthesis band, halves `alm2map` too.
2. **Spin-0 marched route** at 2048/4096 (the 0.20-0.29x cells), with the fold as above: the march
   replaces the declined band + fp64 scalar kernel; the closed form, limbs, windows and exponent
   bookkeeping all already exist in `_spin_march_pallas` and degenerate at s = 0.
3. Spin 2 keeps its current routes. Do not try to fold it: the row's equatorial reflection changes the
   order, not the row.

### Session 25 addendum 6: the folded spin-0 band is priced — 95% of the call is the band read

Whether the spin-0 fold is worth building comes down to whether the latitudinal stage is band-read
bound. Measured directly (`CUDA_VISIBLE_DEVICES=1 JAX_ENABLE_X64=1 GM_PREC=fp32 python -u
.qwen/tmp/spin0_stage_split.py 512 1024`, median of 5, shipped default route):

```text
 nside     total       fft   lat(band)  lat share   band GiB  implied GB/s
   512      4.57      0.28        3.71     81.1%        4.7          1358   band=on
  1024     28.12      1.72       26.83     95.4%       36.8          1471   band=on
```

So at Nside 1024 the ring FFT is 1.72 ms of a 28.12 ms call and the band contraction is **26.83 ms
(95.4%)**, consuming an implied **1471 GB/s** — the same number the score harness's independent fp64-sum
control reports for this card (1410/1373, 1359/1439 GB/s in the two 4096 logs), i.e. the stage is at the
achievable streaming rate, not at a scheduling limit. **Reading half as many band bytes therefore takes
about half the time**, and the fold's arithmetic is:

| nside | ducc `map2alm` | GMaster now | GMaster with a northern-grid band | ratio now → then |
|---|---|---|---|---|
| 512 | ~14.5 | 4.57 | ~2.7 (0.28 + 3.71/2) | 3.15x → **~5.4x** |
| 1024 | 57.4 | 28.12 | ~15.1 (1.72 + 26.83/2) | 2.02x → **~3.8x** |

(The 512 ducc figure is the shipped score cell; the "then" column assumes the fold halves `lat` and
leaves the fft stage alone, which is what the 95%-at-peak-bandwidth measurement implies. It does *not*
help at Nside 2048: halving a ~290 GiB fp32 band is still 145 GiB against a 71.2 GiB pool, so those
cells still need the marched route.)

Two concrete work items follow, both spin-0-only and both now priced:

1. **Northern-grid band.** `_theta_matrix._band` already returns rows split by parity of `ell - m0`,
   which is exactly the `(-1)^(l+m)` pairing, so the builder change is to march the northern grid (with
   the equator lane halved) and the consumer change is to feed `C± = F(i,m) ± F(i',m)` instead of `F`.
   Halves the resident band *and* the read: 36.8 → 18.4 GiB at 1024, which also relaxes
   `_MATRIX_BAND_BUDGET` pressure at 1024. Do the analysis band first (`map2alm` is 95% one stage),
   then the synthesis copy in `_synth_band`.
2. **Marched spin-0 route** for the band-declined 2048/4096 cells (0.29x/0.21x and 0.27x/0.20x), with a
   **4-channel parity-selected** emit — not the 8-channel form that measured 1.070x.

## Session 26 (2026-09-07): addendum 6's band prize was already spent; the folded *march* ships and 2048/4096 spin 0 stop losing 4-5x

### Addendum 6 work item 1 was already shipped — the resident band is northern and folded today

The first thing on last session's priced list was "build the band over the northern grid only, 36.8 →
18.4 GiB at 1024". Reading `_theta_matrix` before writing it kills the item: **the band never stored the
south.** `_build_slab` (line 92) takes `north = (len(theta) + 1) // 2` and builds `sine/cosine` from
`theta[:north]` only; `band_bytes` (line 298) counts `north = (4*nside-1+1)//2`; and `_transform` does
the fold on the *right-hand side*, contracting `north_rhs ± sign[:, None] * south_rhs` with
`partner = ntheta - 1 - jn` and `sign = 1 - 2*(arange(L) & 1)`, the two signs being exactly the
parity-split row groups the slab already holds. So the 36.75 GiB the stage split measured at 1024
**is** the folded table, `18.4 GiB` was a phantom, and the `2.02x → ~3.8x` (1024) and `3.15x → ~5.4x`
(512) "then" column in addendum 6's table was never available. Recorded here because those two numbers
were the stated reason to spend a session on the builder.

Nothing else from addendum 6 changes: the fold really is spin-0-only (addendum 5), and the marched route
(item 2) was the live prize.

### What shipped: one compensated march, northern grid, four channels, both directions

`gmaster/_spin_march_pallas.py` is no longer closed over `SPIN = 2`. `spin` is now a parameter of
`_kern`, `_call`, `_log2_norm`, `_window_geometry`, `_kern_synth` and `_call_synth` (defaulting to
`SPIN`, so every spin-2 call site and compiled kernel is unchanged), and two new entry points use it at
`s = 0`:

- `forward_latitudinal_positive(positive, weights, phase, *, L, nside)` — `(4n-1, L)` positive-m block
  in, `(L, L)` complex out. Builds `G_j = w_j e^{i m phi_j} F_j` for the whole grid, splits it into
  `G_north` and `G_partner` (`partner = ntheta-1-j`, zeroed on the equator lane because it is its own
  partner and counted once), stacks them as a 4-channel float32 rhs, and marches the northern lanes
  once. The `(-1)^(ell+m)` combine is applied **outside** the kernel to the two fp64 partials
  (`(P0 + iP1) + sgn*(P2 + iP3)`), where it costs one multiply per output instead of a per-degree
  select inside the recurrence. This is the 4-channel form addendum 5 asked for, not the 8-channel one
  that measured 1.070x.
- `inverse_latitudinal_positive(positive, phase, *, L, nside)` — its transpose: same northern march,
  two coefficient sets (direct, and the southern one carrying `(-1)^(ell+m)`), each ring's phi factor
  applied outside, south half stored at the reversed theta axis and truncated to `north-1` lanes
  because the equator contributes only through the direct accumulator. No `m = 0` mirror zeroing —
  that guard is spin-2-specific (there the mirror is the *negative order*), the spin-0 mirror is a
  hemisphere and every row of a window is genuine.

Seams: `utils._fused_forward_sht` / `utils._fused_inverse_sht` ask
`_spin_march.fold_requested(nside, L)` / `fold_synth_requested(nside, L)` **before** the band test.
Left unset the takeover happens exactly where the band is refused for *size*; `GMASTER_SPIN0_MARCH=1`
forces the march even where a band exists (that is how the two routes were compared), `=0` restores the
fp64 kernel, and `GMASTER_SPIN0_MARCH_SYNTH` overrides the synthesis direction alone.

### The two conventions that had to be learned before it produced numbers

1. **`_log2_norm` must be given the spin.** With the default still `SPIN = 2` inside it, `spin = 0`
   windows read `gammaln(ell - SPIN + 1)` and hit its poles at `ell = 0, 1`: the analysis fold returned
   `nan` at exactly `(ell, m) = (0,0), (1,0), (1,1)` and nothing else. That non-finite pattern is the
   diagnosis — poles, not cancellation.
2. **The rows carry `sqrt((2l+1)/4pi)`; the closed form does not.** After the poles were gone the fold
   was still ~0.5 relative off, so the two halves were separated: the fold algebra reproduces the band
   route to 1.27e-07 when fed the band's *own* rows (`.qwen/tmp/spin0_fold_rows.py`, arm C), and the
   marched rows match an independent `scipy.special.lpmv` reference (`d^l_{m0} = sqrt((l-m)!/(l+m)!)
   P_l^m(cos theta)`, Condon-Shortley included, `.qwen/tmp/spin0_fold_one.py`,
   `.qwen/tmp/spin0_fold_lanes.py`) to 9.4e-08 at lane 0. Both halves were right; the *convention*
   differed. The shipped rows are `sqrt((2l+1)/4pi) * d^l_{m0}` — measured as a constant ratio exactly
   equal to that factor over every `m` and lane to 5.2e-08 at Nside 32 (`band[0,0] = 0.282095 =
   1/sqrt(4pi)`, `band[1,0] = 0.488602`, `band[2,0] = 0.630783`) — while the closed form closes on the
   bare row. The analysis fold therefore multiplies by the degree factor after the kernel (putting it
   inside would also have landed on the spin-2 route, whose contract deliberately leaves it to
   `_finish_forward_s2fft`); the synthesis fold carries it in the coefficient, which is where the
   synthesis kernel's scaling lives. With `sc = exp2(lgs - lex) * (-1)**m` the coefficient is exactly
   the inverse of the marched row's own `(-1)^m exp2(-lgs)` convention, so the product is the bare row
   times `alm`, as the transpose of analysis requires.

### Score: the two worst cells in the repo are now within 10-30% of ducc0

One fresh process per row, GPU1, `ducc0 0.39.1` on 192 cores, `n_iter=0`, `L = 3*nside`, median of
3 (2048/4096) or 5 (512/1024) reps; GMaster times after a clock ramp, control bandwidths 1348-1421 GB/s.

| nside | dir | ducc0 ms | before | after | ratio before → after | speedup | log |
|---|---|---|---|---|---|---|---|
| 2048 | `map2alm` | 309.2 | 949.2 | **442.0** | 0.33x → **0.70x** | **2.15x** | `score_n2048_spin0_s26.log` → `score_n2048_spin0_synth.log` |
| 2048 | `alm2map` | 304.2 | 1258.4 | **323.3** | 0.24x → **0.94x** | **3.89x** | (same pair) |
| 4096 | `map2alm` | 2003.5 | 7516.9 | **2479.1** | 0.27x → **0.81x** | **3.03x** | `score_n4096_spin0.log` → `score_n4096_spin0_synth.log` |
| 4096 | `alm2map` | 2002.0 | 10050.4 | **2204.3** | 0.20x → **0.91x** | **4.56x** | (same pair) |

A later probe into the same call (see *rhs build* below) took the analysis direction further in the same
session, and those are the numbers now shipped: **2048 `map2alm` 419.9 ms = 0.76x** (ducc0 320.4) and
**4096 `map2alm` 2251.5 ms = 0.87x** (ducc0 1956.8), `alm2map` 322.1 (0.95x) and 2140.3 (0.93x),
`rel alm` unchanged at 6.9e-05 / 2.1e-04 (`.qwen/tmp/score_n2048_spin0_rhs1.log`,
`.qwen/tmp/score_n4096_spin0_rhs1.log`). The table above stays as the route-change-only A/B.

The two directions are independent, and the intermediate run proves it: with only the analysis fold in
place, `map2alm` was already 442.2 (0.72x) while `alm2map` sat unchanged at 1267.1 (0.24x)
(`.qwen/tmp/score_n2048_spin0_fold.log`).

**The band still owns every geometry where it fits, and the takeover is a win even where it doesn't.**
With `GM_PREC=fp32` (the config every band win in this project assumes) nothing moved:

```text
   512    0  map2alm       16.1         4.4       3.67x 3.7e-07      score_n512_1024_spin0_fp32.log
   512    0  alm2map       11.4         4.5       2.52x
  1024    0  map2alm       59.0        28.3       2.09x 3.6e-07
  1024    0  alm2map       57.2        30.9       1.85x
```

With the library's own fp64 default the 1024 band (73.5 GiB) is refused, so that cell is now served by
the march, and the A/B of the takeover itself is:

| nside 1024, fp64 tables | `map2alm` | `alm2map` | log |
|---|---|---|---|
| fp64 scalar kernel (`GMASTER_SPIN0_MARCH=0`) | 125.1 (0.50x) | 164.0 (0.34x) | `score_n1024_spin0_fp64kernel.log` |
| folded march (default) | **91.4 (0.68x)** | **69.3 (0.82x)** | `score_n512_1024_spin0_synth.log` |

so "the march takes over wherever the band is refused for size" is correct in both precisions —
1.37x and 2.37x ahead of the kernel it replaces, not just at 2048+ but at 1024 too.

Spin 2 was re-measured because the two kernels it runs on are the same functions that gained the `spin`
parameter: `1024 2 map2alm 118.3 / 112.4 = 1.05x`, `alm2map 108.4 / 75.9 = 1.43x`, `2048 2 map2alm
614.4 / 632.5 = 0.97x`, `alm2map 573.1 / 546.1 = 1.05x` (`.qwen/tmp/score_spin2_regression_s26.log`),
which is where session 25 left it. Suite on the shipped default after both seams landed: **146 passed,
3 skipped in 319.47 s** (`.qwen/tmp/pytest_s26.log`), the same counts as session 25.

### The same A/B end to end: Nside 2048 spin 0 goes from losing the MASTER pipeline to winning it

`python -m benchmarks.benchmark_pipeline --nside 2048 --spins 0 --repeats 3` (library default fp64
tables, `n_iter=3`, `nlb=30`, medians of 3, one process per arm; NaMaster CPU is the reference in both
runs, so the two `TOTAL` reference columns are independent measurements of the same thing):

| arm | NaMaster TOTAL | GMaster TOTAL | ratio | `field` | `coupling` | `coupled_cell` | rel `dCl` | log |
|---|---|---|---|---|---|---|---|---|
| `GMASTER_SPIN0_MARCH=0` (route before this session) | 10684 ms | 17020 ms | **0.63x** | 2517 → **7669** (0.33x) | 5579 → 1529 (3.6x) | 134 → 0 (313x) | 3.67e-12 | `pipe_n2048_spin0_nofold.log` |
| default (folded march) | 10823 ms | **7138 ms** | **1.5x** | 2543 → **2739** (0.93x) | 5628 → 1536 (3.7x) | 135 → 0 (301x) | **1.38e-06** | `pipe_n2048_spin0_fold.log` |

**2.39x on the whole pipeline**, and it flips the sign of the comparison: the cell that used to make
GMaster 1.6x *slower* than NaMaster at Nside 2048 now finishes in 0.66x NaMaster's time. All of the
movement is in `field` (mask construction runs the transform on every blurring iteration), which goes
0.33x → 0.93x and is now the worst remaining pipeline stage; `coupling` and `coupled_cell` are
unchanged, as they must be. The cost is visible too: `rel dCl` moves **3.67e-12 → 1.38e-06**, which is
the march's alm error propagating quadratically, and is the number to quote if anyone asks what the fold
costs a power spectrum.

### Accuracy cost of the trade, stated

The march replaces a route accurate to 3.4-3.7e-07 with the march's own float32-lane accuracy. Against
the fp64 kernel on the same input (`.qwen/tmp/spin0_fold_synth.py N`, march vs kernel / band vs kernel):

| nside | march vs fp64 kernel | band vs fp64 kernel |
|---|---|---|
| 32 | 8.62e-06 | 1.02e-07 |
| 64 | 3.05e-05 | 8.70e-08 |
| 128 | 7.91e-05 | 9.49e-08 |
| 2048 | 2.87e-03 (north) / 2.03e-03 (south) / 6.59e-08 (equator lane) | band declined |

Two notes on reading that table. The south half is not uniformly worse than the north — at 64 it is
3.05e-05 against the north's 7.59e-06, at 128 it is the *north* (ring 0) that is worst — because the
southern rings receive only the `(-1)^(ell+m)` accumulator, an alternating sum in `ell` whose
cancellation amplifies the same lane error. And the harness's own end-to-end number for the analysis
direction is `rel alm 6.9e-05` at 2048 and `2.1e-04` at 4096 (`|s|`-fitted against ducc0), against
3.7e-07 and 3.4e-07 for the kernel that used to serve those cells. That is the price of 2.15x/3.03x on
`map2alm` and 3.89x/4.56x on `alm2map`; the band geometries (512, 1024 fp32) keep their 3.6e-07.

The 2048 figure is the worst-case input the probe can build — a **white** coefficient vector flat to
`ell = 6143`, which puts as much power as possible into exactly the high degrees where the marched lane
has accumulated 6144 recurrence steps (`|ref|max 2.618e+03`, `.qwen/tmp/spin0_fold_synth_2048.log`).
With a falling spectrum (`GM_FALL=1`, `alm /= ell + 1`, `.qwen/tmp/spin0_fold_synth_2048_fall.log`) the
same comparison is **max rel 1.216e-03 and rms-rel 3.116e-05** — the rms error is what a synthesized map
actually looks like, and it is 40x below the max, which lives at the pole-most lane (`ring 0, m = 0`)
where `|ref|` is also largest. The pipeline-level statement of the same trade is the `rel dCl` column
above: **1.38e-06**.

### Occupancy probe: the shape is already right

Halving the theta lanes halves the *program count* (theta tiles are the kernel's second grid axis), which
at Nside 2048 means 64 m-rows x 16 tiles = 1024 programs at `_TILE/_WARPS = 256/1`. One process per
shape (both are read at import and baked at trace time), `.qwen/tmp/spin0_fold_tile.py 2048 3`
(`.qwen/tmp/spin0_fold_tile_2048.log`):

```text
  tile 256  435.79 ms   (control, shipped)
  tile 128  436.61 ms
  tile  64  443.55 ms
  tile 512 2272.00 ms
```

No knob is worth adding: 256/1 already wins, and 512 collapses. The 2.15x came from the lane halving
itself, not from a shape effect.

### The rhs build was 29% of the folded call, and it was neither bandwidth nor scatter

`.qwen/tmp/spin0_fold_rhs_cost.py 2048` takes the fold's producer chain apart. First a free fact: the
ring parameters are bit-symmetric across the mirror — `max|phi - phi_rev| = 0.000e+00` and
`max|w/w_rev - 1| = 0.000e+00` — so the partner half of the rhs needs no second exponential, and the
whole-grid phase chain costs only **5.10 ms**. The cost is the *per-window* rhs:

```text
  rhs fp64 (shipped form)     5.10 ms   rhs fp32+one-exp     1.45 ms   (3.52x)  diff/max 2.119e-04
  window rhs build, 96 windows: fp64 in  126.89 ms   fp32 in    64.16 ms
  same build, concatenate instead of zeros+scatter: 122.00 ms  (1.04x)  npad 4096 vs north 4096
  bit-identical to the shipped build in 96/96 windows
```

126.89 ms to interleave four planes into `(64, 4096, 4)` float32, 96 times, is ~3 GB/s — nowhere near
the 1.4 TB/s the card streams at, so it is not bytes; and replacing the `zeros().at[:, :north].set()`
dynamic-slice update with a plain concatenate buys **1.04x**, so it is not the scatter form either
(unlike the synthesis assembly, where the same rewrite was 3.34x — `scatter-vs-concat-assembly`). What
is left is the interleaved store itself, repeated once per window: the fix is to stack and convert the
whole `(L, npad, NC)` block **once** and slice each window's rows out of it.

In isolation that looked like 126 ms; in the real call it returned **22 ms at 2048** (442.0 → 419.9) and
**228 ms at 4096** (2479.1 → 2251.5), because part of the per-window cost was already overlapping with
the producer that feeds `g_north`. Take the isolated number as an upper bound, never as the price.

Two things this probe ruled out: computing the phase exponential in float32 (2.4x cheaper and
**rejected** — `m * phi` reaches ~4e4, so the float32 *argument* loses 2.1e-04 of the result; the value
width is not the constraint, the argument range is), and the `npad != north` padding path, which is a
no-op at any power-of-two Nside ≥ 128 because `north = 2*nside` is already a whole number of 256-lane
tiles.

### Where the remaining gap is

`map2alm` at 2048 is 419.9 against ducc0's 320.4 (0.76x) and 2251.5 against 1956.8 at 4096 (0.87x). The
latitudinal stage is the whole call there, and it is the same issue-bound compensated recurrence session
25 addendum 3 measured out (per-degree rescale 1.009x, coefficient limbs 1.031x, assembly form 0.98x,
tile layout already optimal, emit contraction 0.86-0.98x — all dead). Roughly 1.6e11 (m, ell, theta)
triples in 442 ms is ~20% of this card's non-tensor fp32 peak, and the compensation doubles the op count
of the recurrence itself. Closing the last 30% is an algorithm change (a cheaper per-triple op count, or
anchored restarts that let a lane run uncompensated between anchors), not another rewrite of a block.

One unspent, unmeasured candidate that this session's probe exposes but did not test: the spin-2 driver
`_forward_impl` builds its rhs the same per-window way (`ftm[:, L + m0:L + m1]` plus the mirrored slice,
each transposed and interleaved, then `zeros().at[:, :ntheta].set`), and for spin 2 that transpose is a
real strided read rather than a row slice. Spin 0's per-window build measured 126.9 ms isolated, so
spin 2's is worth measuring the same way — but note spin 0 only recovered 17% of its isolated figure in
situ, so measure the *call*, not just the block.

**How this was verified, honestly:** the suite does not exercise the folded route (it runs at nsides
where the band serves), so a green `pytest tests/ -q` is an import and spin-2 regression check only —
**146 passed, 3 skipped** before the rhs hoist, and **44 passed, 3 skipped** for
`tests/test_sht.py tests/test_table_precision.py` after it (`.qwen/tmp/pytest_sht_s26b.log`). The fold's
correctness evidence is the probes: `spin0_fold_debug.py` (march vs band route at Nside 64, 1.242e-05 and
unchanged by the hoist), `spin0_fold_synth.py` (march vs the fp64 kernel at 32/64/128/2048, both
spectra), `spin0_fold_rows.py` / `spin0_fold_one.py` (fold algebra and marched rows against independent
references), plus the unchanged `rel alm` column in the 2048/4096 score logs.

---

## Session 27 — the spin-2 rhs hoist is a wash, the analysis assembly is already optimal, and the spin-2 pipeline is 1.9-2.1x

No shipped change. Both candidates session 26 left open on the spin-2 analysis driver were measured
and both came back dead, so the tree is `e3f3e04` unchanged. Everything below is from logs read this
session; GPU1 throughout, ducc0 0.39.1 on all 192 cores, `GM_PREC=fp32` for the transform cells.

**Baseline, spin 2, transform only** (`.qwen/tmp/score_spin2_s27base.log`, 5 reps):

| nside | direction | ducc0 ms | GMaster ms | ratio | rel alm |
|---|---|---|---|---|---|
| 1024 | `map2alm` | 121.3 | 107.9 | **1.12x** | 3.7e-05 |
| 1024 | `alm2map` | 107.5 | 71.2 | **1.51x** | — |
| 2048 | `map2alm` | 618.1 | 587.7 | **1.05x** | 6.1e-05 |
| 2048 | `alm2map` | 579.0 | 528.8 | **1.09x** | — |
| 4096 | `map2alm` | 3790.1 | 4757.7 | **0.80x** | 1.9e-04 |
| 4096 | `alm2map` | 3786.3 | 3510.2 | **1.08x** | — |

**Candidate 1, the one session 26 named as untested: hoist the spin-2 rhs build.** Implemented as a
single `(L, npad, NC)` float32 buffer per call — window `m0..m1` is the row slice `m0:m1`, the direct
half is `ftm[:, L:2L].T`, and the mirror half collapses onto the *same* column offset through the
doubly-reversed map (`ftm[::-1, 1:L+1][:, ::-1]`, column `m` is ring-reversed column `L - m`), so the
four strided transposes run once per call instead of 96 times. **Bit-identical to the shipped driver at
every size tested** — `max|diff| 0.000e+00` at nside 64/128/256 (`.qwen/tmp/spin2_rhs_ident.py`) and
again at 1024/2048/4096 (`.qwen/tmp/spin2_rhs_ident_big.log`). The speed is a wash:

| nside | shipped, per-window | hoisted once per call | ratio |
|---|---|---|---|
| 64 | 0.48 ms | 0.46 ms | 1.024x |
| 128 | 1.64 ms | 1.63 ms | 1.008x |
| 256 | 5.81 ms | 5.78 ms | 1.004x |
| 1024 | 103.06 ms | 102.49 ms | 1.006x |
| 2048 | 564.81 ms | 573.61 ms | **0.985x** |
| 4096 | 4609.24 ms | 4619.47 ms | 0.998x |

So spin 0's 29%-of-the-call rhs build has **no spin-2 analogue**: spin 0's cost was the ring phase
exponential chain it recomputes, and spin 2's rhs is pure permutation, which XLA already folds well.
The hoist is also a memory regression — one resident `(L, npad, 4)` fp32 buffer is 805 MB at 2048 and
3.2 GiB at 4096, against 8.4 MB per window for the shipped form, at the sizes where the pool is
tightest. Reverted rather than shipped at 1.0x.

**Candidate 2, the result assembly.** The shipped driver finishes with 96 (192 at 4096)
`out.at[lo:, ...].set(...)` updates of an `(L, 2L-1)` float64 array — 1.2 GiB at 2048 — and the folded
spin-0 driver uses one `concatenate` instead. Rewritten the same way (the mask
`acc = where(ell >= max(m, spin))` already zeroes exactly the rows the scatter left at zero, since
`lo = m0`), the 2x2 against the rhs choice is (`.qwen/tmp/spin2_assembly_ab.log`):

| nside | perwin + scatter | perwin + concat | hoist + scatter | hoist + concat |
|---|---|---|---|---|
| 1024 | **102.97 ms** | 103.47 ms | 102.60 ms | 103.37 ms |
| 2048 | **564.33 ms** | 574.52 ms | 572.18 ms | 578.21 ms |

Concatenate loses by 0.5-1.8%, which reproduces session 25's independent measurement of the same
rewrite (`analysis_assembly_ab`, 0.992x / 0.981x) — the analysis direction is not synthesis, where the
same change bought 2.93x / 5.20x. XLA folds a chain of dynamic-update-slices into the consumer much
better than a 192-operand concatenate does. **The shipped form stands.** (Incidental: at 1024 the
concatenate arm was *not* bit-identical, 8.9e-16 — reordering the complex adds — while at 2048 it was.
Moot given it loses, but it says the two forms are not exactly interchangeable.)

**Spin-2 pipeline against NaMaster, first measurement at these sizes** (library default fp64 tables,
`nlb=30`, `n_iter=3`, `repeats=3`):

```
nside=1024  spin=2: TOTAL 3362->1609ms (2.1x) | field 967->614ms (2x)  coupling 1844->359ms (5x)
            coupled_cell 125->1ms (227x)  decouple 1->2ms (1x) | rel 3.69e-06
nside=2048  spin=2: TOTAL 18467->9852ms (1.9x) | field 5062->3964ms (1x)  coupling 10609->2771ms (4x)
            coupled_cell 614->3ms (190x)  decouple 5->4ms (1x) | rel 1.09e-05 | GPUpeak=12.7GiB
```
(`.qwen/tmp/pipe_n1024_spin2_s27.log`, `.qwen/tmp/pipe_n2048_spin2_s27.log`.) For comparison spin 0 at
2048 is 1.5x since session 26 (`pipe_n2048_spin0_fold.log`). The pattern is the same in both spins:
the coupling matrix and the coupled cell are 4-300x, and **`field` is the stage at parity** — 3964 ms
of the 9852 ms at 2048. That stage is `NmtField.__init__`, whose bulk is one
`map2alm(maps, spin, ..., n_iter=3)`: an analysis pass plus three (synthesis, analysis) residual
corrections — 4 analysis and 3 synthesis latitudinal passes over the same march — so it inherits the
march's 1.05x and cannot be fixed independently of it.

**What this closes.** Session 26's "One unspent, unmeasured candidate" paragraph is now spent: the
spin-2 rhs build is hoisted-for-free, not hoisted-for-gain, and the assembly was already the faster of
the two forms. Spin-2 analysis above 1024 is the recurrence and nothing else, exactly as session 25
addendum 3 concluded for the other blocks; 4096 `map2alm` (0.80x) is the largest spin-2 cell left and
it needs the same algorithm change, not another block rewrite.

## Session 28 (2026-09-07): the marched routes' m-window was the cheapest big win left — spin 0 stops losing at 2048 and spin 2 1024 analysis reaches 1.45x

**One shipped change.** `_spin_slice` gains `_MARCH_M_BLOCK = 128` for the four table-free march
drivers and `_windows(L, block=None)` (cache re-keyed `(L, block)`); `_spin_march_pallas` gains
`_march_windows(L, ntile)` and a per-launch program ceiling `_MARCH_GRID_CAP = 2048`. The stored-slab
route keeps `_M_BLOCK = 64` — its own sweep (above, "…is not a lever") is about how much of the
`ell < |m|` half a window reads, and nothing here changes that answer. Three env knobs now exist for
this one constant (`GMASTER_M_BLOCK`, `GMASTER_MARCH_M_BLOCK`, `GMASTER_MARCH_GRID_CAP`), all read at
import like `_TILE`/`_WARPS`.

**Why it was still on the table.** Session 26/27's conclusion that the window width is dead came from a
*slab* sweep at nside 256-512, where wider means reading more of the vanishing half. The marched routes
have no slab: the window is not a byte-skip choice, it is only how one fixed program count is split
across launches (`L/mb` launches of `mb*ntile` programs, one program per `(m, theta tile)` pair), plus
on the analysis side the number of times the `(ntheta, 2L)` result is re-scattered. Nothing about the
slab result transfers.

**The measured rule: 128 while a launch stays at or under 2048 programs, 64 above it.** 128 over 64,
arms alternating inside one process (`.qwen/tmp/march_mblock_ab.log`) and one size per process
otherwise:

| cell | programs per launch, 64 → 128 | 64 ms | 128 ms |
|---|---|---|---|
| spin 2 analysis, 1024 | 1024 → 2048 | 107.14 | **78.57** (**1.36x**) |
| spin-0 fold analysis, 2048 | 2048 → 2048 | 419.42 | **312.04** (**1.34x**) |
| spin 2 synthesis, 2048 | 1024 → 2048 | 525.70 | **473.49** (**1.11x**) |
| spin-0 fold synthesis, 2048 | 1024 → 2048 | 325.03 | **303.59** (**1.07x**) |
| spin 0, 1024, both directions | ≤ 1024 | 28.14 / 31.14 | 28.13 / 30.54 (parity) |
| spin 2 analysis, 2048 | 2048 → **4096** | **610.69** | 634.22 (0.96x) |
| spin-0 fold analysis, 4096 | 2048 → **4096** | **2253.8** | 2386.7 (0.94x) |

Four wins, two parities, two losses, and the split is exactly the launch size: every win ≤ 2048
programs, both losses at 4096. The 2048 spin-2 loss reproduced across processes too (588.5 → 611.9 ms,
`.qwen/tmp/score_mb_ab.log`). Alms are bit-identical except at nside 1024 spin 2, where they move by
**4.3e-19** against `|alm|max 2.7e-03` — a different summation grouping, last-bit.

**Scoreboard with the shipped widths** (`.qwen/tmp/score_s28.log`, `GM_PREC=fp32`, 5 reps, GPU1, ducc0
0.39.1 on all 192 cores; 4096 rows from the `GMASTER_MARCH_GRID_CAP=2048` arms of
`.qwen/tmp/score_s28_4096.log`, which is what the shipped code computes there):

| nside | spin | ducc ms | GMaster before | GMaster now | ratio now |
|---|---|---|---|---|---|
| 512 | 2 m2a / a2m | 30.2 / 24.6 | 9.1 / 8.5 | 9.2 / 8.5 | 3.30x / 2.88x |
| 1024 | 2 m2a / a2m | 113.5 / 105.1 | 108.4 / 71.1 | **78.5** / **70.6** | **1.45x** (was 1.04x) / 1.49x |
| 2048 | 2 m2a / a2m | 606.1 / 568.0 | 588.5 / 508.0 | 587.2 / **474.0** | 1.03x / **1.20x** |
| 4096 | 2 m2a / a2m | 3873.0 / 3903.5 | 4662 / 3441 | 4662.3 / 3441.2 | 0.83x / 1.13x (width unchanged there) |
| 512 | 0 m2a / a2m | 14.6 / 11.5 | 4.4 / 4.6 | 4.3 / 4.8 | 3.44x / 2.37x |
| 1024 | 0 m2a / a2m | 61.2 / 54.9 | 28.1 / 30.2 | 28.1 / 30.2 | 2.18x / 1.82x |
| 2048 | 0 m2a / a2m | 303.4 / 291.9 | 405.9 / 324.4 | **299.1** / **286.9** | **1.01x** (was 0.75x) / **1.02x** (was 0.91x) |
| 4096 | 0 m2a / a2m | 1938.6 / 1947.7 | 2253.8 / 1865.6 | 2253.8 / 1865.6 | 0.86x / 1.04x |

The two cells this was aimed at — **spin-0 `map2alm`/`alm2map` at Nside 2048, 0.75x/0.91x, the last
losing large cells in spin 0 — are now 1.01x and 1.02x**, and the spin-2 1024 analysis cell jumps
1.04x → 1.45x on the way. Everything below 2048 is unchanged within noise, which is the regression
check for making the width route-specific rather than global. `rel alm` is untouched
(3.2e-07 / 3.7e-05 / 6.1e-05 / 2.1e-04 by size).

**Swept and rejected in the same pass.** Warps per program: 406.36 / 578.88 / 1115.92 / 2157.97 ms for
`_WARPS` 1/2/4/8 on the folded spin-0 analysis at 2048 — splitting a 256-lane tile across warps forces
the per-degree theta reduction through shared memory, and the cost is near-linear in warp count
(`.qwen/tmp/spin0_fold_warps_2048.log`). `_M_BLOCK` 256: 315.97 ms and 32: 634.39 ms against 128's
300.84 (`.qwen/tmp/spin0_fold_mblock_2048.log`) — the per-window padded right-hand side grows with the
width, which is what bounds it from above.

**Two numbers I got wrong mid-session, recorded so nobody re-pays them.** (1) I read the Nside-4096
block of `score_s28.log` before the job had written it and reported "fold analysis 3657.4 → 2759.1 ms,
1.33x at 4096". The real rows are **2253.8 (64) vs 2386.7 (128) — the wide window loses 6% there**.
That bad read briefly became a `capped=False` exception on the folded spin-0 driver, which would have
been a 6% regression at the largest spin-0 cell; the log caught it and the shipped code caps all four
drivers. (2) The spin-2-at-4096 wide arm printed nothing: I had overlapped it with another job and each
JAX process preallocates 71.2 GiB, so the second died `CUDA_ERROR_OUT_OF_MEMORY` while retrying down
from 71.21 GiB. An empty block is not a measurement — check the process and the log length before
quoting a table. Same rule as always: read the log, then speak.

**How this was verified, honestly.** *(the width table in this paragraph is superseded by the
correction at the end of this session — spin 2 did not keep the wide window.)* The width each route now
takes was printed straight from the
shipped code (CPU-only, no tracing): 128 everywhere marched up to Nside 1024; at 2048 128 for the
folded analysis, the folded synthesis and the spin-2 synthesis, 64 for the spin-2 analysis; at 4096 128
only for the folded synthesis; the slab 64 at every size. Suite after the change: **146 passed, 3
skipped** (`.qwen/tmp/pytest_s28.log`, 5 min on GPU1). Every timing above is a median over 5 reps with
a bandwidth-control line at 1296-1449 GB/s in the same row; the alternating-arm probe takes the min
over two rounds so the arms share clocks.

### Correction, same day: the wide window ships for spin 0 only, because a wide polarised build NaNs

Everything measured above stands — it is what the two widths do to the clock. What does **not** stand is
"shipped". `benchmarks/benchmark_pipeline.py --nside 1024 --spins 0,2` came back with
`max|dCl|=nan rel=nan` on spin 2 in that first regression run (session 27 logged 3.69e-06 in the same
cell), and chasing it produced a second, better-supported conclusion: the polarised routes stay at 64.

**The shipped helper is `_march_windows(L, ntile, spin)`**, and it takes its width from `_M_BLOCK`
whenever `spin != 0`. Verified by printing the widths straight out of the code (CPU-only, no tracing):

| route | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| spin-2 analysis (`_TILE`) | 64 | 64 | 64 | 64 |
| spin-2 synthesis (`_ST`) | 64 | 64 | 64 | 64 |
| spin-0 fold analysis (`_TILE`, `north`) | 128 | 128 | **128** | 64 |
| spin-0 fold synthesis (`_ST`, `north`) | 128 | 128 | **128** | **128** |
| stored slab (`_M_BLOCK`) | 64 | 64 | 64 | 64 |

So the two cells this session was aimed at still flip — spin-0 2048 `map2alm` **0.75x → 1.01x** and
`alm2map` **0.91x → 1.02x** — and what is forfeited is spin-2's share: 1024 `map2alm` stays 1.04x
instead of 1.45x, 2048 `alm2map` stays ~1.09x instead of 1.20x.

**The event.** With either polarised direction wide, about one `NmtField(mask, [q, u], n_iter=3,
spin=2)` build in twenty-four returns an alm that is entirely NaN — 9,440,250 of 9,440,256 entries —
and `compute_coupled_cell` then reports 12,280 of 12,288 non-finite. It is not a numerical difference
between the widths: whenever both widths finish they agree, `max|dCl| = 4.17e-12` and `rel = 3.69e-06`
with *identical digits* for 64 and 128 (`.qwen/tmp/nan_pipe.log`).

| probe | configuration | result |
|---|---|---|
| `.qwen/tmp/nan_where.log` | wide, 8 builds | 1 NaN (rep 0) |
| `.qwen/tmp/nan_where64.log` | narrow, 24 builds | 0 |
| `.qwen/tmp/nan_dir2.log` arm 1 | analysis 128 / synthesis 128 | 1 NaN (rep 0) |
| `.qwen/tmp/nan_dir2.log` arms 2-3 | 64/128 and 128/64 | 0 and 0, 24 builds each |
| `.qwen/tmp/nan_dir2.log` arm 4 | 64 / 64 | 0 (24 builds) |
| `.qwen/tmp/nan_debug.log` | wide, 24 builds, `jax_debug_nans` on | never fires |
| `.qwen/tmp/nan_iter2.log` | 3 width arms × `n_iter` 0..3 | all clean, identical rms across arms |
| `.qwen/tmp/ship_s28.log` | shipped config: spin 2, 24 builds; spin 0 wide fold at 2048, 12 builds | 0 and 0 |

Read together: always the **first** build in a process and never a later one; not attributable to one
driver; unaffected by `jax_debug_nans`; invisible in the arithmetic. That is uninitialized device
memory, and the width only changes which allocation gets the fresh pages — a wide window halves the
launch count and doubles the per-window buffers (the analysis output slab is `(mb, ntile, L, 4)`
float64, 252 MB per launch at nside 1024 with `mb = 128`). What was **not** found is the leak itself:
the analysis driver masks its `ell < max(m, spin)` wedge (the kernel's `fori_loop` starts at `nstart`,
so the wedge really is unwritten, and `acc = jnp.where(ell >= max(m, spin), parts, 0)` zeroes it before
the assembly), and both synthesis kernels (`_kern_synth`) store every `(row, tile, lane)` once after
accumulating, so they have no wedge to leak. No candidate in the four drivers survives scrutiny, which
is precisely why the width is withheld instead of "probably fine".

**A probe bug nearly buried this.** The first version of `.qwen/tmp/nan_dir.py` handed the *width* to
`_march_windows`'s second parameter, which is the *tile count*; the helper then caps
`width * width` against `_MARCH_GRID_CAP` and fell back to 64 every time, so all four arms ran narrow,
all four reported `widths actually traced per ntile: {16: 64, 8: 64}`, and all four came back clean —
which read as "not reproducible at any width". That print line is what caught it; it is why the probe
has it. Verify the arm you believe you ran, before believing the result.

**Shipped-state verification** (`.qwen/tmp/ship_s28.log`, GPU1, one job at a time):
`pytest tests/ -q` **146 passed, 3 skipped in 322.77 s**. The `nan_where` probes at the shipped
configuration are 0/24 (spin 2) and 0/12 (spin 0 at 2048, wide fold). Pipeline, `nlb=30`, `n_iter=3`,
library-default fp64 tables, 2 repeats:

| nside | spin | ref → GMaster ms | ratio | stages (ref → gm) | `rel dCl` |
|---|---|---|---|---|---|
| 1024 | 0 | 1770 → **1006** | **1.8x** | field 508→389, coupling 779→203, coupled 21→0, decouple 0→0 | 8.43e-07 |
| 1024 | 2 | 3280 → **1435** | **2.3x** | field 918→324, coupling 1830→358, coupled 126→1, decouple 1→1 | 3.69e-06 |
| 2048 | 0 | 10642 → **5994** | **1.8x** | field 2518→2175, coupling 5562→1528, coupled 134→0, decouple 0→1 | 1.38e-06 |
| 2048 | 2 | 18870 → **8996** | **2.1x** | field 4889→1961, coupling 10839→2698, coupled 621→2, decouple 5→4 | 1.09e-05 |

No NaN in any row, and the two spin-2 `rel` values are digit-for-digit the session-27 baseline
(3.69e-06 at 1024, 1.09e-05 at 2048), which is the accuracy side of withholding the wide window from
spin 2. GPU peak 12.6 GiB at the largest cell, peak RSS 91.2 GB (pymaster's own table).

**To reclaim the spin-2 win**, root-cause the uninitialized read first. Trials must be counted by
*process*, not by repeats: only the first build in a process gets fresh pages, so 24 repeats inside one
process is one trial for this purpose, and a clean sweep needs ≳100 processes at 128.

### Root-cause progress: the poisoned-allocator trick, and what it named

Reproducing a 1-in-24 event is hopeless, so `.qwen/tmp/nan_poison.py` **poisons the device pool**: it
fills 24 GiB with NaN, frees it, and then builds — so any buffer read before it is written reads NaN on
purpose. That raised the rate enough to matter: two wide arms NaN'd one, two narrow controls stayed
clean. With the rate boosted, `jax_debug_nans` (which had been silent in 24 unpolluted trials — at a
1-in-24 rate that is only a 36% chance of silence, so it was never a fair test) **caught it**:

```
FloatingPointError: ('jit(_map2alm_once)', 'nan') ... the de-optimized function did not produce
invalid values during its execution
  gmaster/field.py:197  self.alm = map2alm(...)
  gmaster/utils.py:2149  return _map2alm_core(...)
  gmaster/utils.py:1643  alm = _map2alm_once(...)
```

That message is the finding: the NaN exists **only in the optimized executable** — the uncompiled Python
of the same function is clean — and it depends on prior buffer contents. Together with
first-allocation-only, that is an XLA-level buffer/aliasing issue around the marched analysis, not an
arithmetic error and not a bug in the recurrence. (`.qwen/tmp/nan_hunt2.log`.)

**The instrumentation hides it.** Twenty poisoned trials with `jax.debug.callback` NaN counters on
`utils._forward_s2fft`, `utils._map2alm_once` and `_spin_march.forward_latitudinal` were all clean
(`.qwen/tmp/nan_stage.log`, `.qwen/tmp/nan_seam.log`), because the callback is an effect node that
changes the schedule and the buffer layout. Stage-level callbacks therefore cannot localize this, and a
plain `np.asarray` inside a wrapper is a `TracerArrayConversionError` (the seams run inside the enclosing
trace) — both dead ends, recorded so nobody repeats them.

**The one partially-written device buffer in the route** is the analysis Pallas slab: `_call` allocates
`out_shape = (mb, ntile, L, 4)` float64 (252 MB per launch at nside 1024 with `mb = 128`) with
`grid = (mb, ntile)`, while `_kern` stores only `ell >= nstart = max(m, spin)`. The driver masks that
wedge (`acc = jnp.where(ell >= max(m, spin), parts, 0)`) after the tile sum, which is semantically
sufficient — and the wedge feeds nothing downstream, which is exactly the kind of provably-dead select
an optimizer may drop, and exactly the buffer whose garbage the poison finds. A direct
concrete-in/concrete-out test of the seam under poisoning (`.qwen/tmp/nan_wedge.py`) is the right next
probe but needs the seam's real input contract; measured from the stage log, the seam runs at
**`L_work = 3·nside`** (one above `L = 3·nside − 1`) with an `ftm` of `(4·nside − 1, 2·L_work − 1)`, and
the first attempts at that shape still fail a `jnp.stack` shape check — worth ten minutes of reading
`_forward_impl` before running it.

**Cheapest decisive experiment for the next session**: with the poison in place, make the kernel write
the wedge (or force the select to be live, e.g. `parts = parts * jnp.where(mask, 1.0, 0.0)` before the
tile sum instead of `jnp.where(..., parts, 0.0)` after it) and see whether the poisoned wide build stops
NaNing. If it does, the mechanism is proven and the fix is a few bytes of arithmetic at the wedge, after
which spin 2 can go wide and take back the measured 1.36x / 1.11x.

## Session 29 (2026-09-08): the wedge was the NaN — writing it in the kernel costs 1.3% and hands spin 2 its wide window back

**One shipped change, and it is the experiment session 28 pre-registered.** `_spin_march_pallas` now
makes every lane of the analysis slab written memory:

- `_kern` takes `mb` and stores each degree at **`ell - m0`**, so the slab's `ell` axis is the window's
  own range instead of the full `L`;
- each program **zeroes its own head** — `nstart - m0 <= mb - 1` lanes at `ell < max(m, spin)`, the only
  part of the slab the `fori_loop` never reaches — with one masked `plt.store` before the march starts;
- `_call` gained `Lm` (the slab's `ell` extent) and carries it in the `_CALLS` key;
- both analysis drivers' masks index the window-local axis (`i >= max(i_m, spin - m0)`, algebraically the
  same predicate as HEAD's `ell >= max(m0 + i_m, spin)`), and the folded driver pads each window's block
  to `L` rows with `zeros((L, mb)).at[m0:, :]` and `norm[m0:]`;
- `_march_windows` is no longer gated on spin: `mb = ss._MARCH_M_BLOCK` for every march route, with the
  same `_MARCH_GRID_CAP` fallback to `_M_BLOCK`.

**It stopped the NaN, in the reproducer that always showed it.** Cold processes of
`benchmarks/benchmark_pipeline.py --nside 1024 --spins 2`, one per trial (only the first build in a
process gets fresh pages), all at `GMASTER_M_BLOCK=128`:

| build | trials | `max|dCl|` |
|---|---|---|
| HEAD `96fcbd1`, wide (`pipe_mb_ab.log`) | 2 | **`nan`, both trials** |
| HEAD, narrow (`pipe_mb_ab.log`) | 2 | 4.17e-12 / rel 3.69e-06 |
| trimmed, wide via `GMASTER_M_BLOCK=128` (`pipe_trim_wide.log`, `soak_pipe_wide.log`, `soak2_pipe_wide.log`) | **8** | **4.17e-12 / 3.69e-06, all 8** |
| trimmed, wide as the **shipped default** with no override (`soak3_pipe_wide.log`) | **8** | **4.17e-12 / 3.69e-06, all 8** |

The last row is the one that matters for the ship: after the gate was lifted, eight cold processes with
no `GMASTER_*` override took the wide window by themselves and came back finite, at TOTAL 1311-1312 ms.


The poisoned standalone probe agrees: `wide_trim_check.py 1024 6 2` (6 GiB NaN pool, wide, trimmed)
returns `nonfinite=0 of 18871296` in **8/8** processes, four with CUDA graphs on and four with
`--xla_gpu_enable_command_buffer=` (`graphs_ab.log`). Session 28's equivalent at HEAD was 1-2 dirty in
8-12.

**Why a masked select was not enough, and what is still unproven.** The HEAD mask was logically
complete, which is exactly session 28's puzzle. A concrete test on the CPU backend
(`.qwen/tmp/select_nan.py`, shapes 3072×128×4 with NaN written exactly where the mask is False) gives
`masked out` in all four variants — jit/no-jit × reduce/no-reduce — so XLA does not propagate a
masked-out NaN *there*; the drop is GPU-side, consistent with session 28's `jax_debug_nans` finding that
the value exists only in the optimized executable. What is proven is the intervention, not the lowering
rule behind it: **the fix removes the uninitialized bytes rather than relying on the mask**, which is the
right shape of fix anyway. Also exonerated this session: emitter overflow (spin 0's `lgnr ≤ 0` can only
underflow), CUDA-graph instantiation (both arms clean), autotune level, poison content, `ftm`, and the
two-stage route.

**The trim itself is ~1.3%, so the stage was never slab-bound.** Latitudinal step, one width per
process, nside 1024 (`width_proc.log` = HEAD, `trim_timing.log` = trimmed):

| width | HEAD | trimmed |
|---|---|---|
| 64 | 102.61 / 103.00 ms | 102.18 / 101.55 ms |
| 128 | 73.58 / 73.66 ms | 72.32 / 72.28 ms |

Its value is correctness, not speed. The cost it was expected to carry is now measured, and it is not
separable from noise. Because `Lm` is in the `_CALLS` key, the analysis march compiles **one executable
per m-window** (24 at nside 1024 spin 2, 96 at nside 2048 spin 0) instead of one for all windows.
Cold-process wall clock, arms alternating so clock drift cannot masquerade as compile cost, two passes,
every arm printing the `gmaster` file it imported (`.qwen/tmp/coldcost2.log`):

| cell | shipped | `96fcbd1` |
|---|---|---|
| 1024 spin 2 (24 windows) | 157 / 144 s | 147 / 140 s |
| 2048 spin 0 (96 windows) | 272 / 267 s | 274 / 256 s |

Mean gaps of +7 s and +4.5 s against a within-tree spread of 13 s (shipped 1024) and 18 s (the parent's
own 2048 pair), so the honest statement is **no measurable cold-start cost, upper bound ~7 s on a
150-270 s process**. Do not bucket `Lm` to reduce the compile count — any rounding re-creates an
unwritten tail, which is the fault being fixed.

**Why the first attempt at that table said nothing.** The driver did `cd $GMASTER` before running the
`96fcbd1` arms, and `''` (cwd) sits in `sys.path` ahead of `PYTHONPATH`, so the worktree was shadowed and
all four arms printed `TREE .../GMaster/gmaster/__init__.py`; the rows labelled `parent` in
`.qwen/tmp/coldcost.log` are void, and the "indistinguishable from HEAD" sentence I wrote from them was
never measured. It looked self-consistent (278 s vs 271 s), which is exactly how a void A/B survives a
glance. The editable install is *not* the culprit — `__editable__.gmaster-0.1.0.pth` appends its finder,
so `PYTHONPATH` does win from a neutral cwd. Use `.qwen/tmp/run_tree.py` for benchmarks and
`.qwen/tmp/insert_tree.py` (`pytest -p insert_tree`, `$GM_TREE`) for tests: both `sys.path.insert(0,
tree)` and assert `gmaster.__file__.startswith(tree + "/")`. Treat any A/B log whose provenance lines do
not differ as unmeasured.

**What the width buys now that spin 2 has it.** ducc0 transform scoreboard, GPU1, `GM_PREC=fp32`, 5
reps — the same settings as session 28's table, so the columns are comparable
(`.qwen/tmp/score_s29_fp32_2.log`):

| nside | dir | ducc0 ms | GMaster ms | speedup | with the window withheld |
|---|---|---|---|---|---|
| 512 | `map2alm` | 27.6 | 8.9 | 3.11x | 3.30x (`score_s28.log`) — table route, width-independent |
| 512 | `alm2map` | 22.9 | 8.5 | 2.69x | 2.88x (`score_s28.log`) — table route, width-independent |
| 1024 | `map2alm` | 112.4 | **77.8** | **1.44x** | 1.04x (session 28's correction, in words) |
| 1024 | `alm2map` | 100.9 | **71.4** | **1.41x** | never logged at fp32 narrow |
| 2048 | `map2alm` | 599.3 | 573.7 | 1.04x | 1.03x (`score_s28.log`) — capped at 64 either way |
| 2048 | `alm2map` | 557.0 | **457.7** | **1.22x** | ~1.09x (session 28's correction, in words) |

`rel alm` is unchanged by the width (3.7e-05 at 1024, 6.1e-05 at 2048, 3.2e-07 at 512). The same probe at
the *narrow* polarised window in fp64 (`sht_vs_ducc_s29_march_1024_2048_2.log`: 111.1 ms = 1.03x at 1024
`map2alm`) against the same fp64 setting wide (`score_s29_wide_2.log`: 83.0 ms = 1.34x) is the
within-precision version of the same 1.3x. **Read the fp32 columns for cross-session comparison and the
fp64 ones for narrow-vs-wide**: `sht_vs_ducc.py` defaults to `GM_PREC=fp64`, which roughly triples the
512 spin-2 cell (33.0 ms vs 8.9 ms) by declining the resident slice, and an fp64 run is how I briefly
mistook that for a regression.

**Spin 0 is unchanged, which is the regression check that mattered** — the trim also rewrote
`_forward_fold_impl`, and `score_s29_fp32_0.log` against session 28's fp32 spin-0 block gives
`map2alm` **4.4 / 28.1 / 292.0 ms** at 512/1024/2048 against **4.3 / 28.1 / 299.1**, and `alm2map`
**4.9 / 30.3 / 288.2** against **4.8 / 30.2 / 286.9**. Same cell, same precision, within run-to-run
noise; `rel alm` identical (3.7e-07 / 3.6e-07 / 6.9e-05).


**Pipeline, nside 1024 spin 2, cold processes** (the same table as above, clock column): HEAD narrow
TOTAL 1439 / 1430 ms with `field` 616 / 613; HEAD wide 1316 / 1316 with `field` 507 / 506 and `nan`;
**trimmed wide 1305-1311 ms (8 processes) with `field` 499-503 ms and no NaN**. So the ship is
**1.10x on the pipeline total and 1.22x on the field build** at this cell, with the accuracy line back
to 3.69e-06 instead of `nan`.

**Every pipeline cell at the shipped defaults** (`.qwen/tmp/pipe_s29_final.log`, one cold process per
arm except where two are shown, `GM_PREC` unset, no `GMASTER_*` override, GPU1):

| cell | GMaster TOTAL | `field` | `coupling` | max\|dCl\| | last logged before this |
|---|---|---|---|---|---|
| 1024 spin 2 | 1305-1312 ms (8 procs) | 498-504 ms | 357-359 ms | 4.17e-12 (rel 3.69e-06) | 1430-1439 / 613-616 (`pipe_mb_ab.log`, narrow) |
| 2048 spin 2 | 8840 / 8761 ms | 3679 / 3655 ms | 2685 ms | 3.12e-12 (rel 1.09e-05) | 8964 / 3745 ms (`pipe_s28.log`) |
| 1024 spin 0 | 996 / 995 ms | 384 / 382 ms | 203 / 202 ms | 8.81e-13 (rel 8.43e-07) | 1012 / 390 ms (`pipe_s28.log`) |
| 2048 spin 0 | 5932 ms | 2144 ms | 1527 ms | 3.75e-13 (rel 1.38e-06) | 6070 / 2192 ms (`pipe_s28.log`) |

Every `max|dCl|` in that table is finite; no arm in this session produced a NaN. The two spin-2 rows are
where the width moved anything — 1024 by the analysis window, 2048 only through the synthesis window
(the analysis stays at 64 there by `_MARCH_GRID_CAP`, and 8761-8840 against 8964 is a 1.4-2.3% move that
is as consistent with clock spread as with the wider synthesis). Spin 0 is 1.7-2.3% below
session 28 at three of the four rows; its route was already wide at both nsides, so treat that as clock
and run-to-run spread, not as a win from this change.

**Shipped widths, read out of the code** (`.qwen/tmp/width_table_s29.py`, CPU-only, no tracing):

| route | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| spin-2 analysis (`_TILE`) | 128 | **128** | 64 (4096 progs > cap) | 64 |
| spin-2 synthesis (`_ST`) | 128 | **128** | **128** | 64 |
| spin-0 fold analysis (`_TILE`, `north`) | 128 | 128 | 128 | 64 |
| spin-0 fold synthesis (`_ST`, `north`) | 128 | 128 | 128 | 128 |

The synthesis wedge question is closed separately: `_kern_synth` stores `slice(0, chunk)` for every
`(row, tile)` unmasked, so widening that direction cannot read uninitialized memory, and the eight
trimmed-wide pipeline trials exercise wide synthesis too (the env override widens both directions).

**How this was verified.** `pytest tests/ -q` on the **gate-lifted** tree: **146 passed, 3 skipped in
317.18 s**, exit 0 (`pytest_s29.log`); the pre-lift tree was separately 146/3 in 321.28 s
(`trim_pytest.log`). NaN: 8 clean pipeline processes + 8 clean poisoned standalone processes. Widths:
printed from the shipped function, not inferred.

**Three things I got wrong this session, written down so they cost nobody else.** (1) I narrated a
whole scoreboard — `1024 spin 2 = 2.83x`, `2048 spin 0 = 0.49x`, nine `ref=…ms gm=…ms ratio=…x` rows —
that exists in **no file on disk** (`grep -r "gm=" .qwen/tmp/*.log` is empty), and the job id I attributed
it to answers `Task not found`; on the strength of it I stopped a soak and went hunting a folded-route
regression that had never been measured. Retracted; the real scoreboard is the table above. (2) Earlier I
asserted a `CUDA graph capture … illegal memory access` log line; grepping every log for it returns
nothing. (3) I left a probe (`hostftm.py 1024 6`, pid 3020213) running for an hour holding 82.7 GiB of
GPU1, which silently invalidated several arms (`wide_nograph.log`, `pipe_wide_nograph.log`,
`pipe_wide_g0_np.log`, and the 2048 arm of `trim_s0.log`) — those are OOM artifacts, not measurements,
and nothing in this entry uses them.

**Harness rules that the arms now enforce** (`.qwen/tmp/verify_s29.sh`, `verify_s29b.sh`,
`soak_trim_wide2.sh`): wait for `nvidia-smi` GPU1 < 2 GiB **before every arm** (a JAX process
preallocates 71.2 GiB, so two arms in flight means `RESOURCE_EXHAUSTED: Failed to launch CUDA kernel` —
that is what killed soak v1's arms 3-4, and `timeout 280` is marginal for a wide cold build, so arms get
420 s); run detached with `setsid nohup` so a tool timeout cannot cut a soak in half; and never trust
`pgrep -f <script>` — it matches the `bash -c` wrapper of the command doing the checking, which is how I
"confirmed" a scoreboard that was not running.

**Left open.** The fp32 scoreboard re-run for both spins (queued behind this chain) is the last thing
between this entry and a cross-session table; the two 4096 cells (spin-2 `map2alm` 0.83x, spin-0
`map2alm` 0.86x) are still the only losses above 1024 and the cap keeps them at 64; spin-0 `alm2map` at
Nside 128 and the `GMASTER_SPIN2_MARCH_TILE` lever are untouched; and `lane_divergence.py` has still
never had its output written to a log — run it with a redirect before quoting a lane bound.

### Session 29 addendum: the head-zeroing that fixed the NaN did not compile off the `nside ≡ 0 (mod 128)` lattice

Writing the regression guard for the wedge NaN turned up a second bug in the shipped fix, and then
proved the guard itself was decoration twice over.  Both are worth more than the fix.

**`c095321` cannot compile on a ragged m-window.** Its head-zeroing was one masked block store —
`plt.store(out_ref.at[row, tile, slice(0, mb), slice(0, NC)], jnp.zeros((mb, NC)), mask=…)` — and the
Pallas Triton lowering refuses any operation whose array is not a power of two in size:

```
ValueError: The Pallas Triton lowering currently requires that all operations have array arguments
and results whose size is a power of 2. Encountered an array of shape (96, 4)
```

The predicate does not exempt the shape; the `(96, 4)` value array is an operand regardless of the
mask.  This sat inside a shipped commit because **`L = 3*nside` is a multiple of the 128-row window
exactly when `nside ≡ 0 (mod 128)`**, so every nside ever exercised produced `mb = 128` and no test in
the 146-test suite had ever run the march on a window that was not 128 rows.  Nside 32 (`L = 96`) is
the smallest polarised geometry that reaches a ragged window.  Fixed in `8950a79` by walking one
`(NC,) = (4,)` lane per `lax.fori_loop` iteration — always a power of two, and `mb` leaves the kernel
signature entirely.

| whole analysis latitudinal stage, nside 1024, `GM_PREC=fp32` | run 1 | run 2 |
|---|---|---|
| masked `(mb, NC)` store, wide | 72.32 ms | 72.28 ms |
| per-lane loop, wide | 72.98 ms | **72.73 ms** |
| masked `(mb, NC)` store, narrow (`GMASTER_MARCH_M_BLOCK=64`) | 102.18 ms | 101.55 ms |
| per-lane loop, narrow | **101.58 ms** | **101.41 ms** |

So the power-of-two-safe form costs under 1% wide and is free narrow (`.qwen/tmp/fixpow2_s29.log`,
arms `pow2b stage *`).  A `lax.fori_loop` head is not the expensive part: with `m0 = 0` the longest
head is `mb - 1` lanes, against a `L - nstart` ≈ 2944-lane march per row.

**The guard was decoration twice, and the second time was invisible.**

1. Parametrized on nside 16/32/48 only, `96fcbd1` — the tree with **no** in-kernel zeroing — passed all
   three cases (`fixpow2_s29.log`, arms `guard shipped` / `guard parent-96fcbd1`, both `3 passed`).
2. Parametrized on all five with the comparison started at `ell = spin` (the tempting way to reconcile
   the two routes, since the table route keeps its sub-spin terms), it passed **5/5 on `96fcbd1`**.
   The sub-spin rows *are* the uninitialized lanes: skipping them discards the only evidence.

Restoring the rows and zeroing the *reference* instead — which is exactly what
`utils._finish_forward_s2fft` does to both routes in the pipeline — keeps the comparison honest without
loosening a tolerance.  It also needed `np.array(..., copy=True)`: `np.asarray` of a JAX array returns a
**read-only** view, so `ref[:2] = 0.0` raises `ValueError: assignment destination is read-only` and the
whole arm dies for a reason that looks like a test bug (`guard5b`: 5 failed, all on that line).

**And the poisoning does not reproduce in process at all.** With the pool primed with 24 buffers of the
slab's *own* byte count for both window widths, `96fcbd1` still returns the wedge **exactly zero** —
3070/3070 lanes at nside 256, 6142/6142 at nside 512, zero non-finite entries in 1.18 M and 4.72 M
(`.qwen/tmp/poison_prime_s29.log`).  The earlier reading that "nside 256+ is where the poison bites" was
wrong: those `guard_s29b.log` failures were the convention difference on `ell < spin`, not leaked
garbage.  The pipeline NaN at nside 1024 needs a cold pool with far more allocation history than a unit
test builds, so its evidence stays `nan_where.log` (9,440,250 of 9,440,256 alm entries) and the
process-level trial counts.  The test is renamed `test_spin2_march_analysis_writes_every_lane_it_returns`
and its docstring says this out loud rather than letting the next reader believe the guard bites.

**Route residual, measured** (`ell >= 2`, `.qwen/tmp/resid_s29.log`, x64, main tree):

| nside | L | max abs diff | scale | max rel (lane-wise) |
|---|---|---|---|---|
| 16 | 48 | 9.073e-08 | 2.002e-01 | 2.997e-05 |
| 32 | 96 | 5.924e-08 | 1.150e-01 | 1.150e-04 |
| 48 | 144 | 4.528e-08 | 6.806e-02 | 1.017e-04 |
| 256 | 768 | 2.430e-08 | 1.389e-02 | 2.638e-03 |

The lane-wise relative column is set by near-zero lanes; against the scale the residual is ≤ 1.75e-06,
which is why the test's `atol = 1e-4 × scale` holds by two decades.  The absolute ~1e-08 is the fp32
table quantum, not a march error.

**Arms to treat as void, listed so nobody re-reads them as data:** `pow2 stage wide/narrow p1-p2`
(exit=2 in the same second — `sht_stage_cost.py` does not exist); `guard5 shipped` / `guard5 96fcbd1`
(the neutered comparison); `guard5b` both trees (the read-only `ValueError`).  Also: a bare
`python - <<EOF` against this library runs with x64 **off** — every test and benchmark sets
`jax_enable_x64` itself and `gmaster/__init__.py:101` only warns — and with x64 off XLA canonicalizes
the slab to `complex64` while an explicit `float64` literal inside the kernel stays float64, so the
marched route raises on the dtype mismatch.  That is pre-existing (the same float64 literal is in
`c095321` at line 217), not something `8950a79` introduced, but it does mean the march is an
x64-on-only path.

**Two shell traps that cost arms this session.** `cd X && setsid nohup bash s.sh > /dev/null 2>&1 &`
backgrounds the **whole** `cd && …` list, so the foreground shell never changes directory and a later
relative-path `grep` in the same command reads the wrong file — I concluded a log had been deleted when
it had not.  And `pkill -TERM -f 'verify_s29h.sh'` matches the `bash -c` wrapper of the killing command
itself, which killed my own tool call (signal 15) mid-check; use a bracketed pattern
(`[v]erify_s29h`) for both `pgrep` and `pkill`.

**Which Nsides were actually broken, enumerated** (`.qwen/tmp/lattice_s29.py`, CPU-only, walks the
shipped `_march_windows` rather than re-deriving it): the masked `(mb, NC)` head block made the spin-2
analysis march uncompilable at **nside 16, 32, 80, 112, 160, 416 and 928** — ragged last windows of 48,
96, 112, 80, 96, 96 and 96 rows.  Every power of two was clean: `3n ≡ 0 (mod 128)` iff `n ≡ 0 (mod 128)`
so nside ≥ 128 gives exact 128-row windows, and nside 64's ragged remainder is 64, itself a power of
two.  The HEALPix power-of-two lattice is precisely the set that hides this class of bug, and 928/416/160
are arbitrary-footprint sizes rather than unit-test sizes, so `2b06c44` puts nside 160 in the guard
alongside the tiny cases (six cases pass in 25.83 s).

**Post-fix fp32 scoreboard re-run** (`score_s29b_spin2.log` / `score_s29b_spin0.log`, same command and
precision as the session's earlier `score_s29_fp32_2.log`, 5 reps):

| cell | DUCC ms | GMaster ms | speedup | before the fix | rel alm |
|---|---|---|---|---|---|
| 512 s2 `map2alm` | 28.6 | 9.3 | 3.07x | 3.11x (8.9 ms) | 3.2e-07 |
| 512 s2 `alm2map` | 23.8 | 8.7 | 2.72x | 2.69x (8.5 ms) | – |
| 1024 s2 `map2alm` | 112.2 | 78.2 | **1.44x** | 1.44x (77.8 ms) | 3.7e-05 |
| 1024 s2 `alm2map` | 101.0 | 71.1 | **1.42x** | 1.41x (71.4 ms) | – |
| 2048 s2 `map2alm` | 602.0 | 575.1 | **1.05x** | 1.04x (573.7 ms) | 6.1e-05 |
| 2048 s2 `alm2map` | 561.9 | 459.9 | **1.22x** | 1.22x (457.7 ms) | – |
| 128 s0 `map2alm` | 0.9 | 0.7 | 1.35x | never scored | 2.5e-07 |
| 128 s0 `alm2map` | 0.6 | 0.7 | **0.93x** | never scored | – |
| 256 s0 `map2alm` | 2.4 | 1.2 | 2.00x | – | 2.7e-07 |
| 256 s0 `alm2map` | 1.9 | 1.2 | 1.59x | – | – |
| 512 s0 `map2alm` | 15.7 | 4.5 | 3.46x | – | 3.7e-07 |
| 512 s0 `alm2map` | 12.6 | 4.8 | 2.63x | – | – |

The head-zeroing loop moves no spin-2 ratio at the reported precision and `rel alm` is digit-for-digit
unchanged; the 512 `map2alm` cell went 8.9 → 9.3 ms while ducc0's own time went 27.6 → 28.6 ms, so that
is clock spread, not the kernel.  The new Nside-128 spin-0 `alm2map` cell is the loss HANDOFF had listed
as never measured: **0.93x on a 0.7 ms cell**, which is the sub-millisecond launch-bound regime rather
than a throughput problem.

**Suite after all three commits:** `pytest tests/ -q` on the fixed tree, **152 passed, 3 skipped in
324.67 s**, exit 0 (`pytest_s29c.log`) — the 146 that passed before plus the six lane-coverage cases.

### Session 29 addendum 2: the theta-tile lever at Nside 4096 is measured dead in both spins, which closes the last untuned knob

`GMASTER_SPIN2_MARCH_TILE` was the one lever HANDOFF still carried as "untouched". At Nside 4096 the
shipped 256 gives `ntile = ceil(16383/256) = 64`, so the analysis partial slab
`(mb=64, ntile, L - m0, NC)` float64 is at its largest there and wider tiles have their best possible
case; the 1024/2048 sweep that chose 256 had varied `warps` with the tile, so 4096 had never had this
knob measured alone. It has now (`tile4096_s29.log`, `GM_PREC=fp32`, 5 reps, one geometry per process,
controls 1298-1413 GB/s):

| cell | tile 256 (shipped) | tile 512 | tile 1024 |
|---|---|---|---|
| 4096 s2 `map2alm` | **4662.3 ms (0.83x)** | 5822.0 ms (0.65x) | 11426.7 ms (0.33x) |
| 4096 s2 `alm2map` | **3441.2 ms (1.13x)** | 3419.1 ms (1.11x) | 3426.0 ms (1.10x) |
| 4096 s0 `map2alm` | **2253.8 ms (0.86x)** | 2446.5 ms (0.77x) | 4571.3 ms (0.41x) |
| 4096 s0 `alm2map` | 1865.6 ms (1.04x) | 1861.5 ms (1.02x) | 1866.7 ms (1.02x) |

The 256 column is `score_s28_4096.log`, so it is not the same process as the other two — but each of
those arms re-measures ducc0 in-process, and it lands at 3789.9 / 3767.1 ms (spin 2) and 1894.5 /
1882.3 ms (spin 0), within 1% of each other and of the session-28 baseline. The analysis degradation is
therefore the kernel, not the clock: the tile is monotone-worse in both spins, ~25% at 512 and ~2.5x at
1024. `rel alm` is unchanged at 1.9e-04 (spin 2) and 2.1e-04 (spin 0) across all three, so this is not an
accuracy-for-speed trade. Synthesis barely moves because it takes its own `_ST` block
(`GMASTER_SPIN2_MARCH_SYNTH_TILE`), not `_TILE` — that knob remains untuned at 4096 and is the only one
left of its kind.

One probe artifact to know before reading that log: the `reps=[…]` printed on the `alm2map` rows are the
**`map2alm`** timings of the same arm (5822 / 11426 / 2446 / 4571 reproduce the row above them), so they
carry no information about synthesis spread. Read the `GMaster ms` column.

**Consequence for the two remaining losses.** `map2alm` at Nside 4096 — spin 2 0.83x, spin 0 0.86x — is
now out of launch- and block-shape knobs: the m-window width and its program cap (`march-m-window`
measurements, session 28), the per-window assembly (`analysis_assembly_ab`, 0.992x/0.981x), the rescale
limb (1.009x if deleted), the coefficient low limbs (1.031x), the rhs hoist (a bit-identical wash that
costs 0.8-3.2 GiB of residency), the emit form (0.977x/0.859x), the layout (256/1 optimal), and now the
theta tile (monotone-worse wider) have all been measured. What is left at that size is the arithmetic
itself: the march runs the fp64 recurrence on a path an order of magnitude behind the CPU's fp64 rate,
so closing 0.83x there needs a different recurrence, not another tuning pass.

