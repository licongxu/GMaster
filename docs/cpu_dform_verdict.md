# Can a CPU float32 difference-form march beat NaMaster? Verdict (16 September 2026)

Question (Boris, 16:37): "have we excluded that we can't win on CPU with the difference
recursion trick and fp32?"

**Answer: NO, it is not excluded — the opposite.** A finite AVX-512 float32 `(v, D)` full
spin-0 `map2alm` beats NaMaster's `map2alm` (ducc0, float64) at both 1 and 96 threads at every
nside 128–2048 on this host, by 1.7–2.0x at nside 1024–2048, at float32-class accuracy.

## What was run

`gmaster/cpu_dform.py` (harness) and `gmaster/_cpu/dform_rings.c` (kernel), branch
`exp/cpu-dform`.  The kernel is a port of `march_analysis<0,2,1>` in
`gmaster/_cuda/march_v2.cu` to ducc0's loop order: OpenMP over m, 16 northern rings per
instruction, ell innermost in blocks of 8; same seed (mantissa + int32 binade), same two-word
x and coefficients, same per-block emit scale with u(ell) folded in, same normalisation every
8 degrees, south folded into the right-hand side.  Nothing is cached across calls but the ring
geometry: the per-m coefficient rows and per-lane seeds are computed inside the timed call.

Full transform = ducc0 `map2leg` (the ring-FFT stage NaMaster's `map2alm` runs, quadrature
weights included) + C fold (parity combination, complex128 -> float32) + C march.
Same random map, same thread count (`OMP_NUM_THREADS`, one process per count), medians of
3 warm runs.  Host: AMD Ryzen Threadripper PRO 9995WX (96 cores, AVX-512).
Logs: `.qwen/tmp/cpu_dform/full_96.log`, `full_1.log`.  Test: `tests/test_cpu_dform.py`.

## Board (ms)

| nside | thr | NM map2alm | dform full | dform/NM | fft | fold | march | ducc0 leg2alm f64 | ducc0 leg2alm f32 | march/leg64 | rel err dform | rel err leg32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 96 | 0.80 | 0.49 | 0.62 | 0.22 | 0.11 | 0.12 | 0.37 | 0.36 | 0.33 | 1.1e-6 | 4.1e-8 |
| 256 | 96 | 2.63 | 1.44 | 0.55 | 0.34 | 0.25 | 0.66 | 1.40 | 1.62 | 0.47 | 1.8e-6 | 4.5e-8 |
| 512 | 96 | 15.42 | 5.60 | 0.36 | 1.75 | 0.94 | 2.44 | 4.64 | 4.66 | 0.53 | 4.2e-6 | 4.5e-8 |
| 1024 | 96 | 55.28 | 27.96 | 0.51 | 4.89 | 4.74 | 16.82 | 31.78 | 32.06 | 0.53 | 9.3e-6 | 4.7e-8 |
| 2048 | 96 | 308.72 | 160.01 | 0.52 | 14.80 | 15.79 | 126.00 | 236.40 | 234.89 | 0.53 | 1.8e-5 | 4.7e-8 |
| 128 | 1 | 5.74 | 4.75 | 0.83 | 1.08 | 0.83 | 2.80 | 4.57 | 4.61 | 0.61 | 1.1e-6 | 4.1e-8 |
| 256 | 1 | 35.78 | 27.28 | 0.76 | 4.66 | 5.90 | 16.62 | 31.78 | 30.92 | 0.52 | 1.8e-6 | 4.5e-8 |
| 512 | 1 | 250.93 | 170.86 | 0.68 | 20.88 | 39.20 | 110.49 | 223.59 | 223.76 | 0.49 | 4.2e-6 | 4.5e-8 |
| 1024 | 1 | 1743.90 | 1027.94 | 0.59 | 84.62 | 160.69 | 784.49 | 1635.34 | 1628.99 | 0.48 | 9.3e-6 | 4.7e-8 |
| 2048 | 1 | 12840.18 | 6989.01 | 0.54 | 363.53 | 737.12 | 5869.60 | 12357.86 | 12341.27 | 0.47 | 1.8e-5 | 4.7e-8 |

rel err = max |alm − NaMaster| / max |NaMaster|.  Finite everywhere; the 96-thread and
1-thread outputs are bit-identical (per-m reductions).

## Reading

- Excluded? **NO.**  Full SHT, 1 thread, nside 1024: 1028 vs 1744 ms (1.70x).  96 threads:
  28.0 vs 55.3 ms (1.98x).  nside 2048: 6989 vs 12840 ms and 160 vs 309 ms.
- Latitudinal stage alone, like for like: the march is 0.47–0.53x ducc0's `leg2alm` (float64)
  at nside 512–2048, both thread counts.
- ducc0's own float32 `leg2alm` runs at the same speed as its float64 (1629 vs 1635 ms at
  nside 1024, 1 thread) and is 200x more accurate than our march (5e-8).  What buys the 2x is
  the difference-form state surviving float32, not "fp32 lanes" by themselves.
- Accuracy is the march's known sqrt(n)·u floor: 1e-6 at nside 128, 1.8e-5 at 2048, growing
  with nside.  ducc0 float64 is 1e-13-class.
- The fold is a separate transposing pass that ducc0 does not need (161 of 1028 ms at nside
  1024, 1 thread).  Fusing it into the FFT stage would take the total to ~0.50x.
- Scope: spin 0, analysis only, n_iter = 0, one machine.  An experiment on `exp/cpu-dform`,
  not a NaMaster replacement, and not on `main`.

## Paragraph for Boris

We have not excluded it — the opposite.  I ported the GPU difference-form march (the (v, D)
state, two-word coefficients, 8-degree block rescaling) to an AVX-512 float32 OpenMP kernel in
ducc0's loop order (m outer, 16 rings per instruction), fed by ducc0's own ring-FFT stage so
the FFT is exactly what NaMaster runs.  On the 9995WX, same random map and same thread count,
the full spin-0 map2alm is faster than NaMaster/ducc0 at every nside from 128 to 2048 and at
both 1 and 96 threads: at nside 1024 it is 1.03 s vs 1.74 s single-threaded (1.7x) and 28 ms vs
55 ms on 96 threads (2.0x); at nside 2048, 7.0 s vs 12.8 s and 160 ms vs 309 ms.  The
latitudinal stage alone is 0.5x ducc0's leg2alm at both thread counts.  The caveats: (i) the
accuracy is float32-class, 1e-5 relative at nside 1024–2048 against ducc0's 1e-13, and it grows
with nside — the march's known sqrt(n) floor, not a bug; (ii) ducc0's own float32 mode is no
faster than its float64 (and 200x more accurate than ours), so what buys the 2x is the
difference-form recurrence surviving float32, not fp32 lanes by themselves; (iii) it is spin 0,
analysis only, n_iter=0, one machine, and the code is an experiment on exp/cpu-dform, not a
NaMaster replacement.  So: yes, the recursion trick plus fp32 wins on CPU by about 2x, if a
1e-5 transform is acceptable for the use.
