# CMB lensing quadratic estimators on the GPU (`gmaster.lensing`)

`gmaster.lensing` implements the curved-sky lensing quadratic estimators (QE) of falafel
(`falafel.qe.qe_all`, the reconstruction code of the ACT DR6 and SO lensing pipelines through
so-lenspipe) on GMaster's GPU transforms, together with their full-sky normalisation (tempura's
convention). Inputs, outputs and conventions are falafel's: C^-1-filtered alms in, unnormalised
gradient and curl alms out, `phi_LM = A_L g_LM`.

```python
from gmaster.lensing import LensingQE, normalization

qe = LensingQE(nside=2048, mlmax=4000)
rec = qe.qe_all(ucls, fTalm=t, fEalm=e, fBalm=b, estimators=["TT", "TE", "EE", "EB", "TB", "mv"])
A = normalization(["TT", "mv"], ucls, tcls, lmin=100, lmax=3000)       # (2, Lmax+1) each
phi_tt = qe.almxfl(rec["TT"][0], A["TT"][0])
```

## The estimator

Every estimator is a product of real-space maps projected with a spin-1 analysis. For temperature,

    g_LM = sqrt(L(L+1)) A_1[ -S_1(-sqrt(l(l+1)) C_l a_lm) S_0(b_lm) ],

with `S_s` a spin-s synthesis and `A_1` the spin-1 analysis; the polarised legs use spin-1, spin-2
and spin-3 syntheses (falafel's `gradient_spin`). The cost is the transforms', so two changes made
it fast:

1. **Odd-spin transforms on the CUDA march.** The difference-form march (`march_v2.cu`) was
   instantiated for spin 0 and 2 only; spin 1 and 3 fell back to s2fft's generic scatter loop
   (91 ms vs 0.9 ms at Nside 256). Spin enters the kernel only through the seed exponents,
   `M = max(m, s)` and the mirror sign `(-1)^(l+s)`, so spins 1 and 3 are new template
   instantiations (`gm_march_{ana,syn}_s{1,3}`) and spin-parametric drivers. One sign was missing:
   the marched rows are `(-1)^s` times the s2fft convention's, invisible at spin 2; it is folded
   into the per-row sign the drivers already apply. The divide-and-conquer engine (L >= 6144) is
   still validated for spin 2 only and is not used for other spins.
   Accuracy against healpy: spin 1 and 3 match spin 2's float32-march level (4e-6 at Nside 256,
   1.5e-5 at Nside 1024, relative to the maximum).

2. **Each distinct transform once.** falafel synthesises the shared legs once per estimator
   (24 syntheses and 7 analyses for TT, TE, EE, EB, TB, mv, mvpol). All transforms are linear, so
   the gradient leg of mv is the sum of the TT and TE legs, the spin-2 Y maps of (E, 0), (0, B),
   (E, B) are two transforms, and mv / mvpol are sums of the base estimators: 9 syntheses and 5
   analyses, the same outputs up to rounding. Leg maps are dropped as soon as the last estimator
   using them is formed (peak ~6 complex maps).

## The normalisation

For the pair XY, falafel's estimator is `g_LM = (-1)^M sum (l1 l2 L; m1 m2 -M) W X Y` with W the
gradient-leg term(s) of the Okamoto-Hu response f, so `A_L = 1/R_L`,
`R_L = (2L+1)^-1 sum_{l1 l2} W f` (for TT and EE this equals the optimal-weight sum, which is why
it reproduces tempura). Each F factor is expanded with the ladder identity

    [L(L+1) + b(b+1) - a(a+1)] (a L b; s 0 -s)
        = -sqrt(L(L+1)) [ sqrt((b+s)(b-s+1)) (a L b; s -1 1-s) + sqrt((b-s)(b+s+1)) (a L b; s 1 -1-s) ]

(the curl takes the difference), and every product of two 3j symbols summed over (l1, l2) is a
Gauss-Legendre integral of three Wigner-d functions, `1/2 int zeta_a zeta_b d^L`. This is the same
device machinery as GMaster's MASTER coupling matrices: one `lax.scan` over l accumulates all the
correlation functions for every (m, n) group at once, a second projects on `d^L`.

Three details matter for correctness:

* **Parity of the cross terms.** Bringing the azimuthal 3j of a term whose undifferentiated field
  is l2 to the common `(l1 l2 L)` basis costs `(-1)^(l1+l2+L)`. That is invisible in the even
  gradient sums but flips the W-f cross term of every curl (and is Okamoto & Hu's EB minus sign).
  Without it the TT curl normalisation was off by ~70x.
* **EB.** Like tempura, the B-lensing term of the EB response is dropped (the unlensed B is taken
  to vanish).
* **Low-L precision.** At low L the two response terms cancel to O(L/l) and the separable pieces
  cancel ~1e5-fold at L = 2, l ~ 2000. numpy's `leggauss` (and scipy's `roots_legendre`)
  integrate `P_l^2` at n ~ 2000 only to 1e-11 (1e-10), which that cancellation turned into a 0.7 %
  error at L = 2. The nodes are therefore Newton-polished on the device (1e-13) and the Wigner-d
  recurrence runs on the nodes `x` themselves (not `arccos x`, which loses digits near the poles).
  Residual error against an exact 3j sum: 9e-6 at L = 2, < 1e-8 for L >= 64, 2e-10 at high L.

## Validation (`tests/test_lensing.py`)

| check | result |
|---|---|
| spin 1, 2, 3 `alm2map` / `map2alm` vs healpy (Nside 64) | < 2e-5 of max |
| `qe_all` vs falafel `qe_all` (HEALPix pixelization), all 7 estimators, gradient and curl | 2.8e-5 (Nside 1024), 4-7e-5 (Nside 2048) of max |
| `normalization` vs exact 3j band sum (ducc0 Schulten-Gordon), 5 estimators x grad/curl | 1e-9 (lmax 60) |
| `normalization` vs tempura 0.2.0, all estimators incl. MV/MVPOL (lmax 2000) | <= 1.8e-4 (tempura's own low-L error; exact above L ~ 50) |
| first-order lensed sims (healpy lensing, 40 independent skies, Nside 512, lmax 512) | gradient response within 0.4 % of 1 (<= 2 sigma) in every bin L >= 60, all estimators; curl consistent with 0 |

The 1e-3 difference between GMaster and falafel on its CAR geometry is the HEALPix-vs-CAR
pixelisation: falafel on HEALPix differs from falafel on CAR by the same amount.

## Speed (`benchmarks/benchmark_lensing_qe.py`, `benchmarks/lensing_qe_results.jsonl`)

One RTX PRO 6000 Blackwell (96 GB; GPU0, because GPU1 had 7 GB free) against the AMD Threadripper
PRO 9995WX (96 cores / 192 threads, `OMP_NUM_THREADS` unset) on the same machine. Identical
C^-1-filtered alms (CMB l 100 to 0.75 mlmax), all seven estimators unless noted, median of three
warm calls. falafel 985deb9 with pixell 0.30.1 / ducc0 0.39.1 (CAR `fejer1`, float32 maps, as
so-lenspipe sets it up: 2' at mlmax 4000) and healpy 1.18.1.

| mlmax | GMaster (HEALPix) | falafel CAR (production path) | falafel HEALPix | GMaster's graph on CPU (ducc0) | ... its transforms only |
|---|---|---|---|---|---|
| 2000 (Nside 1024 / 4') | **0.103 s** | 6.27 s (61x) | 9.92 s (96x) | 1.98 s (19x) | 1.14 s (11x) |
| 4000 (Nside 2048 / 2') | **0.594 s** | 24.5 s (41x) | 38.0 s (64x) | 7.27 s (12x) | 4.30 s (7.2x) |
| 4000, TT only | **0.129 s** | 2.11 s (16x) | 3.27 s (25x) | 1.08 s (8.4x) | 0.76 s (5.9x) |

* Against the production code (falafel as so-lenspipe runs it) the full estimator set is 41-61x
  faster. About 3x of that is algorithmic (falafel's redundant transforms and single-threaded
  numpy glue) and would also be available to a CPU code; on identical work the GPU contributes
  7-19x. It is not hundreds of times: at Nside 2048 a spin-s transform takes ~38 ms on the
  float32 march and the full set is 14 of them.
* The first call compiles: 19 s (mlmax 2000) and 52 s (mlmax 4000), once per process, amortised
  over the hundreds of reconstructions of a simulation set.
* Normalisation (all estimators): 0.42 s (GPU1) / 1.15 s (GPU0) vs tempura 0.99 s / 3.86 s on the
  CPU (lmax 1500 / 3000).

Not claimed: no comparison with other GPU lensing codes; no C^-1 filtering (the inputs are
already filtered); HEALPix only (CAR maps go through GMaster's slow catalog route).

Possible next steps: batching same-spin transforms through one march launch (the spin-0 pair gave
~13 %), complex64 leg maps, and the D&C engine for spin 1 and 3.
