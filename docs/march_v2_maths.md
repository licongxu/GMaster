# The v2 latitudinal march: the mathematics, written down

This note is the human-readable statement of the mathematics `gmaster/_cuda/march_v2.cu` and
`gmaster/_march_v2.py` rely on, in the notation of `docs/latitudinal_march_maths.md` (the note for
the previous kernel; its §1 closed form, §2 recurrence and coefficients, §6 fold / mirror and §8
polar skip are unchanged and reused here).  **The proof of record is the Lean 4 / Mathlib project in
`formal/lean/`** (`cd formal/lean && lake build`, no `sorry`); §1–§3 are `GMasterMarch/DiffForm.lean`,
§4 is `GMasterMarch/EmitV2.lean`, and each statement below names its theorem.  §5, the error
analysis, is heuristic and is not formalised; its evidence is numerical (HANDOFF addendum).

Notation as before: `L = lmax + 1`, spin `s ∈ {0, 2}`, order `m ≥ 0`, `alpha = m + s`,
`beta = |m - s|`, `M = max(m, s)`, `n = ell - M`, `x = cos theta`.  The row the kernel marches is the
Jacobi sequence of §2 of the old note,

    v_n = (c1_n x + c0_n) v_{n-1} - cb_n v_{n-2},   v_0 = 1,   v_1 = ((alpha+beta+2)/2) x + (alpha-beta)/2,   (1)

(Lean: `GMasterMarch.Recurrence.march`, equal to `P_n^(alpha,beta)(x)` by `march_eq_jacobi`), and the
true row is `d^ell_{m,-s}(theta) = (-1)^(m+s) N(ell) (sin theta/2)^alpha (cos theta/2)^beta v_{ell-M}` with
`N(ell) = 2^(eps_m LG2N(ell,m))` (old note §1).  The differences from the previous kernel are four:
the pair marched is `(v, D)` with `D = v_n - v_{n-1}` instead of `(v_n, v_{n-1})`; the southern
hemisphere marches the reflected row `(-1)^n v_n`; the lane binade is renormalised every eight
degrees to the exponent field of `max(|v|, |D|)`; and the emit forms a plain float32 number with
the normalisation folded into the tables.

## 1. Difference form

Lean: `GMasterMarch.DiffForm` -- `diff_form_step`, `jacobi_C_eq`, `diffMarch` (the `(v, D)` pair the
kernel iterates), `diff_march_eq_rec2` (any two-term recurrence), **`diff_march_eq_march`** (the
Jacobi row), `diff_march_value`, `seed_D1_north`.

**Proposition (difference form).** Let `v_n = A_n v_{n-1} - B_n v_{n-2}` for `n ≥ 2` and
`D_n = v_n - v_{n-1}` for `n ≥ 1`.  Then for every `n ≥ 2`

    D_n = (A_n - 1 - B_n) v_{n-1} + B_n D_{n-1},        v_n = v_{n-1} + D_n.                         (2)

*Proof.* `D_n = A_n v_{n-1} - B_n v_{n-2} - v_{n-1} = (A_n - 1 - B_n) v_{n-1} + B_n (v_{n-1} - v_{n-2})`.
The second identity is the definition of `D_n`. □  (`diff_form_step`, one step.)

**Corollary (the kernel's march).** Starting from `(v_1, D_1) = (v_1, v_1 - v_0)` and iterating
`(v, D) ← (v + D', D')` with `D' = C_n v + B_n D`, `C_n = A_n - 1 - B_n`, the pair after `n - 1`
iterations is `(v_n, v_n - v_{n-1})` for every `n ≥ 1`.
*Proof.* Induction on `n`; the step is (2). □  (`diff_march_eq_rec2`: `diffMarch` from
`(s1, s1 - s0)` equals `(rec2 (k+1), rec2 (k+1) - rec2 k)` at every index `k`, for the generic
`rec2` of `ExponentCarry.lean`.)

Specialised to (1), `A_n = c1_n x + c0_n` and

    C_n = c1_n (x - 1) + E⁺_n,        E⁺_n = c1_n - 1 - cb_n + c0_n                                (3)

(`jacobi_C_eq`).  This is the form the kernel evaluates: the lane carries `x - 1` and the table
carries `c1` and `E⁺`, both as (hi, lo) float32 pairs (§5 explains why).  `diff_march_eq_march`
states that `kernelMarch alpha beta x k` -- `diffMarch` with the coefficients (3), `B = cb`, seeds
`1`, `v_1` -- equals `(march (k+1), march (k+1) - march k)`, hence its value lane is
`P_{k+1}^(alpha,beta)(x)` (`diff_march_value`).  The seed the kernel forms is
`D_1 = v_1 - v_0 = ((alpha+beta+2)/2)(x - 1) + alpha` (`seed_D1_north`; `cst = a` in `seed()`).

## 2. Reflection: the southern hemisphere

Lean: `GMasterMarch.DiffForm` -- `reflect_step`, **`reflect_recurrence`**, `rec2_reflect`,
`kernelMarchSouth`, **`south_march_eq_reflect`**, `seed_D1_south`, `march_reflect_eq_swapped`.

**Proposition (reflection).** If `v_n = (c1_n x + c0_n) v_{n-1} - cb_n v_{n-2}` for all `n ≥ 2`, then
`w_n := (-1)^n v_n` satisfies

    w_n = (c1_n (-x) - c0_n) w_{n-1} - cb_n w_{n-2}.                                                 (4)

*Proof.* Multiply the recurrence by `(-1)^n`: `(-1)^n v_n = (c1_n x + c0_n) (-1)^n v_{n-1} - cb_n (-1)^n v_{n-2}
= -(c1_n x + c0_n) w_{n-1} - cb_n w_{n-2} = (c1_n (-x) - c0_n) w_{n-1} - cb_n w_{n-2}`. □
(`reflect_step` for one step, `reflect_recurrence` for the sequence, `rec2_reflect` at the level of
`rec2`: the recurrence with `a'_n = c1_n(-x) - c0_n` from seeds `(s0, -s1)` is `(-1)^n` times the one
with `a_n = c1_n x + c0_n` from `(s0, s1)`.)

For a southern ring `x < 0`, put `y = |x| = -x`.  Then (4) is the *same* recurrence (1) with `y` in
place of `x` and `-c0` in place of `c0`, and its difference form (3) has `C_n = c1_n (y - 1) + E⁻_n`,
`E⁻_n = c1_n - 1 - cb_n - c0_n`.  So every lane, northern or southern, marches `|x| - 1 ∈ (-1, 0]`
with the same code and the table column `E⁺` or `E⁻` (`load_tab(.., south)`), seeded with
`w_1 = -v_1 = ((alpha+beta+2)/2) y - (alpha-beta)/2`, `D_1 = w_1 - 1 = ((alpha+beta+2)/2)(y - 1) + beta`
(`seed_D1_south`; `cst = b`).  `south_march_eq_reflect`: the value lane of `kernelMarchSouth` at
`y = -x` is `(-1)^(k+1) march (k+1)`.  The emit restores `(-1)^n` on odd `n` (`oddsign`); in the
synthesis kernel it is the `so` sign combining the even- and odd-`n` accumulators.

*Relation to the closed-form reflection.* By `jacobi_reflect` (Mirror.lean),
`(-1)^n P_n^(alpha,beta)(x) = P_n^(beta,alpha)(-x)` (`march_reflect_eq_swapped`): the southern lane
is literally the `(beta, alpha)` Jacobi row at `|x|`.  This is consistent with (4): `c1` and `cb`
are symmetric in `(alpha, beta)` and `c0 ∝ alpha^2 - beta^2` flips sign.  The kernel needs (4), not
the closed form, because it never swaps the half-angle prefactor: the seed
`(sin theta/2)^alpha (cos theta/2)^beta` is that of the *northern* form at the southern `theta`.

## 3. Exact block scaling

Lean: `GMasterMarch.DiffForm` -- `diff_step_homogeneous`, **`diff_march_scaled`**, `block_rescale`
(built on `ExponentCarry.lane_rescale`).

Every lane stores `v · 2^ex` with `v, D` float32 and `ex` int32; initially `v = 2^(f - ⌊f⌋)`,
`ex = ⌊f⌋`, `f = alpha log2 sin(theta/2) + beta log2 cos(theta/2)`, exactly as before.  After each
block of `UNR = 8` degrees (`normalise()`), with `eb` the biased exponent field of
`big = max(|v|, |D|)` (clamped to `≥ 1`), both `v` and `D` are multiplied by `sc = 2^(127 - eb)` and
`ex += eb - 127`; afterwards `big ∈ [1, 2)`.

**Lemma (homogeneity).** The map `(v, D) ↦ (v + D', D')`, `D' = C v + B D`, is linear, so the
difference march from seeds `(λ s0, λ s1)` is `λ` times the march from `(s0, s1)` in both lanes at
every degree.  *Proof.* Induction; one step is `C (λ v) + B (λ D) = λ (C v + B D)`. □
(`diff_step_homogeneous`, `diff_march_scaled`.)  Hence rescaling `(v, D)` by `2^(-k)` and adding `k`
to `ex` leaves every later represented value `v · 2^ex` unchanged (`block_rescale`).

*IEEE remark (not formalised, as in the old §3).* Multiplication by a power of two changes only the
exponent field, so `v *= sc`, `D *= sc` are exact unless they overflow or underflow.  With
`big ∈ [1, 2)` after a normalisation and the per-degree growth of the pair bounded by
`1 + |C| + |cb| ≤ 2^15` for `m < 2^14` (the coefficient bounds of the old §2; `|x - 1| ≤ 2`), eight
degrees grow by at most `2^120 < 2^126`: no overflow before the next normalisation.  A lane cannot
shrink to the denormal range either: marched forward from `ell = M`, the row is the dominant
solution (it grows through the forbidden region up to the turning point and oscillates after it),
and `D` is bounded by `2 max |v|`.  The clamp `eb ≥ 1` only matters for an exactly zero lane
(padding), which stays zero.

## 4. The emit: factorisation and the range of `u`

Lean: `GMasterMarch.EmitV2` -- **`emit_factorisation`**, `biased_exponent`, `u_range`,
`seven_steps`, `step_ratio_bounds_ge`, `step_ratio_bounds_lt`, `step_bound_ge`.

At degree `ell` of a block with base `b` (`b = M` for `ell ∈ {M, M+1}`, then `b = M + 2 + 8 i`) the
lane's true-magnitude value is `v · 2^ex · N(ell)`.  The kernel forms it as

    val = v · sc,   sc = 2^(ex + lex_b),   lex_b = ⌊log2 N(b)⌋,      then    (Σ_lanes val · r) · u(ell),
    u(ell) = 2^(log2 N(ell) - lex_b),                                                              (5)

`sc` per lane by exponent-field assembly (`blockscale`: the table stores `lexp = lex_b + 127` and
`pow2e(ex + lexp)` is `2^(ex + lexp - 127)`) and `u(ell)` once per `(ell, channel)` after the warp
reduction (`mt` in analysis; folded into the coefficients in synthesis).

**Lemma (factorisation).** For real `v, g` and integers `ex, lb`:
`v · 2^ex · 2^g = v · 2^(ex + lb) · 2^(g - lb)`.  *Proof.* `2^a 2^b = 2^(a+b)`. □
(`emit_factorisation`; `biased_exponent` for the `+127 - 127`.)  The point of the factorisation is
that the lane-dependent factor `sc` is an exact power of two while the only non-trivial mantissa,
`u(ell)`, depends on `(ell, m)` alone and can be applied after the sum over lanes.

**Lemma (range of `u`).** For every degree of a block, `u(ell) ∈ [2^-45.5, 2)`; in particular it is
a normal float32 (the kernel comment quotes the looser `(2^-53, 2)`).
*Proof.* `log2 N(ell) - lex_b = (log2 N(ell) - log2 N(b)) + frac_b` with `frac_b = log2 N(b) - lex_b ∈ [0, 1)`,
and `log2 N(ell) - log2 N(b)` is a sum of `ell - b ≤ 7` per-degree steps

    t(ell') = eps_m · ½ log2 ρ(ell'),    ρ(ell') = (ell'+m)(ell'-m) / ((ell'-s)(ell'+s)),    ell' = b+1, ..., ell,

(the cumulative form of `LG2N` in `_window_tables`).  For `m ≥ s` (`eps_m = +1`, `ell' ≥ m + 1`):
`ρ = (ell'^2 - m^2)/(ell'^2 - s^2)` is `≤ 1` and increasing in `ell'`, so it is smallest at
`ell' = m + 1` where `ρ = (2m+1)/((m+1)^2 - s^2) ≥ (2m+1)/(m+1)^2 ≥ 2/(m+2)`
(`step_ratio_bounds_ge`); hence `t ∈ [½ (1 - log2 (m+2)), 0] ⊂ [-6.5, 0]` for `m + 2 ≤ 2^14`
(`step_bound_ge`; `L = 12288` at Nside 4096 is inside).  For `m < s` (`eps_m = -1`, the rows
`m ∈ {0, 1}` of spin 2, `ell' ≥ s + 1`): `1 ≤ ρ ≤ (s+1)^2/(2s+1) = 9/5` (`step_ratio_bounds_lt`), so
`t ∈ [-½ log2 (9/5), 0] ⊂ [-0.5, 0]`.  Seven steps in `[-6.5, 0]` plus `frac_b ∈ [0, 1)` give an
exponent in `[-45.5, 1)` (`seven_steps`) and `u ∈ [2^-45.5, 2)` (`u_range`). □

*Roundings.* `val = v · sc` is exact; the lane product enters the float32 tree sum as before (old
§5, error `delta`); `u` is rounded once when tabulated and once in the product, so the tile partial
carries `(1+u)^2 (1+delta) - 1` as in `Emit.two_roundings`, with `u = 2^-24`.  Lanes with
`ex + lex_b ≤ -127` are flushed by `blockscale` (`pow2e(0) = 0`): at the block base such a lane's
true magnitude is `|v| · 2^(ex + lex_b) · u < 2^-125 · 2 = 2^-124`, and it stays below
`2^-124 · G` over the block, `G` the pair's growth within the block (`O(1)` per degree in
practice, `≤ 2^15` in the worst case of §3).  Every emitted value is a true magnitude `|d| ≤ 1`
contracted against `O(1)` right-hand sides, so a dropped term is far below the float32 resolution
of the partial it enters.  The driver applies the `(-1)^(m+s)` of the closed form and
`sqrt((2ell+1)/4pi)` in float64 and sums the tile partials in float32 -> float64 with no exponent
bookkeeping.

## 5. Error analysis (heuristic, not formalised)

This section is the reasoning behind the design; like §8 of the old note it is asymptotic and
is validated numerically, not proved.  Numbers: `u = 2^-24 ≈ 6e-8`; measured floors against the
float64 recurrence (HANDOFF addendum of the v2 session): rms relative `2.8e-6` (`m = 0`) and
`2.9e-6` (`m = 1000`) at Nside 1024 and `5.7e-6` at Nside 4096, against `4e-5` with a single-word `C`
and `4 %` with a plain single-word three-term recurrence at `m = 0`.

**Local modes.** Over a few degrees the coefficients are nearly constant and `cb_n → 1`, so (1)
is locally `v_n ≈ A v_{n-1} - v_{n-2}` with characteristic roots `e^{±i phi}`, `2 cos phi = A`
(more precisely `2 sqrt(cb) cos phi = A`).  In the oscillatory region `|A| < 2`; `phi → 0` at the
poles (`x → ±1`, where for Legendre `A → 2` exactly) and at every turning point
`m ≈ (ell + ½) sin theta` (the boundary of the forbidden region).  In terms of `phi`,
`C = A - 1 - cb = -(1 - sqrt cb)^2 - 2 sqrt(cb) (1 - cos phi) ≈ -phi^2`: the difference coefficient
vanishes quadratically exactly where the march is ill-conditioned, and `D ≈ 2 sin(phi/2) |v| ≈ phi |v|`.

**Why the plain three-term march loses accuracy.** An error `delta` injected into `v_k` with
`v_{k-1}` untouched is the state perturbation `(delta, 0)`, which decomposes on the two modes
`(e^{±i phi}, 1)` with amplitude `delta / (2 sin phi)` each and evolves as
`delta sin((n-k+1) phi) / sin phi`.  The rounding of `fl(A v_{k-1} - B v_{k-2})` is such an
injection of size `~2u|v|`, so every rounding is amplified by `1/sin phi`; over `n` steps with
random signs the relative error is `~sqrt(n) u / sin phi`.  On the first HEALPix ring at
Nside 4096, `theta ≈ 2e-4`, `1/sin phi ≈ 5e3`, `sqrt(12288) ≈ 110`: `~4e-2`, the measured `4 %`.
A *systematic* error is worse: a relative error `eps` in `x` perturbs `A` by `delta A = c1 x eps`
at every step, and from `2 cos phi = A`, `delta phi = -delta A / (2 sin phi)` per step, i.e. a
coherent phase drift `~ n c1 x eps / (2 sin phi)` after `n` steps -- with single-word `x`
(`eps = u`) at `n = 12288`, `sin phi = 2e-4`, this is `~2` radians.

**Why the difference form removes the amplification.** Its roundings are of two kinds.
(i) `v ← fl(v + D)`: an error `delta ~ u|v|` on `v_n` *with `D_n` fixed*, i.e. `v_{n-1}` shifted by
the same `delta`.  The state perturbation `(delta, delta)` has modal amplitude
`delta e^{-i phi/2} / (2 cos(phi/2)) ≈ delta/2`: not amplified.  (ii) `D ← fl(cb D + C v)`: an error
`eta ~ u (|D| + |C v|) ~ u phi |v|` on `D_n` with `v_n` fixed, i.e. the perturbation `(0, -eta)`,
amplified by `1/(2 sin phi)` to `~u|v|/2`.  Both kinds therefore contribute `O(u|v|)` per step
irrespective of `phi`, and the total is the random-walk floor `~sqrt(n) u`: `sqrt(3072) u ≈ 3.3e-6`
at Nside 1024 and `sqrt(12288) u ≈ 6.6e-6` at 4096, against the measured `2.8e-6` and `5.7e-6`.

**Why `x - 1`, `c1` and `E` are two-word.** The only coefficient that enters the amplified path is
`C`, and the analysis above needs its error to be `O(u|C|)`, not `O(u|A|)`.  Near a pole `x - 1`
is `~ -theta^2/2`; a single-word `x` carries an absolute error `u` into `x - 1`, i.e. a relative
error `u/(theta^2/2)` in the leading term of `C`, and a single-word `E` leaves an error `~u|c1|`
that does not shrink with `C`.  Both are systematic and drift the phase like the `delta A` above.
The kernel therefore stores `|x| - 1 = xh + xl`, `c1 = c1h + c1l`, `E^± = Eh + El` as float32 pairs
(each split exact from float64 to `2^-48` relative) and forms

    C = fma(c1h, xh, Eh) + [ fma(c1h, xl, fma(c1l, xh, El)) ].

The first fma rounds once, by `u|C|` (its result is `C` up to the bracket), and this rounding
vanishes with `C` at the turning point where the amplification peaks; the bracket is `~2^-24` of the
leading terms and its roundings `~2^-48 |c1 (x-1)|` are negligible; the final add rounds once more
by `u|C|`.  A `C` formed from single words instead leaves the coherent `u|c1|`-class error, whose
phase drift `n u / (2 sin phi)` is the `4e-5` floor measured for the single-word `C` (a drift, not
a random walk: it grows like `n`, the two-word floor like `sqrt n`).  `cb` is single-word: its error
enters through `cb D`, i.e. as `u|D| ~ u phi |v|` on `D`, kind (ii) above, already unamplified.
