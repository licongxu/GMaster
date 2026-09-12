# The table-free latitudinal march: the mathematics, written down

This note is the human-readable statement of the mathematics `gmaster/_spin_march_pallas.py`
relies on.  **The proof of record is the Lean 4 / Mathlib project in `formal/lean/`**
(`cd formal/lean && lake build`); each section below names the theorems that prove it.  Most of
the material was validated numerically in the sessions recorded in `HANDOFF.md` but never derived;
§5, §7 and §8 are new in session 36.  The project has no `sorry`: DLMF 18.9.2, the three-term
recurrence of the explicit Jacobi sum, is proved in `GMasterMarch.JacobiSum.jacobi_three_term`
(a telescoping certificate over the summands, see §2), and the spin-2 mirror identity in
`GMasterMarch.Mirror.mirror_identity`; the identification of the Jacobi closed form with Wigner's
function is `GMasterMarch.WignerJacobi.wignerD_eq_jacobiForm`, proved from Wigner's explicit sum
formula.  Outside Lean are the WKB decay estimate of §8, which is asymptotic analysis
justifying a rule that is the reference's own and is checked against ducc0, and the IEEE scaling
remark on the synthesis accumulator at the end of §5.  Notation: `L = lmax + 1`, spin `s >= 0`,
order `m >= 0`, colatitude `theta`, `x = cos theta`.  The transform the kernel implements is,
for each order `m` and one of four real channels `r_c(theta)` (real/imaginary x direct/mirror),

    A_c(ell, m) = sum_theta d^ell_{m,-s}(theta) r_c(theta),          ell = max(m, s) .. L-1.   (1)

## 1. Closed form of `d^ell_{m,-s}` as a Jacobi polynomial

Lean: `GMasterMarch.Indices` -- `alpha_add_beta`, `alpha_sub_beta`, `alpha_sq_sub_beta_sq`,
`jratio_of_ge`, `jratio_of_lt` (the `eps_m` flip), `sign_of_even_spin`, `sign_of_odd_spin`.  `GMasterMarch.WignerJacobi` -- `wignerD` (Wigner's explicit sum for `d^j_{m,-s}`, Wigner 1931 /
Sakurai 3.8.33, specialised to `(m', m) = (m, -s)`), `jacobiForm` (the closed form below with
`sqrt(jratio)` as normalisation), `coeff_ge` / `coeff_lt` (the per-term factorial identities in the
two branches) and **`wignerD_eq_jacobiForm`**: the two agree for every `m, s <= j` and every
colatitude.  The proof matches the powers of `sin(theta/2)`, `cos(theta/2)` term by term
(`(x-1)/2 = -sin^2`, `(x+1)/2 = cos^2`), and the coefficients by
`C(j+s, j-m-k) C(j-s, k) (j-s-k)! (j-m-k)! (k+m+s)! k! = (j+s)! (j-s)!` for `m >= s`
(`(j+m)! (j-m)!` for `m < s`), which is where the `eps_m` flip comes from.

**Proposition.** With `alpha = m + s`, `beta = |m - s|`, `M = max(m, s)`, `n = ell - M`,

    d^ell_{m,-s}(theta) = (-1)^(m+s) * 2^(eps_m LG2N(ell,m)) * (sin theta/2)^alpha (cos theta/2)^beta * P_n^(alpha,beta)(x),
    LG2N = log2 sqrt( (ell+m)! (ell-m)! / ((ell-s)! (ell+s)!) ),   eps_m = +1 (m >= s), -1 (m < s).

*Proof.* The standard Jacobi form of the Wigner small-d matrix (Edmonds 1957 eq. 4.1.23; Wikipedia
"Wigner D-matrix", Jacobi-polynomial form) is, for `d^j_{m' m}`, with `a = |m - m'|`, `b = |m + m'|`,
`n = j - (a + b)/2`,

    d^j_{m' m}(beta) = xi * sqrt( n! (n+a+b)! / ((n+a)! (n+b)!) ) * (sin beta/2)^a (cos beta/2)^b * P_n^(a,b)(cos beta),
    xi = 1 if m >= m', (-1)^(m-m') otherwise.

Put `m' -> m` (our row) and `m -> -s`.  Then `a = m + s = alpha`, `b = |m - s| = beta`, and
`a + b = 2 max(m, s)` so `n = ell - M`.  Since `-s < m` unless `m = s = 0`, `xi = (-1)^(-s-m) = (-1)^(m+s)`.
For the factorial ratio, `n + a + b = ell + M`, `n + a = ell - M + m + s`, `n + b = ell - M + |m - s|`:

* `m >= s`: `n = ell - m`, `n+a+b = ell + m`, `n+a = ell + s`, `n+b = ell - s`, so the ratio is
  `(ell-m)!(ell+m)! / ((ell+s)!(ell-s)!)`, i.e. `2^(+LG2N)`;
* `m < s`: `n = ell - s`, `n+a+b = ell + s`, `n+a = ell + m`, `n+b = ell - m`, so the ratio is the
  reciprocal, i.e. `2^(-LG2N)`.  This is the `eps_m` flip. □

The kernel writes `(-1)^m` for `(-1)^(m+s)`; the two agree for even spin (0 and 2, the only spins
the march serves).  A spin-1 or spin-3 march must restore the `(-1)^s`.  Also checked numerically against
Wigner's explicit sum formula for spins 0-3, `ell <= 12`, four colatitudes: worst relative
difference 6e-13 (`.qwen/tmp/closed_form_check_s36.py`); the same script reproduces `eval_jacobi` from
the §2 coefficients to 4e-12 over 60 degrees.  The Lean theorem `wignerD_eq_jacobiForm` makes the
identity exact for all `m, s <= ell`.

## 2. The three-term recurrence and its seeds

Lean: `GMasterMarch.JacobiSum` -- `jacobi` (DLMF 18.5.7 sum), `tj` (its `k`-th summand in
`u = (x-1)/2`), the ratio identities `tj_succ`, `tj_pred_v`, `tj_pred_u`, their combinations
`tj_pred`, `tj_pred2` (a degree-`n` summand through two / three degree-`(n+1)` / `(n+2)` summands,
with no division by `u` or `u+1`), the telescoping certificate `W` with `summand_telescopes`,
`W_zero`, `W_top`, and DLMF 18.9.2 itself, `jacobi_three_term_succ` / `jacobi_three_term`.
`GMasterMarch.Recurrence` -- `jacobi_zero`, `jacobi_one`, `c0_eq`, `den_pos`, `recurrence_form`
(the exact rearrangement into `(c1 x + c0) v - cb v'`), `march` (the sequence the kernel
computes) and `march_eq_jacobi`.  No `sorry` anywhere in the chain.

*Proof sketch of 18.9.2 for the sum (the Lean file follows it line by line).* With
`t_k = C(n+α,n-k) C(n+β,k) u^k (u+1)^(n-k)` (zero outside `0 ≤ k ≤ n`), the binomial identities
`C(N, j+1)(j+1) = C(N, j)(N-j)` and `C(N, j)(N+1) = C(N+1, j)(N+1-j)` give
`(α+k+1)(k+1)(u+1) t_{k+1} = (n-k)(n+β-k) u t_k` (I1),
`(n+1+α)(n+1+β)(u+1) t^n_k = (n+1-k)(n+1+β-k) t^{n+1}_k` (I2) and
`(n+1+α)(n+1+β) u t^n_k = (α+k+1)(k+1) t^{n+1}_{k+1}` (I3); (I2)-(I3) removes the `u`.  Writing
the recurrence's summand at degree `N` (times `D' = (N+α)(N+β)(N+α-1)(N+β-1)`) through
`t_k, t_{k+1}, t_{k+2}` alone, it equals `W(k+1) - W(k)` for
`W(k) = -q_0(k) t_k + (q_2(k-1) - C (u+1)(k+1)(k+1+α)) t_{k+1}` (explicit polynomials `q_0, q_2, C`
in the file), using (I1) at `k+1`; `W(0) = 0` by (I1) at `0` and `W(N+1) = 0` since `t` vanishes
past `N`, so the sum telescopes to zero.

DLMF 18.9.2 for `P_n^(alpha,beta)`, written with `S = 2n + alpha + beta`:

    2n (n+alpha+beta)(S-2) P_n = (S-1)[ S (S-2) x + (alpha^2 - beta^2) ] P_{n-1} - 2 (n+alpha-1)(n+beta-1) S P_{n-2}.

Hence `v_ell = (c1 x + c0) v_{ell-1} - cb v_{ell-2}` with `den = 2n(n+alpha+beta)(S-2)` and

    c1 = (S-1) S (S-2) / den,   c0 = (S-1)(alpha^2 - beta^2) / den,   cb = 2 (n+alpha-1)(n+beta-1) S / den.

**Lemma (no cancellation in `c0`).** `alpha^2 - beta^2 = 4 m s` for every `m, s >= 0`.
*Proof.* `alpha - beta = m + s - |m - s| = 2 min(m,s)` and `alpha + beta = 2 max(m,s)`. □
So no coefficient is a difference of large numbers; each is a ratio of small integer products,
which is what lets a float32 (high, low) split of the coefficients carry them to `2^-48`.

Seeds: `P_0 = 1`; from DLMF 18.5.7 at `n = 1`, `P_1^(alpha,beta)(x) = (alpha+1) + (alpha+beta+2)(x-1)/2
= ((alpha+beta+2)/2) x + (alpha-beta)/2`.  For `s = 2, m >= 2` that is `(m+1) x + 2`.

## 3. Exact exponent carry

Lean: `GMasterMarch.ExponentCarry` -- `lane_rescale`, `guard_large`, `guard_small`,
`step_homogeneous`, `rec2_scaled` / `march_eq_rec2` / `march_scaled` (the march from seeds scaled by
`lambda` is `lambda` times the march, at every degree), `no_overflow_before_guard`.

Every lane stores its value as `v * 2^ex` with `v` float32 and `ex` int32.  Initially
`v = 2^(f - floor f)`, `ex = floor f` where `f = alpha log2 sin(theta/2) + beta log2 cos(theta/2)`
(so `v in [1,2)`).  Whenever `max(|v_ell|, |v_{ell-1}|)` leaves `[2^-24, 2^24]` both lanes are multiplied
by `2^{-+24}` and `ex` moves by `+-24`.

**Lemma.** The rescaling is exact and commutes with the recurrence.
*Proof.* Multiplying a binary float by a power of two changes only its exponent field, so it is
exact as long as neither overflow nor underflow occurs; with `|v| <= 2^24 * max growth per step`
and the float32 range `2^+-126`, a single step's growth would have to exceed `2^100` to overflow,
which the coefficient bounds of §2 exclude (`|c1|,|c0|,|cb| <= O(m)`, so growth per step
`<= 2^15` at `m < 2^14`).  The recurrence is linear and homogeneous in `(v_ell, v_{ell-1})`, so
scaling both by the same constant scales every later value by it. □

The same reasoning is why the `2^-24` band was chosen: the product `v * r` with `|r| <= 2^24`-class
right-hand sides and a 256-lane sum stays inside float32 range for any lane inside the band.

## 4. The emit and its flush threshold

Lean: `GMasterMarch.Emit` -- `flushed_lane_small`, `flush_bound` (`2^-71 R` over at most 256 lanes).

At degree `ell` the program holds `(v_theta, ex_theta)` for its 256 lanes and reduces
`sum_theta v_theta 2^(ex_theta - emax) r_c(theta)` with `emax = max_theta ex_theta`.  Lanes with
`ex_theta - emax < -126` are flushed to zero by the exact-power builder.

**Lemma (flush bound).** The flushed lanes change the tile partial by less than `2^-70` relative
to the largest lane's term.
*Proof.* Every stored `|v| in [2^-24, 2^24]`; the largest lane's term is at least
`2^-24 * 2^emax * |r|`, a flushed lane's at most `2^24 * 2^(emax-127) * |r|`; the ratio is `2^-79`
and there are at most `2^8` lanes. □

## 5. The binade-split emit (new, session 36)

Lean: `GMasterMarch.Emit` -- `binade_split`, `frac_factor_bounds` (`[1, 2)`), `int_binade_is_zpow`,
`two_roundings` (`(1+u)^2 - 1`), `dropped_partial_small`.

The tile partial of (1) is, from §1,

    A_c^tile(ell,m) = (-1)^(m+s) * 2^(eps LG2N + emax) * sum_theta [ v_theta 2^(ex_theta - emax) ] r_c(theta).   (2)

Write `eps LG2N = lex + frac`, `lex = floor(eps LG2N)`, `frac in [0,1)`.  The kernel stores

    p = fl32( fl32( sum_theta val_theta r_c(theta) ) * fl32( (-1)^(m+s) 2^frac ) ),     e = lex + emax  (int32),

and the driver returns `fl64(p) * 2^e`, then sums the tiles in float64.

**Proposition.** `fl64(p) 2^e` equals (2) with relative error at most `(1 + u)^2 (1 + delta) - 1`,
where `u = 2^-24` and `delta` is the relative error of the float32 tree sum; the multiplication by
`2^e` is exact whenever `-1022 <= e <= 1023`, and a partial whose `e < -1022` is below `2^-1022`
in magnitude and is dropped.
*Proof.* `2^e` is exactly representable in float64 for `e` in that range and multiplication by it
only changes the exponent field.  The two float32 roundings are the stored mantissa `fl32(2^frac)`
and the product; each contributes a factor in `[1-u, 1+u]`.  The tree error `delta` is the same
term the float64-emit kernel carried, since that kernel also summed in float32 before converting.
The dropped partials: the alm entries are `O(1)` sums of these partials, so a partial below
`2^-1022` is below the float64 resolution of any sum it enters. □

The shipped kernel had computed `2^(emax + eps LG2N)` with a warp-wide float64 `exp2`, converted
four float32 partials to float64 and multiplied them in float64, per degree.  On a card with
float64 at 1/64 of the float32 issue rate that tail was 40 % of the analysis kernel: 3.74 -> 2.26 ms
per 128-order window at Nside 1024 (`.qwen/tmp/emit_ablate_s36.py`), and with eight degrees per
loop iteration 1.92 ms.  Measured relative change against the float64-emit kernel over every lane:
7.5e-08 at Nside 1024 (windows m0 = 0 / 1024 / 2560), 7.3e-08 at 2048, 6.6e-08 at 4096 --
i.e. `~ 2u` as the proposition predicts.

*Synthesis accumulator (remark, not formalised).* The synthesis kernel contracts the same marched
row against harmonic coefficients and accumulates per lane.  It used to carry a block exponent per
accumulator, re-derived at every degree for each order sign.  The window's coefficients are now
divided by the power of two at their maximum, so `|c| < 4`; with `|d| <= 1` the accumulator over at
most `L` degrees is below `2^(2 + log2 L)`, inside float32 range with no exponent lane, and the
alignment `nd * 2^tex` of the marched row is one exact power shared by both order signs.  Powers of
two commute with rounding, so the result is bit-identical to the block-exponent form except for
terms below `2^-126` of the window scale, which are flushed; the driver multiplies back by the
window scale.  Map error against ducc0 is unchanged to the printed digit (2.32e-04 at 1024 spin 2,
8.19e-04 at 2048 spin 0, max relative to the map maximum, `.qwen/tmp/acc_s36.py`).  This is an
IEEE scaling statement rather than an identity and has no Lean counterpart.

## 6. Hemisphere fold (spin 0) and mirror channel (spin 2)

Lean: `GMasterMarch.Fold` -- `jacobi_symmetric_parity` (`P_n^{(a,a)}(-x) = (-1)^n P_n^{(a,a)}(x)`,
proved from the explicit sum), `row0_reflect` (the spin-0 row under `theta -> pi - theta`),
`sum_pairs_even`, `sum_pairs_odd` (equator counted once), `fold_even`.  `GMasterMarch.Mirror` --
`jacobi_reflect` (`P_n^{(a,b)}(-x) = (-1)^n P_n^{(b,a)}(x)`), `rowNeg_reflect`
(`d^ell_{m,-s}(pi - theta) = (-1)^(ell+m) d^ell_{m,s}(theta)` for the Jacobi closed forms, `s <= m <= ell`) and
`mirror_identity`, the form the kernel uses (`T[m, pi-theta] = (-1)^(ell+s) T[-m, theta]`, with
`d^ell_{-m,-s} = (-1)^(m+s) d^ell_{m,s}` read off the same Jacobi form).

**Lemma.** `d^ell_{m,0}(pi - theta) = (-1)^(ell+m) d^ell_{m,0}(theta)`.
*Proof.* `d^ell_{m,0}(theta) = sqrt((ell-m)!/(ell+m)!) P_ell^m(cos theta)` and
`P_ell^m(-x) = (-1)^(ell+m) P_ell^m(x)` (DLMF 14.7.17). □

With HEALPix rings symmetric (`theta_{N-1-i} = pi - theta_i`) and ring `i` and its partner sharing
weight and `phi_0`, the sum over the sphere in (1) is, per northern lane,
`sum_north d [ G_i + (-1)^(ell+m) G_i' ]`: one recurrence, two right-hand sides, the parity sign
applied to the second fp64 partial outside the kernel.  This halves the lane count for spin 0.

For spin 2 the negative orders come from `d^ell_{-m,-s}(theta) = (-1)^(ell-m')... ` -- the code uses the
measured identity `T[m, pi-theta, ell] = (-1)^(ell - m') T[-m, theta, ell]` with `m' = -s`
(`_spin_slice._windows` docstring, 2.3e-12 over every row), so the mirror channel is the same row
contracted against the ring-reversed map with the sign `(-1)^(ell+s)` applied outside.

## 7. Where the arithmetic wall is (new, session 36)

Lean: `GMasterMarch.Wall` -- `live_degrees` (`sum_{m<L} (L-m) = L(L+1)/2`), `time_lower_bound`.
The rates are measurements (`.qwen/tmp/fp64_peak_s36.py`: 0.45 TFMA/s float64, 30.5 TFMA/s float32
on GPU1 with independent accumulators).

Let `N` be the number of `(m, ell, theta)` triples the march visits: for the folded spin-0 route
`N = (L(L+1)/2) * ceil(ntheta/2)` and for spin 2 `N = (L(L+1)/2) * ntheta` (before any polar skip).

| Nside | L | rings marched (spin 0 / 2) | N spin 0 | N spin 2 |
|---|---|---|---|---|
| 1024 | 3072 | 2048 / 4095 | 9.7e9 | 1.9e10 |
| 2048 | 6144 | 4096 / 8191 | 7.7e10 | 1.5e11 |
| 4096 | 12288 | 8192 / 16383 | 6.2e11 | 1.2e12 |

A three-term recurrence needs at least two multiply-adds per new value and each channel one
multiply-add per triple, so an analysis pass with `nc` channels costs at least `(2 + nc) N` FMAs.
On this card (RTX PRO 6000 Blackwell, float64 at 1/64 rate) the float64 FMA rate measured with
eight independent accumulators is `R_64 = 0.45e12 FMA/s` (`.qwen/tmp/fp64_peak_s36.py`; the
same kernel reaches 30.5e12 FMA/s in float32, a 68:1 ratio), against a datasheet ceiling of
`0.94e12`.  Hence any float64 SIMT implementation of the spin-0 analysis at Nside 4096 needs

    T >= (2 + 4) * 6.2e11 / 0.94e12 s = 4.0 s  (and >= 1.3 s for the recurrence alone) at the datasheet rate,
    T >= 8.3 s (2.8 s for the recurrence alone) at the measured rate,

against ducc0's measured 2.02 s per pass on the 96-core host.  **No float64 kernel on this card can
beat the CPU reference at the top of the board; only float32-class arithmetic can**, which is why the
march is a compensated float32 recurrence: its per-triple cost is ~30 float32 instructions against
a float32 issue rate of `~6e13 lane-instructions/s`, a floor of ~0.3 s per pass at Nside 4096.

The measured analysis kernel after §5 is 22.5 ms per 128-order window at Nside 4096 against a
recurrence-only floor of 13.7 ms, so the emit is now 39 % of the kernel and the compensated
recurrence 61 %.

## 8. Polar skip: the orders a ring never sees (new, session 36)

Lean: `GMasterMarch.PolarSkip` -- `mlim_is_root` (ducc0's closed form is the larger root of the
quadratic it solves), `mlim_ge`, `mlim_spin_zero`, `skipped_degrees`.  The WKB estimate below is not
formalised.

For `m > (ell + 1/2) sin theta` the function `d^ell_{m,-s}(theta)` is in its forbidden region.
Writing `u = sqrt(sin theta) d^ell_{m,0}` the Legendre equation is `u'' + [(ell+1/2)^2 - (m^2-1/4)/sin^2 theta] u = 0`,
so with `l' = ell + 1/2` the WKB decay from the turning point `theta_t` (`sin theta_t = m/l'`) down to
`theta < theta_t` is `exp(-S)` with `S = int_theta^{theta_t} sqrt(m^2/sin^2 t - l'^2) dt`.  Linearising
`m^2/sin^2 t - l'^2 ~ 2 l'^2 cot(theta_t) (theta_t - t)` and `theta_t - theta ~ delta/(l' cos theta)` for
`delta = m - l' sin theta` gives

    S ~ (2 sqrt 2 / 3) delta^(3/2) / ( sqrt(l' sin theta) cos theta ).

ducc0 (`sharp_get_mlim`) neglects, on the ring at `theta`, every order above
`mlim = s |cos theta| + sqrt((lmax sin theta + ofs)^2 - s^2 sin^2 theta)`, `ofs = max(100, 0.01 lmax)`
(for `s = 0`: `m > lmax sin theta + ofs`).  Every `ell <= lmax` then has `delta >= ofs`, so at
`lmax = 12287`, `sin theta = 1/2`: `S >= 0.943 * 123^1.5 / (sqrt(6144) * 0.866) ~ 19`, i.e. neglected
values below `~1e-8` times the oscillatory-region amplitude `~ 1/sqrt(l' sin theta)`; at smaller
`sin theta` the bound is tighter.  The march applies the same rule per `(m, theta tile)` program,
skipping the program when `m > max_tile mlim(theta)`; because the neglected set is the reference's own,
the alm agreement with ducc0 is unchanged to the printed digit (rel 3.76e-05 at Nside 1024 spin 2,
7.20e-05 at 2048 spin 0, with and without the skip).  Work removed (`.qwen/tmp/mlim_count_s36.py`):

| Nside | analysis (tile 256) | synthesis spin 0 (tile 1024) | synthesis spin 2 (tile 512) |
|---|---|---|---|
| 1024 | 13.9 % | 2.5 % | 9.3 % |
| 2048 | 17.5 % | 10.0 % | 14.8 % |
| 4096 | 19.3 % | 15.1 % | 17.9 % |

The lane-exact fraction (skipping every `(m, theta)` lane above its own `mlim`, which a lane-parallel
kernel cannot cash) is `<(1 - sin theta)^2>/2 / <1 - sin theta / 2>` of the triangle, about 21 % for
rings uniform in `cos theta`.
