/-
§6 of `docs/latitudinal_march_maths.md`, spin-2 half: the mirror channel.

The kernel contracts one marched row against the ring-reversed map and applies `(-1)^(ℓ+s)`,
relying on `T[m, π-θ, ℓ] = (-1)^(ℓ+s) T[-m, θ, ℓ]` (the `_spin_slice._windows` docstring records
it as measured to 2.3e-12).  Here it is derived for the Jacobi closed forms of §1: the general
reflection `P_n^{(α,β)}(-x) = (-1)^n P_n^{(β,α)}(x)` swaps the two half-angle powers, which is
exactly the swap between `d^ℓ_{m,-s}` and `d^ℓ_{m,s}` (or `d^ℓ_{-m,-s}`) in the Jacobi form.
-/
import Mathlib
import GMasterMarch.Fold

namespace GMasterMarch

open Finset

/-- Reflection: `P_n^{(α,β)}(-x) = (-1)^n P_n^{(β,α)}(x)`. -/
theorem jacobi_reflect (n α β : ℕ) (x : ℝ) :
    jacobi n α β (-x) = (-1) ^ n * jacobi n β α x := by
  unfold jacobi
  rw [mul_sum]
  conv_rhs => rw [← sum_range_reflect]
  apply sum_congr rfl
  intro k hk
  have hk' : k ≤ n := Nat.lt_succ_iff.mp (mem_range.mp hk)
  have e1 : n + 1 - 1 - k = n - k := by omega
  have e2 : n - (n - k) = k := by omega
  rw [e1, e2]
  have hx1 : ((-x - 1) / 2) = -((x + 1) / 2) := by ring
  have hx2 : ((-x + 1) / 2) = -((x - 1) / 2) := by ring
  have key : ((-x - 1) / 2) ^ k * ((-x + 1) / 2) ^ (n - k)
      = (-1) ^ n * (((x - 1) / 2) ^ (n - k) * ((x + 1) / 2) ^ k) := by
    rw [hx1, hx2, neg_eq_neg_one_mul ((x + 1) / 2), neg_eq_neg_one_mul ((x - 1) / 2), mul_pow,
      mul_pow]
    have hs : (-1 : ℝ) ^ k * (-1) ^ (n - k) = (-1) ^ n := by
      rw [← pow_add]; congr 1; omega
    linear_combination (((x - 1) / 2) ^ (n - k) * ((x + 1) / 2) ^ k) * hs
  rw [mul_assoc, key]
  ring

/-- The `θ`-dependence of the Jacobi form of `d^ℓ_{m,-s}` for `s ≤ m ≤ ℓ` (§1 with `α = m+s`,
`β = m-s`, `n = ℓ-m`), up to the common factorial normalisation and the sign `(-1)^(m+s)`. -/
noncomputable def rowNeg (ℓ m s : ℕ) (θ : ℝ) : ℝ :=
  Real.sin (θ / 2) ^ (m + s) * Real.cos (θ / 2) ^ (m - s) * jacobi (ℓ - m) (m + s) (m - s) (Real.cos θ)

/-- The same for `d^ℓ_{m,s}`: the general Jacobi form with `(m', m) = (m, s)` has `a = m-s`,
`b = m+s`, `n = ℓ-m`, the same normalisation and the same sign `(-1)^(m-s) = (-1)^(m+s)` for
`s ≤ m`.  It is also the Jacobi form of `d^ℓ_{-m,-s}` (with `(m', m) = (-m, -s)`: `a = m-s`,
`b = m+s`, sign `+1`), so `d^ℓ_{-m,-s} = (-1)^(m+s) d^ℓ_{m,s}`. -/
noncomputable def rowPos (ℓ m s : ℕ) (θ : ℝ) : ℝ :=
  Real.sin (θ / 2) ^ (m - s) * Real.cos (θ / 2) ^ (m + s) * jacobi (ℓ - m) (m - s) (m + s) (Real.cos θ)

/-- **Mirror identity**, closed-form level: `d^ℓ_{m,-s}(π - θ) = (-1)^(ℓ+m) d^ℓ_{m,s}(θ)` for
`s ≤ m ≤ ℓ` — the rows share their normalisation and sign, so it is the row identity below. -/
theorem rowNeg_reflect (ℓ m s : ℕ) (hsm : s ≤ m) (hℓ : m ≤ ℓ) (θ : ℝ) :
    rowNeg ℓ m s (Real.pi - θ) = (-1) ^ (ℓ + m) * rowPos ℓ m s θ := by
  unfold rowNeg rowPos
  have hs : Real.sin ((Real.pi - θ) / 2) = Real.cos (θ / 2) := by
    rw [show (Real.pi - θ) / 2 = Real.pi / 2 - θ / 2 by ring, Real.sin_pi_div_two_sub]
  have hc : Real.cos ((Real.pi - θ) / 2) = Real.sin (θ / 2) := by
    rw [show (Real.pi - θ) / 2 = Real.pi / 2 - θ / 2 by ring, Real.cos_pi_div_two_sub]
  rw [hs, hc, Real.cos_pi_sub, jacobi_reflect, neg_one_pow_sub_eq_add hℓ]
  ring

/-- The form the kernel uses: with `T[-m, θ] = d^ℓ_{-m,-s}(θ) = (-1)^(m+s) d^ℓ_{m,s}(θ)`,
`T[m, π-θ] = (-1)^(ℓ+s) T[-m, θ]`.  Stated for the rows with their signs:
`(-1)^(m+s) rowNeg (π-θ) = (-1)^(ℓ+s) rowPos θ`. -/
theorem mirror_identity (ℓ m s : ℕ) (_hsm : s ≤ m) (hℓ : m ≤ ℓ) (θ : ℝ) :
    (-1) ^ (m + s) * rowNeg ℓ m s (Real.pi - θ) = (-1) ^ (ℓ + s) * rowPos ℓ m s θ := by
  rw [rowNeg_reflect ℓ m s _hsm hℓ θ]
  have : ((-1 : ℝ)) ^ (m + s) * (-1) ^ (ℓ + m) = (-1) ^ (ℓ + s) := by
    rw [← pow_add, show m + s + (ℓ + m) = (ℓ + s) + 2 * m by ring, pow_add, pow_mul]; norm_num
  rw [← mul_assoc, this]

end GMasterMarch
