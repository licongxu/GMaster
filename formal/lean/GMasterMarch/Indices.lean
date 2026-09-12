/-
§1 of `docs/latitudinal_march_maths.md`: the index bookkeeping of the Jacobi closed form of
`d^ℓ_{m,-s}`.  Everything here is exact arithmetic on naturals, rationals and signs.

The classical identity (Edmonds 4.1.23) writes `d^j_{m'm}` with `a = |m - m'|`, `b = |m + m'|`,
`n = j - (a+b)/2`, a factorial ratio `n! (n+a+b)! / ((n+a)! (n+b)!)` and a sign
`ξ = 1` if `m ≥ m'` else `(-1)^(m-m')`.  Substituting `m' ↦ m`, `m ↦ -s` gives the kernel's
`α = m + s`, `β = |m - s|`, `n = ℓ - max m s`, and the two factorial forms below.
-/
import Mathlib

namespace GMasterMarch

/-- `α = m + s`. -/
def alpha (m s : ℕ) : ℕ := m + s

/-- `β = |m - s|`, written without integer subtraction. -/
def beta (m s : ℕ) : ℕ := max m s - min m s

theorem beta_eq_natAbs (m s : ℕ) : beta m s = ((m : ℤ) - s).natAbs := by
  unfold beta; omega

/-- `α + β = 2 max(m, s)`: the Jacobi degree `n = ℓ - (α+β)/2` is `ℓ - max(m,s)`. -/
theorem alpha_add_beta (m s : ℕ) : alpha m s + beta m s = 2 * max m s := by
  unfold alpha beta; omega

/-- `α - β = 2 min(m, s)`. -/
theorem alpha_sub_beta (m s : ℕ) : alpha m s - beta m s = 2 * min m s := by
  unfold alpha beta; omega

theorem beta_le_alpha (m s : ℕ) : beta m s ≤ alpha m s := by
  unfold alpha beta; omega

/-- **Lemma (no cancellation in `c0`)**: `α² - β² = 4 m s` for every `m, s`. -/
theorem alpha_sq_sub_beta_sq (m s : ℕ) :
    ((alpha m s : ℤ)) ^ 2 - ((beta m s : ℤ)) ^ 2 = 4 * m * s := by
  have h1 : ((alpha m s : ℤ)) ^ 2 - ((beta m s : ℤ)) ^ 2
      = ((alpha m s : ℤ) - beta m s) * ((alpha m s : ℤ) + beta m s) := by ring
  have h2 : ((alpha m s : ℤ) - beta m s) = 2 * min m s := by
    have := alpha_sub_beta m s
    have := beta_le_alpha m s
    omega
  have h3 : ((alpha m s : ℤ) + beta m s) = 2 * max m s := by
    have := alpha_add_beta m s
    omega
  rw [h1, h2, h3]
  rcases le_total m s with h | h
  · rw [min_eq_left h, max_eq_right h]; ring
  · rw [min_eq_right h, max_eq_left h]; ring

/-- The real-valued form used by the kernel's coefficient builder. -/
theorem alpha_sq_sub_beta_sq_real (m s : ℕ) :
    ((alpha m s : ℝ)) ^ 2 - ((beta m s : ℝ)) ^ 2 = 4 * m * s := by
  have := alpha_sq_sub_beta_sq m s
  exact_mod_cast this

/-- Jacobi degree `n = ℓ - max(m, s)`. -/
def jdeg (ℓ m s : ℕ) : ℕ := ℓ - max m s

/-- `n + α + β = ℓ + max(m, s)`. -/
theorem jdeg_add_alpha_add_beta {ℓ m s : ℕ} (h : max m s ≤ ℓ) :
    jdeg ℓ m s + alpha m s + beta m s = ℓ + max m s := by
  unfold jdeg alpha beta; omega

/-- The factorial ratio of the Jacobi form of the Wigner function, `n! (n+α+β)! / ((n+α)! (n+β)!)`. -/
noncomputable def jratio (ℓ m s : ℕ) : ℚ :=
  ((jdeg ℓ m s).factorial * (jdeg ℓ m s + alpha m s + beta m s).factorial : ℚ)
    / ((jdeg ℓ m s + alpha m s).factorial * (jdeg ℓ m s + beta m s).factorial)

/-- The `m ≥ s` branch: the ratio is `(ℓ-m)! (ℓ+m)! / ((ℓ+s)! (ℓ-s)!)`, i.e. `2^(+2·LG2N)`. -/
theorem jratio_of_ge {ℓ m s : ℕ} (hsm : s ≤ m) (hℓ : m ≤ ℓ) :
    jratio ℓ m s = ((ℓ - m).factorial * (ℓ + m).factorial : ℚ)
      / ((ℓ + s).factorial * (ℓ - s).factorial) := by
  unfold jratio
  have e1 : jdeg ℓ m s = ℓ - m := by unfold jdeg; omega
  have e2 : jdeg ℓ m s + alpha m s + beta m s = ℓ + m := by unfold jdeg alpha beta; omega
  have e3 : jdeg ℓ m s + alpha m s = ℓ + s := by unfold jdeg alpha; omega
  have e4 : jdeg ℓ m s + beta m s = ℓ - s := by unfold jdeg beta; omega
  rw [e2, e3, e4, e1]

/-- The `m < s` branch: the ratio is the reciprocal of the `m ≥ s` form, which is the
`ε_m = -1` flip of the exponent `LG2N`. -/
theorem jratio_of_lt {ℓ m s : ℕ} (hms : m < s) (hℓ : s ≤ ℓ) :
    jratio ℓ m s = (((ℓ - m).factorial * (ℓ + m).factorial : ℚ)
      / ((ℓ + s).factorial * (ℓ - s).factorial))⁻¹ := by
  unfold jratio
  have e1 : jdeg ℓ m s = ℓ - s := by unfold jdeg; omega
  have e2 : jdeg ℓ m s + alpha m s + beta m s = ℓ + s := by unfold jdeg alpha beta; omega
  have e3 : jdeg ℓ m s + alpha m s = ℓ + m := by unfold jdeg alpha; omega
  have e4 : jdeg ℓ m s + beta m s = ℓ - m := by unfold jdeg beta; omega
  rw [e2, e3, e4, e1, inv_div]
  ring

/-- The sign of the Jacobi form is `(-1)^(m+s)`; the kernel writes `(-1)^m`, which agrees for
even spin (0 and 2), the only spins the march serves. -/
theorem sign_of_even_spin (m s : ℕ) (hs : Even s) : ((-1 : ℤ)) ^ (m + s) = (-1) ^ m := by
  rw [pow_add, hs.neg_one_pow, mul_one]

/-- For odd spin the two signs differ by `-1`: a spin-1 or spin-3 march must restore it. -/
theorem sign_of_odd_spin (m s : ℕ) (hs : Odd s) : ((-1 : ℤ)) ^ (m + s) = -((-1) ^ m) := by
  rw [pow_add, hs.neg_one_pow, mul_neg, mul_one]

end GMasterMarch
