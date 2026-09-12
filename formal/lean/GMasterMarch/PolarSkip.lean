import Mathlib
import GMasterMarch.Wall

/-!
§8 of `docs/latitudinal_march_maths.md`: the polar skip.

Two exact statements.  `mlim = s|c| + sqrt(t₁² - s² sθ²)` is the larger root of
`m² - 2 s |c| m + s² - t₁² = 0` on a ring with `c² + sθ² = 1` (this is what ducc0's
`sharp_get_mlim` solves), and the degrees a cutoff `M` removes from a ring are
`Σ_{M ≤ m < L} (L - m) = (L - M)(L - M + 1)/2`.  The WKB decay estimate behind the choice of
`ofs` is asymptotic analysis and is *not* formalised; it is checked numerically (the alm agreement
with ducc0 is unchanged to the printed digit with the skip on and off).
-/

namespace GMasterMarch

open Finset

/-- ducc0's `mlim`: `s |c| + sqrt(t₁² - s² sθ²)`. -/
noncomputable def mlim (s c sθ t1 : ℝ) : ℝ := s * |c| + Real.sqrt (t1 ^ 2 - s ^ 2 * sθ ^ 2)

/-- `mlim` solves the quadratic `m² - 2 s |c| m + s² - t₁² = 0` when `c² + sθ² = 1` and the
discriminant is non-negative. -/
theorem mlim_is_root (s c sθ t1 : ℝ) (hring : c ^ 2 + sθ ^ 2 = 1)
    (hdisc : 0 ≤ t1 ^ 2 - s ^ 2 * sθ ^ 2) :
    mlim s c sθ t1 ^ 2 - 2 * s * |c| * mlim s c sθ t1 + s ^ 2 - t1 ^ 2 = 0 := by
  unfold mlim
  have hsq := Real.sq_sqrt hdisc
  have hc : |c| ^ 2 = c ^ 2 := sq_abs c
  nlinarith [hsq, hc, hring]

/-- It is the larger root: `mlim ≥ s|c|`. -/
theorem mlim_ge (s c sθ t1 : ℝ) : s * |c| ≤ mlim s c sθ t1 := by
  unfold mlim
  linarith [Real.sqrt_nonneg (t1 ^ 2 - s ^ 2 * sθ ^ 2)]

/-- For spin 0 the rule is `m > lmax sin θ + ofs`. -/
theorem mlim_spin_zero (c sθ t1 : ℝ) (ht : 0 ≤ t1) : mlim 0 c sθ t1 = t1 := by
  unfold mlim
  simp [Real.sqrt_sq ht]

/-- Degrees removed from one ring by skipping every order `m ≥ M`: `(L-M)(L-M+1)/2`. -/
theorem skipped_degrees (L M : ℕ) (h : M ≤ L) :
    ∑ m ∈ Ico M L, (L - m) = (L - M) * (L - M + 1) / 2 := by
  rw [sum_Ico_eq_sum_range]
  have e : ∑ k ∈ range (L - M), (L - (M + k)) = ∑ k ∈ range (L - M), (L - M - k) := by
    apply sum_congr rfl
    intro k _
    omega
  rw [e]
  exact live_degrees (L - M)

end GMasterMarch
