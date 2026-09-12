import Mathlib
import GMasterMarch.Recurrence

/-!
§3 of `docs/latitudinal_march_maths.md`: the `(value, exponent)` lane representation.

A lane holds `v * 2^ex` with `v : ℝ` (a float32 in the kernel) and `ex : ℤ`.  The renormalisation
moves a power of two between the two fields, which is exact, and the recurrence is linear and
homogeneous in `(v_ℓ, v_{ℓ-1})`, so a common rescaling of both lanes rescales every later value.
-/

namespace GMasterMarch

/-- The value a lane represents. -/
noncomputable def lane (v : ℝ) (ex : ℤ) : ℝ := v * (2 : ℝ) ^ ex

/-- Moving `k` binades from the mantissa to the exponent field changes nothing. -/
theorem lane_rescale (v : ℝ) (ex k : ℤ) : lane (v * (2 : ℝ) ^ (-k)) (ex + k) = lane v ex := by
  unfold lane
  rw [zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0), zpow_neg]
  field_simp

/-- The `2^{-24}` guard step (`large`): mantissa above `2^24` is divided by `2^24`, exponent
raised by `24`; the represented value is unchanged. -/
theorem guard_large (v : ℝ) (ex : ℤ) : lane (v * (2 : ℝ) ^ (-24 : ℤ)) (ex + 24) = lane v ex :=
  lane_rescale v ex 24

/-- The `2^{+24}` guard step (`small`). -/
theorem guard_small (v : ℝ) (ex : ℤ) : lane (v * (2 : ℝ) ^ (24 : ℤ)) (ex - 24) = lane v ex := by
  have h := lane_rescale v ex (-24)
  rw [sub_eq_add_neg]
  simpa using h

/-- One marched step is linear and homogeneous in the two carried lanes: rescaling both by `λ`
rescales the result by `λ`.  This is what lets both lanes of a pair be renormalised together. -/
theorem step_homogeneous (a b p1 p0 lam : ℝ) :
    a * (lam * p1) - b * (lam * p0) = lam * (a * p1 - b * p0) := by ring

/-- A two-term linear recurrence `v_{n+2} = a n · v_{n+1} - b n · v_n` from arbitrary seeds. -/
noncomputable def rec2 (a b : ℕ → ℝ) (s0 s1 : ℝ) : ℕ → ℝ
  | 0 => s0
  | 1 => s1
  | n + 2 => a n * rec2 a b s0 s1 (n + 1) - b n * rec2 a b s0 s1 n

/-- **Homogeneity.** The march started from seeds scaled by `λ` is the march scaled by `λ` at
every degree: this is why one common rescaling of the `(v_ℓ, v_{ℓ-1})` pair is exact for the
whole rest of the row. -/
theorem rec2_scaled (a b : ℕ → ℝ) (s0 s1 lam : ℝ) :
    ∀ n, rec2 a b (lam * s0) (lam * s1) n = lam * rec2 a b s0 s1 n := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    match n with
    | 0 => simp [rec2]
    | 1 => simp [rec2]
    | n + 2 =>
      simp only [rec2]
      rw [ih (n + 1) (by omega), ih n (by omega)]
      ring

/-- The kernel's march (`Recurrence.march`) is `rec2` with the §2 coefficients and seeds
`1`, `P_1`, so its homogeneity is the theorem above. -/
theorem march_eq_rec2 (α β : ℕ) (x : ℝ) : ∀ n,
    march α β x n = rec2 (fun n => c1 (n + 2) α β * x + c0 (n + 2) α β)
      (fun n => cb (n + 2) α β) 1 (((α + β + 2) / 2) * x + ((α : ℝ) - β) / 2) n := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    match n with
    | 0 => rfl
    | 1 => rfl
    | n + 2 =>
      simp only [march, rec2]
      rw [ih (n + 1) (by omega), ih n (by omega)]

/-- The march from seeds `λ`, `λ P_1` is `λ ×` the march. -/
theorem march_scaled (α β : ℕ) (x lam : ℝ) (n : ℕ) :
    rec2 (fun n => c1 (n + 2) α β * x + c0 (n + 2) α β) (fun n => cb (n + 2) α β)
      (lam * 1) (lam * (((α + β + 2) / 2) * x + ((α : ℝ) - β) / 2)) n
      = lam * march α β x n := by
  rw [rec2_scaled, march_eq_rec2]

/-- The guard keeps the mantissa inside `[2^-24, 2^24]` after a step whose growth is below
`2^100`: a mantissa in the band multiplied by a factor `g ≤ 2^100` stays below `2^124 < 2^126`,
so no overflow occurs before the guard fires. -/
theorem no_overflow_before_guard (v g : ℝ) (hv : |v| ≤ (2 : ℝ) ^ (24 : ℤ))
    (hg : |g| ≤ (2 : ℝ) ^ (100 : ℤ)) : |v * g| ≤ (2 : ℝ) ^ (124 : ℤ) := by
  rw [abs_mul]
  calc |v| * |g| ≤ (2 : ℝ) ^ (24 : ℤ) * (2 : ℝ) ^ (100 : ℤ) :=
        mul_le_mul hv hg (abs_nonneg _) (by positivity)
    _ = (2 : ℝ) ^ (124 : ℤ) := by rw [← zpow_add₀ (by norm_num)]; norm_num

end GMasterMarch
