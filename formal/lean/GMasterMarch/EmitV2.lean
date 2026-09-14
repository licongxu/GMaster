import Mathlib
import GMasterMarch.DiffForm

/-!
§4 of `docs/march_v2_maths.md`: the v2 emit.

At degree `ℓ` in a block with base `b` the kernel emits `v · 2^(ex + ⌊log₂ N(b)⌋) · u(ℓ)` with
`u(ℓ) = 2^(log₂ N(ℓ) - ⌊log₂ N(b)⌋)` from the table.  The factorisation is an identity of real
numbers; `u` is a bounded float32 mantissa because `log₂ N(ℓ) - log₂ N(b)` is a sum of at most
seven per-degree steps `½ log₂ ((ℓ'+m)(ℓ'-m)/((ℓ'-s)(ℓ'+s)))`, each in `[-6.5, 0]`, plus the
fractional part of `log₂ N(b)`.
-/

namespace GMasterMarch

/-- **`emit_factorisation`.** `v · 2^ex · 2^g = v · 2^(ex + lb) · 2^(g - lb)` for integer `ex`,
`lb` (the block's `⌊log₂ N(b)⌋`) and real `g` (`log₂ N(ℓ)`). -/
theorem emit_factorisation (v : ℝ) (ex lb : ℤ) (g : ℝ) :
    v * (2 : ℝ) ^ ex * (2 : ℝ) ^ g = v * (2 : ℝ) ^ (ex + lb) * (2 : ℝ) ^ (g - lb) := by
  rw [zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
  have h : (2 : ℝ) ^ lb * (2 : ℝ) ^ (g - lb) = (2 : ℝ) ^ g := by
    rw [← Real.rpow_intCast 2 lb, ← Real.rpow_add (by norm_num : (0 : ℝ) < 2)]
    congr 1; ring
  calc v * (2 : ℝ) ^ ex * (2 : ℝ) ^ g
      = v * (2 : ℝ) ^ ex * ((2 : ℝ) ^ lb * (2 : ℝ) ^ (g - lb)) := by rw [h]
    _ = v * ((2 : ℝ) ^ ex * (2 : ℝ) ^ lb) * (2 : ℝ) ^ (g - lb) := by ring

/-- The kernel stores the biased field `lexp = lb + 127` and builds `2^(e - 127)` from
`e = ex + lexp` by exponent-field assembly: the same power of two. -/
theorem biased_exponent (ex lb : ℤ) :
    (2 : ℝ) ^ (ex + (lb + 127) - 127) = (2 : ℝ) ^ (ex + lb) := by
  congr 1; ring

/-- A real exponent in `[-46, 1)` gives a factor in `[2^-46, 2)`: `u(ℓ)` is a normal float32. -/
theorem u_range (e : ℝ) (h1 : -46 ≤ e) (h2 : e < 1) :
    (2 : ℝ) ^ (-46 : ℝ) ≤ (2 : ℝ) ^ e ∧ (2 : ℝ) ^ e < 2 := by
  constructor
  · exact Real.rpow_le_rpow_of_exponent_le (by norm_num) h1
  · calc (2 : ℝ) ^ e < (2 : ℝ) ^ (1 : ℝ) := Real.rpow_lt_rpow_of_exponent_lt (by norm_num) h2
      _ = 2 := by simp

/-- The seven-step sum: `t_i ∈ [-13/2, 0]` for `i < 7` and `f ∈ [0, 1)` give
`Σ t_i + f ∈ [-91/2, 1)`, inside the `[-46, 1)` of `u_range`. -/
theorem seven_steps (t : ℕ → ℝ) (f : ℝ) (ht : ∀ i, i < 7 → -(13 / 2 : ℝ) ≤ t i ∧ t i ≤ 0)
    (hf0 : 0 ≤ f) (hf1 : f < 1) :
    -46 ≤ (∑ i ∈ Finset.range 7, t i) + f ∧ (∑ i ∈ Finset.range 7, t i) + f < 1 := by
  have hlo : -(91 / 2 : ℝ) ≤ ∑ i ∈ Finset.range 7, t i := by
    have : ∑ i ∈ Finset.range 7, (-(13 / 2 : ℝ)) ≤ ∑ i ∈ Finset.range 7, t i :=
      Finset.sum_le_sum fun i hi => (ht i (Finset.mem_range.mp hi)).1
    simp at this; linarith
  have hhi : ∑ i ∈ Finset.range 7, t i ≤ 0 := by
    have : ∑ i ∈ Finset.range 7, t i ≤ ∑ i ∈ Finset.range 7, (0 : ℝ) :=
      Finset.sum_le_sum fun i hi => (ht i (Finset.mem_range.mp hi)).2
    simpa using this
  constructor <;> linarith

/-- **Per-step ratio, `m ≥ s` branch.** For `s ≤ m` and `ℓ ≥ m + 1`,
`2/(m+2) ≤ (ℓ+m)(ℓ-m)/((ℓ-s)(ℓ+s)) ≤ 1` (the ratio increases with `ℓ` and is smallest at
`ℓ = m + 1`). -/
theorem step_ratio_bounds_ge (m s ℓ : ℕ) (hsm : s ≤ m) (hℓ : m + 1 ≤ ℓ) :
    2 / ((m : ℝ) + 2) ≤ (((ℓ : ℝ) + m) * ((ℓ : ℝ) - m)) / (((ℓ : ℝ) - s) * ((ℓ : ℝ) + s))
      ∧ (((ℓ : ℝ) + m) * ((ℓ : ℝ) - m)) / (((ℓ : ℝ) - s) * ((ℓ : ℝ) + s)) ≤ 1 := by
  have hs : (s : ℝ) ≤ m := by exact_mod_cast hsm
  have hl : (m : ℝ) + 1 ≤ ℓ := by exact_mod_cast hℓ
  have hs0 : (0 : ℝ) ≤ s := Nat.cast_nonneg s
  have hden : 0 < ((ℓ : ℝ) - s) * ((ℓ : ℝ) + s) := by
    apply mul_pos <;> linarith
  constructor
  · rw [div_le_div_iff₀ (by linarith) hden]
    nlinarith [mul_nonneg hs0 hs0, sq_nonneg ((ℓ : ℝ) - m - 1), mul_nonneg (Nat.cast_nonneg m) (by linarith : (0 : ℝ) ≤ ℓ - m - 1),
      mul_nonneg (Nat.cast_nonneg m) (by linarith : (0 : ℝ) ≤ ℓ + m + 1)]
  · rw [div_le_one hden]
    nlinarith [mul_nonneg hs0 hs0]

/-- **Per-step ratio, `m < s` branch** (`ε_m = -1`, the rows `m < s = 2`).  For `m < s` and
`ℓ ≥ s + 1` the inverted ratio satisfies `1 ≤ (ℓ+m)(ℓ-m)/((ℓ-s)(ℓ+s)) ≤ (s+1)²/(2s+1)`
(at `s = 2`: `9/5`), so the step `-½ log₂` of it lies in `[-½ log₂ (9/5), 0] ⊂ [-1/2, 0]`. -/
theorem step_ratio_bounds_lt (m s ℓ : ℕ) (hms : m < s) (hℓ : s + 1 ≤ ℓ) :
    1 ≤ (((ℓ : ℝ) + m) * ((ℓ : ℝ) - m)) / (((ℓ : ℝ) - s) * ((ℓ : ℝ) + s))
      ∧ (((ℓ : ℝ) + m) * ((ℓ : ℝ) - m)) / (((ℓ : ℝ) - s) * ((ℓ : ℝ) + s))
          ≤ ((s : ℝ) + 1) ^ 2 / (2 * s + 1) := by
  have hm : (m : ℝ) < s := by exact_mod_cast hms
  have hl : (s : ℝ) + 1 ≤ ℓ := by exact_mod_cast hℓ
  have hm0 : (0 : ℝ) ≤ m := Nat.cast_nonneg m
  have hden : 0 < ((ℓ : ℝ) - s) * ((ℓ : ℝ) + s) := by
    apply mul_pos <;> linarith
  constructor
  · rw [one_le_div hden]
    nlinarith
  · rw [div_le_div_iff₀ hden (by linarith)]
    nlinarith [mul_nonneg hm0 hm0, mul_nonneg (mul_nonneg hm0 hm0) (by linarith : (0 : ℝ) ≤ 2 * s + 1),
      mul_nonneg (by linarith : (0 : ℝ) ≤ s) (by linarith : (0 : ℝ) ≤ s),
      mul_nonneg (by linarith : (0 : ℝ) ≤ ℓ - s - 1) (by linarith : (0 : ℝ) ≤ ℓ + s + 1),
      mul_nonneg (mul_nonneg (by linarith : (0 : ℝ) ≤ s) (by linarith : (0 : ℝ) ≤ s))
        (mul_nonneg (by linarith : (0 : ℝ) ≤ ℓ - s - 1) (by linarith : (0 : ℝ) ≤ ℓ + s + 1))]

/-- **Step bound.** In the `m ≥ s` branch with `m + 2 ≤ 2^14` the per-degree step
`½ log₂ ((ℓ+m)(ℓ-m)/((ℓ-s)(ℓ+s)))` lies in `[-13/2, 0]`. -/
theorem step_bound_ge (m s ℓ : ℕ) (hsm : s ≤ m) (hℓ : m + 1 ≤ ℓ) (hm : (m : ℝ) + 2 ≤ 2 ^ 14) :
    -(13 / 2 : ℝ) ≤ (1 / 2) * Real.logb 2 ((((ℓ : ℝ) + m) * ((ℓ : ℝ) - m)) / (((ℓ : ℝ) - s) * ((ℓ : ℝ) + s)))
      ∧ (1 / 2) * Real.logb 2 ((((ℓ : ℝ) + m) * ((ℓ : ℝ) - m)) / (((ℓ : ℝ) - s) * ((ℓ : ℝ) + s))) ≤ 0 := by
  obtain ⟨hlo, hhi⟩ := step_ratio_bounds_ge m s ℓ hsm hℓ
  set r := (((ℓ : ℝ) + m) * ((ℓ : ℝ) - m)) / (((ℓ : ℝ) - s) * ((ℓ : ℝ) + s)) with hr
  have hm2 : (0 : ℝ) < (m : ℝ) + 2 := by positivity
  have hq : (0 : ℝ) < 2 / ((m : ℝ) + 2) := by positivity
  have hr0 : 0 < r := lt_of_lt_of_le hq hlo
  constructor
  · have h1 : Real.logb 2 (2 / ((m : ℝ) + 2)) ≤ Real.logb 2 r :=
      Real.logb_le_logb_of_le (by norm_num) hq hlo
    have h2 : Real.logb 2 (2 / ((m : ℝ) + 2)) = 1 - Real.logb 2 ((m : ℝ) + 2) := by
      rw [Real.logb_div (by norm_num) hm2.ne', Real.logb_self_eq_one (by norm_num)]
    have h3 : Real.logb 2 ((m : ℝ) + 2) ≤ 14 := by
      calc Real.logb 2 ((m : ℝ) + 2) ≤ Real.logb 2 ((2 : ℝ) ^ 14) :=
            Real.logb_le_logb_of_le (by norm_num) hm2 hm
        _ = 14 := by
            rw [Real.logb_pow, Real.logb_self_eq_one (by norm_num)]; norm_num
    linarith
  · have : Real.logb 2 r ≤ Real.logb 2 1 := Real.logb_le_logb_of_le (by norm_num) hr0 hhi
    rw [Real.logb_one] at this
    linarith

end GMasterMarch
