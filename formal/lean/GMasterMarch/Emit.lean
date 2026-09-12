import Mathlib
import GMasterMarch.ExponentCarry

/-!
§4 and §5 of `docs/latitudinal_march_maths.md`: the emit.

§4: lanes more than 126 binades below the tile maximum are flushed; their total is below `2^-70`
of the largest lane's term.  §5: the binade split `2^(g + emax) = 2^{g - ⌊g⌋} · 2^{⌊g⌋ + emax}`
is an exact real identity, the fractional factor lies in `[1, 2)`, and two roundings of relative
size `u` change a value by at most `(1+u)^2 - 1` relatively.
-/

namespace GMasterMarch

open Finset

/-- A flushed lane (`ex ≤ emax - 127`, `|v| ≤ 2^24`) is below `2^-79` of a lane at the tile
maximum with the smallest mantissa the band allows (`|v| ≥ 2^-24`). -/
theorem flushed_lane_small (v : ℝ) (ex emax : ℤ) (hv : |v| ≤ (2 : ℝ) ^ (24 : ℤ))
    (hex : ex ≤ emax - 127) :
    |lane v ex| ≤ (2 : ℝ) ^ (-79 : ℤ) * ((2 : ℝ) ^ (-24 : ℤ) * (2 : ℝ) ^ emax) := by
  unfold lane
  rw [abs_mul, abs_of_pos (zpow_pos (by norm_num : (0:ℝ) < 2) ex)]
  have h2 : (2 : ℝ) ^ ex ≤ (2 : ℝ) ^ (emax - 127) :=
    zpow_le_zpow_right₀ (by norm_num : (1:ℝ) ≤ 2) hex
  calc |v| * (2 : ℝ) ^ ex ≤ (2 : ℝ) ^ (24 : ℤ) * (2 : ℝ) ^ (emax - 127) :=
        mul_le_mul hv h2 (by positivity) (by positivity)
    _ = (2 : ℝ) ^ (-79 : ℤ) * ((2 : ℝ) ^ (-24 : ℤ) * (2 : ℝ) ^ emax) := by
        rw [← zpow_add₀ (by norm_num : (2:ℝ) ≠ 0), ← zpow_add₀ (by norm_num : (2:ℝ) ≠ 0),
          ← zpow_add₀ (by norm_num : (2:ℝ) ≠ 0)]
        congr 1; ring

/-- **Flush bound.** Over at most `256` flushed lanes with right-hand sides bounded by `R`, the
neglected sum is at most `2^-71 · R` times the smallest possible top-lane term `2^-24 · 2^emax`. -/
theorem flush_bound {ι : Type*} (s : Finset ι) (hs : s.card ≤ 256)
    (v : ι → ℝ) (ex : ι → ℤ) (r : ι → ℝ) (emax : ℤ) (R : ℝ) (hR : 0 ≤ R)
    (hv : ∀ i ∈ s, |v i| ≤ (2 : ℝ) ^ (24 : ℤ))
    (hex : ∀ i ∈ s, ex i ≤ emax - 127)
    (hr : ∀ i ∈ s, |r i| ≤ R) :
    |∑ i ∈ s, lane (v i) (ex i) * r i|
      ≤ (2 : ℝ) ^ (-71 : ℤ) * R * ((2 : ℝ) ^ (-24 : ℤ) * (2 : ℝ) ^ emax) := by
  calc |∑ i ∈ s, lane (v i) (ex i) * r i|
      ≤ ∑ i ∈ s, |lane (v i) (ex i) * r i| := abs_sum_le_sum_abs _ _
    _ ≤ ∑ i ∈ s, (2 : ℝ) ^ (-79 : ℤ) * ((2 : ℝ) ^ (-24 : ℤ) * (2 : ℝ) ^ emax) * R := by
        apply sum_le_sum
        intro i hi
        rw [abs_mul]
        exact mul_le_mul (flushed_lane_small _ _ _ (hv i hi) (hex i hi)) (hr i hi)
          (abs_nonneg _) (by positivity)
    _ = s.card * ((2 : ℝ) ^ (-79 : ℤ) * ((2 : ℝ) ^ (-24 : ℤ) * (2 : ℝ) ^ emax) * R) := by
        rw [sum_const, nsmul_eq_mul]
    _ ≤ 256 * ((2 : ℝ) ^ (-79 : ℤ) * ((2 : ℝ) ^ (-24 : ℤ) * (2 : ℝ) ^ emax) * R) := by
        apply mul_le_mul_of_nonneg_right _ (by positivity)
        exact_mod_cast hs
    _ = (2 : ℝ) ^ (-71 : ℤ) * R * ((2 : ℝ) ^ (-24 : ℤ) * (2 : ℝ) ^ emax) := by
        have : (256 : ℝ) = (2 : ℝ) ^ (8 : ℤ) := by norm_num
        rw [this, ← mul_assoc, ← mul_assoc, ← zpow_add₀ (by norm_num : (2:ℝ) ≠ 0)]
        norm_num; ring

/-- **Binade split.** For real `g` and integer `emax`,
`2^(g + emax) = 2^(g - ⌊g⌋) · 2^(⌊g⌋ + emax)`, with the second factor an integer power of two. -/
theorem binade_split (g : ℝ) (emax : ℤ) :
    (2 : ℝ) ^ (g + emax) = (2 : ℝ) ^ (g - ⌊g⌋) * (2 : ℝ) ^ ((⌊g⌋ + emax : ℤ) : ℝ) := by
  rw [← Real.rpow_add (by norm_num : (0:ℝ) < 2)]
  congr 1; push_cast; ring

/-- The fractional factor is in `[1, 2)`: it is a float32 mantissa. -/
theorem frac_factor_bounds (g : ℝ) :
    1 ≤ (2 : ℝ) ^ (g - ⌊g⌋) ∧ (2 : ℝ) ^ (g - ⌊g⌋) < 2 := by
  constructor
  · have h0 : (0 : ℝ) ≤ g - ⌊g⌋ := by linarith [Int.floor_le g]
    calc (1 : ℝ) = (2 : ℝ) ^ (0 : ℝ) := by simp
      _ ≤ (2 : ℝ) ^ (g - ⌊g⌋) := Real.rpow_le_rpow_of_exponent_le (by norm_num) h0
  · have h1 : g - ⌊g⌋ < 1 := by linarith [Int.lt_floor_add_one g]
    calc (2 : ℝ) ^ (g - ⌊g⌋) < (2 : ℝ) ^ (1 : ℝ) :=
          Real.rpow_lt_rpow_of_exponent_lt (by norm_num) h1
      _ = 2 := by simp

/-- The integer power is exactly representable and multiplying by it is exact in the reals;
in the kernel's driver this is `_pow2_f64_xla`, a bit pattern. -/
theorem int_binade_is_zpow (e : ℤ) : (2 : ℝ) ^ ((e : ℤ) : ℝ) = (2 : ℝ) ^ e :=
  Real.rpow_intCast 2 e

/-- **Two roundings.** If `p = S (1+δ₁)(1+δ₂)` with `|δᵢ| ≤ u`, then `|p - S| ≤ ((1+u)^2 - 1) |S|`. -/
theorem two_roundings (S p d1 d2 u : ℝ) (hu : 0 ≤ u) (h1 : |d1| ≤ u) (h2 : |d2| ≤ u)
    (hp : p = S * (1 + d1) * (1 + d2)) : |p - S| ≤ ((1 + u) ^ 2 - 1) * |S| := by
  have : p - S = S * ((1 + d1) * (1 + d2) - 1) := by rw [hp]; ring
  rw [this, abs_mul, mul_comm]
  apply mul_le_mul_of_nonneg_right _ (abs_nonneg S)
  have e : (1 + d1) * (1 + d2) - 1 = d1 + d2 + d1 * d2 := by ring
  rw [e]
  have hd1 := abs_le.mp h1
  have hd2 := abs_le.mp h2
  have hprod : |d1 * d2| ≤ u * u := by
    rw [abs_mul]; exact mul_le_mul h1 h2 (abs_nonneg _) hu
  calc |d1 + d2 + d1 * d2| ≤ |d1| + |d2| + |d1 * d2| := by
        exact (abs_add_le _ _).trans (add_le_add (abs_add_le _ _) le_rfl)
    _ ≤ u + u + u * u := add_le_add (add_le_add h1 h2) hprod
    _ = (1 + u) ^ 2 - 1 := by ring

/-- A partial whose binade is below `-1022` is below `2^-1022` in magnitude once `|p| < 2`:
dropping it costs less than the float64 resolution of any `O(1)` sum. -/
theorem dropped_partial_small (p : ℝ) (e : ℤ) (hp : |p| < 2) (he : e < -1022) :
    |p * (2 : ℝ) ^ e| < (2 : ℝ) ^ (-1022 : ℤ) := by
  rw [abs_mul, abs_of_pos (zpow_pos (by norm_num : (0:ℝ) < 2) e)]
  have h2 : (2 : ℝ) ^ e ≤ (2 : ℝ) ^ (-1023 : ℤ) :=
    zpow_le_zpow_right₀ (by norm_num : (1:ℝ) ≤ 2) (by omega)
  calc |p| * (2 : ℝ) ^ e < 2 * (2 : ℝ) ^ (-1023 : ℤ) :=
        mul_lt_mul_of_pos_of_nonneg' hp h2 (by positivity) (by norm_num)
    _ = (2 : ℝ) ^ (-1022 : ℤ) := by
        rw [show (-1022 : ℤ) = 1 + (-1023) by norm_num, zpow_add₀ (by norm_num : (2:ℝ) ≠ 0), zpow_one]

end GMasterMarch
