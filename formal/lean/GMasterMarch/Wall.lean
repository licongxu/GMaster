import Mathlib

/-!
§7 of `docs/latitudinal_march_maths.md`: the arithmetic wall.

The march visits `Σ_{m<L} (L - m) = L(L+1)/2` degrees per ring, and a kernel that must perform at
least `c · N` fused multiply-adds on hardware that issues at most `R` per second takes at least
`c N / R` seconds.
-/

namespace GMasterMarch

open Finset

/-- The live-degree count of one ring: `Σ_{m < L} (L - m) = L (L + 1) / 2`. -/
theorem live_degrees (L : ℕ) : ∑ m ∈ range L, (L - m) = L * (L + 1) / 2 := by
  have h : ∑ m ∈ range L, (L - m) = ∑ m ∈ range L, (m + 1) := by
    rw [← sum_range_reflect (fun m => m + 1) L]
    apply sum_congr rfl
    intro m hm
    have := mem_range.mp hm
    omega
  rw [h, sum_add_distrib, sum_const, card_range, smul_eq_mul, mul_one]
  have hg := Finset.sum_range_id_mul_two L
  -- `(Σ_{i<L} i) * 2 = L * (L - 1)`; hence `Σ i + L = L(L+1)/2`
  have : (∑ i ∈ range L, i) * 2 + L * 2 = L * (L + 1) := by
    rw [hg]
    rcases L with _ | L
    · simp
    · simp only [Nat.add_sub_cancel]; ring
  omega

/-- Triples per pass: `N = L(L+1)/2 · rings`. -/
def triples (L rings : ℕ) : ℕ := L * (L + 1) / 2 * rings

/-- **Time bound.** `fmas ≥ c·N` operations at rate `≤ R` take `≥ c·N/R` seconds. -/
theorem time_lower_bound (c N R T fmas : ℝ) (hR : 0 < R) (hf : c * N ≤ fmas) (hT : fmas ≤ R * T) :
    c * N / R ≤ T := by
  rw [div_le_iff₀ hR]
  linarith

end GMasterMarch
