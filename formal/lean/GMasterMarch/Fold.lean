import Mathlib
import GMasterMarch.Recurrence

/-!
§6 of `docs/latitudinal_march_maths.md`: the hemisphere fold.

Two facts: the symmetric Jacobi polynomial has parity `(-1)^n`, so with the spin-0 closed form
(`α = β = m`, `n = ℓ - m`) the row satisfies `d^ℓ_{m,0}(π - θ) = (-1)^(ℓ+m) d^ℓ_{m,0}(θ)`; and a
sum over a ring grid symmetric about the equator splits into northern lanes contracted against
`G_i + σ G_{N-1-i}`.
-/

namespace GMasterMarch

open Finset

/-- Parity of the symmetric Jacobi polynomial: `P_n^{(α,α)}(-x) = (-1)^n P_n^{(α,α)}(x)`. -/
theorem jacobi_symmetric_parity (n α : ℕ) (x : ℝ) :
    jacobi n α α (-x) = (-1) ^ n * jacobi n α α x := by
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

/-- `(-1)^(ℓ - m) = (-1)^(ℓ + m)` for `m ≤ ℓ`: the fold sign in the form the driver uses. -/
theorem neg_one_pow_sub_eq_add {ℓ m : ℕ} (h : m ≤ ℓ) : ((-1 : ℝ)) ^ (ℓ - m) = (-1) ^ (ℓ + m) := by
  have : ℓ + m = (ℓ - m) + 2 * m := by omega
  rw [this, pow_add, pow_mul]; norm_num

/-- The spin-0 row's `θ`-dependence: `(sin θ/2)^m (cos θ/2)^m P_{ℓ-m}^{(m,m)}(cos θ)`.  Under
`θ ↦ π - θ` the half-angle product is invariant and `cos` flips sign. -/
noncomputable def row0 (ℓ m : ℕ) (θ : ℝ) : ℝ :=
  (Real.sin (θ / 2) * Real.cos (θ / 2)) ^ m * jacobi (ℓ - m) m m (Real.cos θ)

theorem row0_reflect (ℓ m : ℕ) (h : m ≤ ℓ) (θ : ℝ) :
    row0 ℓ m (Real.pi - θ) = (-1) ^ (ℓ + m) * row0 ℓ m θ := by
  unfold row0
  have hs : Real.sin ((Real.pi - θ) / 2) = Real.cos (θ / 2) := by
    rw [show (Real.pi - θ) / 2 = Real.pi / 2 - θ / 2 by ring, Real.sin_pi_div_two_sub]
  have hc : Real.cos ((Real.pi - θ) / 2) = Real.sin (θ / 2) := by
    rw [show (Real.pi - θ) / 2 = Real.pi / 2 - θ / 2 by ring, Real.cos_pi_div_two_sub]
  rw [hs, hc, Real.cos_pi_sub, jacobi_symmetric_parity, neg_one_pow_sub_eq_add h]
  ring

/-- A sum over `2n` rings paired `i ↔ 2n-1-i` is a sum over the northern `n` rings of the pair. -/
theorem sum_pairs_even (n : ℕ) (f : ℕ → ℝ) :
    ∑ i ∈ range (2 * n), f i = ∑ i ∈ range n, (f i + f (2 * n - 1 - i)) := by
  rw [sum_add_distrib, two_mul, sum_range_add]
  congr 1
  rw [← sum_range_reflect (fun i => f (n + i)) n]
  apply sum_congr rfl
  intro i hi
  have := mem_range.mp hi
  congr 1; omega

/-- With `2n + 1` rings the equator (`i = n`) is its own partner and is counted once. -/
theorem sum_pairs_odd (n : ℕ) (f : ℕ → ℝ) :
    ∑ i ∈ range (2 * n + 1), f i = ∑ i ∈ range n, (f i + f (2 * n - i)) + f n := by
  have h1 : ∑ i ∈ range (2 * n + 1), f i
      = ∑ i ∈ range n, f i + ∑ i ∈ range (n + 1), f (n + i) := by
    rw [show 2 * n + 1 = n + (n + 1) by ring, sum_range_add]
  have h2 : ∑ i ∈ range (n + 1), f (n + i) = f n + ∑ i ∈ range n, f (2 * n - i) := by
    rw [sum_range_succ']
    simp only [add_zero]
    rw [add_comm]
    congr 1
    rw [← sum_range_reflect (fun i => f (n + (i + 1))) n]
    apply sum_congr rfl
    intro i hi
    have := mem_range.mp hi
    congr 1; omega
  rw [h1, h2, sum_add_distrib]; ring

/-- **Fold.** If the row satisfies `d_{N-1-i} = σ d_i` on a grid of `N = 2n` rings, the latitudinal
sum is `Σ_north d_i (G_i + σ G_{N-1-i})`. -/
theorem fold_even (n : ℕ) (d G : ℕ → ℝ) (σ : ℝ)
    (hd : ∀ i, i < n → d (2 * n - 1 - i) = σ * d i) :
    ∑ i ∈ range (2 * n), d i * G i = ∑ i ∈ range n, d i * (G i + σ * G (2 * n - 1 - i)) := by
  rw [sum_pairs_even]
  apply sum_congr rfl
  intro i hi
  rw [hd i (mem_range.mp hi)]; ring

end GMasterMarch
