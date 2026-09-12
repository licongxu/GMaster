/-
§2 of `docs/latitudinal_march_maths.md`: the three-term recurrence the kernel marches.

`jacobi n α β x` is the explicit finite sum (DLMF 18.5.7) for natural `α, β`, defined in
`GMasterMarch.JacobiSum`, where DLMF 18.9.2 for that sum is proved (`jacobi_three_term`, no
`sorry`).  Here the kernel's coefficients `c1, c0, cb` are the DLMF 18.9.2 coefficients divided by
the common denominator; the rearrangement is exact algebra (`recurrence_form`), the `c0` spelling
is the §1 identity `α² - β² = 4 m s` (`c0_eq`), the seeds are `jacobi_zero`, `jacobi_one`, and
`march_eq_jacobi` shows the marched sequence is the Jacobi polynomial at every degree.
-/
import Mathlib
import GMasterMarch.Indices
import GMasterMarch.JacobiSum

namespace GMasterMarch

open Finset

theorem jacobi_zero (α β : ℕ) (x : ℝ) : jacobi 0 α β x = 1 := by
  simp [jacobi]

/-- The seed at `ℓ = max(m,s) + 1`: `P_1 = ((α+β+2)/2) x + (α-β)/2`, whose linear coefficient is
`(α+β+2)/2` and not `(α+β)/2`. -/
theorem jacobi_one (α β : ℕ) (x : ℝ) :
    jacobi 1 α β x = ((α + β + 2) / 2) * x + ((α : ℝ) - β) / 2 := by
  simp [jacobi, sum_range_succ, Nat.choose_one_right]
  ring

noncomputable def c1 (n α β : ℕ) : ℝ := (S n α β - 1) * S n α β * (S n α β - 2) / den n α β
noncomputable def c0 (n α β : ℕ) : ℝ := (S n α β - 1) * ((α : ℝ) ^ 2 - (β : ℝ) ^ 2) / den n α β
noncomputable def cb (n α β : ℕ) : ℝ := 2 * (n + α - 1) * (n + β - 1) * S n α β / den n α β

/-- The kernel's `a0 = (S-1) * 4 m s / den` is `c0` for `α = m + s`, `β = |m - s|`. -/
theorem c0_eq (n m s : ℕ) :
    c0 n (alpha m s) (beta m s) = (S n (alpha m s) (beta m s) - 1) * (4 * m * s)
      / den n (alpha m s) (beta m s) := by
  unfold c0; rw [alpha_sq_sub_beta_sq_real]

theorem den_pos (n α β : ℕ) (hn : 2 ≤ n) : 0 < den n α β := by
  unfold den S
  have h1 : (0 : ℝ) < 2 * n := by positivity
  have h2 : (0 : ℝ) < n + α + β := by positivity
  have h3 : (0 : ℝ) < 2 * n + α + β - 2 := by
    have : (2 : ℝ) ≤ n := by exact_mod_cast hn
    linarith [(Nat.cast_nonneg α : (0:ℝ) ≤ α), (Nat.cast_nonneg β : (0:ℝ) ≤ β)]
  positivity

/-- DLMF 18.9.2 in the form `den · P_n = (S-1)[S(S-2) x + (α²-β²)] P_{n-1} - 2(n+α-1)(n+β-1) S P_{n-2}`
rearranges exactly to the marched form `P_n = (c1 x + c0) P_{n-1} - cb P_{n-2}`. -/
theorem recurrence_form (n α β : ℕ) (x pn pn1 pn2 : ℝ) (hden : den n α β ≠ 0)
    (h : den n α β * pn
      = (S n α β - 1) * (S n α β * (S n α β - 2) * x + ((α : ℝ) ^ 2 - (β : ℝ) ^ 2)) * pn1
        - 2 * (n + α - 1) * (n + β - 1) * S n α β * pn2) :
    pn = (c1 n α β * x + c0 n α β) * pn1 - cb n α β * pn2 := by
  unfold c1 c0 cb
  field_simp
  linear_combination h

/-- The sequence the kernel marches: seeds `1`, `P_1`, then `(c1 x + c0) v_{n-1} - cb v_{n-2}`. -/
noncomputable def march (α β : ℕ) (x : ℝ) : ℕ → ℝ
  | 0 => 1
  | 1 => ((α + β + 2) / 2) * x + ((α : ℝ) - β) / 2
  | n + 2 => (c1 (n + 2) α β * x + c0 (n + 2) α β) * march α β x (n + 1)
              - cb (n + 2) α β * march α β x n

/-- The marched sequence is the Jacobi polynomial at every degree (given `jacobi_three_term`). -/
theorem march_eq_jacobi (α β : ℕ) (x : ℝ) : ∀ n, march α β x n = jacobi n α β x := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    match n with
    | 0 => simp [march, jacobi_zero]
    | 1 => simp [march, jacobi_one]
    | n + 2 =>
      have h1 := ih (n + 1) (by omega)
      have h0 := ih n (by omega)
      simp only [march]
      rw [h1, h0]
      have hden : den (n + 2) α β ≠ 0 := (den_pos (n + 2) α β (by omega)).ne'
      have key := jacobi_three_term (n + 2) α β (by omega) x
      simp only [show n + 2 - 2 = n by omega] at key
      exact (recurrence_form (n + 2) α β x _ _ _ hden key).symm

end GMasterMarch
