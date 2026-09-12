/-
The Jacobi polynomial as the explicit DLMF 18.5.7 sum, and DLMF 18.9.2 (the three-term recurrence)
proved for that sum.  §2 of `docs/latitudinal_march_maths.md`.

Route.  Write `u = (x-1)/2`, so `(x+1)/2 = u + 1`, and let `tj n α β u k` be the `k`-th summand
(zero outside `0 ≤ k ≤ n`).  Three exact ratio identities between neighbouring summands follow from
`Nat.choose_succ_right_eq` / `Nat.choose_mul_succ_eq`:

  (I1)  (α+k+1)(k+1)(u+1) t_{k+1}   = (n-k)(n+β-k) u t_k                     `tj_succ`
  (I2)  (n+1+α)(n+1+β)(u+1) t^{n}_k = (n+1-k)(n+1+β-k) t^{n+1}_k             `tj_pred_v`
  (I3)  (n+1+α)(n+1+β) u t^{n}_k    = (α+k+1)(k+1) t^{n+1}_{k+1}             `tj_pred_u`

(I2) - (I3) expresses the degree-`n` summand through two degree-`(n+1)` summands with no division by
`u` or `u+1` (`tj_pred`), and applying it twice gives the degree-`n` summand through three
degree-`(n+2)` summands (`tj_pred2`).  With those, the recurrence's summand at degree `N = n+2`,
multiplied by `D' = (N+α)(N+β)(N+α-1)(N+β-1)`, is the exact difference `W(k+1) - W(k)` of

  W(k) = -q0(k) t_k + (q2(k-1) - C (u+1)(k+1)(k+1+α)) t_{k+1},

a certificate found with sympy (session 36); `W(0) = 0` by (I1) at `k = 0` and `W(N+1) = 0` because
the summands vanish past `N`.  Every identity is closed by `linear_combination`.
-/
import Mathlib

namespace GMasterMarch

open Finset

/-- DLMF 18.5.7: `P_n^{(α,β)}(x) = Σ_k C(n+α, n-k) C(n+β, k) ((x-1)/2)^k ((x+1)/2)^(n-k)`. -/
noncomputable def jacobi (n α β : ℕ) (x : ℝ) : ℝ :=
  ∑ k ∈ range (n + 1),
    ((n + α).choose (n - k) : ℝ) * ((n + β).choose k) * ((x - 1) / 2) ^ k * ((x + 1) / 2) ^ (n - k)

/-- The `k`-th summand in the variable `u = (x-1)/2`, zero outside `k ≤ n`. -/
noncomputable def tj (n α β : ℕ) (u : ℝ) (k : ℕ) : ℝ :=
  if k ≤ n then ((n + α).choose (n - k) : ℝ) * ((n + β).choose k) * u ^ k * (u + 1) ^ (n - k) else 0

theorem jacobi_eq_sum_tj (n α β : ℕ) (x : ℝ) :
    jacobi n α β x = ∑ k ∈ range (n + 1), tj n α β ((x - 1) / 2) k := by
  unfold jacobi tj
  refine sum_congr rfl fun k hk => ?_
  have hk' : k ≤ n := Nat.lt_succ_iff.mp (mem_range.mp hk)
  rw [if_pos hk']
  have : (x - 1) / 2 + 1 = (x + 1) / 2 := by ring
  rw [this]

theorem tj_of_gt (n α β : ℕ) (u : ℝ) {k : ℕ} (h : n < k) : tj n α β u k = 0 := by
  unfold tj; rw [if_neg (by omega)]

theorem tj_of_le (n α β : ℕ) (u : ℝ) {k : ℕ} (h : k ≤ n) :
    tj n α β u k = ((n + α).choose (n - k) : ℝ) * ((n + β).choose k) * u ^ k * (u + 1) ^ (n - k) := by
  unfold tj; rw [if_pos h]

/-! ### Binomial ratio identities over `ℝ` -/

theorem choose_succ_right_real (M j : ℕ) (h : j ≤ M) :
    (M.choose (j + 1) : ℝ) * ((j : ℝ) + 1) = (M.choose j : ℝ) * ((M : ℝ) - j) := by
  have := congrArg (fun z : ℕ => (z : ℝ)) (Nat.choose_succ_right_eq M j)
  push_cast [Nat.cast_sub h] at this
  linear_combination this

theorem choose_mul_succ_real (M j : ℕ) (h : j ≤ M + 1) :
    (M.choose j : ℝ) * ((M : ℝ) + 1) = ((M + 1).choose j : ℝ) * ((M : ℝ) + 1 - j) := by
  have := congrArg (fun z : ℕ => (z : ℝ)) (Nat.choose_mul_succ_eq M j)
  push_cast [Nat.cast_sub h] at this
  linear_combination this

theorem succ_mul_choose_real (M j : ℕ) (h : j ≤ M) :
    ((M : ℝ) + 1) * (M.choose j : ℝ) = ((M + 1).choose (j + 1) : ℝ) * ((j : ℝ) + 1) := by
  have h1 := choose_succ_right_real (M + 1) j (by omega)
  have h2 := choose_mul_succ_real M j (by omega)
  push_cast at h1
  linear_combination h2 - h1

/-! ### The three ratio identities between summands -/

/-- (I1) `(α+k+1)(k+1)(u+1) t_{k+1} = (n-k)(n+β-k) u t_k`, for every `k`. -/
theorem tj_succ (n α β : ℕ) (u : ℝ) (k : ℕ) :
    ((α : ℝ) + k + 1) * ((k : ℝ) + 1) * (u + 1) * tj n α β u (k + 1)
      = ((n : ℝ) - k) * ((n : ℝ) + β - k) * u * tj n α β u k := by
  rcases lt_or_ge n k with h | h
  · rw [tj_of_gt n α β u h, tj_of_gt n α β u (by omega)]; ring
  rcases eq_or_lt_of_le h with h | h
  · rw [tj_of_gt n α β u (by omega), h]; ring
  rw [tj_of_le n α β u (by omega : k + 1 ≤ n), tj_of_le n α β u h.le]
  have e1 : ((n + α).choose (n - k) : ℝ) * ((n : ℝ) - k)
      = ((n + α).choose (n - (k + 1)) : ℝ) * ((α : ℝ) + k + 1) := by
    have := choose_succ_right_real (n + α) (n - (k + 1)) (by omega)
    rw [show n - (k + 1) + 1 = n - k by omega] at this
    push_cast [Nat.cast_sub (by omega : k + 1 ≤ n)] at this
    linear_combination this
  have e2 : ((n + β).choose (k + 1) : ℝ) * ((k : ℝ) + 1)
      = ((n + β).choose k : ℝ) * ((n : ℝ) + β - k) := by
    have := choose_succ_right_real (n + β) k (by omega)
    push_cast at this
    linear_combination this
  have hv : (u + 1) ^ (n - k) = (u + 1) ^ (n - (k + 1)) * (u + 1) := by
    rw [← pow_succ, show n - (k + 1) + 1 = n - k by omega]
  rw [hv, pow_succ]
  linear_combination (u ^ k * u * (u + 1) ^ (n - (k + 1)) * (u + 1))
    * (-(((k : ℝ) + 1) * ((n + β).choose (k + 1) : ℝ)) * e1
       + (((n : ℝ) - k) * ((n + α).choose (n - k) : ℝ)) * e2)

/-- (I2) `(n+1+α)(n+1+β)(u+1) t^n_k = (n+1-k)(n+1+β-k) t^{n+1}_k`. -/
theorem tj_pred_v (n α β : ℕ) (u : ℝ) (k : ℕ) :
    ((n : ℝ) + 1 + α) * ((n : ℝ) + 1 + β) * (u + 1) * tj n α β u k
      = ((n : ℝ) + 1 - k) * ((n : ℝ) + 1 + β - k) * tj (n + 1) α β u k := by
  rcases lt_or_ge (n + 1) k with h | h
  · rw [tj_of_gt n α β u (by omega), tj_of_gt (n + 1) α β u h]; ring
  rcases eq_or_lt_of_le h with h | h
  · rw [tj_of_gt n α β u (by omega), h]; push_cast; ring
  have hk : k ≤ n := by omega
  rw [tj_of_le n α β u hk, tj_of_le (n + 1) α β u (by omega)]
  have e1 : ((n : ℝ) + 1 + α) * ((n + α).choose (n - k) : ℝ)
      = ((n + 1 + α).choose (n + 1 - k) : ℝ) * ((n : ℝ) + 1 - k) := by
    have := succ_mul_choose_real (n + α) (n - k) (by omega)
    rw [show n - k + 1 = n + 1 - k by omega, show n + α + 1 = n + 1 + α by omega] at this
    push_cast [Nat.cast_sub hk] at this
    linear_combination this
  have e2 : ((n + β).choose k : ℝ) * ((n : ℝ) + 1 + β)
      = ((n + 1 + β).choose k : ℝ) * ((n : ℝ) + 1 + β - k) := by
    have := choose_mul_succ_real (n + β) k (by omega)
    rw [show n + β + 1 = n + 1 + β by omega] at this
    push_cast at this
    linear_combination this
  have hv : (u + 1) ^ (n + 1 - k) = (u + 1) ^ (n - k) * (u + 1) := by
    rw [← pow_succ, show n - k + 1 = n + 1 - k by omega]
  rw [hv]
  linear_combination (u ^ k * (u + 1) ^ (n - k) * (u + 1))
    * ((((n + β).choose k : ℝ) * ((n : ℝ) + 1 + β)) * e1
       + (((n + 1 + α).choose (n + 1 - k) : ℝ) * ((n : ℝ) + 1 - k)) * e2)

/-- (I3) `(n+1+α)(n+1+β) u t^n_k = (α+k+1)(k+1) t^{n+1}_{k+1}`. -/
theorem tj_pred_u (n α β : ℕ) (u : ℝ) (k : ℕ) :
    ((n : ℝ) + 1 + α) * ((n : ℝ) + 1 + β) * u * tj n α β u k
      = ((α : ℝ) + k + 1) * ((k : ℝ) + 1) * tj (n + 1) α β u (k + 1) := by
  rcases lt_or_ge n k with h | h
  · rw [tj_of_gt n α β u h, tj_of_gt (n + 1) α β u (by omega)]; ring
  rw [tj_of_le n α β u h, tj_of_le (n + 1) α β u (by omega)]
  rw [show n + 1 - (k + 1) = n - k by omega]
  have e1 : ((n + α).choose (n - k) : ℝ) * ((n : ℝ) + 1 + α)
      = ((n + 1 + α).choose (n - k) : ℝ) * ((α : ℝ) + k + 1) := by
    have := choose_mul_succ_real (n + α) (n - k) (by omega)
    rw [show n + α + 1 = n + 1 + α by omega] at this
    push_cast [Nat.cast_sub h] at this
    linear_combination this
  have e2 : ((n : ℝ) + 1 + β) * ((n + β).choose k : ℝ)
      = ((n + 1 + β).choose (k + 1) : ℝ) * ((k : ℝ) + 1) := by
    have := succ_mul_choose_real (n + β) k (by omega)
    rw [show n + β + 1 = n + 1 + β by omega] at this
    push_cast at this
    linear_combination this
  rw [pow_succ]
  linear_combination (u ^ k * u * (u + 1) ^ (n - k))
    * ((((n : ℝ) + 1 + β) * ((n + β).choose k : ℝ)) * e1
       + (((n + 1 + α).choose (n - k) : ℝ) * ((α : ℝ) + k + 1)) * e2)

/-- (E1) the degree-`n` summand through two degree-`(n+1)` summands, no division. -/
theorem tj_pred (n α β : ℕ) (u : ℝ) (k : ℕ) :
    ((n : ℝ) + 1 + α) * ((n : ℝ) + 1 + β) * tj n α β u k
      = ((n : ℝ) + 1 - k) * ((n : ℝ) + 1 + β - k) * tj (n + 1) α β u k
        - ((α : ℝ) + k + 1) * ((k : ℝ) + 1) * tj (n + 1) α β u (k + 1) := by
  linear_combination tj_pred_v n α β u k - tj_pred_u n α β u k

/-- (E2) the degree-`n` summand through three degree-`(n+2)` summands. -/
theorem tj_pred2 (n α β : ℕ) (u : ℝ) (k : ℕ) :
    ((n : ℝ) + 2 + α) * ((n : ℝ) + 2 + β) * ((n : ℝ) + 1 + α) * ((n : ℝ) + 1 + β) * tj n α β u k
      = ((n : ℝ) + 1 - k) * ((n : ℝ) + 1 + β - k) * ((n : ℝ) + 2 - k) * ((n : ℝ) + 2 + β - k)
          * tj (n + 2) α β u k
        - 2 * ((n : ℝ) + 1 - k) * ((n : ℝ) + 1 + β - k) * ((α : ℝ) + k + 1) * ((k : ℝ) + 1)
          * tj (n + 2) α β u (k + 1)
        + ((α : ℝ) + k + 1) * ((k : ℝ) + 1) * ((α : ℝ) + k + 2) * ((k : ℝ) + 2)
          * tj (n + 2) α β u (k + 2) := by
  have h1 := tj_pred n α β u k
  have h2 := tj_pred (n + 1) α β u k
  have h3 := tj_pred (n + 1) α β u (k + 1)
  push_cast at h2 h3
  linear_combination (((n : ℝ) + 2 + α) * ((n : ℝ) + 2 + β)) * h1
    + (((n : ℝ) + 1 - k) * ((n : ℝ) + 1 + β - k)) * h2
    - (((α : ℝ) + k + 1) * ((k : ℝ) + 1)) * h3

/-! ### The recurrence -/

/-- `S = 2n + α + β`. -/
def S (n α β : ℕ) : ℝ := 2 * n + α + β

/-- The common denominator `2n (n+α+β) (S-2)`. -/
def den (n α β : ℕ) : ℝ := 2 * n * (n + α + β) * (S n α β - 2)

/-! The certificate is written over real parameters so that unfolding it produces no `ℕ → ℝ`
casts inside the large terms. -/

/-- `S` over reals. -/
def Sr (n α β : ℝ) : ℝ := 2 * n + α + β
/-- `den` over reals. -/
def denr (n α β : ℝ) : ℝ := 2 * n * (n + α + β) * (Sr n α β - 2)
/-- The coefficient of `P_{n-1}` in DLMF 18.9.2. -/
def Ar (n α β x : ℝ) : ℝ := (Sr n α β - 1) * (Sr n α β * (Sr n α β - 2) * x + (α ^ 2 - β ^ 2))
/-- The coefficient of `P_{n-2}` in DLMF 18.9.2. -/
def Br (n α β : ℝ) : ℝ := 2 * (n + α - 1) * (n + β - 1) * Sr n α β
/-- `D' = (n+α)(n+β)(n+α-1)(n+β-1)`, the denominator cleared by (E1)/(E2). -/
def Dr (n α β : ℝ) : ℝ := (n + α) * (n + β) * (n + α - 1) * (n + β - 1)
/-- The constant of the certificate. -/
def Cr (n α β : ℝ) : ℝ := 2 * Sr n α β * (n + α - 1) * (n + β - 1) * (Sr n α β - 2) * (Sr n α β - 1)
/-- Coefficient of `t_k` in `D' ×` (the recurrence summand), after (E1)/(E2). -/
def q0r (n α β x k : ℝ) : ℝ :=
  Dr n α β * denr n α β
    - Ar n α β x * (n + α - 1) * (n + β - 1) * (n - k) * (n + β - k)
    + Br n α β * (n - 1 - k) * (n - 1 + β - k) * (n - k) * (n + β - k)
/-- `q2(k-1)`: the coefficient of `t_{k+2}` in the summand at `k-1`. -/
def q2m1r (n α β k : ℝ) : ℝ := Br n α β * (α + k) * k * (α + k + 1) * (k + 1)

/-- The telescoping certificate, at top degree `n + 2`. -/
noncomputable def W (n α β : ℕ) (u : ℝ) (k : ℕ) : ℝ :=
  -q0r (n + 2) α β (2 * u + 1) k * tj (n + 2) α β u k
    + (q2m1r (n + 2) α β k - Cr (n + 2) α β * (u + 1) * (k + 1) * (k + 1 + α))
      * tj (n + 2) α β u (k + 1)

/-- The recurrence summand at degree `n+2`, times `D'`, is `W(k+1) - W(k)`. -/
theorem summand_telescopes (n α β : ℕ) (u : ℝ) (k : ℕ) :
    Dr (n + 2) α β * (denr (n + 2) α β * tj (n + 2) α β u k
        - Ar (n + 2) α β (2 * u + 1) * tj (n + 1) α β u k
        + Br (n + 2) α β * tj n α β u k)
      = W n α β u (k + 1) - W n α β u k := by
  have h1 := tj_pred (n + 1) α β u k
  have h2 := tj_pred2 n α β u k
  have h3 := tj_succ (n + 2) α β u (k + 1)
  push_cast at h1 h2 h3
  simp only [W, q0r, q2m1r, Dr, denr, Ar, Br, Cr, Sr]
  push_cast
  linear_combination
    (-((2 * ((n : ℝ) + 2) + α + β - 1) * ((2 * ((n : ℝ) + 2) + α + β)
        * (2 * ((n : ℝ) + 2) + α + β - 2) * (2 * u + 1) + ((α : ℝ) ^ 2 - (β : ℝ) ^ 2))
        * ((n : ℝ) + 1 + α) * ((n : ℝ) + 1 + β))) * h1
    + (2 * ((n : ℝ) + 2 + α - 1) * ((n : ℝ) + 2 + β - 1) * (2 * ((n : ℝ) + 2) + α + β)) * h2
    + (2 * (2 * ((n : ℝ) + 2) + α + β) * ((n : ℝ) + 2 + α - 1) * ((n : ℝ) + 2 + β - 1)
        * (2 * ((n : ℝ) + 2) + α + β - 2) * (2 * ((n : ℝ) + 2) + α + β - 1)) * h3

theorem W_zero (n α β : ℕ) (u : ℝ) : W n α β u 0 = 0 := by
  have h0 := tj_succ (n + 2) α β u 0
  push_cast at h0
  simp only [W, q0r, q2m1r, Dr, denr, Ar, Br, Cr, Sr]
  push_cast
  linear_combination
    (-(2 * (2 * ((n : ℝ) + 2) + α + β) * ((n : ℝ) + 2 + α - 1) * ((n : ℝ) + 2 + β - 1)
        * (2 * ((n : ℝ) + 2) + α + β - 2) * (2 * ((n : ℝ) + 2) + α + β - 1))) * h0

theorem W_top (n α β : ℕ) (u : ℝ) : W n α β u (n + 3) = 0 := by
  unfold W
  rw [tj_of_gt (n + 2) α β u (by omega : n + 2 < n + 3),
      tj_of_gt (n + 2) α β u (by omega : n + 2 < n + 3 + 1)]
  ring

theorem Dr_ne_zero (n α β : ℕ) : Dr ((n : ℝ) + 2) α β ≠ 0 := by
  unfold Dr
  have : (0 : ℝ) ≤ n := Nat.cast_nonneg n
  have : (0 : ℝ) ≤ α := Nat.cast_nonneg α
  have : (0 : ℝ) ≤ β := Nat.cast_nonneg β
  apply mul_ne_zero (mul_ne_zero (mul_ne_zero _ _) _) _ <;> linarith

/-- DLMF 18.9.2 for the explicit sum, at degree `n + 2`, over the real coefficient forms. -/
theorem jacobi_three_term_succ (n α β : ℕ) (x : ℝ) :
    denr ((n : ℝ) + 2) α β * jacobi (n + 2) α β x
      = Ar ((n : ℝ) + 2) α β x * jacobi (n + 1) α β x
        - Br ((n : ℝ) + 2) α β * jacobi n α β x := by
  set u := (x - 1) / 2 with hu
  have hx : x = 2 * u + 1 := by rw [hu]; ring
  rw [jacobi_eq_sum_tj, jacobi_eq_sum_tj, jacobi_eq_sum_tj, ← hu]
  -- extend the shorter sums to `range (n + 3)`: the extra summands vanish
  have hs1 : ∑ k ∈ range (n + 1 + 1), tj (n + 1) α β u k
      = ∑ k ∈ range (n + 2 + 1), tj (n + 1) α β u k := by
    rw [sum_range_succ _ (n + 2), tj_of_gt (n + 1) α β u (by omega : n + 1 < n + 2), add_zero]
  have hs2 : ∑ k ∈ range (n + 1), tj n α β u k = ∑ k ∈ range (n + 2 + 1), tj n α β u k := by
    rw [sum_range_succ _ (n + 2), sum_range_succ _ (n + 1),
      tj_of_gt n α β u (by omega : n < n + 2), tj_of_gt n α β u (by omega : n < n + 1),
      add_zero, add_zero]
  rw [hs1, hs2]
  have key : Dr ((n : ℝ) + 2) α β
      * (denr ((n : ℝ) + 2) α β * ∑ k ∈ range (n + 2 + 1), tj (n + 2) α β u k
        - (Ar ((n : ℝ) + 2) α β x * ∑ k ∈ range (n + 2 + 1), tj (n + 1) α β u k
          - Br ((n : ℝ) + 2) α β * ∑ k ∈ range (n + 2 + 1), tj n α β u k))
      = ∑ k ∈ range (n + 2 + 1), (W n α β u (k + 1) - W n α β u k) := by
    rw [hx, mul_sum, mul_sum, mul_sum, ← sum_sub_distrib, ← sum_sub_distrib, mul_sum]
    refine sum_congr rfl fun k _ => ?_
    have := summand_telescopes n α β u k
    push_cast at this
    rw [← this]; ring
  rw [sum_range_sub, W_top, W_zero, sub_zero] at key
  rcases mul_eq_zero.mp key with h | h
  · exact absurd h (Dr_ne_zero n α β)
  · linear_combination h

/-- DLMF 18.9.2 in the form used downstream (`Recurrence.lean`), for `2 ≤ n`. -/
theorem jacobi_three_term (n α β : ℕ) (hn : 2 ≤ n) (x : ℝ) :
    den n α β * jacobi n α β x
      = (S n α β - 1) * (S n α β * (S n α β - 2) * x + ((α : ℝ) ^ 2 - (β : ℝ) ^ 2))
          * jacobi (n - 1) α β x
        - 2 * (n + α - 1) * (n + β - 1) * S n α β * jacobi (n - 2) α β x := by
  obtain ⟨m, rfl⟩ : ∃ m, n = m + 2 := ⟨n - 2, by omega⟩
  have := jacobi_three_term_succ m α β x
  rw [show m + 2 - 1 = m + 1 by omega, show m + 2 - 2 = m by omega]
  unfold denr Ar Br Sr at this
  unfold den S
  push_cast
  linear_combination this

end GMasterMarch
