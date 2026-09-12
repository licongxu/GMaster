/-
§1 of `docs/latitudinal_march_maths.md`: the Jacobi closed form of `d^ℓ_{m,-s}` *is* Wigner's
function.

`wignerD j m s β` is Wigner's explicit sum for `d^j_{m',m}(β)` (Wigner 1931; Sakurai (3.8.33);
the "Wigner D-matrix" article), specialised to `(m', m) = (m, -s)`:

  d^j_{m,-s}(β) = Σ_k (-1)^(k+s+m) √((j+m)!(j-m)!(j-s)!(j+s)!) / ((j-s-k)!(j-m-k)!(k+m+s)! k!)
                    · cos(β/2)^(2j-s-m-2k) · sin(β/2)^(m+s+2k),

the sum over the `k` for which every factorial argument is non-negative, i.e. `0 ≤ k ≤ j - max m s`.
`jacobiForm` is the §1 closed form with the normalisation `jratio` of `Indices.lean` (which already
carries the `ε_m` flip).  Both are sums over the same `k`; with `x = cos β`, `(x-1)/2 = -sin²(β/2)`
and `(x+1)/2 = cos²(β/2)`, the powers match term by term and the coefficients agree by
`Nat.choose_mul_factorial_mul_factorial`: for `m ≥ s`, `C(j+s, j-m-k) C(j-s, k)` times the
denominator is `(j+s)!(j-s)!`, for `m < s` it is `(j+m)!(j-m)!`, and the square roots recombine.
-/
import Mathlib
import GMasterMarch.Indices
import GMasterMarch.JacobiSum

namespace GMasterMarch

open Finset

/-- Wigner's explicit sum for `d^j_{m,-s}(β)`, `m, s ≤ j`. -/
noncomputable def wignerD (j m s : ℕ) (β : ℝ) : ℝ :=
  ∑ k ∈ range (j - max m s + 1),
    (-1 : ℝ) ^ (k + s + m)
      * (Real.sqrt (((j + m).factorial * (j - m).factorial * (j - s).factorial
          * (j + s).factorial : ℕ))
        / (((j - s - k).factorial * (j - m - k).factorial * (k + m + s).factorial
          * k.factorial : ℕ)))
      * Real.cos (β / 2) ^ (2 * j - s - m - 2 * k) * Real.sin (β / 2) ^ (m + s + 2 * k)

/-- The Jacobi closed form of §1: `(-1)^(m+s) √jratio (sin β/2)^α (cos β/2)^β P_n^{(α,β)}(cos β)`. -/
noncomputable def jacobiForm (j m s : ℕ) (β : ℝ) : ℝ :=
  (-1 : ℝ) ^ (m + s) * Real.sqrt ((jratio j m s : ℚ) : ℝ)
    * Real.sin (β / 2) ^ alpha m s * Real.cos (β / 2) ^ beta m s
    * jacobi (jdeg j m s) (alpha m s) (beta m s) (Real.cos β)

/-- `(cos β - 1)/2 = -sin²(β/2)` and `(cos β + 1)/2 = cos²(β/2)`. -/
theorem cos_eq_two_cos_half_sq (β : ℝ) : Real.cos β = 2 * Real.cos (β / 2) ^ 2 - 1 := by
  rw [← Real.cos_two_mul]; congr 1; ring
theorem half_angle_u (β : ℝ) : (Real.cos β - 1) / 2 = -(Real.sin (β / 2) ^ 2) := by
  have h := cos_eq_two_cos_half_sq β
  have h2 := Real.sin_sq_add_cos_sq (β / 2)
  linarith
theorem half_angle_v (β : ℝ) : (Real.cos β + 1) / 2 = Real.cos (β / 2) ^ 2 := by
  have h := cos_eq_two_cos_half_sq β
  linarith

/-- The coefficient identity, `m ≥ s`: `C(j+s, j-m-k) C(j-s, k) · dens = (j+s)! (j-s)!`. -/
theorem coeff_ge (j m s k : ℕ) (hsm : s ≤ m) (hm : m ≤ j) (hk : k ≤ j - m) :
    (j + s).choose (j - m - k) * (j - s).choose k
      * ((j - s - k).factorial * (j - m - k).factorial * (k + m + s).factorial * k.factorial)
      = (j + s).factorial * (j - s).factorial := by
  have h1 := Nat.choose_mul_factorial_mul_factorial (show j - m - k ≤ j + s by omega)
  have h2 := Nat.choose_mul_factorial_mul_factorial (show k ≤ j - s by omega)
  rw [show j + s - (j - m - k) = k + m + s by omega] at h1
  calc (j + s).choose (j - m - k) * (j - s).choose k
        * ((j - s - k).factorial * (j - m - k).factorial * (k + m + s).factorial * k.factorial)
      = ((j + s).choose (j - m - k) * (j - m - k).factorial * (k + m + s).factorial)
        * ((j - s).choose k * k.factorial * (j - s - k).factorial) := by ring
    _ = (j + s).factorial * (j - s).factorial := by rw [h1, h2]

/-- The coefficient identity, `m < s`: `C(j+m, j-s-k) C(j-m, k) · dens = (j+m)! (j-m)!`. -/
theorem coeff_lt (j m s k : ℕ) (hms : m < s) (hs : s ≤ j) (hk : k ≤ j - s) :
    (j + m).choose (j - s - k) * (j - m).choose k
      * ((j - s - k).factorial * (j - m - k).factorial * (k + m + s).factorial * k.factorial)
      = (j + m).factorial * (j - m).factorial := by
  have h1 := Nat.choose_mul_factorial_mul_factorial (show j - s - k ≤ j + m by omega)
  have h2 := Nat.choose_mul_factorial_mul_factorial (show k ≤ j - m by omega)
  rw [show j + m - (j - s - k) = k + m + s by omega] at h1
  calc (j + m).choose (j - s - k) * (j - m).choose k
        * ((j - s - k).factorial * (j - m - k).factorial * (k + m + s).factorial * k.factorial)
      = ((j + m).choose (j - s - k) * (j - s - k).factorial * (k + m + s).factorial)
        * ((j - m).choose k * k.factorial * (j - m - k).factorial) := by ring
    _ = (j + m).factorial * (j - m).factorial := by rw [h1, h2]

/-- `√(A B) = √(A / B) · B` for `A ≥ 0`, `B > 0`. -/
theorem sqrt_mul_eq_sqrt_div_mul {A B : ℝ} (hA : 0 ≤ A) (hB : 0 < B) :
    Real.sqrt (A * B) = Real.sqrt (A / B) * B := by
  have hsq : Real.sqrt B * Real.sqrt B = B := Real.mul_self_sqrt hB.le
  have hpos : 0 < Real.sqrt B := Real.sqrt_pos.mpr hB
  rw [Real.sqrt_mul hA, Real.sqrt_div hA, div_mul_eq_mul_div, eq_div_iff hpos.ne', mul_assoc, hsq]

/-- **§1.** Wigner's sum for `d^j_{m,-s}` equals the Jacobi closed form, for `m, s ≤ j`. -/
theorem wignerD_eq_jacobiForm (j m s : ℕ) (hm : m ≤ j) (hs : s ≤ j) (β : ℝ) :
    wignerD j m s β = jacobiForm j m s β := by
  unfold wignerD jacobiForm jacobi
  rw [mul_sum]
  unfold jdeg
  refine sum_congr rfl fun k hk => ?_
  have hk' : k ≤ j - max m s := Nat.lt_succ_iff.mp (mem_range.mp hk)
  rw [half_angle_u, half_angle_v, neg_pow (Real.sin (β / 2) ^ 2) k, ← pow_mul, ← pow_mul]
  set c := Real.cos (β / 2) with hc
  set σ := Real.sin (β / 2) with hσ
  -- the powers
  have hpow_s : σ ^ alpha m s * σ ^ (2 * k) = σ ^ (m + s + 2 * k) := by
    rw [← pow_add]; unfold alpha; ring_nf
  have hpow_c : c ^ beta m s * c ^ (2 * (j - max m s - k)) = c ^ (2 * j - s - m - 2 * k) := by
    rw [← pow_add]; congr 1; unfold beta; omega
  -- the sign
  have hsgn : (-1 : ℝ) ^ (k + s + m) = (-1) ^ (m + s) * (-1) ^ k := by
    rw [← pow_add]; congr 1; ring
  -- the coefficient
  have hcoef : Real.sqrt (((j + m).factorial * (j - m).factorial * (j - s).factorial
          * (j + s).factorial : ℕ))
        / (((j - s - k).factorial * (j - m - k).factorial * (k + m + s).factorial
          * k.factorial : ℕ))
      = Real.sqrt ((jratio j m s : ℚ) : ℝ)
        * (((j - max m s + alpha m s).choose (j - max m s - k) : ℝ)
            * ((j - max m s + beta m s).choose k)) := by
    have hdens : (0 : ℝ) < (((j - s - k).factorial * (j - m - k).factorial
        * (k + m + s).factorial * k.factorial : ℕ)) := by positivity
    rcases le_or_gt s m with hsm | hms
    · -- m ≥ s: α = m+s, β = m-s, n = j-m, ratio A/B
      have e1 : j - max m s + alpha m s = j + s := by unfold alpha; omega
      have e2 : j - max m s + beta m s = j - s := by unfold beta; omega
      have e3 : j - max m s - k = j - m - k := by omega
      rw [e1, e2, e3, jratio_of_ge hsm hm]
      have hid := coeff_ge j m s k hsm hm (by omega)
      have hid' := congrArg (fun z : ℕ => (z : ℝ)) hid
      push_cast at hid' hdens ⊢
      have hB : (0 : ℝ) < (j + s).factorial * (j - s).factorial := by positivity
      have hA : (0 : ℝ) ≤ (j - m).factorial * (j + m).factorial := by positivity
      rw [show ((j + m).factorial : ℝ) * (j - m).factorial * (j - s).factorial * (j + s).factorial
          = ((j - m).factorial * (j + m).factorial) * ((j + s).factorial * (j - s).factorial) by ring,
        sqrt_mul_eq_sqrt_div_mul hA hB, div_eq_iff hdens.ne']
      linear_combination (-Real.sqrt (((j - m).factorial * (j + m).factorial : ℝ)
          / ((j + s).factorial * (j - s).factorial))) * hid'
    · -- m < s: α = m+s, β = s-m, n = j-s, ratio B/A
      have e1 : j - max m s + alpha m s = j + m := by unfold alpha; omega
      have e2 : j - max m s + beta m s = j - m := by unfold beta; omega
      have e3 : j - max m s - k = j - s - k := by omega
      rw [e1, e2, e3, jratio_of_lt hms hs]
      have hid := coeff_lt j m s k hms hs (by omega)
      have hid' := congrArg (fun z : ℕ => (z : ℝ)) hid
      push_cast at hid' hdens ⊢
      have hA : (0 : ℝ) < (j - m).factorial * (j + m).factorial := by positivity
      have hB : (0 : ℝ) ≤ (j + s).factorial * (j - s).factorial := by positivity
      rw [inv_div, show ((j + m).factorial : ℝ) * (j - m).factorial * (j - s).factorial
            * (j + s).factorial
          = ((j + s).factorial * (j - s).factorial) * ((j - m).factorial * (j + m).factorial) by ring,
        sqrt_mul_eq_sqrt_div_mul hB hA, div_eq_iff hdens.ne']
      linear_combination (-Real.sqrt (((j + s).factorial * (j - s).factorial : ℝ)
          / ((j - m).factorial * (j + m).factorial))) * hid'
  rw [hsgn, hcoef, ← hpow_s, ← hpow_c]
  ring

end GMasterMarch
