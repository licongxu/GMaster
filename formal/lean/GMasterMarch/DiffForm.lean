import Mathlib
import GMasterMarch.ExponentCarry
import GMasterMarch.Mirror

/-!
§1–§3 of `docs/march_v2_maths.md`: the difference form of the three-term recurrence, the
reflected (southern) row, and the homogeneity of the `(v, D)` march.

`gmaster/_cuda/march_v2.cu` marches, for the Jacobi row `v_n = (c1 x + c0) v_{n-1} - cb v_{n-2}`
(`GMasterMarch.Recurrence.march`), the pair `(v, D = v_n - v_{n-1})` as

    C = c1 (x - 1) + (c1 - 1 - cb + c0),   D ← cb D + C v,   v ← v + D

on the northern hemisphere, and the reflected row `(-1)^n v_n` on the southern one, where the
same code runs with `|x| = -x` in place of `x` and `-c0` in place of `c0`.  Everything here is
exact real algebra; the reason the kernel prefers this form is the (heuristic) §5 of the note.
-/

namespace GMasterMarch

/-! ### §1 Difference form -/

/-- **One step.** If `v₂ = A v₁ - B v₀` then `D₂ = v₂ - v₁ = (A - 1 - B) v₁ + B (v₁ - v₀)`. -/
theorem diff_form_step (A B v0 v1 : ℝ) :
    (A * v1 - B * v0) - v1 = (A - 1 - B) * v1 + B * (v1 - v0) := by ring

/-- The Jacobi specialisation of the coefficient `C = A - 1 - B` with `A = c1 x + c0`:
`C = c1 (x - 1) + (c1 - 1 - cb + c0)`, the form the kernel evaluates from the lane coordinate
`x - 1` and the table entry `E⁺ = c1 - 1 - cb + c0`. -/
theorem jacobi_C_eq (c1 c0 cb x : ℝ) :
    (c1 * x + c0) - 1 - cb = c1 * (x - 1) + (c1 - 1 - cb + c0) := by ring

/-- The `(v, D)` pair the kernel carries.  Index `k` holds `(v_{k+1}, D_{k+1})`; the coefficients
`C k`, `B k` are those that produce degree `k + 2`, as in `rec2` (`a k`, `b k`). -/
noncomputable def diffMarch (C B : ℕ → ℝ) (s0 s1 : ℝ) : ℕ → ℝ × ℝ
  | 0 => (s1, s1 - s0)
  | k + 1 =>
      let p := diffMarch C B s0 s1 k
      let D := C k * p.1 + B k * p.2
      (p.1 + D, D)

/-- **The difference march reproduces the recurrence.** Iterating `(v, D) ← (v + D', D')` with
`D' = (a - 1 - b) v + b D` from `(v₁, v₁ - v₀)` gives `(v_n, v_n - v_{n-1})` for the sequence
`v_{n+2} = a n v_{n+1} - b n v_n` at every degree. -/
theorem diff_march_eq_rec2 (a b : ℕ → ℝ) (s0 s1 : ℝ) : ∀ k,
    diffMarch (fun k => a k - 1 - b k) b s0 s1 k
      = (rec2 a b s0 s1 (k + 1), rec2 a b s0 s1 (k + 1) - rec2 a b s0 s1 k) := by
  intro k
  induction k with
  | zero => simp [diffMarch, rec2]
  | succ k ih =>
    simp only [diffMarch, ih, rec2]
    ext <;> simp only <;> ring

/-- The kernel's northern march for the Jacobi row: `C = c1 (x - 1) + (c1 - 1 - cb + c0)`,
`B = cb`, seeds `1`, `P_1`. -/
noncomputable def kernelMarch (α β : ℕ) (x : ℝ) : ℕ → ℝ × ℝ :=
  diffMarch (fun k => c1 (k + 2) α β * (x - 1) + (c1 (k + 2) α β - 1 - cb (k + 2) α β + c0 (k + 2) α β))
    (fun k => cb (k + 2) α β) 1 (((α + β + 2) / 2) * x + ((α : ℝ) - β) / 2)

/-- **`diff_march_eq_march`.** The value lane of the kernel's march is the three-term march
(`Recurrence.march`, hence the Jacobi polynomial) at degree `k + 1`, and its difference lane is
the difference of consecutive degrees. -/
theorem diff_march_eq_march (α β : ℕ) (x : ℝ) (k : ℕ) :
    kernelMarch α β x k = (march α β x (k + 1), march α β x (k + 1) - march α β x k) := by
  unfold kernelMarch
  have hC : (fun k => c1 (k + 2) α β * (x - 1) + (c1 (k + 2) α β - 1 - cb (k + 2) α β + c0 (k + 2) α β))
      = fun k => (c1 (k + 2) α β * x + c0 (k + 2) α β) - 1 - cb (k + 2) α β := by
    funext k; ring
  rw [hC, diff_march_eq_rec2, march_eq_rec2 α β x (k + 1), march_eq_rec2 α β x k]

/-- The value lane alone. -/
theorem diff_march_value (α β : ℕ) (x : ℝ) (k : ℕ) :
    (kernelMarch α β x k).1 = jacobi (k + 1) α β x := by
  rw [diff_march_eq_march, march_eq_jacobi]

/-- The northern seed `D_1 = v_1 - v_0 = P_1 - 1 = ((α+β+2)/2)(x - 1) + α`: the kernel's
`fma(half, x - 1, cst)` with `cst = α`. -/
theorem seed_D1_north (α β : ℕ) (x : ℝ) :
    (((α + β + 2) / 2) * x + ((α : ℝ) - β) / 2) - 1 = ((α + β + 2) / 2) * (x - 1) + α := by ring

/-! ### §2 Reflection -/

/-- **One reflected step.** If `v₂ = (c1 x + c0) v₁ - cb v₀` then, with `w_n = (-1)^n v_n`
(`w₀ = v₀`, `w₁ = -v₁`, `w₂ = v₂`), `w₂ = (c1 (-x) - c0) w₁ - cb w₀`. -/
theorem reflect_step (c1 c0 cb x v0 v1 : ℝ) :
    (c1 * x + c0) * v1 - cb * v0 = (c1 * (-x) - c0) * (-v1) - cb * v0 := by ring

/-- **`reflect_recurrence`.** If `v_{n+2} = (c1_{n+2} x + c0_{n+2}) v_{n+1} - cb_{n+2} v_n` for
all `n`, then `w_n = (-1)^n v_n` satisfies
`w_{n+2} = (c1_{n+2} (-x) - c0_{n+2}) w_{n+1} - cb_{n+2} w_n`. -/
theorem reflect_recurrence (c1 c0 cb : ℕ → ℝ) (x : ℝ) (v : ℕ → ℝ)
    (h : ∀ n, v (n + 2) = (c1 (n + 2) * x + c0 (n + 2)) * v (n + 1) - cb (n + 2) * v n) :
    ∀ n, (-1) ^ (n + 2) * v (n + 2)
      = (c1 (n + 2) * (-x) - c0 (n + 2)) * ((-1) ^ (n + 1) * v (n + 1))
        - cb (n + 2) * ((-1) ^ n * v n) := by
  intro n
  rw [h n]
  ring

/-- The march-level form: `rec2` with coefficients `a' n = c1 n (-x) - c0 n`, the same `b`, from
seeds `(s0, -s1)` is `(-1)^n` times `rec2` with `a n = c1 n x + c0 n` from `(s0, s1)`. -/
theorem rec2_reflect (c1 c0 b : ℕ → ℝ) (x s0 s1 : ℝ) : ∀ n,
    rec2 (fun n => c1 n * (-x) - c0 n) b s0 (-s1) n
      = (-1) ^ n * rec2 (fun n => c1 n * x + c0 n) b s0 s1 n := by
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

/-- The kernel's southern march: the same difference form with `y = |x| = -x` in place of `x`,
`E⁻ = c1 - 1 - cb - c0` in place of `E⁺`, and seed `w_1 = -P_1(x) = ((α+β+2)/2) y - (α-β)/2`. -/
noncomputable def kernelMarchSouth (α β : ℕ) (y : ℝ) : ℕ → ℝ × ℝ :=
  diffMarch (fun k => c1 (k + 2) α β * (y - 1) + (c1 (k + 2) α β - 1 - cb (k + 2) α β - c0 (k + 2) α β))
    (fun k => cb (k + 2) α β) 1 (((α + β + 2) / 2) * y - ((α : ℝ) - β) / 2)

/-- **The southern march is the reflected row.** At `y = -x` the southern value lane is
`(-1)^(k+1) v_{k+1}` where `v` is the northern three-term march at `x`; the emit restores the
sign `(-1)^n` (`oddsign` in the kernel). -/
theorem south_march_eq_reflect (α β : ℕ) (x : ℝ) (k : ℕ) :
    (kernelMarchSouth α β (-x) k).1 = (-1) ^ (k + 1) * march α β x (k + 1) := by
  unfold kernelMarchSouth
  have hC : (fun k => c1 (k + 2) α β * (-x - 1) + (c1 (k + 2) α β - 1 - cb (k + 2) α β - c0 (k + 2) α β))
      = fun k => (c1 (k + 2) α β * (-x) - c0 (k + 2) α β) - 1 - cb (k + 2) α β := by
    funext k; ring
  have hs : ((α + β + 2) / 2) * (-x) - ((α : ℝ) - β) / 2
      = -(((α + β + 2) / 2) * x + ((α : ℝ) - β) / 2) := by ring
  rw [hC, hs, diff_march_eq_rec2]
  simp only
  rw [rec2_reflect, march_eq_rec2 α β x (k + 1)]

/-- The southern seed `D_1 = w_1 - w_0 = ((α+β+2)/2)(y - 1) + β`: the kernel's `cst = β`. -/
theorem seed_D1_south (α β : ℕ) (y : ℝ) :
    (((α + β + 2) / 2) * y - ((α : ℝ) - β) / 2) - 1 = ((α + β + 2) / 2) * (y - 1) + β := by ring

/-- Relation to `jacobi_reflect`: the reflected row is the `(β, α)` Jacobi polynomial at `-x`,
so the southern hemisphere marches `P_n^{(β,α)}(|x|)`; `c1`, `cb` are symmetric in `(α, β)` and
`c0` flips sign, which is the `c0 → -c0` of the kernel. -/
theorem march_reflect_eq_swapped (α β : ℕ) (x : ℝ) (n : ℕ) :
    (-1) ^ n * march α β x n = march β α (-x) n := by
  rw [march_eq_jacobi, march_eq_jacobi, jacobi_reflect]

/-! ### §3 Homogeneity of the `(v, D)` march -/

/-- One difference step is linear and homogeneous in `(v, D)`. -/
theorem diff_step_homogeneous (C B v D lam : ℝ) :
    C * (lam * v) + B * (lam * D) = lam * (C * v + B * D) := by ring

/-- **`diff_march_scaled`.** The difference march from seeds scaled by `λ` is `λ` times the
difference march, in both lanes, at every degree: the block renormalisation (an exact power of
two applied to both `v` and `D`, with the exponent moved to `ex`) commutes with the march. -/
theorem diff_march_scaled (C B : ℕ → ℝ) (s0 s1 lam : ℝ) : ∀ k,
    diffMarch C B (lam * s0) (lam * s1) k
      = (lam * (diffMarch C B s0 s1 k).1, lam * (diffMarch C B s0 s1 k).2) := by
  intro k
  induction k with
  | zero => simp only [diffMarch]; ext <;> simp only <;> ring
  | succ k ih =>
    simp only [diffMarch, ih]
    ext <;> simp only <;> ring

/-- The lane value `v · 2^ex` is unchanged when `v` and `D` are multiplied by `2^(-k)` and `ex`
raised by `k` (the kernel's `sc = 2^(127 - eb)`, `ex += eb - 127`). -/
theorem block_rescale (v D : ℝ) (ex k : ℤ) :
    lane (v * (2 : ℝ) ^ (-k)) (ex + k) = lane v ex
      ∧ lane (D * (2 : ℝ) ^ (-k)) (ex + k) = lane D ex :=
  ⟨lane_rescale v ex k, lane_rescale D ex k⟩

end GMasterMarch
