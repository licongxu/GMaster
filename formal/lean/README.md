# GMasterMarch — Lean 4 proofs for the table-free latitudinal march

Proof of record for `docs/latitudinal_march_maths.md` and `docs/march_v2_maths.md` (the v2 difference-form march).  Built with the toolchain in
`lean-toolchain` against the pinned Mathlib (`lakefile.toml`).

```bash
~/.elan/bin/lake exe cache get   # once: Mathlib .olean cache
~/.elan/bin/lake build           # must succeed; there is no `sorry` in the project
~/.elan/bin/lake env lean CheckAxioms.lean   # axioms of every main theorem: propext, Classical.choice, Quot.sound
```

| file | section of the note | content |
|---|---|---|
| `GMasterMarch/Indices.lean` | §1 | `α+β = 2 max(m,s)`, `α²-β² = 4ms`, the factorial ratio in both branches (`ε_m` flip), the `(-1)^(m+s)` sign |
| `GMasterMarch/WignerJacobi.lean` | §1 | Wigner's explicit sum for `d^j_{m,-s}` equals the Jacobi closed form, both branches (`wignerD_eq_jacobiForm`) |
| `GMasterMarch/JacobiSum.lean` | §2 | the explicit Jacobi sum and DLMF 18.9.2 for it: summand ratio identities `tj_succ`/`tj_pred_v`/`tj_pred_u`, `tj_pred`/`tj_pred2`, the telescoping certificate `W`, `jacobi_three_term` |
| `GMasterMarch/Recurrence.lean` | §2 | seeds `P_0`, `P_1`, the exact rearrangement into the kernel's `(c1 x + c0) v - cb v'`, `den > 0`, `march = jacobi` |
| `GMasterMarch/ExponentCarry.lean` | §3 | `(v, ex)` lanes: rescaling is exact, the step is homogeneous, no overflow before the guard |
| `GMasterMarch/Emit.lean` | §4–§5 | flush bound `2^-71`, binade split, fractional factor in `[1,2)`, two-rounding bound, dropped partials |
| `GMasterMarch/Fold.lean` | §6 | symmetric-Jacobi parity from the sum, `row0(π-θ) = (-1)^(ℓ+m) row0(θ)`, ring-pair sums, the fold |
| `GMasterMarch/Mirror.lean` | §6 | general reflection `P_n^{(α,β)}(-x) = (-1)^n P_n^{(β,α)}(x)`, the spin-2 mirror identity for the Jacobi closed forms (`rowNeg_reflect`, `mirror_identity`) |
| `GMasterMarch/Wall.lean` | §7 | `Σ_{m<L}(L-m) = L(L+1)/2`, time lower bound from an FMA count and a rate |
| `GMasterMarch/PolarSkip.lean` | §8 | ducc0's `mlim` is the larger root of its quadratic, spin-0 form, degrees removed by a cutoff |
| `GMasterMarch/DiffForm.lean` | v2 note §1–§3 | difference form `(v, D)` of the three-term recurrence (`diff_march_eq_march`), the reflected southern row (`reflect_recurrence`, `south_march_eq_reflect`), homogeneity of the `(v, D)` march (`diff_march_scaled`) |
| `GMasterMarch/EmitV2.lean` | v2 note §4 | emit factorisation `v 2^ex N = v 2^(ex+⌊log₂N_b⌋) u`, the per-step ratio bounds in both `ε_m` branches and the range of `u` |

Not formalised (documented as such in the note): the WKB decay estimate behind the polar-skip
margin (§8).  The numerical cross-check `.qwen/tmp/closed_form_check_s36.py` remains as a sanity
test of the definitions.
Axioms of every theorem: `propext`, `Classical.choice`, `Quot.sound` only (`#print axioms`).
