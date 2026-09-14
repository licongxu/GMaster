/-
Run with `~/.elan/bin/lake env lean CheckAxioms.lean` after `lake build`: prints the axioms each
main theorem depends on.  Expected: `[propext, Classical.choice, Quot.sound]` for every line and no
`sorryAx`.
-/
import GMasterMarch

#print axioms GMasterMarch.wignerD_eq_jacobiForm
#print axioms GMasterMarch.jacobi_three_term
#print axioms GMasterMarch.march_eq_jacobi
#print axioms GMasterMarch.march_scaled
#print axioms GMasterMarch.flush_bound
#print axioms GMasterMarch.binade_split
#print axioms GMasterMarch.two_roundings
#print axioms GMasterMarch.row0_reflect
#print axioms GMasterMarch.fold_even
#print axioms GMasterMarch.mirror_identity
#print axioms GMasterMarch.time_lower_bound
#print axioms GMasterMarch.mlim_is_root
#print axioms GMasterMarch.skipped_degrees
#print axioms GMasterMarch.diff_march_eq_march
#print axioms GMasterMarch.south_march_eq_reflect
#print axioms GMasterMarch.reflect_recurrence
#print axioms GMasterMarch.diff_march_scaled
#print axioms GMasterMarch.emit_factorisation
#print axioms GMasterMarch.step_bound_ge
