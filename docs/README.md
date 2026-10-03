# Documentation

* [architecture.md](architecture.md): how the package is organised, how a power spectrum is
  computed, the transform engines and how one is chosen, precision, memory, environment
  variables, native code, tests.
* [notes/](notes/): technical notes for readers interested in the algorithms.
  * [latitudinal_march_maths.md](notes/latitudinal_march_maths.md) (and `.pdf`): the table-free
    Wigner-d march, its closed form, recurrence, exponent handling, emit step, parity fold,
    arithmetic cost and polar skip.
  * [march_v2_maths.md](notes/march_v2_maths.md): the difference form of the recurrence used by
    the CUDA march.
  * [dc_latitudinal.md](notes/dc_latitudinal.md) and
    [dc_latitudinal_study_note.pdf](notes/dc_latitudinal_study_note.pdf): the divide-and-conquer
    latitudinal transform, its complexity and measurements.
  * [three_way_gpu.md](notes/three_way_gpu.md): GMaster's two GPU engines against SHTns.
  * [lensing_qe.md](notes/lensing_qe.md): CMB lensing quadratic estimators (falafel's) and their
    normalisation on the GPU, validation against falafel / tempura / lensed sims, and timings.
* [figures/](figures/): the figures used by the README and the notes.

The Lean 4 proofs of the identities behind the march are in [../formal/lean](../formal/lean/).
The public API follows NaMaster's; its [documentation](https://namaster.readthedocs.io) applies,
and every public GMaster function has a NumPy-style docstring.
