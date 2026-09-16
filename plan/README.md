# GMaster plan (act-dr6-demo) — as of 2026-09-16

## Ship status (done)
- ~4pp methods-trust Letter: `paper/gmaster_letter/` (+ rebuilt `main.pdf` ≤4 pages)
- Artifact-only timings notebook: `examples/namaster_vs_gmaster_timings.ipynb`
- Trust demos: `examples/act_dr6_tt_example/`, `examples/flamingo_y_tt_example/` (TT@4096; rms ~1.4e-5 / ~2.3e-7; ~9× vs NaMaster)
- CUDA/fair-CPU Letter sentence (PR #5): Dn march is not GPU-intrinsic; shipped kernel is CUDA; fair CPU needs a real port — **not** JAX CPU fallback timings

## Locked for Thu 12:45 (Letter-only)
- Claim: GPU required for *shipped* kernel; algo itself is not GPU-intrinsic
- No JAX-fallback numbers as "new algo on CPU"
- Rotate / small-patch / 1/φ counterexample: **parked** (optional later ≤½ page if Boris still wants it)
- Slack one-liner after Wrap/Code Pair green + merge

## Future spike — fair CPU Dn port (test plan; not required for Thu)
Only if Licong opens this board:
1. Port spin-0 folded analysis from `gmaster/_cuda/march_v2.cu` → C++/OpenMP (or Numba) marching `(v, D)` with same coeffs/binade — **not** `JAX_PLATFORMS=cpu`
2. Gate via e.g. `GMASTER_MARCH_V2_CPU=1`; CUDA remains default
3. Bench same ACT TT cache; warm last-of-2; nside **256 then 512** first; stop if CPU warm ≫ GPU×10 before 1024
4. Done = one table row + Letter sentence update with measured numbers (no invented timings)

## Out of scope / parked
- Stranger-README / `/home/lxu` NaMaster path polish
- Dual-GPU e2e package number; float32 stock-NaMaster deep dive
- nside=8192 / A&A SHT fig.4; Ben/Reinecke share
- FLAMINGO `#tsz_sbi` / CNC science in this Letter

## Next actions
1. Merge PR #5 into `act-dr6-demo`; pull on scratch
2. Thu 12:45: confirm Letter+notebook scope; CPU-Dn spike only if Boris pushes for a number
3. Optional: green-light WorkAssistant Slack one-liner after merge
