# GMaster ship plan (Boris lock, Wed ~12:45)

Locked scope for the next public-facing drop on `act-dr6-demo`. Everything else is parked.

## Ship now

1. **MNRAS Letter (4 pages)** — `paper/gmaster_letter/main.tex` / `main.pdf`; Overleaf `6a9b7e6f30290767963c6afe`
   - Verifiable problem: masked pseudo-\(C_\ell\) (MASTER) on GPU vs NaMaster CPU reference.
   - Key equations and pipeline steps (map \(\leftrightarrow\) alms, coupling, decouple).
   - Short agentic-methods sketch (how the implementation was built and checked).
   - Trust evidence only: ACT DR6 TT + FLAMINGO L2p8 \(y\) overlays, agreement ratios, wall-clock.
   - Frame as **methods trust** (numerical agreement + timing), **not** cosmology or science interpretation.
   - Appendix: small, copy-pasteable code snippets.
   - Wire figures from committed artifacts under `examples/act_dr6_tt_example/` and `examples/flamingo_y_tt_example/` (or regenerate plots from saved `{bins,cl_gmaster,cl_namaster}.npy` only).

2. **GitHub notebook** — `examples/namaster_vs_gmaster_timings.ipynb`
   - Loads the five-file sets in each example directory (no new MASTER / NaMaster runs).
   - Overlay + \((C_\ell^{\mathrm{GM}}/C_\ell^{\mathrm{NM}} - 1)\) panels.
   - Timing / RMS table parsed from each `README.txt`.
   - One prose cell: methods trust only; ACT native `nside=8192` degraded to 4096 for the demo; FLAMINGO full-sky at `nside=4096`.

## Evidence on disk (use as-is)

| Case | rms(GM/NM−1) | GMaster (warm) | NaMaster (192 cores) |
|------|--------------|----------------|----------------------|
| ACT DR6 TT @ 4096 | ~1.4×10⁻⁵ | ~7.6 s | ~70 s |
| FLAMINGO L2p8 \(y\) @ 4096 | ~2.3×10⁻⁷ | ~8.6 s | ~70 s |

TT at 4096 **did not** require “NaMaster N/A” for this demo — both estimators ran.

## Parked (explicitly out of scope)

- Stranger-README polish and “public package stranger-ship complete” claims
- fp32-stock-NaMaster probe
- Dual-GPU end-to-end package numbers
- `nside=8192` board and A&A fig. 4 reproduction
- Ben / Reinecke outreach
- FLAMINGO `tsz_sbi` science, CNC/HILC, feedback science
- Any new end-to-end MASTER / NaMaster runs on ACT or FLAMINGO maps

## Done checklist

- [x] `plan/README.md` matches this scope
- [x] `paper/gmaster_letter/main.pdf` (4 pp MNRAS Letter, difference-form march + ACT/FLAMINGO timings)
- [x] `examples/namaster_vs_gmaster_timings.ipynb` loads artifacts only
- [x] PR on `act-dr6-demo` (or branch off it)

## Reproduce figures from committed npy (no MASTER rerun)

```bash
# optional: regenerate overlay.pdf from saved arrays only
python examples/act_dr6_tt_overlay.py --skip-namaster  # needs cached map; skip if using shipped npy
# notebook path: load cl_*.npy + bins.npy + README.txt directly
```
