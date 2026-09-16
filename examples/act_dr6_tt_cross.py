"""ACT DR6 TT split cross-spectrum: a noise-bias-free CMB power spectrum from real maps.

Two independent night-time PA4 f150 splits (Naess et al. 2025, arXiv:2503.14451), Stokes I,
inverse-variance weighted footprint (the coadd `ivar`, reprojected by `examples/act_dr6_prepare_splits.py`),
and the nominal night beam deconvolved through the mode-coupling matrix.  The cross-spectrum of
the two splits has no noise bias, so the acoustic peaks appear directly; the split auto-spectrum
is shown for the noise level.  GMaster (GPU) and NaMaster (CPU) run the same estimator on the same
inputs and only the estimator calls are timed.

Outputs under `--out`: cl_cross_gmaster.npy, cl_cross_namaster.npy, cl_auto_gmaster.npy,
bins.npy, cross.pdf, README.txt
"""
import argparse
import os
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/rds/datasets/act/act_dr6.02_maps_standard"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nside", type=int, default=4096)
    ap.add_argument("--cache", default=".qwen/tmp/act")
    ap.add_argument("--out", default="examples/act_dr6_tt_cross_example")
    ap.add_argument("--nlb", type=int, default=50)
    ap.add_argument("--n-iter", type=int, default=3)
    ap.add_argument("--skip-namaster", action="store_true")
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    nside, lmax, nlb = args.nside, 3 * args.nside - 1, args.nlb
    m0 = np.load(f"{args.cache}/act_set0_T_nside{nside}.npy").astype(np.float64)
    m1 = np.load(f"{args.cache}/act_set1_T_nside{nside}.npy").astype(np.float64)
    w = np.load(f"{args.cache}/act_ivar_nside{nside}.npy").astype(np.float64)
    beam = np.load(f"{args.cache}/act_beam_f150_night.npy")[: lmax + 1]
    print(f"nside={nside} lmax={lmax} nlb={nlb} f_sky(w>0)={np.mean(w > 0):.4f} "
          f"<w>={w.mean():.4f} b(lmax)={beam[-1]:.2e}", flush=True)

    import jax
    jax.config.update("jax_enable_x64", True)
    import gmaster as gm

    def pipeline(mod, block=lambda x: x):
        f0 = mod.NmtField(w, [m0], beam=beam, n_iter=args.n_iter)
        f1 = mod.NmtField(w, [m1], beam=beam, n_iter=args.n_iter)
        b = mod.NmtBin.from_lmax_linear(lmax, nlb)
        ws = mod.NmtWorkspace()
        ws.compute_coupling_matrix(f0, f0, b)          # same weight on both splits
        cross = ws.decouple_cell(mod.compute_coupled_cell(f0, f1))[0]
        auto = ws.decouple_cell(mod.compute_coupled_cell(f0, f0))[0]
        return np.asarray(block(cross)), np.asarray(block(auto)), np.asarray(b.get_effective_ells())

    t_runs = []
    for r in range(args.repeats):
        t0 = time.perf_counter()
        cross_gm, auto_gm, ell = pipeline(gm, jax.block_until_ready)
        t_runs.append(time.perf_counter() - t0)
        print(f"GMaster run {r}: {t_runs[-1]:.2f} s", flush=True)
    t_gm = t_runs[-1]

    t_nm, cross_nm = None, np.full_like(cross_gm, np.nan)
    if not args.skip_namaster:
        import pymaster as nmt
        from gmaster._nmt_bin64 import patch_pymaster
        patch_pymaster()
        t0 = time.perf_counter()
        cross_nm, auto_nm, _ = pipeline(nmt)
        t_nm = time.perf_counter() - t0
        print(f"NaMaster: {t_nm:.2f} s on {len(os.sched_getaffinity(0))} cores", flush=True)

    np.save(f"{args.out}/cl_cross_gmaster.npy", cross_gm)
    np.save(f"{args.out}/cl_cross_namaster.npy", cross_nm)
    np.save(f"{args.out}/cl_auto_gmaster.npy", auto_gm)
    np.save(f"{args.out}/bins.npy", ell)
    ratio = cross_gm / cross_nm - 1.0
    rms = float(np.nanstd(ratio))
    t_nm_s = "N/A" if t_nm is None else f"{t_nm:.3f}s"

    dl = ell * (ell + 1) / (2 * np.pi)
    sel = ell <= 6000
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 5.5), gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(ell[sel], dl[sel] * cross_gm[sel], label=f"split0 × split1, GMaster (GPU) {t_gm:.1f} s")
    if t_nm is not None:
        ax[0].plot(ell[sel], dl[sel] * cross_nm[sel], "--",
                   label=f"split0 × split1, NaMaster ({len(os.sched_getaffinity(0))} cores) {t_nm:.0f} s")
    ax[0].plot(ell[sel], dl[sel] * auto_gm[sel], ":", color="grey", label="split0 auto (signal + noise)")
    ax[0].set_yscale("log")
    ax[0].set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi\ [\mu\mathrm{K}^2]$")
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"ACT DR6 night PA4 f150, TT, ivar-weighted, beam-deconvolved, nside {nside}")
    ax[1].axhline(0, color="k", lw=0.5)
    if t_nm is not None:
        ax[1].plot(ell[sel], ratio[sel])
        ax[1].set_ylabel("GM/NM − 1")
    ax[1].set_xlabel(r"$\ell$")
    fig.tight_layout()
    fig.savefig(f"{args.out}/cross.pdf")

    with open(f"{args.out}/README.txt", "w") as fh:
        fh.write(
            f"MAPS={ROOT}/act_dr6.02_std_AA_night_pa4_f150_4way_set{{0,1}}_map_srcfree_healpix.fits\n"
            f"WEIGHT={ROOT}/act_dr6.02_std_AA_night_pa4_f150_4way_coadd_ivar.fits (CAR -> HEALPix, "
            f"bilinear, normalised), same array for both estimators and both splits\n"
            f"BEAM=act_dr6.02_beams/main_beams/nominal/coadd_pa4_f150_night_beam_tform_instant.txt\n"
            f"nside={nside} lmax={lmax} nlb={nlb} n_iter={args.n_iter} spin=0 f_sky(w>0)={np.mean(w > 0):.4f}\n"
            f"t_gmaster={t_gm:.3f}s (runs: {', '.join(f'{t:.3f}' for t in t_runs)}; first includes JIT)\n"
            f"t_namaster={t_nm_s} (cores={len(os.sched_getaffinity(0))})\n"
            f"cross ratio GM/NM-1: rms={rms:.3e} max|.|={float(np.nanmax(np.abs(ratio))):.3e}\n"
            f"timed: 2 fields + coupling matrix + 2 coupled cells + 2 decouples; I/O excluded\n"
        )
    print("ship:", args.out, "ratio rms", rms, "t_gm", t_gm, "t_nm", t_nm_s, flush=True)


if __name__ == "__main__":
    main()
