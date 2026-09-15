"""ACT DR6 TT trust overlay: GMaster (GPU) against NaMaster (CPU) on a real map.

Map: the DR6 night-time PA4 f150 source-free coadd (Naess et al. 2025, arXiv:2503.14451), Stokes I,
HEALPix nside 8192, cached (and degraded with `hp.ud_grade`) by `examples/act_dr6_prepare.py`.
Mask: the observed footprint (finite, non-zero pixels), the same array for both estimators.
Only the estimator calls are timed; FITS / cache I/O is outside the clocks.

Outputs under `--out`:
    cl_gmaster.npy, cl_namaster.npy, bins.npy, overlay.pdf, README.txt

`--skip-namaster` writes `cl_namaster.npy` as NaN and `t_namaster=N/A`; no ratio is invented.
"""
import argparse
import os
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/rds/datasets/act/act_dr6.02_maps_standard"
MAP = f"{ROOT}/act_dr6.02_std_AA_night_pa4_f150_4way_coadd_map_srcfree_healpix.fits"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nside", type=int, default=4096)
    ap.add_argument("--cache", default=".qwen/tmp/act")
    ap.add_argument("--T", default=None, help="Stokes I npy; default {cache}/act_T_nside{nside}.npy")
    ap.add_argument("--w", default=None, help="weight npy; default {cache}/act_w_nside{nside}.npy")
    ap.add_argument("--full-sky", action="store_true", help="ignore --w and use ones")
    ap.add_argument("--map-path", default=MAP)
    ap.add_argument("--out", default="examples/act_dr6_tt_example")
    ap.add_argument("--nlb", type=int, default=50)
    ap.add_argument("--n-iter", type=int, default=3)
    ap.add_argument("--skip-namaster", action="store_true")
    ap.add_argument("--exact", action="store_true",
                    help="GMaster's exact float64 route (no float32 march) instead of the default")
    ap.add_argument("--repeats", type=int, default=2,
                    help="GMaster runs; the first includes JIT compilation, the last is reported")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t_path = args.T or f"{args.cache}/act_T_nside{args.nside}.npy"
    m = np.load(t_path).astype(np.float64)
    if args.full_sky:
        w = np.ones_like(m)
    else:
        w_path = args.w or f"{args.cache}/act_w_nside{args.nside}.npy"
        w = np.load(w_path).astype(np.float64)
    nside, lmax, nlb = args.nside, 3 * args.nside - 1, args.nlb
    print(f"nside={nside} lmax={lmax} nlb={nlb} f_sky={float(np.mean(w > 0)):.4f}", flush=True)

    if args.exact:
        os.environ["GMASTER_MARCH_V2"] = "0"
    import jax
    jax.config.update("jax_enable_x64", True)
    import gmaster as gm
    if args.exact:
        gm.set_ring_precision("fp64")
        gm.set_coupling_precision("fp64")

    bins = gm.NmtBin.from_lmax_linear(lmax, nlb)
    ell = np.asarray(bins.get_effective_ells())

    def gmaster_once():
        f = gm.NmtField(w, [m], n_iter=args.n_iter)
        ws = gm.NmtWorkspace()
        ws.compute_coupling_matrix(f, f, bins)
        cl = ws.decouple_cell(gm.compute_coupled_cell(f, f))[0]
        return np.asarray(jax.block_until_ready(cl))

    t_gm_runs = []
    for r in range(args.repeats):
        t0 = time.perf_counter()
        cl_gm = gmaster_once()
        t_gm_runs.append(time.perf_counter() - t0)
        print(f"GMaster run {r}: {t_gm_runs[-1]:.2f} s", flush=True)
    t_gm = t_gm_runs[-1]

    if args.skip_namaster:
        cl_nm, t_nm = np.full_like(cl_gm, np.nan), None
    else:
        import pymaster as nmt
        from gmaster._nmt_bin64 import patch_pymaster
        patch_pymaster()
        b_nm = nmt.NmtBin.from_lmax_linear(lmax, nlb)
        t0 = time.perf_counter()
        f_nm = nmt.NmtField(w, [m], n_iter=args.n_iter)
        ws = nmt.NmtWorkspace()
        ws.compute_coupling_matrix(f_nm, f_nm, b_nm)
        cl_nm = ws.decouple_cell(nmt.compute_coupled_cell(f_nm, f_nm))[0]
        t_nm = time.perf_counter() - t0
        print(f"NaMaster: {t_nm:.2f} s on {len(os.sched_getaffinity(0))} cores", flush=True)

    np.save(f"{args.out}/cl_gmaster.npy", cl_gm)
    np.save(f"{args.out}/cl_namaster.npy", cl_nm)
    np.save(f"{args.out}/bins.npy", ell)
    ratio = cl_gm / cl_nm - 1.0
    rms = float(np.nanstd(ratio))
    # noise reference: rms of the fractional scatter expected from the bandpower itself is not
    # available without a covariance, so the README reports the plain rms and the max over bins.
    t_nm_s = "N/A" if t_nm is None else f"{t_nm:.3f}s"

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 5), gridspec_kw={"height_ratios": [2, 1]})
    dl = ell * (ell + 1) / (2 * np.pi)
    ax[0].plot(ell, dl * cl_gm, label=f"GMaster (GPU) {t_gm:.2f} s")
    if t_nm is not None:
        ax[0].plot(ell, dl * cl_nm, "--", label=f"NaMaster ({len(os.sched_getaffinity(0))} cores) {t_nm:.1f} s")
    ax[0].set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$")
    ax[0].set_yscale("log")
    ax[0].legend()
    ax[0].set_title(f"{args.map_path.rsplit('/', 1)[-1]}, TT, nside {nside}, nlb {nlb}")
    ax[1].axhline(0, color="k", lw=0.5)
    if t_nm is not None:
        ax[1].plot(ell, ratio)
        ax[1].set_ylabel("GM/NM − 1")
    else:
        ax[1].text(0.5, 0.5, "NaMaster N/A at this nside", ha="center", transform=ax[1].transAxes)
    ax[1].set_xlabel(r"$\ell$")
    fig.tight_layout()
    fig.savefig(f"{args.out}/overlay.pdf")

    with open(f"{args.out}/README.txt", "w") as fh:
        fh.write(
            f"MAP={args.map_path}\nT={t_path}\n"
            f"MASK={'full-sky ones' if args.full_sky else 'weight npy, same array for both estimators'}, "
            f"f_sky={float(np.mean(w > 0)):.4f}\n"
            f"nside={nside} lmax={lmax} nlb={nlb} n_iter={args.n_iter} spin=0 "
            f"route={'exact fp64' if args.exact else 'default (float32 march)'}\n"
            f"t_gmaster={t_gm:.3f}s (runs: {', '.join(f'{t:.3f}' for t in t_gm_runs)}; "
            f"first includes JIT compilation)\n"
            f"t_namaster={t_nm_s} (cores={len(os.sched_getaffinity(0))})\n"
            f"ratio GM/NM-1: rms={rms:.3e} max|.|={float(np.nanmax(np.abs(ratio))):.3e}\n"
            f"timed: field + coupling matrix + coupled cell + decouple; I/O excluded\n"
        )
    print("ship:", args.out, "ratio rms", rms, "t_gm", t_gm, "t_nm", t_nm_s, flush=True)


if __name__ == "__main__":
    main()
