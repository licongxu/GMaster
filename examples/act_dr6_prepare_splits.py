"""Prepare the ACT DR6 night f150 4-way splits for a cross-spectrum example.

Caches (float32, under `--out`, at `--nside`, degraded from the native nside 8192 by averaging):
    act_set<k>_T_nside<N>.npy   Stokes I of split k (source-free coadd of that split)
    act_ivar_nside<N>.npy       inverse-variance weight of the full coadd, CAR -> HEALPix by
                                bilinear interpolation at the pixel centres, normalised to max 1
    act_beam_f150_night.npy     normalised beam transfer function b_ell of the night coadd
"""
import argparse
import os
import time

import healpy as hp
import numpy as np

from act_dr6_prepare import ROOT, read_stokes_i

BEAM = ("/rds/datasets/act/act_dr6.02_beams/main_beams/nominal/"
        "coadd_pa4_f150_night_beam_tform_instant.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".qwen/tmp/act")
    ap.add_argument("--nside", type=int, default=4096)
    ap.add_argument("--splits", default="0,1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.perf_counter()

    for k in [int(s) for s in args.splits.split(",")]:
        path = f"{ROOT}/act_dr6.02_std_AA_night_pa4_f150_4way_set{k}_map_srcfree_healpix.fits"
        m, nside = read_stokes_i(path)
        m = np.where(np.isfinite(m), m, 0.0).astype(np.float64)
        if args.nside != nside:
            m = hp.ud_grade(m, args.nside, order_in="RING")
        np.save(f"{args.out}/act_set{k}_T_nside{args.nside}.npy", m.astype(np.float32))
        print(f"split {k}: nside {nside} -> {args.nside}, {time.perf_counter()-t0:.0f} s", flush=True)

    from pixell import enmap, reproject
    ivar = enmap.read_map(f"{ROOT}/act_dr6.02_std_AA_night_pa4_f150_4way_coadd_ivar.fits")
    w = reproject.map2healpix(ivar, nside=args.nside, method="spline", order=1, spin=[0])
    w = np.asarray(w, dtype=np.float64)
    w[~np.isfinite(w) | (w < 0)] = 0.0
    w /= w.max()
    np.save(f"{args.out}/act_ivar_nside{args.nside}.npy", w.astype(np.float32))
    print(f"ivar weight: f_sky(w>0) = {np.mean(w > 0):.4f}, mean w = {w.mean():.4f}, "
          f"{time.perf_counter()-t0:.0f} s", flush=True)

    b = np.loadtxt(BEAM)
    bl = b[:, 1] / b[0, 1]
    np.save(f"{args.out}/act_beam_f150_night.npy", bl.astype(np.float64))
    print(f"beam: {bl.size} multipoles, b(3000) = {bl[3000]:.3f}, b(10000) = {bl[10000]:.3e}")


if __name__ == "__main__":
    main()
