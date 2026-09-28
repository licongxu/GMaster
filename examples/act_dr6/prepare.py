"""Cache the ACT DR6 night f150 maps as numpy arrays for the ACT examples and benchmarks.

    python examples/act_dr6/prepare.py --data-dir /path/to/act_dr6.02_maps_standard \
        [--beam /path/to/coadd_pa4_f150_night_beam_tform_instant.txt] [--out act_cache]

Inputs are the public ACT DR6.02 maps (Naess et al. 2025, arXiv:2503.14451): the PA4 f150 night
source-free HEALPix coadd, its 4-way splits and the coadd inverse-variance map.  FITS reading is
slow at Nside 8192, so every example times the estimator only and reads these caches instead.

Outputs (float32 unless stated, under --out):
    act_{T,Q,U}_nside<N>.npy      Stokes I, Q, U of the coadd (map units), for each --nsides
    act_w_nside<N>.npy            footprint weight: finite, nonzero pixels, degraded by averaging
    act_set<k>_T_nside<N>.npy     Stokes I of split k at --split-nside         (with --splits)
    act_ivar_nside<N>.npy         coadd inverse variance, CAR -> HEALPix, max 1 (needs pixell)
    act_beam_f150_night.npy       beam transfer function b_ell, normalised (float64, with --beam)
"""
import argparse
import os
import time

import healpy as hp
import numpy as np
from astropy.io import fits

COADD = "act_dr6.02_std_AA_night_pa4_f150_4way_coadd_map_srcfree_healpix.fits"
SPLIT = "act_dr6.02_std_AA_night_pa4_f150_4way_set{k}_map_srcfree_healpix.fits"
IVAR = "act_dr6.02_std_AA_night_pa4_f150_4way_coadd_ivar.fits"
COLUMNS = {"T": "TEMPERATURE", "Q": "Q_POLARISATION", "U": "U_POLARISATION"}


def read_column(path, column):
    """One Stokes column of a HEALPix binary table, as float32, without reading the others."""
    with fits.open(path, memmap=True) as hdul:
        hdu = hdul[1]
        nside = int(hdu.header["NSIDE"])
        assert hdu.header["ORDERING"].strip() == "RING", hdu.header["ORDERING"]
        m = np.ascontiguousarray(hdu.data[column]).reshape(-1).astype(np.float32)
    assert m.size == 12 * nside ** 2, (m.size, nside)
    return m, nside


def degrade(m, nside):
    """`ud_grade` by averaging the child pixels (the identity at the native resolution)."""
    if hp.get_nside(m) == nside:
        return m
    return hp.ud_grade(m.astype(np.float64), nside, order_in="RING").astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", required=True, help="directory holding the ACT DR6.02 map files")
    ap.add_argument("--out", default="act_cache")
    ap.add_argument("--nsides", default="8192,4096,2048,1024,512,256")
    ap.add_argument("--splits", default="", help="comma-separated split indices, e.g. 0,1")
    ap.add_argument("--split-nside", type=int, default=4096)
    ap.add_argument("--beam", help="beam transfer-function text file (ell, b_ell)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    nsides = [int(s) for s in args.nsides.split(",")]
    t0 = time.perf_counter()

    coadd = os.path.join(args.data_dir, COADD)
    for tag, column in COLUMNS.items():
        m, native = read_column(coadd, column)
        if tag == "T":
            footprint = (np.isfinite(m) & (m != 0.0)).astype(np.float32)
            print(f"coadd: Nside {native}, footprint f_sky {footprint.mean():.4f}", flush=True)
            for n in nsides:
                if n <= native:
                    np.save(os.path.join(args.out, f"act_w_nside{n}.npy"), degrade(footprint, n))
        m = np.where(np.isfinite(m), m, 0.0).astype(np.float32)
        for n in nsides:
            if n <= native:
                np.save(os.path.join(args.out, f"act_{tag}_nside{n}.npy"), degrade(m, n))
        print(f"  {tag} cached at Nside {[n for n in nsides if n <= native]} "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)
        del m

    for k in [int(s) for s in args.splits.split(",") if s]:
        m, _ = read_column(os.path.join(args.data_dir, SPLIT.format(k=k)), "TEMPERATURE")
        m = degrade(np.where(np.isfinite(m), m, 0.0).astype(np.float32), args.split_nside)
        np.save(os.path.join(args.out, f"act_set{k}_T_nside{args.split_nside}.npy"), m)
        print(f"split {k} cached at Nside {args.split_nside}", flush=True)
    if args.splits:
        from pixell import enmap, reproject

        ivar = enmap.read_map(os.path.join(args.data_dir, IVAR))
        w = np.asarray(reproject.map2healpix(ivar, nside=args.split_nside, method="spline",
                                             order=1, spin=[0]), dtype=np.float64)
        w[~np.isfinite(w) | (w < 0)] = 0.0
        w /= w.max()
        np.save(os.path.join(args.out, f"act_ivar_nside{args.split_nside}.npy"), w.astype(np.float32))
        print(f"inverse-variance weight cached, f_sky(w>0) = {np.mean(w > 0):.4f}", flush=True)

    if args.beam:
        b = np.loadtxt(args.beam)
        np.save(os.path.join(args.out, "act_beam_f150_night.npy"), b[:, 1] / b[0, 1])
        print("beam cached", flush=True)
    print(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
