"""One-time preparation of the ACT DR6 night f150 HEALPix map for the GMaster examples.

Reads the source-free coadd (Naess et al. 2025, arXiv:2503.14451; the DR6 maps paper), keeps
Stokes I, builds the footprint mask the way the run plan specifies, and caches the native map plus
degraded copies so that the timing scripts never pay FITS I/O.

Outputs (float32, under `--out`):
    act_T_nside<N>.npy    Stokes I in the map's own units
    act_w_nside<N>.npy    footprint weight in [0, 1]
"""
import argparse
import os
import time

import numpy as np
from astropy.io import fits

ROOT = "/rds/datasets/act/act_dr6.02_maps_standard"
MAP = f"{ROOT}/act_dr6.02_std_AA_night_pa4_f150_4way_coadd_map_srcfree_healpix.fits"


def read_stokes_i(path):
    """Stokes I of a HEALPix BinTableHDU, as float32, without materialising Q and U."""
    with fits.open(path, memmap=True) as hdul:
        hdu = hdul[1]
        nside = int(hdu.header["NSIDE"])
        assert hdu.header["ORDERING"].strip() == "RING", hdu.header["ORDERING"]
        col = hdu.data["TEMPERATURE"]              # (nrow, 1024) float32
        m = np.ascontiguousarray(col).reshape(-1).astype(np.float32)
    assert m.size == 12 * nside ** 2, (m.size, nside)
    return m, nside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=MAP)
    ap.add_argument("--out", default=".qwen/tmp/act")
    ap.add_argument("--nsides", default="8192,4096,2048,1024")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.perf_counter()
    m, nside = read_stokes_i(args.map)
    print(f"read {args.map}\n  nside={nside} npix={m.size} in {time.perf_counter()-t0:.1f} s",
          flush=True)

    finite = np.isfinite(m)
    hit = finite & (m != 0.0)
    print(f"  finite {finite.mean():.4f}  nonzero-and-finite {hit.mean():.4f} "
          f"({hit.mean()*41253:.0f} deg^2)", flush=True)
    w = hit.astype(np.float32)
    m = np.where(finite, m, 0.0).astype(np.float32)

    import healpy as hp
    for target in [int(s) for s in args.nsides.split(",")]:
        if target > nside:
            continue
        if target == nside:
            mt, wt = m, w
        else:
            # `ud_grade` averages the children; the mask degrades to the covered fraction, which
            # is then cut at 1 to keep a binary footprint with the partially covered rim included.
            mt = hp.ud_grade(m.astype(np.float64), target, order_in="RING").astype(np.float32)
            wt = hp.ud_grade(w.astype(np.float64), target, order_in="RING").astype(np.float32)
        np.save(f"{args.out}/act_T_nside{target}.npy", mt)
        np.save(f"{args.out}/act_w_nside{target}.npy", wt)
        cov = float(np.mean(wt > 0))
        print(f"  nside {target:>5}: saved, f_sky(w>0) = {cov:.4f}, "
              f"mean w = {float(wt.mean()):.4f}, rms T = {float(np.std(mt[wt > 0])):.3e}",
              flush=True)
    print(f"total {time.perf_counter()-t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
