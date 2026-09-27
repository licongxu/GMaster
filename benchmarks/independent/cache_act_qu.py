"""Cache ACT DR6 Q and U. The FITS file has no E or B map."""
import time

import healpy as hp
import numpy as np
from astropy.io import fits

MAP = "/rds/datasets/act/act_dr6.02_maps_standard/act_dr6.02_std_AA_night_pa4_f150_4way_coadd_map_srcfree_healpix.fits"
OUT = ".qwen/tmp/act"
NSIDES = (8192, 4096, 2048, 1024, 512, 256)


def column(name):
    with fits.open(MAP, memmap=True) as hdul:
        hdu = hdul[1]
        nside = int(hdu.header["NSIDE"])
        m = np.ascontiguousarray(hdu.data[name]).reshape(-1).astype(np.float32)
    return np.where(np.isfinite(m), m, 0.0).astype(np.float32), nside


def main():
    t0 = time.perf_counter()
    for name, tag in (("Q_POLARISATION", "Q"), ("U_POLARISATION", "U")):
        m, nside = column(name)
        print(f"read {tag} nside={nside} in {time.perf_counter()-t0:.1f}s", flush=True)
        for target in NSIDES:
            mt = m if target == nside else hp.ud_grade(m.astype(np.float64), target).astype(np.float32)
            path = f"{OUT}/act_{tag}_nside{target}.npy"
            np.save(path, mt)
            print(f"  wrote {path}", flush=True)
        del m
    print(f"total {time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
