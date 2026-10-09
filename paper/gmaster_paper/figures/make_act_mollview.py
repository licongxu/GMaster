"""Three separate mollweide panels of the ACT DR6 T, Q, U coadd.

Native HEALPix Nside 8192, averaged to Nside 512 for display.
Unobserved pixels (not finite, or zero in Stokes I) are grey.
The paper stacks the three PDFs; this script does not composite them.
"""
import os

import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

FITS = (
    "/rds/datasets/act/act_dr6.02_maps_standard/"
    "act_dr6.02_std_AA_night_pa4_f150_4way_coadd_map_srcfree_healpix.fits"
)
SHOW_NSIDE = 512
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = "/tmp/act_tqu_nside512.npz"
PANELS = (
    ("TEMPERATURE", "fig_act_T", r"$T$", 100.0),
    ("Q_POLARISATION", "fig_act_Q", r"$Q$", 20.0),
    ("U_POLARISATION", "fig_act_U", r"$U$", 20.0),
)


def read_column(path, column):
    with fits.open(path, memmap=True) as hdul:
        hdu = hdul[1]
        nside = int(hdu.header["NSIDE"])
        m = np.ascontiguousarray(hdu.data[column]).reshape(-1).astype(np.float32)
    return m, nside


def load_maps():
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return z["maps"]
    t_native, nside = read_column(FITS, "TEMPERATURE")
    footprint = np.isfinite(t_native) & (t_native != 0.0)
    print(f"native nside {nside}  f_sky {footprint.mean():.4f}", flush=True)
    del t_native
    maps = []
    for column, _stem, _title, _lim in PANELS:
        m, _ = read_column(FITS, column)
        m = np.where(np.isfinite(m), m, 0.0).astype(np.float32)
        m = hp.ud_grade(m, SHOW_NSIDE)
        w = hp.ud_grade(footprint.astype(np.float32), SHOW_NSIDE)
        m = m.astype(np.float64)
        m[w <= 0.5] = hp.UNSEEN
        maps.append(m.astype(np.float32))
    maps = np.stack(maps)
    np.savez(CACHE, maps=maps)
    return maps


def main():
    mpl.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath}",
        "font.family": "serif",
        "pdf.fonttype": 42,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })
    maps = load_maps()
    # Each panel prints at 0.32\textwidth (~2.3 in); draw at that size so text stays legible.
    fontsize = {"title": 11}
    for m, (_column, stem, title, lim) in zip(maps, PANELS):
        plt.close("all")
        fig = plt.figure(figsize=(2.4, 1.75))
        hp.mollview(
            m.astype(np.float64),
            fig=fig.number,
            title=title,
            cbar=False,
            min=-lim,
            max=lim,
            xsize=1200,
            fontsize=fontsize,
        )
        ax = plt.gca()
        cb = fig.colorbar(ax.get_images()[0], ax=ax, orientation="horizontal",
                          fraction=0.06, pad=0.03, shrink=0.6, aspect=18)
        cb.set_ticks([-lim, 0, lim])
        cb.ax.tick_params(labelsize=9, length=2, pad=1.5)
        cb.set_label(r"$\mu\mathrm{K}$", fontsize=10, labelpad=1)
        fig.savefig(os.path.join(HERE, stem + ".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(os.path.join(HERE, stem + ".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
        print("wrote", stem, flush=True)


if __name__ == "__main__":
    main()
