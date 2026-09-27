"""Pseudo-C_ell power spectra of your own HEALPix maps.

    python examples/power_spectrum_from_fits.py MAP.fits MASK.fits [options]

MAP.fits holds a temperature map (field 0) and optionally Q and U (fields 1 and 2); MASK.fits
holds the weight map (a binary mask is apodized on request).  The script computes the decoupled
bandpowers with GMaster and writes them to an .npz file; with --plot it also saves a figure.

Examples
    # TT only, 1-degree C1 apodization, bins of 20, 5 arcmin Gaussian beam deconvolved
    python examples/power_spectrum_from_fits.py cmb.fits mask.fits --apodize 1.0 --nlb 20 --fwhm 5

    # TT, TE, EE, BB (reads fields 0, 1, 2), cross-correlating two splits
    python examples/power_spectrum_from_fits.py split1.fits mask.fits --pol --cross split2.fits

Outputs (in --out, default `cl.npz`): `ell_eff`, one array per spectrum (`TT`, `TE`, `TB`, `EE`,
`EB`, `BE`, `BB` as available, in the map units squared), and `bandpower_windows_<pair>` for
comparing with theory.
"""

import argparse
import os
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")      # before JAX is imported

import healpy as hp
import numpy as np

import gmaster as nmt

SPECTRA = {(0, 0): ["TT"], (0, 2): ["TE", "TB"], (2, 2): ["EE", "EB", "BE", "BB"]}


def read_fields(path, pol):
    maps = hp.read_map(path, field=(0, 1, 2) if pol else 0)
    maps = np.atleast_2d(np.asarray(maps, dtype=np.float64))
    maps[~np.isfinite(maps) | (maps == hp.UNSEEN)] = 0.0
    return maps


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("map", help="HEALPix FITS map (I, or I Q U with --pol)")
    ap.add_argument("mask", help="HEALPix FITS mask / weight map")
    ap.add_argument("--cross", help="second map for a cross-spectrum (e.g. another data split)")
    ap.add_argument("--pol", action="store_true", help="also read Q and U (fields 1, 2)")
    ap.add_argument("--apodize", type=float, default=0.0, help="C1 apodization scale in degrees")
    ap.add_argument("--nlb", type=int, default=16, help="bandpower width in multipoles")
    ap.add_argument("--lmax", type=int, help="band limit of the fields and bins (default 3*nside-1; "
                    "HEALPix spectra are usually trusted up to ~2*nside)")
    ap.add_argument("--fwhm", type=float, default=0.0, help="Gaussian beam FWHM in arcmin")
    ap.add_argument("--purify-b", action="store_true", help="B-mode purification (needs a smooth mask)")
    ap.add_argument("--out", default="cl.npz")
    ap.add_argument("--plot", help="save a figure of the spectra to this path")
    args = ap.parse_args()

    maps1 = read_fields(args.map, args.pol)
    maps2 = read_fields(args.cross, args.pol) if args.cross else None
    nside = hp.get_nside(maps1[0])
    mask = hp.ud_grade(hp.read_map(args.mask), nside)
    if args.apodize > 0:
        mask = np.asarray(nmt.mask_apodization(mask, args.apodize, apotype="C1"))
    lmax = args.lmax or 3 * nside - 1
    beam = hp.gauss_beam(np.radians(args.fwhm / 60.0), lmax=lmax) if args.fwhm > 0 else None
    bins = nmt.NmtBin.from_lmax_linear(lmax, args.nlb)       # NaMaster: bins and fields share lmax
    print(f"Nside {nside}, f_sky {mask.mean():.3f}, {bins.get_n_bands()} bandpowers")

    t0 = time.perf_counter()

    def fields(maps):
        out = {0: nmt.NmtField(mask, maps[:1], beam=beam, lmax=lmax, lmax_mask=lmax)}
        if args.pol:
            out[2] = nmt.NmtField(mask, maps[1:3], beam=beam, purify_b=args.purify_b, lmax=lmax,
                                  lmax_mask=lmax)
        return out

    fa = fields(maps1)
    fb = fields(maps2) if maps2 is not None else fa
    result = {"ell_eff": np.asarray(bins.get_effective_ells())}
    for (s1, s2), names in SPECTRA.items():
        if s1 not in fa or s2 not in fb:
            continue
        workspace = nmt.NmtWorkspace.from_fields(fa[s1], fb[s2], bins)
        cl = np.asarray(workspace.decouple_cell(nmt.compute_coupled_cell(fa[s1], fb[s2])))
        result.update(zip(names, cl))
        result[f"bandpower_windows_{s1}{s2}"] = np.asarray(workspace.get_bandpower_windows())
    print(f"computed {', '.join(k for k in result if k.isupper())} in {time.perf_counter() - t0:.1f} s")
    np.savez(args.out, **result)
    print(f"wrote {args.out}")

    if args.plot:
        import matplotlib.pyplot as plt

        ell = result["ell_eff"]
        names = [k for k in ("TT", "TE", "EE", "BB") if k in result]
        fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.5), squeeze=False)
        for ax, name in zip(axes[0], names):
            ax.plot(ell, ell * (ell + 1) / (2 * np.pi) * result[name], "o-", ms=3)
            ax.set_title(name)
            ax.set_xlabel(r"$\ell$")
        axes[0, 0].set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
