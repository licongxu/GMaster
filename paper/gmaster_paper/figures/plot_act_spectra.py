"""TT, EE, TE of the NILC maps: raw Delta-ell = 30 bandpowers.

Upper: D_ell. Lower: (D - D_NaMaster) / D_NaMaster on a linear axis,
with the cosmic-variance band of one bin.
"""
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter

from make_figures import BLACK, GREY, MN_W, TEX_RC

RED = "#D62728"
HERE = Path(__file__).resolve().parent
DATA = HERE / "cl_roundtrip" / "act_nilc_n2048.npz"
# SHTns a_lm of the same masked HEALPix fields (n_iter = 3, float32 recurrence): shtns_hp_nilc.py
SHT32 = HERE / "cl_roundtrip" / "act_nilc_shtns_hp_fp32.npz"
MASK = HERE.parents[2] / ".qwen" / "tmp" / "act_nilc" / "mask_n2048.npy"
ELL_MIN = 500
ELL_MAX = 4000
NLB = 30


def _style(ax, logy):
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=20))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=3.0, labelsize=8)
    ax.tick_params(which="minor", length=1.4)
    ax.set_xlim(ELL_MIN, ELL_MAX)


def _fsky():
    w = np.clip(np.load(MASK), 0, None)
    return float(np.mean(w > 0))


def _dell(ell, cl):
    return ell * (ell + 1) * cl / (2 * np.pi)


def _cv(z, ell, key, fsky):
    nmode = (2 * ell + 1) * fsky * NLB
    if key == "TE":
        var = (z["nm_TT"] * z["nm_EE"] + z["nm_TE"] ** 2) / nmode
        return np.sqrt(np.maximum(var, 0.0)) / np.maximum(np.abs(z["nm_TE"]), 1e-30)
    return np.sqrt(2.0 / nmode)


def _healpix_gm(z, alms=None):
    """Decouple stored HEALPix alms (GMaster's, or SHTns's from `alms`) with GMaster's M."""
    lmax = 3 * int(z["nside"]) - 1
    alms = z if alms is None else alms

    def bp(a, b, m):
        cl = hp.alm2cl(a, b, lmax=lmax)
        binned = np.array([cl[2 + NLB * i:2 + NLB * (i + 1)].mean() for i in range(m.shape[0])])
        return np.linalg.solve(m, binned)

    return {
        "TT": bp(alms["almT"], alms["almT"], z["mTT"]),
        "EE": bp(alms["almE"], alms["almE"], z["mEE"]),
        "TE": bp(alms["almT"], alms["almE"], z["mTE"]),
    }


def main():
    z = np.load(DATA)
    gm = _healpix_gm(z)
    fp = _healpix_gm(z, np.load(SHT32))
    ell = z["ell"]
    keep = (ell >= ELL_MIN) & (ell <= ELL_MAX)
    fsky = _fsky()
    curves = (
        (z, "nm", BLACK, 1.5, "-", "NaMaster"),
        (gm, None, RED, 1.1, "--", "GMaster"),
        (fp, None, GREY, 1.0, ":", "SHTns fp32"),
    )
    panels = (
        ("TT", r"$TT$", True, (50, 4000), (-0.05, 0.05)),
        ("EE", r"$EE$", False, (0, 90), (-0.05, 0.05)),
        ("TE", r"$TE$", False, (-160, 80), (-0.5, 0.5)),
    )
    with plt.rc_context(TEX_RC):
        fig, axes = plt.subplots(
            2, 3, figsize=(MN_W, 4.2), sharex=True, layout="constrained",
            height_ratios=(1.35, 1.0))
        for j, (key, title, logy, ylim, rlim) in enumerate(panels):
            ax, lo = axes[0, j], axes[1, j]
            dnm = _dell(ell, z["nm_" + key])
            band = _cv(z, ell, key, fsky)
            lo.fill_between(
                ell[keep], -band[keep], band[keep], color="#C8C8C8", lw=0,
                label="cosmic variance", zorder=0)
            lo.axhline(0, color=BLACK, lw=1.1, zorder=2)
            for src, prefix, color, lw, ls, label in curves:
                cl = src[key] if prefix is None else src[f"{prefix}_{key}"]
                d = _dell(ell, cl)
                ax.plot(ell[keep], d[keep], color=color, lw=lw, ls=ls, label=label)
                if prefix == "nm":
                    continue
                frac = (d - dnm) / dnm
                lo.plot(ell[keep], frac[keep], color=color, lw=lw, ls=ls, zorder=3)
            if key == "TE":
                ax.axhline(0, color="#888888", lw=0.4)
            _style(ax, logy)
            _style(lo, False)
            ax.set_ylim(*ylim)
            lo.set_ylim(*rlim)
            ax.set_title(title, fontsize=9)
            lo.set_xlabel(r"$\ell$", fontsize=9)
        axes[0, 0].set_ylabel(
            r"$D_\ell=\ell(\ell+1)C_\ell/2\pi\;[\mu\mathrm{K}^{2}]$", fontsize=9)
        axes[1, 0].set_ylabel(r"$\Delta D_\ell/D_\ell$", fontsize=9)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        h2, lab2 = axes[1, 0].get_legend_handles_labels()
        cv_handle = [(h, lab) for h, lab in zip(h2, lab2) if lab == "cosmic variance"]
        fig.legend(
            handles + [cv_handle[0][0]], labels + [cv_handle[0][1]],
            loc="outside upper center", ncol=4, frameon=False,
            fontsize=8, handlelength=2.2, columnspacing=0.9, borderaxespad=0.02)
        fig.savefig(HERE / "fig_act_spectra.pdf", dpi=300)
        fig.savefig(HERE / "fig_act_spectra.png", dpi=300)
        plt.close(fig)
    print(f"wrote fig_act_spectra  f_sky={fsky:.3f}")


if __name__ == "__main__":
    main()
