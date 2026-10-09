"""GMaster vs NaMaster decoupled bandpowers on the ACT DR6 maps of Fig. 1.

Existing spectra only (no NaMaster reruns), at the native Nside = 8192 of the maps:
  GMaster  float32 Wigner-d march, re-decoupled with the Gauss-Legendre-fixed MCM:
           .qwen/tmp/boris_phase4/cl/act_cl_gmaster_glducc_n8192_s{0,2}.npy (via cv_ratio.spectra)
  NaMaster 96 CPU cores (27 Sep board): .qwen/tmp/accuracy/march_rerun/hist/act_cl_namaster96_n8192_s{0,2}.npy
TT is the spin-0 cell, EE row 0 of the spin-2 cell. Bins are
NmtBin.from_lmax_linear(3 Nside - 1, 30): bin i covers ell = 2 + 30 i ... 31 + 30 i.
The grey band is the cosmic variance of one bandpower, sigma_CV/C (cv_ratio.py); the dashed
grey line is 0.1 sigma_CV/C. The lower panels show |C_GM/C_NM - 1|.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, MultipleLocator, NullFormatter

from cv_ratio import TARGET, leff, sigma_cv, spectra
from make_figures import BLACK, GREY, TEX_RC

RED = "#D62728"
HERE = __import__("pathlib").Path(__file__).resolve().parent
NSIDE = 8192
COL_W = 244.0 / 72.27


def _grid(ax):
    # Linear bins of 30, so a linear ell axis; minor grid every 1000 in ell, 2-9 x 10^k in y.
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color="#D9D9D9", lw=0.5)
    ax.grid(True, which="minor", color="#EEEEEE", lw=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=3.0, labelsize=8)
    ax.tick_params(which="minor", length=1.4)


def main():
    with plt.rc_context(TEX_RC):
        fig, axes = plt.subplots(
            2, 2, figsize=(COL_W, 2.45), sharex=True, layout="constrained",
            gridspec_kw=dict(height_ratios=(1, 1)))
        for j, (spin, name) in enumerate(((0, "TT"), (2, "EE"))):
            gm, nm, _ = spectra(NSIDE, spin)
            ell = leff(gm.size) / 1e3
            rel = np.abs(gm / nm - 1)
            top, bot = axes[0, j], axes[1, j]
            top.plot(ell, nm, color=BLACK, lw=1.8, label=r"NaMaster")
            top.plot(ell, gm, color=RED, lw=0.7, label=r"GMaster fp32")
            cv = sigma_cv(NSIDE, gm.size)
            bot.fill_between(ell, 1e-9, cv, color="#BBBBBB", alpha=0.5, lw=0,
                             label=r"$\sigma_{\mathrm{CV}}/C_\ell$")
            bot.plot(ell, cv, color=GREY, lw=0.6)
            bot.plot(ell, TARGET * cv, color=GREY, lw=0.6, ls="--",
                     label=rf"${TARGET:g}\,\sigma_{{\mathrm{{CV}}}}/C_\ell$")
            bot.plot(ell, rel, color=RED, lw=0.7, label=r"$\lvert C_\ell^{\mathrm{GM}}/C_\ell^{\mathrm{NM}}-1\rvert$")
            for ax in (top, bot):
                _grid(ax)
            top.set_title(name, fontsize=9)
            bot.set_xlabel(r"$\ell/10^{3}$", fontsize=9)
            bot.set_ylim(1e-7, 2e-1)
        axes[0, 0].set_ylabel(r"$C_\ell$", fontsize=9)
        axes[1, 0].set_ylabel(r"fractional", fontsize=9)
        axes[1, 1].set_xlim(0, 3 * NSIDE / 1e3)
        for ax in axes[:, 1]:
            ax.tick_params(labelleft=True)
        h0, l0 = axes[0, 0].get_legend_handles_labels()
        h1, l1 = axes[1, 0].get_legend_handles_labels()
        fig.legend(h0 + h1[::-1], l0 + l1[::-1], loc="outside upper center", ncol=3,
                   frameon=False, fontsize=7, handlelength=1.4, columnspacing=0.6,
                   borderaxespad=0.02)
        fig.savefig(HERE / "fig_act_cl.pdf", dpi=300)
        fig.savefig(HERE / "fig_act_cl.png", dpi=300)
        plt.close(fig)
    print("wrote fig_act_cl")


if __name__ == "__main__":
    main()
