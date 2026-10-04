"""ACT DR6 TT overlay for the letter. Writes fig_act_overlay.pdf."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EXAMPLE = REPO / "examples" / "act_dr6_tt_example"

F_SKY = 0.481
DELTA_ELL = 50.0


def cosmic_variance_fraction(ell):
    return np.sqrt(2.0 / ((2.0 * ell + 1.0) * F_SKY * DELTA_ELL))


def main():
    ell = np.load(EXAMPLE / "bins.npy")
    cl_gm = np.load(EXAMPLE / "cl_gmaster.npy")
    cl_nm = np.load(EXAMPLE / "cl_namaster.npy")
    ratio = cl_gm / cl_nm - 1.0
    dl = ell * (ell + 1) / (2 * np.pi)
    cv = cosmic_variance_fraction(ell)

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.35))

    ax = axes[0]
    ax.plot(ell, dl * cl_gm, color="#CC0000", lw=1.1, label="GMaster")
    ax.plot(ell, dl * cl_nm, color="#333333", lw=1.0, ls="--", label="NaMaster")
    ax.set_yscale("log")
    ax.set_ylabel(r"$D_\ell^{\mathrm{TT}}$")
    ax.set_xlabel(r"$\ell$")
    ax.legend(frameon=False, fontsize=8)
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.grid(True, which="major", linewidth=0.45, alpha=0.65)
    ax.grid(True, which="minor", linewidth=0.25, alpha=0.45)

    ax = axes[1]
    ax.axhline(0, color="#333333", lw=0.6)
    ax.plot(ell, ratio, color="#CC0000", lw=0.9, label=r"$C_\ell^{\mathrm{GM}}/C_\ell^{\mathrm{NM}}-1$")
    ax.plot(
        ell,
        cv,
        color="#666666",
        lw=0.8,
        ls=":",
        label=r"$\sqrt{2/[(2\ell+1)f_{\mathrm{sky}}\Delta_\ell]}$",
    )
    ax.plot(ell, -cv, color="#666666", lw=0.8, ls=":")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel("fractional difference")
    ax.legend(frameon=False, fontsize=7)
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.grid(True, which="major", linewidth=0.45, alpha=0.65)
    ax.grid(True, which="minor", linewidth=0.25, alpha=0.45)

    fig.tight_layout()
    out = HERE / "fig_act_overlay.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
