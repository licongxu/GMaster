"""End-to-end MASTER wall time for the letter (Table tab:e2e). Writes fig_scoreboard.pdf."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

HERE = Path(__file__).resolve().parent

# Milliseconds from paper/gmaster_letter/main.tex Table tab:e2e (96-core NaMaster reference).
ROWS = [
    (64, 0, 9, 2),
    (64, 2, 12, 2),
    (128, 0, 20, 1),
    (128, 2, 40, 3),
    (256, 0, 58, 4),
    (256, 2, 161, 8),
    (512, 0, 314, 17),
    (512, 2, 641, 30),
    (1024, 0, 1682, 102),
    (1024, 2, 3017, 212),
    (2048, 0, 10311, 1014),
    (2048, 2, 17767, 1244),
    (4096, 0, 71583, 7458),
    (4096, 2, None, 8805),
]


def style(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(48, 9000)
    ax.set_xticks([64, 128, 256, 512, 1024, 2048, 4096])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.set_xlabel(r"$N_{\mathrm{side}}$")
    ax.set_ylabel("wall time (ms)")
    ax.grid(True, which="both", linewidth=0.35, alpha=0.55)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, spin in zip(axes, (0, 2)):
        rows = [r for r in ROWS if r[1] == spin and r[2] is not None]
        x = np.array([r[0] for r in rows], dtype=float)
        nm = np.array([r[2] for r in rows], dtype=float)
        gm = np.array([r[3] for r in rows], dtype=float)
        ax.plot(x, nm, color="#333333", marker="s", ms=4.5, lw=1.2, label="NaMaster (96 cores)")
        ax.plot(x, gm, color="#CC0000", marker="o", ms=4.5, lw=1.2, label="GMaster")
        if spin == 2:
            ax.annotate("NaMaster absent\n(32-bit MCM indices)", xy=(4096, 8805), xytext=(1800, 2e4),
                        fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.6), ha="center")
        style(ax)
        ax.set_title(f"spin {spin}")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    out = HERE / "fig_scoreboard.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
