"""Peak memory versus Nside: GMaster (device), SHTns (device) and NaMaster (host).

Reads `results.json` (next to this file) and writes `memory_vs_nside.png`.  Top row: the
transforms; bottom row: the end-to-end MASTER estimator on ACT DR6.  GMaster's figure is XLA's
accounted device peak, SHTns's the growth of device memory in use over its cell, NaMaster's the
host peak RSS of its process.  One process per cell, so every peak is that cell's own.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

matplotlib.rcParams.update({"font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8,
                            "xtick.labelsize": 8, "ytick.labelsize": 8})

HERE = Path(__file__).resolve().parent
NSIDES = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
CARD_GIB = 95.6
STYLE = {"gmaster": ("GMaster (GPU)", "#0072B2", "o"),
         "shtns": ("SHTns (GPU)", "#D55E00", "^"),
         "namaster": ("NaMaster, 96 cores (host RSS)", "#111111", "D")}


def main():
    data = json.loads((HERE / "results.json").read_text())["memory"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), sharex=True)
    for col, spin in enumerate((0, 2)):
        for row, kind in enumerate(("transforms", "end_to_end")):
            ax = axes[row, col]
            for engine, (label, colour, marker) in STYLE.items():
                rows = sorted((r for r in data[kind] if r["engine"] == engine and r["spin"] == spin),
                              key=lambda r: r["nside"])
                if rows:
                    ax.plot([r["nside"] for r in rows], [r["peak_gib"] for r in rows], marker=marker,
                            color=colour, markersize=4.5, linewidth=1.0, label=label)
            ax.axhline(CARD_GIB, color="#999999", linewidth=0.7, linestyle="--", label="one GPU (96 GB)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(48, 12000)
            ax.set_xticks(NSIDES)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_locator(LogLocator(base=10))
            ax.set_xlabel(r"$N_{\rm side}$")
            ax.set_ylabel("peak memory (GiB)")
            ax.grid(True, which="major", linewidth=0.3, alpha=0.5)
        axes[0, col].set_title(f"spin {spin}", pad=14)
    axes[0, 0].annotate("transforms (alm2map + map2alm)", xy=(1.12, 1.04), xycoords="axes fraction",
                        ha="center")
    axes[1, 0].annotate(r"end-to-end $C_\ell$, ACT DR6", xy=(1.12, 1.04), xycoords="axes fraction",
                        ha="center")
    handles, labels = [], []
    for ax in axes.ravel():
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.17, hspace=0.45, wspace=0.28)
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Peak memory of one cell (one process per cell)", fontsize=10, y=0.985)
    out = HERE / "memory_vs_nside.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
