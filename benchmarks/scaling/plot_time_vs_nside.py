"""Warm wall time versus Nside: GMaster, NaMaster (32 and 96 cores) and SHTns.

Reads `results.json` (next to this file) and writes `time_vs_nside.png`.  Rows: `alm2map`,
one `map2alm` pass, and the end-to-end MASTER estimator on ACT DR6; columns: spin 0 and 2.
Markers are the mean of the warm runs; bars span their min and max; hollow markers are cells
with a single (median) value.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

matplotlib.rcParams.update({"font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8,
                            "xtick.labelsize": 8, "ytick.labelsize": 8})

HERE = Path(__file__).resolve().parent
NSIDES = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
SERIES = [  # (label, selector, colour, marker, horizontal dodge in octaves)
    ("NaMaster, 32 cores", lambda r: r["engine"] == "namaster" and r.get("cores") == 32, "#555555", "s", -0.12),
    ("NaMaster, 96 cores", lambda r: r["engine"] == "namaster" and r.get("cores") == 96, "#111111", "D", -0.04),
    ("GMaster (one GPU)", lambda r: r["engine"] == "gmaster", "#0072B2", "o", 0.04),
    ("SHTns fp64 (GPU)", lambda r: r["engine"] == "shtns", "#D55E00", "^", 0.12),
]


def draw(ax, rows, label, colour, marker, dodge):
    rows = sorted(rows, key=lambda r: r["nside"])
    if not rows:
        return
    x = np.array([r["nside"] for r in rows]) * 2.0 ** dodge
    t = [np.asarray(r["seconds"]) for r in rows]
    mean = np.array([v.mean() for v in t])
    err = np.array([[m - v.min() for m, v in zip(mean, t)], [v.max() - m for m, v in zip(mean, t)]])
    ax.errorbar(x, mean, yerr=err, color=colour, marker=marker, markersize=4.5, linewidth=1.0,
                capsize=2.0, elinewidth=0.7, label=label)
    single = np.array([len(v) == 1 for v in t])
    if single.any():
        ax.plot(x[single], mean[single], linestyle="None", marker=marker, markersize=4.5,
                markerfacecolor="white", markeredgecolor=colour, zorder=5)


def style(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(48, 12000)
    ax.set_xticks(NSIDES)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.set_xlabel(r"$N_{\rm side}$")
    ax.set_ylabel("wall time (s)")
    ax.grid(True, which="major", linewidth=0.3, alpha=0.5)


def main():
    data = json.loads((HERE / "results.json").read_text())
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.6), sharex=True)
    for col, spin in enumerate((0, 2)):
        for row, op in enumerate(("alm2map", "map2alm")):
            cells = [r for r in data["transforms"] if r["spin"] == spin and r["op"] == op]
            for label, pick, colour, marker, dodge in SERIES:
                shown = label if not (label.startswith("SHTns") and spin == 2) else "SHTns vector (spin 1), fp64"
                draw(axes[row, col], [r for r in cells if pick(r)], shown, colour, marker, dodge)
        cells = [r for r in data["end_to_end"] if r["spin"] == spin]
        for label, pick, colour, marker, dodge in SERIES[:3]:
            draw(axes[2, col], [r for r in cells if pick(r)], label, colour, marker, dodge)
        for ax in axes[:, col]:
            style(ax)
        axes[0, col].set_title(f"spin {spin}", pad=14)
    for row, title in enumerate([r"alm2map", r"map2alm (one pass)", r"end-to-end $C_\ell$, ACT DR6"]):
        axes[row, 0].annotate(title, xy=(1.12, 1.04), xycoords="axes fraction", ha="center", fontsize=9)
    handles, labels = [], []
    for ax in axes.ravel():
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.14, hspace=0.45, wspace=0.28)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Warm wall time (bars: min and max of the warm runs; hollow: single median)",
                 fontsize=10, y=0.985)
    out = HERE / "time_vs_nside.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
