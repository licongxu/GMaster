"""Accuracy panels for the letter. Writes fig_accuracy.pdf.

Left: float32 march vs float64 recurrence (Table tab:dform).
Right: |GM/NM - 1| on decoupled TT bandpowers from benchmarks/scaling/results.json.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RESULTS = ROOT / "benchmarks" / "scaling" / "results.json"

DFORM = {
    "Plain, single word": [4.2e-2, 8.8e-5, 3.1e-5],
    "Difference, 1-word $C$": [4.1e-5, 8.6e-5, 5.1e-5],
    "Difference, 2-word $C$ (shipped)": [2.8e-6, 2.9e-6, 1.6e-5],
}
M_LABELS = [r"$m=0$", r"$m=1000$", r"$m=L/2$"]


def main():
    data = json.loads(RESULTS.read_text())
    tt = sorted(
        (r for r in data["agreement"] if r["spin"] == 0 and r["spectrum"] == "TT"),
        key=lambda r: r["nside"],
    )
    nsides = np.array([r["nside"] for r in tt], dtype=float)
    mx_all = np.array([r["per_bandpower_max"] for r in tt])
    mx_lo = np.array([r["per_bandpower_max_below_2nside"] for r in tt])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    ax = axes[0]
    x = np.arange(len(M_LABELS))
    width = 0.25
    for i, (lab, vals) in enumerate(DFORM.items()):
        ax.bar(x + (i - 1) * width, vals, width, label=lab, color=["#888888", "#555555", "#CC0000"][i])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(M_LABELS)
    ax.set_ylabel(r"max relative error ($N_{\mathrm{side}}=1024$)")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(True, which="both", axis="y", linewidth=0.35, alpha=0.55)

    ax = axes[1]
    ax.plot(nsides, mx_lo, color="#CC0000", marker="o", ms=4, lw=1.2,
            label=r"max $|\mathrm{GM}/\mathrm{NM}-1|$, $\ell<2N_{\mathrm{side}}$")
    ax.plot(nsides, mx_all, color="#333333", marker="s", ms=4, lw=1.2, ls="--",
            label=r"max $|\mathrm{GM}/\mathrm{NM}-1|$, all bins")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$N_{\mathrm{side}}$")
    ax.set_ylabel(r"decoupled TT bandpowers (ACT DR6 board)")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.grid(True, which="both", linewidth=0.35, alpha=0.55)
    ax.legend(frameon=False, fontsize=7)

    fig.tight_layout()
    out = HERE / "fig_accuracy.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
