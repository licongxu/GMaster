"""End-to-end MASTER wall time for the letter. Writes fig_scoreboard.pdf."""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RESULTS = ROOT / "benchmarks" / "scaling" / "results.json"

HARDWARE = (
    "GMaster: one RTX PRO 6000 Blackwell (96 GB); "
    "NaMaster 3.0 / ducc0 0.39.1: Threadripper PRO 9995WX, 96 cores"
)


def _median_seconds(row):
    secs = row["seconds"]
    return statistics.median(secs) if len(secs) > 1 else secs[0]


def load_e2e():
    data = json.loads(RESULTS.read_text())
    out = {0: {"gmaster": {}, "namaster": {}}, 2: {"gmaster": {}, "namaster": {}}}
    for row in data["end_to_end"]:
        spin = row["spin"]
        eng = row["engine"]
        if eng not in out[spin]:
            continue
        if eng == "namaster" and row.get("cores") != 96:
            continue
        out[spin][eng][row["nside"]] = _median_seconds(row) * 1000.0
    return out


def style(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(48, 12000)
    ax.set_xticks([64, 128, 256, 512, 1024, 2048, 4096, 8192])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.set_xlabel(r"$N_{\mathrm{side}}$")
    ax.set_ylabel("end-to-end wall time (ms)")
    ax.grid(True, which="major", linewidth=0.45, alpha=0.65)
    ax.grid(True, which="minor", linewidth=0.25, alpha=0.45)


def main():
    e2e = load_e2e()
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.35), sharey=True)
    for ax, spin in zip(axes, (0, 2)):
        ns = sorted(set(e2e[spin]["gmaster"]) | set(e2e[spin]["namaster"]))
        x = np.array(ns, dtype=float)
        gm = np.array([e2e[spin]["gmaster"].get(n, np.nan) for n in ns])
        nm = np.array([e2e[spin]["namaster"].get(n, np.nan) for n in ns])
        mask = np.isfinite(nm)
        ax.plot(x[mask], nm[mask], color="#333333", marker="s", ms=4.5, lw=1.25, label="NaMaster")
        ax.plot(x, gm, color="#CC0000", marker="o", ms=4.5, lw=1.25, label="GMaster")
        style(ax)
        ax.set_title(f"spin {spin}")
    axes[0].legend(frameon=False, loc="upper left", fontsize=7.5)
    fig.text(0.5, 0.01, HARDWARE, ha="center", va="bottom", fontsize=6.5, color="#333333")
    fig.subplots_adjust(bottom=0.17, wspace=0.08)
    out = HERE / "fig_scoreboard.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
