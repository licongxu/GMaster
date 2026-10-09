"""SHT and ACT end-to-end figures.

Timings come from benchmarks/scaling/results.json (the 27 Sep board). The
GMaster/NaMaster agreement panels come from the march-only float32 rerun,
.qwen/tmp/accuracy/march_rerun/paper_metrics.json (new cells at Nside
1024-4096, the 27 Sep board rows at 8192; each cell carries its "source"),
and from the board's own agreement rows at Nside 64-512. The accuracy panels of
fig_act_perf show that error in units of the bandpower cosmic variance
(cv_ratios.json from cv_ratio.py; run that first).

MNRAS letter width, usetex, 300 dpi. Markers are medians. The SHT timing
figure draws those medians only.
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.container import ErrorbarContainer
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter

from make_figures import BLACK, GREY, MN_W, TEX_RC

# GMaster in red; every reference code in black or grey.
RED = "#D62728"

HERE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get(
    "GMASTER_RESULTS_JSON", HERE.parents[2] / "benchmarks" / "scaling" / "results.json"))
SHTNS_HP = HERE.parents[2] / "benchmarks" / "shtns_healpix"
METRICS = Path(os.environ.get(
    "GMASTER_PAPER_METRICS",
    HERE.parents[2] / ".qwen" / "tmp" / "accuracy" / "march_rerun" / "paper_metrics.json"))
NSIDES = np.array([64, 128, 256, 512, 1024, 2048, 4096, 8192], dtype=float)
# MNRAS letter: \columnwidth and \textwidth. Figures drawn at the size they
# are included, so these font sizes are the page sizes.
COL_W = 244.0 / 72.27
TEXT_W = 508.0 / 72.27
TICKS = np.array([64, 256, 1024, 8192], dtype=float)


def _load():
    data = json.loads(RESULTS.read_text())
    # SHTns: the HEALPix port (benchmarks/shtns_healpix), spin 0 and spin 2, fp64 and fp32.
    # It replaces the board's Gauss-Legendre SHTns rows.  Cells that ran out of memory
    # (seconds null) are dropped.
    new = [json.loads(x) for x in (SHTNS_HP / "sht_times.jsonl").read_text().splitlines() if x]
    keep = {}
    for r in new:
        keep[(r["nside"], r["spin"], r["op"], r["precision"])] = r
    data["transforms"] = ([r for r in data["transforms"] if r["engine"] != "shtns"]
                          + [r for r in keep.values() if r["seconds"]])
    data["e2e_shtns"] = _e2e_shtns()
    # Accuracy panels: march-only float32 metrics where they exist (Nside
    # 1024-4096, plus the board rows they carry at 8192); every other
    # benchmarked Nside (64-512) takes the board's own agreement row.
    cells = json.loads(METRICS.read_text())["cells"]
    have = {(c["nside"], c["spin"]) for c in cells}
    board = [dict(r, source="results.json") for r in data["agreement"]
             if (r["nside"], r["spin"]) not in have]
    data["agreement"] = cells + board
    data["cv"] = json.loads((HERE / "cv_ratios.json").read_text())
    return data


def _e2e_shtns():
    """SHTns end to end = SHTns analysis (n_iter 3) + GMaster mask/MCM/decouple stage."""
    path = SHTNS_HP / "e2e_shtns.jsonl"
    if not path.exists():
        return []
    mcm, ana = {}, {}
    for r in (json.loads(x) for x in path.read_text().splitlines() if x):
        if not r["seconds"]:
            continue
        if r["part"] == "gmaster_mcm":
            mcm[(r["nside"], r["spin"])] = np.median(r["seconds"])
        else:
            ana[(r["nside"], r["spin"], r["precision"])] = np.median(r["seconds"])
    rows = []
    for (n, s, p), t in ana.items():
        if (n, s) in mcm:
            rows.append({"engine": "shtns", "nside": n, "spin": s, "precision": p,
                         "cores": None, "seconds": [t + mcm[(n, s)]]})
    return rows


def _times(rows, engine, spin, op=None, cores=None, precision=None):
    sel = []
    for r in rows:
        if r["engine"] != engine or r["spin"] != spin:
            continue
        if op is not None and r.get("op") != op:
            continue
        if r.get("cores") != cores:
            continue
        if engine == "shtns" and r.get("precision", "fp64") != (precision or "fp64"):
            continue
        sel.append(r)
    sel.sort(key=lambda r: r["nside"])
    nside = np.array([r["nside"] for r in sel], dtype=float)
    sec = [np.asarray(r["seconds"], dtype=float) for r in sel]
    med = np.array([np.median(v) for v in sec])
    lo = np.array([v.min() for v in sec])
    hi = np.array([v.max() for v in sec])
    nrun = np.array([len(v) for v in sec])
    return nside, med, lo, hi, nrun


def _draw(ax, nside, med, lo, hi, nrun, color, marker, label, ls="-", mfc=None, zorder=None, bars=True):
    gm = color == RED      # GMaster is drawn heavier and on top
    kw = dict(
        color=color, marker=marker, linestyle=ls,
        lw=1.0, ms=3.2,
        markerfacecolor=color if mfc is None else mfc, markeredgecolor=color, markeredgewidth=0.6,
        label=label, zorder=5 if gm else (3 if zorder is None else zorder),
    )
    if not bars:
        ax.plot(nside, med, **kw)
        return
    yerr = np.vstack((med - lo, hi - med))
    yerr[:, nrun == 1] = 0.0
    ax.errorbar(nside, med, yerr=yerr, capsize=1.2, elinewidth=0.5, **kw)


def _line_only(h):
    """Legend handle without the error-bar glyph."""
    return h[0] if isinstance(h, ErrorbarContainer) else h


def _lmax_top(ax):
    """Upper axis: ell_max = 3 Nside - 1, same log positions as Nside."""
    top = ax.twiny()
    top.set_xscale("log", base=2)
    top.set_xlim(ax.get_xlim())
    top.set_xticks(TICKS)
    top.set_xticklabels([rf"${int(3 * v - 1)}$" for v in TICKS])
    top.tick_params(which="major", direction="in", length=3.0, labelsize=8, top=True)
    top.tick_params(which="minor", length=0, top=False)
    top.set_xlabel(r"$\ell_{\max}$", fontsize=9)
    return top


def _style(ax, top=True):
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(TICKS)
    ax.set_xticklabels([rf"${int(v)}$" for v in TICKS])
    ax.tick_params(which="major", direction="in", top=top, right=True, length=3.0, labelsize=8)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.title.set_size(9)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=1.4)
    ax.set_xlim(64, 8192)
    # Minor grid: every benchmarked Nside on x, 2-9 x 10^k within each decade on y.
    ax.xaxis.set_minor_locator(FixedLocator(NSIDES))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color="#D9D9D9", lw=0.5)
    ax.grid(True, which="minor", color="#EEEEEE", lw=0.35)
    ax.set_axisbelow(True)


def _guide(ax, nside, med, label):
    iref = int(np.where(nside == 2048)[0][0])
    grid = np.geomspace(nside[0], nside[-1], 200)
    ax.plot(
        grid, med[iref] * (grid / 2048.0) ** 3,
        linestyle=(0, (1.2, 1.8)), color=BLACK, lw=1.0, zorder=1,
        label=label,
    )


def _series(ax, rows, spin, op, unit, bars=True, shtns=True):
    specs = (
        ("namaster", 32, "#9A9A9A", "o", (0, (4, 1.5)), r"NaMaster, 32 cores", "white", None),
        ("namaster", 96, "#222222", "D", "-", r"NaMaster, 96 cores", None, None),
        ("shtns", None, "#6E6E6E", "^", (0, (1, 1)), r"SHTns fp64", "white", "fp64"),
        ("shtns", None, BLACK, "o", (0, (3.0, 1.2)), r"SHTns fp32", None, "fp32all"),
        ("gmaster", None, RED, "s", "-", r"GMaster fp32 march", None, None),
    )
    drawn = []
    for engine, cores, color, marker, ls, label, mfc, precision in specs:
        if engine == "shtns" and (op is None or not shtns):
            continue
        nside, med, lo, hi, nrun = _times(rows, engine, spin, op, cores, precision)
        if len(nside) == 0:
            continue
        med = med * unit
        lo = lo * unit
        hi = hi * unit
        _draw(ax, nside, med, lo, hi, nrun, color, marker, label, ls=ls, mfc=mfc,
              zorder=2 if mfc == "white" else None, bars=bars)
        drawn.append((nside, med, lo, hi, engine))
    gm = [d for d in drawn if d[4] == "gmaster"][0]
    lo_all = np.concatenate([d[2] for d in drawn])
    hi_all = np.concatenate([d[3] for d in drawn])
    return gm[0], gm[1], lo_all, hi_all


def _act_series(ax, rows, spin):
    """ACT timing panel. Same marks as before; GMaster uses the NaMaster stroke."""
    specs = (
        ("namaster", 32, "#BBBBBB", "o", "--", r"NaMaster, 32 cores", None),
        ("namaster", 96, "#444444", "D", "-", r"NaMaster, 96 cores", None),
        ("shtns", None, "#6E6E6E", "^", (0, (1, 1)), r"SHTns fp64", "fp64"),
        ("shtns", None, BLACK, "o", (0, (3.0, 1.2)), r"SHTns fp32", "fp32all"),
        ("gmaster", None, RED, "s", "-", r"GMaster fp32", None),
    )
    drawn = []
    for engine, cores, color, marker, ls, label, prec in specs:
        nside, med, lo, hi, nrun = _times(rows, engine, spin, None, cores, prec)
        if len(nside) == 0:
            continue
        yerr = np.vstack((med - lo, hi - med))
        yerr[:, nrun == 1] = 0.0
        ax.errorbar(
            nside, med, yerr=yerr, color=color, marker=marker, linestyle=ls,
            lw=1.05, ms=3.4, capsize=1.6, elinewidth=0.6,
            label=label, zorder=4 if color == RED else 2,
        )
        drawn.append((nside, med, lo, hi, engine))
    gm = [d for d in drawn if d[4] == "gmaster"][0]
    lo_all = np.concatenate([d[2] for d in drawn])
    hi_all = np.concatenate([d[3] for d in drawn])
    return gm[0], gm[1], lo_all, hi_all


# max|x - x_ref| / max|x_ref|, spin 0, lmax = 3 Nside - 1.
# GMaster: float32 march vs NaMaster on HEALPix, random inputs; map2alm is n_iter = 0.
# SHTns: one transform on its Gauss-Legendre grid, against the exact samples of the
# NaMaster coefficients of the masked ACT temperature map.
_ERR_NSIDE = np.array([64, 128, 256, 512, 1024, 2048, 4096, 8192], dtype=float)
_ERR = {
    "map2alm": {
        "fp64": [2.20e-14, 2.50e-14, 3.42e-14, 1.04e-13, 1.17e-13, 6.39e-13, 9.26e-13, 5.90e-12],
        "fp32": [1.24e-6, 2.24e-6, 2.85e-6, 2.59e-6, 5.29e-6, 8.91e-6, 8.39e-6, 9.16e-6],
        "fp32all": [2.60e-6, 2.61e-6, 6.34e-6, 1.32e-5, 2.21e-5, 2.97e-5, 7.31e-5, 3.30e-4],
        "gm": [2.1e-6, 3.7e-6, 7.4e-6, 1.3e-5, 2.8e-5, 5.4e-5, 1.3e-4, 2.3e-4],
    },
    "alm2map": {
        "fp64": [7.30e-15, 1.61e-14, 5.34e-14, 1.56e-13, 9.50e-14, 6.38e-13, 7.77e-13, 2.96e-12],
        "fp32": [2.70e-7, 2.89e-7, 3.09e-7, 3.38e-7, 2.68e-7, 1.94e-7, 2.33e-7, 3.40e-7],
        "fp32all": [1.68e-6, 3.44e-6, 1.30e-5, 2.12e-5, 5.19e-5, 5.97e-5, 7.60e-5, 1.20e-4],
        "gm": [1.5e-6, 2.5e-6, 4.7e-6, 1.2e-5, 2.4e-5, 4.6e-5, 8.5e-5, 1.7e-4],
    },
}


def fig_sht_error():
    specs = (
        ("fp64", "#6E6E6E", "^", (0, (1, 1.2)), r"SHTns fp64", "white"),
        ("fp32", "#222222", "v", (0, (1, 1.2)), r"SHTns fp32 data", "white"),
        ("fp32all", BLACK, "o", (0, (3, 1.4)), r"SHTns fp32 all", None),
        ("gm", RED, "s", "-", r"GMaster fp32 march", None),
    )
    ylim = {"map2alm": (8e-15, 1e-3), "alm2map": (2e-15, 1e-3)}
    ylabel = {
        "map2alm": r"$\max|\Delta a_{\ell m}|/\max|a_{\ell m}^{\mathrm{ref}}|$",
        "alm2map": r"$\max|\Delta\mathrm{map}|/\max|\mathrm{map}_{\mathrm{ref}}|$",
    }
    with plt.rc_context(TEX_RC):
        fig, axes = plt.subplots(2, 1, figsize=(COL_W, 3.85), sharex=True, layout="constrained")
        for i, op in enumerate(("map2alm", "alm2map")):
            ax = axes[i]
            for key, color, marker, ls, label, mfc in specs:
                ax.plot(
                    _ERR_NSIDE, _ERR[op][key], color=color, marker=marker, ls=ls,
                    lw=1.0, ms=3.2, markerfacecolor=color if mfc is None else mfc,
                    markeredgecolor=color, markeredgewidth=0.6, label=label,
                    zorder=4 if color == RED else 3,
                )
            _style(ax, top=(i == 1))
            if i == 0:
                _lmax_top(ax)
            ax.set_ylim(*ylim[op])
            ax.set_ylabel(ylabel[op])
            ax.text(0.03, 0.95, r"\texttt{" + op + "}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=8)
            if i == 1:
                ax.set_xlabel(r"$N_{\mathrm{side}}$")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="outside upper center", ncol=2, frameon=False,
            fontsize=8, handlelength=2.4, columnspacing=0.8, borderaxespad=0.02,
        )
        fig.savefig(HERE / "fig_sht_error.pdf", dpi=300)
        fig.savefig(HERE / "fig_sht_error.png", dpi=300)
        plt.close(fig)


def fig_sht(data):
    with plt.rc_context(TEX_RC):
        fig, axes = plt.subplots(2, 2, figsize=(TEXT_W, 4.05), sharex=True, layout="constrained")
        y_all = []
        for i, op in enumerate(("map2alm", "alm2map")):
            for j, spin in enumerate((0, 2)):
                ax = axes[i, j]
                nside, med, lo, hi = _series(
                    ax, data["transforms"], spin, op, 1e3, bars=False, shtns=True)
                _guide(
                    ax, nside, med,
                    r"$\propto N_{\mathrm{side}}^{3}$" if (i, j) == (0, 0) else None)
                _style(ax, top=(i == 1))
                if i == 0:
                    _lmax_top(ax)
                y_all.append(lo)
                y_all.append(hi)
                if i == 0:
                    ax.set_title(rf"spin ${spin}$")
                if i == 1:
                    ax.set_xlabel(r"$N_{\mathrm{side}}$")
                if j == 0:
                    ax.set_ylabel(op + r" (ms)")
        ys = np.concatenate(y_all)
        for ax in axes.ravel():
            ax.set_ylim(ys.min() / 2.2, ys.max() * 2.8)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        handles = [_line_only(h) for h in handles]
        order = [r"$\propto N_{\mathrm{side}}^{3}$", "NaMaster, 32 cores", "NaMaster, 96 cores",
                 "SHTns fp64", "SHTns fp32", "GMaster fp32 march"]
        pick = [labels.index(k) for k in order if k in labels]
        order = [k for k in order if k in labels]
        fig.legend(
            [handles[i] for i in pick], order, loc="outside upper center", ncol=4, frameon=False,
            fontsize=8, handlelength=2.4, columnspacing=0.8, borderaxespad=0.02,
        )
        fig.savefig(HERE / "fig_sht_times.pdf", dpi=300)
        fig.savefig(HERE / "fig_sht_times.png", dpi=300)
        plt.close(fig)


def fig_perf(data):
    with plt.rc_context(TEX_RC):
        fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.45), sharey=True, layout="constrained")
        fig.set_constrained_layout_pads(w_pad=0.04, wspace=0.08)
        y_all = []
        for j, spin in enumerate((0, 2)):
            ax = axes[j]
            nside, med, lo, hi = _act_series(ax, data["end_to_end"] + data["e2e_shtns"], spin)
            _guide(ax, nside, med, r"$\propto N_{\mathrm{side}}^{3}$" if j == 0 else None)
            _style(ax, top=False)
            _lmax_top(ax)
            ax.set_title(rf"spin ${spin}$")
            ax.set_xlabel(r"$N_{\mathrm{side}}$")
            y_all.append(lo)
            y_all.append(hi)
        for ax in axes:
            ax.set_ylim(min(np.concatenate(y_all)) / 2.5, max(np.concatenate(y_all)) * 3.0)
        axes[0].set_ylabel(r"End-to-end (s)")
        handles, labels = axes[0].get_legend_handles_labels()
        handles = [_line_only(h) for h in handles]
        fig.legend(
            handles, labels, loc="outside upper center", ncol=3, frameon=False,
            fontsize=7, handlelength=1.6, columnspacing=0.6, borderaxespad=0.02,
        )
        fig.savefig(HERE / "fig_act_perf.pdf", dpi=300)
        fig.savefig(HERE / "fig_act_perf.png", dpi=300)
        plt.close(fig)


def main():
    data = _load()
    fig_sht(data)
    fig_perf(data)
    print("wrote fig_sht_times and fig_act_perf")


if __name__ == "__main__":
    main()
