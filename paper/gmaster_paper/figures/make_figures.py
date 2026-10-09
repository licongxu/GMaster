"""Publication figures for the GMaster MNRAS paper. Measured numbers only."""

import json
from pathlib import Path

import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent

BLACK = "#000000"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
BLUE = "#0072B2"
VERM = "#D55E00"
GREY = "#666666"
LIGHT = "#F4F4F4"

# MNRAS two-column \textwidth is 508 pt; lettering ~8 pt at that final size
# (https://academic.oup.com/mnras/pages/general_instructions).
MN_W = 508.0 / 72.27

TEX_RC = {
    "text.usetex": True,
    "text.latex.preamble": (
        r"\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{mathptmx}"
    ),
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.1,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": None,
}

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    }
)


def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf", dpi=300)
    fig.savefig(HERE / f"{name}.png", dpi=300)
    plt.close(fig)


PLOT_JSON = HERE.parents[2] / "benchmarks/independent/results_for_plots.json"
GM_PIPE_JSON = HERE.parents[2] / "benchmarks/independent/results_gm_noprefill.json"
GM_4096S2_JSON = HERE.parents[2] / "benchmarks/independent/results_gm_4096s2_historical.json"
NM_RSS_JSON = HERE.parents[2] / "benchmarks/independent/results_nm_rss.json"


def _load_json(path):
    return json.loads(path.read_text())


def _top_legend(fig, ax, ncol):
    """Shared legend just above the panels, 8 pt, no reserved empty band."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=ncol,
        frameon=False,
        fontsize=8,
        handlelength=1.8,
        handletextpad=0.4,
        columnspacing=1.1,
        borderaxespad=0.15,
    )


def _ticks(ax, nsides):
    ax.set_xticks(nsides)
    ax.set_xticklabels([rf"${int(v)}$" for v in nsides])
    ax.tick_params(which="major", direction="in", top=True, right=True, length=3.2)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=1.6)


def _rows_by_spin(rows, spin):
    out = [r for r in rows if r["spin"] == spin]
    out.sort(key=lambda r: r["nside"])
    return out


def _namaster_pipeline_gib(nside, spin, nthreads):
    """Computed NaMaster working set for the paper pipeline (CPU, float64).

    Counts the arrays the benchmark worker holds live: four input maps,
    NmtField (mask, maps, alms, mask alms), unbinned MCM, binned MCM plus
    inverse, and a per-thread OpenMP/ducc scratch term. NaMaster RSS was
    not recorded; 32- vs 96-core difference is the thread term.
    """
    npix = 12 * nside**2
    lmax = 3 * nside - 1
    nalm = (lmax + 1) * (lmax + 2) // 2
    nmaps = 1 if spin == 0 else 2
    ncls = 1 if spin == 0 else 4
    nell = lmax + 1
    nbands = max(nell // 30, 1)
    py_maps = 4 * npix * 8
    field = (1 + nmaps) * npix * 8 + (nmaps + 1) * nalm * 16
    mcm = (ncls * nell) ** 2 * 8
    mcm_binned = 2 * (ncls * nbands) ** 2 * 8
    # ponytail: 2 MiB OMP stack + 8 MiB ducc scratch per thread; measure ru_maxrss if the split matters
    omp = nthreads * (2 + 8) * 1024**2
    baseline = 0.4 * 1024**3
    return (py_maps + field + mcm + mcm_binned + omp + baseline) / 1024**3


def fig_map2alm():
    _fig_sht_time("ana", "fig_map2alm")


def fig_alm2map():
    _fig_sht_time("syn", "fig_alm2map")


def fig_sht_times():
    """One letter figure: map2alm (top) and alm2map (bottom), spin 0/2."""
    rows = _load_json(PLOT_JSON)["sht"]
    with mpl.rc_context(TEX_RC):
        fig, axes = plt.subplots(
            2, 2, figsize=(MN_W, 4.55), sharex=True, sharey=True, layout="constrained"
        )
        y_all = []
        for i, (kind, ylab) in enumerate((
            ("ana", r"map2alm (ms)"),
            ("syn", r"alm2map (ms)"),
        )):
            for j, spin in enumerate((0, 2)):
                ax = axes[i, j]
                r = _rows_by_spin(rows, spin)
                nside = np.array([x["nside"] for x in r], dtype=float)
                gm = np.array([x[f"gm_fp32_{kind}_ms"] for x in r], dtype=float)
                d32 = np.array([x[f"ducc32_{kind}_ms"] for x in r], dtype=float)
                d96 = np.array([x[f"ducc96_{kind}_ms"] for x in r], dtype=float)
                y_all.append(np.concatenate((d32, d96, gm)))
                ax.plot(nside, d32, "o--", color="#BBBBBB", lw=1.1, ms=3.5,
                        label=r"NaMaster, 32 cores")
                ax.plot(nside, d96, "o-", color=GREY, lw=1.15, ms=3.5,
                        label=r"NaMaster, 96 cores")
                nref = 2048.0
                iref = int(np.where(nside == nref)[0][0])
                n_guide = np.geomspace(nside[0], nside[-1], 200)
                t_guide = gm[iref] * (n_guide / nside[iref]) ** 3
                y_all.append(t_guide)
                ax.plot(n_guide, t_guide, linestyle=(0, (1.2, 1.8)), color=BLACK,
                        lw=1.1, zorder=4, dash_capstyle="butt",
                        label=r"$\propto N_{\mathrm{side}}^{3}$"
                        if (i, j) == (0, 0) else None)
                ax.plot(nside, gm, "s-", color=BLUE, lw=1.15, ms=3.5, zorder=3,
                        label=r"GMaster fp32")
                ax.set_xscale("log", base=2)
                ax.set_yscale("log")
                _ticks(ax, nside)
                ax.set_xlim(nside[0] / 1.15, nside[-1] * 1.15)
                if i == 0:
                    ax.set_title(rf"spin ${spin}$")
                if i == 1:
                    ax.set_xlabel(r"$N_{\mathrm{side}}$")
                if j == 0:
                    ax.set_ylabel(ylab)
        ys = np.concatenate(y_all)
        axes[0, 0].set_ylim(max(ys.min(), 1e-2) / 1.4, ys.max() * 2.2)
        _top_legend(fig, axes[0, 0], ncol=4)
        save(fig, "fig_sht_times")


def _fig_sht_time(kind, name):
    rows = _load_json(PLOT_JSON)["sht"]
    with mpl.rc_context(TEX_RC):
        fig, axes = plt.subplots(1, 2, figsize=(MN_W, 2.55), sharey=True, layout="constrained")
        y_all = []
        for ax, spin in zip(axes, (0, 2)):
            r = _rows_by_spin(rows, spin)
            nside = np.array([x["nside"] for x in r], dtype=float)
            gm = np.array([x[f"gm_fp32_{kind}_ms"] for x in r], dtype=float)
            d32 = np.array([x[f"ducc32_{kind}_ms"] for x in r], dtype=float)
            d96 = np.array([x[f"ducc96_{kind}_ms"] for x in r], dtype=float)
            y_all.append(np.concatenate((d32, d96, gm)))
            ax.plot(nside, d32, "o--",
                    color="#BBBBBB", lw=1.1, ms=3.5, label=r"NaMaster, 32 cores")
            ax.plot(nside, d96, "o-",
                    color=GREY, lw=1.15, ms=3.5, label=r"NaMaster, 96 cores")
            nref = 2048.0
            iref = int(np.where(nside == nref)[0][0]) if np.any(nside == nref) else -1
            n_guide = np.geomspace(nside[0], nside[-1], 200)
            t_guide = gm[iref] * (n_guide / nside[iref]) ** 3
            ax.plot(n_guide, t_guide, linestyle=(0, (1.2, 1.8)), color=BLACK,
                    lw=1.1, zorder=4, dash_capstyle="butt",
                    label=r"$\propto N_{\mathrm{side}}^{3}$" if ax is axes[0] else None)
            ax.plot(nside, gm, "s-", color=BLUE, lw=1.15, ms=3.5, zorder=3,
                    label=r"GMaster fp32")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            _ticks(ax, nside)
            ax.set_xlabel(r"$N_{\mathrm{side}}$")
            ax.set_title(rf"spin ${spin}$")
            ax.set_xlim(nside[0] / 1.15, nside[-1] * 1.15)
        ys = np.concatenate(y_all)
        axes[0].set_ylim(ys.min() / 1.8, ys.max() * 2.2)
        axes[0].set_ylabel(r"Time (ms)")
        _top_legend(fig, axes[0], ncol=4)
        save(fig, name)


def fig_pipeline_memory():
    gm = _load_json(GM_PIPE_JSON)["pipeline"]
    hist = _load_json(GM_4096S2_JSON)["pipeline"]
    nm = _load_json(NM_RSS_JSON)["pipeline"]
    gm_key = {(r["nside"], r["spin"]): r for r in gm}
    for r in hist:
        gm_key[(r["nside"], r["spin"])] = r
    nm_key = {(r["nside"], r["spin"]): r for r in nm}
    nsides = np.array([64, 128, 256, 512, 1024, 2048, 4096], dtype=float)
    with mpl.rc_context(TEX_RC):
        fig, axes = plt.subplots(1, 2, figsize=(MN_W, 2.55), sharey=True, layout="constrained")
        y_all = []
        for ax, spin in zip(axes, (0, 2)):
            nm_n, nm_m, gm_n, gm_m = [], [], [], []
            for n in nsides:
                nmi = nm_key.get((int(n), spin))
                if nmi is not None and nmi.get("peak_rss_gb") is not None:
                    nm_n.append(n)
                    nm_m.append(nmi["peak_rss_gb"])
                row = gm_key.get((int(n), spin))
                mem = None if row is None else row.get("fp32", {}).get("gpu_peak_gb")
                if mem is not None:
                    gm_n.append(n)
                    gm_m.append(mem)
            gm_n = np.asarray(gm_n, dtype=float)
            gm_m = np.asarray(gm_m, dtype=float)
            y_all.append(np.concatenate((np.asarray(nm_m, dtype=float), gm_m)))
            ax.plot(nm_n, nm_m, "o-", color=GREY, lw=1.15, ms=3.5,
                    label=r"NaMaster peak RSS")
            nref = 2048.0
            iref = int(np.where(gm_n == nref)[0][0]) if np.any(gm_n == nref) else -1
            n_guide = np.geomspace(nsides[0], nsides[-1], 200)
            m_guide = gm_m[iref] * (n_guide / gm_n[iref]) ** 2
            y_all.append(m_guide)
            ax.plot(n_guide, m_guide, linestyle=(0, (1.2, 1.8)), color=BLACK,
                    lw=1.1, zorder=4, dash_capstyle="butt",
                    label=r"$\propto N_{\mathrm{side}}^{2}$" if ax is axes[0] else None)
            ax.plot(gm_n, gm_m, "s-", color=BLUE, lw=1.15, ms=3.5, zorder=3,
                    label=r"GMaster fp32")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            _ticks(ax, nsides)
            ax.set_xlabel(r"$N_{\mathrm{side}}$")
            ax.set_title(rf"spin ${spin}$")
            ax.set_xlim(nsides[0] / 1.15, nsides[-1] * 1.15)
        ys = np.concatenate(y_all)
        axes[0].set_ylim(ys.min() / 1.4, ys.max() * 2.2)
        axes[0].set_ylabel(r"Pipeline memory (GiB)")
        _top_legend(fig, axes[0], ncol=3)
        save(fig, "fig_pipeline_memory")


def fig_e2e_time():
    """End-to-end MASTER wall time vs Nside, same board as the letter table."""
    nsides = np.array([64, 128, 256, 512, 1024, 2048, 4096], dtype=float)
    nm96 = {
        0: np.array([9.0, 20.0, 58.0, 314.0, 1682.0, 10311.0, 71583.0]),
        2: np.array([12.0, 40.0, 161.0, 641.0, 3017.0, 17767.0, 109663.0]),
    }
    nm32 = {
        0: np.array([35.0, 54.0, 130.0, 549.0, 3194.0, 19717.0, 160225.0]),
        2: np.array([40.0, 84.0, 490.0, 1467.0, 5485.0, 31622.0, 228964.0]),
    }
    gm = {
        0: np.array([2.0, 1.0, 4.0, 17.0, 102.0, 1014.0, 7458.0]),
        2: np.array([2.0, 3.0, 8.0, 30.0, 212.0, 1244.0, 8805.0]),
    }
    with mpl.rc_context(TEX_RC):
        fig, axes = plt.subplots(1, 2, figsize=(MN_W, 2.55), sharey=True, layout="constrained")
        y_all = []
        for ax, spin in zip(axes, (0, 2)):
            g = gm[spin]
            n96, t96 = nsides[~np.isnan(nm96[spin])], nm96[spin][~np.isnan(nm96[spin])]
            n32, t32 = nsides[~np.isnan(nm32[spin])], nm32[spin][~np.isnan(nm32[spin])]
            y_all.append(np.concatenate((t32, t96, g)))
            ax.plot(n32, t32, "o--", color="#BBBBBB", lw=1.1, ms=3.5,
                    label=r"NaMaster, 32 cores")
            ax.plot(n96, t96, "o-", color=GREY, lw=1.15, ms=3.5,
                    label=r"NaMaster, 96 cores")
            nref = 2048.0
            iref = int(np.where(nsides == nref)[0][0])
            n_guide = np.geomspace(nsides[0], nsides[-1], 200)
            t_guide = g[iref] * (n_guide / nref) ** 3
            y_all.append(t_guide)
            ax.plot(n_guide, t_guide, linestyle=(0, (1.2, 1.8)), color=BLACK,
                    lw=1.1, zorder=4, dash_capstyle="butt",
                    label=r"$\propto N_{\mathrm{side}}^{3}$" if ax is axes[0] else None)
            ax.plot(nsides, g, "s-", color=BLUE, lw=1.15, ms=3.5, zorder=3,
                    label=r"GMaster fp32")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            _ticks(ax, nsides)
            ax.set_xlabel(r"$N_{\mathrm{side}}$")
            ax.set_title(rf"spin ${spin}$")
            ax.set_xlim(nsides[0] / 1.15, nsides[-1] * 1.15)
        ys = np.concatenate(y_all)
        finite = ys[np.isfinite(ys) & (ys > 0)]
        axes[0].set_ylim(finite.min() / 1.6, finite.max() * 2.4)
        axes[0].set_ylabel(r"End-to-end MASTER (ms)")
        _top_legend(fig, axes[0], ncol=4)
        save(fig, "fig_e2e_time")


def fig_error():
    """Maximum relative deviation of the coupled spectrum from NaMaster."""
    nsides = np.array([64, 128, 256, 512, 1024, 2048, 4096], dtype=float)
    rel = {
        0: np.array([1.15e-7, 4.92e-7, 6.36e-7, 7.07e-7, 1.45e-6, 7.25e-6, 7.0e-6]),
        2: np.array([2.72e-7, 4.37e-7, 6.30e-7, 2.99e-6, 7.79e-6, 1.19e-5, 6.5e-5]),
    }
    with mpl.rc_context(TEX_RC):
        fig, axes = plt.subplots(1, 2, figsize=(MN_W, 2.35), sharey=True, layout="constrained")
        for ax, spin in zip(axes, (0, 2)):
            ax.axhline(1e-6, color=GREY, ls=":", lw=0.9, zorder=1)
            ax.plot(nsides, rel[spin], "s-", color=BLUE, lw=1.15, ms=3.5, zorder=3)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            _ticks(ax, nsides)
            ax.set_xlabel(r"$N_{\mathrm{side}}$")
            ax.set_title(rf"spin ${spin}$")
            ax.set_xlim(nsides[0] / 1.15, nsides[-1] * 1.15)
        axes[0].set_ylim(3e-8, 3e-4)
        axes[0].set_ylabel(r"$\max\lvert C_\ell^{\mathrm{GM}}/C_\ell^{\mathrm{NM}}-1\rvert$")
        save(fig, "fig_error")


def fig_stages():
    # Stage marginals of the session 37 board (.qwen/tmp/chain_s37j.log), fp64 default,
    # ring/coupling precision auto.  "Transform" is field + mask; at spin 0 the mask is
    # paired into the field on the GMaster side, so its GMaster column is the pair.
    cats = ["1024\nspin 0", "1024\nspin 2", "2048\nspin 0", "2048\nspin 2"]
    tr_n = [456 + 421, 833 + 405, 2263 + 2326, 4497 + 2203]
    tr_g = [80 + 0, 118 + 67, 659 + 0, 734 + 358]
    coup_n = [777, 1685, 5557, 10188]
    coup_g = [22, 25, 363, 143]

    x = np.arange(len(cats))
    w = 0.3
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for ax, n, g, title in (
        (axes[0], tr_n, tr_g, "Transform (field + mask)"),
        (axes[1], coup_n, coup_g, "Coupling matrix"),
    ):
        ax.bar(x - w / 2, n, w, color=GREY, label="NaMaster (96 cores)")
        ax.bar(x + w / 2, g, w, color=BLUE, label="GMaster")
        ax.set_xticks(x)
        ax.set_xticklabels(cats, fontsize=8)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel("Stage time (ms)" if ax is axes[0] else "")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "fig_stages")


def fig_ducc():
    # Isolated latitudinal pass vs CPU ducc0, warmed, n_iter=0: board_s37.md table C.
    n = np.array([1024, 2048, 4096])
    ducc_a0 = np.array([58.2, 310.2, 1917.3])
    gm_a0 = np.array([7.2, 42.0, 266.4])
    ducc_s0 = np.array([57.1, 295.7, 1962.6])
    gm_s0 = np.array([7.1, 36.6, 302.6])
    ducc_a2 = np.array([107.6, 603.2, 3811.9])
    gm_a2 = np.array([13.1, 82.3, 676.5])
    ducc_s2 = np.array([110.2, 576.9, 3873.3])
    gm_s2 = np.array([12.9, 81.5, 658.8])

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex="col")
    specs = (
        (axes[0, 0], ducc_a0, gm_a0, r"spin 0, map$\to$alm", ORANGE),
        (axes[0, 1], ducc_s0, gm_s0, r"spin 0, alm$\to$map", ORANGE),
        (axes[1, 0], ducc_a2, gm_a2, r"spin 2, map$\to$alm", VERM),
        (axes[1, 1], ducc_s2, gm_s2, r"spin 2, alm$\to$map", VERM),
    )
    for ax, ducc, gm, title, color in specs:
        ax.plot(n, ducc, "o-", color=GREY, lw=1.6, ms=6, label="ducc0 CPU (96 cores)")
        ax.plot(n, gm, "s-", color=color, lw=1.6, ms=6, label="GMaster v2 march")
        for xv, dv, gv in zip(n, ducc, gm):
            ax.annotate(f"{dv / gv:.1f}x", xy=(xv, gv), xytext=(0, -13),
                        textcoords="offset points", ha="center", fontsize=7, color=color)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(n)
        ax.set_xticklabels([str(int(v)) for v in n])
        ax.set_title(title)
    axes[0, 0].set_ylabel("Warm time (ms)")
    axes[1, 0].set_ylabel("Warm time (ms)")
    axes[1, 0].set_xlabel(r"$N_{\mathrm{side}}$")
    axes[1, 1].set_xlabel(r"$N_{\mathrm{side}}$")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "fig_ducc")


def fig_accuracy():
    # Left: float32 march vs the fp64 recurrence, max over every lane, Nside 1024 spin 0
    # (HANDOFF addendum 34 section 1).  Right: decoupled-bandpower agreement with pymaster,
    # shipped route vs the exact route (board_s37.md table D).
    ms = np.array([0, 1, 2])
    plain = np.array([4.2e-2, 8.8e-5, 3.1e-5])
    one_word = np.array([4.1e-5, 8.6e-5, 5.1e-5])
    two_word = np.array([2.8e-6, 2.9e-6, 1.6e-5])

    cells = ["256\nspin 0", "256\nspin 2", "512\nspin 0", "512\nspin 2"]
    shipped = np.array([7.12e-7, 6.30e-7, 6.20e-7, 3.13e-6])
    exact = np.array([4.79e-13, 2.74e-8, 1.71e-12, 3.56e-7])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    ax = axes[0]
    w = 0.26
    ax.bar(ms - w, plain, w, color=GREY, label="plain float32")
    ax.bar(ms, one_word, w, color=ORANGE, label="difference form, 1-word $C$")
    ax.bar(ms + w, two_word, w, color=BLUE, label="difference form, 2-word $C$")
    ax.set_yscale("log")
    ax.set_xticks(ms)
    ax.set_xticklabels(["$m=0$", "$m=1000$", "$m=L/2$"])
    ax.set_ylabel("Relative error vs float64 march")
    ax.set_ylim(1e-7, 1e-1)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")

    ax = axes[1]
    x = np.arange(len(cells))
    ax.bar(x - 0.18, shipped, 0.36, color=BLUE, label="shipped route")
    ax.bar(x + 0.18, exact, 0.36, color=GREEN, label="exact route (flag)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(cells, fontsize=8)
    ax.set_ylabel(r"Max rel. dev. of $C_\ell$ vs NaMaster")
    ax.set_ylim(1e-14, 1e-4)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_accuracy")


def fig_memory():
    nside = np.array([256, 512, 1024, 2048, 4096])
    L = 3 * nside
    ntheta = 4 * nside - 1
    band_fp64 = L * ntheta * L * 8 / 1024**3
    band_fp32 = band_fp64 / 2
    polar_fp32 = (L * ntheta * L * 4) / 1024**3
    # Six fp32 (high, low) coefficient arrays, 64-order window: 6 * 64 * L * 4 bytes.
    march_win = 6 * 64 * L * 4 / 1024**3
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.plot(nside, band_fp64, "o-", color=BLUE, lw=1.6, label="Scalar Legendre band, fp64")
    ax.plot(nside, band_fp32, "s-", color=ORANGE, lw=1.6, label="Scalar Legendre band, fp32")
    ax.plot(nside, polar_fp32, "D-", color=VERM, lw=1.6, label=r"Polar Wigner-$d$ layout, fp32")
    ax.plot(nside, march_win, "^--", color=GREEN, lw=1.6, label=r"March coefficient window, fp32")
    ax.axhline(71.2, color=GREY, ls="--", lw=1.2, label="Default XLA pool (71.2 GiB)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(nside)
    ax.set_xticklabels([str(int(v)) for v in nside])
    ax.set_xlabel(r"$N_{\mathrm{side}}$")
    ax.set_ylabel("Bytes (GiB)")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_memory")


def fig_prefix():
    L = np.linspace(32, 4096, 200)
    sequential = L
    blocked = 2 * np.sqrt(L)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(L, sequential, color=GREY, lw=1.8, label=r"On-the-fly recurrence, depth $L$")
    ax.plot(L, blocked, color=GREEN, lw=1.8, label=r"Blocked prefix, depth $2\sqrt{L}$")
    ax.set_xlabel(r"Band limit $L$")
    ax.set_ylabel("Sequential recurrence depth")
    ax.set_xlim(0, 4096)
    ax.legend(frameon=False)
    ax.annotate(
        r"$N_{\mathrm{side}}=1024$",
        xy=(3072, 2 * np.sqrt(3072)),
        xytext=(2200, 900),
        arrowprops=dict(arrowstyle="->", color=BLACK, lw=0.8),
        fontsize=8,
    )
    fig.tight_layout()
    save(fig, "fig_prefix")


def fig_offset():
    n = np.arange(8, 1601)
    full = n**3
    blocked = n**3 / 3.0
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(n, full / 1e9, color=GREY, lw=1.8, label=r"Dense $n^3$ offset loop")
    ax.plot(n, blocked / 1e9, color=BLUE, lw=1.8, label=r"Offset blocks, $\sim n^3/3$")
    ax.set_xlabel(r"Matrix size $n=\ell_{\max}+1$")
    ax.set_ylabel(r"Visited cells ($\times 10^9$)")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig_offset")


def _eq_cart(m, xsize=1800, lonra=(-180.0, 180.0), latra=(-70.0, 30.0)):
    proj = hp.projector.CartesianProj(
        coord="C",
        xsize=xsize,
        lonra=list(lonra),
        latra=list(latra),
        flipconv="astro",
    )
    nside = hp.npix2nside(m.size)
    img = np.asarray(
        proj.projmap(m, lambda x, y, z: hp.vec2pix(nside, x, y, z)),
        dtype=np.float64,
    )
    img[~np.isfinite(img) | (np.abs(img) > 1e20)] = np.nan
    return img, proj.get_extent()


def fig_act():
    # ACT DR6 night PA4 f150: footprint + Stokes I + MASTER overlay.
    # Spectra: examples/act_dr6_tt_example (nside 4096). Maps: nside-1024 cache, shown at 256.
    gmaster = HERE.parents[2]
    cache = gmaster / ".qwen/tmp/act"
    ex = gmaster / "examples/act_dr6_tt_example"
    nside_show = 256
    t = np.load(cache / "act_T_nside1024.npy").astype(np.float64)
    w = np.clip(np.load(cache / "act_w_nside1024.npy").astype(np.float64), 0.0, 1.0)
    t[w <= 0] = hp.UNSEEN
    t = hp.ud_grade(t, nside_show)
    w = np.clip(hp.ud_grade(w, nside_show), 0.0, 1.0)
    t[w <= 0] = hp.UNSEEN
    good = np.isfinite(t) & (np.abs(t) < 1e20)
    vmax = float(np.percentile(np.abs(t[good]), 95))
    ell = np.load(ex / "bins.npy")
    cl_gm = np.load(ex / "cl_gmaster.npy")
    cl_nm = np.load(ex / "cl_namaster.npy")
    dl = ell * (ell + 1) / (2 * np.pi)
    ratio = cl_gm / cl_nm - 1.0
    img_w, _ = _eq_cart(w)
    img_t, _ = _eq_cart(t)
    extent = (180.0, -180.0, -70.0, 30.0)
    cmap_w = LinearSegmentedColormap.from_list("actw", ["#FFFFFF", BLUE])
    cmap_w.set_bad("#E6E6E6")
    cmap_t = plt.get_cmap("RdBu_r").copy()
    cmap_t.set_bad("#E6E6E6")
    tex = {
        "text.usetex": True,
        "text.latex.preamble": (
            r"\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{mathptmx}"
        ),
        "font.family": "serif",
        "font.serif": ["Times"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
    xticks = [180, 120, 60, 0, -60, -120, -180]
    yticks = [-60, -40, -20, 0, 20]
    with mpl.rc_context(tex):
        fig, axes = plt.subplots(
            4,
            1,
            figsize=(7.16, 8.15),
            gridspec_kw={"height_ratios": [1.4, 1.4, 1.35, 0.68], "hspace": 0.48},
        )
        ax_w, ax_t, ax_cl, ax_r = axes
        im_w = ax_w.imshow(
            img_w, origin="lower", extent=extent, cmap=cmap_w,
            vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest",
        )
        im_t = ax_t.imshow(
            img_t, origin="lower", extent=extent, cmap=cmap_t,
            vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest",
        )
        ax_w.set_title(r"(a) night PA4 $f150$ footprint")
        ax_t.set_title(r"(b) Stokes $I$")
        for ax in (ax_w, ax_t):
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_ylabel(r"Dec $(^\circ)$")
            ax.tick_params(direction="in")
            ax.grid(True, color="k", alpha=0.12, lw=0.4)
        ax_w.set_xticklabels([])
        ax_t.set_xlabel(r"RA $(^\circ)$")
        fig.colorbar(im_w, ax=ax_w, fraction=0.02, pad=0.015).set_label(r"$w$")
        fig.colorbar(im_t, ax=ax_t, fraction=0.02, pad=0.015).set_label(r"$\mu\mathrm{K}$")
        ax_cl.plot(ell, dl * cl_gm, color=BLUE, lw=1.35, label=r"GMaster $7.63\,\mathrm{s}$")
        ax_cl.plot(
            ell, dl * cl_nm, color=GREY, lw=1.15, ls="--",
            label=r"NaMaster ($192$ cores) $70.4\,\mathrm{s}$",
        )
        ax_cl.set_yscale("log")
        ax_cl.set_ylabel(r"$D_\ell=\ell(\ell+1)C_\ell/2\pi$")
        ax_cl.legend(frameon=False, loc="upper left", fontsize=9)
        ax_cl.text(0.99, 0.08, r"(c)", transform=ax_cl.transAxes, ha="right", va="bottom")
        ax_r.axhline(0.0, color=BLACK, lw=0.6)
        ax_r.plot(ell, 1e5 * ratio, color=BLUE, lw=0.9)
        ax_r.set_xlabel(r"$\ell$")
        ax_r.set_ylabel(r"$(C_\ell^{\mathrm{GM}}/C_\ell^{\mathrm{NM}}-1)\times 10^{5}$")
        ax_r.set_xlim(float(ell.min()), float(ell.max()))
        ax_r.text(0.99, 0.10, r"(d)", transform=ax_r.transAxes, ha="right", va="bottom")
        for ax in (ax_cl, ax_r):
            ax.tick_params(direction="in", which="both")
        fig.subplots_adjust(left=0.11, right=0.90, top=0.955, bottom=0.055)
        save(fig, "fig_act_overlay")


def fig_schematic():
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.15, 3.35)
    ax.axis("off")

    def box(x, y, w, h, text, fc):
        p = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=fc,
            edgecolor=BLACK,
            linewidth=0.8,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=10,
                lw=0.9,
                color=BLACK,
            )
        )

    box(0.15, 1.15, 1.55, 0.9, "HEALPix\nmap", LIGHT)
    box(1.95, 1.15, 2.25, 0.9, "Belt FFT +\ncap chirp-$Z$", SKY)
    box(4.45, 2.15, 2.15, 0.85, "Difference-form\nCUDA march", GREEN)
    box(4.45, 1.10, 2.15, 0.85, "Legendre /\nWigner table", ORANGE)
    box(4.45, 0.05, 2.15, 0.85, "Generic\nscatter loop", GREY)
    box(6.95, 1.15, 1.35, 0.9, r"$a_{\ell m}$", LIGHT)
    box(8.50, 1.15, 1.35, 0.9, "Offset-blocked\nMASTER $M$", SKY)

    arrow(1.70, 1.6, 1.95, 1.6)
    arrow(4.20, 1.6, 4.45, 2.55)
    arrow(4.20, 1.6, 4.45, 1.50)
    arrow(4.20, 1.6, 4.45, 0.45)
    arrow(6.60, 2.55, 6.95, 1.90)
    arrow(6.60, 1.50, 6.95, 1.60)
    arrow(6.60, 0.45, 6.95, 1.30)
    arrow(8.30, 1.6, 8.50, 1.6)

    ax.text(5.52, 3.05, r"default, $L \geq 192$ (spin 0: northern fold)",
            ha="center", fontsize=6.5, color=BLACK)
    ax.text(5.52, 2.00, r"exact fp64 route, where the table fits", ha="center",
            fontsize=6.5, color=BLACK)
    ax.text(5.52, 0.0, "exact fp64 route, otherwise", ha="center", fontsize=6.5, color=BLACK)
    save(fig, "fig_schematic")


if __name__ == "__main__":
    fig_e2e_time()
    fig_sht_times()
    fig_map2alm()
    fig_alm2map()
    fig_pipeline_memory()
    fig_error()
    fig_schematic()
    fig_stages()
    fig_ducc()
    fig_accuracy()
    fig_memory()
    fig_prefix()
    fig_offset()
    fig_act()
    print("wrote", sorted(p.name for p in HERE.glob("fig_*.pdf")))
