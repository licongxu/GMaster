"""Figures from the published three-way board plus the existing NaMaster/ducc0 table.

Does not time anything. Sources:
  docs/three_way_gpu.md            D&C, v2 march, SHTns (24 Sep 2026, GPU0)
  docs/v2_march_benchmark_report.md  NaMaster's ducc0 SHT, 96 cores, n_iter=0
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NS = [64, 128, 256, 512, 1024, 2048, 4096]

# synthesis / map2alm n_iter=0, milliseconds
S0_SYN = {
    "D&C":    [0.26, 0.34, 0.52, 1.77, 5.85, 23.67, 122.61],
    "march":  [0.28, 0.44, 0.52, 1.87, 6.90, 36.90, 510.97],
    "SHTns":  [0.04, 0.08, 0.32, 1.93, 13.31, 99.92, 767.16],
    "NaMaster": [0.38, 0.82, 1.93, 11.97, 50.27, 269.26, 1833.50],
}
S0_ANA = {
    "D&C":    [0.27, 0.31, 0.62, 1.97, 6.67, 29.24, 145.07],
    "march":  [0.25, 0.29, 0.48, 1.93, 7.34, 42.32, 453.28],
    "SHTns":  [0.03, 0.06, 0.27, 1.72, 12.71, 98.14, 767.03],
    "NaMaster": [0.52, 1.15, 2.87, 13.55, 55.10, 285.48, 1832.55],
}
S2_SYN = {
    "D&C":    [0.45, 0.65, 1.23, 2.72, 12.23, 62.40, 253.46],
    "march":  [0.33, 0.43, 0.84, 2.74, 13.14, 90.35, 695.29],
    "SHTns":  [0.13, 0.22, 0.94, 5.55, 35.22, 259.27, 1989.39],
    "NaMaster": [0.65, 1.45, 3.37, 21.93, 95.91, 538.71, 3673.44],
}
S2_ANA = {
    "D&C":    [0.74, 0.86, 1.64, 3.63, 14.46, 71.61, 294.35],
    "march":  [0.30, 0.35, 0.67, 2.35, 13.46, 91.75, 706.69],
    "SHTns":  [0.08, 0.16, 0.79, 4.95, 35.08, 265.20, 2066.43],
    "NaMaster": [0.84, 1.95, 4.88, 25.13, 104.73, 571.50, 3794.93],
}
# GPU: device-memory growth, GiB. SHTns spin 0 is the fp64 process (fp32 not separable).
# NaMaster: host peak RSS of the full MASTER pipeline, benchmarks/independent/results_nm_rss.json.
S0_MEM = {
    "D&C": [0.06, 0.12, 0.35, 0.72, 2.73, 11.09, 40.96],
    "march": [0.09, 0.12, 0.32, 0.75, 2.66, 10.21, 40.55],
    "SHTns": [0.01, 0.03, 0.08, 0.30, 1.29, 5.14, 20.55],
    "NaMaster": [0.105, 0.123, 0.187, 0.437, 1.469, 5.722, 22.621],
}
S2_MEM = {
    "D&C": [0.02, 0.08, 0.30, 1.31, 4.60, 17.71, 71.90],
    "march": [0.09, 0.18, 0.38, 1.42, 5.39, 24.35, 64.75],
    "SHTns": [0.01, 0.04, 0.12, 0.41, 1.72, 6.84, 27.30],
    "NaMaster": [0.127, 0.183, 0.418, 1.315, 4.748, 18.900, 39.882],
}

BLUE, ORANGE, AQUA, INK, INK2, GRID, SURF = (
    "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb")
PURPLE = "#6b4c9a"
STYLE = {
    "D&C": ("-o", BLUE, "GMaster D&C"),
    "march": ("--s", ORANGE, "GMaster v2 march"),
    "SHTns": (":^", AQUA, "SHTns GPU"),
    "NaMaster": ("-.D", PURPLE, "NaMaster, 96 cores"),
}


def panel(ax, series, title, ylabel):
    ax.set_facecolor(SURF)
    for key, (fmt, col, lab) in STYLE.items():
        if key not in series:
            continue
        ax.plot(NS, series[key], fmt, color=col, lw=2, ms=6, mec=SURF, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(NS)
    ax.set_xticklabels([str(n) for n in NS])
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel(r"$N_{\mathrm{side}}$  ($\ell_{\max}=3N_{\mathrm{side}}-1$)", color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=INK2)


fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), facecolor=SURF)
panel(axes[0, 0], S0_ANA, "spin 0   map2alm   n_iter = 0", "time [ms]")
panel(axes[0, 1], S0_SYN, "spin 0   alm2map", "time [ms]")
panel(axes[1, 0], S2_ANA, "spin 2   map2alm   n_iter = 0", "time [ms]")
panel(axes[1, 1], S2_SYN, "spin 2   alm2map", "time [ms]")
axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
fig.suptitle(
    "HEALPix transforms vs SHTns on its Gauss grid.  RTX PRO 6000 and 96-core NaMaster.\n"
    "Spin-2 SHTns is its fp64 spin-1 vector transform, not a spin-2 HEALPix SHT.",
    color=INK, fontsize=11, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig("docs/three_way_time.png", dpi=160, facecolor=SURF)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURF, sharey=True)
panel(axes[0], S0_MEM, "spin 0", "memory [GiB]")
panel(axes[1], S2_MEM, "spin 2", "memory [GiB]")
axes[0].legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
fig.suptitle(
    "GPU curves: device-memory growth of the transform (cudaMemGetInfo).\n"
    "NaMaster: host peak RSS of the full MASTER pipeline.  SHTns spin 2 is its fp64 spin-1 vector transform.",
    color=INK, fontsize=11, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig("docs/three_way_memory.png", dpi=160, facecolor=SURF)
print("wrote docs/three_way_time.png docs/three_way_memory.png")
