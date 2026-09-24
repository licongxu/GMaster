"""v2-march pipeline wall time across the published hardware boards. No new timings.

RTX PRO 6000 + 96-core NaMaster: docs/v2_march_benchmark_report.md (fp32, spin 0).
Kaggle Tesla T4 + 4 session CPUs: examples/kaggle_t4_nside_ladder.md on github/main.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NS = np.array([64, 128, 256, 512, 1024, 2048, 4096])
# seconds, spin 0, end-to-end MASTER, n_iter=3
GM_RTX = np.array([5.6, 1.7, 5.3, 22.5, 126.8, 1614.0, 7458.0]) / 1e3
NM_96 = np.array([8.1, 20.3, 58.7, 310.2, 1658.2, 10239.9, 71583.0]) / 1e3
GM_T4 = np.array([0.04, 0.04, 0.07, 0.36, 2.02, np.nan, np.nan])
NM_4 = np.array([0.08, 0.25, 1.77, 12.93, 98.70, np.nan, np.nan])
# PR #7, Cursor cloud VM, 4 Xeon threads, full-sky ones mask, nlb=50. 4096 OOM.
NM_CLOUD = np.array([0.018, 0.074, 0.440, 3.174, 24.205, 165.138, np.nan])

BLUE, ORANGE, PURPLE, AQUA = "#2a78d6", "#eb6834", "#6b4c9a", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

fig, ax = plt.subplots(figsize=(7.6, 5.0), facecolor=SURF)
ax.set_facecolor(SURF)
series = (
    (GM_RTX, "-o", BLUE, "GMaster v2 march, RTX PRO 6000"),
    (GM_T4, "--s", AQUA, "GMaster v2 march, Kaggle T4"),
    (NM_96, "-.D", PURPLE, "NaMaster, 96-core EPYC"),
    (NM_4, ":^", ORANGE, "NaMaster, Kaggle 4 CPU"),
    (NM_CLOUD, "-.v", "#9a4c6b", "NaMaster, Cursor VM Intel Xeon 4 thr"),
)
for y, fmt, col, lab in series:
    ax.plot(NS, y, fmt, color=col, lw=2, ms=6, mec=SURF, label=lab)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(NS)
ax.set_xticklabels([str(n) for n in NS])
ax.set_xlabel(r"$N_{\mathrm{side}}$", color=INK2)
ax.set_ylabel("pipeline wall time [s]", color=INK2)
ax.set_title("Spin 0 MASTER, n_iter = 3.  T4 OOM at 2048; Xeon OOM at 4096.", loc="left", color=INK, fontsize=11)
ax.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
for sp in ax.spines.values():
    sp.set_color(GRID)
ax.tick_params(colors=INK2)
ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
fig.tight_layout()
fig.savefig("docs/hardware_v2_march.png", dpi=160, facecolor=SURF)
print("wrote docs/hardware_v2_march.png")
