"""Latitudinal pass time vs Nside, D&C vs march, spin 0 and spin 2 (from the LAT rows of suite2.log)."""
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG = sys.argv[1] if len(sys.argv) > 1 else ".qwen/tmp/s38/suite2.log"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/dc_lat_complexity.png"
BLUE, ORANGE = "#2a78d6", "#eb6834"              # categorical slots 1, 2 (validated, light)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

pat = re.compile(r"eng=(\w+) spin=(\d) nside=(\d+) L=\d+ synth=([\d.]+) ana=([\d.]+)")
data = {}
for line in open(LOG):
    if line.startswith("LAT"):
        e, s, n, sy, an = pat.search(line).groups()
        data[(e, int(s), int(n))] = (float(sy), float(an))
ns = np.array(sorted({k[2] for k in data}), float)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, facecolor=SURF)
for ax, spin in zip(axes, (0, 2)):
    ax.set_facecolor(SURF)
    for eng, col, lab in (("dc", BLUE, "D\\&C + FMM"), ("march", ORANGE, "v2 march")):
        for k, ls, mk, dname in ((0, "-", "o", "synthesis"), (1, "--", "s", "analysis")):
            t = np.array([data[(eng, spin, int(n))][k] for n in ns])
            ax.plot(ns, t, ls, color=col, lw=2, marker=mk, ms=6, mec=SURF, mew=1.5,
                    label=f"{lab.replace(chr(92), '')}, {dname}")
        t = np.array([data[(eng, spin, int(n))][0] for n in ns])
        slope = np.log2(t[-1] / t[-2])
        ax.annotate(f"{lab.replace(chr(92), '')}\nslope {slope:.1f}", (ns[-1], t[-1]),
                    xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
    # reference slopes over 1024-4096: N^2 log N anchored on the D&C synthesis, N^3 on the march's
    ref = ns[ns >= 1024]
    a0, m0 = data[("dc", spin, 1024)][0], data[("march", spin, 1024)][0]
    ax.plot(ref, a0 * (ref / 1024) ** 2 * np.log2(3 * ref) / np.log2(3 * 1024), "-.", color=INK2, lw=1,
            label=r"reference $N^2 \log N$")
    ax.plot(ref, m0 * (ref / 1024) ** 3, ":", color=INK2, lw=1.3, label=r"reference $N^3$")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(ns); ax.set_xticklabels([str(int(n)) for n in ns])
    ax.set_xlim(ns[0] / 1.3, ns[-1] * 2.6)
    ax.set_title(f"spin {spin}", color=INK, fontsize=11, loc="left")
    ax.set_xlabel("Nside  (L = 3 Nside)", color=INK2)
    ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=INK2)
axes[0].set_ylabel("one latitudinal pass [ms]", color=INK2)
axes[0].legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
fig.suptitle("GMaster latitudinal transform, RTX PRO 6000 (GPU0): measured scaling",
             color=INK, fontsize=12, x=0.02, ha="left")
fig.tight_layout()
fig.savefig(OUT, dpi=160, facecolor=SURF)
print("wrote", OUT)
