"""Tables and a figure from bench_three_way.py output (lines starting with RESULT)."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG = sys.argv[1] if len(sys.argv) > 1 else ".qwen/tmp/s38/three_way.log"
FIG = sys.argv[2] if len(sys.argv) > 2 else "docs/three_way_gpu.png"
rows = [json.loads(l[7:]) for l in open(LOG) if l.startswith("RESULT ")]
get = {(r["route"], r["nside"], r["spin"]): r for r in rows}
nsides = sorted({r["nside"] for r in rows})


def f(v, fmt="{:.2f}"):
    return "--" if v is None else fmt.format(v)


for spin in (0, 2):
    print(f"\n### spin {spin}: time (ms) synthesis / analysis n_iter=0 / analysis n_iter=3\n")
    sh_hdr = "SHTns GPU fp32 (GL) syn / ana" if spin == 0 else "SHTns GPU fp64 spin-1 vector (GL) syn / ana"
    print(f"| Nside | GMaster D&C | GMaster v2 march | {sh_hdr} |")
    print("|---|---|---|---|")
    for n in nsides:
        d, m, s = get.get(("dc", n, spin)), get.get(("march", n, spin)), get.get(("shtns", n, spin))
        cell = lambda r: "--" if r is None else f"{r['syn_ms']:.2f} / {r['ana0_ms']:.2f} / {r['ana3_ms']:.2f}"
        if s is None:
            sc = "--"
        elif spin == 0:
            sc = f"{s['fp32_syn_ms']:.2f} / {s['fp32_ana_ms']:.2f} (fp64 {s['fp64_syn_ms']:.2f} / {s['fp64_ana_ms']:.2f})"
        else:
            sc = f"{s['vec1_syn_ms']:.2f} / {s['vec1_ana_ms']:.2f}"
        print(f"| {n} | {cell(d)} | {cell(m)} | {sc} |")
    print(f"\n### spin {spin}: memory (GiB, device growth; XLA peak in brackets) and accuracy\n")
    print("| Nside | D&C mem | march mem | SHTns mem | D&C err syn / ana0 / ana3 | march err syn / ana0 / ana3 | SHTns err |")
    print("|---|---|---|---|---|---|---|")
    for n in nsides:
        d, m, s = get.get(("dc", n, spin)), get.get(("march", n, spin)), get.get(("shtns", n, spin))
        mem = lambda r: "--" if r is None else f"{r['dev_gib']:.2f} [{r['peak_gib']:.2f}]"
        err = lambda r: "--" if r is None else f"{r['syn_err']:.1e} / {r['ana0_err']:.1e} / {r['ana3_err']:.1e}"
        if s is None:
            sm = se = "--"
        elif spin == 0:
            sm = f"{s['fp64_dev_gib']:.2f} (fp64; fp32 n/a)"   # fp32 ran after fp64 in one process
            se = f"fp32 {s['fp32_syn_err']:.1e}, fp64 {s['fp64_syn_err']:.1e}"
        else:
            sm, se = f"{s['vec1_dev_gib']:.2f}", f"round trip {s['vec1_rt_err']:.1e}"
        print(f"| {n} | {mem(d)} | {mem(m)} | {sm} | {err(d)} | {err(m)} | {se} |")

BLUE, ORANGE, AQUA, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, facecolor=SURF)
for ax, spin in zip(axes, (0, 2)):
    ax.set_facecolor(SURF)
    for route, col, lab in (("dc", BLUE, "GMaster D&C"), ("march", ORANGE, "GMaster v2 march")):
        xs = [n for n in nsides if (route, n, spin) in get]
        ax.plot(xs, [get[(route, n, spin)]["ana3_ms"] for n in xs], "-o", color=col, lw=2, ms=6, mec=SURF,
                label=f"{lab}, map2alm n_iter=3")
        ax.plot(xs, [get[(route, n, spin)]["syn_ms"] for n in xs], "--s", color=col, lw=2, ms=6, mec=SURF,
                label=f"{lab}, alm2map")
    key = "fp32_syn_ms" if spin == 0 else "vec1_syn_ms"
    xs = [n for n in nsides if ("shtns", n, spin) in get]
    ax.plot(xs, [get[("shtns", n, spin)][key] for n in xs], ":^", color=AQUA, lw=2, ms=6, mec=SURF,
            label="SHTns GPU synthesis (" + ("fp32, GL grid" if spin == 0 else "spin-1 vector, fp64, GL") + ")")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(nsides); ax.set_xticklabels([str(n) for n in nsides])
    ax.set_title(f"spin {spin}", loc="left", color=INK, fontsize=11)
    ax.set_xlabel("Nside  (lmax = 3 Nside - 1)", color=INK2)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=INK2)
axes[0].set_ylabel("time [ms]", color=INK2)
axes[0].legend(frameon=False, fontsize=7.5, loc="upper left", labelcolor=INK)
fig.suptitle("GPU transforms, RTX PRO 6000 (GPU0): GMaster D&C vs v2 march vs SHTns",
             color=INK, fontsize=12, x=0.02, ha="left")
fig.tight_layout()
fig.savefig(FIG, dpi=160, facecolor=SURF)
print("\nwrote", FIG)
