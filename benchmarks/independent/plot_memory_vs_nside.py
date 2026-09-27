"""Peak memory versus Nside, the companion of plot_time_vs_nside.py.

Transforms: GMaster is XLA's accounted device peak (`peak_gib`); SHTns is the growth of device
memory in use across its cell (`*_dev_gib`, cudaMemGetInfo), the only figure it exposes.
End-to-end ACT DR6: GMaster is the device peak, NaMaster the host peak RSS of its process.
Each cell is one process, so each peak is that cell's.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

ROOT = Path(__file__).resolve().parent
NSIDES = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
CARD_GIB = 95.6   # RTX PRO 6000 Blackwell, as nvidia-smi reports it


def _rows(name):
    path = ROOT / name
    if not path.exists():
        return []
    return [json.loads(line[len("RESULT "):]) for line in path.read_text().splitlines()
            if line.startswith("RESULT ")]


def _load():
    gm, shtns = {}, {}
    for row in _rows("three_way_gpu0.log") + _rows("sht_warm5.log") + _rows("sht8192_s2.log"):
        key = (row["spin"], row["nside"])
        if row["engine"] == "gm" and row["route"] == "march":
            gm[key] = row["peak_gib"]
        elif row["engine"] == "shtns":
            shtns[key] = row["fp64_dev_gib"] if row["spin"] == 0 else row["vec1_dev_gib"]
    act_gm = {(r["spin"], r["nside"]): r["gpu_peak_gib"] for r in _rows("act_mem_gmaster8.log")}
    act_nm = {(r["spin"], r["nside"]): r["host_peak_gib"] for r in _rows("act_mem_namaster96.log")}
    return gm, shtns, act_gm, act_nm


def _draw(ax, table, spin, **style):
    xs = [n for n in NSIDES if (spin, n) in table]
    if xs:
        ax.plot(xs, [table[(spin, n)] for n in xs], linestyle="-", linewidth=1.0,
                markersize=4.5, **style)


def main():
    gm, shtns, act_gm, act_nm = _load()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), sharex=True)
    for col, spin in enumerate((0, 2)):
        shtns_label = r"SHTns fp64" if spin == 0 else r"SHTns vector (spin 1), fp64"
        top, bottom = axes[0, col], axes[1, col]
        _draw(top, gm, spin, color="#0072B2", marker="o", label=r"GMaster march (GPU)")
        _draw(top, shtns, spin, color="#D55E00", marker="^", label=shtns_label + r" (GPU)")
        _draw(bottom, act_nm, spin, color="#111111", marker="D",
              label=r"NaMaster, 96 cores (host RSS)")
        _draw(bottom, act_gm, spin, color="#0072B2", marker="o", label=r"GMaster MASTER (GPU)")
        for ax in (top, bottom):
            ax.axhline(CARD_GIB, color="#999999", linewidth=0.7, linestyle="--",
                       label=r"one GPU (96 GB)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(48, 12000)
            ax.set_xticks(NSIDES)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_locator(LogLocator(base=10))
            ax.set_xlabel(r"$N_{\mathrm{side}}$")
            ax.set_ylabel(r"peak memory (GiB)")
            ax.grid(True, which="major", linewidth=0.3, alpha=0.5)
        top.set_title(r"spin 0" if spin == 0 else r"spin 2", pad=14)
    handles, labels, seen = [], [], set()
    for ax in axes.ravel():
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in seen:
                seen.add(label)
                handles.append(handle)
                labels.append(label)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.26, hspace=0.5, wspace=0.28)
    for row, title in enumerate((r"transform (\texttt{alm2map} + \texttt{map2alm}, $n_{\rm iter}=0, 3$)",
                                 r"end-to-end $C_\ell$, ACT DR6")):
        pos = axes[row, 0].get_position()
        fig.text(0.53, pos.y1 + 0.01, title, ha="center", va="bottom", fontsize=9)
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.16))
    fig.suptitle(r"Peak memory of one cell (one process per cell).", fontsize=10, y=0.98)
    fig.text(
        0.5, 0.01,
        "\n".join([
            r"GMaster: XLA's accounted device peak. SHTns: growth of device memory in use over its cell.",
            r"NaMaster: host peak RSS. NaMaster transform memory was not recorded.",
        ]),
        ha="center", va="bottom", fontsize=7,
    )
    out = ROOT / "memory_vs_nside.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
