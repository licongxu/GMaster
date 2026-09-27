"""Wall time versus Nside. Error bars are the min and max of five warm runs."""

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
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

ROOT = Path(__file__).resolve().parent
NSIDES = [64, 128, 256, 512, 1024, 2048, 4096, 8192]


def _mean_span(samples):
    values = np.asarray(samples, dtype=float)
    mean = float(values.mean())
    if values.size == 1:
        return mean, 0.0, 0.0, 1
    return mean, mean - float(values.min()), float(values.max()) - mean, int(values.size)


def _series(points):
    xs, ys, lo, hi, nrep = [], [], [], [], []
    for nside, samples in points:
        if not samples:
            continue
        mean, below, above, count = _mean_span(samples)
        xs.append(nside)
        ys.append(mean)
        lo.append(below)
        hi.append(above)
        nrep.append(count)
    return map(np.asarray, (xs, ys, lo, hi, nrep))


def _draw(ax, points, *, color, marker, label, dodge):
    xs, ys, lo, hi, nrep = _series(points)
    if xs.size == 0:
        return None
    xs = xs * 2.0 ** dodge
    artist = ax.errorbar(
        xs, ys, yerr=np.vstack([lo, hi]),
        color=color, marker=marker, markersize=4.5, linestyle="-",
        linewidth=1.0, capsize=2.0, elinewidth=0.7, label=label,
        markerfacecolor=color, markeredgecolor=color,
    )
    once = nrep == 1
    if np.any(once):
        ax.plot(
            xs[once], ys[once], linestyle="None", marker=marker, markersize=4.5,
            markerfacecolor="none", markeredgecolor=color, markeredgewidth=1.0,
        )
    return artist


def _load_master():
    runs = json.loads((ROOT / "act_warm_times.json").read_text())["runs"]
    out = {}
    for run in runs:
        if run["cores"] == 94:
            continue
        out[(run["engine"], run["cores"], run["spin"], run["nside"])] = run["seconds"]
    line = (ROOT / "gmaster8192_s0.log").read_text().split("RESULT ", 1)[1].splitlines()[0]
    saved = json.loads(line)
    out[("gmaster", 8, 0, 8192)] = saved["seconds"]
    for line in _result_lines("gmaster8192_s2_act.log"):
        out[("gmaster", 8, 2, 8192)] = line["seconds"]
    for line in (ROOT / "recheck_s0.log").read_text().splitlines():
        if not line.startswith("RESULT "):
            continue
        row = json.loads(line[len("RESULT "):])
        if row["engine"] == "gmaster" and row["spin"] == 0 and row["nside"] < 512:
            out[(row["engine"], row["cores"], row["spin"], row["nside"])] = row["seconds"]
    for line in (ROOT / "recheck_nm32_s0.log").read_text().splitlines():
        if not line.startswith("RESULT "):
            continue
        row = json.loads(line[len("RESULT "):])
        if row["cores"] == 32 and row["spin"] == 0 and row["nside"] <= 1024:
            out[(row["engine"], row["cores"], row["spin"], row["nside"])] = row["seconds"]
    return out


def _result_lines(name):
    path = ROOT / name
    if not path.exists():
        return []
    return [json.loads(line[len("RESULT "):]) for line in path.read_text().splitlines()
            if line.startswith("RESULT ")]


def _load_sht():
    march = {}
    for line in (ROOT / "three_way_gpu0.log").read_text().splitlines():
        if not line.startswith("RESULT "):
            continue
        row = json.loads(line[len("RESULT "):])
        if row.get("engine") == "gm" and row.get("route") == "march":
            march[(row["spin"], row["nside"])] = row
    shtns = {}
    gm8192 = {}
    for row in _result_lines("sht_warm5.log") + _result_lines("sht8192_s2.log"):
        if row.get("engine") == "shtns":
            shtns[(row["spin"], row["nside"])] = row
        elif row.get("engine") == "gm" and row.get("nside") == 8192:
            gm8192[row["spin"]] = row
    return march, shtns, gm8192


def _panel(ax, builders, ylabel):
    for build in builders:
        build(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(48, 12000)
    ax.set_xticks(NSIDES)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.set_xlabel(r"$N_{\mathrm{side}}$")
    ax.set_ylabel(r"wall time (s)")
    ax.grid(True, which="major", linewidth=0.3, alpha=0.5)


def _load_ducc():
    rows = json.loads((ROOT / "results_for_plots.json").read_text())["sht"]
    return {(row["spin"], row["nside"]): row for row in rows}


def main():
    master = _load_master()
    march, shtns, gm8192 = _load_sht()
    ducc = _load_ducc()

    def cl_points(engine, cores, spin):
        points = []
        for nside in NSIDES:
            points.append((nside, master.get((engine, cores, spin, nside))))
        return points

    # Container runs (`bench_nm_sht.py`, five warm runs) where they exist: all of 32 cores, and
    # 96 cores at 4096-8192.  Elsewhere the earlier medians.
    nm_sht = {(r["cores"], r["spin"], r["nside"]): r for r in _result_lines("nm_sht_8192.log")}

    def sht_ducc(spin, syn, cores):
        field = f"ducc{cores}_{'syn' if syn else 'ana'}_ms"
        points = []
        for nside in NSIDES:
            run = nm_sht.get((cores, spin, nside))
            if run is not None:
                times = run["syn_times_ms"] if syn else run["ana_times_ms"]
                points.append((nside, [t / 1000.0 for t in times]))
                continue
            row = ducc.get((spin, nside))
            points.append((nside, [row[field] / 1000.0] if row else None))
        return points

    def sht_gm(spin, key):
        points = []
        for nside in NSIDES:
            if nside == 8192 and spin in gm8192:
                points.append((nside, [t / 1000.0 for t in gm8192[spin][key]]))
                continue
            row = march.get((spin, nside))
            points.append((nside, [row[key[:-9] + "_ms"] / 1000.0] if row else None))
        return points

    def sht_shtns(spin, syn):
        points = []
        for nside in NSIDES:
            row = shtns.get((spin, nside))
            if row is None:
                points.append((nside, None))
                continue
            if spin == 0:
                field = "fp64_syn_times_ms" if syn else "fp64_ana_times_ms"
            else:
                field = "vec1_syn_times_ms" if syn else "vec1_ana_times_ms"
            points.append((nside, [t / 1000.0 for t in row[field]]))
        return points

    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.6), sharex=True)
    titles = [r"spin 0", r"spin 2"]
    row_titles = [
        r"\texttt{alm2map}",
        r"\texttt{map2alm} (one pass)",
        r"end-to-end $C_\ell$",
    ]
    for col, spin in enumerate((0, 2)):
        shtns_label = r"SHTns fp64" if spin == 0 else r"SHTns vector (spin 1), fp64"
        specs = [
            (
                axes[0, col],
                [
                    lambda ax, s=spin: _draw(
                        ax, sht_ducc(s, True, 32),
                        color="#555555", marker="s", label=r"NaMaster, 32 cores", dodge=-0.12,
                    ),
                    lambda ax, s=spin: _draw(
                        ax, sht_ducc(s, True, 96),
                        color="#111111", marker="D", label=r"NaMaster, 96 cores", dodge=-0.04,
                    ),
                    lambda ax, s=spin: _draw(
                        ax, sht_gm(s, "syn_times_ms"),
                        color="#0072B2", marker="o", label=r"GMaster march", dodge=0.04,
                    ),
                    lambda ax, s=spin, lab=shtns_label: _draw(
                        ax, sht_shtns(s, True),
                        color="#D55E00", marker="^", label=lab, dodge=0.12,
                    ),
                ],
            ),
            (
                axes[1, col],
                [
                    lambda ax, s=spin: _draw(
                        ax, sht_ducc(s, False, 32),
                        color="#555555", marker="s", label=r"NaMaster, 32 cores", dodge=-0.12,
                    ),
                    lambda ax, s=spin: _draw(
                        ax, sht_ducc(s, False, 96),
                        color="#111111", marker="D", label=r"NaMaster, 96 cores", dodge=-0.04,
                    ),
                    lambda ax, s=spin: _draw(
                        ax, sht_gm(s, "ana0_times_ms"),
                        color="#0072B2", marker="o", label=r"GMaster march", dodge=0.04,
                    ),
                    lambda ax, s=spin, lab=shtns_label: _draw(
                        ax, sht_shtns(s, False),
                        color="#D55E00", marker="^", label=lab, dodge=0.12,
                    ),
                ],
            ),
            (
                axes[2, col],
                [
                    lambda ax, s=spin: _draw(
                        ax, cl_points("namaster", 32, s),
                        color="#555555", marker="s", label=r"NaMaster, 32 cores", dodge=-0.08,
                    ),
                    lambda ax, s=spin: _draw(
                        ax, cl_points("namaster", 96, s),
                        color="#111111", marker="D", label=r"NaMaster, 96 cores", dodge=0.0,
                    ),
                    lambda ax, s=spin: _draw(
                        ax, cl_points("gmaster", 8, s),
                        color="#0072B2", marker="o", label=r"GMaster MASTER", dodge=0.08,
                    ),
                ],
            ),
        ]
        for ax, builders in specs:
            _panel(ax, builders, r"wall time (s)")
        axes[0, col].set_title(titles[col], pad=14)
    handles, labels = [], []
    seen = set()
    for ax in axes.ravel():
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label in seen:
                continue
            seen.add(label)
            handles.append(handle)
            labels.append(label)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.30, hspace=0.55, wspace=0.28)
    for row, title in enumerate(row_titles):
        pos = axes[row, 0].get_position()
        fig.text(0.53, pos.y1 + 0.008, title, ha="center", va="bottom", fontsize=9)
    fig.legend(
        handles, labels, loc="upper center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 0.145),
    )
    fig.suptitle(
        r"Warm wall time. Bars are the min and max of five runs.",
        fontsize=10, y=0.98,
    )
    fig.text(
        0.5, 0.012,
        "\n".join([
            r"Bottom row is ACT DR6. Transforms: NaMaster 32 cores and 96 cores at $N_{\mathrm{side}}\ge 4096$ are five container runs, 96 cores below that are medians; march medians through $4096$.",
            r"Spin 0, NaMaster 32 cores at $N_{\mathrm{side}}\le 1024$, and GMaster at 64, 128, 256, are one fresh run.",
            r"$N_{\mathrm{side}}=8192$ spin 2: transforms are five warm runs; GMaster end-to-end $C_\ell$ is one warm run (one GPU). The 94-core NaMaster runs are not shown.",
        ]),
        ha="center", va="bottom", fontsize=7,
    )
    out = ROOT / "time_vs_nside.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
