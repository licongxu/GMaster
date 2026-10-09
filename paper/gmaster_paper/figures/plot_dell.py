"""D_ell and per-multipole C_ell error from measure_dell.py.

Input a_lm are scaled so D_ell = ell(ell+1) C_ell / 2pi = 1 for ell >= 2.
SHTns round trips are on its Gauss-Legendre grid, so float64 recovers the input.
GMaster is compared with NaMaster on the same HEALPix grid.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter

from make_figures import BLACK, TEX_RC

RED = "#D62728"
HERE = Path(__file__).resolve().parent
OUT = HERE / "cl_roundtrip"
TEXT_W = 508.0 / 72.27
BIN = 60


def load(name):
    path = OUT / name
    if not path.exists():
        return None
    z = np.load(path)
    ell = np.arange(z["target"].size)
    return ell, z


def binned(ell, y, width=BIN):
    y = np.asarray(y, dtype=float)
    edges = np.arange(2, ell[-1] + width, width)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sl = y[lo:min(hi, y.size)]
        sl = sl[np.isfinite(sl)]
        if sl.size:
            xs.append(0.5 * (lo + min(hi, y.size) - 1))
            ys.append(np.median(sl))
    return np.asarray(xs), np.asarray(ys)


def safe_div(num, den):
    out = np.full(np.shape(num), np.nan, dtype=float)
    ok = np.asarray(den) > 0
    out[ok] = np.asarray(num)[ok] / np.asarray(den)[ok]
    return out


def ratio(got, ref):
    r = np.abs(safe_div(got, ref) - 1)
    r[:2] = np.nan
    return r


def style(ax):
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color="#D9D9D9", lw=0.5)
    ax.grid(True, which="minor", color="#EEEEEE", lw=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=3.0, labelsize=8)
    ax.tick_params(which="minor", length=1.4)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)


def main():
    fp32all = load("shtns_fp32all_n8192.npz")
    fp32 = load("shtns_fp32_n8192.npz")
    fp64 = load("shtns_fp64_n8192.npz")
    fp32all_4k = load("shtns_fp32all_n4096.npz")
    gm = load("gmaster_n8192.npz") or load("gmaster_n2048.npz")
    if fp32all is None:
        raise SystemExit("missing shtns_fp32all_n8192.npz")
    ell, z = fp32all
    # Left panel is absolute D_ell. fp64 and fp32-data sit on the input line;
    # they are distinguished on the right.
    dell_curves = [
        (ell, safe_div(z["got"], z["target"]), BLACK, (0, (3.0, 1.2)),
         r"SHTns fp32 all, $N_{\mathrm{side}}=8192$"),
    ]
    frac = [
        (ell, ratio(z["got"], z["target"]), BLACK, (0, (3.0, 1.2)),
         r"SHTns fp32 all, $N_{\mathrm{side}}=8192$"),
    ]
    if fp32all_4k is not None:
        e, w = fp32all_4k
        frac.append((e, ratio(w["got"], w["target"]), "#888888", "-",
                     r"SHTns fp32 all, $N_{\mathrm{side}}=4096$"))
    if fp32 is not None:
        e, w = fp32
        frac.append((e, ratio(w["got"], w["target"]), "#222222", (0, (0.5, 1.4)),
                     r"SHTns fp32 data, $N_{\mathrm{side}}=8192$"))
    # fp64 round trip is ~1e-12, below the axis. The input line on the left is that result.
    gm_curve = None
    if gm is not None:
        e, w = gm
        nside = int(np.asarray(w["nside"]).reshape(-1)[0]) if "nside" in w.files else (
            8192 if e.size > 20000 else 2048)
        gm_curve = (e, ratio(w["got"], w["ref"]),
                    rf"GMaster / NaMaster, $N_{{\mathrm{{side}}}}={nside}$")

    with plt.rc_context(TEX_RC):
        fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.55), sharex=True, layout="constrained")
        left, right = axes
        left.axhline(1.0, color=BLACK, lw=0.9, zorder=1, label=r"input, $D_\ell=1$")
        right.set_yscale("log")
        right.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
        right.yaxis.set_minor_formatter(NullFormatter())
        for e, dell, color, ls, label in dell_curves:
            left.plot(*binned(e, dell), color=color, ls=ls, lw=1.15, label=label)
        for e, rel, color, ls, label in frac:
            right.plot(*binned(e, rel), color=color, ls=ls, lw=1.05, label=label)
        if gm_curve is not None:
            e, rel, label = gm_curve
            right.plot(*binned(e, rel), color=RED, ls="-", lw=1.15, label=label)
        for ax in axes:
            style(ax)
            ax.set_xlim(2, 3 * 8192)
            ax.set_xlabel(r"$\ell$")
        left.set_ylabel(r"$D_\ell$")
        left.set_ylim(0.85, 5.0)
        right.set_ylabel(r"$\lvert C_\ell/C_\ell^{\mathrm{ref}}-1\rvert$")
        right.set_ylim(1e-8, 8)
        # GMaster's reference is NaMaster; every SHTns curve uses the input C_ell.
        h_in, lab_in = left.get_legend_handles_labels()
        h_r, lab_r = right.get_legend_handles_labels()
        handles = [h_in[0]] + h_r
        labels = [lab_in[0]] + lab_r
        fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False,
                   fontsize=7.5, handlelength=2.6, columnspacing=0.9, borderaxespad=0.02)
        fig.savefig(HERE / "fig_dell.pdf", dpi=300)
        fig.savefig(HERE / "fig_dell.png", dpi=300)
        plt.close(fig)
    if fp64 is not None:
        print("fp64 max |C/Cin-1|", np.nanmax(ratio(fp64[1]["got"], fp64[1]["target"])))
    print("wrote fig_dell")


if __name__ == "__main__":
    main()
