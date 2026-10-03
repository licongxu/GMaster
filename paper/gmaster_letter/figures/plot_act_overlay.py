"""Four-panel ACT DR6 overlay for the letter. Writes fig_act_overlay.pdf.

Panels (a--b) need Stokes I and weight npy files (--T and --w). Panels (c--d) use committed
examples/act_dr6_tt_example/cl_*.npy when present.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EXAMPLE = REPO / "examples" / "act_dr6_tt_example"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=Path, default=None, help="Stokes I map npy for panels (a--b)")
    ap.add_argument("--w", type=Path, default=None, help="weight / footprint npy")
    ap.add_argument("--example", type=Path, default=EXAMPLE)
    args = ap.parse_args()

    ell = np.load(args.example / "bins.npy")
    cl_gm = np.load(args.example / "cl_gmaster.npy")
    cl_nm = np.load(args.example / "cl_namaster.npy")
    ratio = cl_gm / cl_nm - 1.0
    dl = ell * (ell + 1) / (2 * np.pi)

    have_map = args.T and args.T.is_file() and args.w and args.w.is_file()
    if have_map:
        m = np.load(args.T)
        w = np.load(args.w)
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4),
                                 gridspec_kw={"height_ratios": [1, 1.2], "width_ratios": [1, 1.3]})
        ax_m, ax_map, ax_cl, ax_r = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
        ax_m.imshow(w > 0, cmap="gray", origin="lower", aspect="auto")
        ax_m.set_title(r"(a) footprint mask")
        ax_m.set_xticks([])
        ax_m.set_yticks([])
        vmin, vmax = np.percentile(m[w > 0], [2, 98])
        ax_map.imshow(m, cmap="RdBu_r", origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
        ax_map.set_title(r"(b) Stokes $I$")
        ax_map.set_xticks([])
        ax_map.set_yticks([])
    else:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
        ax_cl, ax_r = axes
        print("Note: --T/--w not provided; skipping mask/map panels (c--d only).")

    ax_cl.plot(ell, dl * cl_gm, color="#CC0000", lw=1.1, label="GMaster")
    ax_cl.plot(ell, dl * cl_nm, color="#333333", lw=1.0, ls="--", label="NaMaster")
    ax_cl.set_yscale("log")
    ax_cl.set_ylabel(r"$D_\ell$")
    ax_cl.legend(frameon=False, fontsize=8)
    ax_cl.grid(True, which="both", linewidth=0.35, alpha=0.55)
    if have_map:
        ax_cl.set_title(r"(c) decoupled TT")
    ax_r.axhline(0, color="#333333", lw=0.6)
    ax_r.plot(ell, ratio, color="#CC0000", lw=0.9)
    ax_r.set_xlabel(r"$\ell$")
    ax_r.set_ylabel(r"$C_\ell^{\mathrm{GM}}/C_\ell^{\mathrm{NM}}-1$")
    ax_r.grid(True, which="both", linewidth=0.35, alpha=0.55)
    if have_map:
        ax_r.set_title(r"(d) fractional difference")

    fig.tight_layout()
    out = HERE / "fig_act_overlay.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
