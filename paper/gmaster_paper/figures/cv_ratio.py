"""Cosmic-variance comparison of the GMaster - NaMaster bandpower differences.

sigma_CV / C = sqrt(2 / ((2 l_eff + 1) f_sky Delta_l)) per bandpower, with
Delta_l = 30 and l_eff the bin centre of NmtBin.from_lmax_linear(3 Nside - 1, 30)
(bin i covers l = 2 + 30 i ... 31 + 30 i; the incomplete last bin is dropped),
and f_sky the mean of the binary ACT mask (w > 0) that the benchmark uses
(act_cache/act_w_nside*.npy).

GMaster bandpowers: by default those re-decoupled with the Gauss-Legendre fix (ducc0 nodes
and weights in the MCM quadrature; .qwen/tmp/boris_phase4/remcm.py), where they exist; set
GMASTER_CL_FIXED=0 for the original paper-run bandpowers.

Per-bandpower ratios |C_GM/C_NM - 1| / (sigma_CV/C) where both spectra exist:
  * Nside 1024-8192, TT and EE: march_rerun/cl/act_cl_gmaster_n*_s*.npy (float32
    march) against march_rerun/hist/act_cl_namaster96_n*_s*.npy (96-core NaMaster);
  * Nside 256 and 512, TT: accuracy/cells/cell_{gmaster,namaster}_n*_s0_l{3N-1}.npz
    (same estimator, GMaster default float32 march).
For the other benchmarked cells (Nside 64, 128 both spins; 256, 512 EE) no
per-bandpower spectra exist; an upper bound is given from the board maximum
(results.json agreement row) over the smallest sigma_CV/C of the cell (top bin),
with f_sky = 0.481 where no mask cache exists.

l_cut (per Nside): the largest l such that every bandpower whose upper edge is
below l_cut has |C_GM/C_NM - 1| <= 0.1 sigma_CV/C (10x below cosmic variance),
in both TT and EE where spectra exist; for bound-only cells, the full range if
the bound is <= 0.1. Writes act_fsky.json, cv_ratios.json.
"""
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = Path(os.environ.get(
    "GMASTER_MARCH_RERUN", HERE.parents[2] / ".qwen" / "tmp" / "accuracy" / "march_rerun"))
ACC = RUN.parent
CACHE = ACC / "act_cache"
FIXED = Path(os.environ.get("GMASTER_CL_FIXED_DIR", ACC.parent / "boris_phase4" / "cl"))
RESULTS = Path(os.environ.get(
    "GMASTER_RESULTS_JSON", HERE.parents[2] / "benchmarks" / "scaling" / "results.json"))
NLB = 30
FSKY_PAPER = 0.481
FSKY_JSON = HERE / "act_fsky.json"


def fsky(nside):
    known = json.loads(FSKY_JSON.read_text()) if FSKY_JSON.exists() else {}
    if str(nside) not in known:
        path = CACHE / f"act_w_nside{nside}.npy"
        if not path.exists():
            return None
        w = np.load(path, mmap_mode="r")
        n, chunk = 0, 1 << 24
        for i in range(0, w.size, chunk):
            n += int(np.count_nonzero(np.asarray(w[i:i + chunk]) > 0))
        known[str(nside)] = n / w.size
        FSKY_JSON.write_text(json.dumps(known, indent=1, sort_keys=True))
    return known[str(nside)]


def nbins(nside):
    return (3 * nside - 1 - 2 + 1) // NLB


def leff(n):
    return 2 + NLB * np.arange(n) + (NLB - 1) / 2


def sigma_cv(nside, n, fs=None):
    return np.sqrt(2.0 / ((2 * leff(n) + 1) * (fs or fsky(nside)) * NLB))


def spectra(nside, spin, fixed=None):
    """(GMaster, NaMaster, source) TT or EE bandpowers, or None without spectra.

    ``fixed`` (default: env GMASTER_CL_FIXED != "0") takes the GMaster bandpowers
    re-decoupled with the ducc0 Gauss-Legendre MCM fix (.qwen/tmp/boris_phase4/cl)
    where they exist; otherwise the paper runs' original GMaster bandpowers.
    """
    if fixed is None:
        fixed = os.environ.get("GMASTER_CL_FIXED", "1") != "0"
    fix = FIXED / f"act_cl_gmaster_glducc_n{nside}_s{spin}.npy"
    if nside >= 1024:
        nm = np.atleast_2d(np.load(RUN / "hist" / f"act_cl_namaster96_n{nside}_s{spin}.npy"))[0]
        if fixed and fix.exists():
            return np.atleast_2d(np.load(fix))[0], nm, "GL-fixed MCM (boris_phase4) vs hist/namaster96"
        gm = np.atleast_2d(np.load(RUN / "cl" / f"act_cl_gmaster_n{nside}_s{spin}.npy"))[0]
        return gm, nm, "march_rerun cl/ vs hist/namaster96"
    if spin == 0 and nside in (256, 512):
        lm = 3 * nside - 1
        g = np.load(ACC / "cells" / f"cell_gmaster_n{nside}_s0_l{lm}.npz")
        n = np.load(ACC / "cells" / f"cell_namaster_n{nside}_s0_l{lm}.npz")
        assert np.allclose(g["leff"], leff(g["leff"].size)), "bin centres differ"
        if fixed and fix.exists():
            return np.atleast_2d(np.load(fix))[0], n["cl_dec"][0], "GL-fixed MCM (boris_phase4) vs cells npz"
        return g["cl_dec"][0], n["cl_dec"][0], "accuracy/cells npz"
    return None


TARGET = 0.1


def ratio(nside, spin):
    """Per-bandpower |C_GM/C_NM - 1| / (sigma_CV/C), or None without spectra."""
    got = spectra(nside, spin)
    if got is None:
        return None
    gm, nm, _ = got
    return np.abs(gm / nm - 1) / sigma_cv(nside, gm.size)


def upper_edge(n):
    return 2 + NLB * np.arange(n) + NLB - 1


def lcut(nside, target=TARGET):
    """Exclusive l bound below which every bandpower is <= target x sigma_CV/C."""
    n = nbins(nside)
    cut = int(upper_edge(n)[-1]) + 1
    for spin in (0, 2):
        r = ratio(nside, spin)
        if r is None:
            continue
        bad = np.where(r > target)[0]
        if bad.size:
            cut = min(cut, int(upper_edge(n)[bad[0]] - NLB + 1))
    return cut


def main():
    board = {(r["nside"], r["spin"]): r for r in json.loads(RESULTS.read_text())["agreement"]}
    out = []
    for nside in (64, 128, 256, 512, 1024, 2048, 4096, 8192):
        for spin, name in ((0, "TT"), (2, "EE")):
            got = spectra(nside, spin)
            if got is not None:
                gm, nm, src = got
                assert gm.size == nbins(nside)
                rel = np.abs(gm / nm - 1)
                ratio = rel / sigma_cv(nside, gm.size)
                below = (leff(gm.size) + (NLB - 1) / 2) < 2 * nside   # upper bin edge < 2 Nside
                keep = upper_edge(gm.size) < lcut(nside)
                row = dict(nside=nside, spectrum=name, kind="per-bandpower", source=src,
                           lcut=lcut(nside), max_below_lcut=float(ratio[keep].max()),
                           fsky=fsky(nside), nbins=int(gm.size),
                           max_rel_all=float(rel.max()), max_rel_below_2nside=float(rel[below].max()),
                           max_all=float(ratio.max()), median_all=float(np.median(ratio)),
                           max_below_2nside=float(ratio[below].max()),
                           median_below_2nside=float(np.median(ratio[below])),
                           argmax_leff=float(leff(gm.size)[np.argmax(ratio)]))
            else:
                b = board[(nside, spin)]
                fs = fsky(nside)
                s = sigma_cv(nside, nbins(nside), fs or FSKY_PAPER)
                below = (leff(nbins(nside)) + (NLB - 1) / 2) < 2 * nside
                row = dict(nside=nside, spectrum=name, kind="upper bound (no spectra on disk)",
                           lcut=lcut(nside),
                           source="results.json agreement row / min sigma_CV/C",
                           fsky=fs or FSKY_PAPER, fsky_assumed=fs is None, nbins=nbins(nside),
                           max_all_bound=float(b["per_bandpower_max"] / s.min()),
                           max_below_2nside_bound=float(b["per_bandpower_max_below_2nside"] / s[below].min()))
            out.append(row)
            print(json.dumps(row))
    (HERE / "cv_ratios.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
