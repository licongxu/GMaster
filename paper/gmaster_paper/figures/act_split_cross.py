"""Noise-free ACT DR6 TT, TE, EE from night PA4 f150 split crosses.

The coadd auto-spectrum is noise-dominated, so D_ell rises like ell^2.  A cross
of independent splits has no noise bias.  Same ivar weight and night beam on
every field; the beam is deconvolved by the mode-coupling matrix.

  python act_split_cross.py
"""
import os
import sys
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("GMASTER_DC", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import healpy as hp
import numpy as np
from astropy.io import fits

NSIDE = 2048
NLB = 40
MAPDIR = "/rds/datasets/act/act_dr6.02_maps_standard"
BEAM = ("/rds/datasets/act/act_dr6.02_beams/main_beams/nominal/"
        "coadd_pa4_f150_night_beam_tform_instant.txt")
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".qwen", "tmp", "act_cross"))
COLS = {"T": "TEMPERATURE", "Q": "Q_POLARISATION", "U": "U_POLARISATION"}


def read_column(path, column):
    with fits.open(path, memmap=True) as hdul:
        hdu = hdul[1]
        nside = int(hdu.header["NSIDE"])
        m = np.ascontiguousarray(hdu.data[column]).reshape(-1).astype(np.float32)
    return m, nside


def split_maps(k):
    os.makedirs(CACHE, exist_ok=True)
    out = {}
    for tag, column in COLS.items():
        path = os.path.join(CACHE, f"set{k}_{tag}_n{NSIDE}.npy")
        if os.path.exists(path):
            out[tag] = np.load(path)
            continue
        src = os.path.join(
            MAPDIR, f"act_dr6.02_std_AA_night_pa4_f150_4way_set{k}_map_srcfree_healpix.fits")
        t0 = time.perf_counter()
        m, native = read_column(src, column)
        m = np.where(np.isfinite(m), m, 0.0)
        if native != NSIDE:
            m = hp.ud_grade(m.astype(np.float64), NSIDE)
        m = np.ascontiguousarray(m, dtype=np.float64)
        np.save(path, m)
        print(f"cached set{k} {tag} in {time.perf_counter()-t0:.0f}s", flush=True)
        out[tag] = m
    return out


def ivar():
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"ivar_n{NSIDE}.npy")
    if os.path.exists(path):
        return np.load(path)
    from pixell import enmap, reproject
    t0 = time.perf_counter()
    src = os.path.join(MAPDIR, "act_dr6.02_std_AA_night_pa4_f150_4way_coadd_ivar.fits")
    w = np.asarray(reproject.map2healpix(
        enmap.read_map(src), nside=NSIDE, method="spline", order=1, spin=[0]), dtype=np.float64)
    w[~np.isfinite(w) | (w < 0)] = 0.0
    w /= w.max()
    np.save(path, w)
    print(f"cached ivar f_sky={np.mean(w>0):.3f} in {time.perf_counter()-t0:.0f}s", flush=True)
    return w


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    import gmaster as nmt
    nmt.set_latitudinal_method("march")

    w = ivar()
    maps = [split_maps(k) for k in range(4)]
    beam = np.loadtxt(BEAM)[:, 1]
    beam = beam / beam[0]
    lmax = 3 * NSIDE - 1
    beam = beam[:lmax + 1]
    bins = nmt.NmtBin.from_lmax_linear(lmax, NLB)

    def field(spin, stokes):
        mp = [stokes["T"]] if spin == 0 else [stokes["Q"], stokes["U"]]
        f = nmt.NmtField(w, mp, beam=beam, spin=spin, n_iter=3) if spin else nmt.NmtField(w, mp, beam=beam, n_iter=3)
        jax.block_until_ready(f.get_alms())
        return f

    print("fields", flush=True)
    t0 = time.perf_counter()
    ft = [field(0, s) for s in maps]
    fp = [field(2, s) for s in maps]
    print(f"fields {time.perf_counter()-t0:.0f}s", flush=True)

    def workspace(a, b):
        ws = nmt.NmtWorkspace()
        ws.compute_coupling_matrix(a, b, bins)
        return ws

    print("coupling", flush=True)
    t0 = time.perf_counter()
    wtt = workspace(ft[0], ft[0])
    wee = workspace(fp[0], fp[0])
    wte = workspace(ft[0], fp[0])
    print(f"coupling {time.perf_counter()-t0:.0f}s", flush=True)

    def dec(ws, a, b, row):
        cl = ws.decouple_cell(nmt.compute_coupled_cell(a, b))
        return np.asarray(jax.block_until_ready(cl))[row]

    acc = {k: [] for k in ("TT", "TE", "EE")}
    for i in range(4):
        for j in range(i + 1, 4):
            acc["TT"].append(dec(wtt, ft[i], ft[j], 0))
            acc["EE"].append(dec(wee, fp[i], fp[j], 0))
            acc["TE"].append(0.5 * (dec(wte, ft[i], fp[j], 0) + dec(wte, ft[j], fp[i], 0)))
            print(f"pair {i}x{j}", flush=True)
    ell = np.asarray(bins.get_effective_ells())
    out = os.path.join(HERE, "cl_roundtrip", "act_split_cross_n2048.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, ell=ell, TT=np.mean(acc["TT"], axis=0), TE=np.mean(acc["TE"], axis=0),
             EE=np.mean(acc["EE"], axis=0), nside=NSIDE, nlb=NLB)
    d = np.load(out)
    for name in ("TT", "TE", "EE"):
        dl = d["ell"] * (d["ell"] + 1) * d[name] / (2 * np.pi)
        print(name, "D at", {int(e): float(dl[np.argmin(np.abs(d["ell"]-e))])
                             for e in (500, 1000, 1500, 2000, 3000)})
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
