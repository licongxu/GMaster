"""Pseudo-Cl of the ACT+Planck DR6 NILC blackbody maps: TT, TE, EE.

These are the component-separated CMB maps, not the single-array coadd whose
auto-spectrum is noise-dominated.  T and E are projected to HEALPix and
decoupled with the NILC footprint.

  python act_nilc_cl.py
"""
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("GMASTER_DC", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np

NSIDE = 2048
NLB = 30
NILC = "/rds/datasets/act/NILC"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".qwen", "tmp", "act_nilc"))


def healpix(name):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{name}_n{NSIDE}.npy")
    if os.path.exists(path):
        return np.load(path)
    from pixell import enmap, reproject
    src = {
        "T": "act-planck_dr6.02_nilc_blackbody_T.fits",
        "E": "act-planck_dr6.02_nilc_blackbody_E.fits",
        "mask": "ilc_footprint_mask.fits",
    }[name]
    m = enmap.read_map(os.path.join(NILC, src))
    hp = np.asarray(reproject.map2healpix(
        m, nside=NSIDE, method="spline", order=1, spin=[0]), dtype=np.float64)
    np.save(path, hp)
    print("cached", name, hp.shape, flush=True)
    return hp


OUT = os.path.join(HERE, "cl_roundtrip", "act_nilc_n2048.npz")


def bin_decouple(pcl, mcm):
    """Same uniform Delta-ell bins as NmtBin.from_lmax_linear, then M^{-1}."""
    ells = np.arange(pcl.size)
    bpws = ((ells - 2) // NLB).astype(int)
    bpws[:2] = -1
    if np.sum(bpws == bpws[-1]) != NLB:
        bpws[bpws == bpws[-1]] = -1
    nb = int(bpws.max() + 1)
    binned = np.array([pcl[bpws == i].mean() for i in range(nb)])
    return np.linalg.solve(np.asarray(mcm), binned)


def gmaster_namaster():
    import jax
    jax.config.update("jax_enable_x64", True)
    import gmaster as nmt
    import pymaster as nm
    import pymaster.utils as pu
    nmt.set_latitudinal_method("march")
    cores = min(32, len(os.sched_getaffinity(0)))
    orig = pu._ducc_kwargs

    def _threads(*args, **kwargs):
        out = orig(*args, **kwargs)
        out["nthreads"] = cores
        return out

    pu._ducc_kwargs = _threads

    T, E, mask = healpix("T"), healpix("E"), healpix("mask")
    w = np.clip(mask, 0, None)
    lmax = 3 * NSIDE - 1

    def run(mod, block):
        bins = mod.NmtBin.from_lmax_linear(lmax, NLB)
        fT = mod.NmtField(w, [T], n_iter=3)
        fE = mod.NmtField(w, [E], n_iter=3)
        block(fT.get_alms())
        block(fE.get_alms())
        wtt, wee, wte = mod.NmtWorkspace(), mod.NmtWorkspace(), mod.NmtWorkspace()
        wtt.compute_coupling_matrix(fT, fT, bins)
        wee.compute_coupling_matrix(fE, fE, bins)
        wte.compute_coupling_matrix(fT, fE, bins)

        def dec(ws, a, b):
            return np.asarray(block(ws.decouple_cell(mod.compute_coupled_cell(a, b))))[0]

        return {
            "ell": np.asarray(bins.get_effective_ells()),
            "TT": dec(wtt, fT, fT), "EE": dec(wee, fE, fE), "TE": dec(wte, fT, fE),
            "almT": np.asarray(block(fT.get_alms()))[0],
            "almE": np.asarray(block(fE.get_alms()))[0],
            "mTT": np.asarray(wtt.mcm_binned), "mEE": np.asarray(wee.mcm_binned),
            "mTE": np.asarray(wte.mcm_binned),
        }

    print("GMaster", flush=True)
    gm = run(nmt, jax.block_until_ready)
    print("NaMaster", flush=True)
    ref = run(nm, lambda x: x)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, ell=gm["ell"], nside=NSIDE, nlb=NLB,
             gm_TT=gm["TT"], gm_EE=gm["EE"], gm_TE=gm["TE"],
             nm_TT=ref["TT"], nm_EE=ref["EE"], nm_TE=ref["TE"],
             almT=gm["almT"], almE=gm["almE"],
             mTT=gm["mTT"], mEE=gm["mEE"], mTE=gm["mTE"])
    print("wrote", OUT, flush=True)


def cl_cross(a, b, ell, m, lmax):
    weight = np.where(m == 0, 1.0, 2.0)
    prod = weight * np.real(np.asarray(a) * np.conj(np.asarray(b)))
    return np.bincount(ell, weights=prod, minlength=lmax + 1) / (2 * np.arange(lmax + 1) + 1)


def shtns_fp32all():
    """One SHTns map2alm of the NILC HEALPix maps. No synthesis.

    Each Gauss-Legendre node takes the value of the HEALPix pixel that
    contains it. SHTns only analyses that map.
    """
    os.environ["SHTNS_GPU_REC_PREC"] = "1"
    sys.path.insert(0, "/scratch/scratch-lxu/shtns_build/shtns-git")
    import ctypes
    import cupy as cp
    import healpy as hp
    import shtns
    from scipy.special import roots_legendre

    loaded = np.load(OUT)
    z = {k: loaded[k] for k in loaded.files}
    loaded.close()
    lmax = 3 * NSIDE - 1
    nlat = lmax + 1 + ((lmax + 1) % 2)
    nphi = 2 * lmax + 2
    sh = shtns.sht(lmax)
    flags = shtns.sht_gauss | shtns.SHT_THETA_CONTIGUOUS | shtns.SHT_ALLOW_GPU | shtns.SHT_FP32
    sh.set_grid(nlat, nphi, flags=flags)
    print("CONFIG", flush=True)
    sh.print_info()
    ell_idx, m_idx = np.asarray(sh.l), np.asarray(sh.m)
    mod = sys.modules.get("_shtns_cuda") or sys.modules.get("_shtns")
    lib = ctypes.CDLL(mod.__file__)
    lib.cu_spat_to_SH_float.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    cfg = int(sh.this)

    # Gauss nodes run from the north pole toward the south, phi starts at 0.
    theta = np.arccos(roots_legendre(nlat)[0][::-1])
    phi = 2 * np.pi * np.arange(nphi) / nphi
    T, E, mask = healpix("T"), healpix("E"), healpix("mask")
    w = np.clip(mask, 0, None)

    def on_grid(hpmap):
        buf = np.empty((nphi, nlat), np.float32)
        step = 64
        for i0 in range(0, nphi, step):
            i1 = min(i0 + step, nphi)
            tt = np.broadcast_to(theta, (i1 - i0, nlat))
            pp = np.broadcast_to(phi[i0:i1, None], (i1 - i0, nlat))
            buf[i0:i1] = hpmap[hp.ang2pix(NSIDE, tt, pp)]
        return buf

    def analyze(hpmap):
        x_d = cp.asarray(np.ascontiguousarray(on_grid(hpmap)))
        b_d = cp.empty(sh.nlm, np.complex64)
        lib.cu_spat_to_SH_float(cfg, x_d.data.ptr, b_d.data.ptr, lmax)
        cp.cuda.Device().synchronize()
        return cp.asnumpy(b_d).astype(np.complex128)

    print("analyse T", flush=True)
    aT = analyze(w * T)
    print("analyse E", flush=True)
    aE = analyze(w * E)
    spec = {
        "shtns_TT": bin_decouple(cl_cross(aT, aT, ell_idx, m_idx, lmax), z["mTT"]),
        "shtns_EE": bin_decouple(cl_cross(aE, aE, ell_idx, m_idx, lmax), z["mEE"]),
        "shtns_TE": bin_decouple(cl_cross(aT, aE, ell_idx, m_idx, lmax), z["mTE"]),
    }
    z.update(spec)
    np.savez(OUT, **z)
    for name in ("TT", "EE", "TE"):
        rel = np.max(np.abs(spec[f"shtns_{name}"] / z[f"gm_{name}"] - 1))
        print(f"SHTns/GMaster {name} max |C/C-1|={rel:.3e}", flush=True)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    {"shtns": shtns_fp32all}.get(sys.argv[1] if len(sys.argv) > 1 else "", gmaster_namaster)()
