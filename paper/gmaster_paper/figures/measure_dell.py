"""Round-trip D_ell at one Nside: SHTns precisions on its Gauss-Legendre grid,
GMaster float32 march and float64 NaMaster on the same HEALPix grid.

Input a_lm are scaled per ell so the realised C_ell is exactly
C_ell = 2 pi / (ell (ell+1)) for ell >= 2 (D_ell = 1). One process per engine,
because SHTns reads SHTNS_GPU_REC_PREC at GPU init.

  python measure_dell.py shtns-fp64 2048
  python measure_dell.py shtns-fp32 2048
  python measure_dell.py shtns-fp32all 2048
  python measure_dell.py gmaster 2048
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "cl_roundtrip"
NSIDE_DEFAULT = 2048


def cl_of(alm, ell_idx, m_idx, lmax):
    alm = np.asarray(alm)
    w = np.where(np.asarray(m_idx) == 0, 1.0, 2.0)
    ell = np.arange(lmax + 1)
    return np.bincount(ell_idx, weights=w * np.abs(alm) ** 2, minlength=lmax + 1) / (2 * ell + 1)


def act_tt(lmax):
    """NaMaster ACT TT bandpowers, constant across each Delta-ell = 30 bin."""
    path = HERE.parents[2] / ".qwen/tmp/accuracy/march_rerun/hist/act_cl_namaster96_n8192_s0.npy"
    cl = np.load(path)
    if cl.ndim == 2:
        cl = cl[0]
    target = np.zeros(lmax + 1)
    for i, c in enumerate(cl):
        lo = 2 + 30 * i
        if lo > lmax:
            break
        target[lo:min(lo + 30, lmax + 1)] = c
    return target


def realize(rng, ell_idx, m_idx, lmax, target=None):
    """Gaussian a_lm, then one scale per ell so C_ell equals ``target``.

    Default target is D_ell = 1. Pass ``act_tt(lmax)`` for the ACT TT shape.
    """
    alm = rng.normal(size=ell_idx.size) + 1j * rng.normal(size=ell_idx.size)
    m0 = np.asarray(m_idx) == 0
    alm[m0] = alm[m0].real
    alm[np.asarray(ell_idx) < 2] = 0
    raw = cl_of(alm, ell_idx, m_idx, lmax)
    if target is None:
        ell = np.arange(lmax + 1)
        target = np.zeros(lmax + 1)
        target[2:] = 2 * np.pi / (ell[2:] * (ell[2:] + 1))
    else:
        target = np.asarray(target, dtype=float).copy()
        target[:2] = 0
    alm *= np.sqrt(target / np.maximum(raw, 1e-300))[np.asarray(ell_idx)]
    return alm, target


def run_shtns(mode, nside, target=None):
    if mode == "fp32all":
        os.environ["SHTNS_GPU_REC_PREC"] = "1"
    sys.path.insert(0, "/scratch/scratch-lxu/shtns_build/shtns-git")
    import ctypes
    import cupy as cp
    import shtns

    lmax = 3 * nside - 1
    nlat = lmax + 1 + ((lmax + 1) % 2)
    nphi = 2 * lmax + 2
    sh = shtns.sht(lmax)
    flags = shtns.sht_gauss | shtns.SHT_THETA_CONTIGUOUS | shtns.SHT_ALLOW_GPU
    fp32 = mode != "fp64"
    if fp32:
        flags |= shtns.SHT_FP32
    sh.set_grid(nlat, nphi, flags=flags)
    ell_idx, m_idx = np.asarray(sh.l), np.asarray(sh.m)
    alm, target = realize(np.random.default_rng(0), ell_idx, m_idx, lmax, target)
    ft, ct = (np.float32, np.complex64) if fp32 else (np.float64, np.complex128)
    a_d = cp.asarray(alm.astype(ct))
    x_d = cp.empty(nphi * nlat, ft)
    b_d = cp.empty_like(a_d)
    mod = sys.modules.get("_shtns_cuda") or sys.modules.get("_shtns")
    if mod is None:
        names = [k for k in sys.modules if "sht" in k.lower()]
        raise RuntimeError(f"SHTns extension not loaded: {names}")
    lib = ctypes.CDLL(mod.__file__)
    for name in ("cu_SH_to_spat_float", "cu_spat_to_SH_float"):
        getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    if fp32:
        cfg = int(sh.this)
        lib.cu_SH_to_spat_float(cfg, a_d.data.ptr, x_d.data.ptr, lmax)
        cp.cuda.Device().synchronize()
        lib.cu_spat_to_SH_float(cfg, x_d.data.ptr, b_d.data.ptr, lmax)
    else:
        sh.cu_SH_to_spat(a_d.data.ptr, x_d.data.ptr)
        cp.cuda.Device().synchronize()
        sh.cu_spat_to_SH(x_d.data.ptr, b_d.data.ptr)
    cp.cuda.Device().synchronize()
    got = cl_of(cp.asnumpy(b_d).astype(np.complex128), ell_idx, m_idx, lmax)
    return target, got


def run_gmaster(nside):
    os.environ["JAX_ENABLE_X64"] = "1"
    os.environ["GMASTER_DC"] = "0"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    import jax
    jax.config.update("jax_enable_x64", True)
    import gmaster as nmt
    import pymaster as ref

    nmt.set_latitudinal_method("march")
    lmax, npix = 3 * nside - 1, 12 * nside ** 2
    ainfo = nmt.NmtAlmInfo(lmax)
    ell_idx, m_idx = np.asarray(ainfo._ell), np.asarray(ainfo._m)
    alm, target = realize(np.random.default_rng(0), ell_idx, m_idx, lmax)
    minfo = nmt.NmtMapInfo(None, (npix,))
    d = jax.devices()[0]
    alm_d = jax.device_put(alm[None, :], d)
    syn = np.asarray(nmt.alm2map(alm_d, 0, minfo, ainfo))
    back = np.asarray(nmt.map2alm(jax.device_put(syn, d), 0, minfo, ainfo, n_iter=0))[0]
    packed = alm[None, :]
    r_minfo, r_ainfo = ref.NmtMapInfo(None, (npix,)), ref.NmtAlmInfo(lmax)
    r_syn = ref.alm2map(packed, 0, r_minfo, r_ainfo)
    r_back = np.asarray(ref.map2alm(r_syn, 0, r_minfo, r_ainfo, n_iter=0))[0]
    return (
        target,
        cl_of(back, ell_idx, m_idx, lmax),
        cl_of(r_back, ell_idx, m_idx, lmax),
    )


def main():
    mode, nside = sys.argv[1], int(sys.argv[2] if len(sys.argv) > 2 else NSIDE_DEFAULT)
    use_act = len(sys.argv) > 3 and sys.argv[3] == "act"
    OUT.mkdir(exist_ok=True)
    tag = "_acttt" if use_act else ""
    if mode == "gmaster":
        target, got, ref = run_gmaster(nside)
        path = OUT / f"gmaster_n{nside}{tag}.npz"
        np.savez(path, target=target, got=got, ref=ref, nside=nside)
    else:
        key = {"shtns-fp64": "fp64", "shtns-fp32": "fp32", "shtns-fp32all": "fp32all"}[mode]
        lmax = 3 * nside - 1
        given = act_tt(lmax) if use_act else None
        target, got = run_shtns(key, nside, given)
        path = OUT / f"{mode.replace('-', '_')}_n{nside}{tag}.npz"
        np.savez(path, target=target, got=got, nside=nside)
    z = np.load(path)
    rel = np.abs(z["got"][2:] / z["target"][2:] - 1)
    print(f"wrote {path}  median |C/Cin-1|={np.median(rel):.3e}  max={rel.max():.3e}", flush=True)


if __name__ == "__main__":
    main()
