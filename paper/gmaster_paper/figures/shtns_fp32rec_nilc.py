"""SHTns gpu fp32/fp32 round trip of a NILC-shaped spectrum at Nside 8192.

SHTNS_GPU_REC_PREC=1 forces the Wigner-d march into float32 and disables
Ishioka's recurrence. print_info must report gpu fp32/fp32.

The input C_ell follows the NILC bandpowers (constant inside each Delta-ell=30
bin). Past the last NILC bin, C_ell is held at that bin so the high-ell
round trip has a target.
"""
import os
import sys

os.environ["SHTNS_GPU_REC_PREC"] = "1"
sys.path.insert(0, "/scratch/scratch-lxu/shtns_build/shtns-git")

import ctypes
from pathlib import Path

import cupy as cp
import numpy as np
import shtns

HERE = Path(__file__).resolve().parent
NILC = HERE / "cl_roundtrip" / "act_nilc_n2048.npz"
OUT = HERE / "cl_roundtrip" / "shtns_fp32rec_nilc_n8192.npz"
NSIDE = 8192
LMAX = 3 * NSIDE - 1
NLB = 30


def per_ell(cl_bin):
    target = np.zeros(LMAX + 1)
    for i, c in enumerate(np.asarray(cl_bin)):
        lo = 2 + NLB * i
        if lo > LMAX:
            break
        target[lo:min(lo + NLB, LMAX + 1)] = c
    if target[2] == 0:
        raise RuntimeError("empty NILC spectrum")
    last = target[target > 0][-1] if np.any(target > 0) else 0.0
    # hold the last measured bin through ell_max
    filled = np.where(target > 0)[0]
    if filled.size:
        target[filled[-1] + 1:] = target[filled[-1]]
    target[:2] = 0.0
    return target


def unit(rng, ell, m, lmax):
    a = rng.normal(size=ell.size) + 1j * rng.normal(size=ell.size)
    m0 = m == 0
    a[m0] = a[m0].real
    w = np.where(m0, 1.0, 2.0)
    raw = np.bincount(ell, weights=w * np.abs(a) ** 2, minlength=lmax + 1) / (2 * np.arange(lmax + 1) + 1)
    a *= np.sqrt(1.0 / np.maximum(raw, 1e-300))[ell]
    a[ell < 2] = 0
    return a


def cl_cross(a, b, ell, m, lmax):
    w = np.where(m == 0, 1.0, 2.0)
    prod = w * np.real(a * np.conj(b))
    return np.bincount(ell, weights=prod, minlength=lmax + 1) / (2 * np.arange(lmax + 1) + 1)


def main():
    z = np.load(NILC)
    ctt, cee, cte = per_ell(z["gm_TT"]), per_ell(z["gm_EE"]), per_ell(z["gm_TE"])
    # keep the cross inside the Cauchy bound of the two autos
    cte = np.clip(cte, -np.sqrt(np.maximum(ctt * cee, 0)), np.sqrt(np.maximum(ctt * cee, 0)))

    nlat = LMAX + 1 + ((LMAX + 1) % 2)
    nphi = 2 * LMAX + 2
    sh = shtns.sht(LMAX)
    flags = shtns.sht_gauss | shtns.SHT_THETA_CONTIGUOUS | shtns.SHT_ALLOW_GPU | shtns.SHT_FP32
    sh.set_grid(nlat, nphi, flags=flags)
    print("CONFIG", flush=True)
    sh.print_info()
    ell, m = np.asarray(sh.l), np.asarray(sh.m)
    rng = np.random.default_rng(0)
    xi, eta = unit(rng, ell, m, LMAX), unit(rng, ell, m, LMAX)
    sig_t = np.sqrt(np.maximum(ctt, 0.0))
    resid = np.sqrt(np.maximum(cee - np.where(ctt > 0, cte ** 2 / np.maximum(ctt, 1e-300), 0.0), 0.0))
    aT = (sig_t[ell] * xi).astype(np.complex64)
    aE = ((np.where(ctt > 0, cte / np.maximum(sig_t, 1e-300), 0.0))[ell] * xi + resid[ell] * eta).astype(np.complex64)
    aT[m == 0] = aT[m == 0].real
    aE[m == 0] = aE[m == 0].real

    mod = sys.modules.get("_shtns_cuda") or sys.modules.get("_shtns")
    lib = ctypes.CDLL(mod.__file__)
    for name in ("cu_SH_to_spat_float", "cu_spat_to_SH_float"):
        getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    cfg = int(sh.this)

    def roundtrip(alm):
        a_d = cp.asarray(alm)
        x_d = cp.empty(nphi * nlat, np.float32)
        b_d = cp.empty_like(a_d)
        lib.cu_SH_to_spat_float(cfg, a_d.data.ptr, x_d.data.ptr, LMAX)
        cp.cuda.Device().synchronize()
        lib.cu_spat_to_SH_float(cfg, x_d.data.ptr, b_d.data.ptr, LMAX)
        cp.cuda.Device().synchronize()
        return cp.asnumpy(b_d).astype(np.complex128)

    print("round trip T", flush=True)
    bT = roundtrip(aT)
    print("round trip E", flush=True)
    bE = roundtrip(aE)
    got = {
        "TT": cl_cross(bT, bT, ell, m, LMAX),
        "EE": cl_cross(bE, bE, ell, m, LMAX),
        "TE": cl_cross(bT, bE, ell, m, LMAX),
    }
    np.savez(OUT, TT=got["TT"], EE=got["EE"], TE=got["TE"],
             in_TT=ctt, in_EE=cee, in_TE=cte, lmax=LMAX)
    for name, src in (("TT", ctt), ("EE", cee), ("TE", cte)):
        rel = np.abs(got[name][2:] / np.maximum(np.abs(src[2:]), 1e-30) - 1)
        print(f"{name} median {np.median(rel):.3e} max {rel.max():.3e}", flush=True)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
