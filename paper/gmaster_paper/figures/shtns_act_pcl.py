"""ACT TT pseudo-Cl with the spherical-harmonic step done by SHTns.

SHTns has no mask coupling. The a_lm of the masked ACT temperature map are
already stored (GMaster, Nside 8192, n_iter=3). This script synthesizes that
field on the SHTns Gauss-Legendre grid in float64, analyzes it back with the
all-float32 recurrence, and decouples with the same binned mode-coupling
matrix that recovers the paper's TT bandpowers from the stored coupled spectrum.

  python shtns_act_pcl.py synth
  python shtns_act_pcl.py analys
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / ".qwen" / "tmp" / "boris_item5" / "data"
MAP = HERE.parents[2] / ".qwen" / "tmp" / "shtns_act_gl_map.npy"
OUT = HERE / "cl_roundtrip" / "shtns_fp32all_act_tt.npz"
LMAX = 3 * 8192 - 1
NLB = 30


def grid():
    nlat = LMAX + 1 + ((LMAX + 1) % 2)
    nphi = 2 * LMAX + 2
    return nlat, nphi


def shtns(fp32):
    if fp32:
        os.environ["SHTNS_GPU_REC_PREC"] = "1"
    sys.path.insert(0, "/scratch/scratch-lxu/shtns_build/shtns-git")
    import shtns as lib

    nlat, nphi = grid()
    sh = lib.sht(LMAX)
    flags = lib.sht_gauss | lib.SHT_THETA_CONTIGUOUS | lib.SHT_ALLOW_GPU
    if fp32:
        flags |= lib.SHT_FP32
    sh.set_grid(nlat, nphi, flags=flags)
    ell = np.asarray(sh.l)
    m = np.asarray(sh.m)
    # Stored a_lm are healpy / NaMaster m-major. SHTns uses that same order.
    hp_l = np.empty(ell.size, np.int32)
    hp_m = np.empty(ell.size, np.int32)
    k = 0
    for mm in range(LMAX + 1):
        n = LMAX - mm + 1
        hp_l[k:k + n] = np.arange(mm, LMAX + 1)
        hp_m[k:k + n] = mm
        k += n
    if not (np.array_equal(ell, hp_l) and np.array_equal(m, hp_m)):
        raise RuntimeError("SHTns a_lm order is not healpy m-major")
    return sh, ell, m


def cl_of(alm, ell, m):
    w = np.where(m == 0, 1.0, 2.0)
    ls = np.arange(LMAX + 1)
    return np.bincount(ell, weights=w * np.abs(alm) ** 2, minlength=LMAX + 1) / (2 * ls + 1)


def decouple(pcl):
    """Uniform Delta-ell=30 bins, then the stored binned MCM. Matches GMaster TT."""
    ells = np.arange(pcl.size)
    bpws = ((ells - 2) // NLB).astype(int)
    bpws[:2] = -1
    if np.sum(bpws == bpws[-1]) != NLB:
        bpws[bpws == bpws[-1]] = -1
    nb = int(bpws.max() + 1)
    binned = np.array([pcl[bpws == i].mean() for i in range(nb)])
    mcm = np.load(DATA / "mcm_binned_auto.npy")
    return np.linalg.solve(mcm, binned)


def synth():
    import cupy as cp

    sh, ell, m = shtns(fp32=False)
    alm = np.load(DATA / "alm_T_gm.npy")
    print("alm", alm.shape, alm.dtype, flush=True)
    a_d = cp.asarray(alm)
    del alm
    nlat, nphi = grid()
    x_d = cp.empty(nphi * nlat, np.float64)
    sh.cu_SH_to_spat(a_d.data.ptr, x_d.data.ptr)
    cp.cuda.Device().synchronize()
    mp = cp.asnumpy(x_d).astype(np.float32)
    np.save(MAP, mp)
    print(f"wrote {MAP}  shape {mp.shape}  rms {mp.std():.4e}", flush=True)


def analys():
    import ctypes
    import cupy as cp

    sh, ell, m = shtns(fp32=True)
    mp = np.load(MAP)
    x_d = cp.asarray(mp)
    del mp
    a_d = cp.empty(ell.size, np.complex64)
    mod = sys.modules.get("_shtns_cuda") or sys.modules.get("_shtns")
    lib = ctypes.CDLL(mod.__file__)
    lib.cu_spat_to_SH_float.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.cu_spat_to_SH_float(int(sh.this), x_d.data.ptr, a_d.data.ptr, LMAX)
    cp.cuda.Device().synchronize()
    pcl = cl_of(cp.asnumpy(a_d).astype(np.complex128), ell, m)
    cl = decouple(pcl)
    ref = decouple(np.load(DATA / "pcl_T_gm.npy"))
    OUT.parent.mkdir(exist_ok=True)
    np.savez(OUT, pcl=pcl, cl=cl, ref=ref, lmax=LMAX, nlb=NLB)
    rel = np.abs(cl / ref - 1)
    print(f"wrote {OUT}  median |C/Cref-1|={np.median(rel):.3e}  max={rel.max():.3e}", flush=True)


if __name__ == "__main__":
    {"synth": synth, "analys": analys}[sys.argv[1]]()
