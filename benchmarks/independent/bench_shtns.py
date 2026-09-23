"""SHTns (CPU OpenMP fp64, GPU fp64, GPU fp32) spin-0 SHT timing at lmax = 3 nside - 1.

Operator: SHTns' own Gauss-Legendre grid (nlat = lmax + 1 rounded up to even, nphi = 2 lmax + 2),
theta-contiguous, orthonormal harmonics.  This is NOT the HEALPix grid (SHTns has none): it has
~1.5 nside ring pairs per m against HEALPix's 2 nside, i.e. 25 % less latitudinal work than the
HEALPix transforms it is compared with.  GPU arrays are device-resident CuPy buffers passed by
pointer (cu_SH_to_spat / cu_spat_to_SH), so no host transfer is timed.  Accuracy: synthesis
against ducc0 synthesis_2d on the same GL grid (max |diff| / max |ref|).

Usage: python bench_shtns.py <nside> [<nside> ...]
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get("SHTNS_DIR", "/scratch/scratch-lxu/shtns_build/shtns-git"))
import ctypes

import cupy as cp
import ducc0
import shtns

# SHTns implements fp32 device transforms in C but its Python layer only exposes the fp64 ones.
_lib = ctypes.CDLL((sys.modules.get("_shtns_cuda") or sys.modules["_shtns"]).__file__)
for _name in ("cu_SH_to_spat_float", "cu_spat_to_SH_float"):
    getattr(_lib, _name).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


def med(f, reps=7, sync=None, pre=None):
    """Median wall time of f(); `pre` runs untimed before every call (SHTns analysis destroys its input)."""
    if pre:
        pre()
    f()
    if sync:
        sync()
    ts = []
    for _ in range(reps):
        if pre:
            pre()
            if sync:
                sync()
        t = time.perf_counter()
        f()
        if sync:
            sync()
        ts.append(time.perf_counter() - t)
    return 1e3 * float(np.median(ts))


def main():
    nthreads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count()))
    for nside in [int(a) for a in sys.argv[1:]]:
        lmax = 3 * nside - 1
        nlat = lmax + 1 + ((lmax + 1) % 2)
        nphi = 2 * lmax + 2
        rng = np.random.default_rng(0)
        base = shtns.sht(lmax)
        nlm = base.nlm
        alm = rng.normal(size=nlm) + 1j * rng.normal(size=nlm)
        alm[base.m == 0] = alm[base.m == 0].real
        # reference on the same grid; ducc0 wants healpy-ordered alm, which is SHTns' m-major order
        ref = ducc0.sht.experimental.synthesis_2d(
            alm=alm[None].astype(np.complex128), spin=0, lmax=lmax, geometry="GL",
            ntheta=nlat, nphi=nphi, nthreads=nthreads)[0]
        # SHTns orthonormal uses the Condon-Shortley phase like ducc0; theta-contiguous is (nphi, nlat)
        rows = []
        for mode in ("cpu", "gpu64", "gpu32"):
            sh = shtns.sht(lmax)
            flags = shtns.sht_gauss | shtns.SHT_THETA_CONTIGUOUS
            if mode != "cpu":
                flags |= shtns.SHT_ALLOW_GPU
            if mode == "gpu32":
                flags |= shtns.SHT_FP32
            sh.set_grid(nlat, nphi, flags=flags)
            if mode == "cpu":
                x = np.asarray(sh.synth(alm))
                got = x.T.copy()
                xin = x.copy()
                ts = med(lambda: sh.synth(alm))
                ta = med(lambda: sh.analys(x), pre=lambda: np.copyto(x, xin))
            else:
                ft, ct = (np.float32, np.complex64) if mode == "gpu32" else (np.float64, np.complex128)
                a_d = cp.asarray(alm.astype(ct))
                x_d = cp.empty(nphi * nlat, ft)
                b_d = cp.empty_like(a_d)
                sync = cp.cuda.Device().synchronize
                if mode == "gpu32":
                    cfg = int(sh.this)
                    syn = lambda: _lib.cu_SH_to_spat_float(cfg, a_d.data.ptr, x_d.data.ptr, lmax)
                    ana = lambda: _lib.cu_spat_to_SH_float(cfg, x_d.data.ptr, b_d.data.ptr, lmax)
                else:
                    syn = lambda: sh.cu_SH_to_spat(a_d.data.ptr, x_d.data.ptr)
                    ana = lambda: sh.cu_spat_to_SH(x_d.data.ptr, b_d.data.ptr)
                ts = med(syn, sync=sync)
                got = cp.asnumpy(x_d).astype(np.float64).reshape(nphi, nlat).T
                x_keep = x_d.copy()
                ta = med(ana, sync=sync, pre=lambda: x_d.__setitem__(Ellipsis, x_keep))
            err = float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))
            rows.append((mode, ts, ta, err))
            del sh
        line = "  ".join(f"{m}: syn {s:.2f} ana {a:.2f} ms err {e:.1e}" for m, s, a, e in rows)
        print(f"SHTns nside={nside} lmax={lmax} grid={nlat}x{nphi} threads={nthreads} | {line}", flush=True)


if __name__ == "__main__":
    main()
