"""ducc0-style CPU difference-form march: OpenMP over m, AVX-512 over rings (spin 0, analysis).

Experiment harness for plan/README.md ("can CPU (v, D) + fp32 beat NaMaster?").  The kernel in
``_cpu/dform_rings.c`` is a port of the CUDA spin-0 folded analysis and computes its coefficient
rows and seeds inside the call, so nothing is cached across calls but the ring geometry.

The full CPU transform is ducc0's own ring-FFT stage (``map2leg``, the code NaMaster's ``map2alm``
runs, quadrature weights included) followed by the C fold and the C march.  NaMaster's number is
``pymaster.map2alm`` (ducc0 ``adjoint_synthesis``, float64), and ducc0's ``leg2alm`` is timed in
float64 and float32 as the like-for-like latitudinal references.

Run under the venv with ``OMP_NUM_THREADS`` set: ducc0 reads it once at import, so one process
per thread count::

    OMP_NUM_THREADS=96 python -m gmaster.cpu_dform --nsides 128,256,512,1024
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import time
from functools import lru_cache

import numpy as np

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cpu", "dform_rings.c")
_LIB = None
V = 16
_FLAGS = ["-O3", "-fopenmp", "-march=native", "-mavx512f", "-mavx512dq", "-mfma"]
_F32 = ctypes.POINTER(ctypes.c_float)
_F64 = ctypes.POINTER(ctypes.c_double)


def _lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    cache = os.environ.get("GMASTER_CUDA_CACHE",
                           os.path.join(os.path.expanduser("~"), ".cache", "gmaster"))
    os.makedirs(cache, exist_ok=True)
    src = open(_SRC).read()
    digest = hashlib.sha1((src + " ".join(_FLAGS)).encode()).hexdigest()[:12]
    so = os.path.join(cache, f"libgm_dform_rings_{digest}.so")
    if not os.path.exists(so):
        tmp = so + f".{os.getpid()}.tmp"
        subprocess.check_call(["gcc", *_FLAGS, "-shared", "-fPIC", "-o", tmp, _SRC, "-lm"])
        os.replace(tmp, so)
    lib = ctypes.CDLL(so)
    lib.dform_rings_spin0.argtypes = [ctypes.c_int, ctypes.c_int, _F32, _F32, _F64, _F32, _F32,
                                      _F32, _F32, ctypes.c_int]
    lib.dform_rings_spin0.restype = None
    lib.fold_leg.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, _F32,
                             ctypes.c_int]
    lib.fold_leg.restype = None
    _LIB = lib
    return _LIB


@lru_cache(maxsize=8)
def geometry(nside, L):
    """HEALPix RING geometry (ducc0's ring arrays) and the kernel's northern-lane arrays."""
    nring = 4 * nside - 1
    rings = np.arange(1, 4 * nside)
    north_idx = np.where(rings > 2 * nside, 4 * nside - rings, rings)
    cap = north_idx < nside
    npix = 12 * nside ** 2
    theta = np.empty(nring)
    phi0 = np.empty(nring)
    nphi = np.empty(nring, np.int64)
    theta[cap] = 2 * np.arcsin(north_idx[cap] / (np.sqrt(6) * nside))
    phi0[cap] = np.pi / (4 * north_idx[cap])
    nphi[cap] = 4 * north_idx[cap]
    theta[~cap] = np.arccos((2 * nside - north_idx[~cap]) * (8 * nside / npix))
    phi0[~cap] = np.pi / (4 * nside) * (((north_idx[~cap] - nside) & 1) == 0)
    nphi[~cap] = 4 * nside
    south = north_idx != rings
    theta[south] = np.pi - theta[south]
    theta += 8 * np.finfo(np.float64).eps           # as `utils._stable_thetas`
    offsets = np.concatenate([[0], np.cumsum(nphi)[:-1]])

    north = (nring + 1) // 2                        # rings 0 .. north-1, equator last
    npad = -(-north // V) * V
    th = np.zeros(npad)
    th[:north] = theta[:north]
    valid = np.zeros(npad, bool)
    valid[:north] = True
    xv = np.cos(th)
    xm1 = np.where(valid, np.abs(xv) - 1.0, 0.0)
    xh = xm1.astype(np.float32)
    xl = (xm1 - xh.astype(np.float64)).astype(np.float32)
    lmax = float(L - 1)
    mlim = lmax * np.sin(th) + max(100.0, 0.01 * lmax)
    mlim = np.where(valid, mlim, -1.0).astype(np.float32)
    lg = np.where(valid, np.log2(np.sin(th / 2.0)) + np.log2(np.cos(th / 2.0)), 0.0)
    norm = np.sqrt((2.0 * np.arange(L) + 1.0) / (4.0 * np.pi)).astype(np.float32)
    return dict(theta=theta, phi0=phi0, nphi=nphi.astype(np.uint64),
                offsets=offsets.astype(np.uint64), npix=npix, nring=nring, north=north,
                npad=npad, xh=xh, xl=xl, mlim=mlim, lg=lg, norm=norm,
                ringfactor=np.full(nring, 4.0 * np.pi / npix))


def map2leg(maps, nside, L, nthreads):
    """ducc0's ring-FFT stage with the HEALPix quadrature weight: ``(nring, L)`` complex."""
    import ducc0
    g = geometry(nside, L)
    m = np.asarray(maps)
    return ducc0.sht.map2leg(map=m.reshape(1, -1), nphi=g["nphi"], phi0=g["phi0"],
                             ringstart=g["offsets"], ringfactor=g["ringfactor"], mmax=L - 1,
                             nthreads=nthreads)[0]


def fold(leg, nside, L, nthreads):
    """``(nring, L)`` complex128 -> the kernel's folded rhs ``(L, 4, npad)`` float32."""
    g = geometry(nside, L)
    leg = np.ascontiguousarray(leg, np.complex128)
    rhs = np.empty((L, 4, g["npad"]), np.float32)
    _lib().fold_leg(leg.ctypes.data, g["nring"], L, g["npad"], rhs.ctypes.data_as(_F32),
                    nthreads)
    return rhs


def dform_latitudinal(rhs, nside, L, nthreads=0):
    """The march: folded rhs in, ``(L, L, 2)`` float32 ``[m, ell, (re, im)]`` out, carrying the
    closed form's ``(-1)^m`` and ``sqrt((2l+1)/4pi)`` as `_fold_analyze` does."""
    g = geometry(nside, L)
    out = np.empty((L, L, 2), np.float32)
    _lib().dform_rings_spin0(
        g["npad"], L, g["xh"].ctypes.data_as(_F32), g["xl"].ctypes.data_as(_F32),
        g["lg"].ctypes.data_as(_F64), g["mlim"].ctypes.data_as(_F32), rhs.ctypes.data_as(_F32),
        g["norm"].ctypes.data_as(_F32), out.ctypes.data_as(_F32), nthreads)
    return out


def map2alm_spin0(maps, nside, nthreads=0):
    """Full CPU transform: ducc0 map2leg -> C fold -> C march.  ``(L, L, 2)`` float32."""
    L = 3 * nside
    return dform_latitudinal(fold(map2leg(maps, nside, L, nthreads), nside, L, nthreads),
                             nside, L, nthreads)


def as_alm(out):
    """``(m, ell, 2)`` kernel output -> ``(ell, m)`` complex block."""
    return (out[..., 0] + 1j * out[..., 1]).T


def to_packed(alm_lm, L):
    """(ell, m) block -> healpy / NaMaster packed order."""
    lmax = L - 1
    ell, m = np.tril_indices(L)
    idx = (m * (2 * lmax + 1 - m)) // 2 + ell
    packed = np.zeros((L * (L + 1)) // 2, np.complex128)
    packed[idx] = alm_lm[ell, m]
    return packed


def _median_time(fn, reps=3):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def _rel(got, want):
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


if __name__ == "__main__":
    import argparse

    import ducc0
    import pymaster as nm

    ap = argparse.ArgumentParser()
    ap.add_argument("--nsides", default="128,256,512,1024")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    thr = int(os.environ.get("OMP_NUM_THREADS", "0"))
    if thr <= 0:
        raise SystemExit("set OMP_NUM_THREADS (ducc0 reads it at import)")
    rng = np.random.default_rng(0)
    print(f"threads={thr}.  All times ms, medians of {args.reps} warm runs, one random map.\n"
          "NM       : pymaster.map2alm (ducc0 float64, FFT + latitudinal), n_iter=0\n"
          "dform    : ducc0 map2leg (FFT) + C fold + AVX-512 fp32 (v,D) march  [full transform]\n"
          "march    : the (v,D) march alone;  leg64/leg32: ducc0 leg2alm alone, float64 / float32\n"
          "rel      : max |alm - NM| / max |NM|", flush=True)
    print(f"{'nside':>5} {'thr':>3} {'NM':>8} {'dform':>8} {'dform/NM':>8} | {'fft':>7} "
          f"{'fold':>6} {'march':>8} {'leg64':>8} {'leg32':>8} {'march/leg64':>11} | "
          f"{'rel dform':>9} {'rel leg32':>9} {'finite':>6}", flush=True)
    for nside in [int(s) for s in args.nsides.split(",")]:
        L = 3 * nside
        npix = 12 * nside ** 2
        maps = rng.normal(size=npix)
        g = geometry(nside, L)
        minfo = nm.NmtMapInfo(None, (npix,))
        ainfo = nm.NmtAlmInfo(L - 1)
        want = nm.map2alm(maps[None, :], 0, minfo, ainfo, n_iter=0)[0]

        leg = map2leg(maps, nside, L, thr)
        rhs = fold(leg, nside, L, thr)
        got = to_packed(as_alm(dform_latitudinal(rhs, nside, L, thr)), L)
        got_full = to_packed(as_alm(map2alm_spin0(maps, nside, thr)), L)
        assert np.array_equal(got, got_full)
        finite = bool(np.isfinite(got).all())
        leg32 = leg.astype(np.complex64)
        alm32 = ducc0.sht.leg2alm(leg=leg32[None], lmax=L - 1, theta=g["theta"], spin=0,
                                  nthreads=thr)[0]
        rel_d, rel_32 = _rel(got, want), _rel(alm32.astype(np.complex128), want)

        t_nm = _median_time(lambda: nm.map2alm(maps[None, :], 0, minfo, ainfo, n_iter=0),
                            args.reps)
        t_full = _median_time(lambda: map2alm_spin0(maps, nside, thr), args.reps)
        t_fft = _median_time(lambda: map2leg(maps, nside, L, thr), args.reps)
        t_fold = _median_time(lambda: fold(leg, nside, L, thr), args.reps)
        t_march = _median_time(lambda: dform_latitudinal(rhs, nside, L, thr), args.reps)
        t_leg64 = _median_time(lambda: ducc0.sht.leg2alm(leg=leg[None], lmax=L - 1,
                                                         theta=g["theta"], spin=0,
                                                         nthreads=thr), args.reps)
        t_leg32 = _median_time(lambda: ducc0.sht.leg2alm(leg=leg32[None], lmax=L - 1,
                                                         theta=g["theta"], spin=0,
                                                         nthreads=thr), args.reps)
        ms = 1e3
        print(f"{nside:5d} {thr:3d} {t_nm*ms:8.2f} {t_full*ms:8.2f} {t_full/t_nm:8.2f} | "
              f"{t_fft*ms:7.2f} {t_fold*ms:6.2f} {t_march*ms:8.2f} {t_leg64*ms:8.2f} "
              f"{t_leg32*ms:8.2f} {t_march/t_leg64:11.2f} | {rel_d:9.1e} {rel_32:9.1e} "
              f"{str(finite):>6}", flush=True)
