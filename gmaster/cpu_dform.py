"""ducc0-style CPU difference-form march: OpenMP over m, SIMD over rings."""
from __future__ import annotations

import ctypes
import os
import subprocess
import time

import numpy as np

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cpu", "dform_rings.c")
_SO = None
_LIB = None


def _lib():
    global _SO, _LIB
    if _LIB is not None:
        return _LIB
    cache = os.environ.get("GMASTER_CUDA_CACHE",
                           os.path.join(os.path.expanduser("~"), ".cache", "gmaster"))
    os.makedirs(cache, exist_ok=True)
    _SO = os.path.join(cache, "libgm_dform_rings.so")
    subprocess.check_call(
        ["gcc", "-O3", "-fopenmp", "-ffast-math", "-march=native", "-shared", "-fPIC",
         "-o", _SO, _SRC],
    )
    lib = ctypes.CDLL(_SO)
    lib.dform_rings_spin0.argtypes = [
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    _LIB = lib
    return _LIB


def _geometry(nside):
    nring = 4 * nside - 1
    rings = np.arange(1, 4 * nside)
    north = np.where(rings > 2 * nside, 4 * nside - rings, rings)
    cap = north < nside
    npix = 12 * nside ** 2
    theta = np.empty(nring)
    phi0 = np.empty(nring)
    nphi = np.empty(nring, np.int64)
    theta[cap] = 2 * np.arcsin(north[cap] / (np.sqrt(6) * nside))
    phi0[cap] = np.pi / (4 * north[cap])
    nphi[cap] = 4 * north[cap]
    theta[~cap] = np.arccos((2 * nside - north[~cap]) * (8 * nside / npix))
    phi0[~cap] = np.pi / (4 * nside) * (((north[~cap] - nside) & 1) == 0)
    nphi[~cap] = 4 * nside
    south = north != rings
    theta[south] = np.pi - theta[south]
    offsets = np.concatenate([[0], np.cumsum(nphi)[:-1]])
    x = (np.abs(np.cos(theta)) - 1.0).astype(np.float32)
    return theta, phi0, nphi, offsets, x, south.astype(np.uint8)


def ring_fft(maps, nside, L):
    """Real HEALPix map -> (L, nring) complex64 ring spectrum, m >= 0."""
    theta, phi0, nphi, offsets, x, south = _geometry(nside)
    nring = theta.size
    m = np.asarray(maps, np.float64).reshape(-1)
    ftm = np.zeros((L, nring), np.complex64)
    ms = np.arange(L)
    for i in range(nring):
        ring = m[offsets[i]:offsets[i] + nphi[i]]
        spec = np.fft.fft(ring)
        take = min(L, spec.size)
        ftm[:take, i] = spec[:take].astype(np.complex64)
        ftm[:take, i] *= np.exp(-1j * ms[:take] * phi0[i]).astype(np.complex64)
    return ftm, x, south


def dform_latitudinal(ftm, x, south, nthreads=0):
    L, nr = ftm.shape
    lib = _lib()
    rhs_re = np.ascontiguousarray(ftm.real.astype(np.float32))
    rhs_im = np.ascontiguousarray(ftm.imag.astype(np.float32))
    alm_re = np.zeros((L, L), np.float32)
    alm_im = np.zeros((L, L), np.float32)
    x = np.ascontiguousarray(x, np.float32)
    south = np.ascontiguousarray(south, np.uint8)
    lib.dform_rings_spin0(
        nr, L,
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        south.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        rhs_re.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        rhs_im.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        alm_re.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        alm_im.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        nthreads,
    )
    return alm_re + 1j * alm_im


def map2alm_spin0(maps, nside, nthreads=0):
    L = 3 * nside
    ftm, x, south = ring_fft(maps, nside, L)
    return dform_latitudinal(ftm, x, south, nthreads=nthreads)


if __name__ == "__main__":
    import pymaster as nm

    nthreads = int(os.environ.get("OMP_NUM_THREADS", "96"))
    rng = np.random.default_rng(0)
    print(f"nthreads={nthreads}", flush=True)
    print(f"{'nside':>6} {'fft':>8} {'dform':>8} {'cpuSHT':>8} {'NM':>8} {'dform/NM':>8}",
          flush=True)
    for nside in (128, 256, 512, 1024):
        L = 3 * nside
        npix = 12 * nside ** 2
        maps = rng.normal(size=npix)
        ftm, x, south = ring_fft(maps, nside, L)
        dform_latitudinal(ftm, x, south, nthreads)
        ts_fft, ts_d, ts_nm = [], [], []
        for _ in range(3):
            t0 = time.perf_counter()
            ftm, x, south = ring_fft(maps, nside, L)
            ts_fft.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            alm = dform_latitudinal(ftm, x, south, nthreads)
            ts_d.append(time.perf_counter() - t0)
        minfo = nm.NmtMapInfo(None, (npix,))
        ainfo = nm.NmtAlmInfo(L - 1)
        nm.map2alm(maps[None, :], 0, minfo, ainfo, n_iter=0)
        for _ in range(3):
            t0 = time.perf_counter()
            nm.map2alm(maps[None, :], 0, minfo, ainfo, n_iter=0)
            ts_nm.append(time.perf_counter() - t0)
        fft = float(np.median(ts_fft))
        dform = float(np.median(ts_d))
        tnm = float(np.median(ts_nm))
        print(f"{nside:6d} {fft*1e3:8.1f} {dform*1e3:8.1f} {(fft+dform)*1e3:8.1f} "
              f"{tnm*1e3:8.1f} {dform/tnm:8.2f}", flush=True)

    nside = 256
    L = 3 * nside
    nr = 4 * nside - 1
    x = np.linspace(-2.0, 0.0, nr, dtype=np.float32)
    south = np.zeros(nr, np.uint8)
    south[nr // 2:] = 1
    ftm = (rng.normal(size=(L, nr)) + 1j * rng.normal(size=(L, nr))).astype(np.complex64)
    a1 = dform_latitudinal(ftm, x, south, nthreads)
    a2 = dform_latitudinal(ftm, x, south, 1)
    rel = float(np.max(np.abs(a1 - a2)) / (np.max(np.abs(a1)) + 1e-30))
    print(f"96 vs 1 thread rel {rel:.2e} finite={np.isfinite(a1).all()}", flush=True)
