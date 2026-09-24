"""GPU spherical-harmonic transforms, three engines: GMaster D&C, GMaster v2 march, SHTns.

One (engine, nside, spin) point per process, so each memory figure is that point's own.

  python bench_three_way.py gm <nside> <spin>      # route chosen by the environment:
        GMASTER_DC_MIN_L=0  -> the divide-and-conquer engine at every size
        GMASTER_DC=0        -> the v2 march
  python bench_three_way.py shtns <nside> <spin>   # spin 0: GPU fp64 and fp32; spin 2: see below

Operators.  GMaster: HEALPix, lmax = 3 nside - 1, alm2map, map2alm with n_iter 0 and 3 (NaMaster's
default).  Accuracy: max |d| / max |ref| against NaMaster (ducc0) on the same operator.  SHTns:
its own Gauss-Legendre grid (nlat = lmax + 1, nphi = 2 lmax + 2) -- exact quadrature, so no
iteration, and ~25 % fewer rings than HEALPix.  SHTns has no spin-2 transform: for spin 2 the
closest it offers, its spin-1 vector (spheroidal / toroidal) GPU transform in fp64, is timed and
labelled as such, with a round-trip error.  Memory: `dev` is the growth of device memory in use
(cudaMemGetInfo) from before the first transform to after the last; `peak` is XLA's accounted
peak (GMaster only).  Output: one JSON line.
"""
import json
import os
import sys
import time

import numpy as np


def med(f, reps, sync):
    f(); sync()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(); sync(); ts.append(time.perf_counter() - t)
    return 1e3 * float(np.median(ts))


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def dev_used():
    import ctypes

    for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        try:
            rt = ctypes.CDLL(name)
            break
        except OSError:
            continue
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    rt.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
    return (total.value - free.value) / 2**30


def run_gm(nside, spin, reps):
    import jax
    jax.config.update("jax_enable_x64", True)
    import gmaster as nmt
    import pymaster as ref

    lmax, npix = 3 * nside - 1, 12 * nside**2
    nmaps = 1 if spin == 0 else 2
    rng = np.random.default_rng(7)
    ainfo, minfo = nmt.NmtAlmInfo(lmax), nmt.NmtMapInfo(None, (npix,))
    alm = rng.normal(size=(nmaps, ainfo.nelem)) + 1j * rng.normal(size=(nmaps, ainfo.nelem))
    alm[:, ainfo._m == 0] = alm[:, ainfo._m == 0].real
    alm[:, ainfo._ell < spin] = 0
    maps = rng.normal(size=(nmaps, npix))
    # references first (host, untimed)
    r_minfo, r_ainfo = ref.NmtMapInfo(None, (npix,)), ref.NmtAlmInfo(lmax)
    r_syn = ref.alm2map(alm, spin, r_minfo, r_ainfo)
    r_ana = {k: ref.map2alm(maps, spin, r_minfo, r_ainfo, n_iter=k) for k in (0, 3)}
    d = jax.devices()[0]
    used0 = dev_used()
    alm_d, maps_d = jax.device_put(alm, d), jax.device_put(maps, d)
    sync = lambda: jax.block_until_ready(jax.numpy.zeros(1))
    out = {}
    syn = nmt.alm2map(alm_d, spin, minfo, ainfo); jax.block_until_ready(syn)
    out["syn_ms"] = med(lambda: jax.block_until_ready(nmt.alm2map(alm_d, spin, minfo, ainfo)), reps, sync)
    out["syn_err"] = rel(syn, r_syn)
    for k in (0, 3):
        a = nmt.map2alm(maps_d, spin, minfo, ainfo, n_iter=k); jax.block_until_ready(a)
        out[f"ana{k}_ms"] = med(lambda: jax.block_until_ready(
            nmt.map2alm(maps_d, spin, minfo, ainfo, n_iter=k)), reps, sync)
        out[f"ana{k}_err"] = rel(a, r_ana[k])
    st = d.memory_stats() or {}
    out["peak_gib"] = st.get("peak_bytes_in_use", 0) / 2**30
    out["dev_gib"] = dev_used() - used0
    return out


def run_shtns(nside, spin, reps):
    sys.path.insert(0, os.environ.get("SHTNS_DIR", "/scratch/scratch-lxu/shtns_build/shtns-git"))
    import ctypes
    import cupy as cp
    import ducc0
    import shtns

    lib = ctypes.CDLL((sys.modules.get("_shtns_cuda") or sys.modules["_shtns"]).__file__)
    for name in ("cu_SH_to_spat_float", "cu_spat_to_SH_float"):
        getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lmax = 3 * nside - 1
    nlat, nphi = lmax + 1 + ((lmax + 1) % 2), 2 * lmax + 2
    sync = cp.cuda.Device().synchronize
    rng = np.random.default_rng(0)
    out = {"grid": f"GL {nlat}x{nphi}"}
    if spin == 0:
        base = shtns.sht(lmax)
        alm = rng.normal(size=base.nlm) + 1j * rng.normal(size=base.nlm)
        alm[base.m == 0] = alm[base.m == 0].real
        r = ducc0.sht.experimental.synthesis_2d(alm=alm[None], spin=0, lmax=lmax, geometry="GL",
                                                ntheta=nlat, nphi=nphi, nthreads=8)[0]
        for prec in ("fp64", "fp32"):
            used0 = dev_used()
            sh = shtns.sht(lmax)
            flags = shtns.sht_gauss | shtns.SHT_THETA_CONTIGUOUS | shtns.SHT_ALLOW_GPU
            if prec == "fp32":
                flags |= shtns.SHT_FP32
            sh.set_grid(nlat, nphi, flags=flags)
            ft, ct = (np.float32, np.complex64) if prec == "fp32" else (np.float64, np.complex128)
            a_d = cp.asarray(alm.astype(ct)); x_d = cp.empty(nphi * nlat, ft); b_d = cp.empty_like(a_d)
            if prec == "fp32":
                cfg = int(sh.this)
                syn = lambda: lib.cu_SH_to_spat_float(cfg, a_d.data.ptr, x_d.data.ptr, lmax)
                ana = lambda: lib.cu_spat_to_SH_float(cfg, x_d.data.ptr, b_d.data.ptr, lmax)
            else:
                syn = lambda: sh.cu_SH_to_spat(a_d.data.ptr, x_d.data.ptr)
                ana = lambda: sh.cu_spat_to_SH(x_d.data.ptr, b_d.data.ptr)
            syn(); sync()
            got = cp.asnumpy(x_d).astype(np.float64).reshape(nphi, nlat).T
            keep = x_d.copy()
            out[f"{prec}_syn_ms"] = med(syn, reps, sync)
            x_d[...] = keep
            ana(); sync()
            out[f"{prec}_rt_err"] = rel(cp.asnumpy(b_d), alm)
            ts = []
            for _ in range(reps):
                x_d[...] = keep; sync(); t = time.perf_counter(); ana(); sync(); ts.append(time.perf_counter() - t)
            out[f"{prec}_ana_ms"] = 1e3 * float(np.median(ts))
            out[f"{prec}_syn_err"] = rel(got, r)
            out[f"{prec}_dev_gib"] = dev_used() - used0
            del sh, a_d, x_d, b_d, keep
            cp.get_default_memory_pool().free_all_blocks()
    else:   # spin-1 vector transform (SHTns has no spin 2), fp64
        used0 = dev_used()
        sh = shtns.sht(lmax)
        sh.set_grid(nlat, nphi, flags=shtns.sht_gauss | shtns.SHT_THETA_CONTIGUOUS | shtns.SHT_ALLOW_GPU)
        s = rng.normal(size=sh.nlm) + 1j * rng.normal(size=sh.nlm)
        t_ = rng.normal(size=sh.nlm) + 1j * rng.normal(size=sh.nlm)
        for a in (s, t_):
            a[sh.m == 0] = a[sh.m == 0].real
            a[sh.l < 1] = 0
        s_d, t_d = cp.asarray(s), cp.asarray(t_)
        vt, vp = cp.empty(nphi * nlat), cp.empty(nphi * nlat)
        s2, t2 = cp.empty_like(s_d), cp.empty_like(t_d)
        syn = lambda: sh.cu_SHsphtor_to_spat(s_d.data.ptr, t_d.data.ptr, vt.data.ptr, vp.data.ptr)
        ana = lambda: sh.cu_spat_to_SHsphtor(vt.data.ptr, vp.data.ptr, s2.data.ptr, t2.data.ptr)
        syn(); sync(); kt, kp = vt.copy(), vp.copy()
        out["vec1_syn_ms"] = med(syn, reps, sync)
        ts = []
        for _ in range(reps):
            vt[...] = kt; vp[...] = kp; sync(); t0 = time.perf_counter(); ana(); sync(); ts.append(time.perf_counter() - t0)
        out["vec1_ana_ms"] = 1e3 * float(np.median(ts))
        out["vec1_rt_err"] = max(rel(cp.asnumpy(s2), s), rel(cp.asnumpy(t2), t_))
        out["vec1_dev_gib"] = dev_used() - used0
    return out


if __name__ == "__main__":
    engine, nside, spin = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    reps = int(os.environ.get("REPS", "5"))
    res = run_gm(nside, spin, reps) if engine == "gm" else run_shtns(nside, spin, reps)
    route = ("dc" if os.environ.get("GMASTER_DC_MIN_L") == "0" else
             "march" if os.environ.get("GMASTER_DC") == "0" else "default") if engine == "gm" else "shtns"
    print("RESULT " + json.dumps({"engine": engine, "route": route, "nside": nside, "spin": spin, **res}),
          flush=True)
