"""Sub-cubic spin-0 latitudinal transform: divide-and-conquer eigenbasis + 1-D FMM (CUDA).

For order ``m`` and parity ``p`` the orthonormal Legendre functions of that parity,
``phi_k(y) = psi_{m+p+2k}(sqrt y)`` with ``y = cos^2 theta`` and ``psi = sqrt(2 pi) lambda_lm``
(positive ``psi_mm``, no Condon-Shortley phase), obey a symmetric three-term recurrence
``y phi_k = E_k phi_{k+1} + D_k phi_k + E_{k-1} phi_{k-1}``.  With ``T = V Y V^T`` its Jacobi matrix
(``Y`` the zeros ``y_j`` of ``phi_n``), the Christoffel-Darboux identity turns synthesis at any
ring ``y_r`` into

    f(y_r) = sum_k c_k phi_k(y_r) = E_{n-1} phi_n(y_r) sum_j V[n-1, j] (V^T c)_j / (y_r - y_j).

``V^T`` is applied through Cuppen's divide-and-conquer tree of ``T`` (Gu & Eisenstat's stable form):
16 x 16 leaves (eigenvectors rebuilt on the fly from stored eigenvalues), then one Cauchy-like rank-one-update merge per level, each a 1-D Cauchy sum
``sum_i q_i / (lam_j - d_i)`` done directly for small nodes and by a 1-D FMM for large ones; the
final sum over ``j`` is another 1-D FMM from the Gauss nodes to the rings.  Every step costs
``O(n)`` or ``O(n log n)`` for an ``n``-term order, so a transform is ``O(L^2 log L)`` against the
march's ``O(L^2 N_ring) = O(L^3)``.  Analysis is the exact adjoint (same plan, transposed kernels).

The plan depends on the geometry only (``L`` and the ring colatitudes) and is built once on the
host (``_cpu/dc_plan.cpp``), cached on disk, and kept on the device.  Contract of the two public
calls is the folded march's (``_march_v2.forward/inverse_latitudinal_positive``), so the router
can swap engines per geometry.
"""
from __future__ import annotations

import ctypes
import hashlib
import math
import os
import subprocess
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CU = os.path.join(_HERE, "_cuda", "dc_lat.cu")
_CC = os.path.join(_HERE, "_cpu", "dc_plan.cpp")
_CACHE = os.environ.get("GMASTER_CUDA_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "gmaster"))
# Below this bandlimit the march is at least as fast: at L 3072 this route is 0.98x (synthesis) / 0.91x
# (analysis) of it, at 6144 1.77x / 1.66x (`.qwen/tmp/s38/dc_vs_march.py`, GPU0).
_MIN_L = int(os.environ.get("GMASTER_DC_MIN_L", "6144"))
# The plan builder runs on the host; keep it small on shared machines.
_THREADS = int(os.environ.get("GMASTER_DC_THREADS", "8"))
_DIRECT_MAX = int(os.environ.get("GMASTER_DC_DIRECT_MAX", "256"))
_DIRECT_THREADS = 128        # merge_direct's block size: a direct node carries <= this many deflations
_CD_SKIP = 1e-9              # CD rings where |E phi_n| < skip * max are in the forbidden region
_P = 12                      # FMM expansion order (dc_lat.cu)
_FL, _CDF = 32, 4            # merge FMM leaf size and CD leaf coarsening (dc_lat.cu defaults)
_FIELDS = ("leaf_off", "leaf_sz", "leaf_mk", "leaf_lam", "nodes", "sbase", "dh", "dl", "gap", "tau", "z", "c",
           "gidx", "slot", "dsrc", "ddst", "cd_desc", "cd_ln", "cd_lr", "cd_nh", "cd_nl",
           "vlast", "scale", "ring_h", "ring_l")
_DTYPES = dict(leaf_off=np.int32, leaf_sz=np.int32, leaf_mk=np.int32, nodes=np.int32, sbase=np.int32, cd_desc=np.int32,
               cd_ln=np.int32, cd_lr=np.int32, gidx=np.int16, slot=np.int16, dsrc=np.int16,
               ddst=np.int16, lev_kind=np.int32, lev_nnode=np.int32, lev_node0=np.int32,
               lev_maxk=np.int32, lev_sb0=np.int32)
_LIB = None
_LIB_ERROR = None


def _digest(*parts):
    return hashlib.sha1("".join(parts).encode()).hexdigest()[:12]


def _build():
    """Compile the CUDA apply and the host plan builder (cached by source hash)."""
    global _LIB, _LIB_ERROR
    if _LIB is not None or _LIB_ERROR is not None:
        return _LIB
    try:
        from ._march_v2 import _nvcc

        devs = [d for d in jax.devices() if d.platform == "gpu"]
        if not devs:
            raise RuntimeError("no GPU device")
        cc = str(getattr(devs[0], "compute_capability", "")).replace(".", "")
        os.makedirs(_CACHE, exist_ok=True)
        cu_src, cc_src = open(_CU).read(), open(_CC).read()
        defs = [f"-D{d}" for d in os.environ.get("GMASTER_DC_DEFS", "").split()]
        so = os.path.join(_CACHE, f"libgm_dc_{_digest(cu_src, cc, jax.__version__, ' '.join(defs))}.so")
        plan_so = os.path.join(_CACHE, f"libgm_dcplan_{_digest(cc_src)}.so")
        if not os.path.exists(so):
            nvcc = _nvcc()
            if nvcc is None:
                raise RuntimeError("nvcc not found (set GMASTER_NVCC)")
            tmp = so + f".{os.getpid()}.tmp"
            out = subprocess.run([nvcc, "-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC",
                                  f"-arch=sm_{cc}", "-diag-suppress", "940,2473", *defs,
                                  "-I", jax.ffi.include_dir(), "-o", tmp, _CU],
                                 capture_output=True, text=True, timeout=900)
            if out.returncode != 0:
                raise RuntimeError("nvcc failed:\n" + out.stderr[-4000:])
            os.replace(tmp, so)
        if not os.path.exists(plan_so):
            tmp = plan_so + f".{os.getpid()}.tmp"
            out = subprocess.run(["g++", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC",
                                  "-std=c++17", "-o", tmp, _CC],
                                 capture_output=True, text=True, timeout=300)
            if out.returncode != 0:
                raise RuntimeError("g++ dc plan failed:\n" + out.stderr[-4000:])
            os.replace(tmp, plan_so)
        lib = ctypes.CDLL(so)
        for name in ("gm_dc_synth", "gm_dc_synth2", "gm_dc_ana", "gm_dc_ana2"):
            jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(getattr(lib, name)), platform="CUDA")
        plan = ctypes.CDLL(plan_so)
        plan.dc_plan.restype = ctypes.c_int64
        plan.dc_plan.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
                                 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
        plan.dc_size.restype = ctypes.c_int64
        plan.dc_size.argtypes = [ctypes.c_char_p]
        plan.dc_copy.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
        _LIB = (lib, plan, _digest(cc_src))
    except Exception as exc:  # noqa: BLE001 - any failure means "route unavailable"
        _LIB_ERROR = exc
        if os.environ.get("GMASTER_DC_VERBOSE"):
            print(f"[gmaster] dc latitudinal route unavailable: {exc}")
    return _LIB


def enabled(L=None) -> bool:
    """True when this route serves spin-0 latitudinal transforms at bandlimit ``L``."""
    if os.environ.get("GMASTER_DC", "1") == "0":
        return False
    if L is not None and int(L) < _MIN_L:
        return False
    return _build() is not None


def _ring_x(L, nside):
    from .utils import _stable_thetas

    theta = np.asarray(_stable_thetas(L, nside), dtype=np.float64)
    return np.cos(theta[: 2 * nside])          # northern rings, pole to equator (equator last)


def _host_plan(L, nside):
    """The packed plan as numpy arrays, from the disk cache or built on the host."""
    lib, plan, tag = _build()
    x = _ring_x(L, nside)
    key = _digest(tag, hashlib.sha1(x.tobytes()).hexdigest(), str((_DIRECT_MAX, _DIRECT_THREADS, _CD_SKIP)))
    path = os.path.join(_CACHE, f"dcplan_L{L}_n{nside}_{key}.npz")
    if os.path.exists(path):
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    xs = np.ascontiguousarray(x)
    plan.dc_plan(int(L), xs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), xs.size,
                 _THREADS, _DIRECT_MAX, _DIRECT_THREADS, _CD_SKIP)
    out = {}
    for name in _FIELDS + ("lev_kind", "lev_nnode", "lev_node0", "lev_maxk", "lev_sb0"):
        n = plan.dc_size(name.encode())
        arr = np.empty(n, dtype=_DTYPES.get(name, np.float32))
        if n:
            plan.dc_copy(name.encode(), arr.ctypes.data)
        out[name] = arr
    out["nbox"] = np.array(plan.dc_size(b"nbox"))
    out["cd_nbox"] = np.array(plan.dc_size(b"cd_nbox"))
    plan.dc_release()
    tmp = path + f".{os.getpid()}.tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, path)
    return out


class _Plan:
    def __init__(self, L, nside):
        h = _host_plan(L, nside)
        self.L, self.nside = L, nside
        self.R = 2 * nside
        self.nprob = 2 * L - 1
        self.ntot = int(h["vlast"].size)
        self._layout(h)
        self.args = tuple(jnp.asarray(h[k]) for k in _FIELDS)
        # packing: problem t = 2 m + p holds ell = m + p + 2 k at offset off[t] + k
        n = np.array([len(range(m + p, L, 2)) for m in range(L) for p in (0, 1) if m + p < L])
        off = np.concatenate([[0], np.cumsum(n)])
        ell = np.empty(self.ntot, np.int32)
        order = np.empty(self.ntot, np.int32)
        t = 0
        for m in range(L):
            for p in (0, 1):
                if m + p < L:
                    ell[off[t]:off[t + 1]] = np.arange(m + p, L, 2)
                    order[off[t]:off[t + 1]] = m
                    t += 1
        self.ell, self.order = jnp.asarray(ell), jnp.asarray(order)
        # analysis writes back by *gather*: position ell * L + m -> packed index (ntot = zero slot)
        inv = np.full(L * L, self.ntot, np.int32)
        inv[ell.astype(np.int64) * L + order] = np.arange(self.ntot, dtype=np.int32)
        self.inv = jnp.asarray(inv)
        # alm_CS = (-1)^m psi / sqrt(2 pi) on both sides of the transform
        self.fac = jnp.asarray(((-1.0) ** order / math.sqrt(2.0 * math.pi)).astype(np.float32))

    def _layout(self, h):
        """Level-local scratch layout, recomputed from the plan's node table.

        The tree levels run one after another, so every per-level scratch (FMM boxes, staged
        strengths, deflations, root positions) is sized for the largest single level and reused;
        the CD step sizes its boxes for its own (coarsened) leaves.  ``h['sbase']`` and the CD
        box offsets in ``h['cd_desc']`` are rewritten here.
        """
        defs = dict(d.split("=") for d in os.environ.get("GMASTER_DC_DEFS", "").split() if "=" in d)
        fl, cdf = int(defs.get("FL", _FL)), int(defs.get("CDF", _CDF))
        nodes = h["nodes"].reshape(-1, 8)
        sbase = np.zeros_like(h["sbase"])
        lev, nbox, nkept, ndefl = [], 1, 1, 1
        for kind, nn, node0, maxk, sb0 in zip(h["lev_kind"], h["lev_nnode"], h["lev_node0"],
                                              h["lev_maxk"], h["lev_sb0"]):
            rec = nodes[node0:node0 + nn]
            if kind == 1:
                nb = np.array([_nbox(-(-int(k) // fl)) for k in rec[:, 1]], np.int64)
                sbase[sb0:sb0 + nn] = np.concatenate([[0], np.cumsum(nb)[:-1]])
                nbox = max(nbox, int(nb.sum()))
            nkept = max(nkept, int(rec[:, 1].sum()))
            ndefl = max(ndefl, int(rec[:, 2].sum()))
            lev.append((int(kind), int(nn), int(node0), int(maxk), int(sb0), int(rec[0, 3]), int(rec[0, 4])))
        h["sbase"] = sbase
        cd = h["cd_desc"].reshape(-1, 8).copy()
        nb = np.array([_nbox(-(-int(n) // cdf)) for n in cd[:, 3]], np.int64)
        cd[:, 4] = np.concatenate([[0], np.cumsum(nb)[:-1]])
        h["cd_desc"] = cd.ravel().astype(np.int32)
        self.levels = np.asarray(lev, np.int64).ravel()
        # scratch element counts per right-hand side: FMM boxes, staged strengths, deflations, kept roots
        self.sizes = (max(nbox, int(nb.sum())), nkept, ndefl, nkept)

    def nbytes(self):
        return sum(int(a.size) * a.dtype.itemsize for a in self.args)

    def static(self):
        """Hashable description for the jitted wrappers (the arrays go in as arguments)."""
        return (self.nprob, self.R, self.ntot, tuple(int(v) for v in self.levels), self.sizes)


def _nbox(nleaf):
    """Boxes of a binary 1-D FMM tree over ``nleaf`` leaves (the kernels' own level loop)."""
    total, size = 0, max(int(nleaf), 1)
    while True:
        total += size
        if size <= 3:
            return total
        size = (size + 1) // 2


def _scratch(static, nr):
    nb, nq, nd, nk = static[4]
    return (jax.ShapeDtypeStruct((2 * nb,), jnp.float32),
            jax.ShapeDtypeStruct((_P * nb * nr,), jnp.complex64),
            jax.ShapeDtypeStruct((_P * nb * nr,), jnp.complex64),
            jax.ShapeDtypeStruct((nq * nr,), jnp.complex64),
            jax.ShapeDtypeStruct((nd * nr,), jnp.complex64),
            jax.ShapeDtypeStruct((2 * nk,), jnp.float32))


def _synth(coef, args, static):
    """``(ntot, nr)`` packed coefficients -> ``(nprob, R, nr)`` parity-resolved northern values."""
    nprob, R, ntot, levels = static[:4]
    nr = coef.shape[1]
    return jax.ffi.ffi_call(
        "gm_dc_synth" if nr == 1 else "gm_dc_synth2",
        (jax.ShapeDtypeStruct((nprob, R, nr), jnp.complex64),
         jax.ShapeDtypeStruct((ntot, nr), jnp.complex64)) + _scratch(static, nr))(
        coef.astype(jnp.complex64), *args, levels=np.asarray(levels, np.int64))[0]


def _analyse(ring, args, static):
    """Adjoint of :func:`_synth`."""
    nprob, R, ntot, levels = static[:4]
    nr = ring.shape[2]
    return jax.ffi.ffi_call(
        "gm_dc_ana" if nr == 1 else "gm_dc_ana2",
        (jax.ShapeDtypeStruct((ntot, nr), jnp.complex64),) + _scratch(static, nr))(
        ring.astype(jnp.complex64), *args, levels=np.asarray(levels, np.int64))[0]


@lru_cache(maxsize=2)
def plan_for(L, nside):
    return _Plan(int(L), int(nside))


def _hemispheres(ring, L, nside):
    """(nprob, R) parity-resolved northern values -> (4 nside - 1, L) ring block (m >= 0)."""
    even = ring[0::2]                                                    # (L, R)
    odd = jnp.concatenate([ring[1::2], jnp.zeros((1, ring.shape[1]), ring.dtype)], axis=0)
    north = (even + odd).T                                               # (R, L), pole -> equator
    south = (even - odd).T[: 2 * nside - 1][::-1]                        # equator excluded
    return jnp.concatenate([north, south], axis=0)


def _ring_phase(phase, L):
    """``exp(i m phase_r)`` as complex64: the angle is reduced mod 2 pi in float64 (``m phase`` reaches
    ~4e4 rad, which float32 would carry to 1e-3), the sincos is float32.  The card's float64 rate is
    1/64 of float32, so a float64 ``exp`` over the ``(4 nside - 1, L)`` block was most of a pass."""
    ang = jnp.mod(jnp.arange(L, dtype=jnp.float64)[None, :] * phase[:, None], 2 * jnp.pi)
    ang = ang.astype(jnp.float32)
    return jax.lax.complex(jnp.cos(ang), jnp.sin(ang))


def _fold_hemispheres(g, L, nside):
    """(4 nside - 1, L) weighted ring block -> (nprob, R) parity-resolved northern sums."""
    R = 2 * nside
    north = g[:R]                                                        # pole -> equator
    south = jnp.concatenate([g[R:][::-1], jnp.zeros((1, L), g.dtype)], axis=0)   # mirror of north
    return jnp.stack([(north + south).T, (north - south).T], axis=1).reshape(2 * L, R)[: 2 * L - 1]


@partial(jax.jit, static_argnames=("L", "nside", "static"))
def _inverse_impl(positives, phase, args, ell, order, fac, *, L, nside, static):
    coef = jnp.stack([p[ell, order].astype(jnp.complex64) * fac for p in positives], axis=1)
    ring = _synth(coef, args, static)
    rp = _ring_phase(phase, L)
    return tuple(_hemispheres(ring[:, :, k], L, nside) * rp for k in range(len(positives)))


@partial(jax.jit, static_argnames=("L", "nside", "static"))
def _forward_impl(positives, weights, phase, args, inv, fac, *, L, nside, static):
    wp = weights[:, None].astype(jnp.float32) * _ring_phase(phase, L)
    ring = jnp.stack([_fold_hemispheres(p.astype(jnp.complex64) * wp, L, nside) for p in positives],
                     axis=2)
    coef = _analyse(ring, args, static) * fac[:, None]
    coef = jnp.concatenate([coef, jnp.zeros((1, coef.shape[1]), coef.dtype)], axis=0)
    return tuple(coef[inv, k].reshape(L, L).astype(jnp.complex128) for k in range(len(positives)))


def inverse_latitudinal_positive(positive, phase, *, L, nside):
    """``(L, L)`` ``[ell, m]`` coefficients -> ``(4 nside - 1, L)`` complex64 ring block, phase included."""
    return inverse_latitudinal_positive_pair(positive, None, phase, L=L, nside=nside)[0]


def forward_latitudinal_positive(positive, weights, phase, *, L, nside):
    """``(4 nside - 1, L)`` ring block -> ``(L, L)`` ``[ell, m]`` coefficients."""
    return forward_latitudinal_positive_pair(positive, None, weights, phase, L=L, nside=nside)[0]


def inverse_latitudinal_positive_pair(positive_a, positive_b, phase, *, L, nside):
    """Two syntheses over one traversal of the plan (``positive_b=None``: one)."""
    pl = plan_for(L, nside)
    maps = (positive_a,) if positive_b is None else (positive_a, positive_b)
    return _inverse_impl(maps, phase, pl.args, pl.ell, pl.order, pl.fac,
                         L=int(L), nside=int(nside), static=pl.static())


def forward_latitudinal_positive_pair(positive_a, positive_b, weights, phase, *, L, nside):
    """Two analyses over one traversal of the plan (``positive_b=None``: one)."""
    pl = plan_for(L, nside)
    maps = (positive_a,) if positive_b is None else (positive_a, positive_b)
    return _forward_impl(maps, weights, phase, pl.args, pl.inv, pl.fac,
                         L=int(L), nside=int(nside), static=pl.static())


# ------------------------------------------------------------------------ packed-state interface
# The MASTER refinement loop can keep its alm state in this engine's packed layout (problem
# t = 2m + p, ell = m + p + 2k) for all of its iterations and convert once at the end: going
# through the (L, L) complex128 block on every pass was ~45 ms of XLA gathers, pads and casts per
# field at Nside 2048.  Packed values are alm (Condon-Shortley), i.e. already scaled by `fac`.

@partial(jax.jit, static_argnames=("L", "nside", "static"))
def _forward_packed_impl(positives, weights, phase, args, fac, *, L, nside, static):
    wp = weights[:, None].astype(jnp.float32) * _ring_phase(phase, L)
    ring = jnp.stack([_fold_hemispheres(p.astype(jnp.complex64) * wp, L, nside) for p in positives],
                     axis=2)
    return _analyse(ring, args, static) * fac[:, None]


@partial(jax.jit, static_argnames=("L", "nside", "static"))
def _inverse_packed_impl(coef, phase, args, fac, *, L, nside, static):
    ring = _synth(coef.astype(jnp.complex64) * fac[:, None], args, static)
    rp = _ring_phase(phase, L)
    return tuple(_hemispheres(ring[:, :, k], L, nside) * rp for k in range(coef.shape[1]))


def forward_packed(ftms, weights, phase, *, L, nside):
    """Ring blocks (one per map) -> ``(ntot, nmaps)`` complex64 packed alm."""
    pl = plan_for(L, nside)
    return _forward_packed_impl(tuple(ftms), weights, phase, pl.args, pl.fac,
                                L=int(L), nside=int(nside), static=pl.static())


def inverse_packed(coef, phase, *, L, nside):
    """``(ntot, nmaps)`` packed alm -> tuple of complex64 ring blocks, phase included."""
    pl = plan_for(L, nside)
    return _inverse_packed_impl(coef, phase, pl.args, pl.fac,
                                L=int(L), nside=int(nside), static=pl.static())


@partial(jax.jit, static_argnames=("L",))
def _packed_to_alm_impl(coef, inv, ell, order, *, L):
    coef = jnp.concatenate([coef, jnp.zeros(1, coef.dtype)])
    return coef[inv[ell * L + order]]


def packed_to_alm(coef, ell, order, *, L, nside):
    """One map's packed alm -> the pipeline's ``(ell, order)`` packing."""
    pl = plan_for(L, nside)
    return _packed_to_alm_impl(coef, pl.inv, ell, order, L=int(L))
