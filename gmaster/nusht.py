"""GPU general (non-uniform) spherical harmonic transform.

The pipeline is the one ``cunuSHT`` and ``ducc0``'s ``synthesis_general`` use, with GMaster's
float32 CUDA march in the latitudinal slot:

    a_lm --(v2 march on a Fejer-1 ring grid)--> F(theta, m)
         --(doubling in theta)--> F on the full circle
         --(FFT in theta)--> c_{k,m}, a 2-D Fourier series on the torus
         --(cufinufft type 2)--> f(theta_n, phi_n) at arbitrary points.

The Fejer-1 grid ``theta_j = (2j+1) pi / (2 ntheta)`` is equidistant, symmetric about pi/2 and
excludes the poles, so (i) the march's north/south folding is exactly the ring reversal and
(ii) the doubled ring stack ``F(2pi - theta, m) = (-1)^(m+s) F(theta, m)`` is again equidistant,
which is what makes the theta direction a plain FFT.  Both stages are verified in
``tests/test_nusht.py`` against ``ducc0.sht.experimental``.

``adjoint_synthesis_general`` runs the pipeline backwards -- type-1 NUFFT, inverse latitude FFT,
undoubling, and the march's analysis kernel, which is the transpose of its synthesis kernel
(checked by a bilinear-form test in the test module; the mirror channel needs an explicit
``(-1)^(ell + s)``).  It reproduces ducc0's ``adjoint_synthesis_general``, which is the
*analysis-shaped* adjoint ``abar_lm = sum_p map_p conj(Y_lm(p))`` and so is not quite the
transpose of ducc0's own synthesis: the m < 0 half of the hermitian alm is not folded back onto
m > 0, which would double it.  See the comment in ``_torus_to_alm``.
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys
from functools import lru_cache, partial

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

from . import _march_v2 as v2

__all__ = [
    "synthesis_general",
    "adjoint_synthesis_general",
    "grid_for",
    "default_mstart",
]

# --------------------------------------------------------------------------- cufinufft loading
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NU_PATH = os.environ.get(
    "GMASTER_CUFINUFFT_PATH", os.path.join(_REPO, ".qwen", "tmp", "nu_pkgs")
)


@lru_cache(maxsize=1)
def _cufinufft():
    """Import ``cufinufft``, preloading the CUDA runtime libraries it links against.

    The wheel here is built against the pip CUDA 12 runtime; those ``.so``s live inside the
    ``nvidia`` wheels and are not on the loader path, so they are dlopen'd RTLD_GLOBAL first.
    """
    try:
        import nvidia

        root = os.path.dirname(nvidia.__file__)
        for pat in ("cuda_runtime", "cufft", "cublas", "cuda_nvrtc"):
            for cand in sorted(glob.glob(os.path.join(root, f"*{pat}*", "lib", "*.so.*"))):
                try:
                    ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
    except ImportError:  # pragma: no cover - only when the nvidia wheels are absent
        pass
    if _NU_PATH and _NU_PATH not in sys.path:
        sys.path.append(_NU_PATH)
    import cufinufft

    return cufinufft


@lru_cache(maxsize=1)
def _cupy():
    import cupy

    return cupy


def _to_cupy(x):
    """Zero-copy jax device array -> cupy array."""
    return _cupy().from_dlpack(x)


def _to_jax(x):
    return jax.dlpack.from_dlpack(x)


# ------------------------------------------------------------------------------------ geometry
def _good_size(n):
    """Smallest even 2-3-5-smooth integer >= n (FFT- and NUFFT-friendly)."""
    n = int(n)
    best = 2 ** int(np.ceil(np.log2(max(n, 2))))
    f2 = 2
    while f2 < 2 * n:
        f3 = f2
        while f3 < 2 * n:
            f5 = f3
            while f5 < 2 * n:
                if n <= f5 < best and f5 % 2 == 0:
                    best = f5
                f5 *= 5
            f3 *= 3
        f2 *= 2
    return int(best)


def grid_for(lmax, ntheta=None, nphi=None):
    """Ring grid for a band limit.

    ``ntheta >= lmax + 1``: ``F(theta, m)`` is a trigonometric polynomial of degree ``lmax`` in
    theta, so the doubled ring stack carries the ``2 lmax + 1`` modes ``|k| <= lmax`` and
    ``2 ntheta >= 2 lmax + 2`` equispaced samples resolve them with one slot to spare.  Taking
    the bound at equality is what makes the sizes powers of two at ``lmax = 2^k - 1`` (4096, not
    4320, at lmax 4095), which is worth ~20% of the theta FFT and ~10% of the NUFFT; the extra
    ring costs accuracy nothing (6.29e-06 against 6.15e-06 at lmax 4095, both float32 march
    noise -- ``.qwen/tmp/nu_grid_s37.py``).
    ``nphi >= 2 lmax + 1``: the phi direction carries orders ``-lmax .. lmax``.  Both are rounded
    up to even 2-3-5-smooth sizes because the theta FFT and the NUFFT fine grid are sized on them.
    """
    ntheta = _good_size(lmax + 1) if ntheta is None else int(ntheta)
    nphi = _good_size(2 * lmax + 1) if nphi is None else int(nphi)
    if ntheta < lmax + 1 or nphi < 2 * lmax + 1:
        raise ValueError(f"grid too small: ntheta={ntheta}, nphi={nphi}, lmax={lmax}")
    return ntheta, nphi


def _thetas(ntheta):
    return (2 * np.arange(ntheta) + 1) * np.pi / (2 * ntheta)


_GEO_KEYS = ("lane_ring", "valid", "lane_of_ring", "xs_hi", "xs_lo", "mlim", "hemi", "lsh", "lch")


@lru_cache(maxsize=16)
def _geometry(ntheta, L, spin):
    """The lane layout ``_march_v2`` builds for HEALPix, for the Fejer-1 grid.

    Returns ``(arrays, npad)``; ``arrays`` is a tuple in ``_GEO_KEYS`` order so it can cross a
    ``jit`` boundary as a pytree of device arrays.
    """
    theta = _thetas(ntheta)
    nt = theta.shape[0]
    x = np.cos(theta)
    if spin == 0:
        blocks = [np.arange((nt + 1) // 2)]  # northern half; partner is nt-1-j
    else:
        blocks = [np.nonzero(x >= 0)[0], np.nonzero(x < 0)[0]]
    lanes, hemi = [], []
    for b, blk in enumerate(blocks):
        n = blk.shape[0]
        ntile = -(-n // v2.TILE_ANA)
        lanes.append(np.concatenate([blk, np.full(ntile * v2.TILE_ANA - n, -1, blk.dtype)]))
        hemi += [b] * (ntile * v2.TILE_ANA // v2.HEMI_BLOCK)
    lane_ring = np.concatenate(lanes)
    valid = lane_ring >= 0
    ring = np.where(valid, lane_ring, 0)
    lane_of_ring = np.zeros(nt, np.int64)
    lane_of_ring[lane_ring[valid]] = np.nonzero(valid)[0]
    th = theta[ring]
    xv = np.cos(th)
    xs_hi, xs_lo = v2._split64(np.where(valid, np.abs(xv) - 1.0, 0.0))
    sth = np.sin(th)
    t1 = (L - 1) * sth + max(100.0, 0.01 * (L - 1))
    ml = spin * np.abs(xv) + np.sqrt(np.maximum(t1 * t1 - (spin * sth) ** 2, 0.0))
    arrays = (
        jnp.asarray(lane_ring),
        jnp.asarray(valid),
        jnp.asarray(lane_of_ring),
        jnp.asarray(xs_hi),
        jnp.asarray(xs_lo),
        jnp.asarray(np.where(valid, ml, -1.0).astype(np.float32)),
        jnp.asarray(np.asarray(hemi, np.int32)),
        jnp.asarray(np.log2(np.sin(th / 2.0))),
        jnp.asarray(np.log2(np.cos(th / 2.0))),
    )
    return arrays, int(lane_ring.shape[0])


def _geo(garr, npad, ntheta):
    g = dict(zip(_GEO_KEYS, garr))
    g["npad"] = npad
    g["ntheta"] = ntheta
    return g


@lru_cache(maxsize=8)
def _tables(ntheta, L, spin):
    """Every window's ``(man, ex0, tab)`` for this geometry, as resident device arrays.

    Same reason as ``_march_v2._build_tables``: built inside the transform's trace they are a few
    hundred ms of ``gammaln`` and float64 cumsum *per call* on this 1/64-rate card, and under
    ``ensure_compile_time_eval`` inside a trace they would become HLO constants.  They cross as
    jit arguments instead (0.95 GiB at lmax 4095 spin 2).
    """
    garr, npad = _geometry(ntheta, L, spin)
    with jax.ensure_compile_time_eval():
        geo = _geo(garr, npad, ntheta)
        out = []
        for (m0, _mb, mbp) in v2._windows(L):
            t = v2._window_tables(jnp.int32(m0), geo, mbp=mbp, L=L, spin=spin)
            out.append(tuple(jax.block_until_ready(x) for x in t))
    return tuple(out)


def drop_tables():
    _tables.cache_clear()
    _geometry.cache_clear()


# --------------------------------------------------------------------------------------- march
def _scale(L, spin):
    """``sqrt((2l+1)/4pi) (-1)^m``: the closed form the march kernel expects (see
    ``_march_v2._fold_synthesise`` / ``_inverse_impl``), times the ``(-1)^s`` of the s2fft
    spin-harmonic convention (``utils._finish_inverse_s2fft``)."""
    norm = jnp.sqrt((2.0 * jnp.arange(L, dtype=jnp.float64) + 1.0) / (4.0 * jnp.pi))
    rowsign = 1.0 - 2.0 * (jnp.arange(L) % 2).astype(jnp.float64)
    return ((-1.0) ** abs(spin)) * norm[:, None] * rowsign[None, :]


@partial(jax.jit, static_argnames=("L", "npad", "ntheta"))
def _march_syn_s0(alm, garr, tabs, *, L, npad, ntheta):
    """``(L, L)`` ``alm[ell, m]`` -> ``(L, ntheta)`` ``F[m, theta]`` for ``m >= 0``, spin 0."""
    geo = _geo(garr, npad, ntheta)
    Lp = L + v2.UNR
    north = (ntheta + 1) // 2
    nsouth = ntheta - north
    a = jnp.asarray(alm) * _scale(L, 0)
    rows_n, rows_s = [], []
    for w, (m0, mb, mbp) in enumerate(v2._windows(L)):
        man, ex0, tab = tabs[w]
        aa = a[:, m0:m0 + mb].T
        coef = jnp.zeros((mbp, Lp, 4), jnp.float32)
        coef = coef.at[:mb, :L, 0].set(aa.real.astype(jnp.float32))
        coef = coef.at[:mb, :L, 1].set(aa.imag.astype(jnp.float32))
        v = v2._ffi_synthesis(0, geo, m0, mbp, L, man, ex0, tab, coef)[:mb]
        rows_n.append(v[:, :north, 0] + 1j * v[:, :north, 1])
        rows_s.append(v[:, :nsouth, 2] + 1j * v[:, :nsouth, 3])
    fn = jnp.concatenate(rows_n, axis=0)                       # (L, north)   rings 0..north-1
    fs = jnp.concatenate(rows_s, axis=0)                       # (L, nsouth) rings ntheta-1 down
    return jnp.concatenate([fn, jnp.flip(fs, axis=1)], axis=1)


@partial(jax.jit, static_argnames=("L", "npad", "ntheta", "spin"))
def _march_syn_spin(almp, almm, garr, tabs, *, L, npad, ntheta, spin):
    """``(L, L)`` coefficient blocks for orders ``+m`` and ``-m`` -> ``F[+m, theta]``,
    ``F[-m, theta]``, both ``(L, ntheta)``."""
    geo = _geo(garr, npad, ntheta)
    Lp = L + v2.UNR
    sc = _scale(L, spin)
    ap_all = jnp.asarray(almp) * sc
    am_all = jnp.asarray(almm) * sc
    lane_of_ring = geo["lane_of_ring"]
    lane_of_mirror = lane_of_ring[::-1]
    pos, neg = [], []
    for w, (m0, mb, mbp) in enumerate(v2._windows(L)):
        man, ex0, tab = tabs[w]
        ap = ap_all[:, m0:m0 + mb].T
        am = am_all[:, m0:m0 + mb].T
        coef = jnp.zeros((mbp, Lp, 4), jnp.float32)
        coef = coef.at[:mb, :L, 0].set(ap.real.astype(jnp.float32))
        coef = coef.at[:mb, :L, 1].set(ap.imag.astype(jnp.float32))
        coef = coef.at[:mb, :L, 2].set(am.real.astype(jnp.float32))
        coef = coef.at[:mb, :L, 3].set(am.imag.astype(jnp.float32))
        v = v2._ffi_synthesis(spin, geo, m0, mbp, L, man, ex0, tab, coef)[:mb]
        fp = v[..., 0] + 1j * v[..., 1]                       # order +m at the lane's own ring
        fm = v[..., 2] + 1j * v[..., 3]                       # order -m at the mirror ring
        pos.append(fp[:, lane_of_ring])
        neg.append(fm[:, lane_of_mirror])
    return jnp.concatenate(pos, axis=0), jnp.concatenate(neg, axis=0)


@partial(jax.jit, static_argnames=("L", "npad", "ntheta"))
def _march_ana_s0(G, garr, tabs, *, L, npad, ntheta):
    """Transpose of :func:`_march_syn_s0`: ``(L, ntheta)`` -> ``(L, L)``."""
    geo = _geo(garr, npad, ntheta)
    north = (ntheta + 1) // 2
    jn = jnp.arange(north)
    partner = ntheta - 1 - jn
    G = jnp.asarray(G)
    g_n = G[:, jn]
    g_p = jnp.where(partner == jn, 0.0, G[:, partner])
    valid = geo["valid"]
    ring = jnp.where(valid, geo["lane_ring"], 0)
    plus = (g_n + g_p)[:, ring]
    minus = (g_n - g_p)[:, ring]
    chan = jnp.stack([plus.real, plus.imag, minus.real, minus.imag], axis=-1)
    rhs_all = lax.optimization_barrier(
        lax.convert_element_type(jnp.where(valid[None, :, None], chan, 0.0), jnp.float32))
    cols = []
    for w, (m0, mb, mbp) in enumerate(v2._windows(L)):
        man, ex0, tab = tabs[w]
        r = rhs_all[m0:m0 + mb]
        if mbp != mb:
            r = jnp.concatenate([r, jnp.zeros((mbp - mb,) + r.shape[1:], r.dtype)], axis=0)
        parts = v2._ffi_analysis(0, geo, m0, mbp, L, man, ex0, tab, r)[:mb]
        block = (parts[..., 0] + 1j * parts[..., 1]).astype(jnp.complex128)
        cols.append(jnp.concatenate([jnp.zeros((m0, mb), jnp.complex128), block.T], axis=0))
    return jnp.concatenate(cols, axis=1) * _scale(L, 0)


@partial(jax.jit, static_argnames=("L", "npad", "ntheta", "spin"))
def _march_ana_spin(Gp, Gm, garr, tabs, *, L, npad, ntheta, spin):
    """Transpose of :func:`_march_syn_spin`."""
    geo = _geo(garr, npad, ntheta)
    valid = geo["valid"]
    ring = jnp.where(valid, geo["lane_ring"], 0)
    mring = jnp.where(valid, ntheta - 1 - geo["lane_ring"], 0)
    direct = jnp.asarray(Gp)[:, ring]
    mirror = jnp.asarray(Gm)[:, mring]
    chan = jnp.stack([direct.real, direct.imag, mirror.real, mirror.imag], axis=-1)
    rhs_all = lax.optimization_barrier(
        lax.convert_element_type(jnp.where(valid[None, :, None], chan, 0.0), jnp.float32))
    cp, cm = [], []
    for w, (m0, mb, mbp) in enumerate(v2._windows(L)):
        man, ex0, tab = tabs[w]
        r = rhs_all[m0:m0 + mb]
        if mbp != mb:
            r = jnp.concatenate([r, jnp.zeros((mbp - mb,) + r.shape[1:], r.dtype)], axis=0)
        parts = v2._ffi_analysis(spin, geo, m0, mbp, L, man, ex0, tab, r)[:mb]
        if m0:
            parts = jnp.concatenate([jnp.zeros((mb, m0, 4), parts.dtype), parts], axis=1)
        p = (parts[..., 0] + 1j * parts[..., 1]).astype(jnp.complex128).T
        q = (parts[..., 2] + 1j * parts[..., 3]).astype(jnp.complex128).T
        cp.append(p)
        cm.append(q)
    sc = _scale(L, spin)
    # The kernel's mirror accumulator is the transpose of its mirror synthesis channel only up to
    # `(-1)^(ell + s)`: the synthesis writes `f_{-m}(pi - theta)` and the analysis reads the same
    # ring back, and `d^l_{m,-s}(pi - theta) = (-1)^(l + s) d^l_{-m,-s}(theta)` sits between them
    # (the same factor `_march_v2._forward_impl` applies).  Measured, not assumed -- see
    # `test_march_is_transpose`.
    esign = 1.0 - 2.0 * ((jnp.arange(L) + abs(spin)) % 2).astype(jnp.float64)
    return jnp.concatenate(cp, axis=1) * sc, jnp.concatenate(cm, axis=1) * sc * esign[:, None]


# --------------------------------------------------------------- doubling + latitude FFT
def _mode_signs(nphi, spin):
    m = np.fft.fftfreq(nphi, d=1.0 / nphi).astype(int)
    return jnp.asarray((-1.0) ** (np.abs(m) + abs(spin)))


def _theta_phase(ntheta, conj=False):
    k = np.fft.fftfreq(2 * ntheta, d=1.0 / (2 * ntheta)).astype(int)
    h = np.pi / ntheta
    s = 1.0 if conj else -1.0
    return jnp.asarray(np.exp(s * 1j * k * h / 2.0))


@partial(jax.jit, static_argnames=("ntheta", "nphi", "spin", "cdtype", "wdtype"))
def _double_and_fft(F, *, ntheta, nphi, spin, cdtype, wdtype):
    """``(nphi, ntheta)`` ring spectrum -> ``(2 ntheta, nphi)`` torus Fourier coefficients.

    ``wdtype`` is the working precision of the doubled stack, which is the largest array in the
    transform (1.07 GiB at lmax 4095 in complex128).  complex64 there costs 3e-07 relative --
    a decade under the march's own float32 noise -- and is 2.7x faster (2.7 ms against 7.1).
    """
    sgn = _mode_signs(nphi, spin)[:, None].astype(wdtype)
    F = F.astype(wdtype)
    Fd = jnp.concatenate([F, sgn * jnp.flip(F, axis=1)], axis=1)          # (nphi, 2 ntheta)
    C = jnp.fft.fft(Fd, axis=1) / jnp.asarray(2.0 * ntheta, wdtype)
    C = C * _theta_phase(ntheta)[None, :].astype(wdtype)
    return C.T.astype(cdtype)


@partial(jax.jit, static_argnames=("ntheta", "nphi", "spin", "wdtype"))
def _undouble_and_ifft(C, *, ntheta, nphi, spin, wdtype):
    """Transpose of :func:`_double_and_fft`; the result carries ``wdtype``."""
    C = C.astype(wdtype).T                                                # (nphi, 2 ntheta)
    C = C * _theta_phase(ntheta, conj=True)[None, :].astype(wdtype)
    Fd = jnp.fft.ifft(C, axis=1)
    sgn = _mode_signs(nphi, spin)[:, None].astype(wdtype)
    return Fd[:, :ntheta] + sgn * jnp.flip(Fd[:, ntheta:], axis=1)


# ---------------------------------------------------------------------------- alm packing
def default_mstart(lmax, mmax=None):
    mmax = lmax if mmax is None else mmax
    m = np.arange(mmax + 1)
    return (m * (2 * lmax + 1 - m) // 2).astype(np.uint64)


@lru_cache(maxsize=8)
def _alm_index(lmax, mmax, mstart_key, lstride, spin, nalm=0):
    """Index pair for the two directions of the ``mstart`` packing.

    ``idx``/``ok`` are the ``(L, L)`` gather into the packed alm (unpacking).  ``back``/``bok``
    are its *inverse* gather, ``(nalm,)`` into the flattened ``(L, L)`` block, used for packing.
    Packing was a ``.at[idx].add`` scatter, which is the wrong shape entirely: the ``ell < m``
    half of the block is masked to index 0, so half of ``L**2`` complex128 atomics land on one
    address -- 26.8 ms of the 45 ms adjoint tail at lmax 4095, against 0.9 ms for this gather.
    """
    L = lmax + 1
    mstart = np.asarray(mstart_key, np.int64)
    ell = np.arange(L)[:, None]
    m = np.arange(L)[None, :]
    ok = (m <= np.minimum(ell, mmax)) & (ell >= abs(spin))
    idx = np.where(ok, mstart[np.minimum(m, mmax)] + ell * lstride, 0)
    nalm = int(nalm) or int(idx[ok].max() + 1)
    back = np.zeros(nalm, np.int64)
    bok = np.zeros(nalm, bool)
    back[idx[ok]] = (ell * L + m)[ok]
    bok[idx[ok]] = True
    return jnp.asarray(idx), jnp.asarray(ok), jnp.asarray(back), jnp.asarray(bok)


@partial(jax.jit, static_argnames=("L",))
def _unpack(alm, idx, ok, *, L):
    return jnp.where(ok, jnp.take(alm, idx, axis=0), 0.0 + 0.0j)


@jax.jit
def _pack(a2d, back, bok):
    return jnp.where(bok, jnp.take(a2d.ravel(), back, axis=0), 0.0 + 0.0j)


# ------------------------------------------------------------------------------ NUFFT layer
class _NufftPlan:
    """A cufinufft type-1 or type-2 2-D plan bound to a point set."""

    def __init__(self, ntype, ntheta, nphi, loc, epsilon, upsampfac, dtype):
        cf = _cufinufft()
        cp = _cupy()
        self.dtype = np.dtype(dtype)
        rdtype = np.float32 if self.dtype == np.complex64 else np.float64
        isign = 1 if ntype == 2 else -1
        kw = dict(eps=float(epsilon), isign=isign, dtype=str(self.dtype), modeord=1,
                  gpu_device_id=0)
        if upsampfac is not None:
            kw["upsampfac"] = float(upsampfac)
        self.plan = cf.Plan(ntype, (2 * ntheta, nphi), **kw)
        loc = np.ascontiguousarray(loc)
        self.npoint = loc.shape[0]
        self.plan.setpts(cp.asarray(loc[:, 0].astype(rdtype)),
                         cp.asarray(loc[:, 1].astype(rdtype)))

    def __call__(self, data):
        # cufinufft runs on the CUDA default stream and jax on its own, and dlpack does not
        # insert the cross-stream event, so each hand-off is bracketed by an explicit sync.
        cp = _cupy()
        data = jnp.asarray(data, self.dtype)
        data.block_until_ready()
        out = self.plan.execute(_to_cupy(data))
        cp.cuda.get_current_stream().synchronize()
        return _to_jax(cp.ascontiguousarray(out))


# --------------------------------------------------------------------------------- drivers
def _prep(alm, spin, lmax, mmax, mstart, lstride):
    alm = jnp.asarray(alm)
    if alm.ndim == 1:
        alm = alm[None, :]
    ncomp = 1 if spin == 0 else 2
    if alm.shape[0] != ncomp:
        raise ValueError(f"spin {spin} needs {ncomp} alm component(s), got {alm.shape[0]}")
    mmax = lmax if mmax is None else int(mmax)
    if mstart is None:
        mstart = default_mstart(lmax, mmax)
    key = tuple(int(v) for v in np.asarray(mstart).ravel())
    nalm = int(max(key) + lmax * int(lstride) + 1)
    idx, ok, back, bok = _alm_index(lmax, mmax, key, int(lstride), abs(spin), nalm)
    return alm, idx, ok, back, bok, mmax


def synthesis_general(alm, *, spin, lmax, loc, epsilon=1e-6, mmax=None, mstart=None,
                      lstride=1, nthreads=None, upsampfac=1.25, single=False, fft_single=True,
                      ntheta=None, nphi=None, plan=None):
    """Spin-weighted synthesis at arbitrary points on the sphere, on the GPU.

    Signature and conventions follow ``ducc0.sht.experimental.synthesis_general``: ``alm`` is
    ``(ncomp, nalm)`` in the ``mstart`` packing (``ncomp`` 1 for spin 0, 2 for spin != 0),
    ``loc`` is ``(npoint, 2)`` with ``loc[:, 0] = theta``, ``loc[:, 1] = phi``, and the result is
    ``(ncomp, npoint)`` real.  ``nthreads`` is accepted and ignored (GPU).

    ``upsampfac`` is the cufinufft upsampling factor (1.25 is both faster and, measured, as
    accurate as 2.0 here).  ``epsilon`` 1e-6 is the knee: 1e-7 and 1e-8 buy no accuracy (the
    march's float32 error is the binding constraint) and cost 20-30% of the NUFFT, while 1e-5
    saves only ~6% -- the cost at upsampfac 1.25 is the spread/interp over the points, not the
    kernel width -- and is already visible in eps_eff at lmax 1023 (6.2e-06 against 4.6e-06).
    ``single=True`` runs the NUFFT itself in complex64, which is *not* recommended: it floors
    eps_eff at ~1.5e-04.  ``fft_single`` sets the latitude FFT's working precision; complex64 is
    the default because it is 2.5x faster there and measurably free (6.400e-06 against 6.392e-06
    at lmax 4095).
    """
    L = int(lmax) + 1
    spin = int(spin)
    alm, idx, ok, _back, _bok, mmax = _prep(alm, spin, lmax, mmax, mstart, lstride)
    ntheta, nphi = grid_for(lmax, ntheta, nphi)
    garr, npad = _geometry(ntheta, L, spin)
    cdtype = jnp.complex64 if single else jnp.complex128

    C = _ftm_to_torus(alm, idx, ok, garr, _tables(ntheta, L, spin), npad=npad, L=L, spin=spin,
                      ntheta=ntheta, nphi=nphi, cdtype=cdtype,
                      wdtype=jnp.complex64 if fft_single else jnp.complex128)
    if plan is None:
        plan = _NufftPlan(2, ntheta, nphi, np.asarray(loc, np.float64), epsilon, upsampfac,
                          np.complex64 if single else np.complex128)
    vals = plan(C)
    if spin == 0:
        return jnp.real(vals)[None, :].astype(jnp.float64)
    return jnp.stack([jnp.real(vals), jnp.imag(vals)], axis=0).astype(jnp.float64)


@partial(jax.jit, static_argnames=("L", "spin", "ntheta", "nphi", "npad", "cdtype",
                                   "wdtype"))
def _ftm_to_torus(alm, idx, ok, garr, tabs, *, npad, L, spin, ntheta, nphi, cdtype,
                  wdtype):
    if spin == 0:
        a = _unpack(alm[0], idx, ok, L=L)
        Fp = _march_syn_s0(a, garr, tabs, L=L, npad=npad, ntheta=ntheta)
        F = jnp.zeros((nphi, ntheta), jnp.complex128).at[:L].set(Fp)
        F = F.at[nphi - L + 1:].set(jnp.conj(jnp.flip(Fp[1:L], axis=0)))
    else:
        a1 = _unpack(alm[0], idx, ok, L=L)
        a2 = _unpack(alm[1], idx, ok, L=L)
        rs = (1.0 - 2.0 * (jnp.arange(L) % 2))[None, :]
        ap = -(a1 + 1j * a2)                                  # order +m
        am = -rs * jnp.conj(a1 - 1j * a2)                     # order -m (reality of the two maps)
        Fp, Fm = _march_syn_spin(ap, am, garr, tabs, L=L, npad=npad, ntheta=ntheta, spin=spin)
        F = jnp.zeros((nphi, ntheta), jnp.complex128).at[:L].set(Fp)
        F = F.at[nphi - L + 1:].set(jnp.flip(Fm[1:L], axis=0))
    return _double_and_fft(F, ntheta=ntheta, nphi=nphi, spin=spin, cdtype=cdtype,
                           wdtype=wdtype)


def adjoint_synthesis_general(map, *, spin, lmax, loc, epsilon=1e-6, mmax=None,
                              mstart=None, lstride=1, nthreads=None, upsampfac=1.25,
                              single=False, fft_single=True, ntheta=None, nphi=None, plan=None):
    """Transpose of :func:`synthesis_general`; matches
    ``ducc0.sht.experimental.adjoint_synthesis_general``."""
    L = int(lmax) + 1
    spin = int(spin)
    mp = jnp.asarray(map)
    if mp.ndim == 1:
        mp = mp[None, :]
    ncomp = 1 if spin == 0 else 2
    if mp.shape[0] != ncomp:
        raise ValueError(f"spin {spin} needs {ncomp} map component(s), got {mp.shape[0]}")
    mmax = lmax if mmax is None else int(mmax)
    if mstart is None:
        mstart = default_mstart(lmax, mmax)
    key = tuple(int(v) for v in np.asarray(mstart).ravel())
    nalm = int(max(key) + lmax * int(lstride) + 1)
    _idx, _ok, back, bok = _alm_index(lmax, mmax, key, int(lstride), abs(spin), nalm)
    ntheta, nphi = grid_for(lmax, ntheta, nphi)
    garr, npad = _geometry(ntheta, L, spin)
    cdtype = jnp.complex64 if single else jnp.complex128

    vals = mp[0] if spin == 0 else mp[0] + 1j * mp[1]
    if plan is None:
        plan = _NufftPlan(1, ntheta, nphi, np.asarray(loc, np.float64), epsilon, upsampfac,
                          np.complex64 if single else np.complex128)
    C = plan(vals)
    return _torus_to_alm(C, back, bok, garr, _tables(ntheta, L, spin), npad=npad, L=L,
                         spin=spin, ntheta=ntheta, nphi=nphi,
                         wdtype=jnp.complex64 if fft_single else jnp.complex128)


@partial(jax.jit, static_argnames=("L", "spin", "ntheta", "nphi", "npad", "wdtype"))
def _torus_to_alm(C, back, bok, garr, tabs, *, npad, L, spin, ntheta, nphi, wdtype):
    F = _undouble_and_ifft(C, ntheta=ntheta, nphi=nphi, spin=spin, wdtype=wdtype)
    rs = (1.0 - 2.0 * (jnp.arange(L) % 2))[None, :]
    # ducc0's adjoint is the *analysis-shaped* adjoint -- `abar_lm = sum_p map_p conj(Y_lm)`, with
    # the m < 0 half of the (hermitian) alm not counted a second time.  So only the m >= 0 rows of
    # the ring spectrum feed the march for spin 0; for spin != 0 both helicities are needed,
    # because `conj((-s)Y_lm) = (-1)^(s+m) (+s)Y_{l,-m}` ties the second component to the -m rows.
    if spin == 0:
        a = _march_ana_s0(F[:L], garr, tabs, L=L, npad=npad, ntheta=ntheta)
        return _pack(a, back, bok)[None, :]
    Gp = F[:L]
    Gm = jnp.zeros((L, ntheta), F.dtype).at[1:L].set(jnp.flip(F[nphi - L + 1:], axis=0))
    u, v = _march_ana_spin(Gp, Gm, garr, tabs, L=L, npad=npad, ntheta=ntheta, spin=spin)
    v = v.at[:, 0].set(u[:, 0])              # order 0 is its own mirror
    q = rs * jnp.conj(v)
    a1 = -0.5 * (u + q)
    a2 = 0.5j * (u - q)
    return jnp.stack([_pack(a1, back, bok), _pack(a2, back, bok)], axis=0)
