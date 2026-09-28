"""CUDA float32 difference-form Wigner-d march for the spin-0 and spin-2 latitudinal transforms.

Implements the latitudinal step of the HEALPix SHT (spin-0 folded analysis / synthesis of the
positive-m block, spin-2 analysis / synthesis of the full ``(L, 2L-1)`` block) with the same
contracts as the Pallas march in :mod:`gmaster._sht.spin_march`.  The Wigner-d rows are generated
in registers, never stored.  The CUDA kernels (``gmaster/_native/cuda/march_v2.cu``) are compiled
with ``nvcc`` on first GPU use and cached by source hash; when JAX runs on CPU the OpenMP port
(``gmaster/_native/cpu/march_v2_cpu.cc``) is compiled with ``g++`` instead.  Both are reached
through the JAX FFI.  For ``L`` in the divide-and-conquer range the public entry points delegate
to :mod:`gmaster._sht.dc`.

Mathematics (``docs/march_v2_maths.md``): for the Jacobi row
``v_n = (c1 x + c0) v_{n-1} - cb v_{n-2}`` the kernel marches the pair ``(v, D = v_n - v_{n-1})`` as

    C = c1 (x - 1) + (c1 - 1 - cb + c0),    D <- cb D + C v,    v <- v + D

on the northern hemisphere, and the reflected row ``(-1)^n v_n`` (``x -> |x|``, ``c0 -> -c0``) on
the southern one.  Near the poles and turning points the three-term recurrence has a near-double
characteristic root, so a plain float32 march amplifies each rounding by ``1/sin(phase step)``.
The difference form injects errors at the scale of ``D`` rather than ``v``; with the lane
coordinate ``|x| - 1`` and the coefficients carried as (hi, lo) float32 pairs, the phase error is
the random-walk floor ``~sqrt(n) u``.  Against the fp64 recurrence the rms relative error is
``1-2e-6`` at Nside 1024 and ``5e-6`` at Nside 4096, for every order.

Each lane keeps an int32 binade ``ex`` (renormalised every ``UNR = 8`` degrees).  The kernels emit
the true-magnitude float32 value ``v * 2^(ex + floor(log2 N(base)) - 127) * u(ell)``, with
``u(ell) = 2^(log2 N(ell) - floor(log2 N(base)))`` folded into the tables, so the tile partials are
plain float32 numbers that the driver sums without exponent bookkeeping.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.special import gammaln

UNR = int(os.environ.get("GMASTER_V2_UNR", "8"))
LPT_ANA = int(os.environ.get("GMASTER_V2_LPT_ANA", "16"))
LPT_SYN = int(os.environ.get("GMASTER_V2_LPT_SYN", "8"))
LPT_ANA_PAIR = int(os.environ.get("GMASTER_V2_LPT_ANA_PAIR", "8"))
TILE_ANA, TILE_SYN = 32 * LPT_ANA, 32 * LPT_SYN
TILE_ANA_PAIR = 32 * LPT_ANA_PAIR
HEMI_BLOCK = 256
KT = 12
EX_PAD = -10_000_000
_M_WINDOW = int(os.environ.get("GMASTER_V2_M_WINDOW", "512"))
# The analysis partials of one window are ``(mb, ntile, L - m0, nch)`` float32 and XLA materialises
# them before the tile sum, so at large geometries the m-window is narrowed to keep that buffer
# under this budget (Nside 4096 spin 2: 12.6 MB per row, so 128 rows).
_PARTS_BUDGET = int(float(os.environ.get("GMASTER_V2_PARTS_GB", "2")) * 1024 ** 3)
# Window tables depend on the geometry alone.  Below this budget they stay resident on the device;
# above it they are rebuilt inside each call (50-60 ms for a whole Nside 4096 geometry, against a
# 260-900 ms pass).  The default holds Nside <= 2048 and refuses Nside 4096 (7.5-8.3 GiB per
# geometry), whose polarised pipeline needs that memory for its own buffers.
# `_config.make_room` evicts whatever is held.  Default: max(4 GiB, 12% of the device pool).
_TABLE_CACHE_GB = os.environ.get("GMASTER_V2_TABLE_CACHE_GB")


def _table_cache_bytes():
    if _TABLE_CACHE_GB is not None:
        return int(float(_TABLE_CACHE_GB) * 1024 ** 3)
    try:
        stats = jax.devices()[0].memory_stats() or {}
        limit = stats.get("bytes_limit") or 0
    except Exception:  # noqa: BLE001 - a backend without pool accounting
        limit = 0
    return max(4 * 1024 ** 3, int(limit * 0.12))
_TABLE_CACHE = {}
_GEO_CACHE = {}
_TABLE_CACHE_SIZE = [0]

_NATIVE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_native")
_CU = os.path.join(_NATIVE, "cuda", "march_v2.cu")
_CPU_CC = os.path.join(_NATIVE, "cpu", "march_v2_cpu.cc")
_NAMES = ("gm_march_ana_s0", "gm_march_ana_s0_pair", "gm_march_ana_s2",
          "gm_march_syn_s0", "gm_march_syn_s0_pair", "gm_march_syn_s2")
_LIB = None
_LIB_ERROR = None
_CPU_LIB = None
_CPU_LIB_ERROR = None


def _nvcc():
    cand = [os.environ.get("GMASTER_NVCC", ""), "/usr/local/cuda/bin/nvcc", "nvcc"]
    for c in cand:
        if not c:
            continue
        try:
            out = subprocess.run([c, "--version"], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if out.returncode == 0 and "release 1" in out.stdout:
            return c
    return None


def _defs():
    """Extra `-D` flags for the kernel build (`GMASTER_V2_DEFS="LPT_ANA=8 UNR=4"`)."""
    raw = os.environ.get("GMASTER_V2_DEFS", "").split()
    return [f"-D{d}" for d in raw]


@lru_cache(maxsize=1)
def _cudart():
    """The CUDA runtime library, or None."""
    import glob

    names = ["libcudart.so", "libcudart.so.13", "libcudart.so.12"]
    for d in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        names += sorted(glob.glob(os.path.join(d, "libcudart.so*")))
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def device_memory_info():
    """`(free, total)` bytes of the current CUDA device from the driver, or None."""
    rt = _cudart()
    if rt is None:
        return None
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    if rt.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) != 0:
        return None
    return int(free.value), int(total.value)


def _retain_pool_memory(ndev):
    """Keep freed device memory in the CUDA memory pool instead of returning it at every sync.

    JAX's `cuda_async` allocator draws from each device's current memory pool, whose release
    threshold defaults to 0.  With `XLA_PYTHON_CLIENT_PREALLOCATE=false` (recommended in the README
    for large maps) every large temporary would then be unmapped at the next synchronisation and
    mapped again on the next call, about half the Nside 2048 spin-2 coupling stage.  A threshold of
    UINT64_MAX keeps the process's high-water mark, as a caching allocator does.
    `GMASTER_RETAIN_POOL=0` leaves the pool alone (e.g. on a shared GPU).
    """
    if os.environ.get("GMASTER_RETAIN_POOL", "1") == "0":
        return
    rt = _cudart()
    if rt is None:
        return
    value = ctypes.c_uint64(2 ** 64 - 1)
    for dev in range(ndev):
        pool = ctypes.c_void_p()
        if rt.cudaDeviceGetMemPool(ctypes.byref(pool), dev) == 0:
            rt.cudaMemPoolSetAttribute(pool, 4, ctypes.byref(value))   # cudaMemPoolAttrReleaseThreshold


def _build():
    """Compile the CUDA kernels and register their FFI targets; return the library or None.

    The shared library is keyed by a hash of the source, compute capability, JAX version and
    `GMASTER_V2_DEFS`, and cached in `GMASTER_CUDA_CACHE` (default ``~/.cache/gmaster``).  Any
    failure is recorded in `_LIB_ERROR` and makes the route unavailable.
    """
    global _LIB, _LIB_ERROR
    if _LIB is not None or _LIB_ERROR is not None:
        return _LIB
    try:
        devs = [d for d in jax.devices() if d.platform == "gpu"]
        if not devs:
            raise RuntimeError("no GPU device")
        try:
            _retain_pool_memory(len(devs))
        except Exception:  # noqa: BLE001 - an allocator tweak must never cost the route
            pass
        cc = str(getattr(devs[0], "compute_capability", "")).replace(".", "")
        if not cc:
            raise RuntimeError("unknown compute capability")
        src = open(_CU).read()
        digest = hashlib.sha1(
            (src + cc + jax.__version__ + " ".join(_defs())).encode()).hexdigest()[:12]
        cache = os.environ.get("GMASTER_CUDA_CACHE",
                               os.path.join(os.path.expanduser("~"), ".cache", "gmaster"))
        os.makedirs(cache, exist_ok=True)
        so = os.path.join(cache, f"libgm_march_v2_{digest}.so")
        if not os.path.exists(so):
            nvcc = _nvcc()
            if nvcc is None:
                raise RuntimeError("nvcc not found (set GMASTER_NVCC)")
            inc = jax.ffi.include_dir()
            tmp = so + f".{os.getpid()}.tmp"
            cmd = [nvcc, "-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC",
                   f"-arch=sm_{cc}", "-diag-suppress", "940,2473",
                   f"-DLPT_ANA_PAIR={LPT_ANA_PAIR}", *_defs(),
                   "-I", inc, "-o", tmp, _CU]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            if out.returncode != 0:
                raise RuntimeError("nvcc failed:\n" + out.stderr[-4000:])
            os.replace(tmp, so)
        lib = ctypes.CDLL(so)
        for name in _NAMES + ("gm_ring_fold", "gm_ring_fold_c"):
            jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(getattr(lib, name)),
                                        platform="CUDA")
        _LIB = lib
    except Exception as exc:  # noqa: BLE001 - any failure means "route unavailable"
        _LIB_ERROR = exc
        if os.environ.get("GMASTER_MARCH_V2_VERBOSE"):
            print(f"[gmaster] v2 march unavailable: {exc}")
    return _LIB


def _build_cpu():
    """Compile the OpenMP CPU march (same recurrence as the CUDA kernels), cached like `_build`."""
    global _CPU_LIB, _CPU_LIB_ERROR
    if _CPU_LIB is not None or _CPU_LIB_ERROR is not None:
        return _CPU_LIB
    try:
        src = open(_CPU_CC).read()
        digest = hashlib.sha1((src + "cpu" + jax.__version__).encode()).hexdigest()[:12]
        cache = os.environ.get("GMASTER_CUDA_CACHE",
                               os.path.join(os.path.expanduser("~"), ".cache", "gmaster"))
        os.makedirs(cache, exist_ok=True)
        so = os.path.join(cache, f"libgm_march_v2_cpu_{digest}.so")
        if not os.path.exists(so):
            inc = jax.ffi.include_dir()
            tmp = so + f".{os.getpid()}.tmp"
            cmd = ["g++", "-O3", "-std=c++17", "-shared", "-fPIC", "-fopenmp",
                   "-I", inc, "-o", tmp, _CPU_CC]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if out.returncode != 0:
                raise RuntimeError("g++ cpu march failed:\n" + out.stderr[-4000:])
            os.replace(tmp, so)
        lib = ctypes.CDLL(so)
        for name in _NAMES:
            jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(getattr(lib, name)),
                                        platform="cpu")
        _CPU_LIB = lib
    except Exception as exc:  # noqa: BLE001
        _CPU_LIB_ERROR = exc
        if os.environ.get("GMASTER_MARCH_V2_VERBOSE"):
            print(f"[gmaster] v2 cpu march unavailable: {exc}")
    return _CPU_LIB


def _want_cpu():
    if os.environ.get("GMASTER_MARCH_V2_CPU") == "1":
        return True
    try:
        return jax.default_backend() == "cpu"
    except Exception:  # noqa: BLE001
        return True


# Lowest band limit served by the v2 march (default 0: every band limit, once the library builds).
# `GMASTER_MARCH_V2=0` selects the exact fp64 band / slice routes instead (accurate to 1e-13 on
# small geometries).  Below Nside 64 the geometry pads to one 512-lane tile, so small maps spend
# most of each tile on padding; raise this floor to route them elsewhere.
_MIN_L = int(os.environ.get("GMASTER_MARCH_V2_MIN_L", "0"))


def enabled(L=None) -> bool:
    """True when the v2 march serves the marched latitudinal routes at band limit ``L``.

    Default is every ``L`` once the CUDA or CPU library has built.  ``GMASTER_MARCH_V2=0``
    disables it; ``GMASTER_MARCH_V2_MIN_L`` raises a floor if one is wanted.
    ``GMASTER_MARCH_V2_CPU=1`` forces the OpenMP CPU kernels even if a GPU is present.
    """
    flag = os.environ.get("GMASTER_MARCH_V2", "1")
    if flag != "1":
        return False
    if L is not None and int(L) < _MIN_L:
        return False
    if _want_cpu():
        return _build_cpu() is not None
    return _build() is not None


def unavailable_reason():
    """The exception that made the active (CPU or CUDA) library unavailable, or None."""
    return _CPU_LIB_ERROR if _want_cpu() else _LIB_ERROR


# ------------------------------------------------------------------------------------ geometry
def _split64(a):
    h = np.asarray(a, np.float64).astype(np.float32)
    return h, (np.asarray(a, np.float64) - h.astype(np.float64)).astype(np.float32)


@lru_cache(maxsize=16)
def _geometry_numpy(L, nside, spin):
    """Lane layout for one geometry (lane = one ring, ``-1`` for padding).

    Spin 0: the northern rings (``x >= 0``) in ring order.  Spin 2: the northern block, then the
    southern block, each padded to whole analysis tiles so that every tile of either kernel lies
    in one hemisphere.
    """
    from s2fft.sampling import s2_samples

    # Same grid as `healpix._stable_thetas`, built on the host so that the geometry is concrete.
    theta = (np.asarray(s2_samples.thetas(L, "healpix", nside), np.float64)
             + 8 * np.finfo(np.float64).eps)
    ntheta = theta.shape[0]
    x = np.cos(theta)
    if spin == 0:
        blocks = [np.arange((ntheta + 1) // 2)]
    else:
        blocks = [np.nonzero(x >= 0)[0], np.nonzero(x < 0)[0]]
    lanes, hemi = [], []
    for b, blk in enumerate(blocks):
        nb = blk.shape[0]
        nt = -(-nb // TILE_ANA)
        lanes.append(np.concatenate([blk, np.full(nt * TILE_ANA - nb, -1, blk.dtype)]))
        hemi += [b] * (nt * TILE_ANA // HEMI_BLOCK)
    lane_ring = np.concatenate(lanes)
    valid = lane_ring >= 0
    ring = np.where(valid, lane_ring, 0)
    lane_of_ring = np.zeros(ntheta, np.int64)
    lane_of_ring[lane_ring[valid]] = np.nonzero(valid)[0]
    th = theta[ring]
    xv = np.cos(th)
    xs_hi, xs_lo = _split64(np.where(valid, np.abs(xv) - 1.0, 0.0))
    lmax = float(L - 1)
    sth = np.sin(th)
    t1 = lmax * sth + max(100.0, 0.01 * lmax)
    mlim = spin * np.abs(xv) + np.sqrt(np.maximum(t1 * t1 - (spin * sth) ** 2, 0.0))
    mlim = np.where(valid, mlim, -1.0).astype(np.float32)
    lsh = np.log2(np.sin(th / 2.0))
    lch = np.log2(np.cos(th / 2.0))
    return dict(lane_ring=lane_ring, valid=valid, lane_of_ring=lane_of_ring, theta=th,
                xs_hi=xs_hi, xs_lo=xs_lo, mlim=mlim, hemi=np.asarray(hemi, np.int32),
                lsh=lsh, lch=lch, npad=int(lane_ring.shape[0]), ntheta=ntheta)


def _geometry(L, nside, spin):
    g = _geometry_numpy(L, nside, spin)
    return {k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v) for k, v in g.items()}


_GEO_KEYS = ("xs_hi", "xs_lo", "mlim", "hemi", "lane_ring", "valid", "lane_of_ring",
             "lsh", "lch")


def geo_arrays(L, nside, spin):
    """The kernel's per-lane geometry as device arrays, to cross a jit boundary as arguments."""
    key = (int(L), int(nside), int(spin))
    hit = _GEO_CACHE.get(key)
    if hit is None:
        g = _geometry_numpy(L, nside, spin)
        with jax.ensure_compile_time_eval():
            hit = tuple(jax.block_until_ready(jnp.asarray(g[k])) for k in _GEO_KEYS)
        _GEO_CACHE[key] = hit
    return hit


def _geo_dict(arrays, L, nside, spin):
    g = _geometry_numpy(L, nside, spin)
    out = dict(zip(_GEO_KEYS, arrays))
    out["npad"] = g["npad"]
    out["ntheta"] = g["ntheta"]
    return out


@lru_cache(maxsize=64)
def _windows(L, nside=None, spin=None):
    """The m-windows ``(m0, mb, mbp)``: first order, width, and width padded to a multiple of 4.

    The width is `GMASTER_V2_M_WINDOW`, narrowed to a power of two so that one window's analysis
    partials fit in `_PARTS_BUDGET`.
    """
    width = _M_WINDOW
    if nside is not None:
        geo = _geometry_numpy(L, nside, spin)
        # Spin 0 may be launched paired (two maps: the smaller `TILE_ANA_PAIR` tile and twice the
        # channels, so four times the partials); the window is sized for that case.
        nch, tile = (4, TILE_ANA_PAIR) if spin == 0 else (4, TILE_ANA)
        per_row = (geo["npad"] // tile) * L * nch * 4
        width = min(width, max(4, _PARTS_BUDGET // max(per_row, 1)))
        width = max(4, 1 << (int(width).bit_length() - 1))
    out, m0 = [], 0
    while m0 < L:
        mb = min(width, L - m0)
        out.append((m0, mb, -(-mb // 4) * 4))
        m0 += mb
    return tuple(out)


@partial(jax.jit, static_argnames=("mbp", "L", "spin"))
def _window_tables(m0, geo, *, mbp, L, spin):
    """Kernel tables of one window (rows ``m0 .. m0 + mbp - 1``; rows past ``L`` are dummies).

    Returns the seeds ``man`` (float32 mantissa) and ``ex0`` (int32 binade), both ``(mbp, npad)``,
    and ``tab`` ``(mbp, L + UNR, KT)`` float32 holding the recurrence coefficients as (hi, lo)
    pairs, the normalisation factor ``u`` and the bit-cast exponent.

    ``log2 N(ell, m)`` is one ``gammaln`` per row at ``ell = M`` plus a float64 cumulative sum of
    ``0.5 log2((ell+m)(ell-m)/((ell-s)(ell+s)))`` along ``ell``, which avoids four transcendentals
    per entry (costly on GPUs with reduced float64 throughput).
    """
    Lp = L + UNR
    ms = jnp.asarray(m0, jnp.float64) + jnp.arange(mbp, dtype=jnp.float64)
    m_ = ms[:, None]
    alpha = m_ + spin
    beta = jnp.abs(m_ - spin)
    M = jnp.maximum(m_, spin)
    f = alpha * geo["lsh"][None, :] + beta * geo["lch"][None, :]
    ex0 = jnp.floor(f)
    man = jnp.exp2((f - ex0).astype(jnp.float32))
    valid = geo["valid"][None, :]
    man = jnp.where(valid, man, 0.0).astype(jnp.float32)
    ex0 = jnp.where(valid, ex0, EX_PAD).astype(jnp.int32)
    ell = jnp.arange(Lp, dtype=jnp.float64)[None, :]
    n = jnp.maximum(ell - M, 0.0)
    S = 2.0 * n + alpha + beta
    ok = n >= 2
    den = jnp.where(ok, 2.0 * n * (n + alpha + beta) * (S - 2.0), 1.0)
    c1 = jnp.where(ok, (S - 1.0) * S * (S - 2.0) / den, 1.0)
    c0 = jnp.where(ok, (S - 1.0) * (4.0 * m_ * spin) / den, 0.0)
    cb = jnp.where(ok, 2.0 * (n + alpha - 1.0) * (n + beta - 1.0) * S / den, 1.0)
    G = jnp.where(ok, c1 - 1.0 - cb, 0.0)
    # log2 N(ell, m) = eps_m log2 sqrt((ell+m)!(ell-m)!/((ell-s)!(ell+s)!)), by cumulative ratio
    eps = jnp.where(ms >= spin, 1.0, -1.0)[:, None]
    Mv = M[:, 0]
    lg_seed = 0.5 * (gammaln(Mv + ms + 1) + gammaln(jnp.maximum(Mv - ms, 0.0) + 1)
                     - gammaln(jnp.maximum(Mv - spin, 0.0) + 1) - gammaln(Mv + spin + 1)) / np.log(2.0)
    ratio = ((ell + m_) * jnp.maximum(ell - m_, 1.0)) / (jnp.maximum(ell - spin, 1.0) * (ell + spin))
    step = jnp.where(ell > M, 0.5 * jnp.log2(jnp.maximum(ratio, 1e-300)), 0.0)
    lg2n = eps * (lg_seed[:, None] + jnp.cumsum(step, axis=1))
    lg2n = jnp.where(ell >= M, lg2n, 0.0)
    ellc = jnp.minimum(ell, L - 1)
    lg2n = jnp.where(ell <= L - 1, lg2n, jnp.take_along_axis(lg2n, jnp.full(ell.shape, L - 1, jnp.int32), axis=1))
    nb = jnp.maximum(ell - M - 2.0, 0.0)
    base = jnp.where(ell < M + 2.0, M, M + 2.0 + UNR * jnp.floor(nb / UNR))
    base = jnp.minimum(base, L - 1)
    lg_base = jnp.take_along_axis(lg2n, jnp.broadcast_to(base.astype(jnp.int32), (mbp, Lp)), axis=1)
    lex = jnp.floor(lg_base)
    u = jnp.exp2((lg2n - lex).astype(jnp.float32))
    lexp = (lex + 127.0).astype(jnp.int32)

    def split(a):
        h = a.astype(jnp.float32)
        return h, (a - h.astype(jnp.float64)).astype(jnp.float32)

    c1h, c1l = split(c1)
    eph, epl = split(G + c0)
    emh, eml = split(G - c0)
    zero = jnp.zeros_like(u)
    tab = jnp.stack([c1h, c1l, cb.astype(jnp.float32), u, eph, epl, emh, eml,
                     lax.bitcast_convert_type(lexp, jnp.float32), zero, zero, zero], axis=-1)
    return man, ex0, tab


# The paired analysis kernel holds two maps' right-hand sides in registers, so it runs on the
# half-size theta tile and writes four times a single launch's tile partials.  It is faster than
# two unpaired launches at every size tested (about 13% on an `NmtField` building a map and its
# mask at Nside 2048 and 4096), so by default there is no cap.  `GMASTER_V2_PAIR_MAX_L` sets one.
_PAIR_MAX_L = int(os.environ.get("GMASTER_V2_PAIR_MAX_L", "0")) or (1 << 30)


def pair_fits(L, nside):
    """True when a paired spin-0 analysis is allowed and fits in half the device pool."""
    if int(L) > _PAIR_MAX_L:
        return False
    stats = jax.devices()[0].memory_stats() or {}
    limit = stats.get("bytes_limit") if stats else None
    return (not limit) or pair_bytes(L, nside) * 2 <= limit


def pair_bytes(L, nside):
    """Device bytes of one paired spin-0 analysis: rhs, one window's tile partials, two alms.

    The march holds no Wigner-d table, so this is the whole demand (9.6 GiB at Nside 4096).
    """
    geo = _geometry_numpy(L, nside, 0)
    npad = geo["npad"]
    mbp = max(mbp for (_, _, mbp) in _windows(L, nside, 0))
    rhs = L * npad * 8 * 4
    parts = mbp * (npad // TILE_ANA_PAIR) * L * 4 * 4
    alms = 2 * L * L * 16
    return rhs + parts + alms


def _tables_bytes(L, nside, spin):
    geo = _geometry_numpy(L, nside, spin)
    per_row = geo["npad"] * 8 + (L + UNR) * KT * 4
    return sum(mbp for (_, _, mbp) in _windows(L, nside, spin)) * per_row


def _build_tables(L, nside, spin):
    """Every window's ``(man, ex0, tab)`` as concrete device arrays, built outside any trace.

    Call this outside the transform's trace: under a trace, `jax.ensure_compile_time_eval` would
    turn the tables into HLO constants of the outer program, which XLA copies to the host at
    lowering (gigabytes at large Nside).  The tables are passed to the transform as jit arguments.
    """
    with jax.ensure_compile_time_eval():
        geo = _geometry(L, nside, spin)
        out = []
        for (m0, _mb, mbp) in _windows(L, nside, spin):
            # `m0` is traced and only `mbp` is static, so `_window_tables` compiles once per
            # geometry rather than once per window (about 2 s each at Nside 4096).
            tabs = _window_tables(jnp.int32(m0), geo, mbp=mbp, L=L, spin=spin)
            out.append(tuple(jax.block_until_ready(t) for t in tabs))
    return tuple(out)


def tables_for(L, nside, spin):
    """Resident window tables for one geometry, or ``None`` when they are too large to hold.

    ``None`` means "build each window inside the transform's trace": the tables are then part of
    the program, XLA frees each window's as it goes, and nothing is pinned between calls.  This is
    required at large geometries, e.g. the mask geometry of an Nside 4096 spin-2 pipeline
    (``L = 2 lmax + 1 = 24575``, 192 windows, 30 GiB if materialised at once).  Tables are cached
    only within `GMASTER_V2_TABLE_CACHE_GB` and when `_pool_is_tight` leaves room.
    """
    key = (int(L), int(nside), int(spin))
    hit = _TABLE_CACHE.get(key)
    if hit is not None:
        return hit
    nbytes = _tables_bytes(L, nside, spin)
    if (_TABLE_CACHE_SIZE[0] + nbytes > _table_cache_bytes()
            or _pool_is_tight(nbytes, L, nside, spin)):
        return None
    _register_room_hook()
    tabs = _build_tables(L, nside, spin)
    _TABLE_CACHE[key] = tabs
    _TABLE_CACHE_SIZE[0] += nbytes
    return tabs


# A geometry may pin its tables only if half the pool can also hold this many ring spectra beside
# them.  The spectrum (`ntheta x (2L or L)` complex128) is the unit the transform allocates in, and
# a pipeline holds several at once.  With a 71.2 GiB pool at Nside 4096 this admits the scalar
# geometry (7.5 GiB of tables + 6 x 3.0 GiB = 25.5 GiB) and refuses the polarised one
# (8.25 + 6 x 6.0 = 44.3 GiB), whose pipeline peaks above 56 GiB.
_TABLE_ROOM_SPECTRA = float(os.environ.get("GMASTER_V2_TABLE_ROOM_SPECTRA", "6.0"))


def _pool_is_tight(nbytes, L, nside, spin):
    try:
        stats = jax.devices()[0].memory_stats() or {}
        limit = stats.get("bytes_limit") or 0
    except Exception:  # noqa: BLE001 - a backend without pool accounting
        return False
    if not limit:
        return False
    ntheta = 4 * nside - 1
    spectrum = ntheta * (2 * L if spin else L) * 16
    return (nbytes + _TABLE_ROOM_SPECTRA * spectrum) > 0.5 * limit


def _win_tables(tabs, w, m0, mbp, geo, L, spin):
    """Window ``w``'s tables: the resident ones, or a build inside the current trace."""
    if tabs is not None:
        return tabs[w]
    return _window_tables(jnp.int32(m0), geo, mbp=mbp, L=L, spin=spin)


def drop_table_cache():
    _TABLE_CACHE.clear()
    _TABLE_CACHE_SIZE[0] = 0


def _register_room_hook():
    from gmaster import _config

    if drop_table_cache not in _config._ROOM_HOOKS_LAST:
        _config._ROOM_HOOKS_LAST.append(drop_table_cache)


def _ffi_analysis(spin, geo, m0, mbp, L, man, ex0, tab, rhs, nmaps=1):
    """One analysis launch.  ``rhs`` is ``(mbp, npad, 4 * nmaps)``; the result is
    ``(mbp, L - m0, nch * nmaps)`` summed over theta tiles."""
    if not enabled():
        raise RuntimeError(f"v2 march unavailable: {unavailable_reason()}")
    nch = 2 if spin == 0 else 4
    tile = TILE_ANA_PAIR if nmaps > 1 else TILE_ANA
    ntile = geo["npad"] // tile
    out_t = jax.ShapeDtypeStruct((mbp, ntile, L - m0, nch * nmaps), jnp.float32)
    if spin == 0:
        name = "gm_march_ana_s0_pair" if nmaps == 2 else "gm_march_ana_s0"
    else:
        name = "gm_march_ana_s2"
    parts = jax.ffi.ffi_call(name, out_t)(
        geo["xs_hi"], geo["xs_lo"], man, ex0, tab, rhs, geo["mlim"], geo["hemi"],
        m0=np.int64(m0), L=np.int64(L))
    return jnp.sum(parts, axis=1)


def _ffi_synthesis(spin, geo, m0, mbp, L, man, ex0, tab, coef, nmaps=1):
    if not enabled():
        raise RuntimeError(f"v2 march unavailable: {unavailable_reason()}")
    out_t = jax.ShapeDtypeStruct((mbp, geo["npad"], 4 * nmaps), jnp.float32)
    if spin == 0:
        name = "gm_march_syn_s0_pair" if nmaps == 2 else "gm_march_syn_s0"
    else:
        name = "gm_march_syn_s2"
    return jax.ffi.ffi_call(name, out_t)(
        geo["xs_hi"], geo["xs_lo"], man, ex0, tab, coef, geo["mlim"], geo["hemi"],
        m0=np.int64(m0), L=np.int64(L))


@lru_cache(maxsize=8)
def _ring_nphi(nside):
    i = np.arange(1, 4 * nside)
    return np.where(i < nside, 4 * i, np.where(i <= 3 * nside, 4 * nside, 4 * (4 * nside - i))).astype(np.int32)


def fold_available() -> bool:
    """True when the CUDA library (which carries ``gm_ring_fold``) is loaded."""
    return enabled() and _LIB is not None


def ring_fold_residual(ftm_synth, ftm_map, *, nside):
    """Ring spectrum of ``IFFT(ftm_synth) - map`` without either ring FFT (CUDA only).

    For a HEALPix ring with ``n_phi`` pixels the forward FFT of the inverse FFT of a Hermitian
    band-limited spectrum is its ``n_phi``-periodic fold, ``n_phi * sum_{m = k mod n_phi} F[m]``, so
    the Richardson residual needs only the map's own spectrum ``ftm_map`` (taken once, by the first
    analysis) and one pass over ``ftm_synth``.  Both are ``(4 nside - 1, L)`` positive-order blocks.
    """
    F = jnp.asarray(ftm_synth).astype(jnp.complex64)
    Mf = jnp.asarray(ftm_map).astype(jnp.complex64)
    nphi = jnp.asarray(_ring_nphi(int(nside)))
    return jax.ffi.ffi_call("gm_ring_fold", jax.ShapeDtypeStruct(F.shape, jnp.complex64))(F, Mf, nphi)


def ring_fold_residual_complex(ftm_synth, ftm_map, *, nside):
    """Complex-field version of :func:`ring_fold_residual` on centred ``(nring, 2L-1)`` spectra."""
    F = jnp.asarray(ftm_synth).astype(jnp.complex64)
    Mf = jnp.asarray(ftm_map).astype(jnp.complex64)
    nphi = jnp.asarray(_ring_nphi(int(nside)))
    return jax.ffi.ffi_call("gm_ring_fold_c", jax.ShapeDtypeStruct(F.shape, jnp.complex64))(F, Mf, nphi)


# ------------------------------------------------------------------------------- spin 0, folded
def _fold_rhs(positives, weights, phase, geo, L):
    """``(len(positives), L, npad, 4)`` float32: per order and northern lane the parity-combined
    ``[(G + G') re, im, (G - G') re, im]`` with ``G = positive w e^{i m phi}`` and ``G'`` the
    partner ring at ``pi - theta`` (zero for the equator, which is its own partner)."""
    ntheta = geo["ntheta"]
    north = (ntheta + 1) // 2
    jn = jnp.arange(north)
    partner = ntheta - 1 - jn
    m = jnp.arange(L, dtype=jnp.float64)[:, None]
    p2phi = jnp.exp(1j * (m * jnp.asarray(phase, jnp.float64)[None, :]))
    lane_ring = geo["lane_ring"]
    valid = geo["valid"]
    ring = jnp.where(valid, lane_ring, 0)
    out = []
    for positive in positives:
        folded = jnp.asarray(positive).T * jnp.asarray(weights)[None, :] * p2phi   # (L, ntheta)
        g_n = folded[:, jn]
        g_p = jnp.where(partner == jn, 0.0, folded[:, partner])
        plus = (g_n + g_p)[:, ring]
        minus = (g_n - g_p)[:, ring]
        chan = jnp.stack([plus.real, plus.imag, minus.real, minus.imag], axis=-1)
        chan = jnp.where(valid[None, :, None], chan, 0.0)
        # Materialised behind a barrier: each window slices `mb` of these `L` rows, and if the
        # block were left fusable XLA would rebuild the parity combination and lane gather inside
        # every window's slice instead of once (see the note in `_forward_impl`).
        out.append(lax.optimization_barrier(lax.convert_element_type(chan, jnp.float32)))
    return out


def _fold_analyze(positives, weights, phase, garr, tabs, *, L, nside):
    geo = _geo_dict(garr, L, nside, 0)
    rhs_all = _fold_rhs(positives, weights, phase, geo, L)
    norm = jnp.sqrt((2.0 * jnp.arange(L, dtype=jnp.float64) + 1.0) / (4.0 * jnp.pi))
    rowsign = 1.0 - 2.0 * (jnp.arange(L) % 2).astype(jnp.float64)
    cols = [[] for _ in positives]
    nmaps = len(positives)
    if nmaps == 2:
        # One march, both maps: the recurrence depends on (m, ell, theta) alone, so a field and
        # its mask join the emit as extra channels instead of marching the rows twice.
        rhs_all = [jnp.concatenate(
            [rhs_all[0][:, :, None, :], rhs_all[1][:, :, None, :]], axis=2).reshape(
                rhs_all[0].shape[:2] + (8,))]
    for w, (m0, mb, mbp) in enumerate(_windows(L, nside, 0)):
        man, ex0, tab = _win_tables(tabs, w, m0, mbp, geo, L, 0)
        for rhs in rhs_all:
            r = rhs[m0:m0 + mb]
            if mbp != mb:
                r = jnp.concatenate([r, jnp.zeros((mbp - mb,) + r.shape[1:], r.dtype)], axis=0)
            parts = _ffi_analysis(0, geo, m0, mbp, L, man, ex0, tab, r,
                                  nmaps=nmaps if len(rhs_all) == 1 else 1)[:mb]
            scale = rowsign[m0:m0 + mb][:, None] * norm[m0:][None, :]
            for k in range(nmaps if len(rhs_all) == 1 else 1):
                # the closed form's (-1)^(m+s) (docs/latitudinal_march_maths.md section 1)
                block = (parts[..., 2 * k] + 1j * parts[..., 2 * k + 1]).astype(jnp.complex128)
                cols[k].append(jnp.concatenate(
                    [jnp.zeros((m0, mb), jnp.complex128), (block * scale).T], axis=0))
    return [jnp.concatenate(c, axis=1) for c in cols]


@partial(jax.jit, static_argnames=("L", "nside"))
def _forward_fold_impl(positive, weights, phase, garr, tabs, *, L, nside):
    return _fold_analyze([positive], weights, phase, garr, tabs, L=L, nside=nside)[0]


@partial(jax.jit, static_argnames=("L", "nside"))
def _forward_fold_pair_impl(positive_a, positive_b, weights, phase, garr, tabs, *, L, nside):
    return _fold_analyze([positive_a, positive_b], weights, phase, garr, tabs, L=L, nside=nside)


def _dc(L, *arrays):
    """The divide-and-conquer engine (:mod:`gmaster._sht.dc`) if it serves this band limit, else None.

    Never used inside a trace, where its plan arrays would be captured as program constants; a
    traced call keeps the march.
    """
    from . import dc as _dc_lat

    if any(isinstance(a, jax.core.Tracer) for a in arrays):
        return None
    return _dc_lat if _dc_lat.enabled(L) else None


def forward_latitudinal_positive(positive, weights, phase, *, L, nside):
    """Spin-0 folded analysis: ``(ntheta, L)`` positive-order ring spectrum to ``(L, L)`` alm.

    ``weights`` are the ring quadrature weights and ``phase`` the ring phase offsets.
    """
    dc = _dc(L, positive)
    if dc is not None:
        return dc.forward_latitudinal_positive(positive, weights, phase, L=L, nside=nside)
    return _forward_fold_impl(positive, weights, phase, geo_arrays(L, nside, 0),
                              tables_for(L, nside, 0), L=L, nside=nside)


def forward_latitudinal_positive_pair(positive_a, positive_b, weights, phase, *, L, nside):
    """Two folded scalar analyses sharing one march; returns a list of two alm blocks."""
    dc = _dc(L, positive_a)
    if dc is not None:
        return dc.forward_latitudinal_positive_pair(positive_a, positive_b, weights, phase,
                                                    L=L, nside=nside)
    return _forward_fold_pair_impl(positive_a, positive_b, weights, phase,
                                   geo_arrays(L, nside, 0), tables_for(L, nside, 0),
                                   L=L, nside=nside)


@partial(jax.jit, static_argnames=("L", "nside", "cdtype"))
def _inverse_fold_impl(positive, phase, garr, tabs, *, L, nside, cdtype=jnp.complex128):
    return _fold_synthesise([positive], phase, garr, tabs, L=L, nside=nside, cdtype=cdtype)[0]


@partial(jax.jit, static_argnames=("L", "nside"))
def _inverse_fold_pair_impl(positive_a, positive_b, phase, garr, tabs, *, L, nside):
    return _fold_synthesise([positive_a, positive_b], phase, garr, tabs, L=L, nside=nside)


def _fold_synthesise(positives, phase, garr, tabs, *, L, nside, cdtype=jnp.complex128):
    """One folded synthesis march driving any number of scalar coefficient sets.

    ``cdtype`` is the output type; the phased rows are rounded to it where they are formed.

    The recurrence is shared and each extra map needs only its own accumulators.  For spin 0
    these occupy the channels the spin-2 kernel uses for its second helicity, so the paired
    kernel needs no extra registers.
    """
    geo = _geo_dict(garr, L, nside, 0)
    ntheta = geo["ntheta"]
    north = (ntheta + 1) // 2
    norm = jnp.sqrt((2.0 * jnp.arange(L, dtype=jnp.float64) + 1.0) / (4.0 * jnp.pi))
    rowsign = 1.0 - 2.0 * (jnp.arange(L) % 2).astype(jnp.float64)     # (-1)^m of the closed form
    scale = norm[:, None] * rowsign[None, :]
    alms = [lax.optimization_barrier(jnp.asarray(p) * scale) for p in positives]
    nmaps = len(alms)
    Lp = L + UNR
    phase = jnp.asarray(phase, jnp.float64)
    rows_n = [[] for _ in alms]
    rows_s = [[] for _ in alms]
    for w, (m0, mb, mbp) in enumerate(_windows(L, nside, 0)):
        man, ex0, tab = _win_tables(tabs, w, m0, mbp, geo, L, 0)
        coef = jnp.zeros((mbp, Lp, 4), jnp.float32)
        for k, alm in enumerate(alms):
            a = alm[:, m0:m0 + mb].T                                  # (mb, L)
            coef = coef.at[:mb, :L, 2 * k].set(a.real.astype(jnp.float32))
            coef = coef.at[:mb, :L, 2 * k + 1].set(a.imag.astype(jnp.float32))
        v = _ffi_synthesis(0, geo, m0, mbp, L, man, ex0, tab, coef, nmaps=nmaps)[:mb]
        ms = (m0 + jnp.arange(mb, dtype=jnp.float64))[:, None]
        nf = jnp.exp(1j * (ms * phase[:north][None, :]))
        sf = jnp.exp(1j * (ms * jnp.flip(phase)[:north - 1][None, :]))
        for k in range(nmaps):
            fn = v[..., 4 * k] + 1j * v[..., 4 * k + 1]
            fs = v[..., 4 * k + 2] + 1j * v[..., 4 * k + 3]
            rows_n[k].append((fn[:, :north] * nf).astype(cdtype))
            rows_s[k].append((fs[:, :north - 1] * sf).astype(cdtype))
    out = []
    for k in range(nmaps):
        north_vals = jnp.concatenate(rows_n[k], axis=0)               # (L, north)
        south_vals = jnp.concatenate(rows_s[k], axis=0)               # (L, north - 1)
        out.append(jnp.transpose(
            jnp.concatenate([north_vals, jnp.flip(south_vals, axis=1)], axis=1)))
    return out


def inverse_latitudinal_positive(positive, phase, *, L, nside, cdtype=jnp.complex128):
    """Spin-0 folded synthesis: ``(L, L)`` alm to the ``(ntheta, L)`` positive-order spectrum."""
    dc = _dc(L, positive)
    if dc is not None:
        return dc.inverse_latitudinal_positive(positive, phase, L=L, nside=nside)
    return _inverse_fold_impl(positive, phase, geo_arrays(L, nside, 0), tables_for(L, nside, 0),
                              L=L, nside=nside, cdtype=cdtype)


def inverse_latitudinal_positive_pair(positive_a, positive_b, phase, *, L, nside):
    """Two folded scalar syntheses, one march (same contract as the single form, applied twice)."""
    dc = _dc(L, positive_a)
    if dc is not None:
        return dc.inverse_latitudinal_positive_pair(positive_a, positive_b, phase, L=L, nside=nside)
    return _inverse_fold_pair_impl(positive_a, positive_b, phase, geo_arrays(L, nside, 0),
                                   tables_for(L, nside, 0), L=L, nside=nside)


# ------------------------------------------------------------------------------------- spin 2
@partial(jax.jit, static_argnames=("L", "nside", "cdtype"))
def _forward_impl(ftm, garr, tabs, *, L, nside, cdtype=jnp.complex128):
    """``(ntheta, 2L)`` ring spectrum (column ``L + m`` is order ``m``) to ``(L, 2L-1)``.

    ``cdtype`` is the output's element type.  The kernel's partials are float32, so complex64
    holds them exactly; Nside 8192 uses it (its complex128 block would be 18 GiB).
    """
    spin = 2
    geo = _geo_dict(garr, L, nside, spin)
    ftm = jnp.asarray(ftm)
    ntheta = geo["ntheta"]
    lane_ring = geo["lane_ring"]
    valid = geo["valid"]
    ring = jnp.where(valid, lane_ring, 0)
    # The spectrum is transposed once for the whole launch and never reversed array-wide: XLA
    # lowers a full reversal to a `loop_reverse_slice_fusion` that costs more than the march
    # itself.  The ring reversal is folded into the lane gather (`mring`) and the order reversal
    # into each window's own `mb`-row slice, which are free.
    # The transposes are materialised behind a barrier: `_forward_impl` is inlined into the
    # caller's trace (a `jit` inside a `jit` is not a fusion boundary), and if left fusable XLA
    # re-derives them inside every window's lane gather, one strided pass over the whole ring
    # spectrum per window.
    dirT = lax.optimization_barrier(ftm[:, L:2 * L].T)      # row m is order +m
    negT = lax.optimization_barrier(ftm[:, 1:L + 1].T)      # row k is order -m at k = L-1-m
    mring = jnp.where(valid, ntheta - 1 - lane_ring, 0)
    sign = 1.0 - 2.0 * ((jnp.arange(L) + spin) % 2).astype(jnp.float64)      # (-1)^(ell + s)
    rowsign = 1.0 - 2.0 * (jnp.arange(L) % 2).astype(jnp.float64)            # (-1)^m of the row
    rows, mrows = [], []
    for w, (m0, mb, mbp) in enumerate(_windows(L, nside, spin)):
        man, ex0, tab = _win_tables(tabs, w, m0, mbp, geo, L, spin)
        direct = dirT[m0:m0 + mb][:, ring]                                     # (mb, npad)
        mirror = negT[L - m0 - mb:L - m0][::-1][:, mring]
        chan = jnp.stack([direct.real, direct.imag, mirror.real, mirror.imag], axis=-1)
        chan = jnp.where(valid[None, :, None], chan, 0.0)
        rhs = lax.convert_element_type(chan, jnp.float32)
        if mbp != mb:
            rhs = jnp.concatenate([rhs, jnp.zeros((mbp - mb,) + rhs.shape[1:], rhs.dtype)], 0)
        parts = _ffi_analysis(spin, geo, m0, mbp, L, man, ex0, tab, rhs)[:mb]  # (mb, L-m0, 4)
        if m0:
            parts = jnp.concatenate(
                [jnp.zeros((mb, m0, 4), parts.dtype), parts], axis=1)           # (mb, L, 4)
        parts = parts * rowsign[m0:m0 + mb][:, None, None].astype(jnp.float32)
        rows.append(parts[:, :, :2])
        # The mirror half is stacked in descending order here, at float32, so that its transpose
        # lands on the output columns directly; reversing the assembled complex128 `(L, 2L-1)`
        # block instead would cost a full-size reverse fusion.
        mrows.append(parts[::-1, :, 2:])
    # One concatenate and one transpose per half, rather than per-window transposes fused into
    # the final concatenate (which XLA lowers to one slow reverse-slice fusion over the output).
    dpart = jnp.concatenate(rows, axis=0)                       # (L, L, 2) rows ascending in m
    mpart = jnp.concatenate(mrows[::-1], axis=0)                # (L, L, 2) rows descending in m
    direct = (dpart[..., 0] + 1j * dpart[..., 1]).astype(cdtype).T
    mirror = (mpart[..., 0] + 1j * mpart[..., 1]).astype(cdtype).T * sign[:, None].astype(cdtype)
    # `mirror` column c already holds order -(L-1-c), which is the output column for the negative
    # half; column L-1 (order 0) belongs to the direct half and is dropped.
    return jnp.concatenate([mirror[:, :L - 1], direct], axis=1)


def forward_latitudinal(ftm, *, L, spin, nside, cdtype=jnp.complex128):
    """Spin-2 analysis: ``(ntheta, 2L)`` ring spectrum to ``(L, 2L-1)`` (see `_forward_impl`)."""
    if int(spin) != 2:
        raise ValueError(f"v2 march implements spin=+2, got spin={spin}")
    dc = _dc(L, ftm)
    if dc is not None:
        return dc.forward_latitudinal_spin(ftm, L=L, spin=2, nside=nside)
    return _forward_impl(ftm, geo_arrays(L, nside, 2), tables_for(L, nside, 2), L=L, nside=nside,
                         cdtype=cdtype)


@partial(jax.jit, static_argnames=("L", "nside", "cdtype", "group"))
def _inverse_impl(flm, garr, tabs, *, L, nside, cdtype=jnp.complex128, group=None):
    """``(L, 2L-1)`` (``flm[ell, L-1+m]``) to ``(ntheta, 2L)`` (column ``L + m``; column 0 zero).

    ``cdtype`` as in :func:`_forward_impl`: the kernel writes float32.  ``group = (w0, w1)`` runs
    only those windows and returns their direct block and their mirror block column-reversed, so
    that `inverse_latitudinal` places every block with one concatenation.
    """
    spin = 2
    geo = _geo_dict(garr, L, nside, spin)
    flm = jnp.asarray(flm)
    ntheta = geo["ntheta"]
    Lp = L + UNR
    lane_of_ring = geo["lane_of_ring"]                       # lane holding ring r
    lane_of_mirror = lane_of_ring[::-1]                      # lane holding ring pi - theta_r
    dirs, mirs = [], []
    for w, (m0, mb, mbp) in enumerate(_windows(L, nside, spin)):
        if group is not None and not group[0] <= w < group[1]:
            continue
        man, ex0, tab = _win_tables(tabs, w, m0, mbp, geo, L, spin)
        rs = (1.0 - 2.0 * ((m0 + jnp.arange(mb)) % 2))[:, None]                    # (-1)^m of the row
        ap = flm[:, L - 1 + m0:L - 1 + m0 + mb].T * rs                             # (mb, L)
        am = flm[:, L - 1 - m0 - mb + 1:L - 1 - m0 + 1][:, ::-1].T * rs            # order -m
        coef = jnp.zeros((mbp, Lp, 4), jnp.float32)
        coef = coef.at[:mb, :L, 0].set(ap.real.astype(jnp.float32))
        coef = coef.at[:mb, :L, 1].set(ap.imag.astype(jnp.float32))
        coef = coef.at[:mb, :L, 2].set(am.real.astype(jnp.float32))
        coef = coef.at[:mb, :L, 3].set(am.imag.astype(jnp.float32))
        v = _ffi_synthesis(spin, geo, m0, mbp, L, man, ex0, tab, coef)[:mb]        # (mb, npad, 4)
        fp = (v[..., 0] + 1j * v[..., 1]).astype(cdtype).T                         # (npad, mb) f+(theta)
        fm = (v[..., 2] + 1j * v[..., 3]).astype(cdtype).T                         # f-(pi - theta)
        dirs.append(fp[lane_of_ring])                                              # (ntheta, mb), order m
        mirs.append(fm[lane_of_mirror])                                            # (ntheta, mb), order -m
    direct = jnp.concatenate(dirs, axis=1)                                         # column m: order m
    mirror = jnp.concatenate(mirs, axis=1)                                         # column m: order -m
    if group is not None:
        return direct, mirror[:, ::-1]
    return jnp.concatenate([jnp.zeros((ntheta, 1), cdtype), mirror[:, 1:][:, ::-1], direct],
                           axis=1)

# Windows per program when the synthesis runs in complex64 (Nside 8192).  As a single program it
# needs 34.5 GiB beside its input (a 12 GiB output plus 22.5 GiB of temporaries), which does not fit
# alongside an Nside 8192 spin-2 pipeline.  Groups of 64 windows (6 programs for 384 windows) peak
# at 24.0 GiB, the blocks plus the output; smaller groups (4 or 16) peak higher, at 35.9 GiB.
_INVERSE_GROUP = int(os.environ.get("GMASTER_V2_INVERSE_GROUP", "64"))


def inverse_latitudinal(flm, *, L, spin, nside, cdtype=jnp.complex128):
    """Spin-2 synthesis: ``(L, 2L-1)`` to the ``(ntheta, 2L)`` ring spectrum (see `_inverse_impl`).

    In complex64 the windows run in groups of `GMASTER_V2_INVERSE_GROUP` to bound peak memory.
    """
    if int(spin) != 2:
        raise ValueError(f"v2 march implements spin=+2, got spin={spin}")
    dc = _dc(L, flm)
    if dc is not None:
        return dc.inverse_latitudinal_spin(flm, L=L, spin=2, nside=nside)
    garr, tabs = geo_arrays(L, nside, 2), tables_for(L, nside, 2)
    if cdtype != jnp.complex64:
        return _inverse_impl(flm, garr, tabs, L=L, nside=nside, cdtype=cdtype)
    nwin = len(_windows(L, nside, 2))
    parts = [_inverse_impl(flm, garr, tabs, L=L, nside=nside, cdtype=cdtype,
                           group=(w0, min(w0 + _INVERSE_GROUP, nwin)))
             for w0 in range(0, nwin, _INVERSE_GROUP)]
    # Column-reversed mirror blocks in reverse group order run from order -(L-1) up; the last
    # column of the first group's block is order 0, which belongs to the direct half.
    mirrors = [m for _, m in parts[::-1]]
    mirrors[-1] = mirrors[-1][:, :-1]
    directs = [d for d, _ in parts]
    del parts
    zero = jnp.zeros((directs[0].shape[0], 1), cdtype)
    return jnp.concatenate([zero] + mirrors + directs, axis=1)


def clear_cache():
    _geometry_numpy.cache_clear()
    _windows.cache_clear()
    _GEO_CACHE.clear()
    drop_table_cache()
