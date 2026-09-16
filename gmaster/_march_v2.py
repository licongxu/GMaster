"""The v2 latitudinal march: a CUDA difference-form float32 Wigner-d recurrence behind the JAX FFI.

Same contracts as the Pallas march in :mod:`gmaster._spin_march_pallas` (spin-0 folded analysis /
synthesis of the positive-m block, spin-2 analysis / synthesis of the full ``(L, 2L-1)`` block),
served by four CUDA kernels in ``gmaster/_cuda/march_v2.cu`` that ``nvcc`` builds on first use.

The mathematics (docs/march_v2_maths.md): for the Jacobi row ``v_n = (c1 x + c0) v_{n-1} - cb v_{n-2}``
the kernel marches the pair ``(v, D = v_n - v_{n-1})`` as

    C = c1 (x - 1) + (c1 - 1 - cb + c0),    D <- cb D + C v,    v <- v + D

on the northern hemisphere, and the reflected row ``(-1)^n v_n`` (``x -> |x|``, ``c0 -> -c0``) on
the southern one.  Near the poles and near every turning point the three-term recurrence has a
near-double characteristic root and a plain float32 march amplifies each rounding by ``1/sin(phase
step)``; the difference form injects errors at the scale of ``D`` instead of ``v``, and with the
lane coordinate ``|x| - 1`` and the coefficients carried as (hi, lo) float32 pairs the phase error
is the random-walk floor ``~sqrt(n) u``.  Measured against the fp64 recurrence: rms relative
``1-2e-6`` at Nside 1024, ``5e-6`` at 4096, every order.

Every lane keeps an int32 binade ``ex`` (normalised every ``UNR = 8`` degrees) and the kernels emit
the true-magnitude float32 value ``v * 2^(ex + floor(log2 N(base)) - 127) * u(ell)`` with the uniform
``u(ell) = 2^(log2 N(ell) - floor(log2 N(base)))`` folded into the tables, so the tile partials are
plain float32 numbers and the driver sums them without any exponent bookkeeping.
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
# them before the tile sum, so the window is narrowed at large geometries to hold that buffer under
# this budget (Nside 4096 spin 2: 12.6 MB per row, so 128 rows).
_PARTS_BUDGET = int(float(os.environ.get("GMASTER_V2_PARTS_GB", "2")) * 1024 ** 3)
# Window tables depend on the geometry alone, so they stay resident on the device below this
# budget and are rebuilt per call above it (50-60 ms for a whole Nside 4096 geometry, against a
# 260-900 ms pass).  The budget holds Nside <= 2048 and refuses Nside 4096 (7.5-8.3 GiB per
# geometry, and the polarised pipeline needs its pool): pinning those cost the Nside 4096 spin-2
# pipeline its command buffer.  `utils.make_room` evicts whatever is held.
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

_CU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cuda", "march_v2.cu")
_NAMES = ("gm_march_ana_s0", "gm_march_ana_s0_pair", "gm_march_ana_s2",
          "gm_march_syn_s0", "gm_march_syn_s0_pair", "gm_march_syn_s2")
_LIB = None
_LIB_ERROR = None


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


def _build():
    """Compile the CUDA source into a per-source-hash shared library (cached under ~/.cache)."""
    global _LIB, _LIB_ERROR
    if _LIB is not None or _LIB_ERROR is not None:
        return _LIB
    try:
        devs = [d for d in jax.devices() if d.platform == "gpu"]
        if not devs:
            raise RuntimeError("no GPU device")
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
        for name in _NAMES:
            jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(getattr(lib, name)),
                                        platform="CUDA")
        _LIB = lib
    except Exception as exc:  # noqa: BLE001 - any failure means "route unavailable"
        _LIB_ERROR = exc
        if os.environ.get("GMASTER_MARCH_V2_VERBOSE"):
            print(f"[gmaster] v2 march unavailable: {exc}")
    return _LIB


# Default 0: the v2 march serves every band limit when the CUDA library builds.
# `GMASTER_MARCH_V2=0` restores the exact fp64 band / slice routes (the small-geometry tests
# pin those to 1e-13).  `GMASTER_MARCH_V2_MIN_L` can raise the floor again; below Nside 64 the
# geometry pads to one 512-lane tile, so small maps spend most of each tile on padding.
_MIN_L = int(os.environ.get("GMASTER_MARCH_V2_MIN_L", "0"))


def enabled(L=None) -> bool:
    """True when the v2 march serves the marched latitudinal routes at band limit ``L``.

    Default is every ``L`` once the CUDA library has built.  ``GMASTER_MARCH_V2=0`` disables
    it; ``GMASTER_MARCH_V2_MIN_L`` raises a floor if one is wanted.
    """
    flag = os.environ.get("GMASTER_MARCH_V2", "1")
    if flag != "1":
        return False
    if L is not None and int(L) < _MIN_L:
        return False
    return _build() is not None


def unavailable_reason():
    return _LIB_ERROR


# ------------------------------------------------------------------------------------ geometry
def _split64(a):
    h = np.asarray(a, np.float64).astype(np.float32)
    return h, (np.asarray(a, np.float64) - h.astype(np.float64)).astype(np.float32)


@lru_cache(maxsize=16)
def _geometry_numpy(L, nside, spin):
    """Lane layout for one geometry.  Spin 0: the northern rings (``x >= 0``) in ring order;
    spin 2: the northern block then the southern block, each padded to whole analysis tiles so
    that every tile (of either kernel) lies in one hemisphere."""
    from s2fft.sampling import s2_samples

    # the grid `utils._stable_thetas` gives, built on the host so that the geometry is concrete
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
    width = _M_WINDOW
    if nside is not None:
        geo = _geometry_numpy(L, nside, spin)
        # Spin 0 may be launched paired (two maps, the smaller `TILE_ANA_PAIR` tile and twice the
        # channels), which is four times this stage's partials, so the window is sized for that.
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
    """Kernel tables of one window (rows ``m0 .. m0 + mbp - 1``, rows past ``L`` are dummies):
    ``man`` / ``ex0`` ``(mbp, npad)`` seeds and ``tab`` ``(mbp, L + UNR, KT)`` float32.

    ``log2 N(ell, m)`` is one ``gammaln`` per row at ``ell = M`` plus a float64 cumulative sum of
    ``0.5 log2((ell+m)(ell-m)/((ell-s)(ell+s)))`` along ``ell``: four transcendentals per entry
    measured 41 ms per window at Nside 4096 on this 1/64-rate float64 card, the cumsum a few ms.
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
# half-size theta tile and writes four times a single launch's tile partials.  Measured against the
# unpaired analysis with the (bit-identical, always-cheaper) paired synthesis in place, on a
# `NmtField` that builds a map and its mask together: Nside 2048 spin 0 527 -> 461 ms, Nside 4096
# 3596 -> 3107 ms.  It wins at every size, so there is no cap; the knob is kept for A/B.
_PAIR_MAX_L = int(os.environ.get("GMASTER_V2_PAIR_MAX_L", "0")) or (1 << 30)


def pair_fits(L, nside):
    if int(L) > _PAIR_MAX_L:
        return False
    stats = jax.devices()[0].memory_stats() or {}
    limit = stats.get("bytes_limit") if stats else None
    return (not limit) or pair_bytes(L, nside) * 2 <= limit


def pair_bytes(L, nside):
    """Device bytes one paired spin-0 analysis launch asks for (rhs, one window's tile partials,
    and the two assembled alms).  The v2 march holds no Wigner-d table, so this is the whole
    demand: at Nside 4096 it is 9.6 GiB against the 43.9 GiB slab the Pallas route needed."""
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

    They must never be built *inside* the transform's trace: `jax.ensure_compile_time_eval` there
    turns them into HLO constants of the outer program, which XLA copies to the host at lowering
    (the trap session 36 hit with the 4 GiB ring table, HANDOFF addendum 32 section 7).  They cross
    as jit arguments instead.
    """
    with jax.ensure_compile_time_eval():
        geo = _geometry(L, nside, spin)
        out = []
        for (m0, _mb, mbp) in _windows(L, nside, spin):
            # `m0` is a traced argument and only `mbp` is static, so the whole geometry compiles
            # this builder once (a fresh `jax.jit(partial(...))` per window instead recompiled it
            # every call: 24 windows x ~2 s at Nside 4096, which is the 57 s pass this replaced).
            tabs = _window_tables(jnp.int32(m0), geo, mbp=mbp, L=L, spin=spin)
            out.append(tuple(jax.block_until_ready(t) for t in tabs))
    return tuple(out)


def tables_for(L, nside, spin):
    """Resident window tables for one geometry, or ``None`` when they are too large to hold.

    ``None`` means "build each window inside the transform's trace": the tables are then part of
    the program, XLA frees each window's as it goes, and nothing is pinned between calls.  That is
    the only workable answer at the mask geometry of an Nside 4096 spin-2 pipeline
    (``L = 2 lmax + 1 = 24575`` over 192 windows, 30 GiB if materialised at once, which is what
    exhausted the pool and failed a cuFFT plan there).
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


# A geometry may pin its tables only if the pool can also hold this many ring spectra beside
# them.  The spectrum is the unit the transform allocates in (`ntheta x (2L or L)` complex128) and
# the pipeline holds several at once, so it is the right yardstick: at Nside 4096 it admits the
# scalar geometry (7.5 GiB of tables + 6 x 3.0 GiB = 25.5 of 71.2 GiB) and refuses the polarised
# one (8.25 + 6 x 6.0 = 44.3), whose pipeline peaks above 56 GiB and whose coupling stage is what
# ran out of memory with the tables held.
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
    from gmaster import utils

    if drop_table_cache not in utils._ROOM_HOOKS_LAST:
        utils._ROOM_HOOKS_LAST.append(drop_table_cache)


def _ffi_analysis(spin, geo, m0, mbp, L, man, ex0, tab, rhs, nmaps=1):
    """One analysis launch.  ``rhs`` is ``(mbp, npad, 4 * nmaps)``; the result is
    ``(mbp, L - m0, nch * nmaps)`` summed over theta tiles."""
    if _build() is None:
        raise RuntimeError(f"v2 march unavailable: {_LIB_ERROR}")
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
    if _build() is None:
        raise RuntimeError(f"v2 march unavailable: {_LIB_ERROR}")
    out_t = jax.ShapeDtypeStruct((mbp, geo["npad"], 4 * nmaps), jnp.float32)
    if spin == 0:
        name = "gm_march_syn_s0_pair" if nmaps == 2 else "gm_march_syn_s0"
    else:
        name = "gm_march_syn_s2"
    return jax.ffi.ffi_call(name, out_t)(
        geo["xs_hi"], geo["xs_lo"], man, ex0, tab, coef, geo["mlim"], geo["hemi"],
        m0=np.int64(m0), L=np.int64(L))


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
        # Materialised behind a barrier: each window slices `mb` of these `L` rows, and with the
        # block left fusable XLA rebuilds the parity combination and the lane gather inside every
        # window's slice instead of once (the same inlining trap the spin-2 analysis hit, where it
        # was 73 ms of a 156 ms pass at Nside 2048 -- `.qwen/tmp/s2split_s37.py`).
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
        # One march, both maps: the recurrence is a function of (m, ell, theta) alone, so a field
        # and its mask join the emit as extra channels instead of marching the rows twice (at
        # Nside 2048 spin 0 the two separate launches were 190 ms of the 559 ms `NmtField`).
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


def forward_latitudinal_positive(positive, weights, phase, *, L, nside):
    return _forward_fold_impl(positive, weights, phase, geo_arrays(L, nside, 0),
                              tables_for(L, nside, 0), L=L, nside=nside)


def forward_latitudinal_positive_pair(positive_a, positive_b, weights, phase, *, L, nside):
    return _forward_fold_pair_impl(positive_a, positive_b, weights, phase,
                                   geo_arrays(L, nside, 0), tables_for(L, nside, 0),
                                   L=L, nside=nside)


@partial(jax.jit, static_argnames=("L", "nside"))
def _inverse_fold_impl(positive, phase, garr, tabs, *, L, nside):
    return _fold_synthesise([positive], phase, garr, tabs, L=L, nside=nside)[0]


@partial(jax.jit, static_argnames=("L", "nside"))
def _inverse_fold_pair_impl(positive_a, positive_b, phase, garr, tabs, *, L, nside):
    return _fold_synthesise([positive_a, positive_b], phase, garr, tabs, L=L, nside=nside)


def _fold_synthesise(positives, phase, garr, tabs, *, L, nside):
    """One folded synthesis march driving any number of scalar coefficient sets.

    Like the analysis pair, the recurrence is shared and each extra map costs its own
    accumulators -- which for spin 0 are the lanes the spin-2 kernel uses for its second
    helicity and this one leaves idle, so the paired kernel needs no extra registers.
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
            rows_n[k].append((fn[:, :north] * nf).astype(jnp.complex128))
            rows_s[k].append((fs[:, :north - 1] * sf).astype(jnp.complex128))
    out = []
    for k in range(nmaps):
        north_vals = jnp.concatenate(rows_n[k], axis=0)               # (L, north)
        south_vals = jnp.concatenate(rows_s[k], axis=0)               # (L, north - 1)
        out.append(jnp.transpose(
            jnp.concatenate([north_vals, jnp.flip(south_vals, axis=1)], axis=1)))
    return out


def inverse_latitudinal_positive(positive, phase, *, L, nside):
    return _inverse_fold_impl(positive, phase, geo_arrays(L, nside, 0), tables_for(L, nside, 0),
                              L=L, nside=nside)


def inverse_latitudinal_positive_pair(positive_a, positive_b, phase, *, L, nside):
    """Two folded scalar syntheses, one march (same contract as the single form, applied twice)."""
    return _inverse_fold_pair_impl(positive_a, positive_b, phase, geo_arrays(L, nside, 0),
                                   tables_for(L, nside, 0), L=L, nside=nside)


# ------------------------------------------------------------------------------------- spin 2
@partial(jax.jit, static_argnames=("L", "nside"))
def _forward_impl(ftm, garr, tabs, *, L, nside):
    """``(ntheta, 2L)`` ring spectrum (column ``L + m`` is order ``m``) to ``(L, 2L-1)``."""
    spin = 2
    geo = _geo_dict(garr, L, nside, spin)
    ftm = jnp.asarray(ftm)
    ntheta = geo["ntheta"]
    lane_ring = geo["lane_ring"]
    valid = geo["valid"]
    ring = jnp.where(valid, lane_ring, 0)
    # Transposed once for the whole launch, and with no array-wide reversal: `ftm[::-1]` and the
    # column flip that used to build the mirror channel lowered to one `loop_reverse_slice_fusion`
    # of 85 ms per pass at Nside 2048 spin 2 (`.qwen/tmp/nsys_s2.nsys-rep`), against 50 ms for the
    # march itself.  The ring reversal is folded into the lane gather (`mring`) and the order
    # reversal into each window's own `mb`-row slice, which costs nothing.
    # Materialised once, behind a barrier: `_forward_impl` is inlined into the caller's trace (a
    # `jit` inside a `jit` is not a fusion boundary), and left fusable XLA re-derives these
    # transposes inside every window's lane gather -- 48 strided passes over the 1.6 GiB ring
    # spectrum, 73 ms of a 156 ms analysis pass at Nside 2048 spin 2 that does not appear when the
    # march is timed on its own (`.qwen/tmp/s2split_s37.py`).
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
        # The mirror half is stacked in *descending* order right here, at float32, so that its
        # transpose lands on the output columns directly.  Reversing the assembled complex128
        # block instead fused the reversal, the sign and the final concatenate into one
        # `loop_reverse_slice_fusion` writing the whole 1.2 GiB `(L, 2L-1)` buffer: 85 ms per pass
        # at Nside 2048 spin 2 against 50 ms for the march (`.qwen/tmp/nsys_s2.nsys-rep`).
        mrows.append(parts[::-1, :, 2:])
    # One pad, one concatenate, one transpose.  Per-window `parts.T` fused into the final
    # `(L, 2L-1)` concatenate instead lowered to a single 85 ms `loop_reverse_slice_fusion` per
    # pass at Nside 2048 spin 2 -- 48 small transposes and two reversals writing one 1.2 GiB
    # complex128 buffer, against 50 ms for the march itself (`.qwen/tmp/nsys_s2.nsys-rep`).
    dpart = jnp.concatenate(rows, axis=0)                       # (L, L, 2) rows ascending in m
    mpart = jnp.concatenate(mrows[::-1], axis=0)                # (L, L, 2) rows descending in m
    direct = (dpart[..., 0] + 1j * dpart[..., 1]).astype(jnp.complex128).T
    mirror = (mpart[..., 0] + 1j * mpart[..., 1]).astype(jnp.complex128).T * sign[:, None]
    # `mirror` column c already holds order -(L-1-c), which is the output column for the negative
    # half; column L-1 (order 0) belongs to the direct half and is dropped.
    return jnp.concatenate([mirror[:, :L - 1], direct], axis=1)


def forward_latitudinal(ftm, *, L, spin, nside):
    if int(spin) != 2:
        raise ValueError(f"v2 march implements spin=+2, got spin={spin}")
    return _forward_impl(ftm, geo_arrays(L, nside, 2), tables_for(L, nside, 2), L=L, nside=nside)


@partial(jax.jit, static_argnames=("L", "nside"))
def _inverse_impl(flm, garr, tabs, *, L, nside):
    """``(L, 2L-1)`` (``flm[ell, L-1+m]``) to ``(ntheta, 2L)`` (column ``L + m``; column 0 zero)."""
    spin = 2
    geo = _geo_dict(garr, L, nside, spin)
    flm = jnp.asarray(flm)
    ntheta = geo["ntheta"]
    Lp = L + UNR
    lane_of_ring = geo["lane_of_ring"]                       # lane holding ring r
    lane_of_mirror = lane_of_ring[::-1]                      # lane holding ring pi - theta_r
    dirs, mirs = [], []
    for w, (m0, mb, mbp) in enumerate(_windows(L, nside, spin)):
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
        fp = (v[..., 0] + 1j * v[..., 1]).astype(jnp.complex128).T                 # (npad, mb) f+(theta)
        fm = (v[..., 2] + 1j * v[..., 3]).astype(jnp.complex128).T                 # f-(pi - theta)
        dirs.append(fp[lane_of_ring])                                              # (ntheta, mb), order m
        mirs.append(fm[lane_of_mirror])                                            # (ntheta, mb), order -m
    direct = jnp.concatenate(dirs, axis=1)                                         # column m: order m
    mirror = jnp.concatenate(mirs, axis=1)                                         # column m: order -m
    return jnp.concatenate([jnp.zeros((ntheta, 1), jnp.complex128), mirror[:, 1:][:, ::-1], direct],
                           axis=1)


def inverse_latitudinal(flm, *, L, spin, nside):
    if int(spin) != 2:
        raise ValueError(f"v2 march implements spin=+2, got spin={spin}")
    return _inverse_impl(flm, geo_arrays(L, nside, 2), tables_for(L, nside, 2), L=L, nside=nside)


def clear_cache():
    _geometry_numpy.cache_clear()
    _windows.cache_clear()
    _GEO_CACHE.clear()
    drop_table_cache()
