"""Precomputed-Legendre-matrix analysis theta stage.

The fused Pallas kernel regenerates every d^l_{m,0}(theta_j) inside each analysis
call, which makes the theta stage recurrence-bound. These numbers are a fixed
function of (nside, L), so generating them once turns the transform into a pure
multiply-reduce that is memory-bound instead.

Layout: the m-banded row set (only ell >= m, so there is no 2x padding waste)
split into m-blocks of BLOCK. Block b covers m in [m0, m0+mb) and holds a slab of
shape (L - m0, mb, north) with `j` contiguous — the natural lax.scan output order,
so building the band never needs a transpose copy. Each block's result is a
rectangular region at (m0, m0) of the (ell, m) output, so the blocks are
assembled with pad-and-concatenate. That matters more than it sounds: the
equivalent scatter with 2-D index arrays measures ~1500x slower in XLA.

The north/south fold `alm = A_north + (-1)^(ell+m) * A_south` distributes over the
j-reduction, so it can be applied after the contraction instead of inside it. That
needs (-1)^(ell+m) to be constant across each reduction, which it is once rows are
split by parity of i = ell - m0: each half then contracts with a single complex RHS
instead of four real columns, halving the arithmetic at identical bytes.

Row values are identical to `_sht_pallas._analysis_kernel`'s: same normalised
3-term recurrence, same seed log2|diag[m]| + m*log2(sin), same
exp2-rescale-every-16 schedule (value-preserving), same equator-counted-once rule.
"""

import gc
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from gmaster._sht_pallas import (
    _diagonal_normalization,
    _normalized_coefficients_numpy,
)

BLOCK = 64


def _initial_factor(exponent):
    return jnp.where(exponent >= -1022, lax.exp2(exponent.astype(jnp.float64)), 0.0)


def _build_slab(theta, L, diag, c1, c2, m0, mb, store_name="float64"):
    """(L - m0, mb, north) slab, slab[ell - m0, m_local, j].

    `store_name` rounds the *emitted* values only.  The recurrence carries on in
    float64 and never sees them, so a float32 slab here is bit-for-bit the
    float64 slab cast afterwards -- and it is what lets the largest geometries be
    built at all: the scan's output is the band's dominant allocation, so emitting
    it in float32 halves the peak instead of leaving a full-precision twin next
    to the storage copy.
    """
    north = (len(theta) + 1) // 2
    store = jnp.dtype(store_name)
    sine = jnp.sin(theta[:north])
    cosine = jnp.cos(theta[:north])
    mi = jnp.arange(m0, m0 + mb)[:, None]
    mf = mi.astype(jnp.float64)
    d = diag[m0:m0 + mb]
    log2_scale = jnp.log2(jnp.abs(d))[:, None] + mf * jnp.log2(sine)[None, :]
    exponent = jnp.floor(log2_scale).astype(jnp.int32)
    seed = jnp.where(d < 0, -1.0, 1.0)[:, None] * jnp.exp2(log2_scale - exponent)

    def degree(state, ell):
        qm2, qm1, exponent, factor = state
        started = ell >= mi
        current = jnp.where(
            ell == mi,
            seed,
            c1[m0:m0 + mb, ell][:, None] * cosine[None, :] * qm1
            - c2[m0:m0 + mb, ell][:, None] * qm2,
        )
        value = jnp.where(started, current * factor, 0.0).astype(store)
        do = (ell >= mi + 2) & (jnp.bitwise_and(ell - mi, 15) == 0)
        largest = jnp.maximum(jnp.abs(qm1), jnp.abs(current))
        large = do & (largest > 2.0**100)
        small = do & (largest < 2.0**-100) & (largest > 0)
        mult = jnp.where(large, 2.0**-100, jnp.where(small, 2.0**100, 1.0))
        exponent = exponent + jnp.where(large, 100, jnp.where(small, -100, 0))
        factor = jnp.where(large | small, _initial_factor(exponent), factor)
        return (qm1 * mult, current * mult, exponent, factor), value

    e0 = jnp.broadcast_to(exponent, (mb, north)).copy()
    _, vals = lax.scan(
        degree,
        (jnp.zeros((mb, north)), jnp.zeros((mb, north)), e0, _initial_factor(e0)),
        jnp.arange(m0, L),
    )
    return vals


def _build_pair(theta, L, diag, c1, c2, m0, mb, store_name):
    """One m-block's two parity halves, in the storage precision.

    The slab is emitted in that precision (`_build_slab`), so no float64 copy of
    a block is ever resident alongside it, and the split is inside the jit so the
    unsplit slab dies with the call too.  At Nside 1024 that is the difference
    between a float32 band that builds and one that runs the pool out of memory
    next to a 73.5 GiB float64 twin.
    """
    vals = _build_slab(theta, L, diag, c1, c2, m0, mb, store_name)
    return vals[0::2], vals[1::2]


_BAND_CACHE = {}
# Geometries whose build ran the pool out of memory.  Cleared by `release()`,
# because that is exactly the call that reclaims the room they needed.
_BAND_FAILED = set()
# Each band is gigabytes, so this is a bound on device memory, not on keys.
_BAND_MAX_GEOMETRIES = 4
# m-blocks built before the compilation caches are dropped mid-build, so that the
# executables of blocks already stored stop pinning their outputs (see `_band`).
_BUILD_CLEAR_EVERY = 8


def _band(geometry):
    """Parity-split m-banded slabs for (nside, L, block), or None if unavailable.

    Rows of each block are split by parity of i = ell - m0 so that each half can
    be contracted with a single complex RHS (see `_transform`).

    The band is only usable as a device buffer, so it must be built outside a
    trace; `lru_cache` would happily cache tracers (the `UnexpectedTracerError`
    trap of HANDOFF Session 5j), hence the explicit dict, which stores a
    geometry only once its slabs are concrete. Under `jax.grad` /
    `jax.linear_transpose` the builder returns tracers even inside
    `ensure_compile_time_eval`, so this returns None and the caller falls back
    to the fused kernel: gradients keep flowing, they just take the kernel.
    """
    from gmaster import utils

    cached = _BAND_CACHE.get(geometry)
    if cached is not None:
        return cached
    if geometry in _BAND_FAILED:
        return None
    nside, L, block, store = geometry
    store = jnp.dtype(store)
    if geometry not in _BAND_CACHE and len(_BAND_CACHE) >= _BAND_MAX_GEOMETRIES:
        _BAND_CACHE.clear()
    theta = utils._stable_thetas(L, nside)
    diag = jnp.asarray(_diagonal_normalization(L))
    c1, c2 = (jnp.asarray(t) for t in _normalized_coefficients_numpy(L, 0, L))
    builder = jax.jit(partial(_build_pair, theta, L, diag, c1, c2),
                      static_argnames=("m0", "mb", "store_name"))
    even, odd = [], []
    try:
        with jax.ensure_compile_time_eval():
            for i, m0 in enumerate(range(0, L, block)):
                # The recurrence is fp64 and stays fp64; only the *storage* may be
                # fp32, and the cast is part of the block's kernel so the fp64
                # original is dead before the next block is built.  Casting the
                # finished band would need both precisions resident at once, which
                # is exactly the budget the fp32 option exists to avoid.
                pair = builder(m0, min(block, L - m0), store.name)
                even.append(pair[0])
                odd.append(pair[1])
                # A cached executable pins a copy of the buffers it produced, so a
                # band built as L/BLOCK separate programs costs twice its table:
                # measured at Nside 256, 1.22 GiB of band holding 2.5 GiB of pool,
                # and the excess vanished from `jax.clear_caches()` while the band
                # stayed (`.qwen/tmp/band_build_memory.py`).  Each program here is
                # used exactly once, so dropping it as soon as its block lands costs
                # nothing and keeps the peak at one table plus one block -- which is
                # what makes the Nside 1024 float32 band (36.8 GiB) reachable on a
                # 71 GiB pool instead of needing 73.6.
                if (i + 1) % _BUILD_CLEAR_EVERY == 0:
                    for slab in pair:
                        slab.block_until_ready()
                    jax.clear_caches()
        groups = (tuple(even), tuple(odd))
        # Tracers lack `block_until_ready`; that is the concrete-buffer test.
        if not all(hasattr(slab, "block_until_ready") for group in groups
                   for slab in group):
            return None
        # Dispatch is asynchronous, so a pool that ran out during the loop is
        # only reported here -- at Nside 1024 this is the call that raised.
        for group in groups:
            for slab in group:
                slab.block_until_ready()
        # Release the executables of the final, un-cleared partial batch too.
        jax.clear_caches()
    except jax.errors.JaxRuntimeError as exc:
        # The byte gate is a static estimate; the pool gets the final say.  A band
        # that does not fit must hand the caller None so it can take the fused
        # kernel — letting the OOM escape kills the run at the next allocation
        # instead, which is what happened at Nside 1024 with a float64 band.
        if "RESOURCE_EXHAUSTED" not in str(exc):
            raise
        gc.collect()
        # Negative cache: without it every one of the ~7 latitudinal passes in a
        # field build retries a multi-gigabyte build that just failed.
        _BAND_FAILED.add(geometry)
        return None
    _BAND_CACHE[geometry] = groups
    return groups


def release():
    """Drop the cached Legendre bands; they rebuild on the next scalar transform.

    The polar Wigner-d block sets and these bands are the two large resident
    tables in GMaster, and at Nside 512 they do not both fit.  The caller that
    needs room calls this rather than failing, because the transform each one
    enables differs by an order of magnitude in cost.
    """
    _BAND_CACHE.clear()
    _SYNTH_CACHE.clear()
    # A geometry that did not fit may fit once the room has been reclaimed.
    _BAND_FAILED.clear()


def band_bytes(nside, L, block=BLOCK, dtype=jnp.float64):
    """Device bytes the band occupies, for the fit check before dispatching."""
    north = (4 * nside - 1 + 1) // 2
    itemsize = jnp.dtype(dtype).itemsize
    return sum(min(block, L - m0) * (L - m0) * north * itemsize
               for m0 in range(0, L, block))


def band_geometry(nside, L):
    """Cache key for the band: the shape *and* the storage precision.

    ``nmt_params.table_dtype`` is user-selectable (`set_table_precision`), and a
    cached fp64 band must not be handed to a caller that asked for fp32 or vice
    versa, so the dtype is part of the key rather than an implicit input.
    """
    from gmaster import utils

    return (nside, L, BLOCK, utils.table_dtype())


def _contract_theta(slab, chan):
    """``acc[e, m, c] = sum_j slab[e, m, j] * chan[m, j, c]`` in one sweep.

    Both channels must be reduced together.  Spelling it
    ``sum(slab[..., None] * chan[None], axis=2)`` leaves the two channels in a
    trailing dimension *after* the reduction axis, and XLA then materialises the
    (rows, mb, north, 2) product rather than folding it into the reduce: 12.16 ms
    for the Nside 512 band against 8.14 ms here (823 vs 1238 GB/s of slab read,
    control 1400 GB/s, three interleaved rounds, values agree to 2.5e-16).  A
    tuple reduce carries both accumulators in a single pass.

    A dot is not the answer either: ``einsum("emj,mjc->emc")`` measures 4x
    *slower* inside this program (203 GB/s) while measuring 4x faster when each
    block is compiled on its own.  Only the whole-band program is what ships.

    The reduce runs in the *storage* width and only the block partials are
    widened.  Widening the slab first is the natural thing to write and it is
    numerically the better estimator, but it hands XLA a float64 temporary the
    size of the band: at Nside 512 with `set_table_precision("fp32")` that costs
    9.37 ms against 3.87 ms here, control 1339 GB/s (the float64 band is
    identical either way, rel 0.00e+00).  The precision bought back is worth
    less than the time: the difference between the two is 1.28e-07 on the alms,
    while the float32 table itself is already good to ~1e-7.
    """
    chan = lax.convert_element_type(chan, slab.dtype)
    zero = jnp.zeros((), slab.dtype)
    re, im = lax.reduce(
        (slab * chan[None, :, :, 0], slab * chan[None, :, :, 1]), (zero, zero),
        lambda a, b: (a[0] + b[0], a[1] + b[1]), (2,))
    return jnp.stack([re.astype(jnp.float64), im.astype(jnp.float64)], axis=-1)


@partial(jax.jit, static_argnames=("L", "widths"))
def _transform(slabs_even, slabs_odd, ftm_pos, weights, phase, *, L, widths):
    ntheta = weights.shape[0]
    north = (ntheta + 1) // 2
    m = jnp.arange(L, dtype=jnp.float64)[:, None]
    folded = ftm_pos.T * weights[None, :] * jnp.exp(1j * (m * phase[None, :]))
    jn = jnp.arange(north)
    partner = ntheta - 1 - jn
    # The equator ring is its own partner and is counted once, matching
    # `_sht_pallas._analysis_kernel`, which skips the south half there.
    north_rhs = folded[:, jn]
    south_rhs = jnp.where(partner == jn, 0.0, folded[:, partner])
    # p = (-1)^(ell+m) is a per-column sign once rows are split by parity of
    # i = ell - m0, and BLOCK is even so (-1)^(m-m0) == (-1)^m.
    sign = 1.0 - 2.0 * jnp.bitwise_and(jnp.arange(L), 1).astype(jnp.float64)
    accs = []
    for group, rhs in ((slabs_even, north_rhs + sign[:, None] * south_rhs),
                       (slabs_odd, north_rhs - sign[:, None] * south_rhs)):
        rhs2 = jnp.stack([rhs.real, rhs.imag], axis=-1)
        accs.append([_contract_theta(slab, rhs2[m0:m0 + w])
                     for m0, w, slab in zip(range(0, L, BLOCK), widths, group)])
    even_accs, odd_accs = accs
    blocks = []
    for m0, w, even_acc, odd_acc in zip(range(0, L, BLOCK), widths, even_accs,
                                        odd_accs):
        rows = L - m0
        # Interleave the two parity groups back into contiguous ell order.
        odd_acc = jnp.pad(odd_acc, ((0, even_acc.shape[0] - odd_acc.shape[0]),
                                    (0, 0), (0, 0)))
        block = jnp.stack([even_acc, odd_acc], axis=1).reshape(
            2 * even_acc.shape[0], w, 2)[:rows]
        blocks.append(jnp.pad(block[..., 0] + 1j * block[..., 1], ((m0, 0), (0, 0))))
    return jnp.concatenate(blocks, axis=1)


def positive_latitudinal(positive, *, L, nside, weights, phase):
    """Positive-m analysis theta transform from the cached band.

    `positive` is the positive-m block of the azimuthal-FFT map, the same slice
    `_fused_forward_sht` hands to `scalar_forward_latitudinal`; the result is the
    (ell, m) complex array that kernel returns. `theta` is not needed — the band
    was built for these rings already, which is exactly the work being removed.
    Returns None when the band cannot be materialized as device buffers, i.e. on
    gradient paths, so the caller can fall back to the fused kernel.
    """
    assert BLOCK % 2 == 0, "the parity split assumes an even m-block"
    band = _band(band_geometry(nside, L))
    if band is None:
        return None
    slabs_even, slabs_odd = band
    return _transform(slabs_even, slabs_odd, positive, weights, phase, L=L,
                      widths=tuple(min(BLOCK, L - m0) for m0 in range(0, L, BLOCK)))


def forward_latitudinal(ftm, *, L, nside, theta, weights, phase):
    """`positive_latitudinal` taking the full FFT map, for the existing probes."""
    return positive_latitudinal(ftm[:, L:], L=L, nside=nside, weights=weights,
                                phase=phase)


_SYNTH_CACHE = {}
# Room left in the pool for the pipeline itself (map blocks, the (L, ntheta) FFT
# buffer, the coupling matrix) when deciding whether a second copy of the band
# fits.  4 GiB was tried and was not enough: at Nside 1024 float32 the copy was
# allowed, the process reached 96.9 GiB of a 97.9 GiB card, and it stalled.
_SYNTH_RESERVE = 16 * 1024 ** 3
ELL_CONTIG = "ell_contig"        # (m, j, ell) -- synthesis reduces a contiguous axis
THETA_CONTIG = "theta_contig"    # (ell, m, j) -- the analysis layout, reduced strided


def _pool_bytes():
    """Total bytes the device will give this process, or +inf when it won't say.

    With a preallocated pool `pool_bytes` is that number; with
    `XLA_PYTHON_CLIENT_PREALLOCATE=false` — the invocation this repo's README
    prescribes for the large geometries — it is reported as 0 while the pool is
    growable up to `bytes_limit`, which is the number the fit test actually wants.
    """
    try:
        stats = jax.local_devices()[0].memory_stats()
    except Exception:  # pragma: no cover - backend without statistics
        return float("inf")
    for key in ("pool_bytes", "bytes_limit"):
        value = stats.get(key) or 0
        if value:
            return float(value)
    return float("inf")


def _synth_band(geometry):
    """The band laid out for synthesis: ``(blocks, layout)``.

    With room to spare the blocks are the band re-laid out as
    ``(m_local, j, ell_row)``, which is what `_contract_ell` wants; a transposed
    *view* is what the theta stage punishes, so each block is materialised and the
    `+ 0.0` is what forces the copy.

    That copy is exactly another band, and both of them have to live alongside the
    pipeline's own working set for the whole synthesis stage.  Asking only whether
    the copy fits *right now* is wrong: measured at Nside 1024 with float32 tables,
    the pool had room for the second 36.8 GiB when the question was asked and the
    process still ended up holding 96.9 GiB of the card's 97.9 and stalling.  So
    the test bounds the whole table footprint against the pool and leaves
    `_SYNTH_RESERVE` for the pipeline.  When it does not fit, the analysis layout is
    handed to the strided contraction — `_spin_slice` measures that at 2.1x the cost
    of the contiguous one, against the fused fp64 kernel, which is the alternative
    to declining.
    """
    cached = _SYNTH_CACHE.get(geometry)
    if cached is not None:
        return cached
    band = _band(geometry)
    if band is None:
        return None
    if geometry not in _SYNTH_CACHE and len(_SYNTH_CACHE) >= _BAND_MAX_GEOMETRIES:
        # Same bound as the analysis band's, or the two caches together would
        # hold twice as many geometries as either alone allows.
        _SYNTH_CACHE.clear()
    nbytes = sum(slab.nbytes for group in band for slab in group)
    if 2 * nbytes + _SYNTH_RESERVE > _pool_bytes():
        # Two copies do not fit with the pipeline: synthesis reduces the band
        # where it lies.
        _SYNTH_CACHE[geometry] = (band, THETA_CONTIG)
        return band, THETA_CONTIG
    try:
        groups = tuple(
            tuple((slab.transpose(1, 2, 0) + 0.0) for slab in group) for group in band)
    except jax.errors.JaxRuntimeError as exc:
        # The headroom check is a snapshot and the transpose is asynchronous, so
        # the pool can still have the last word.  Decline the *copy*, not the route.
        if "RESOURCE_EXHAUSTED" not in str(exc):
            raise
        gc.collect()
        _SYNTH_CACHE[geometry] = (band, THETA_CONTIG)
        return band, THETA_CONTIG
    for group in groups:
        for slab in group:
            slab.block_until_ready()
    _SYNTH_CACHE[geometry] = (groups, ELL_CONTIG)
    return groups, ELL_CONTIG


def _contract_ell(slab, rhs):
    """``acc[m, j] = sum_e slab[m, j, e] * rhs[m, e]``, reducing the ell axis.

    The synthesis twin of `_contract_theta`.  Spelled as a single complex product,
    ``sum(slab * rhs[:, None, :], axis=-1)``, the operand that has to be
    materialised before the reduce is complex — twice the slab — and the reduce
    sits behind that materialisation instead of absorbing it: 22.55 ms for the
    Nside 512 synth band against 16.78 ms here (446 vs 600 GB/s of slab read,
    control 1420 GB/s, three interleaved rounds, values agree to 3.1e-16).

    Like `_contract_theta` this reduces in the storage width and widens the
    partials, for the same reason and the same measured size of effect on a
    float32 table.  The width cast has to be applied to the real and imaginary
    parts separately: casting a complex operand to a real dtype truncates it, and
    the imaginary half of the synthesis sum silently becomes zero (Nside 512,
    decoupled cell wrong by 7.5 relative).
    """
    re_rhs = lax.convert_element_type(rhs.real, slab.dtype)
    im_rhs = lax.convert_element_type(rhs.imag, slab.dtype)
    zero = jnp.zeros((), slab.dtype)
    re, im = lax.reduce(
        (slab * re_rhs[:, None, :], slab * im_rhs[:, None, :]), (zero, zero),
        lambda a, b: (a[0] + b[0], a[1] + b[1]), (2,))
    return re.astype(jnp.float64) + 1j * im.astype(jnp.float64)


def _contract_ell_leading(slab, rhs):
    """`_contract_ell` on a band that stayed in its analysis (ell, m, j) layout.

    The same sum with the reduced axis leading rather than trailing: XLA gathers
    it with a stride instead of streaming it, which `_spin_slice` measures at ~2x
    on the contraction.  That is the price of not holding a second copy of a table
    that is already tens of gigabytes (see `_synth_band`), and it is paid against
    the fused fp64 kernel, which is the alternative.
    """
    re_rhs = lax.convert_element_type(rhs.T.real, slab.dtype)
    im_rhs = lax.convert_element_type(rhs.T.imag, slab.dtype)
    zero = jnp.zeros((), slab.dtype)
    re, im = lax.reduce(
        (slab * re_rhs[..., None], slab * im_rhs[..., None]), (zero, zero),
        lambda a, b: (a[0] + b[0], a[1] + b[1]), (0,))
    return re.astype(jnp.float64) + 1j * im.astype(jnp.float64)


@partial(jax.jit, static_argnames=("L", "widths", "strided"))
def _inverse(slabs_even, slabs_odd, alm, weights, phase, *, L, widths, strided):
    # In the analysis layout the ring axis is last instead of second.
    north = slabs_even[0].shape[2 if strided else 1]
    contract = _contract_ell_leading if strided else _contract_ell
    m = jnp.arange(L, dtype=jnp.float64)[:, None]
    sign = 1.0 - 2.0 * jnp.bitwise_and(jnp.arange(L), 1).astype(jnp.float64)
    # The fused kernel applies each ring's quadrature weight and phi phase to
    # its own half, so the band has to as well -- the south half with the
    # *partner* ring's weight and phase, in descending theta.
    north_factor = weights[:north] * jnp.exp(1j * (m * phase[:north]))
    south_factor = (
        jnp.flip(weights)[: north - 1]
        * jnp.exp(1j * (m * jnp.flip(phase)[: north - 1]))
    )
    north_parts, south_parts = [], []
    for m0, w, even, odd in zip(range(0, L, BLOCK), widths, slabs_even, slabs_odd):
        # `alm` is (ell, m); a block needs its own m window in the two parity
        # row-groups the band was split into.
        rhs_even = alm[m0::2, m0:m0 + w].T
        rhs_odd = alm[m0 + 1::2, m0:m0 + w].T
        acc_e = contract(even, rhs_even)
        acc_o = contract(odd, rhs_odd)
        north_parts.append((acc_e + acc_o) * north_factor[m0:m0 + w])
        # (-1)^(ell+m) = (-1)^(ell-m0) * (-1)^(m+m0): the parity split makes the
        # first factor a constant per group, leaving a per-m sign for the south.
        block_sign = (1.0 if m0 % 2 == 0 else -1.0) * sign[m0:m0 + w]
        south_parts.append(
            ((acc_e - acc_o)[:, :north - 1] * block_sign[:, None])
            * south_factor[m0:m0 + w])
    north_vals = jnp.concatenate(north_parts, axis=0)   # (L, north)
    south_vals = jnp.concatenate(south_parts, axis=0)   # (L, north - 1)
    # ntheta = 4*nside - 1 is odd: row `north - 1` is the equator and is its own
    # mirror image, where the south sum equals the north one exactly (every
    # d^l_{m,0}(pi/2) with odd l+m vanishes), so it is the north half's row and
    # the south contributes only its first north-1 rows.  They follow in
    # descending theta, which is the order the fused kernel writes them in.
    return jnp.transpose(jnp.concatenate(
        [north_vals, jnp.flip(south_vals, axis=1)], axis=1))


def inverse_latitudinal(positive_alm, *, L, nside, weights, phase):
    """Positive-m synthesis theta transform from the cached band.

    `positive_alm` is the (ell, m) positive-m block, the same array
    `_alm2map_core_pallas` hands to `scalar_inverse_latitudinal`; the result is
    the centred (ntheta, L) positive-m block that kernel returns, quadrature
    weight and phi phase included.  Returns None when the band is unavailable so
    the caller can fall back to the fused kernel.
    """
    assert BLOCK % 2 == 0, "the parity split assumes an even m-block"
    geometry = band_geometry(nside, L)
    synth = _synth_band(geometry)
    if synth is None:
        return None
    slabs_even, slabs_odd = synth[0]
    strided = synth[1] == THETA_CONTIG
    return _inverse(slabs_even, slabs_odd, jnp.asarray(positive_alm),
                    weights, phase, L=L, strided=strided,
                    widths=tuple(min(BLOCK, L - m0) for m0 in range(0, L, BLOCK)))
