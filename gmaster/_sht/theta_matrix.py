"""Spin-0 latitudinal transform as a contraction with a precomputed Legendre band.

The fused Pallas kernel (`gmaster._sht.sht_pallas`) regenerates every
d^l_{m,0}(theta_j) inside each call, so its latitudinal stage is bound by the
recurrence.  These values depend only on (nside, L), so this module builds them
once, keeps them on the device while they fit, and turns the transform into a
memory-bound multiply-reduce.  When the band cannot be built or cached (not
enough device memory, or inside a `jax.grad` / transpose trace) the public
functions return None and the caller uses the fused kernel.

Layout: only the rows with ell >= m are stored, split into m-blocks of width
`BLOCK`.  Block b covers m in [m0, m0 + mb) and is a slab of shape
(L - m0, mb, north), indexed slab[ell - m0, m_local, j], with the ring index `j`
contiguous (the natural `lax.scan` output order, so no transpose copy is
needed).  Each block's output is the rectangle at (m0, m0) of the (ell, m)
array, so blocks are assembled with pad-and-concatenate; the equivalent scatter
with 2-D index arrays is about three orders of magnitude slower in XLA.

North/south fold: alm = A_north + (-1)^(ell+m) A_south distributes over the ring
sum, so it is applied after the contraction.  Splitting each block's rows by the
parity of i = ell - m0 makes (-1)^(ell+m) a per-column constant within each half,
so each half contracts with a single complex right-hand side.

Row values match `sht_pallas._analysis_kernel`: same normalised three-term
recurrence, same seed log2|diag[m]| + m*log2(sin theta), same value-preserving
exp2 rescale every 16 degrees, and the equator ring counted once.
"""

import contextlib
import gc
import os
import warnings
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
from jax import lax

from gmaster._sht.cuda_gpu import on_cuda_gpu
from gmaster._sht.sht_pallas import (
    _diagonal_normalization,
    _initial_factor,
    _normalized_coefficients_numpy,
)

BLOCK = 64

# Band builder, from GMASTER_BAND_BUILDER: "pallas" (default; one Triton program per
# m-block, see `gmaster._sht.band_pallas`) or "scan" (the XLA reference, also the
# fallback off NVIDIA GPUs).  Both produce the same values.
_BAND_BUILDER = os.environ.get("GMASTER_BAND_BUILDER", "pallas").strip().lower()


def set_band_builder(name):
    """Choose the band builder at runtime: `"pallas"` (default) or `"scan"`."""
    global _BAND_BUILDER
    name = name.strip().lower()
    if name not in ("pallas", "scan"):
        raise ValueError(f"unknown band builder: {name!r}")
    _BAND_BUILDER = name


def band_builder():
    """The configured band builder name."""
    return _BAND_BUILDER


@lru_cache(maxsize=1)
def _has_nvidia_gpu():
    """True on a CUDA GPU (some devices, e.g. ``Tesla T4``, omit ``NVIDIA``)."""
    return on_cuda_gpu()


def _build_slab(theta, L, diag, c1, c2, m0, mb, store_name="float64"):
    """Build one m-block with `lax.scan`: shape (L - m0, mb, north), slab[ell - m0, m_local, j].

    The recurrence always runs in float64; `store_name` only rounds the emitted
    values, so a float32 slab is bit-for-bit the float64 slab cast afterwards.
    Emitting in the storage dtype keeps the scan output (the dominant allocation)
    from coexisting with a float64 copy.

    The recurrence coefficients are sliced and transposed once outside the scan
    so each step receives a contiguous (mb,) row instead of gathering a strided
    column of `c1`/`c2` per degree; the result is identical.
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
    # (rows, mb): row i holds the coefficients for ell = m0 + i.
    k1 = c1[m0:m0 + mb, m0:L].T
    k2 = c2[m0:m0 + mb, m0:L].T

    def degree(state, coeffs):
        qm2, qm1, exponent, factor = state
        ell, a, b = coeffs
        current = jnp.where(
            ell == mi,
            seed,
            a[:, None] * cosine[None, :] * qm1 - b[:, None] * qm2,
        )
        value = jnp.where(ell >= mi, current * factor, 0.0).astype(store)
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
        (jnp.arange(m0, L), k1, k2),
    )
    return vals


def _build_pair(theta, L, diag, c1, c2, m0, mb, store_name):
    """One m-block split into its (even, odd) ell - m0 parity halves, in storage precision.

    The split happens inside the jit so the unsplit slab is freed with the call
    and no float64 copy of the block stays resident.
    """
    vals = _build_slab(theta, L, diag, c1, c2, m0, mb, store_name)
    return vals[0::2], vals[1::2]


_BAND_CACHE = {}
# Geometries whose build ran out of device memory; cleared by `release()`, which
# is the call that frees room for them.
_BAND_FAILED = set()
# Each band is gigabytes, so this bounds device memory rather than key count.
_BAND_MAX_GEOMETRIES = 4
# Number of m-blocks built between `jax.clear_caches()` calls during a build, so
# that executables of finished blocks stop pinning their outputs (see `_band`).
_BUILD_CLEAR_EVERY = 8


def _band(geometry):
    """Parity-split m-banded slabs ``(even_blocks, odd_blocks)`` for a geometry, or None.

    `geometry` is ``(nside, L, block, store_dtype)`` from `band_geometry`.  Rows
    of each block are split by parity of i = ell - m0 (see `_transform`).

    The band is only useful as concrete device buffers, so it must be built
    outside a trace.  `lru_cache` would cache tracers (`UnexpectedTracerError`);
    the explicit dict stores a geometry only once its slabs are concrete.  Under
    `jax.grad` / `jax.linear_transpose` the builders return tracers, so this
    returns None and the caller uses the fused kernel, which keeps gradients
    working.  Also returns None if the build runs out of device memory.
    """
    from gmaster._sht import healpix

    cached = _BAND_CACHE.get(geometry)
    if cached is not None:
        return cached
    if geometry in _BAND_FAILED:
        return None
    nside, L, block, store = geometry
    store = jnp.dtype(store)
    if geometry not in _BAND_CACHE and len(_BAND_CACHE) >= _BAND_MAX_GEOMETRIES:
        _BAND_CACHE.clear()
    theta = healpix._stable_thetas(L, nside)
    diag = jnp.asarray(_diagonal_normalization(L))
    c1_np, c2_np = _normalized_coefficients_numpy(L, 0, L)

    def scan_builder():
        # Created lazily so the emitter route does not stage the coefficient
        # tables as jit constants.
        return jax.jit(partial(_build_pair, theta, L, diag,
                               jnp.asarray(c1_np), jnp.asarray(c2_np)),
                       static_argnames=("m0", "mb", "store_name"))

    emitter = _BAND_BUILDER == "pallas" and _has_nvidia_gpu()
    if emitter:
        # One Triton program per m-block, writing straight into the parity halves.
        # The scan route pays an XLA compile per block, which dominates its build
        # time (minutes vs seconds at Nside 1024).
        from gmaster._sht import band_pallas as _band_pallas

        builder = partial(_band_pallas.build_pair, theta, L, diag, c1_np, c2_np)
    else:
        builder = scan_builder()
    # `ensure_compile_time_eval` makes the scan build eagerly even when called from
    # inside a trace.  `pallas_call` has no eager rule (`pl.program_id` raises
    # NotImplementedError), so the emitter route skips it and instead relies on
    # the concrete-buffer checks below to decline a build inside a trace.
    fold = contextlib.nullcontext() if emitter else jax.ensure_compile_time_eval()
    even, odd = [], []
    try:
        with fold:
            for i, m0 in enumerate(range(0, L, block)):
                # The recurrence is float64; only storage may be float32, and the
                # cast happens inside the block's program so both precisions are
                # never resident for the whole band.
                mb = min(block, L - m0)
                try:
                    pair = builder(m0, mb, store.name)
                except Exception as exc:
                    # RESOURCE_EXHAUSTED goes to the outer handler.  Any other
                    # emitter failure means it cannot compile this shape: switch
                    # to the scan builder for the rest of the band.
                    if not emitter or isinstance(exc, jax.errors.JaxRuntimeError):
                        raise
                    emitter = False
                    builder = scan_builder()
                    warnings.warn(
                        f"GMaster band emitter declined m0={m0} "
                        f"({type(exc).__name__}: {str(exc)[:200]}); this band is "
                        "being built with the scan builder instead", stacklevel=2)
                    pair = builder(m0, mb, store.name)
                if not hasattr(pair[0], "block_until_ready"):
                    # Emitter blocks built inside an outer trace are tracers;
                    # decline now so the caller falls back to the fused kernel.
                    return None
                even.append(pair[0])
                odd.append(pair[1])
                # A cached executable pins a copy of its outputs, which would double
                # the band's footprint.  Each program is used once, so clearing the
                # caches periodically keeps the peak near one band plus one block.
                if (i + 1) % _BUILD_CLEAR_EVERY == 0:
                    for slab in pair:
                        slab.block_until_ready()
                    jax.clear_caches()
        groups = (tuple(even), tuple(odd))
        # Tracers lack `block_until_ready`; that is the concrete-buffer test.
        if not all(hasattr(slab, "block_until_ready") for group in groups
                   for slab in group):
            return None
        # Dispatch is asynchronous, so an out-of-memory error during the loop
        # surfaces here.
        for group in groups:
            for slab in group:
                slab.block_until_ready()
        # Release the executables of the final partial batch as well.
        jax.clear_caches()
    except jax.errors.JaxRuntimeError as exc:
        # The byte estimate is static; the allocator decides.  A band that does not
        # fit returns None so the caller takes the fused kernel instead of failing.
        if "RESOURCE_EXHAUSTED" not in str(exc):
            raise
        gc.collect()
        # Negative cache so later latitudinal passes do not retry the failed build.
        _BAND_FAILED.add(geometry)
        return None
    _BAND_CACHE[geometry] = groups
    return groups


def release():
    """Drop the cached Legendre bands; they rebuild on the next scalar transform.

    The polar Wigner-d block sets (`spin_slice`) and these bands are the two large
    resident tables in GMaster and may not both fit; `spin_slice` calls this to
    reclaim room rather than fail.
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
    """Cache key for the band: ``(nside, L, BLOCK, table_dtype)``.

    The storage precision is user-selectable (`set_table_precision`), so it is
    part of the key: a float64 band is never served to a float32 request or
    vice versa.
    """
    from gmaster._config import table_dtype

    return (nside, L, BLOCK, table_dtype())


def _contract_theta(slab, chan):
    """``acc[e, m, c] = sum_j slab[e, m, j] * chan[m, j, c]`` in one sweep.

    `chan` holds the real and imaginary parts as its last axis (c = 0, 1).  Both
    channels are reduced together in one tuple `lax.reduce`: the broadcast
    spelling ``sum(slab[..., None] * chan[None], axis=2)`` makes XLA
    materialise the (rows, mb, north, 2) product, and an `einsum` is slower
    inside the whole-band program, so this form is markedly faster.

    The reduction runs in the storage dtype and only the block partials are
    widened to float64.  Widening the slab first would create a float64
    temporary the size of the band; with float32 tables the resulting difference
    (~1e-7 relative) is at the level of the table precision itself.
    """
    chan = lax.convert_element_type(chan, slab.dtype)
    zero = jnp.zeros((), slab.dtype)
    re, im = lax.reduce(
        (slab * chan[None, :, :, 0], slab * chan[None, :, :, 1]), (zero, zero),
        lambda a, b: (a[0] + b[0], a[1] + b[1]), (2,))
    return jnp.stack([re.astype(jnp.float64), im.astype(jnp.float64)], axis=-1)


def _assemble_blocks(even_accs, odd_accs, widths, L):
    """Interleave the two parity groups back into contiguous ``ell`` order."""
    blocks = []
    for m0, w, even_acc, odd_acc in zip(range(0, L, BLOCK), widths, even_accs,
                                        odd_accs):
        rows = L - m0
        odd_acc = jnp.pad(odd_acc, ((0, even_acc.shape[0] - odd_acc.shape[0]),
                                    (0, 0), (0, 0)))
        block = jnp.stack([even_acc, odd_acc], axis=1).reshape(
            2 * even_acc.shape[0], w, 2)[:rows]
        blocks.append(jnp.pad(block[..., 0] + 1j * block[..., 1], ((m0, 0), (0, 0))))
    return jnp.concatenate(blocks, axis=1)


@partial(jax.jit, static_argnames=("L", "widths"))
def _transform(slabs_even, slabs_odd, ftm_pos, weights, phase, *, L, widths):
    """Analysis contraction over the whole band; returns the (L, L) complex (ell, m) alms."""
    ntheta = weights.shape[0]
    north = (ntheta + 1) // 2
    m = jnp.arange(L, dtype=jnp.float64)[:, None]
    folded = ftm_pos.T * weights[None, :] * jnp.exp(1j * (m * phase[None, :]))
    jn = jnp.arange(north)
    partner = ntheta - 1 - jn
    # The equator ring is its own partner and is counted once, matching
    # `sht_pallas._analysis_kernel`, which skips the south half there.
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
    return _assemble_blocks(even_accs, odd_accs, widths, L)


def positive_latitudinal(positive, *, L, nside, weights, phase):
    """Positive-m analysis latitudinal transform using the cached band.

    Parameters
    ----------
    positive : (ntheta, L) complex array
        Positive-m block of the azimuthal-FFT map (the slice
        `healpix._fused_forward_sht` passes to `scalar_forward_latitudinal`).
    weights, phase : (ntheta,) arrays
        Per-ring quadrature weights and azimuthal phase offsets.

    Returns
    -------
    (L, L) complex array indexed (ell, m), as returned by the fused kernel, or
    None if the band is unavailable (gradient trace, or insufficient memory),
    in which case the caller should use the fused kernel.
    """
    assert BLOCK % 2 == 0, "the parity split assumes an even m-block"
    band = _band(band_geometry(nside, L))
    if band is None:
        return None
    slabs_even, slabs_odd = band
    return _transform(slabs_even, slabs_odd, positive, weights, phase, L=L,
                      widths=tuple(min(BLOCK, L - m0) for m0 in range(0, L, BLOCK)))


def forward_latitudinal(ftm, *, L, nside, theta, weights, phase):
    """`positive_latitudinal` taking the full (ntheta, 2L) FFT map; `theta` is unused."""
    return positive_latitudinal(ftm[:, L:], L=L, nside=nside, weights=weights,
                                phase=phase)


def _contract_theta_pair(slab, chan_a, chan_b):
    """`_contract_theta` for two right-hand sides over one read of `slab`.

    Four accumulators in one tuple reduce.  The stage is bandwidth-bound, so the
    second map adds arithmetic but no slab traffic: two maps cost roughly 0.55x
    of two separate calls at Nside >= 256 (less gain at small Nside, where the
    stage is not bandwidth-bound).  Outputs are bit-identical to separate calls.
    """
    chan_a = lax.convert_element_type(chan_a, slab.dtype)
    chan_b = lax.convert_element_type(chan_b, slab.dtype)
    zero = jnp.zeros((), slab.dtype)
    acc = lax.reduce(
        (slab * chan_a[None, :, :, 0], slab * chan_a[None, :, :, 1],
         slab * chan_b[None, :, :, 0], slab * chan_b[None, :, :, 1]),
        (zero, zero, zero, zero),
        lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3]), (2,))
    return (jnp.stack([acc[0].astype(jnp.float64), acc[1].astype(jnp.float64)],
                      axis=-1),
            jnp.stack([acc[2].astype(jnp.float64), acc[3].astype(jnp.float64)],
                      axis=-1))


@partial(jax.jit, static_argnames=("L", "widths"))
def _transform_pair(slabs_even, slabs_odd, ftm_a, ftm_b, weights, phase, *,
                    L, widths):
    """`_transform` for two maps sharing one sweep of the band.

    Per-map work (ring fold, parity combination, assembly) is still done twice;
    only the slab read is shared, which is what matters for a bandwidth-bound
    stage.
    """
    ntheta = weights.shape[0]
    north = (ntheta + 1) // 2
    m = jnp.arange(L, dtype=jnp.float64)[:, None]
    jn = jnp.arange(north)
    partner = ntheta - 1 - jn
    sign = 1.0 - 2.0 * jnp.bitwise_and(jnp.arange(L), 1).astype(jnp.float64)
    rhs = {}
    for tag, ftm_pos in (("a", ftm_a), ("b", ftm_b)):
        folded = ftm_pos.T * weights[None, :] * jnp.exp(1j * (m * phase[None, :]))
        north_rhs = folded[:, jn]
        south_rhs = jnp.where(partner == jn, 0.0, folded[:, partner])
        for delta, parity in ((1.0, "even"), (-1.0, "odd")):
            combined = north_rhs + delta * sign[:, None] * south_rhs
            rhs[tag + parity] = jnp.stack([combined.real, combined.imag], axis=-1)
    accs = ([], [], [], [])
    even_a, odd_a, even_b, odd_b = accs
    for parity, group in (("even", slabs_even), ("odd", slabs_odd)):
        for m0, w, slab in zip(range(0, L, BLOCK), widths, group):
            out_a, out_b = _contract_theta_pair(slab, rhs["a" + parity][m0:m0 + w],
                                                rhs["b" + parity][m0:m0 + w])
            (even_a if parity == "even" else odd_a).append(out_a)
            (even_b if parity == "even" else odd_b).append(out_b)
    return (_assemble_blocks(even_a, odd_a, widths, L),
            _assemble_blocks(even_b, odd_b, widths, L))


def positive_latitudinal_pair(positive_a, positive_b, *, L, nside, weights, phase):
    """Two positive-m analysis latitudinal transforms from one pass over the band.

    Arguments are as in `positive_latitudinal`.  Returns ``(alm_a, alm_b)``, or
    None under the same conditions as `positive_latitudinal`, so the caller can
    fall back to two separate calls.
    """
    assert BLOCK % 2 == 0, "the parity split assumes an even m-block"
    band = _band(band_geometry(nside, L))
    if band is None:
        return None
    return _transform_pair(band[0], band[1], positive_a, positive_b, weights,
                           phase, L=L,
                           widths=tuple(min(BLOCK, L - m0)
                                        for m0 in range(0, L, BLOCK)))


_SYNTH_CACHE = {}
# Device memory reserved for the rest of the pipeline (map blocks, the (L, ntheta)
# FFT buffer, the coupling matrix) when deciding whether a second, synthesis-layout
# copy of the band fits.  4 GiB proved too small at Nside 1024 float32 (the process
# filled the card and stalled).
_SYNTH_RESERVE = 16 * 1024 ** 3
ELL_CONTIG = "ell_contig"        # (m, j, ell) -- synthesis reduces a contiguous axis
THETA_CONTIG = "theta_contig"    # (ell, m, j) -- the analysis layout, reduced strided


def _pool_bytes():
    """Total bytes the device will give this process, or +inf when it won't say.

    With a preallocated pool this is `pool_bytes`; with
    `XLA_PYTHON_CLIENT_PREALLOCATE=false` (recommended for large geometries)
    `pool_bytes` is 0 and the pool can grow up to `bytes_limit`, which is then
    the relevant limit.
    """
    try:
        # The CPU backend returns None rather than raising.
        stats = jax.local_devices()[0].memory_stats() or {}
    except Exception:  # pragma: no cover - backend without statistics
        return float("inf")
    for key in ("pool_bytes", "bytes_limit"):
        value = stats.get(key) or 0
        if value:
            return float(value)
    return float("inf")


def _synth_band(geometry):
    """The band laid out for synthesis: ``(blocks, layout)``, or None without a band.

    If memory allows, each block is copied to ``(m_local, j, ell_row)`` layout
    (`ELL_CONTIG`), so `_contract_ell` reduces a contiguous axis.  The ``+ 0.0``
    forces a materialised copy; a transposed view would be reduced strided.

    That copy is a second band, and both must coexist with the pipeline's working
    set for the whole synthesis stage.  The test therefore bounds the total table
    footprint (two bands plus `_SYNTH_RESERVE`) against the pool size, not the
    currently free memory.  If it does not fit, the analysis layout is returned
    (`THETA_CONTIG`) and reduced strided, which is about 2x slower but still far
    cheaper than the fused float64 kernel.
    """
    cached = _SYNTH_CACHE.get(geometry)
    if cached is not None:
        return cached
    band = _band(geometry)
    if band is None:
        return None
    if geometry not in _SYNTH_CACHE and len(_SYNTH_CACHE) >= _BAND_MAX_GEOMETRIES:
        # Same bound as the analysis cache.
        _SYNTH_CACHE.clear()
    nbytes = sum(slab.nbytes for group in band for slab in group)
    if 2 * nbytes + _SYNTH_RESERVE > _pool_bytes():
        # Two copies do not fit alongside the pipeline: reduce the band in place.
        _SYNTH_CACHE[geometry] = (band, THETA_CONTIG)
        return band, THETA_CONTIG
    try:
        groups = tuple(
            tuple((slab.transpose(1, 2, 0) + 0.0) for slab in group) for group in band)
    except jax.errors.JaxRuntimeError as exc:
        # The size check is an estimate and the copy is asynchronous, so the
        # allocator may still refuse it.  Decline the copy, not the route.
        if "RESOURCE_EXHAUSTED" not in str(exc):
            raise
        gc.collect()
        _SYNTH_CACHE[geometry] = (band, THETA_CONTIG)
        return band, THETA_CONTIG
    if not all(hasattr(slab, "block_until_ready") for group in groups
               for slab in group):
        # Under a trace the copy is a tracer: return the analysis layout and cache
        # nothing (caching a tracer raises `UnexpectedTracerError` later).
        return band, THETA_CONTIG
    for group in groups:
        for slab in group:
            slab.block_until_ready()
    _SYNTH_CACHE[geometry] = (groups, ELL_CONTIG)
    return groups, ELL_CONTIG


def warm(nside, L):
    """Build the band and its synthesis layout now; return whether they are resident.

    A traced program can read the band but cannot build it (the builders decline
    under a trace), so a geometry first touched inside a trace would silently use
    the slow fused float64 kernel.  Callers about to trace call this first, at
    top level.

    The return value reflects residency: builders cache a geometry only once its
    slabs are concrete, so True means a traced program may read these buffers.
    Called inside a trace, nothing is cached and this returns False.
    """
    geometry = band_geometry(nside, L)
    if _band(geometry) is None:
        return False
    if geometry in _SYNTH_CACHE:
        return True
    _synth_band(geometry)
    return geometry in _SYNTH_CACHE


def synth_pair_ready(nside, L):
    """Whether two syntheses can share this geometry's band (call after `warm`).

    Sharing requires the contiguous `ELL_CONTIG` layout; on the strided layout a
    second right-hand side is slower than two separate calls (see
    `inverse_latitudinal_pair`).  If this returns False, the caller should run the
    two syntheses separately.
    """
    cached = _SYNTH_CACHE.get(band_geometry(nside, L))
    return cached is not None and cached[1] == ELL_CONTIG


def _contract_ell(slab, rhs):
    """``acc[m, j] = sum_e slab[m, j, e] * rhs[m, e]``, reducing the contiguous ell axis.

    The synthesis counterpart of `_contract_theta`.  The complex `rhs` is split
    into real and imaginary parts reduced as a tuple: a single complex product
    would materialise a complex operand twice the slab's size before the reduce.

    As in `_contract_theta`, the reduction runs in the storage dtype and only the
    partials are widened.  The cast must be applied to real and imaginary parts
    separately: casting a complex array to a real dtype drops the imaginary part.
    """
    re_rhs = lax.convert_element_type(rhs.real, slab.dtype)
    im_rhs = lax.convert_element_type(rhs.imag, slab.dtype)
    zero = jnp.zeros((), slab.dtype)
    re, im = lax.reduce(
        (slab * re_rhs[:, None, :], slab * im_rhs[:, None, :]), (zero, zero),
        lambda a, b: (a[0] + b[0], a[1] + b[1]), (2,))
    return re.astype(jnp.float64) + 1j * im.astype(jnp.float64)


def _contract_ell_leading(slab, rhs):
    """`_contract_ell` on a band still in its analysis (ell, m, j) layout.

    The reduced axis is leading, so XLA reads it strided (about 2x slower).  This
    is the cost of not holding a second copy of the band (see `_synth_band`).
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
    """Synthesis contraction over the whole band; returns the (ntheta, L) positive-m block."""
    # In the analysis layout the ring axis is last instead of second.
    north = slabs_even[0].shape[2 if strided else 1]
    contract = _contract_ell_leading if strided else _contract_ell
    m = jnp.arange(L, dtype=jnp.float64)[:, None]
    sign = 1.0 - 2.0 * jnp.bitwise_and(jnp.arange(L), 1).astype(jnp.float64)
    # Each ring gets its own quadrature weight and phi phase, as in the fused
    # kernel: the south half uses the partner ring's, in descending theta.
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
    """Positive-m synthesis latitudinal transform using the cached band.

    Parameters
    ----------
    positive_alm : (L, L) complex array
        Positive-m alms indexed (ell, m), as passed by
        `healpix._alm2map_core_pallas` to `scalar_inverse_latitudinal`.
    weights, phase : (ntheta,) arrays
        Per-ring quadrature weights and azimuthal phase offsets.

    Returns
    -------
    (ntheta, L) complex array, the positive-m block of the ring FFT with
    weights and phases applied (as the fused kernel returns), or None if the
    band is unavailable, in which case the caller should use the fused kernel.
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


def _contract_ell_pair(slab, rhs_a, rhs_b):
    """`_contract_ell` for two right-hand sides over one read of `slab`.

    Contiguous-ell layout only; see `inverse_latitudinal_pair` for why there is
    no strided variant.
    """
    re_a = lax.convert_element_type(rhs_a.real, slab.dtype)
    im_a = lax.convert_element_type(rhs_a.imag, slab.dtype)
    re_b = lax.convert_element_type(rhs_b.real, slab.dtype)
    im_b = lax.convert_element_type(rhs_b.imag, slab.dtype)
    zero = jnp.zeros((), slab.dtype)
    acc = lax.reduce((slab * re_a[:, None, :], slab * im_a[:, None, :],
                      slab * re_b[:, None, :], slab * im_b[:, None, :]),
                     (zero,) * 4,
                     lambda x, y: (x[0] + y[0], x[1] + y[1],
                                   x[2] + y[2], x[3] + y[3]), (2,))
    return (acc[0].astype(jnp.float64) + 1j * acc[1].astype(jnp.float64),
            acc[2].astype(jnp.float64) + 1j * acc[3].astype(jnp.float64))


@partial(jax.jit, static_argnames=("L", "widths"))
def _inverse_pair(slabs_even, slabs_odd, alm_a, alm_b, weights, phase, *, L,
                  widths):
    """`_inverse` for two sets of alms over one pass over the synthesis band."""
    north = slabs_even[0].shape[1]
    m = jnp.arange(L, dtype=jnp.float64)[:, None]
    sign = 1.0 - 2.0 * jnp.bitwise_and(jnp.arange(L), 1).astype(jnp.float64)
    north_factor = weights[:north] * jnp.exp(1j * (m * phase[:north]))
    south_factor = (
        jnp.flip(weights)[: north - 1]
        * jnp.exp(1j * (m * jnp.flip(phase)[: north - 1]))
    )
    north_a, south_a, north_b, south_b = [], [], [], []
    for m0, w, even, odd in zip(range(0, L, BLOCK), widths, slabs_even, slabs_odd):
        rhs_even_a = alm_a[m0::2, m0:m0 + w].T
        rhs_odd_a = alm_a[m0 + 1::2, m0:m0 + w].T
        rhs_even_b = alm_b[m0::2, m0:m0 + w].T
        rhs_odd_b = alm_b[m0 + 1::2, m0:m0 + w].T
        acc_e_a, acc_e_b = _contract_ell_pair(even, rhs_even_a, rhs_even_b)
        acc_o_a, acc_o_b = _contract_ell_pair(odd, rhs_odd_a, rhs_odd_b)
        block_sign = (1.0 if m0 % 2 == 0 else -1.0) * sign[m0:m0 + w]
        for acc_e, acc_o, nlist, slist in ((acc_e_a, acc_o_a, north_a, south_a),
                                           (acc_e_b, acc_o_b, north_b, south_b)):
            nlist.append((acc_e + acc_o) * north_factor[m0:m0 + w])
            slist.append(((acc_e - acc_o)[:, :north - 1] * block_sign[:, None])
                         * south_factor[m0:m0 + w])
    outs = []
    for nlist, slist in ((north_a, south_a), (north_b, south_b)):
        north_vals = jnp.concatenate(nlist, axis=0)   # (L, north)
        south_vals = jnp.concatenate(slist, axis=0)   # (L, north - 1)
        outs.append(jnp.transpose(jnp.concatenate(
            [north_vals, jnp.flip(south_vals, axis=1)], axis=1)))
    return outs[0], outs[1]


def inverse_latitudinal_pair(positive_alm_a, positive_alm_b, *, L, nside,
                             weights, phase):
    """Two positive-m synthesis latitudinal transforms from one pass over the band.

    Arguments are as in `inverse_latitudinal`.  Returns ``(ftm_a, ftm_b)``, or
    None if the band is unavailable or only the strided analysis layout is
    resident.  On the strided layout a shared pass is several times slower than
    two separate calls (the strided read has no spare bandwidth for a second
    right-hand side), whereas on the contiguous layout it costs about 0.6x of
    two calls.
    """
    assert BLOCK % 2 == 0, "the parity split assumes an even m-block"
    geometry = band_geometry(nside, L)
    synth = _synth_band(geometry)
    if synth is None or synth[1] == THETA_CONTIG:
        return None
    return _inverse_pair(synth[0][0], synth[0][1], jnp.asarray(positive_alm_a),
                         jnp.asarray(positive_alm_b), weights, phase, L=L,
                         widths=tuple(min(BLOCK, L - m0)
                                      for m0 in range(0, L, BLOCK)))
