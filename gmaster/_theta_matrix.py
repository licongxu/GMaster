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


def _build_slab(theta, L, diag, c1, c2, m0, mb):
    """(L - m0, mb, north) fp64 slab, slab[ell - m0, m_local, j]."""
    north = (len(theta) + 1) // 2
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
        value = jnp.where(started, current * factor, 0.0)
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


_BAND_CACHE = {}
# Each band is gigabytes, so this is a bound on device memory, not on keys.
_BAND_MAX_GEOMETRIES = 4


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
    nside, L, block = geometry
    if geometry not in _BAND_CACHE and len(_BAND_CACHE) >= _BAND_MAX_GEOMETRIES:
        _BAND_CACHE.clear()
    theta = utils._stable_thetas(L, nside)
    diag = jnp.asarray(_diagonal_normalization(L))
    c1, c2 = (jnp.asarray(t) for t in _normalized_coefficients_numpy(L, 0, L))
    builder = jax.jit(partial(_build_slab, theta, L, diag, c1, c2),
                      static_argnames=("m0", "mb"))
    even, odd = [], []
    with jax.ensure_compile_time_eval():
        for m0 in range(0, L, block):
            vals = builder(m0, min(block, L - m0))
            even.append(vals[0::2])
            odd.append(vals[1::2])
    groups = (tuple(even), tuple(odd))
    # Tracers lack `block_until_ready`; that is the concrete-buffer test.
    if not all(hasattr(slab, "block_until_ready") for group in groups
               for slab in group):
        return None
    for group in groups:
        for slab in group:
            slab.block_until_ready()
    _BAND_CACHE[geometry] = groups
    return groups


def band_bytes(nside, L, block=BLOCK):
    """Device bytes the band occupies, for the fit check before dispatching."""
    north = (4 * nside - 1 + 1) // 2
    return sum(min(block, L - m0) * (L - m0) * north * 8 for m0 in range(0, L, block))


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
        accs.append([jnp.sum(slab[:, :, :, None] * rhs2[m0:m0 + w][None], axis=2)
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
    band = _band((nside, L, BLOCK))
    if band is None:
        return None
    slabs_even, slabs_odd = band
    return _transform(slabs_even, slabs_odd, positive, weights, phase, L=L,
                      widths=tuple(min(BLOCK, L - m0) for m0 in range(0, L, BLOCK)))


def forward_latitudinal(ftm, *, L, nside, theta, weights, phase):
    """`positive_latitudinal` taking the full FFT map, for the existing probes."""
    return positive_latitudinal(ftm[:, L:], L=L, nside=nside, weights=weights,
                                phase=phase)
