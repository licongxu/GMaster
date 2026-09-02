"""Precomputed Wigner-d slice for the polarised (spin != 0) latitudinal step.

The generic s2fft latitudinal step recurses over ``m`` inside a ``lax.fori_loop``
and scatters every ``m`` slice into ``dl``/``flm`` with a dynamic row index. That
is scatter- and latency-bound: ~194 ms per call at nside 256, paid seven times per
polarised field (one analysis plus three refinement analyse+synthesise rounds).

On HEALPix the latitudinal step is a *single* contraction against the
``m' = -spin`` slice of the Wigner-d matrix -- one ``m'``, no sum over ``m'``:

    flm[ell, L-1+m] = sum_theta slice[theta, ell, m] * ftm[theta, L+m]

(the ``+1`` on the ftm column is s2fft's HEALPix Fourier padding). Verified exact
against ``utils._forward_latitudinal`` to <1e-15 for spin +/-2 and +/-1 -- the path
that already agrees with NaMaster -- and its transpose reproduces the synthesis
step to the same accuracy.

So the slice is built once per geometry with a scatter-free ``lax.scan`` and every
transform afterwards is one memory-bound contraction, 15-25x cheaper per call.

The contraction is diagonal in ``m`` (a batched matvec, not a GEMM), so the slab
layout decides whether XLA streams the buffer once or gathers it: with the
reduction axis contiguous both directions run at ~700 GB/s, with a strided view
of the same numbers analysis drops to ~200 GB/s (measured 1.7 ms vs 6.0 ms per
call at nside 128).  Two materialised copies -- theta-contiguous for analysis,
ell-contiguous for synthesis -- are therefore worth their 2x memory; a transposed
view of one buffer is not.

There is no parity shortcut here: unlike the scalar m'=0 band, no signed relation
reproduces ``d^l_{m,-2}(pi - theta)`` from the same slice (measured ratios span
0.34-2.98), so the full theta range must be stored.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from s2fft.recursions.price_mcewen import generate_precomputes_jax

THETA_CONTIG = "theta_contig"  # (m, ell, theta) -- analysis reduces over theta
ELL_CONTIG = "ell_contig"  # (m, theta, ell) -- synthesis reduces over ell

_CACHE = {}
_MAX_GEOMETRIES = 2
# A slab pair is 2 * (2L-1) * ntheta * L * 8 bytes: 2.2 GiB at nside 128, 18 GiB
# at 256, 154 GiB at 512 (lmax = 3*nside).  Above the budget the caller keeps the
# generic scatter loop rather than trading a slower theta stage for swapping.
_PAIR_BUDGET = 48 * 1024**3


def _march(theta_trig, half_slice, cpi, cp2, vsign_rows, lrenorm, indices, L, which):
    """Recursion over m for one m-half, returning one slice row per m.

    Mirrors ``s2fft.recursions.price_mcewen.compute_all_slices_jax`` exactly but
    replaces its per-step ``dl.at[row].set(...)`` scatters with scan outputs.
    ``which`` picks the m-half: 0 fills rows ``0..L-1`` in march order, 1 fills
    rows ``L-1..2L-2`` (returned in ascending row order).
    """
    c, s, omc, el = theta_trig
    ntheta = c.shape[0]
    lind = L - 1

    lamb0 = (((el + 1.0)[None, :] * omc[:, None]
              + (2.0 - L + el)[None, :] * c[:, None]
              - half_slice[None, :]) / s[:, None])
    dl_iter = jnp.ones((2, ntheta, L), dtype=jnp.float64)
    dl_iter = dl_iter.at[1, :, lind:].set(
        cpi[0, lind:][None, :] * dl_iter[0, :, lind:] * lamb0[:, lind:]
    )

    # The two rows the recursion seeds before it has two history rows.
    first = jnp.zeros((ntheta, L), dtype=jnp.float64).at[:, lind:].set(
        dl_iter[0, :, lind:] * vsign_rows[0][lind:] * jnp.exp(lrenorm[:, lind:])
    )
    second = jnp.zeros((ntheta, L), dtype=jnp.float64).at[:, lind - 1:].set(
        dl_iter[1, :, lind - 1:] * vsign_rows[1][lind - 1:]
        * jnp.exp(lrenorm[:, lind - 1:])
    )

    def step(carry, xs):
        dl_prev, dl_cur, lrenorm_cur, dl_entry = carry
        m, cpi_m, cp2_m, vsign_m = xs
        index = indices >= L - m - 1
        lamb = (((el + 1.0)[None, :] * omc[:, None]
                 + (m - L + el + 1.0)[None, :] * c[:, None]
                 - half_slice[None, :]) / s[:, None])
        dl_entry = jnp.where(index,
                             cpi_m[None, :] * dl_cur * lamb - cp2_m[None, :] * dl_prev,
                             dl_entry)
        # d^l_{m,m'} at its minimal l, where the recursion has no history.
        dl_entry = jnp.where(indices == L - m - 1, 1.0, dl_entry)
        row = jnp.where(index, dl_entry * vsign_m[None, :] * jnp.exp(lrenorm_cur), 0.0)
        bigi = 1.0 / jnp.abs(dl_entry)
        lbig = jnp.log(jnp.abs(dl_entry))
        return (jnp.where(index, bigi * dl_cur, dl_prev),
                jnp.where(index, bigi * dl_entry, dl_cur),
                jnp.where(index, lrenorm_cur + lbig, lrenorm_cur),
                dl_entry), row

    _, rows = lax.scan(
        step,
        (dl_iter[0], dl_iter[1], lrenorm, jnp.zeros((ntheta, L), dtype=jnp.float64)),
        (jnp.arange(2, L), cpi[1:L - 1], cp2[1:L - 1], vsign_rows[2:]),
    )
    rows = jnp.concatenate([first[None], second[None], rows], axis=0)
    return rows if which == 0 else rows[::-1]


@partial(jax.jit, static_argnames=("L", "spin"))
def _build(theta, L, spin):
    """(2L-1, ntheta, L) slice of the Wigner-d matrix at m' = -spin.

    Row ``L-1+m`` holds ``d^l_{m,-spin}(theta)`` for every theta and ell.
    """
    mm = -spin
    el = jnp.arange(L, dtype=jnp.float64)
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    lrenorm, vsign, cpi, cp2, indices = generate_precomputes_jax(
        L, spin, "healpix", None, True, 0, betas=theta
    )
    trig = (c, s, 1.0 - c, el)
    rows_pos = _march(trig, el + mm + 1.0, cpi, cp2, vsign[:L], lrenorm[0],
                      indices, L, 0)
    rows_neg = _march(trig, el - mm + 1.0, cpi, cp2, vsign[::-1][:L], lrenorm[1],
                      indices, L, 1)
    # Row L-1 (m = 0) comes out of both halves; keep the second, which is what the
    # generic path's loop order leaves in place.
    return jnp.concatenate([rows_pos[:L - 1], rows_neg], axis=0)


def slice_bytes(nside, L):
    return (2 * L - 1) * (4 * nside - 1) * L * 8


def pair_bytes(nside, L):
    return 2 * slice_bytes(nside, L)


def _blocked(array):
    """Concrete built array, or ``None`` when traced (grad/transpose context)."""
    if not hasattr(array, "block_until_ready"):
        return None
    array.block_until_ready()
    return array


def _cached(theta, L, spin, key, layout):
    cached = _CACHE.get((key, layout))
    if cached is not None:
        return cached
    table = _build(theta, L, spin)  # (m, theta, ell)
    if layout == THETA_CONTIG:
        # `+ 0.0` forces a real theta-contiguous buffer; a bare transpose can
        # stay a view, which is exactly the layout the contraction punishes.
        table = table.transpose(0, 2, 1) + 0.0
    table = _blocked(table)
    if table is None:
        # Inside a transpose/grad trace: nothing concrete to cache, so let the
        # caller fall back to the generic path rather than bake a constant in.
        return None
    keys = {k for k, _ in _CACHE}
    if key not in keys and len(keys) >= _MAX_GEOMETRIES:
        _CACHE.clear()
    _CACHE[(key, layout)] = table
    return table


def slabs_for(theta, *, L, spin, nside):
    """``(analysis, synthesis)`` slabs for this geometry, or ``(None, None)``.

    Two materialised copies of the same numbers, laid out so that each direction
    reduces over its contiguous axis: ``(m, ell, theta)`` for
    :func:`forward_latitudinal`, ``(m, theta, ell)`` for
    :func:`inverse_latitudinal`. Returns ``(None, None)`` when the pair does not
    fit the memory budget or this is a trace context, in which case the caller
    keeps the generic scatter loop.
    """
    if pair_bytes(nside, L) > _PAIR_BUDGET:
        return None, None
    theta = jnp.asarray(theta)
    key = (nside, L, spin)
    analysis = _cached(theta, L, spin, key, THETA_CONTIG)
    if analysis is None:
        return None, None
    synthesis = _cached(theta, L, spin, key, ELL_CONTIG)
    if synthesis is None:
        return None, None
    return analysis, synthesis


def clear_cache():
    _CACHE.clear()


def forward_latitudinal(ftm, slab, *, L):
    """flm[ell, L-1+m] = sum_theta slice[theta, ell, m] * ftm[theta, L+m].

    ``slab`` is the ``(m, ell, theta)`` copy from :func:`slabs_for`.
    """
    ftm = jnp.asarray(ftm)[:, 1:]
    return jnp.einsum("cet,tc->ec", slab, ftm, optimize=True)


def inverse_latitudinal(flm, slab, *, L):
    """Transpose of :func:`forward_latitudinal`; ftm is padded to 2L columns.

    ``slab`` is the ``(m, theta, ell)`` copy from :func:`slabs_for`. Diagonal in
    ``m`` like the forward step: column ``m + L - 1`` of ``flm`` feeds column
    ``m + L`` of the padded ``ftm`` (the same +1 padding offset).
    """
    contracted = jnp.einsum("cte,ec->tc", slab, jnp.asarray(flm), optimize=True)
    ftm = jnp.zeros((contracted.shape[0], 2 * L), dtype=jnp.complex128)
    return ftm.at[:, 1:].set(contracted)
