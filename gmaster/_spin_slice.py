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
transform afterwards is one memory-bound contraction, ~15x cheaper per call.

The contraction is diagonal in ``m`` (a batched matvec, not a GEMM), so the slab
layout decides whether XLA streams it once or materialises a transpose of a
multi-GiB array: read theta-contiguous it runs at ~760 GB/s, read ell-contiguous
at ~730 GB/s, and a mismatched pairing drops to ~290 GB/s. One
``(m, theta, ell)`` buffer plus a fused view for the analysis direction gets both
directions close to the first number without storing the slab twice.

There is no parity shortcut here: unlike the scalar m'=0 band, no signed relation
reproduces ``d^l_{m,-2}(pi - theta)`` from the same slice (measured ratios span
0.34-2.98), so the full theta range must be stored.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from s2fft.recursions.price_mcewen import generate_precomputes_jax

_CACHE = {}
_MAX_GEOMETRIES = 2
# A slab is (2L-1) * ntheta * L * 8 bytes: 1.1 GiB at nside 128, 9.0 GiB at 256,
# 77 GiB at 512 (lmax = 3*nside). Above the budget the caller keeps the generic
# scatter loop rather than swapping.
_TABLE_BUDGET = 48 * 1024**3


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


def table_budget():
    return _TABLE_BUDGET


def _blocked(array):
    """Concrete built array, or ``None`` when traced (grad/transpose context)."""
    if not hasattr(array, "block_until_ready"):
        return None
    array.block_until_ready()
    return array


def _cached(theta, L, spin, key):
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    table = _blocked(_build(theta, L, spin))
    if table is None:
        # Inside a transpose/grad trace: nothing concrete to cache, so let the
        # caller fall back to the generic path rather than bake a constant in.
        return None
    if key not in _CACHE and len(_CACHE) >= _MAX_GEOMETRIES:
        _CACHE.clear()
    _CACHE[key] = table
    return table


def slab_for(theta, *, L, spin, nside):
    """Wigner-d slab for this geometry, or ``None`` if it would not fit or is traced.

    One physical buffer in ``(m, theta, ell)`` order serves both directions: the
    analysis contraction reads it as a ``(theta, m, ell)`` view, which XLA fuses
    into the reduction instead of materialising a transpose. A second,
    ell-contiguous copy would have doubled the memory for no measured gain.
    """
    if slice_bytes(nside, L) > _TABLE_BUDGET:
        return None
    return _cached(jnp.asarray(theta), L, spin, (nside, L, spin))


def clear_cache():
    _CACHE.clear()


def forward_latitudinal(ftm, slab, *, L):
    """flm[ell, L-1+m] = sum_theta slice[theta, ell, m] * ftm[theta, L+m]."""
    ftm = jnp.asarray(ftm)[:, 1:]
    return jnp.einsum("tce,tc->ec", jnp.moveaxis(slab, 1, 0), ftm, optimize=True)


def inverse_latitudinal(flm, slab, *, L):
    """Transpose of :func:`forward_latitudinal`; ftm is padded to 2L columns.

    Diagonal in ``m`` like the forward step: column ``m + L - 1`` of ``flm``
    feeds column ``m + L`` of the padded ``ftm`` (the same +1 padding offset).
    """
    contracted = jnp.einsum("cte,ec->tc", slab, jnp.asarray(flm), optimize=True)
    ftm = jnp.zeros((contracted.shape[0], 2 * L), dtype=jnp.complex128)
    return ftm.at[:, 1:].set(contracted)
