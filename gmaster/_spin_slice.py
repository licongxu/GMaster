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

The contraction is diagonal in ``m`` (a batched matvec, not a GEMM): the row
holding ``d^ell_{m,-spin}`` pairs with exactly one ``m`` column of ``ftm``, and
each contiguous run of the reduction is one ``theta``.  So what decides the speed
is which axis the buffer lays out innermost, not how big it is -- analysis reduces
over ``theta``, synthesis over ``ell``, and making both directions reduce over a
contiguous axis costs memory, not time (1.7 ms vs 6.0 ms per call at nside 128 for
the strided spelling).

Storing only the ``ell >= |m|`` triangle of each layout -- the head below it is
identically zero -- is where the memory comes back: the same contraction runs at
the same speed (12.1 vs 11.5 ms on analysis, 9.3 vs 9.3 ms on synthesis at nside
256) for 1.85x fewer bytes, 9.7 GiB of block sets against 18 GiB of full slabs, and
37 GiB rather than 72 GiB at nside 512, which is what puts 512 inside a 96 GiB card
at all.  When even two triangles will not fit, one theta-contiguous triangle serves
both directions and synthesis pays 2.1x reducing over a strided axis (19.2 vs
9.3 ms at nside 256) -- still ~10x cheaper than the scatter loop it replaces.  See
:func:`slabs_for`.

There is no parity shortcut here: unlike the scalar m'=0 band, no signed relation
reproduces ``d^l_{m,-2}(pi - theta)`` from the same slice (measured ratios span
0.34-2.98), so the full theta range must be stored.
"""

import gc
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from s2fft.recursions.price_mcewen import generate_precomputes_jax

THETA_CONTIG = "theta_contig"  # (m, ell, theta) -- analysis reduces over theta
ELL_CONTIG = "ell_contig"  # (m, theta, ell) -- synthesis reduces over ell

_CACHE = {}
_MAX_GEOMETRIES = 2
# One windowed layout is the ``ell >= lo`` triangle of the slab, one block per
# ``m``-window: 0.65 GiB at nside 128, 4.9 at 256, 37 at 512, 125 at 768
# (lmax = 3*nside).  Both layouts are built while the pair fits, and one layout is
# still worth far more than the generic scatter loop, so the budget admits 512
# (37 GiB against a 71 GiB pool) and refuses 768.
_SLAB_BUDGET = 56 * 1024**3
# Target size for the transient tables the build materialises per theta chunk; the
# march cannot be split along ``m`` (see `_build_triangles`).  Smaller chunks look
# cheaper and are not: at nside 512 a 2 GiB target makes 36 pieces per window, the
# pool fills with 40 MB blocks, and the join then fails to find one contiguous
# 1.3 GiB.  A 6 GiB target (12 pieces per window) builds the same 37.5 GiB layout
# with the peak at 41 GiB.
_BUILD_CHUNK_TARGET = 6 * 1024**3
# Headroom required on top of the layout before the build is approved: the chunk
# table and its per-window copies are alive while the finished blocks accumulate,
# the rest of the pipeline holds maps and coupling matrices, and a pool that is
# nominally large enough but fragmented still refuses a 1.6 GiB block.
_BUILD_RESERVE = 16 * 1024**3
# Row ``L-1+m`` of a slab holds ``d^ell_{m,-spin}``, which vanishes for ``ell <
# |m|`` -- just under half the buffer, streamed on every polarised transform.  A
# contiguous block of rows shares the bound ``lo = min |m|`` in the block, so the
# contraction reads only ``[lo, L)`` (see :func:`forward_latitudinal`).  64 skips
# 42% of the bytes and measures 1.67x on analysis / 1.60x on synthesis at nside
# 128; 16-row blocks skip 48% but pay 4x the kernels and give only 1.34x/1.20x.
_M_BLOCK = 64
_WINDOW_CACHE = {}


def _windows(L):
    """``(r0, r1, lo)`` per m-block: the first ell nonzero anywhere in the block."""
    cached = _WINDOW_CACHE.get(L)
    if cached is None:
        nrows = 2 * L - 1
        cached = tuple(
            (r0, min(r0 + _M_BLOCK, nrows),
             min(abs(r - (L - 1)) for r in range(r0, min(r0 + _M_BLOCK, nrows))))
            for r0 in range(0, nrows, _M_BLOCK)
        )
        _WINDOW_CACHE[L] = cached
    return cached


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


def _table_bytes(ntheta, L):
    return (2 * L - 1) * ntheta * L * 8


def triangle_bytes(nside, L):
    """Bytes of one windowed layout: one block per m-window holding ``ell >= lo``."""
    ntheta = 4 * nside - 1
    return sum((r1 - r0) * (L - lo) * ntheta * 8 for r0, r1, lo in _windows(L))


class _BlockSet(tuple):
    """Tuple of per-window blocks tagged with the axis order they are stored in.

    Subclasses ``tuple`` so caching, indexing and iteration all behave normally;
    the layout tag is what lets :func:`inverse_latitudinal` pick the right
    contraction when a single layout has to serve both directions.
    """

    def __new__(cls, blocks, layout):
        self = super().__new__(cls, blocks)
        self.layout = layout
        return self


# The block set travels through `jax.jit` boundaries as an argument, so it has to
# be a container of array leaves (like the single slab it replaces) with the layout
# as aux data.  An unregistered tuple subclass is treated as a non-array leaf and
# the trace rejects it.
jax.tree_util.register_pytree_node(
    _BlockSet,
    lambda slab: (tuple(slab), slab.layout),
    lambda layout, children: _BlockSet(children, layout),
)


def _build_triangles(theta, L, spin, want_theta, want_ell):
    """Per-m-window triangles (``ell >= lo``) in the requested layouts.

    The march recurses over ``m``, so a window's rows cannot be produced without
    their predecessors and one call always materialises the full slab.  It *can*
    run over a subset of theta (the recurrence is independent per ring), so the
    build walks theta in chunks and copies each window's triangle out immediately:
    peak memory is the accumulated blocks plus two chunk tables instead of the
    full slab plus its transpose.  Chunk lengths are uniform so the march
    compiles once.

    ``+ 0.0`` forces a real buffer: a bare transpose can stay a view, which is
    exactly the layout the contraction punishes.
    """
    ntheta = len(theta)
    chunks = max(1, -(-_table_bytes(ntheta, L) // _BUILD_CHUNK_TARGET))
    per = -(-ntheta // chunks)
    chunks = -(-ntheta // per)
    tail = per * chunks - ntheta
    if tail:
        # Repeat the last ring rather than padding with zero: theta = 0 divides by
        # sin(theta) in the recurrence.  The padded rings are never contracted.
        theta = jnp.concatenate([theta, jnp.repeat(theta[-1:], tail)])
    wins = _windows(L)
    theta_pieces = [[] for _ in wins] if want_theta else None
    ell_pieces = [[] for _ in wins] if want_ell else None
    for start in range(0, ntheta, per):
        table = _build(theta[start:start + per], L, spin)  # (m, theta, ell)
        nvalid = min(per, ntheta - start)  # the padded tail rings are dropped
        pieces = []
        for i, (r0, r1, lo) in enumerate(wins):
            sub = table[r0:r1, :nvalid, lo:]  # (m, theta, ell), zero head dropped
            if want_theta:
                out = sub.transpose(0, 2, 1) + 0.0
                theta_pieces[i].append(out)
                pieces.append(out)
            if want_ell:
                out = sub + 0.0
                ell_pieces[i].append(out)
                pieces.append(out)
        del table
        # Dispatch is asynchronous, so without this the chunk tables of every
        # chunk stay alive at once -- tens of GiB of live buffer at nside 512
        # instead of one chunk's worth on top of the finished blocks.
        for arr in pieces:
            arr.block_until_ready()

    def join(parts, axis):
        """Join one window's chunks, releasing them as soon as the join lands.

        Joining every window first would hold the chunk copies and the finished
        blocks at once -- twice the layout, which is the memory this whole build
        exists to avoid.
        """
        joined = []
        for part in parts:
            arr = part[0] if len(part) == 1 else jnp.concatenate(part, axis=axis)
            arr.block_until_ready()
            part.clear()
            joined.append(arr)
        return joined

    out = {}
    if want_theta:
        out[THETA_CONTIG] = _BlockSet(join(theta_pieces, 2), THETA_CONTIG)
    if want_ell:
        out[ELL_CONTIG] = _BlockSet(join(ell_pieces, 1), ELL_CONTIG)
    return out


def _build_or_none(theta, L, spin, want_theta, want_ell):
    """Build the block sets; release everything reclaimable and retry once on OOM.

    A pool can hold enough bytes in aggregate and still refuse a 1.6 GiB block, and
    the reclaimable memory in the process is usually somebody else's table.  A
    second refusal means the geometry genuinely does not fit here, so the caller
    takes the generic scatter loop instead of dying.
    """
    for attempt in (0, 1):
        try:
            return _build_triangles(theta, L, spin, want_theta, want_ell)
        except jax.errors.JaxRuntimeError as exc:
            if "RESOURCE_EXHAUSTED" not in str(exc):
                raise
        _CACHE.clear()
        gc.collect()
        from . import _theta_matrix
        _theta_matrix.release()
    return None


def _blocked(array):
    """Concrete built array, or ``None`` when traced (grad/transpose context)."""
    if not hasattr(array, "block_until_ready"):
        return None
    array.block_until_ready()
    return array


def _pool_headroom():
    """Bytes free in the active JAX pool, or +inf when the device won't say."""
    try:
        stats = jax.local_devices()[0].memory_stats()
    except Exception:  # pragma: no cover - backend without statistics
        return float("inf")
    pool = stats.get("pool_bytes") or 0
    if not pool:
        return float("inf")
    return pool - stats.get("bytes_in_use", 0)


def slabs_for(theta, *, L, spin, nside):
    """``(analysis, synthesis)`` block sets for this geometry, or ``(None, None)``.

    Both are tuples of per-``m``-window triangles: a window stores only
    ``ell >= lo``, the head below that being identically zero.  Analysis is always
    ``(m, ell, theta)`` so it reduces over its contiguous axis.  Synthesis prefers
    ``(m, theta, ell)`` for the same reason and gets it while both layouts fit --
    9.7 GiB at nside 256 against the 18 GiB full-slab pair, at the same speed in
    both directions (0.95-1.01x).  When a single triangle is all that fits (nside
    512: 37 GiB against a 71 GiB pool) both slots hold the theta-contiguous layout
    and synthesis pays 2.1x reducing over a strided axis (19.2 vs 9.3 ms at nside
    256).  ``(None, None)`` means the caller keeps the generic scatter
    loop: too large, or a trace context with nothing concrete to cache.
    """
    tri = triangle_bytes(nside, L)
    if tri > _SLAB_BUDGET:
        return None, None
    want_ell = 2 * tri <= _SLAB_BUDGET
    key = (nside, L, spin, ELL_CONTIG if want_ell else THETA_CONTIG)
    cached = _CACHE.get(key)
    if cached is not None:
        # A resident layout is free to hand back.  Gating it would decline the very
        # calls that already paid the build: the pool is by definition short by the
        # size of the layout it is holding, so the gate fires on every call after
        # the first and the pipeline silently reverts to the scatter loop.
        return cached
    need = (2 * tri if want_ell else tri) + _BUILD_RESERVE
    if need > _pool_headroom():
        # The other large resident table in GMaster is the scalar Legendre band, up
        # to 19 GiB at Nside 512.  It rebuilds on demand; the generic polar path
        # costs more than that rebuild by a wide margin, so take its room rather
        # than decline.  If that still is not enough, decline: the generic path is
        # slow but never out of memory.
        from . import _theta_matrix
        _theta_matrix.release()
        if need > _pool_headroom():
            return None, None
    theta = jnp.asarray(theta)
    layouts = _build_or_none(theta, L, spin, True, want_ell)
    if layouts is None:
        return None, None
    analysis = layouts[THETA_CONTIG]
    if _blocked(analysis[0]) is None:
        # Inside a transpose/grad trace: nothing concrete to cache, so let the
        # caller fall back to the generic path rather than bake a constant in.
        return None, None
    synthesis = layouts[ELL_CONTIG] if want_ell else analysis
    if len({k[:3] for k in _CACHE}) >= _MAX_GEOMETRIES:
        _CACHE.clear()
    _CACHE[key] = (analysis, synthesis)
    return _CACHE[key]


def clear_cache():
    _CACHE.clear()


def forward_latitudinal(ftm, slab, *, L):
    """flm[ell, L-1+m] = sum_theta slice[theta, ell, m] * ftm[theta, L+m].

    ``slab`` is the ``(m, ell, theta)`` analysis block set from :func:`slabs_for`:
    one block per entry of :func:`_windows`, each already sliced to its
    ``ell >= lo`` triangle.  Below that bound the Wigner-d slice vanishes (the
    triangle condition on ``d^ell_{m,-spin}``), so the skipped output rows keep the
    zero they already hold and the result is bit-identical to the full slab.
    """
    rhs = jnp.asarray(ftm)[:, 1:]
    out = jnp.zeros((L, 2 * L - 1), dtype=jnp.result_type(slab[0], rhs))
    for (r0, r1, lo), block in zip(_windows(L), slab):
        out = out.at[lo:, r0:r1].set(
            jnp.einsum("cet,tc->ec", block, rhs[:, r0:r1], optimize=True))
    return out


def inverse_latitudinal(flm, slab, *, L):
    """Transpose of :func:`forward_latitudinal`; ftm is padded to 2L columns.

    Diagonal in ``m`` like the forward step: column ``m + L - 1`` of ``flm`` feeds
    column ``m + L`` of the padded ``ftm`` (the same +1 padding offset), and the
    identically-zero ``ell < |m|`` head is dropped either way.

    ``slab`` is the ``(m, theta, ell)`` synthesis block set, or the
    ``(m, ell, theta)`` analysis blocks when a single layout serves both
    directions -- there the reduction runs over a strided axis and costs 2.1x
    (19.2 vs 9.3 ms at nside 256).
    """
    alm = jnp.asarray(flm)
    theta_contig = slab.layout == THETA_CONTIG
    ntheta = slab[0].shape[2 if theta_contig else 1]
    out = jnp.zeros((ntheta, 2 * L), dtype=jnp.result_type(slab[0], alm))
    subscripts = "cet,ec->tc" if theta_contig else "cte,ec->tc"
    for (r0, r1, lo), block in zip(_windows(L), slab):
        out = out.at[:, r0 + 1:r1 + 1].set(
            jnp.einsum(subscripts, block, alm[lo:, r0:r1], optimize=True))
    return out
