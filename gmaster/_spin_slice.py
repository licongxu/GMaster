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

Two independent shrinkages make the table fit.  Storing only the ``ell >= |m|``
triangle of each layout -- the head below it is identically zero -- costs nothing in
speed (12.1 vs 11.5 ms on analysis, 9.3 vs 9.3 ms on synthesis at nside 256) and saves
1.85x of the bytes.  Storing only *non-negative orders* then halves what is left:
for every order except ``m = 0``,

    d^l_{m,-spin}(pi - theta) = (-1)**(l + spin) * d^l_{-m,-spin}(theta)

measured on the built slice over every row at spin 1 and 2 (worst per-row relative
residue 2.3e-12).  The HEALPix ring grid is symmetric about pi/2 to 4e-15, so
``theta -> pi - theta`` is exactly the ring reversal ``i -> ntheta-1-i``, and one
stored row serves both ``+m`` and ``-m`` -- see :func:`forward_latitudinal`.
``m = 0`` is its own mirror and does *not* satisfy the relation (residue 1.33 at spin
2, 2.00 at spin 1), which is why an earlier whole-table residue test recorded this
shortcut as absent.  The reconstruction is not free internally: against a float128
oracle the full slab is 8e-16 / 1e-15 (analysis / synthesis) and the reconstructed
path 1.5e-12 / 1.2e-11, because the two halves of the s2fft march agree only to
~1e-13.  End to end it is invisible -- the pipeline's ``rel`` against NaMaster is
identical either way.

Net at nside 512 (lmax = 3*nside): a layout pair is 37.48 GiB instead of the 143.9 GiB
two full slabs need, and at nside 256 it is 4.87 GiB instead of 18.  Both layouts still
fit through 512; 640 gets one layout (36.31 GiB); 768 (62.42) and 1024 (146.96) are
declined, so those sizes keep the generic scatter loop -- the table is cubic in nside
and no further shrinking reaches 1024 on one card.  When only one layout fits, synthesis
pays 2.1x reducing over a strided axis (19.2 vs 9.3 ms at nside 256) -- still ~10x
cheaper than the scatter loop it replaces.  See :func:`slabs_for`.
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
# Geometries whose build the pool refused.  Without this every latitudinal pass of a
# pipeline re-attempts a multi-gigabyte build that just failed: the Nside 1024
# float32 spin-2 pipeline spent 142 s per rep (.qwen/tmp/score_n1024_spin2_98.log,
# ten `Allocator ran out of memory trying to allocate 2.69GiB` warnings, one block of
# one window) instead of taking the generic loop in 6 s.  Dropped by `clear_cache`,
# which is the call that reclaims the room.
_BUILD_FAILED = set()
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
# Chunks built before the compilation caches are dropped mid-build.  The march is one
# jit reused over every chunk, and a cached executable pins the buffers it last
# produced, so without this two chunk tables are alive at once: the Nside 1024
# float32 layout measured a peak of 86.2 GiB against 73.5 declared, and the excess is
# almost exactly two 6 GiB float64 chunk tables.
_BUILD_CLEAR_EVERY = 8
# Headroom required on top of the layout before the build is approved: the chunk
# table and its per-window copies are alive while the finished blocks accumulate
# (measured peak/declared: 41.0/37.5 at nside 512, 86.2/73.5 at Nside 1024 float32 --
# about two float64 chunk tables), the rest of the pipeline holds maps and coupling
# matrices, and a pool that is nominally large enough but fragmented still refuses a
# 1.6 GiB block.  It is also what keeps Nside 1024 spin 2 off a pool it cannot share:
# 73.48 GiB in float32, and every pool tried so far refuses the build -- fractions
# 0.95 (90.2 GiB) and 0.98 (93.0 GiB), `_BUILD_RESERVE` 16/10/6,
# `TF_GPU_ALLOCATOR=cuda_malloc_async`, `PREALLOCATE=false` -- always on a ~2.7 GiB
# request (exactly one joined window block) with the peak at 86.8 GiB and
# `largest_free_block_bytes: 0`. The pipeline then runs the generic polar loop at 87 s:
# every logged Nside 1024 spin-2 cell is 0.02-0.04x
# (.qwen/tmp/score_n1024_spin2_halfmarch.log).
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
    """``(m0, m1, lo)`` per stored window of *non-negative* orders.

    Only orders ``0..L-1`` are stored.  ``T[m, pi-theta, ell] == (-1)**(ell - m') *
    T[-m, theta, ell]`` -- measured over every row at spin 1 and 2, exact to
    2.3e-12 relative -- makes the negative rows redundant, and each stored row is
    then contracted twice in one pass: straight, and against the sky map with its
    rings reversed (see :func:`forward_latitudinal`).  ``lo = m0`` because within a
    window of non-negative orders the smallest ``|m|`` is the first one, and
    ``d^ell_{m,-spin}`` vanishes below ``ell = |m|``.
    """
    cached = _WINDOW_CACHE.get(L)
    if cached is None:
        cached = tuple((a, min(a + _M_BLOCK, L), a) for a in range(0, L, _M_BLOCK))
        _WINDOW_CACHE[L] = cached
    return cached


def _sign(L, spin):
    """``(-1)**(ell - m')`` with ``m' = -spin``: the theta -> pi-theta sign."""
    return 1.0 - 2.0 * ((jnp.arange(L) + abs(spin)) % 2).astype(jnp.float64)


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
    """(L, ntheta, L) slice of the Wigner-d matrix at m' = -spin, orders ``m >= 0``.

    Row ``m`` holds ``d^l_{m,-spin}(theta)`` for every theta and ell.  The negative
    orders are not marched: :func:`forward_latitudinal` reconstructs them with
    ``T[m, pi-theta, ell] == (-1)**(ell-spin) * T[-m, theta, ell]``, so the old
    ``(2L-1, ...)`` table's first ``L-1`` rows were never read by
    :func:`_build_triangles`, which slices the non-negative window rows.  Dropping
    that march halves the build work and the chunk table -- at Nside 1024 float32,
    3 GiB off a peak that was refusing the layout.  Row 0 is the m = 0 row, which
    both marches produce; as before it is the ``which=1`` copy, matching the generic
    path's loop order.
    """
    mm = -spin
    el = jnp.arange(L, dtype=jnp.float64)
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    lrenorm, vsign, cpi, cp2, indices = generate_precomputes_jax(
        L, spin, "healpix", None, True, 0, betas=theta
    )
    trig = (c, s, 1.0 - c, el)
    return _march(trig, el - mm + 1.0, cpi, cp2, vsign[::-1][:L], lrenorm[1],
                  indices, L, 1)


def _table_bytes(ntheta, L):
    return L * ntheta * L * 8


def triangle_bytes(nside, L, dtype=jnp.float64):
    """Bytes of one windowed layout: one block per m-window holding ``ell >= lo``.

    ``dtype`` is the table *storage* precision (`set_table_precision`), which is
    what the fit test has to count; the march that generates the values is always
    float64.
    """
    ntheta = 4 * nside - 1
    itemsize = jnp.dtype(dtype).itemsize
    return sum((m1 - m0) * (L - lo) * ntheta * itemsize
               for m0, m1, lo in _windows(L))


class _BlockSet(tuple):
    """Tuple of per-window blocks tagged with the axis order they are stored in.

    Subclasses ``tuple`` so caching, indexing and iteration all behave normally;
    the layout tag is what lets :func:`inverse_latitudinal` pick the right
    contraction when a single layout has to serve both directions.
    """

    def __new__(cls, blocks, layout, spin):
        self = super().__new__(cls, blocks)
        self.layout = layout
        self.spin = spin
        return self


# The block set travels through `jax.jit` boundaries as an argument, so it has to
# be a container of array leaves (like the single slab it replaces) with the layout
# and spin as aux data.  An unregistered tuple subclass is treated as a non-array
# leaf and the trace rejects it.
jax.tree_util.register_pytree_node(
    _BlockSet,
    lambda slab: (tuple(slab), (slab.layout, slab.spin)),
    lambda aux, children: _BlockSet(children, aux[0], aux[1]),
)


def _build_triangles(theta, L, spin, want_theta, want_ell, store=jnp.float64):
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
    for i_chunk, start in enumerate(range(0, ntheta, per)):
        table = _build(theta[start:start + per], L, spin)  # (m, theta, ell)
        nvalid = min(per, ntheta - start)  # the padded tail rings are dropped
        pieces = []
        for i, (m0, m1, lo) in enumerate(wins):
            # Row m holds order m; the negative half is reconstructed from it.
            sub = table[m0:m1, :nvalid, lo:]
            if want_theta:
                # The cast and the transpose go in one kernel: an fp32 copy of the
                # slice followed by a transposed copy of *that* is two buffers per
                # window per chunk, and it is the second one that made the Nside
                # 1024 float32 build peak ~12 GiB above the layout it was writing.
                # `astype` materialises, so the fp32 path needs no `+ 0.0`; the
                # fp64 path does, because a bare transpose can stay a view.
                out = sub.transpose(0, 2, 1)
                out = out.astype(store) if store != jnp.float64 else out + 0.0
                theta_pieces[i].append(out)
                pieces.append(out)
            if want_ell:
                out = sub.astype(store) if store != jnp.float64 else sub + 0.0
                ell_pieces[i].append(out)
                pieces.append(out)
        del table
        # Dispatch is asynchronous, so without this the chunk tables of every
        # chunk stay alive at once -- tens of GiB of live buffer at nside 512
        # instead of one chunk's worth on top of the finished blocks.
        for arr in pieces:
            arr.block_until_ready()
        # ... and a cached executable pins the buffers it produced, so with `chunks`
        # programs alive the peak is the layout plus a second copy of every chunk:
        # 86.2 GiB against the 73.5 GiB declared at Nside 1024 float32.  The program
        # is identical from chunk to chunk (uniform lengths, static L/spin only), so
        # dropping it periodically costs one recompile per clear and returns the peak
        # to the layout plus one chunk.  Same trick as `_theta_matrix._band`.
        if chunks > _BUILD_CLEAR_EVERY and (i_chunk + 1) % _BUILD_CLEAR_EVERY == 0:
            jax.clear_caches()

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
        # Drop the chunk programs before the join.  A cached executable keeps the
        # buffers it produced alive, and here those buffers are the *pieces* of the
        # layout: at Nside 1024 float32 the pieces alone are 73.5 GiB, so the first
        # window join (`2.69 GiB`, `.qwen/tmp/score_n1024_spin2_98.log`, peak 87.7 of
        # a 93.0 GiB pool, `largest_free_block_bytes: 0`) is refused while 73.5 GiB of
        # data nothing will read again sits pinned.
        jax.clear_caches()
        out[THETA_CONTIG] = _BlockSet(join(theta_pieces, 2), THETA_CONTIG, spin)
    if want_ell:
        out[ELL_CONTIG] = _BlockSet(join(ell_pieces, 1), ELL_CONTIG, spin)
    return out


def _build_or_none(theta, L, spin, want_theta, want_ell, store=jnp.float64):
    """Build the block sets; release everything reclaimable and retry once on OOM.

    A pool can hold enough bytes in aggregate and still refuse a 1.6 GiB block, and
    the reclaimable memory in the process is usually somebody else's table.  A
    second refusal means the geometry genuinely does not fit here, so the caller
    takes the generic scatter loop instead of dying.
    """
    for attempt in (0, 1):
        try:
            return _build_triangles(theta, L, spin, want_theta, want_ell, store)
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
    from . import utils

    store = utils.table_dtype()
    tri = triangle_bytes(nside, L, store)
    if tri > _SLAB_BUDGET:
        return None, None
    want_ell = 2 * tri <= _SLAB_BUDGET
    key = (nside, L, spin, ELL_CONTIG if want_ell else THETA_CONTIG, store)
    if key in _BUILD_FAILED:
        return None, None
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
    layouts = _build_or_none(theta, L, spin, True, want_ell, store)
    if layouts is None:
        _BUILD_FAILED.add(key)
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
    _BUILD_FAILED.clear()


def _reduce_channels(prod, axis):
    """Sum a list of same-shaped products along ``axis``, one accumulator each.

    Spelling the channel contraction as ``sum(block[..., None] * rhs[:, None, :, :],
    axis=2)`` leaves the channels in a dimension *after* the axis being reduced, and
    XLA then materialises the whole ``(m, ell, theta, 4)`` product instead of folding
    the multiply into the reduce -- a write and a read of the product on top of the
    block read.  One ``lax.reduce`` over a tuple of accumulators reads the block once
    and nothing wide is ever materialised: the Nside 256 spin-2 ``field`` stage goes
    64.9 -> 47.7 ms (1.36x, three interleaved rounds inside the production stage,
    control 1430 GB/s, alms agree to 5.4e-16).  This is the polar twin of
    ``_theta_matrix._contract_theta``, which wins 1.50x on the scalar band.

    ``einsum`` is not the answer: it lowers this batched matvec to 717 GB/s (HANDOFF
    session 8), and it loses to both of the forms above.
    """
    return jnp.stack(lax.reduce(tuple(prod), (0.0,) * len(prod),
                                lambda a, b: tuple(x + y for x, y in zip(a, b)),
                                (axis,)), axis=-1)


def forward_latitudinal(ftm, slab, *, L):
    """flm[ell, L-1+m] = sum_theta slice[theta, ell, m] * ftm[theta, L+m].

    ``slab`` is the ``(m, ell, theta)`` analysis block set from :func:`slabs_for`:
    one block per entry of :func:`_windows`, non-negative orders only, each already
    sliced to its ``ell >= lo`` triangle.  Splitting the theta sum at pi/2 and
    applying ``T[m, pi-theta, ell] = (-1)**(ell - m') T[-m, theta, ell]`` turns the
    negative orders into a second contraction of the *same* block against the sky
    map with its rings reversed and its orders negated:

        flm[ell, L-1+m] = sum_theta T[m, theta, ell] * ftm[theta, L+m]
        flm[ell, L-1-m] = (-1)**(ell - m') * sum_theta T[m, theta, ell]
                                                    * ftm[pi-theta, L-m]

    Both channels read the same block, so half the table produces all of ``flm``.
    Order ``m = 0`` is its own mirror and does *not* satisfy the relation (measured),
    so only the direct channel writes that column.  Below ``ell = |m|`` the slice
    vanishes, so the skipped output rows keep the zero they already hold.

    The contraction is a multiply-and-reduce over four *real* right-hand sides
    (direct re, direct im, mirror re, mirror im) rather than an ``einsum`` over the
    complex map.  ``einsum`` lowers the per-window batched matvec to 717 GB/s here,
    and every alternative spelling of it -- one call over a stacked ``h`` axis
    (nside 512 field 694 -> 1392 ms), the same two calls without the scatter chain
    -- measured no better or worse.  The four real channels then have to be reduced
    through ``_reduce_channels``, not a stacked broadcast: that second choice is
    worth 1.36x on the ``field`` stage at nside 256 (64.9 -> 47.7 ms) at 5.4e-16.
    The channel split itself is bit-for-bit the same answer as ``einsum``
    (``|new - einsum| = 0.0`` at nside 128 against the full slab and at 512 against
    shipped).
    """
    ftm = jnp.asarray(ftm)
    off = L - 1
    rev = ftm[::-1]  # ring i of rev is the ring at pi - theta_i
    sign = _sign(L, slab.spin)
    out = jnp.zeros((L, 2 * L - 1), dtype=jnp.result_type(slab[0], ftm))
    for (m0, m1, lo), block in zip(_windows(L), slab):
        # Table storage may be fp32 (`set_table_precision`); the accumulation is not.
        block = lax.convert_element_type(block, jnp.float64)
        direct = ftm[:, L + m0:L + m1]
        mirror = rev[:, L - m1 + 1:L - m0 + 1][:, ::-1]
        rhs = jnp.stack([direct.real.T, direct.imag.T,
                         mirror.real.T, mirror.imag.T], axis=-1)  # (m, theta, 4)
        acc = _reduce_channels([block * rhs[:, None, :, c] for c in range(4)], 2)
        out = out.at[lo:, off + m0:off + m1].set(
            (acc[..., 0] + 1j * acc[..., 1]).T)
        mir = sign[lo:, None] * (acc[..., 2] + 1j * acc[..., 3]).T
        if m0:
            out = out.at[lo:, off - m1 + 1:off - m0 + 1].set(mir[:, ::-1])
        else:
            # Column off belongs to the direct channel alone.
            out = out.at[lo:, off - m1 + 1:off].set(mir[:, 1:][:, ::-1])
    return out


def inverse_latitudinal(flm, slab, *, L):
    """Transpose of :func:`forward_latitudinal`; ftm is padded to 2L columns.

    Reading the forward map as a sum of two contractions over one block gives the
    adjoint directly: the direct channel writes ``ftm[:, L+m]`` and the mirrored
    channel writes the ring-reversed ``ftm[:, L-m]``.  The two channels touch
    disjoint columns except at ``m = 0``, which again only the direct one writes.

    ``slab`` is the ``(m, theta, ell)`` synthesis block set, or the
    ``(m, ell, theta)`` analysis blocks when a single layout has to serve both
    directions -- there the reduction runs over a strided axis and costs 2.1x
    (19.2 vs 9.3 ms at nside 256).  Spelled as one multiply-and-reduce over four
    real right-hand sides for the same reason as :func:`forward_latitudinal`
    (57.4 -> 34.0 ms at nside 512, identical output).
    """
    alm = jnp.asarray(flm)
    off = L - 1
    theta_contig = slab.layout == THETA_CONTIG
    ntheta = slab[0].shape[2 if theta_contig else 1]
    sign = _sign(L, slab.spin)
    out = jnp.zeros((ntheta, 2 * L), dtype=jnp.result_type(slab[0], alm))
    for (m0, m1, lo), block in zip(_windows(L), slab):
        block = lax.convert_element_type(block, jnp.float64)
        direct = alm[lo:, off + m0:off + m1]
        mirror = sign[lo:, None] * alm[lo:, off - m1 + 1:off - m0 + 1][:, ::-1]
        rhs = jnp.stack([direct.real, direct.imag,
                         mirror.real, mirror.imag], axis=-1)
        rhs = rhs.transpose(1, 0, 2)  # (m, ell, 4)
        if theta_contig:  # block (m, ell, theta): reduce over the strided axis
            acc = _reduce_channels([block * rhs[:, :, c, None] for c in range(4)], 1)
        else:  # block (m, theta, ell)
            acc = _reduce_channels([block * rhs[:, None, :, c] for c in range(4)], 2)
        out = out.at[:, L + m0:L + m1].set((acc[..., 0] + 1j * acc[..., 1]).T)
        mir = (acc[..., 2] + 1j * acc[..., 3]).T
        # The mirrored channel reads ftm ring-reversed, so its adjoint writes back
        # ring-reversed as well.
        if m0:
            out = out.at[:, L - m1 + 1:L - m0 + 1].set(mir[::-1][:, ::-1])
        else:
            out = out.at[:, L - m1 + 1:L].set(mir[::-1][:, 1:][:, ::-1])
    return out
