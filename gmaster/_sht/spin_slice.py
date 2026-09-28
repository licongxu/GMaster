"""Spin-s latitudinal transform with precomputed Wigner-d slabs (small band limits).

The generic s2fft latitudinal step recurses over ``m`` in a ``lax.fori_loop`` and
scatters each ``m`` slice with a dynamic row index, which is scatter- and
latency-bound.  On HEALPix the step is a *single* contraction against the
``m' = -spin`` slice of the Wigner-d matrix (one ``m'``, no sum over ``m'``):

    flm[ell, L-1+m] = sum_theta slice[theta, ell, m] * ftm[theta, L+m]

(the ``+1`` column offset on ``ftm`` is s2fft's HEALPix Fourier padding).  This
agrees with ``healpix._forward_latitudinal`` to <1e-15 for spin +/-1 and +/-2,
and its transpose reproduces the synthesis step to the same accuracy.  The slice
is built once per geometry with a scatter-free ``lax.scan`` and cached on the
device, so each transform is one memory-bound contraction (roughly 15-25x
cheaper per call than the generic step).

Layouts.  The contraction is diagonal in ``m`` (a batched matvec, not a GEMM),
so speed is set by which axis is innermost: analysis reduces over ``theta`` and
synthesis over ``ell``.  Two layouts are kept when memory allows:
``THETA_CONTIG`` ``(m, ell, theta)`` for analysis and ``ELL_CONTIG``
``(m, theta, ell)`` for synthesis.

Two storage reductions make the tables fit:

* Only the ``ell >= |m|`` triangle is stored, per window of orders (the rows
  below are identically zero): about 1.85x fewer bytes at no speed cost.
* Only non-negative orders are stored.  For every ``m != 0``

      d^l_{m,-spin}(pi - theta) = (-1)**(l + spin) * d^l_{-m,-spin}(theta)

  (checked on the built slice to 2.3e-12 relative per row at spin 1 and 2).  The
  HEALPix rings are symmetric about pi/2 to 4e-15, so ``theta -> pi - theta`` is
  the ring reversal ``i -> ntheta-1-i`` and one stored row serves both ``+m`` and
  ``-m`` (see :func:`forward_latitudinal`).  ``m = 0`` is its own mirror and does
  *not* satisfy the relation, so only the direct channel writes it.  Against a
  float128 reference the reconstructed half is accurate to ~1e-12 (vs ~1e-15 for
  a full slab), because the two halves of the s2fft recursion agree only to
  ~1e-13; the effect on the final power spectra is negligible.

Memory (lmax = 3*nside, float64 storage): both layouts together are 4.9 GiB at
nside 256 and 37.5 GiB at nside 512; nside 640 gets one layout (36.3 GiB); nside
768 (62.4 GiB per layout) and above are declined and use the generic scatter
loop.  The table grows as nside^3.  When only one layout fits, synthesis reduces
over a strided axis at about 2.1x the cost, still ~10x cheaper than the scatter
loop.  See :func:`slabs_for`.
"""

import gc
import os
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
from jax import lax

from s2fft.recursions.price_mcewen import generate_precomputes_jax

from gmaster._sht.cuda_gpu import on_cuda_gpu

THETA_CONTIG = "theta_contig"  # (m, ell, theta) -- analysis reduces over theta
ELL_CONTIG = "ell_contig"  # (m, theta, ell) -- synthesis reduces over ell

# Contraction used by the latitudinal step, from GMASTER_POLAR_CONTRACT.  "pallas"
# (default) reads each block once and holds the four real channels in register
# accumulators (`gmaster._sht.spin_contract`); "xla" uses the multiply-and-reduce
# form.  "pallas" applies wherever :func:`_kernel_contract` allows (float32 tables on
# an NVIDIA GPU), and there it has no silent fallback: the kernel runs or the call
# raises, so a kernel failure is never masked.  Use GMASTER_POLAR_CONTRACT=xla or
# `set_polar_contract("xla")` to select the XLA form.
_CONTRACT = os.environ.get("GMASTER_POLAR_CONTRACT", "pallas").strip().lower()


def set_polar_contract(name):
    """Select the latitudinal contraction, ``"pallas"`` (default) or ``"xla"``; return the previous one."""
    global _CONTRACT
    name = name.strip().lower()
    if name not in ("pallas", "xla"):
        raise ValueError(
            f"GMaster polar contraction must be 'pallas' or 'xla', got {name!r}")
    previous, _CONTRACT = _CONTRACT, name
    return previous


def polar_contract():
    """The configured polar contraction."""
    return _CONTRACT


@lru_cache(maxsize=1)
def _pallas_ok():
    """True on a CUDA GPU (some devices, e.g. ``Tesla T4``, omit ``NVIDIA``)."""
    return on_cuda_gpu()


def _kernel_contract(slab, *, synthesis=False):
    """Whether the fused Triton contraction should serve this block set.

    All of the following must hold:

    * **float32 storage.**  The kernel is bandwidth-oriented; with float64 tables
      Triton's float64 path is slower than XLA's (about 0.85x on analysis and
      0.66x on synthesis), so the default float64 configuration keeps the XLA
      form and the kernel is used with ``set_table_precision("fp32")``.
    * **an NVIDIA GPU** (Triton), see :func:`_pallas_ok`.
    * **for synthesis, the ``(m, theta, ell)`` layout.**  The kernel reduces along the
      block's last axis; when :func:`slabs_for` could afford only the analysis layout,
      the strided reduction stays with XLA.
    """
    if _CONTRACT != "pallas" or not _pallas_ok():
        return False
    if slab[0].dtype != jnp.float32:
        return False
    return not synthesis or slab.layout == ELL_CONTIG


_CACHE = {}
# Geometries whose build ran out of device memory.  Without this every latitudinal
# pass of a pipeline would retry a multi-gigabyte build that just failed instead of
# going straight to the generic loop.  Cleared by `clear_cache`.
_BUILD_FAILED = set()
_MAX_GEOMETRIES = 2
# Budget for the stored tables.  One windowed layout (float64, lmax = 3*nside) is
# about 2.4 GiB at nside 256, 18.7 at 512, 36.3 at 640 and 62.4 at 768.  Both
# layouts are built while the pair fits; a single layout is still far faster than
# the generic scatter loop, so the budget admits one layout at 640 and refuses 768.
_SLAB_BUDGET = 56 * 1024**3
# Target size of the transient table built per theta chunk (the march cannot be
# split along ``m``; see `_build_triangles`).  Smaller targets fragment the pool into
# many small blocks so that the per-window join later fails to find contiguous space;
# 6 GiB keeps the nside 512 build peak at about 1.1x the layout size.
_BUILD_CHUNK_TARGET = 6 * 1024**3
# Chunks built between `jax.clear_caches()` calls.  The march is one jit reused for
# every chunk and a cached executable pins the buffers it last produced, so without
# periodic clearing extra chunk tables stay alive and raise the peak.
_BUILD_CLEAR_EVERY = 8
# Free device memory required on top of the layout before a build is attempted.  It
# covers the chunk table and its per-window copies during the build (the peak is
# about the layout plus two float64 chunk tables), the rest of the pipeline (maps,
# coupling matrices), and fragmentation: a pool with enough free bytes in total can
# still refuse a single block of a few GiB.
_BUILD_RESERVE = 16 * 1024**3
# Width of the m-windows of the stored slabs (GMASTER_M_BLOCK).  ``d^ell_{m,-spin}``
# vanishes for ``ell < |m|`` (just under half the full slab), and all rows in a
# window share ``lo = min |m|``, so the contraction reads only ``ell`` in ``[lo, L)``
# (see :func:`forward_latitudinal`).  64 skips 42% of the bytes (about 1.6x faster);
# 16 skips 48% but the 4x kernel count makes it slower overall.
_M_BLOCK = int(os.environ.get("GMASTER_M_BLOCK", "64"))
# Order-window width for the table-free marched routes (GMASTER_MARCH_M_BLOCK).  With
# no stored slab there are no bytes to skip; the width only sets how the programs,
# one per ``(m, theta tile)``, are split across ``L/mb`` launches.  128 is best up to
# a per-launch program ceiling: 256 doubles the padded per-window right-hand side and
# 32 multiplies the launches.  Both are slower.  See
# :func:`gmaster._sht.spin_march._march_windows`.  (The stored-slab width above is a
# separate choice, where a wider window reads more of the zero ``ell < |m|`` region.)
_MARCH_M_BLOCK = int(os.environ.get("GMASTER_MARCH_M_BLOCK", "128"))
# Upper bound on the order window of the folded spin-0 *synthesis* launch
# (GMASTER_MARCH_M_SYNTH0_MAX).  That route has few theta tiles (`_ST0` gives 4 at
# nside 2048, 8 at 4096), so a wider window is needed to fill the program ceiling:
# about 1.14x at nside 2048 and 1.06x at 4096, with unchanged alms.  Raising
# `_MARCH_M_BLOCK` instead would not help, because the analysis route then exceeds
# the ceiling and falls back to `_M_BLOCK`.  See
# :func:`gmaster._sht.spin_march._synth_windows`.
_MARCH_M_SYNTH0_MAX = int(os.environ.get("GMASTER_MARCH_M_SYNTH0_MAX", "512"))
_WINDOW_CACHE = {}


def _windows(L, block=None):
    """``(m0, m1, lo)`` per stored window of *non-negative* orders.

    Only orders ``0..L-1`` are stored: ``T[m, pi-theta, ell] == (-1)**(ell - m') *
    T[-m, theta, ell]`` (``m != 0``) makes the negative rows redundant, and each
    stored row is contracted twice in one pass, directly and against the
    ring-reversed map (see :func:`forward_latitudinal`).  ``lo = m0`` because the
    first order in a window has the smallest ``|m|`` and ``d^ell_{m,-spin}``
    vanishes below ``ell = |m|``.

    ``block`` overrides the width for the table-free marches
    (:data:`_MARCH_M_BLOCK`); every stored layout uses :data:`_M_BLOCK` so that the
    builder, the budget and the contraction agree.
    """
    block = _M_BLOCK if block is None else block
    cached = _WINDOW_CACHE.get((L, block))
    if cached is None:
        cached = tuple((a, min(a + block, L), a) for a in range(0, L, block))
        _WINDOW_CACHE[(L, block)] = cached
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

    Row ``m`` holds ``d^l_{m,-spin}(theta)`` for every theta and ell.  Negative
    orders are not marched, since :func:`forward_latitudinal` reconstructs them
    with ``T[m, pi-theta, ell] == (-1)**(ell-spin) * T[-m, theta, ell]``; this
    halves the build work and the chunk table.  Row 0 (m = 0) is taken from the
    ``which=1`` march, matching the generic path's loop order.
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

    ``dtype`` is the table *storage* precision (`set_table_precision`); the march
    that generates the values is always float64.
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


# The block set is passed through `jax.jit` boundaries, so it is registered as a
# pytree of array leaves with layout and spin as aux data; an unregistered tuple
# subclass would be treated as a non-array leaf and rejected by the trace.
jax.tree_util.register_pytree_node(
    _BlockSet,
    lambda slab: (tuple(slab), (slab.layout, slab.spin)),
    lambda aux, children: _BlockSet(children, aux[0], aux[1]),
)


def _build_triangles(theta, L, spin, want_theta, want_ell, store=jnp.float64):
    """Per-m-window triangles (``ell >= lo``) in the requested layouts.

    Returns a dict mapping `THETA_CONTIG` and/or `ELL_CONTIG` to a `_BlockSet`.

    The march recurses over ``m``, so a window's rows need all their predecessors
    and each call produces the full slab.  The recurrence is independent per ring,
    so the build walks theta in chunks and copies each window's triangle out
    immediately: peak memory is the accumulated blocks plus about two chunk tables
    rather than the full slab plus its transpose.  Chunks have uniform length so
    the march compiles once.

    ``+ 0.0`` forces a real buffer: a bare transpose can remain a view, which the
    contraction would then read strided.
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
                # Transpose first, then cast, so only one new buffer is created per
                # window per chunk.  `astype` materialises, so the fp32 path needs
                # no `+ 0.0`; the fp64 path does, because a bare transpose can stay
                # a view.
                out = sub.transpose(0, 2, 1)
                out = out.astype(store) if store != jnp.float64 else out + 0.0
                theta_pieces[i].append(out)
                pieces.append(out)
            if want_ell:
                out = sub.astype(store) if store != jnp.float64 else sub + 0.0
                ell_pieces[i].append(out)
                pieces.append(out)
        del table
        # Dispatch is asynchronous; without this wait every chunk's table could be
        # alive at once instead of one chunk on top of the finished blocks.
        for arr in pieces:
            arr.block_until_ready()
        # A cached executable pins the buffers it produced.  The program is the same
        # for every chunk (uniform lengths, static L/spin), so clearing periodically
        # costs one recompile per clear and keeps the peak at the layout plus one
        # chunk.  Same approach as `theta_matrix._band`.
        if chunks > _BUILD_CLEAR_EVERY and (i_chunk + 1) % _BUILD_CLEAR_EVERY == 0:
            jax.clear_caches()

    def join(parts, axis):
        """Join each window's chunks, releasing them as soon as the join completes.

        Joining all windows before releasing would hold the chunks and the
        finished blocks at once, i.e. twice the layout.
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
        # Drop the chunk programs before the join: their cached executables pin
        # the pieces being joined, which would otherwise stay alive alongside the
        # joined blocks and can make the join run out of memory.
        jax.clear_caches()
        out[THETA_CONTIG] = _BlockSet(join(theta_pieces, 2), THETA_CONTIG, spin)
    if want_ell:
        out[ELL_CONTIG] = _BlockSet(join(ell_pieces, 1), ELL_CONTIG, spin)
    return out


def _build_or_none(theta, L, spin, want_theta, want_ell, store=jnp.float64):
    """Build the block sets; release everything reclaimable and retry once on OOM.

    A pool can have enough free bytes in total and still refuse a large block; the
    retry first drops this module's cache and the scalar Legendre band
    (`theta_matrix.release`).  Returns None if the second attempt also fails, so
    the caller uses the generic scatter loop.
    """
    for attempt in (0, 1):
        try:
            return _build_triangles(theta, L, spin, want_theta, want_ell, store)
        except jax.errors.JaxRuntimeError as exc:
            if "RESOURCE_EXHAUSTED" not in str(exc):
                raise
        _CACHE.clear()
        gc.collect()
        from . import theta_matrix as _theta_matrix
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
        # The CPU backend returns None rather than raising.
        stats = jax.local_devices()[0].memory_stats() or {}
    except Exception:  # pragma: no cover - backend without statistics
        return float("inf")
    pool = stats.get("pool_bytes") or 0
    if not pool:
        return float("inf")
    return pool - stats.get("bytes_in_use", 0)


def slabs_for(theta, *, L, spin, nside):
    """``(analysis, synthesis)`` block sets for this geometry, or ``(None, None)``.

    Both are `_BlockSet` tuples of per-``m``-window triangles (only ``ell >= lo``
    is stored).  Analysis is always ``(m, ell, theta)`` so it reduces over its
    contiguous axis.  Synthesis uses ``(m, theta, ell)`` for the same reason when
    both layouts fit `_SLAB_BUDGET`; otherwise both slots hold the analysis layout
    and synthesis reduces over a strided axis (about 2.1x slower).  Results are
    cached per ``(nside, L, spin, layout, storage dtype)``.

    Returns ``(None, None)`` when the caller should use the generic scatter loop:
    the table exceeds the budget or the free device memory, the build ran out of
    memory, or this is a trace context with nothing concrete to cache.
    """
    from .._config import table_dtype

    store = table_dtype()
    tri = triangle_bytes(nside, L, store)
    if tri > _SLAB_BUDGET:
        return None, None
    want_ell = 2 * tri <= _SLAB_BUDGET
    key = (nside, L, spin, ELL_CONTIG if want_ell else THETA_CONTIG, store)
    if key in _BUILD_FAILED:
        return None, None
    cached = _CACHE.get(key)
    if cached is not None:
        # Return a resident layout without the headroom check: the pool is short by
        # exactly the layout it holds, so the check would reject every later call.
        return cached
    need = (2 * tri if want_ell else tri) + _BUILD_RESERVE
    if need > _pool_headroom():
        # Free the scalar Legendre band (the other large resident table; it
        # rebuilds on demand far more cheaply than the generic polar path costs).
        # If there is still not enough room, decline: the generic path is slow but
        # does not run out of memory.
        from . import theta_matrix as _theta_matrix
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
        # Inside a transpose/grad trace: nothing concrete to cache, so the caller
        # uses the generic path rather than baking in a constant.
        return None, None
    synthesis = layouts[ELL_CONTIG] if want_ell else analysis
    if len({k[:3] for k in _CACHE}) >= _MAX_GEOMETRIES:
        _CACHE.clear()
    _CACHE[key] = (analysis, synthesis)
    return _CACHE[key]


def clear_cache():
    """Drop cached block sets and the record of failed builds."""
    _CACHE.clear()
    _BUILD_FAILED.clear()


def _reduce_channels(prod, axis):
    """Sum a list of same-shaped products along ``axis``, one accumulator each.

    Returns the sums stacked on a new last axis.  The broadcast spelling
    ``sum(block[..., None] * rhs[:, None, :, :], axis=2)`` places the channels
    after the reduced axis, and XLA then materialises the full
    ``(m, ell, theta, 4)`` product.  A tuple ``lax.reduce`` reads the block once
    and materialises nothing wide (about 1.36x faster on the spin-2 field stage,
    identical results to 5e-16).  ``einsum`` lowers this batched matvec poorly
    and is slower than either.  Counterpart of `theta_matrix._contract_theta`.
    """
    return jnp.stack(lax.reduce(tuple(prod), (0.0,) * len(prod),
                                lambda a, b: tuple(x + y for x, y in zip(a, b)),
                                (axis,)), axis=-1)


def forward_latitudinal(ftm, slab, *, L):
    """flm[ell, L-1+m] = sum_theta slice[theta, ell, m] * ftm[theta, L+m].

    Parameters
    ----------
    ftm : (ntheta, 2L) complex array
        Azimuthal FFT of the rings (s2fft HEALPix padding).
    slab : _BlockSet
        Analysis block set from :func:`slabs_for`.

    Returns
    -------
    (L, 2L-1) complex array ``flm``.

    When :func:`_kernel_contract` allows (float32 tables on an NVIDIA GPU), this
    runs the fused Triton kernel in :mod:`gmaster._sht.spin_contract`, which reads
    each block once, forms products in the storage precision and accumulates tile
    partials in float64.  It is about 4x faster than the XLA form at nside 256-512,
    with results differing by ~6e-8, the order of the float32 table precision.
    Otherwise (or with ``GMASTER_POLAR_CONTRACT=xla``) the XLA form is used.
    """
    if _kernel_contract(slab):
        from gmaster._sht.spin_contract import forward

        return forward(slab, ftm, L=L)
    return _forward_latitudinal_xla(ftm, slab, L=L)


def _forward_latitudinal_xla(ftm, slab, *, L):
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
    Order ``m = 0`` is its own mirror and does *not* satisfy the relation, so only
    the direct channel writes that column.  Below ``ell = |m|`` the slice
    vanishes, so the skipped output rows keep their zeros.

    The contraction is a multiply-and-reduce over four *real* right-hand sides
    (direct re, direct im, mirror re, mirror im) via :func:`_reduce_channels`,
    rather than an ``einsum`` over the complex map, which XLA lowers poorly here.
    The result is bit-for-bit identical to the ``einsum`` form.
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

    Takes ``flm`` of shape (L, 2L-1) and the synthesis block set from
    :func:`slabs_for`; returns ``ftm`` of shape (ntheta, 2L).  Dispatched like
    :func:`forward_latitudinal`, except that the Triton kernel reduces over the
    block's last axis and so serves only the ``(m, theta, ell)`` layout; when only
    the analysis layout is resident, the strided reduction runs in XLA.
    """
    if _kernel_contract(slab, synthesis=True):
        from gmaster._sht.spin_contract import inverse

        return inverse(slab, flm, L=L)
    return _inverse_latitudinal_xla(flm, slab, L=L)


def _inverse_latitudinal_xla(flm, slab, *, L):
    """Transpose of :func:`forward_latitudinal`; ftm is padded to 2L columns.

    Reading the forward map as a sum of two contractions over one block gives the
    adjoint directly: the direct channel writes ``ftm[:, L+m]`` and the mirrored
    channel writes the ring-reversed ``ftm[:, L-m]``.  The two channels touch
    disjoint columns except at ``m = 0``, which again only the direct one writes.

    ``slab`` is the ``(m, theta, ell)`` synthesis block set, or the
    ``(m, ell, theta)`` analysis blocks when a single layout has to serve both
    directions; there the reduction runs over a strided axis (about 2.1x slower).
    Uses the same four-real-channel multiply-and-reduce as the forward XLA form.
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
