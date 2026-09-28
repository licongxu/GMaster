"""Pallas (Triton) contraction of the spin-weighted Wigner-d slabs.

This is the GPU kernel behind the polarised latitudinal step of
:mod:`gmaster._sht.spin_slice`.  Each ``m`` window of the precomputed Wigner-d
slab is contracted against four *real* right-hand sides: the direct ``+m``
columns (real and imaginary parts) and the ``pi - theta`` mirror that serves
``-m`` (real and imaginary parts).  The XLA form expresses these as four operands
of one ``lax.reduce`` and so streams the slab once per channel; here each Triton
program owns a tile of outputs, loops over the reduction axis, loads each table
tile **once** and multiplies it by all four right-hand sides in registers.  The
contraction is memory-bound, so reading the table once is the whole gain
(roughly 4.5x over the XLA form, close to the pure-read floor of the slab).

Precision rules that keep the kernel near the byte floor:

* Products and within-tile sums are taken in the storage precision of the slab;
  only per-tile partials are accumulated in float64.  Widening every element to
  float64 (``cvt.f32.f64``) makes a float32 slab arithmetic-bound at a fraction
  of its bandwidth.
* The right-hand side is cast to the slab's precision, never widened.  Synthesis
  receives complex128 ``a_lm`` (NaMaster's storage), but a float32 table cannot
  inform the result beyond ~7 significant digits anyway, and float64 products
  would again make the kernel arithmetic-bound.

The ``flm`` / ``ftm`` assembly stays in XLA outside the kernel: it is
``L x (2L-1)`` complex values, negligible next to the slab.
"""

from functools import partial

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

# Tuning constants, chosen by sweeps at Nside 256 and 512 in both directions.
# Output elements per program: 8 is best or tied everywhere (16 and 32 lose 10-40%).
_E_TILE = 8
# Reduction elements per tile: longer tiles win up to a plateau at 1024 (at Nside
# 512 analysis, 64-element tiles are ~3x slower; 2048 is slightly worse than 1024).
_RED_CHUNK = 1024
# 2 warps per program beat 4 in every configuration swept (~7% at Nside 512).
_NUM_WARPS = 2


def _reduce_tile(rhs_ref, blk_ref, lane, out_idx, red_idx, accs, *, out_mask,
                 red_mask):
    """One tile of the contraction: load the table once, accumulate four channels.

    Products and the within-tile sum are in the operands' own precision (float32
    under ``set_table_precision("fp32")``); only the per-tile partial is converted
    to float64.  Widening per element would make the kernel bound by the
    ``cvt.f32.f64`` issue rate rather than by memory bandwidth.  The right-hand
    side is re-read once per output tile, so its width multiplies traffic directly
    and it is never widened in memory either.
    """
    if out_mask is None:
        v = plt.load(blk_ref.at[lane, out_idx[:, None], red_idx[None, :]])
    else:
        v = plt.load(blk_ref.at[lane, out_idx[:, None], red_idx[None, :]],
                     mask=out_mask, other=0.0)
    for c in range(4):
        if red_mask is None:
            r = plt.load(rhs_ref.at[c, lane, red_idx])
        else:
            r = plt.load(rhs_ref.at[c, lane, red_idx], mask=red_mask, other=0.0)
        part = jnp.sum(v * r[None, :], axis=1)
        accs = accs[:c] + (accs[c] + part.astype(jnp.float64),) + accs[c + 1:]
    return accs


def _contract(rhs_ref, blk_ref, acc_ref, *, n_out, n_red, out_tile, red_chunk,
              store_mask):
    """``acc[c, m, o] = sum_r blk[m, o, r] * rhs[c, m, r]`` over a contiguous ``r``.

    Whole tiles first (no predicate), then a single masked tail tile.  One masked
    1024-wide tile is faster than decomposing the remainder into smaller unmasked
    tiles (about 3x at Nside 256, ``n_red = 1023``): narrow vector loads cost more
    than a predicate on a wide one.
    """
    lane = pl.program_id(0)
    tile = pl.program_id(1)
    o = tile * out_tile + jnp.arange(out_tile)
    full = n_red // red_chunk
    zero = jnp.zeros((out_tile,), jnp.float64)
    # Static: with no tail on the output axis, no predicate is emitted anywhere in
    # the hot loop, which is what lets Triton issue vectorised loads.
    omask = None if not (n_out % out_tile) else jnp.broadcast_to(
        (o < n_out)[:, None], (out_tile, red_chunk))

    def step(ti, accs):
        r = ti * red_chunk + jnp.arange(red_chunk)
        return _reduce_tile(rhs_ref, blk_ref, lane, o, r, accs, out_mask=omask,
                            red_mask=None)

    accs = lax.fori_loop(0, full, step, (zero, zero, zero, zero))
    if n_red % red_chunk:
        r = full * red_chunk + jnp.arange(red_chunk)
        tail = (r < n_red)[None, :] if omask is None else omask & (r < n_red)[None, :]
        accs = _reduce_tile(rhs_ref, blk_ref, lane, o, r, accs, out_mask=tail,
                            red_mask=(r < n_red))
    for c in range(4):
        if store_mask:
            plt.store(acc_ref.at[c, lane, o], accs[c], mask=o < n_out)
        else:
            plt.store(acc_ref.at[c, lane, o], accs[c])


def _forward_kernel(rhs_ref, blk_ref, acc_ref, *, ncol, ntheta, e_tile, t_chunk):
    """Analysis: reduce each ``blk[m, ell, theta]`` over ``theta``."""
    _contract(rhs_ref, blk_ref, acc_ref, n_out=ncol, n_red=ntheta, out_tile=e_tile,
              red_chunk=t_chunk, store_mask=bool(ncol % e_tile))


def _inverse_kernel(rhs_ref, blk_ref, acc_ref, *, ncol, ntheta, t_tile, e_chunk):
    """Synthesis: reduce each ``blk[m, theta, ell]`` over ``ell``."""
    _contract(rhs_ref, blk_ref, acc_ref, n_out=ntheta, n_red=ncol, out_tile=t_tile,
              red_chunk=e_chunk, store_mask=bool(ntheta % t_tile))


_CALLS = {}


def _make(kind, mb, ncol, ntheta, store_name, rhs_name, tile, chunk):
    """Cached ``pallas_call`` for one (window shape, storage) pair.

    Cached like :mod:`gmaster._sht.band_pallas`: each window shape is one
    program, and recompiling it with Triton on every transform call would cost
    far more host time than the kernel saves.
    """
    key = (kind, mb, ncol, ntheta, store_name, rhs_name, tile, chunk)
    call = _CALLS.get(key)
    if call is None:
        tail = ncol if kind == "f" else ntheta
        out = jax.ShapeDtypeStruct((4, mb, tail), jnp.float64)
        if kind == "f":
            kern = partial(_forward_kernel, ncol=ncol, ntheta=ntheta, e_tile=tile,
                           t_chunk=chunk)
            grid_tail = -(-ncol // tile)
        else:
            kern = partial(_inverse_kernel, ncol=ncol, ntheta=ntheta, t_tile=tile,
                           e_chunk=chunk)
            grid_tail = -(-ntheta // tile)
        call = jax.jit(pl.pallas_call(
            kern, out_shape=out, grid=(mb, grid_tail),
            compiler_params=plt.CompilerParams(num_warps=_NUM_WARPS),
            name=f"gmaster_spin_contract_{kind}",
        ))
        _CALLS[key] = call
    return call


def _rhs_forward(ftm, m0, m1, L, rhs_dtype):
    """The four real right-hand sides of one analysis window, channel-major."""
    rev = ftm[::-1]
    direct = ftm[:, L + m0:L + m1]
    other = rev[:, L - m1 + 1:L - m0 + 1][:, ::-1]
    return jnp.stack([direct.real.T, direct.imag.T,
                      other.real.T, other.imag.T], axis=0).astype(rhs_dtype)


def forward(slab, ftm, *, L):
    """Analysis contraction over the theta-contiguous block set.

    Returns the assembled ``(L, 2L-1)`` array; agrees with
    :func:`gmaster._sht.spin_slice.forward_latitudinal` to summation-order
    round-off (~6e-16 relative in float64).
    """
    from gmaster._sht import spin_slice as ss

    ftm = jnp.asarray(ftm)
    off = L - 1
    sign = ss._sign(L, slab.spin)
    out = jnp.zeros((L, 2 * L - 1), dtype=jnp.result_type(slab[0], ftm))
    for (m0, m1, lo), block in zip(ss._windows(L), slab):
        ncol, ntheta = block.shape[1], block.shape[2]
        # Cast the right-hand side to the slab's precision.  Under
        # ``set_table_precision("fp32")`` the ring transform is already complex64 and
        # this is a no-op; for a float32 slab with a float64 map, widening the table
        # instead would make the kernel float64-arithmetic-bound.
        rhs = _rhs_forward(ftm, m0, m1, L, jnp.dtype(block.dtype))
        acc = _make("f", block.shape[0], ncol, ntheta, str(block.dtype),
                    str(rhs.dtype), _E_TILE, _RED_CHUNK)(rhs, block)
        out = out.at[lo:, off + m0:off + m1].set((acc[0] + 1j * acc[1]).T)
        mir = sign[lo:, None] * (acc[2] + 1j * acc[3]).T
        if m0:
            out = out.at[lo:, off - m1 + 1:off - m0 + 1].set(mir[:, ::-1])
        else:
            out = out.at[lo:, off - m1 + 1:off].set(mir[:, 1:][:, ::-1])
    return out


def inverse(slab, flm, *, L):
    """Synthesis contraction over the ell-contiguous block set."""
    from gmaster._sht import spin_slice as ss

    alm = jnp.asarray(flm)
    off = L - 1
    sign = ss._sign(L, slab.spin)
    ntheta = slab[0].shape[1]
    out = jnp.zeros((ntheta, 2 * L), dtype=jnp.result_type(slab[0], alm))
    for (m0, m1, lo), block in zip(ss._windows(L), slab):
        ncol, ntheta = block.shape[2], block.shape[1]
        direct = alm[lo:, off + m0:off + m1]
        mirror = sign[lo:, None] * alm[lo:, off - m1 + 1:off - m0 + 1][:, ::-1]
        # Precision follows the slab, not the caller's storage: ``alm`` is complex128
        # (NaMaster's convention) but a float32 slab cannot resolve it past ~7
        # digits, and float64 products would make the kernel arithmetic-bound.
        rhs = jnp.stack([direct.real.T, direct.imag.T,
                         mirror.real.T, mirror.imag.T],
                        axis=0).astype(block.dtype)
        acc = _make("i", block.shape[0], ncol, ntheta, str(block.dtype),
                    str(rhs.dtype), _E_TILE, _RED_CHUNK)(rhs, block)
        out = out.at[:, L + m0:L + m1].set((acc[0] + 1j * acc[1]).T)
        mir = (acc[2] + 1j * acc[3]).T
        if m0:
            out = out.at[:, L - m1 + 1:L - m0 + 1].set(mir[::-1][:, ::-1])
        else:
            out = out.at[:, L - m1 + 1:L].set(mir[::-1][:, 1:][:, ::-1])
    return out
