"""Fused polar Wigner-d contraction: one read of the slice, four accumulators.

:func:`gmaster._spin_slice.forward_latitudinal` contracts one table block against
four *real* right-hand sides (direct re/im and the ``pi - theta`` mirror re/im), and
that channel split is what makes the XLA form fast -- ``einsum`` on the complex map
caps at 717 GB/s.  It is also what makes it slow: the four products are four separate
operands of the tuple ``lax.reduce``, so XLA walks the block set once per channel.
Here one Triton program owns a tile of outputs and loops the reduction axis, loading
the table **once** and multiplying it by all four right-hand sides in registers.

Three things had to be true for that to pay off, and each one is a measurement:

* **One read, not four.**  The premise of the kernel.
* **Products in the storage precision.**  Writing the accumulate as
  ``acc += v.astype(f64) * r.astype(f64)`` -- which is what "careful" suggests --
  runs the float32 slice at 210 GB/s against a 1361 GB/s control, because
  ``cvt.f32.f64`` issues at a small fraction of the FMA rate and the right-hand side
  is loaded four times per table element.  Same kernel, same tile, products in
  float32 with float64 tile partials: 1065 GB/s (``.qwen/tmp/spin_contract_tune3.log``
  against `.qwen/tmp/spin_contract_tune5.log`).
* **The right-hand side never widened in memory.**  In synthesis ``alm`` arrives
  complex128 because that is how NaMaster keeps ``a_lm``.  Feeding that to a float32
  slice as float64 leaves the kernel arithmetic-bound: 48.85 ms against 7.49 once the
  channels are taken at the slice's own width, 6.5x, at the same 7.1e-08 relative
  difference (`.qwen/tmp/spin_contract_tune5.log` against
  `.qwen/tmp/spin_contract_tune8.log`).  A float32 table cannot inform a caller past
  ~7 digits anyway.

Net on the block set itself (fp32 tables, control 1359-1377 GB/s,
`.qwen/tmp/spin_contract_tune9.log`): analysis 31.69 -> 6.97 ms (4.55x, 1444 GB/s
against a 9.37 GiB layout whose pure-read floor is 6.16 ms -- 88% of the floor) and
synthesis 33.39 -> 7.26 ms (4.60x) at Nside 512; 3.77x and 4.01x at Nside 256.  In the
pipeline that flips every spin-2 cell against ducc0 -- Nside 512 ``map2alm`` 34.2 ms
(0.76x) -> 11.1 ms (**2.53x**), ``alm2map`` 37.6 (0.59x) -> 8.8 (**2.71x**), Nside 256
0.83x -> **1.64x** and 0.52x -> **1.84x** (`.qwen/tmp/sht_spin2_pallas.log`, with
``rel alm`` 3.0-3.2e-07, the fp32 route's existing 1.1-1.8e-07 order).

Masking is *not* one of the levers, contrary to the first reading of this kernel: one
fully masked 1024-element tile beats a decomposition into unmasked 128- and
16-element tiles by 3x (see :func:`_contract`).

The ``flm`` / ``ftm`` assembly stays in XLA outside the kernel -- ``L x (2L-1)`` is
37 MB at Nside 512 against the table's 9.37 GiB, so it is not where the time is.
"""

from functools import partial

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

# Output elements per program.  Swept over {4,8,16,32}: 8 is best or tied in every
# cell of both directions at both 256 and 512 (16 and 32 lose 10-40%).
_E_TILE = 8
# Reduction elements per tile.  Swept over {64,128,256,512,1024,2048}: long wins and
# 1024 is the plateau.  At Nside 512 the analysis contraction runs 22.79 ms with
# 64-element tiles, 13.56 at 128, 8.97 at 256, 7.53 at 512, 6.97 at 1024, and 7.21 at
# 2048 (`.qwen/tmp/spin_contract_tune8.log`, `.qwen/tmp/spin_contract_tune9.log`; the
# 64/128/256 points are the 4-column series, the rest the 8-column one).  Nothing
# here is derived from a register budget: the formula this replaced produced 1024 for
# float32 by accident and 512 for float64, and cost 1.3x on float32.
_RED_CHUNK = 1024
# 2 warps beat 4 in every cell of the sweep (6.97 vs 7.50 ms analysis at Nside 512).
_NUM_WARPS = 2


def _reduce_tile(rhs_ref, blk_ref, lane, out_idx, red_idx, accs, *, out_mask,
                 red_mask):
    """One tile of the contraction: load the table once, accumulate four channels.

    The products and the within-tile sum are in the operands' own precision -- in
    production float32, because both the slice and the ring transform run at
    ``set_table_precision("fp32")`` -- and only the per-tile partial is converted to
    float64.  Widening per element is what makes the float64 spelling slow here
    rather than merely careful: ``cvt.f32.f64`` issues at a small fraction of the
    FMA rate on this card, the right-hand side is loaded four times per table
    element, and the tile sweep says the result is a kernel that runs at 211 GB/s
    against a 1361 GB/s control (``.qwen/tmp/spin_contract_tune3.log``).

    The right-hand side's width is a direct multiplier on traffic because it is
    re-read once per output tile of its lane, so it is never widened in memory
    either: it keeps the ring transform's own ``complex64``/``complex128`` real part.
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

    Whole tiles first (no predicate at all), then a single masked tail.  Splitting
    the ragged remainder against progressively smaller tiles -- the "get rid of the
    mask" instinct -- measures *worse* at every configuration tried: at Nside 256
    (``n_red = 1023``) one fully masked 1024-element tile does analysis in 1.15 ms,
    while the same length decomposed into 128- and 16-element unmasked tiles plus a
    masked tail takes 3.44 ms (`.qwen/tmp/spin_contract_tune5.log` against
    `.qwen/tmp/spin_contract_tune7.log`, same card, controls 1360 and 1402 GB/s).  A
    16-wide vector load costs more than a predicate on a 1024-wide one.
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

    Keyed and cached like :mod:`gmaster._band_pallas`: one window is one program,
    and paying Triton compilation on every transform call would put the whole win
    back into the host.
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
    :func:`gmaster._spin_slice.forward_latitudinal` to the summation-order floor
    (6.1e-16 relative at Nside 128 float64, ``.qwen/tmp/spin_contract_gate.log``).
    """
    from gmaster import _spin_slice as ss

    ftm = jnp.asarray(ftm)
    off = L - 1
    sign = ss._sign(L, slab.spin)
    out = jnp.zeros((L, 2 * L - 1), dtype=jnp.result_type(slab[0], ftm))
    for (m0, m1, lo), block in zip(ss._windows(L), slab):
        ncol, ntheta = block.shape[1], block.shape[2]
        # The right-hand side is taken to the slice's precision.  Under
        # ``set_table_precision("fp32")`` the ring transform is already complex64 so
        # this is a no-op; it matters only when a float32 slice meets a float64 map,
        # where widening the *table* instead (what the XLA form does) makes the
        # kernel float64-arithmetic-bound at a quarter of its byte rate.
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
    from gmaster import _spin_slice as ss

    alm = jnp.asarray(flm)
    off = L - 1
    sign = ss._sign(L, slab.spin)
    ntheta = slab[0].shape[1]
    out = jnp.zeros((ntheta, 2 * L), dtype=jnp.result_type(slab[0], alm))
    for (m0, m1, lo), block in zip(ss._windows(L), slab):
        ncol, ntheta = block.shape[2], block.shape[1]
        direct = alm[lo:, off + m0:off + m1]
        mirror = sign[lo:, None] * alm[lo:, off - m1 + 1:off - m0 + 1][:, ::-1]
        # The contraction takes its precision from the slice, not from the caller's
        # storage: `alm` arrives complex128 because that is how NaMaster keeps a_lm,
        # but a float32 slice cannot inform it past ~7 digits, and taking the
        # products in float64 leaves this kernel arithmetic-bound at ~200 GFMA/s
        # (48.6 ms against XLA's 32.7 at Nside 512, `.qwen/tmp/spin_contract_tune5.log`)
        # instead of near the byte floor.
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
