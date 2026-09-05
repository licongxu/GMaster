"""Pallas emitter for the Legendre band slabs.

`_theta_matrix._build_slab` produces the band with a `lax.scan` over the degree, and the
shipped build at Nside 1024 costs 340 s for 36.73 GiB of output (`.qwen/tmp/block_band_check.log`).
The device work inside that is small: the same scan warm is 26 ms for the leading block, and the
whole band's worth of XLA kernel time is under a second (`.qwen/tmp/pallas_band_fullbuild.log`).
What costs 340 s is that `block` m-blocks means `block` distinct XLA programs -- a static
`m0`/`mb` per block -- so the build pays a full compile per block, and each of those programs
also pins a copy of what it produced until the caches are dropped.

This module emits one m-block with a single Triton program instead: the recurrence runs in
registers, one lane per `m`, storing each degree straight into the parity-split halves that
`_build_pair` returns.  Measured on the whole band at Nside 1024, float32 storage, 48 blocks
warm: **0.08 s** against the 340 s shipped build (`.qwen/tmp/pallas_band_fullbuild2.log`), and
the block compiles cost 5.14 s cold instead of minutes.  Parity splitting is free -- two masked
stores per degree measured 37.4 G values/s against 37.8 for one unmasked store
(`.qwen/tmp/fullbuild_probe.log`), so there is no separate `vals[0::2]` pass.

Row values match `_build_slab`: same normalised 3-term recurrence, same seed
`sign(diag[m]) * 2**(log2|diag[m]| + m*log2(sin theta) - exponent)`, same rescale-every-16
schedule, same `_initial_factor` bit-assembled power of two.  The march carries in float64 and
only the store is rounded, so `float32` storage is the float64 band cast, exactly as in the
scan builder.  The two differ in the last bits the way any Triton program differs from an XLA
one -- max absolute difference 5.96e-08 at Nside 1024 on values up to 1.9e-01, which is float32
storage quantum, not a recurrence difference.

Two lowering facts this kernel exists to respect, both from
`.qwen/tmp/emitter_bisect2.log`:

* a `jnp.where(scalar_pred, scalar, scalar)` whose result feeds a vector multiply does not
  lower (`AssertionError: ('tensor<128xi1>', 'tensor<128xf64>')`).  The sign of the seed is
  therefore written `-jnp.ones_like(cosine)` / `jnp.ones_like(cosine)`, matching
  `_sht_pallas._analysis_kernel`.
* the `lax.cond` that carries the 16-degree guard must advance the recurrence in BOTH arms.
  Returning the incoming carry from the non-firing arm freezes the march on 15 degrees of
  every 16 and produces garbage that still looks plausible (~2e+00 relative).
"""

from functools import partial

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
import numpy as np

from gmaster._sht_pallas import _initial_factor

# Theta lanes per program.  The store-only probe on this layout peaked here and regressed at
# 512 (`.qwen/tmp/pallas_band_emitter.log`: 127.8 / 123.3 / 89.5 G values/s for 128 / 256 / 512).
_CHUNK = 128
_NUM_WARPS = 2


def _band_kernel(cos_ref, l2sin_ref, diag_ref, c1_ref, c2_ref,
                 even_ref, odd_ref, *, m0, L, north, chunk):
    """One m-block: `mb` lanes march independently, `north/chunk` tiles cover theta."""
    lane = pl.program_id(0)
    tile = pl.program_id(1)
    m = m0 + lane
    mf = m.astype(jnp.float64)
    t = tile * chunk + jnp.arange(chunk)
    valid = t < north

    cosine = plt.load(cos_ref.at[t], mask=valid, other=0.0)
    diagonal = plt.load(diag_ref.at[lane])
    log2_sin = plt.load(l2sin_ref.at[t], mask=valid, other=0.0)
    log2_scale = jnp.log2(jnp.abs(diagonal)) + mf * log2_sin
    exponent = jnp.floor(log2_scale).astype(jnp.int32)
    # Vector arms: a scalar-armed `where` here does not lower (see module docstring).
    seed = jnp.where(diagonal < 0, -jnp.ones_like(cosine),
                     jnp.ones_like(cosine)) * jnp.exp2(log2_scale - exponent)
    factor = _initial_factor(exponent)
    rows = L - m0

    def store_degree(i, value):
        """Write row `i = ell - m0` into the parity half that owns it.

        Row 0 of the block is `ell = m0`, so parity of `i` decides the half and `i >> 1` is
        its row.  Both stores carry the parity as a mask, which costs nothing and keeps the
        degree loop branch-free.
        """
        is_even = (i & 1) == 0
        within = (i < rows) & valid
        plt.store(even_ref.at[i >> 1, lane, t], value,
                  mask=jnp.broadcast_to(is_even, (chunk,)) & within)
        plt.store(odd_ref.at[i >> 1, lane, t], value,
                  mask=jnp.broadcast_to(jnp.logical_not(is_even), (chunk,)) & within)

    # Rows with ell < m are outside the band and read as zero, like the scan's
    # `where(ell >= mi, ..., 0.0)`.  Buffers handed to a pallas call are not zeroed.
    def fill(i, _):
        store_degree(i, jnp.zeros(chunk, dtype=even_ref.dtype))
        return ()

    lax.fori_loop(0, lane, fill, ())
    store_degree(lane, (seed * factor).astype(even_ref.dtype))

    # ell = m + 1 has c2 == 0, so it is the seed times c1(m+1) = sqrt(2m+3).
    previous = jnp.sqrt(2.0 * mf + 3.0) * cosine * seed
    store_degree(lane + 1, (previous * factor).astype(even_ref.dtype))

    def degree(ell, state):
        q_m2, q_m1, exponent, factor = state
        a = plt.load(c1_ref.at[lane, ell - m0])
        b = plt.load(c2_ref.at[lane, ell - m0])
        current = a * cosine * q_m1 - b * q_m2
        store_degree(ell - m0, (current * factor).astype(even_ref.dtype))

        def rescale(state):
            q_m2, q_m1, exponent, factor = state
            largest = jnp.maximum(jnp.abs(q_m1), jnp.abs(current))
            large = largest > 2.0**100
            small = (largest < 2.0**-100) & (largest > 0)
            mult = jnp.where(large, 2.0**-100, jnp.where(small, 2.0**100, 1.0))
            exponent = exponent + jnp.where(large, 100, jnp.where(small, -100, 0))
            factor = jnp.where(large | small, _initial_factor(exponent), factor)
            return q_m1 * mult, current * mult, exponent, factor

        def keep(state):
            q_m2, q_m1, exponent, factor = state
            return q_m1, current, exponent, factor

        return lax.cond(jnp.bitwise_and(ell - m, 15) == 0, rescale, keep,
                        operand=(q_m2, q_m1, exponent, factor))

    lax.fori_loop(m + 2, L, degree, (seed, previous, exponent, factor))


_CALLS = {}


def _call(m0, mb, L, north, store_name, chunk):
    """Cached `pallas_call` for one block geometry.

    Keyed on the geometry because the block's row count is baked into the output shapes; the
    cache is what keeps a build from paying Triton compilation on every call, which is the
    same trap the XLA route is in.
    """
    key = (m0, mb, L, north, store_name, chunk)
    call = _CALLS.get(key)
    if call is None:
        store = jnp.dtype(store_name)
        # `_build_pair` splits with `vals[0::2]` / `vals[1::2]`, so the halves have
        # ceil(rows/2) and floor(rows/2) rows.
        shapes = (jax.ShapeDtypeStruct(((L - m0 + 1) // 2, mb, north), store),
                  jax.ShapeDtypeStruct(((L - m0) // 2, mb, north), store))
        call = jax.jit(pl.pallas_call(
            partial(_band_kernel, m0=m0, L=L, north=north, chunk=chunk),
            out_shape=shapes,
            grid=(mb, -(-north // chunk)),
            compiler_params=plt.CompilerParams(num_warps=_NUM_WARPS),
            name="gmaster_band_emitter",
        ))
        _CALLS[key] = call
    return call


def build_pair(theta, L, diag, c1, c2, m0, mb, store_name="float64"):
    """`(even, odd)` halves of one m-block, matching `_theta_matrix._build_pair`.

    `theta` is the full north+south colatitude array; only the north half enters the band, the
    south being recovered by the `(-1)**(ell+m)` fold in the contraction.

    Not buildable inside `jax.ensure_compile_time_eval()`: that context exists to make `jit`
    evaluate eagerly, and a `pallas_call` has no eager rule, so it raises
    `NotImplementedError: Evaluation rule for 'program_id' not implemented`.  Neither
    `jax.disable_jit(False)` nor a nested `jit` undoes it -- the fold is a trace-time flag, not
    the jit setting -- which is why `_band` leaves the fold out when it picks this builder.
    """
    north = (len(theta) + 1) // 2
    chunk = _CHUNK if north >= _CHUNK else max(32, 1 << (int(north).bit_length() - 1))
    call = _call(m0, mb, L, north, store_name, chunk)
    k1 = jnp.asarray(np.ascontiguousarray(c1[m0:m0 + mb, m0:L]))
    k2 = jnp.asarray(np.ascontiguousarray(c2[m0:m0 + mb, m0:L]))
    return call(jnp.cos(theta[:north]), jnp.log2(jnp.sin(theta[:north])),
                jnp.asarray(diag[m0:m0 + mb]), k1, k2)
