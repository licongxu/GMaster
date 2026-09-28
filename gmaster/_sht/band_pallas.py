"""Pallas emitter for the Legendre band tables.

Builds one m-block of the parity-split Legendre band (the ``(even, odd)`` halves returned by
`_theta_matrix._build_pair`) with a single Triton program.  Each program lane owns one order
`m`, runs the degree recurrence in registers and stores every degree straight into its parity
half.  The XLA builder (`_theta_matrix._build_slab`) needs one compiled program per m-block,
because `m0`/`mb` are static there; this kernel compiles once per block geometry, which turns a
build dominated by compilation into one dominated by the device stores.

Row values match `_build_slab`: the same normalised three-term recurrence, the same seed
`sign(diag[m]) * 2**(log2|diag[m]| + m*log2(sin theta) - exponent)`, the same rescale every 16
degrees and the same bit-assembled power of two from `_initial_factor`.  The march carries in
float64 and only the store is rounded, so `float32` storage is the float64 band cast.  The two
builders agree to the storage quantum (last-bit differences between a Triton and an XLA
program, e.g. 6e-08 absolute at Nside 1024 in float32 storage).

Two lowering constraints the kernel respects:

* a `jnp.where(scalar_pred, scalar, scalar)` whose result feeds a vector multiply does not
  lower, so the sign of the seed is written with vector arms
  (`-jnp.ones_like(cosine)` / `jnp.ones_like(cosine)`), as in `_sht_pallas._analysis_kernel`.
* the `lax.cond` that applies the 16-degree rescale must advance the recurrence in BOTH arms.
  Returning the incoming carry from the non-rescaling arm freezes the march on 15 of every 16
  degrees and yields plausible-looking garbage.
"""

from functools import partial

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
import numpy as np

from gmaster._sht.sht_pallas import _initial_factor

# Theta lanes per program: store throughput on this layout peaks at 128 and drops at 512.
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
    cache avoids paying Triton compilation on every call.
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

    Stores are masked by parity, so no separate `vals[0::2]` pass is needed.

    Not callable inside `jax.ensure_compile_time_eval()`: a `pallas_call` has no eager
    evaluation rule (`NotImplementedError: Evaluation rule for 'program_id' not implemented`),
    and neither `jax.disable_jit(False)` nor a nested `jit` lifts that trace-time flag.  This is
    why `_band` does not use compile-time evaluation when it picks this builder.
    """
    north = (len(theta) + 1) // 2
    chunk = _CHUNK if north >= _CHUNK else max(32, 1 << (int(north).bit_length() - 1))
    call = _call(m0, mb, L, north, store_name, chunk)
    k1 = jnp.asarray(np.ascontiguousarray(c1[m0:m0 + mb, m0:L]))
    k2 = jnp.asarray(np.ascontiguousarray(c2[m0:m0 + mb, m0:L]))
    return call(jnp.cos(theta[:north]), jnp.log2(jnp.sin(theta[:north])),
                jnp.asarray(diag[m0:m0 + mb]), k1, k2)
