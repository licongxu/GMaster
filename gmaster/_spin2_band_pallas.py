"""GPU window emitter for spin-2 spherical harmonic transforms (d^ell_{m, -2}(theta)).

Evaluates the exact normalized Jacobi 3-term recurrence:
    v_ell = (c_1 * cos(theta) + c_0) * v_{ell-1} - c_2 * v_{ell-2}
in float64 registers with bit-exact periodic exponent rescaling, storing float32
window blocks of shape (mb, L - m0, ntheta) directly to device memory.
"""

from functools import partial
import numpy as np
from scipy.special import gammaln

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from gmaster._sht_pallas import _initial_factor

_CHUNK = 256
_NUM_WARPS = 8



def _spin2_window_kernel(cos_ref, l2s_ref, l2c_ref, l2norm_ref, sgn_ref,
                         c1_ref, c0_ref, c2_ref, m0_ref,
                         out_ref, *, L, ntheta, chunk):
    """Emit one m-block window: mb lanes march in registers, chunk covers theta."""
    lane = pl.program_id(0)
    tile = pl.program_id(1)
    m0 = plt.load(m0_ref.at[0])
    m = m0 + lane
    mf = m.astype(jnp.float64)
    t = tile * chunk + jnp.arange(chunk)
    valid = t < ntheta

    cosine = plt.load(cos_ref.at[t], mask=valid, other=0.0)
    l2s = plt.load(l2s_ref.at[t], mask=valid, other=0.0)
    l2c = plt.load(l2c_ref.at[t], mask=valid, other=0.0)
    l2norm = plt.load(l2norm_ref.at[lane])
    sgn = plt.load(sgn_ref.at[lane])

    alpha = jnp.abs(mf + 2.0)
    beta = jnp.abs(mf - 2.0)
    M = jnp.maximum(m, 2)

    log2_scale = l2norm + alpha * l2s + beta * l2c
    exponent = jnp.floor(log2_scale).astype(jnp.int32)

    seed = (sgn * jnp.ones_like(cosine)) * jnp.exp2(log2_scale - exponent)
    factor = _initial_factor(exponent)

    # 1. Degrees ell < M do not exist for spin 2: store zero.
    def fill(ell, _):
        plt.store(out_ref.at[lane, ell, t], jnp.zeros(chunk, dtype=out_ref.dtype),
                  mask=valid)
        return ()

    lax.fori_loop(0, M, fill, ())

    # 2. Store seed at ell = M
    plt.store(out_ref.at[lane, M, t], (seed * factor).astype(out_ref.dtype),
              mask=valid)

    # 3. Advance to ell = M + 1
    c1_1 = plt.load(c1_ref.at[lane, M + 1])
    c0_1 = plt.load(c0_ref.at[lane, M + 1])
    v_1 = (c1_1 * cosine + c0_1) * seed

    @pl.when(M + 1 < L)
    def store_first():
        plt.store(out_ref.at[lane, M + 1, t],
                  (v_1 * factor).astype(out_ref.dtype), mask=valid)

    # 4. March degrees ell from M + 2 to L - 1
    def degree(ell, state):
        v_m2, v_m1, exponent, factor = state
        c1 = plt.load(c1_ref.at[lane, ell])
        c0 = plt.load(c0_ref.at[lane, ell])
        c2 = plt.load(c2_ref.at[lane, ell])
        current = (c1 * cosine + c0) * v_m1 - c2 * v_m2
        plt.store(out_ref.at[lane, ell, t], (current * factor).astype(out_ref.dtype),
                  mask=valid)

        def rescale(state):
            v_m2, v_m1, exponent, factor = state
            largest = jnp.maximum(jnp.abs(v_m1), jnp.abs(current))
            large = largest > 2.0**100
            small = (largest < 2.0**-100) & (largest > 0)
            mult = jnp.where(large, 2.0**-100, jnp.where(small, 2.0**100, 1.0))
            exponent = exponent + jnp.where(large, 100, jnp.where(small, -100, 0))
            factor = jnp.where(large | small, _initial_factor(exponent), factor)
            return v_m1 * mult, current * mult, exponent, factor

        def keep(state):
            v_m2, v_m1, exponent, factor = state
            return v_m1, current, exponent, factor

        return lax.cond(jnp.bitwise_and(ell - m, 15) == 0, rescale, keep,
                        operand=(v_m2, v_m1, exponent, factor))

    lax.fori_loop(M + 2, L, degree, (seed, v_1, exponent, factor))


_CALLS = {}
_COEFF_CACHE_FULL = {}


@partial(jax.jit, static_argnums=(0,))
def _precompute_spin2_coeffs_gpu(L):
    m = jnp.arange(L, dtype=jnp.float64)[:, None]
    ell = jnp.arange(L, dtype=jnp.float64)[None, :]
    M = jnp.maximum(m, 2.0)
    valid = ell > M
    safe_ell = jnp.where(valid, ell, M + 1.0)
    den = jnp.sqrt((safe_ell**2 - m**2) * (safe_ell**2 - 4.0))

    c1 = jnp.where(valid, safe_ell * (2.0 * safe_ell - 1.0) / den, 0.0)
    c0 = jnp.where(valid, 2.0 * m * (2.0 * safe_ell - 1.0) / ((safe_ell - 1.0) * den), 0.0)
    num2 = jnp.sqrt(jnp.maximum(0.0, (safe_ell - 1.0)**2 - m**2) * ((safe_ell - 1.0)**2 - 4.0))
    c2 = jnp.where(valid, (safe_ell / (safe_ell - 1.0)) * (num2 / den), 0.0)
    return c1, c0, c2


@partial(jax.jit, static_argnums=(0,))
def _l2norm_gpu(L):
    m = jnp.arange(L, dtype=jnp.float64)
    m_safe = jnp.maximum(m, 2.0)
    norm = 0.5 * (lax.lgamma(2.0 * m_safe + 1.0) - lax.lgamma(m_safe + 3.0) - lax.lgamma(m_safe - 1.0)) / jnp.log(2.0)
    norm0 = 0.5 * (lax.lgamma(5.0) - lax.lgamma(3.0) - lax.lgamma(3.0)) / jnp.log(2.0)
    norm1 = 0.5 * (lax.lgamma(5.0) - lax.lgamma(4.0) - lax.lgamma(2.0)) / jnp.log(2.0)
    norm = jnp.where(m == 0, norm0, jnp.where(m == 1, norm1, norm))
    sgn = 1.0 - 2.0 * (m % 2)
    return norm, sgn


def _get_coefficients(L, m0, mb):
    """Precompute normalized recurrence coefficients and normalization on GPU."""
    cached = _COEFF_CACHE_FULL.get(L)
    if cached is None:
        c1_all, c0_all, c2_all = _precompute_spin2_coeffs_gpu(L)
        l2norm_all, sgn_all = _l2norm_gpu(L)
        cached = (l2norm_all, sgn_all, c1_all, c0_all, c2_all)
        _COEFF_CACHE_FULL[L] = cached

    l2norm_all, sgn_all, c1_all, c0_all, c2_all = cached
    return (l2norm_all[m0:m0 + mb], sgn_all[m0:m0 + mb],
            c1_all[m0:m0 + mb], c0_all[m0:m0 + mb], c2_all[m0:m0 + mb])


def _call(mb, L, ntheta, store_name, chunk):
    """Cached `pallas_call` compiled ONCE per geometry, reusable across all windows."""
    key = (mb, L, ntheta, store_name, chunk)
    call = _CALLS.get(key)
    if call is None:
        store = jnp.dtype(store_name)
        out_shape = jax.ShapeDtypeStruct((mb, L, ntheta), store)
        grid = (mb, -(-ntheta // chunk))
        call = jax.jit(pl.pallas_call(
            partial(_spin2_window_kernel, L=L, ntheta=ntheta, chunk=chunk),
            out_shape=out_shape,
            grid=grid,
            compiler_params=plt.CompilerParams(num_warps=_NUM_WARPS),
            name="gmaster_spin2_window_emitter",
        ))
        _CALLS[key] = call
    return call


def build_window(theta, L, m0, mb, store_name="float32"):
    """Emit one window of spin-2 spherical harmonics of fixed shape (mb, L, ntheta)."""
    ntheta = len(theta)
    chunk = _CHUNK if ntheta >= _CHUNK else max(32, 1 << (int(ntheta).bit_length() - 1))
    call = _call(mb, L, ntheta, store_name, chunk)
    l2norm, sgn, c1, c0, c2 = _get_coefficients(L, m0, mb)
    m0_arr = jnp.asarray([m0], dtype=jnp.int32)
    return call(
        jnp.cos(theta),
        jnp.log2(jnp.sin(theta / 2.0)),
        jnp.log2(jnp.cos(theta / 2.0)),
        l2norm, sgn, c1, c0, c2,
        m0_arr,
    )
