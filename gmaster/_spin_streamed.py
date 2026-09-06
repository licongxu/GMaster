"""Streamed window-by-window latitudinal transforms for spin-2 spherical harmonics.

Allows arbitrary Nside (1024, 2048, 4096) on single GPU within minimal memory
footprint (~1.6 GiB working buffer) by evaluating normalized Wigner-d recurrences
on the fly in high-throughput registers and contracting without intermediate DRAM blocks.
"""

from functools import partial
import numpy as np

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from gmaster import _spin_slice as ss
from gmaster import _spin_contract_pallas as scp
from gmaster import _spin2_band_pallas as s2bp
from gmaster._sht_pallas import _initial_factor
import gmaster.utils as utils

_E_TILE = 8
_RED_CHUNK = 1024
_SYNTH_CHUNK = 128
_NUM_WARPS = 4


def _fused_spin2_synth_mixed(cos_ref, l2s_ref, l2c_ref, l2norm_ref, sgn_ref,
                             c1_ref, c0_ref, c2_ref, rhs_ref, m0_ref,
                             out_ref, *, L, ntheta, chunk):
    """Fused Wigner-d recurrence + contraction in registers, eliminating DRAM tables."""
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

    acc0 = jnp.zeros(chunk, dtype=jnp.float32)
    acc1 = jnp.zeros(chunk, dtype=jnp.float32)
    acc2 = jnp.zeros(chunk, dtype=jnp.float32)
    acc3 = jnp.zeros(chunk, dtype=jnp.float32)

    v_0 = (seed * factor).astype(jnp.float32)
    r0 = plt.load(rhs_ref.at[0, lane, M])
    r1 = plt.load(rhs_ref.at[1, lane, M])
    r2 = plt.load(rhs_ref.at[2, lane, M])
    r3 = plt.load(rhs_ref.at[3, lane, M])
    acc0 = acc0 + v_0 * r0
    acc1 = acc1 + v_0 * r1
    acc2 = acc2 + v_0 * r2
    acc3 = acc3 + v_0 * r3

    c1_1 = plt.load(c1_ref.at[lane, M + 1])
    c0_1 = plt.load(c0_ref.at[lane, M + 1])
    v_1 = (c1_1 * cosine + c0_1) * seed

    v_1_f = (v_1 * factor).astype(jnp.float32)
    r0_1 = plt.load(rhs_ref.at[0, lane, M + 1])
    r1_1 = plt.load(rhs_ref.at[1, lane, M + 1])
    r2_1 = plt.load(rhs_ref.at[2, lane, M + 1])
    r3_1 = plt.load(rhs_ref.at[3, lane, M + 1])
    cond1 = M + 1 < L
    acc0 = acc0 + jnp.where(cond1, v_1_f * r0_1, 0.0)
    acc1 = acc1 + jnp.where(cond1, v_1_f * r1_1, 0.0)
    acc2 = acc2 + jnp.where(cond1, v_1_f * r2_1, 0.0)
    acc3 = acc3 + jnp.where(cond1, v_1_f * r3_1, 0.0)

    def degree(ell, state):
        v_m2, v_m1, exponent, factor, a0, a1, a2, a3 = state
        c1 = plt.load(c1_ref.at[lane, ell])
        c0 = plt.load(c0_ref.at[lane, ell])
        c2 = plt.load(c2_ref.at[lane, ell])
        current = (c1 * cosine + c0) * v_m1 - c2 * v_m2
        val = (current * factor).astype(jnp.float32)

        r0_l = plt.load(rhs_ref.at[0, lane, ell])
        r1_l = plt.load(rhs_ref.at[1, lane, ell])
        r2_l = plt.load(rhs_ref.at[2, lane, ell])
        r3_l = plt.load(rhs_ref.at[3, lane, ell])
        a0 = a0 + val * r0_l
        a1 = a1 + val * r1_l
        a2 = a2 + val * r2_l
        a3 = a3 + val * r3_l

        def rescale(state):
            v_m2, v_m1, exponent, factor, a0, a1, a2, a3 = state
            largest = jnp.maximum(jnp.abs(v_m1), jnp.abs(current))
            large = largest > 2.0**100
            small = (largest < 2.0**-100) & (largest > 0)
            mult = jnp.where(large, 2.0**-100, jnp.where(small, 2.0**100, 1.0))
            exponent = exponent + jnp.where(large, 100, jnp.where(small, -100, 0))
            factor = jnp.where(large | small, _initial_factor(exponent), factor)
            return v_m1 * mult, current * mult, exponent, factor, a0, a1, a2, a3

        def keep(state):
            v_m2, v_m1, exponent, factor, a0, a1, a2, a3 = state
            return v_m1, current, exponent, factor, a0, a1, a2, a3

        return lax.cond(jnp.bitwise_and(ell - m, 15) == 0, rescale, keep,
                        operand=(v_m2, v_m1, exponent, factor, a0, a1, a2, a3))

    _, _, _, _, acc0, acc1, acc2, acc3 = lax.fori_loop(
        M + 2, L, degree, (seed, v_1, exponent, factor, acc0, acc1, acc2, acc3))

    plt.store(out_ref.at[0, lane, t], acc0, mask=valid)
    plt.store(out_ref.at[1, lane, t], acc1, mask=valid)
    plt.store(out_ref.at[2, lane, t], acc2, mask=valid)
    plt.store(out_ref.at[3, lane, t], acc3, mask=valid)


_ANALYSIS_CHUNK = 256
_ANALYSIS_FUSED_CALLS = {}


def _fused_spin2_analysis_mixed(cos_ref, l2s_ref, l2c_ref, l2norm_ref, sgn_ref,
                               c1_ref, c0_ref, c2_ref, rhs_ref, m0_ref,
                               out_ref, *, L, ntheta, chunk):
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

    r0 = plt.load(rhs_ref.at[0, lane, t], mask=valid, other=0.0)
    r1 = plt.load(rhs_ref.at[1, lane, t], mask=valid, other=0.0)
    r2 = plt.load(rhs_ref.at[2, lane, t], mask=valid, other=0.0)
    r3 = plt.load(rhs_ref.at[3, lane, t], mask=valid, other=0.0)

    alpha = jnp.abs(mf + 2.0)
    beta = jnp.abs(mf - 2.0)
    M = jnp.maximum(m, 2)

    log2_scale = l2norm + alpha * l2s + beta * l2c
    exponent = jnp.floor(log2_scale).astype(jnp.int32)

    seed = (sgn * jnp.ones_like(cosine)) * jnp.exp2(log2_scale - exponent)
    factor = _initial_factor(exponent)

    z = jnp.float32(0.0)
    def fill(ell, _):
        plt.store(out_ref.at[0, lane, tile, ell], z)
        plt.store(out_ref.at[1, lane, tile, ell], z)
        plt.store(out_ref.at[2, lane, tile, ell], z)
        plt.store(out_ref.at[3, lane, tile, ell], z)
        return ()

    lax.fori_loop(0, M, fill, ())

    v_0 = jnp.where(valid, (seed * factor).astype(jnp.float32), z)
    plt.store(out_ref.at[0, lane, tile, M], jnp.sum(v_0 * r0))
    plt.store(out_ref.at[1, lane, tile, M], jnp.sum(v_0 * r1))
    plt.store(out_ref.at[2, lane, tile, M], jnp.sum(v_0 * r2))
    plt.store(out_ref.at[3, lane, tile, M], jnp.sum(v_0 * r3))

    c1_1 = plt.load(c1_ref.at[lane, M + 1])
    c0_1 = plt.load(c0_ref.at[lane, M + 1])
    v_1 = (c1_1 * cosine + c0_1) * seed

    v_1_f = jnp.where(valid, (v_1 * factor).astype(jnp.float32), z)
    cond1 = M + 1 < L
    plt.store(out_ref.at[0, lane, tile, M + 1], jnp.where(cond1, jnp.sum(v_1_f * r0), z))
    plt.store(out_ref.at[1, lane, tile, M + 1], jnp.where(cond1, jnp.sum(v_1_f * r1), z))
    plt.store(out_ref.at[2, lane, tile, M + 1], jnp.where(cond1, jnp.sum(v_1_f * r2), z))
    plt.store(out_ref.at[3, lane, tile, M + 1], jnp.where(cond1, jnp.sum(v_1_f * r3), z))

    def degree(ell, state):
        v_m2, v_m1, exponent, factor = state
        c1 = plt.load(c1_ref.at[lane, ell])
        c0 = plt.load(c0_ref.at[lane, ell])
        c2 = plt.load(c2_ref.at[lane, ell])
        current = (c1 * cosine + c0) * v_m1 - c2 * v_m2
        val = jnp.where(valid, (current * factor).astype(jnp.float32), z)

        plt.store(out_ref.at[0, lane, tile, ell], jnp.sum(val * r0))
        plt.store(out_ref.at[1, lane, tile, ell], jnp.sum(val * r1))
        plt.store(out_ref.at[2, lane, tile, ell], jnp.sum(val * r2))
        plt.store(out_ref.at[3, lane, tile, ell], jnp.sum(val * r3))

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


def _get_fused_analysis(mb, L, ntheta):
    key = (mb, L, ntheta, _ANALYSIS_CHUNK)
    call = _ANALYSIS_FUSED_CALLS.get(key)
    if call is None:
        n_tiles = -(-ntheta // _ANALYSIS_CHUNK)
        out_shape = jax.ShapeDtypeStruct((4, mb, n_tiles, L), jnp.float32)
        call = jax.jit(pl.pallas_call(
            partial(_fused_spin2_analysis_mixed, L=L, ntheta=ntheta, chunk=_ANALYSIS_CHUNK),
            out_shape=out_shape,
            grid=(mb, n_tiles),
            compiler_params=plt.CompilerParams(num_warps=8),
            name="gmaster_spin2_fused_analysis",
        ))
        _ANALYSIS_FUSED_CALLS[key] = call
    return call


_SYNTH_FUSED_CALLS = {}


def _get_fused_synth(mb, L, ntheta):
    key = (mb, L, ntheta, _SYNTH_CHUNK)
    call = _SYNTH_FUSED_CALLS.get(key)
    if call is None:
        grid = (mb, -(-ntheta // _SYNTH_CHUNK))
        out_shape = jax.ShapeDtypeStruct((4, mb, ntheta), jnp.float32)
        call = jax.jit(pl.pallas_call(
            partial(_fused_spin2_synth_mixed, L=L, ntheta=ntheta, chunk=_SYNTH_CHUNK),
            out_shape=out_shape,
            grid=grid,
            compiler_params=plt.CompilerParams(num_warps=_NUM_WARPS),
            name="gmaster_spin2_fused_synth",
        ))
        _SYNTH_FUSED_CALLS[key] = call
    return call


def forward_latitudinal(ftm, *, L, nside, spin=2):
    """Streamed forward latitudinal contraction for spin 2."""
    ftm = jnp.asarray(ftm)
    off = L - 1
    theta = np.asarray(utils._stable_thetas(L, nside), dtype=np.float64)
    ntheta = len(theta)
    sign = ss._sign(L, spin)

    cos_theta = jnp.cos(theta)
    l2s = jnp.log2(jnp.sin(theta / 2.0))
    l2c = jnp.log2(jnp.cos(theta / 2.0))

    rev = ftm[::-1]
    directs = []
    mirrors = []

    for m0, m1, lo in ss._windows(L):
        mb = m1 - m0
        call = _get_fused_analysis(mb, L, ntheta)
        l2norm, sgn, c1, c0, c2 = s2bp._get_coefficients(L, m0, mb)
        m0_arr = jnp.asarray([m0], dtype=jnp.int32)

        direct = ftm[:, L + m0:L + m1]
        other = rev[:, L - m1 + 1:L - m0 + 1][:, ::-1]
        rhs = jnp.stack([direct.real.T, direct.imag.T,
                         other.real.T, other.imag.T], axis=0).astype(jnp.float32)

        acc_tiles = call(
            cos_theta, l2s, l2c, l2norm, sgn, c1, c0, c2, rhs, m0_arr
        )
        acc = jnp.sum(acc_tiles, axis=2)

        directs.append((acc[0, :mb] + 1j * acc[1, :mb]).T)
        mirrors.append(sign[:, None] * (acc[2, :mb] + 1j * acc[3, :mb]).T)

    d_all = jnp.concatenate(directs, axis=1)
    m_all = jnp.concatenate(mirrors, axis=1)
    m_neg = m_all[:, 1:][:, ::-1]
    return jnp.concatenate([m_neg, d_all], axis=1)


def inverse_latitudinal(flm, *, L, nside, spin=2):
    """Streamed monolithic inverse latitudinal contraction for spin 2."""
    alm = jnp.asarray(flm)
    off = L - 1
    theta = np.asarray(utils._stable_thetas(L, nside), dtype=np.float64)
    ntheta = len(theta)
    sign = ss._sign(L, spin)

    fn_synth = _get_fused_synth(L, L, ntheta)

    cos_th = jnp.cos(theta)
    l2s_th = jnp.log2(jnp.sin(theta / 2.0))
    l2c_th = jnp.log2(jnp.cos(theta / 2.0))
    l2norm_all, sgn_all, c1_all, c0_all, c2_all = s2bp._get_coefficients(L, 0, L)

    direct = alm[:, off:off + L]
    mirror = sign[:, None] * alm[:, off - L + 1:off + 1][:, ::-1]
    rhs = jnp.stack([direct.real.T, direct.imag.T,
                     mirror.real.T, mirror.imag.T], axis=0).astype(jnp.float32)

    acc = fn_synth(cos_th, l2s_th, l2c_th,
                   l2norm_all, sgn_all, c1_all, c0_all, c2_all,
                   rhs, jnp.asarray([0], dtype=jnp.int32))

    d_all = (acc[0] + 1j * acc[1]).T
    m_all = (acc[2] + 1j * acc[3]).T
    m_neg = m_all[::-1, 1:][:, ::-1]
    return jnp.concatenate([jnp.zeros((ntheta, 1), dtype=d_all.dtype), m_neg, d_all], axis=1)
