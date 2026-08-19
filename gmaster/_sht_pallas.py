"""Fused, rescaled scalar HEALPix latitude transforms for NVIDIA GPUs."""

from functools import lru_cache, partial

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
import jax.numpy as jnp
import numpy as np


@lru_cache(maxsize=16)
def _diagonal_normalization(L):
    values = np.empty(L)
    values[0] = 1 / np.sqrt(4 * np.pi)
    for m in range(1, L):
        values[m] = -np.sqrt(1 + 1 / (2 * m)) * values[m - 1]
    return values


def _scaled_value(value, exponent):
    biased = (exponent.astype(jnp.int64) + 1023) << 52
    factor = lax.bitcast_convert_type(biased, jnp.float64)
    factor = jnp.where(exponent >= -1022, factor, 0.0)
    return value * factor


def _renormalize(previous, current, exponent):
    largest = jnp.maximum(jnp.abs(previous), jnp.abs(current))
    large = largest > 2.0**100
    small = (largest < 2.0**-100) & (largest > 0)
    multiplier = jnp.where(
        large, 2.0**-100, jnp.where(small, 2.0**100, 1.0)
    )
    exponent += jnp.where(large, 100, jnp.where(small, -100, 0))
    return previous * multiplier, current * multiplier, exponent


def _renormalize_periodically(previous, current, exponent, ell, m):
    return lax.cond(
        jnp.bitwise_and(ell - m, 15) == 0,
        lambda: _renormalize(previous, current, exponent),
        lambda: (previous, current, exponent),
    )


def _analysis_kernel(
    ftm_real_ref,
    ftm_imag_ref,
    sine_ref,
    cosine_ref,
    diagonal_ref,
    weight_ref,
    phase_ref,
    _zero_real_ref,
    _zero_imag_ref,
    out_real_ref,
    out_imag_ref,
    *,
    L,
    ntheta,
    block_size,
):
    m = pl.program_id(0)
    m_float = m.astype(jnp.float64)
    offsets = jnp.arange(block_size)
    diagonal = plt.load(diagonal_ref.at[m])
    north_count = (ntheta + 1) // 2

    def apply_factor(real, imag, theta, valid):
        weight = plt.load(weight_ref.at[theta], mask=valid, other=0.0)
        angle = m_float * plt.load(
            phase_ref.at[theta], mask=valid, other=0.0
        )
        cosine_phase = jnp.cos(angle)
        sine_phase = jnp.sin(angle)
        return (
            weight * (cosine_phase * real - sine_phase * imag),
            weight * (sine_phase * real + cosine_phase * imag),
        )

    def add_coefficient(
        ell,
        values,
        north_real,
        north_imag,
        south_real,
        south_imag,
    ):
        parity = 1.0 - 2.0 * jnp.bitwise_and(ell + m, 1).astype(jnp.float64)
        real = jnp.sum(values * (north_real + parity * south_real))
        imag = jnp.sum(values * (north_imag + parity * south_imag))
        plt.store(
            out_real_ref.at[ell, m],
            plt.load(out_real_ref.at[ell, m]) + real,
        )
        plt.store(
            out_imag_ref.at[ell, m],
            plt.load(out_imag_ref.at[ell, m]) + imag,
        )

    def latitude_chunk(chunk, _):
        theta = chunk * block_size + offsets
        valid = theta < north_count
        south_theta = ntheta - 1 - theta
        has_pair = valid & (south_theta != theta)
        sine = plt.load(sine_ref.at[theta], mask=valid, other=1.0)
        cosine = plt.load(cosine_ref.at[theta], mask=valid, other=0.0)
        north_real = plt.load(
            ftm_real_ref.at[theta, m], mask=valid, other=0.0
        )
        north_imag = plt.load(
            ftm_imag_ref.at[theta, m], mask=valid, other=0.0
        )
        south_real = plt.load(
            ftm_real_ref.at[south_theta, m], mask=has_pair, other=0.0
        )
        south_imag = plt.load(
            ftm_imag_ref.at[south_theta, m], mask=has_pair, other=0.0
        )
        north_real, north_imag = apply_factor(
            north_real, north_imag, theta, valid
        )
        south_real, south_imag = apply_factor(
            south_real, south_imag, south_theta, has_pair
        )
        log2_scale = jnp.log2(jnp.abs(diagonal)) + m_float * jnp.log2(sine)
        scale_exponent = jnp.floor(log2_scale).astype(jnp.int32)
        qmm = jnp.where(
            diagonal < 0, -jnp.ones_like(sine), jnp.ones_like(sine)
        ) * jnp.exp2(log2_scale - scale_exponent)
        add_coefficient(
            m,
            _scaled_value(qmm, scale_exponent),
            north_real,
            north_imag,
            south_real,
            south_imag,
        )

        qm1 = jnp.sqrt(2.0 * m_float + 3.0) * cosine * qmm

        @pl.when(m + 1 < L)
        def add_first_off_diagonal():
            add_coefficient(
                m + 1,
                _scaled_value(qm1, scale_exponent),
                north_real,
                north_imag,
                south_real,
                south_imag,
            )

        def degree_step(ell, state):
            qm2, qm1, scale_exponent = state
            ell_float = ell.astype(jnp.float64)
            denominator = ell_float**2 - m_float**2
            coefficient_1 = jnp.sqrt((4 * ell_float**2 - 1) / denominator)
            coefficient_2 = jnp.sqrt(
                (2 * ell_float + 1)
                / (2 * ell_float - 3)
                * (((ell_float - 1) ** 2 - m_float**2) / denominator)
            )
            current = coefficient_1 * cosine * qm1 - coefficient_2 * qm2
            add_coefficient(
                ell,
                _scaled_value(current, scale_exponent),
                north_real,
                north_imag,
                south_real,
                south_imag,
            )
            qm1, current, scale_exponent = _renormalize_periodically(
                qm1, current, scale_exponent, ell, m
            )
            return qm1, current, scale_exponent

        lax.fori_loop(
            m + 2, L, degree_step, (qmm, qm1, scale_exponent)
        )

    lax.fori_loop(0, pl.cdiv(north_count, block_size), latitude_chunk, None)


def _synthesis_kernel(
    alm_real_ref,
    alm_imag_ref,
    sine_ref,
    cosine_ref,
    diagonal_ref,
    weight_ref,
    phase_ref,
    _zero_real_ref,
    _zero_imag_ref,
    out_real_ref,
    out_imag_ref,
    *,
    L,
    ntheta,
    block_size,
):
    m = pl.program_id(0)
    chunk = pl.program_id(1)
    m_float = m.astype(jnp.float64)
    theta = chunk * block_size + jnp.arange(block_size)
    north_count = (ntheta + 1) // 2
    valid = theta < north_count
    south_theta = ntheta - 1 - theta
    has_pair = valid & (south_theta != theta)
    sine = plt.load(sine_ref.at[theta], mask=valid, other=1.0)
    cosine = plt.load(cosine_ref.at[theta], mask=valid, other=0.0)
    diagonal = plt.load(diagonal_ref.at[m])
    log2_scale = jnp.log2(jnp.abs(diagonal)) + m_float * jnp.log2(sine)
    scale_exponent = jnp.floor(log2_scale).astype(jnp.int32)
    qmm = jnp.where(
        diagonal < 0, -jnp.ones_like(sine), jnp.ones_like(sine)
    ) * jnp.exp2(log2_scale - scale_exponent)

    alm_real = plt.load(alm_real_ref.at[m, m])
    alm_imag = plt.load(alm_imag_ref.at[m, m])
    value = _scaled_value(qmm, scale_exponent)
    result_real = value * alm_real
    result_imag = value * alm_imag
    south_result_real = result_real
    south_result_imag = result_imag

    qm1 = jnp.sqrt(2.0 * m_float + 3.0) * cosine * qmm
    first_valid = m + 1 < L
    alm_real = plt.load(
        alm_real_ref.at[m + 1, m], mask=first_valid, other=0.0
    )
    alm_imag = plt.load(
        alm_imag_ref.at[m + 1, m], mask=first_valid, other=0.0
    )
    value = _scaled_value(qm1, scale_exponent)
    result_real += value * alm_real
    result_imag += value * alm_imag
    south_result_real -= value * alm_real
    south_result_imag -= value * alm_imag

    def degree_step(ell, state):
        (
            qm2,
            qm1,
            scale_exponent,
            result_real,
            result_imag,
            south_result_real,
            south_result_imag,
        ) = state
        ell_float = ell.astype(jnp.float64)
        denominator = ell_float**2 - m_float**2
        coefficient_1 = jnp.sqrt((4 * ell_float**2 - 1) / denominator)
        coefficient_2 = jnp.sqrt(
            (2 * ell_float + 1)
            / (2 * ell_float - 3)
            * (((ell_float - 1) ** 2 - m_float**2) / denominator)
        )
        current = coefficient_1 * cosine * qm1 - coefficient_2 * qm2
        value = _scaled_value(current, scale_exponent)
        alm_real = plt.load(alm_real_ref.at[ell, m])
        alm_imag = plt.load(alm_imag_ref.at[ell, m])
        result_real += value * alm_real
        result_imag += value * alm_imag
        parity = 1.0 - 2.0 * jnp.bitwise_and(ell + m, 1).astype(jnp.float64)
        south_result_real += parity * value * alm_real
        south_result_imag += parity * value * alm_imag
        qm1, current, scale_exponent = _renormalize_periodically(
            qm1, current, scale_exponent, ell, m
        )
        return (
            qm1,
            current,
            scale_exponent,
            result_real,
            result_imag,
            south_result_real,
            south_result_imag,
        )

    (
        _,
        _,
        _,
        result_real,
        result_imag,
        south_result_real,
        south_result_imag,
    ) = lax.fori_loop(
        m + 2,
        L,
        degree_step,
        (
            qmm,
            qm1,
            scale_exponent,
            result_real,
            result_imag,
            south_result_real,
            south_result_imag,
        ),
    )
    north_weight = plt.load(weight_ref.at[theta], mask=valid, other=0.0)
    north_angle = m_float * plt.load(
        phase_ref.at[theta], mask=valid, other=0.0
    )
    north_cosine = jnp.cos(north_angle)
    north_sine = jnp.sin(north_angle)
    result_real, result_imag = (
        north_weight
        * (north_cosine * result_real - north_sine * result_imag),
        north_weight
        * (north_sine * result_real + north_cosine * result_imag),
    )
    south_weight = plt.load(
        weight_ref.at[south_theta], mask=has_pair, other=0.0
    )
    south_angle = m_float * plt.load(
        phase_ref.at[south_theta], mask=has_pair, other=0.0
    )
    south_cosine = jnp.cos(south_angle)
    south_sine = jnp.sin(south_angle)
    south_result_real, south_result_imag = (
        south_weight
        * (
            south_cosine * south_result_real
            - south_sine * south_result_imag
        ),
        south_weight
        * (
            south_sine * south_result_real
            + south_cosine * south_result_imag
        ),
    )
    plt.store(out_real_ref.at[theta, m], result_real, mask=valid)
    plt.store(out_imag_ref.at[theta, m], result_imag, mask=valid)
    plt.store(
        out_real_ref.at[south_theta, m], south_result_real, mask=has_pair
    )
    plt.store(
        out_imag_ref.at[south_theta, m], south_result_imag, mask=has_pair
    )


def _scalar_forward_latitudinal_impl(
    positive, theta, weights, phase, L, block_size
):
    sine = jnp.sin(theta)
    cosine = jnp.cos(theta)
    diagonal = jnp.asarray(_diagonal_normalization(L))
    zeros = jnp.zeros((L, L), dtype=jnp.float64)
    shape = jax.ShapeDtypeStruct((L, L), jnp.float64)
    real, imag = pl.pallas_call(
        partial(
            _analysis_kernel,
            L=L,
            ntheta=len(theta),
            block_size=block_size,
        ),
        out_shape=(shape, shape),
        grid=(L,),
        input_output_aliases={7: 0, 8: 1},
        compiler_params=plt.CompilerParams(num_warps=2),
        name="gmaster_scalar_analysis",
    )(
        jnp.real(positive),
        jnp.imag(positive),
        sine,
        cosine,
        diagonal,
        weights,
        phase,
        zeros,
        zeros,
    )
    return real + 1j * imag


def _scalar_inverse_latitudinal_impl(
    positive_alm, theta, weights, phase, L, block_size
):
    sine = jnp.sin(theta)
    cosine = jnp.cos(theta)
    diagonal = jnp.asarray(_diagonal_normalization(L))
    zeros = jnp.zeros((len(theta), L), dtype=jnp.float64)
    shape = jax.ShapeDtypeStruct((len(theta), L), jnp.float64)
    real, imag = pl.pallas_call(
        partial(
            _synthesis_kernel,
            L=L,
            ntheta=len(theta),
            block_size=block_size,
        ),
        out_shape=(shape, shape),
        grid=(L, pl.cdiv((len(theta) + 1) // 2, block_size)),
        input_output_aliases={7: 0, 8: 1},
        compiler_params=plt.CompilerParams(num_warps=2),
        name="gmaster_scalar_synthesis",
    )(
        jnp.real(positive_alm),
        jnp.imag(positive_alm),
        sine,
        cosine,
        diagonal,
        weights,
        phase,
        zeros,
        zeros,
    )
    return real + 1j * imag


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _scalar_forward_adjoint(
    positive_ftm, theta, weights, phase, L, block_size
):
    return _scalar_forward_latitudinal_impl(
        positive_ftm, theta, weights, phase, L, block_size
    )


def _scalar_forward_fwd(
    positive_ftm, theta, weights, phase, L, block_size
):
    result = _scalar_forward_latitudinal_impl(
        positive_ftm, theta, weights, phase, L, block_size
    )
    return result, (theta, weights, phase)


def _scalar_forward_bwd(L, block_size, residual, cotangent):
    theta, weights, phase = residual
    return (
        _scalar_inverse_latitudinal_impl(
            cotangent, theta, weights, phase, L, block_size
        ),
        jnp.zeros_like(theta),
        jnp.zeros_like(weights),
        jnp.zeros_like(phase),
    )


_scalar_forward_adjoint.defvjp(_scalar_forward_fwd, _scalar_forward_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _scalar_inverse_adjoint(
    positive_alm, theta, weights, phase, L, block_size
):
    return _scalar_inverse_latitudinal_impl(
        positive_alm, theta, weights, phase, L, block_size
    )


def _scalar_inverse_fwd(
    positive_alm, theta, weights, phase, L, block_size
):
    result = _scalar_inverse_latitudinal_impl(
        positive_alm, theta, weights, phase, L, block_size
    )
    return result, (theta, weights, phase)


def _scalar_inverse_bwd(L, block_size, residual, cotangent):
    theta, weights, phase = residual
    return (
        _scalar_forward_latitudinal_impl(
            cotangent, theta, weights, phase, L, block_size
        ),
        jnp.zeros_like(theta),
        jnp.zeros_like(weights),
        jnp.zeros_like(phase),
    )


_scalar_inverse_adjoint.defvjp(_scalar_inverse_fwd, _scalar_inverse_bwd)


@partial(jax.jit, static_argnames=("L", "block_size"))
def scalar_forward_latitudinal(
    positive_ftm,
    theta,
    weights=None,
    phase=None,
    *,
    L,
    block_size=256,
):
    weights = jnp.ones_like(theta) if weights is None else weights
    phase = jnp.zeros_like(theta) if phase is None else phase
    return _scalar_forward_adjoint(
        positive_ftm, theta, weights, phase, L, block_size
    )


@partial(jax.jit, static_argnames=("L", "block_size"))
def scalar_inverse_latitudinal(
    positive_alm,
    theta,
    weights=None,
    phase=None,
    *,
    L,
    block_size=256,
):
    weights = jnp.ones_like(theta) if weights is None else weights
    phase = jnp.zeros_like(theta) if phase is None else phase
    return _scalar_inverse_adjoint(
        positive_alm, theta, weights, phase, L, block_size
    )
