"""Fused Pallas (Triton) kernels for the latitudinal step of the HEALPix SHT.

After the per-ring azimuthal FFT, a spin-0 spherical-harmonic transform reduces, for
each order ``m``, to a sum over rings of ring coefficients times ``lambda_lm(theta)``.
These kernels evaluate ``lambda_lm`` on the fly with the three-term recurrence in
``ell`` inside one fused GPU program per order (and theta chunk), so no
``(L, L, ntheta)`` table is ever stored.  They run in float64 with a power-of-two
rescaling that keeps the recurrence in range for high ``m`` near the poles, and use
the north/south ring symmetry to process ring pairs together.  Spin-weighted
transforms use the Wigner-d engines (`march_v2`, `spin_slice`, `dc`) instead.

Scalar layout: ``positive`` arrays hold orders ``m = m_start ... m_start +
m_count - 1`` along the last axis; ring data are ``(ntheta, m_count)`` and
harmonic data ``(L, m_count)`` (rows ``ell``; entries with ``ell < m`` are zero).
"""

from functools import lru_cache, partial

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
import jax.numpy as jnp
import numpy as np


# Warps per program for the fused latitudinal kernels.  The hot loop keeps about
# ten float64 vectors live per theta lane, so `block_size / (32 * num_warps)`
# doubles per thread decides whether they stay in registers or spill.
_NUM_WARPS = 2


@lru_cache(maxsize=16)
def _diagonal_normalization(L):
    """Sectoral coefficients ``lambda_mm = values[m] * sin(theta)^m``, m < L.

    Orthonormal spherical harmonics with the Condon-Shortley phase:
    ``values[0] = 1/sqrt(4 pi)``, ``values[m] = -sqrt(1 + 1/(2m)) values[m-1]``.
    """
    values = np.empty(L)
    values[0] = 1 / np.sqrt(4 * np.pi)
    for m in range(1, L):
        values[m] = -np.sqrt(1 + 1 / (2 * m)) * values[m - 1]
    return values


@lru_cache(maxsize=128)
def _normalized_coefficients_numpy(L, m_start, m_count):
    """Normalised Legendre recurrence coefficients, shape ``(m_count, L)`` each.

    ``lambda_lm = c1 * cos(theta) * lambda_{l-1,m} - c2 * lambda_{l-2,m}`` with
    ``c1 = sqrt((4l^2 - 1) / (l^2 - m^2))`` and
    ``c2 = sqrt((2l + 1) / (2l - 3) * ((l - 1)^2 - m^2) / (l^2 - m^2))``;
    entries with ``l <= m`` are zero.  Precomputed so the sequential degree loop
    in the kernels performs no square roots or divisions.
    """
    m = (m_start + np.arange(m_count, dtype=np.float64))[:, None]
    ell = np.arange(L, dtype=np.float64)[None, :]
    denominator = ell**2 - m**2
    safe = np.where(denominator > 0.0, denominator, 1.0)
    valid = denominator > 0.0
    with np.errstate(invalid="ignore"):
        c1 = np.sqrt((4.0 * ell**2 - 1.0) / safe)
        c2 = np.sqrt(
            (2.0 * ell + 1.0)
            / (2.0 * ell - 3.0)
            * ((ell - 1.0) ** 2 - m**2)
            / safe
        )
    c1 = np.where(valid, c1, 0.0)
    c2 = np.where(valid, c2, 0.0)
    return c1, c2


def _initial_factor(exponent):
    """``2.0 ** exponent`` for an integer exponent, assembled bit by bit.

    Writing the biased exponent into the exponent field is exact for every normal
    double, so the rescale never perturbs a mantissa bit.  ``lax.exp2`` is not
    exact at integer arguments (errors up to ~1e-13 relative) and, on the GPU,
    lowers to a slower software sequence.  Exponents below -1022 give 0 and above
    1023 give ``inf``.

    Also used by the precomputed band (``theta_matrix._build_slab``), which
    evaluates it at every degree; here :func:`_renormalize_periodically` calls
    it only every sixteenth degree.
    """
    biased = jnp.clip(exponent.astype(jnp.int64) + 1023, 1, 2046)
    bits = lax.bitcast_convert_type(biased << 52, jnp.float64)
    return jnp.where(exponent >= -1022,
                     jnp.where(exponent <= 1023, bits, jnp.inf), 0.0)


def _renormalize_with_factor(previous, current, exponent, factor):
    """Rescale the recurrence pair by ``2^-+100`` when it leaves [2^-100, 2^100].

    The true values are ``state * factor`` with ``factor = 2^exponent``; the
    rescale is exact (powers of two), so the represented values are unchanged.
    """
    largest = jnp.maximum(jnp.abs(previous), jnp.abs(current))
    large = largest > 2.0**100
    small = (largest < 2.0**-100) & (largest > 0)
    multiplier = jnp.where(
        large, 2.0**-100, jnp.where(small, 2.0**100, 1.0)
    )
    exponent += jnp.where(large, 100, jnp.where(small, -100, 0))
    factor = jnp.where(
        large | small,
        _initial_factor(exponent),
        factor,
    )
    return previous * multiplier, current * multiplier, exponent, factor


def _renormalize_periodically(previous, current, exponent, factor, ell, m):
    """Apply :func:`_renormalize_with_factor` only when ``ell - m`` is a multiple of 16.

    The thresholds sit far inside the float64 range, so checking every sixteenth
    degree suffices, and the ``lax.cond`` keeps the check out of most iterations.
    """
    return lax.cond(
        jnp.bitwise_and(ell - m, 15) == 0,
        lambda: _renormalize_with_factor(
            previous, current, exponent, factor
        ),
        lambda: (previous, current, exponent, factor),
    )


def _analysis_kernel(
    ftm_real_ref,
    ftm_imag_ref,
    sine_ref,
    cosine_ref,
    diagonal_ref,
    weight_ref,
    phase_ref,
    coefficient_1_ref,
    coefficient_2_ref,
    _zero_real_ref,
    _zero_imag_ref,
    out_real_ref,
    out_imag_ref,
    *,
    L,
    ntheta,
    block_size,
    m_start,
):
    """Scalar analysis for one order: ``alm[m, l] += sum_theta lambda_lm(theta) G_m(theta)``.

    One program per order ``m`` loops over chunks of northern rings; ``G`` is the
    ring coefficient times ``weight * exp(i m phase)``.  Output rows are
    accumulated in place into the zero-initialised aliased buffers.
    """
    local_m = pl.program_id(0)
    m = local_m + m_start
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
        combined_real,
        combined_imag,
        mask=None,
    ):
        real = jnp.sum(values * combined_real)
        imag = jnp.sum(values * combined_imag)
        if mask is None:
            previous_real = plt.load(out_real_ref.at[local_m, ell])
            previous_imag = plt.load(out_imag_ref.at[local_m, ell])
        else:
            previous_real = plt.load(
                out_real_ref.at[local_m, ell], mask=mask, other=0.0
            )
            previous_imag = plt.load(
                out_imag_ref.at[local_m, ell], mask=mask, other=0.0
            )
        plt.store(
            out_real_ref.at[local_m, ell], previous_real + real, mask=mask
        )
        plt.store(
            out_imag_ref.at[local_m, ell], previous_imag + imag, mask=mask
        )

    def latitude_chunk(chunk, _):
        theta = chunk * block_size + offsets
        valid = theta < north_count
        south_theta = ntheta - 1 - theta
        has_pair = valid & (south_theta != theta)
        sine = plt.load(sine_ref.at[theta], mask=valid, other=1.0)
        cosine = plt.load(cosine_ref.at[theta], mask=valid, other=0.0)
        north_real = plt.load(
            ftm_real_ref.at[local_m, theta], mask=valid, other=0.0
        )
        north_imag = plt.load(
            ftm_imag_ref.at[local_m, theta], mask=valid, other=0.0
        )
        south_real = plt.load(
            ftm_real_ref.at[local_m, south_theta],
            mask=has_pair,
            other=0.0,
        )
        south_imag = plt.load(
            ftm_imag_ref.at[local_m, south_theta],
            mask=has_pair,
            other=0.0,
        )
        north_real, north_imag = apply_factor(
            north_real, north_imag, theta, valid
        )
        south_real, south_imag = apply_factor(
            south_real, south_imag, south_theta, has_pair
        )
        log2_scale = jnp.log2(jnp.abs(diagonal)) + m_float * jnp.log2(sine)
        scale_exponent = jnp.floor(log2_scale).astype(jnp.int32)
        scale_factor = _initial_factor(scale_exponent)
        qmm = jnp.where(
            diagonal < 0, -jnp.ones_like(sine), jnp.ones_like(sine)
        ) * jnp.exp2(log2_scale - scale_exponent)
        # lambda_lm(pi - theta) = (-1)^(ell+m) lambda_lm(theta), so each southern
        # ring folds onto its northern partner with a sign alternating in ell.
        # Form `north + south` (even ell - m) and `north - south` (odd) once per
        # chunk; the degree loop is unrolled by two so each step uses a fixed one.
        plus_real = north_real + south_real
        plus_imag = north_imag + south_imag
        minus_real = north_real - south_real
        minus_imag = north_imag - south_imag
        add_coefficient(m, qmm * scale_factor, plus_real, plus_imag)

        qm1 = jnp.sqrt(2.0 * m_float + 3.0) * cosine * qmm

        @pl.when(m + 1 < L)
        def add_first_off_diagonal():
            add_coefficient(m + 1, qm1 * scale_factor, minus_real, minus_imag)

        def degree_pair(step, state):
            qm2, qm1, scale_exponent, scale_factor = state
            ell = m + 2 + 2 * step
            coefficient_1 = plt.load(coefficient_1_ref.at[local_m, ell])
            coefficient_2 = plt.load(coefficient_2_ref.at[local_m, ell])
            current = coefficient_1 * cosine * qm1 - coefficient_2 * qm2
            add_coefficient(ell, current * scale_factor, plus_real, plus_imag)
            qm1, current, scale_exponent, scale_factor = (
                _renormalize_periodically(
                    qm1, current, scale_exponent, scale_factor, ell, m
                )
            )
            # An odd offset from m is never a renormalisation point, so only the
            # first degree of the pair needs the rescale check.
            partner = ell + 1
            partner_valid = partner < L
            coefficient_1 = plt.load(
                coefficient_1_ref.at[local_m, partner],
                mask=partner_valid,
                other=0.0,
            )
            coefficient_2 = plt.load(
                coefficient_2_ref.at[local_m, partner],
                mask=partner_valid,
                other=0.0,
            )
            paired = coefficient_1 * cosine * current - coefficient_2 * qm1
            add_coefficient(
                partner,
                paired * scale_factor,
                minus_real,
                minus_imag,
                mask=partner_valid,
            )
            return current, paired, scale_exponent, scale_factor

        lax.fori_loop(
            0,
            (L - m - 1) // 2,
            degree_pair,
            (qmm, qm1, scale_exponent, scale_factor),
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
    coefficient_1_ref,
    coefficient_2_ref,
    _zero_real_ref,
    _zero_imag_ref,
    out_real_ref,
    out_imag_ref,
    *,
    L,
    ntheta,
    block_size,
    m_start,
):
    """Scalar synthesis: ``ftm[theta, m] = sum_l lambda_lm(theta) alm[m, l]``.

    Grid ``(m_count, northern-ring chunks)``; each program writes one northern
    ring chunk and its mirrored southern rings (sign ``(-1)^(ell+m)``), then
    applies ``weight * exp(i m phase)`` per ring.
    """
    local_m = pl.program_id(0)
    m = local_m + m_start
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
    scale_factor = _initial_factor(scale_exponent)
    qmm = jnp.where(
        diagonal < 0, -jnp.ones_like(sine), jnp.ones_like(sine)
    ) * jnp.exp2(log2_scale - scale_exponent)

    alm_real = plt.load(alm_real_ref.at[local_m, m])
    alm_imag = plt.load(alm_imag_ref.at[local_m, m])
    value = qmm * scale_factor
    result_real = value * alm_real
    result_imag = value * alm_imag
    south_result_real = result_real
    south_result_imag = result_imag

    qm1 = jnp.sqrt(2.0 * m_float + 3.0) * cosine * qmm
    first_valid = m + 1 < L
    alm_real = plt.load(
        alm_real_ref.at[local_m, m + 1], mask=first_valid, other=0.0
    )
    alm_imag = plt.load(
        alm_imag_ref.at[local_m, m + 1], mask=first_valid, other=0.0
    )
    value = qm1 * scale_factor
    result_real += value * alm_real
    result_imag += value * alm_imag
    south_result_real -= value * alm_real
    south_result_imag -= value * alm_imag

    def degree_step(ell, state):
        (
            qm2,
            qm1,
            scale_exponent,
            scale_factor,
            result_real,
            result_imag,
            south_result_real,
            south_result_imag,
        ) = state
        coefficient_1 = plt.load(coefficient_1_ref.at[local_m, ell])
        coefficient_2 = plt.load(coefficient_2_ref.at[local_m, ell])
        current = coefficient_1 * cosine * qm1 - coefficient_2 * qm2
        value = current * scale_factor
        alm_real = plt.load(alm_real_ref.at[local_m, ell])
        alm_imag = plt.load(alm_imag_ref.at[local_m, ell])
        result_real += value * alm_real
        result_imag += value * alm_imag
        parity = 1.0 - 2.0 * jnp.bitwise_and(ell + m, 1).astype(jnp.float64)
        south_result_real += parity * value * alm_real
        south_result_imag += parity * value * alm_imag
        qm1, current, scale_exponent, scale_factor = (
            _renormalize_periodically(
                qm1, current, scale_exponent, scale_factor, ell, m
            )
        )
        return (
            qm1,
            current,
            scale_exponent,
            scale_factor,
            result_real,
            result_imag,
            south_result_real,
            south_result_imag,
        )

    (
        _,
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
            scale_factor,
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
    plt.store(out_real_ref.at[theta, local_m], result_real, mask=valid)
    plt.store(out_imag_ref.at[theta, local_m], result_imag, mask=valid)
    plt.store(
        out_real_ref.at[south_theta, local_m],
        south_result_real,
        mask=has_pair,
    )
    plt.store(
        out_imag_ref.at[south_theta, local_m],
        south_result_imag,
        mask=has_pair,
    )


def _scalar_forward_latitudinal_impl(
    positive, theta, weights, phase, L, block_size, m_start
):
    """Launch the analysis kernel: ``(ntheta, m_count)`` -> ``(L, m_count)``."""
    sine = jnp.sin(theta)
    cosine = jnp.cos(theta)
    diagonal = jnp.asarray(_diagonal_normalization(L))
    coefficient_1, coefficient_2 = _normalized_coefficients_numpy(
        L, m_start, positive.shape[1]
    )
    m_count = positive.shape[1]
    transposed_input = jnp.asarray(positive).T
    zeros = jnp.zeros((m_count, L), dtype=jnp.float64)
    shape = jax.ShapeDtypeStruct((m_count, L), jnp.float64)
    real, imag = pl.pallas_call(
        partial(
            _analysis_kernel,
            L=L,
            ntheta=len(theta),
            block_size=block_size,
            m_start=m_start,
        ),
        out_shape=(shape, shape),
        grid=(m_count,),
        input_output_aliases={9: 0, 10: 1},
        compiler_params=plt.CompilerParams(num_warps=_NUM_WARPS),
        name="gmaster_scalar_analysis",
    )(
        jnp.real(transposed_input),
        jnp.imag(transposed_input),
        sine,
        cosine,
        diagonal,
        weights,
        phase,
        jnp.asarray(coefficient_1),
        jnp.asarray(coefficient_2),
        zeros,
        zeros,
    )
    return (real + 1j * imag).T


def _scalar_inverse_latitudinal_impl(
    positive_alm, theta, weights, phase, L, block_size, m_start
):
    """Launch the synthesis kernel: ``(L, m_count)`` -> ``(ntheta, m_count)``."""
    sine = jnp.sin(theta)
    cosine = jnp.cos(theta)
    diagonal = jnp.asarray(_diagonal_normalization(L))
    coefficient_1, coefficient_2 = _normalized_coefficients_numpy(
        L, m_start, positive_alm.shape[1]
    )
    m_count = positive_alm.shape[1]
    transposed = jnp.asarray(positive_alm).T
    zeros = jnp.zeros((len(theta), m_count), dtype=jnp.float64)
    shape = jax.ShapeDtypeStruct((len(theta), m_count), jnp.float64)
    real, imag = pl.pallas_call(
        partial(
            _synthesis_kernel,
            L=L,
            ntheta=len(theta),
            block_size=block_size,
            m_start=m_start,
        ),
        out_shape=(shape, shape),
        grid=(m_count, pl.cdiv((len(theta) + 1) // 2, block_size)),
        input_output_aliases={9: 0, 10: 1},
        compiler_params=plt.CompilerParams(num_warps=_NUM_WARPS),
        name="gmaster_scalar_synthesis",
    )(
        jnp.real(transposed),
        jnp.imag(transposed),
        sine,
        cosine,
        diagonal,
        weights,
        phase,
        jnp.asarray(coefficient_1),
        jnp.asarray(coefficient_2),
        zeros,
        zeros,
    )
    return real + 1j * imag


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7))
def _scalar_forward_adjoint(
    positive_ftm, theta, weights, phase, L, block_size, m_start, operand_dtype
):
    """Analysis latitudinal stage with the synthesis kernel as its own transpose.

    The kernel always computes in float64, and so does its transpose.  With
    ``set_ring_precision("fp32")`` the azimuthal stage supplies a complex64
    ``positive_ftm``, so the cotangent is cast back to ``operand_dtype``;
    otherwise JAX's transpose of the upstream ring FFT sees mismatched dtypes
    (complex64 vs complex128) and fails.
    """
    return _scalar_forward_latitudinal_impl(
        positive_ftm, theta, weights, phase, L, block_size, m_start
    )


def _scalar_forward_fwd(
    positive_ftm, theta, weights, phase, L, block_size, m_start, operand_dtype
):
    result = _scalar_forward_latitudinal_impl(
        positive_ftm, theta, weights, phase, L, block_size, m_start
    )
    return result, (theta, weights, phase)


def _scalar_forward_bwd(L, block_size, m_start, operand_dtype, residual, cotangent):
    theta, weights, phase = residual
    return (
        _scalar_inverse_latitudinal_impl(
            cotangent, theta, weights, phase, L, block_size, m_start
        ).astype(operand_dtype),
        jnp.zeros_like(theta),
        jnp.zeros_like(weights),
        jnp.zeros_like(phase),
    )


_scalar_forward_adjoint.defvjp(_scalar_forward_fwd, _scalar_forward_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7))
def _scalar_inverse_adjoint(
    positive_alm, theta, weights, phase, L, block_size, m_start, operand_dtype
):
    """Synthesis latitudinal stage with the analysis kernel as its transpose."""
    return _scalar_inverse_latitudinal_impl(
        positive_alm, theta, weights, phase, L, block_size, m_start
    )


def _scalar_inverse_fwd(
    positive_alm, theta, weights, phase, L, block_size, m_start, operand_dtype
):
    result = _scalar_inverse_latitudinal_impl(
        positive_alm, theta, weights, phase, L, block_size, m_start
    )
    return result, (theta, weights, phase)


def _scalar_inverse_bwd(L, block_size, m_start, operand_dtype, residual, cotangent):
    theta, weights, phase = residual
    return (
        _scalar_forward_latitudinal_impl(
            cotangent, theta, weights, phase, L, block_size, m_start
        ).astype(operand_dtype),
        jnp.zeros_like(theta),
        jnp.zeros_like(weights),
        jnp.zeros_like(phase),
    )


_scalar_inverse_adjoint.defvjp(_scalar_inverse_fwd, _scalar_inverse_bwd)


@partial(jax.jit, static_argnames=("L", "block_size", "m_start"))
def scalar_forward_latitudinal(
    positive_ftm,
    theta,
    weights=None,
    phase=None,
    *,
    L,
    block_size=256,
    m_start=0,
):
    """Scalar latitudinal analysis on the GPU (float64, differentiable).

    Computes ``alm[l, m] = sum_theta lambda_lm(theta) weight(theta)
    exp(i m phase(theta)) ftm[theta, m]`` for ``l >= m``.

    Parameters
    ----------
    positive_ftm : complex array, shape ``(ntheta, m_count)``
        Ring Fourier coefficients for orders ``m_start ... m_start + m_count - 1``.
        Rings must be ordered so that ring ``i`` and ``ntheta - 1 - i`` are
        mirror images about the equator.
    theta : float64 array, shape ``(ntheta,)``
        Ring colatitudes.
    weights, phase : float64 arrays, shape ``(ntheta,)``, optional
        Per-ring quadrature weights (default 1) and azimuthal phase offsets
        (default 0).
    L : int
        Band limit; degrees ``0 ... L-1``.
    block_size : int
        Rings per program chunk.
    m_start : int
        First order handled (lets callers split the orders across calls).

    Returns
    -------
    complex128 array, shape ``(L, m_count)``; entries with ``l < m`` are zero.
    """
    weights = jnp.ones_like(theta) if weights is None else weights
    phase = jnp.zeros_like(theta) if phase is None else phase
    return _scalar_forward_adjoint(
        positive_ftm, theta, weights, phase, L, block_size, m_start,
        positive_ftm.dtype,
    )


@partial(jax.jit, static_argnames=("L", "block_size", "m_start"))
def scalar_inverse_latitudinal(
    positive_alm,
    theta,
    weights=None,
    phase=None,
    *,
    L,
    block_size=256,
    m_start=0,
):
    """Scalar latitudinal synthesis on the GPU (float64, differentiable).

    Computes ``ftm[theta, m] = weight(theta) exp(i m phase(theta))
    sum_l lambda_lm(theta) alm[l, m]``.  Arguments as in
    :func:`scalar_forward_latitudinal`, with ``positive_alm`` of shape
    ``(L, m_count)``; returns a complex128 array of shape ``(ntheta, m_count)``.
    """
    weights = jnp.ones_like(theta) if weights is None else weights
    phase = jnp.zeros_like(theta) if phase is None else phase
    return _scalar_inverse_adjoint(
        positive_alm, theta, weights, phase, L, block_size, m_start,
        positive_alm.dtype,
    )
