"""Fused, rescaled scalar HEALPix latitude transforms for NVIDIA GPUs."""

from functools import lru_cache, partial

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
import jax.numpy as jnp
import numpy as np


# Thread-block width for the fused latitudinal kernels.  The hot loop keeps about
# ten float64 vectors live per theta lane, so `block_size / (32 * num_warps)`
# doubles per thread is what decides whether they sit in registers or spill.
_NUM_WARPS = 2


@lru_cache(maxsize=16)
def _diagonal_normalization(L):
    values = np.empty(L)
    values[0] = 1 / np.sqrt(4 * np.pi)
    for m in range(1, L):
        values[m] = -np.sqrt(1 + 1 / (2 * m)) * values[m - 1]
    return values


@lru_cache(maxsize=128)
def _normalized_coefficients_numpy(L, m_start, m_count):
    """Stable normalized-recurrence coefficient tables c1[l, m], c2[l, m].

    Computed as parallel vector operations outside the kernel so the
    sequential degree loop performs no square roots or divisions.
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
    """Exact power-of-two scale factor; zero below the double range."""
    return jnp.where(
        exponent >= -1022,
        lax.exp2(exponent.astype(jnp.float64)),
        0.0,
    )


def _renormalize_with_factor(previous, current, exponent, factor):
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
        # d^ell_{m0}(pi - theta) = (-1)^(ell+m) d^ell_{m0}(theta), so the southern
        # half of the ring sum collapses onto the northern one with a sign that
        # alternates in ell.  Within the degree loop that sign is fixed every
        # other step, so the two combinations are formed once per theta chunk and
        # the loop multiplies by a fixed vector instead of recomputing
        # `north + parity * south` at every degree.
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
            # An odd offset from m is never a renormalization point, so the pair
            # needs the tracked rescale only at its first degree.
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


def _seed_terms(m, mp, ell):
    """Closed-form Wigner-d seed terms (coefficient, cos-power, sin-power)."""
    from scipy.special import gammaln

    kmin = max(0, m - mp)
    kmax = min(ell + m, ell - mp)
    terms = []
    if kmax < kmin or ell < max(abs(m), abs(mp)):
        return terms
    log_norm = 0.5 * (
        gammaln(ell + m + 1)
        + gammaln(ell - m + 1)
        + gammaln(ell + mp + 1)
        + gammaln(ell - mp + 1)
    )
    for k in range(kmin, kmax + 1):
        log_term = (
            log_norm
            - gammaln(ell + m - k + 1)
            - gammaln(k + 1)
            - gammaln(mp - m + k + 1)
            - gammaln(ell - mp - k + 1)
        )
        terms.append(
            (
                (-1.0) ** k * np.exp(log_term),
                float(2 * ell + m - mp - 2 * k),
                float(mp - m + 2 * k),
            )
        )
    return terms


@lru_cache(maxsize=32)
def _spin_tables_numpy(L, spin, m_count):
    """Per-order tables for fused spin-weighted latitudinal kernels.

    Rows cover orders m = row - (m_count // 2); mp = -spin throughout.
    Seed columns l0 and l0+1 hold up to four closed-form terms each.
    """
    half = m_count // 2
    max_terms = 4
    seed_coeff = [
        np.zeros((m_count, max_terms)),
        np.zeros((m_count, max_terms)),
    ]
    seed_pow_c = [
        np.zeros((m_count, max_terms)),
        np.zeros((m_count, max_terms)),
    ]
    seed_pow_s = [
        np.zeros((m_count, max_terms)),
        np.zeros((m_count, max_terms)),
    ]
    n_terms0 = np.zeros(m_count, dtype=np.int32)
    n_terms1 = np.zeros(m_count, dtype=np.int32)
    start = np.full(m_count, L, dtype=np.int32)
    b_linear = np.zeros((m_count, L))
    b_constant = np.zeros((m_count, L))
    amp_current = np.zeros((m_count, L))
    inv_next = np.zeros((m_count, L))
    mp = -spin
    for row in range(m_count):
        m = row - half
        l0 = max(abs(m), abs(spin))
        if l0 >= L:
            continue
        start[row] = l0
        for slot, ell in enumerate((l0, min(l0 + 1, L - 1))):
            terms = _seed_terms(m, mp, ell)[:max_terms]
            (n_terms0 if slot == 0 else n_terms1)[row] = len(terms)
            for term, (coeff, pa, pb) in enumerate(terms):
                seed_coeff[slot][row, term] = coeff
                seed_pow_c[slot][row, term] = pa
                seed_pow_s[slot][row, term] = pb

        def amplitude(ell):
            value = (ell * ell - m * m) * (ell * ell - mp * mp)
            return np.sqrt(value) / ell if value > 0 else 0.0

        # Entry [ell] advances the pair (d^{ell-2}, d^{ell-1}) to d^ell via
        # A_ell d^ell = B_{ell-1} d^{ell-1} - A_{ell-1} d^{ell-2}.
        for ell in range(l0 + 2, L):
            previous = ell - 1
            b_linear[row, ell] = float(2 * previous + 1)
            b_constant[row, ell] = float(2 * previous + 1) * m * mp / (
                previous * (previous + 1.0)
            )
            amp_current[row, ell] = amplitude(previous)
            next_amplitude = amplitude(ell)
            inv_next[row, ell] = (
                1.0 / next_amplitude if next_amplitude > 0 else 0.0
            )
    def decompose(tables):
        log_abs = []
        signs = []
        for table in tables:
            finite = np.abs(table) > 0
            log_abs.append(np.where(finite, np.log2(np.abs(table)), 0.0))
            signs.append(np.where(table >= 0, 1.0, -1.0))
        return log_abs, signs

    log_abs_coeff, sign_coeff = decompose(seed_coeff)
    return (
        seed_pow_c[0], seed_pow_s[0],
        log_abs_coeff[0], sign_coeff[0],
        seed_pow_c[1], seed_pow_s[1],
        log_abs_coeff[1], sign_coeff[1],
        n_terms0,
        n_terms1,
        start,
        b_linear,
        b_constant,
        amp_current,
        inv_next,
    )


def _spin_tables(L, spin, m_count):
    return tuple(jnp.asarray(t) for t in _spin_tables_numpy(L, spin, m_count))


def _evaluate_seed_tracked(
    seed_pow_c_ref,
    seed_pow_s_ref,
    seed_log_abs_ref,
    seed_sign_ref,
    n_terms_ref,
    local_m,
    cosine_half,
    sine_half,
):
    """Exponent-tracked Wigner-d seed evaluation.

    Direct factorial sums cancel catastrophically for |m| -> l; evaluating
    every term at a shared base-two scale and recombining via frexp yields
    exact zeros where the true value underflows and bounded relative error
    where it does not.
    """
    count = plt.load(n_terms_ref.at[local_m])
    log_cos = jnp.log2(jnp.abs(cosine_half))
    log_sin = jnp.log2(jnp.abs(sine_half))
    log_terms = []
    signs = []
    for term in range(4):
        active = lax.convert_element_type(term < count, jnp.float64)
        power_c = plt.load(seed_pow_c_ref.at[local_m, term])
        power_s = plt.load(seed_pow_s_ref.at[local_m, term])
        log_abs = plt.load(seed_log_abs_ref.at[local_m, term])
        sign = plt.load(seed_sign_ref.at[local_m, term])
        magnitude = log_abs + power_c * log_cos + power_s * log_sin
        magnitude = jnp.where(
            jnp.isfinite(magnitude), magnitude, -jnp.inf
        )
        log_terms.append(jnp.where(active > 0, magnitude, -jnp.inf))
        signs.append(sign)
    e_max = log_terms[0]
    for term in range(1, 4):
        e_max = jnp.maximum(e_max, log_terms[term])
    e_scale = jnp.maximum(e_max, -1024.0)
    accumulator = jnp.zeros_like(cosine_half)
    for term in range(4):
        accumulator += signs[term] * lax.exp2(log_terms[term] - e_scale)
    mantissa, exponent = jnp.frexp(accumulator)
    total_exponent = e_scale + exponent
    return jnp.where(
        total_exponent >= -1073.0,
        mantissa * lax.exp2(total_exponent),
        0.0,
    )


def _spin_synthesis_kernel(
    alm_real_ref,
    alm_imag_ref,
    cos_theta_ref,
    cos_half_ref,
    sin_half_ref,
    seed_pow_c0_ref,
    seed_pow_s0_ref,
    seed_log_abs0_ref,
    seed_sign0_ref,
    seed_pow_c1_ref,
    seed_pow_s1_ref,
    seed_log_abs1_ref,
    seed_sign1_ref,
    n_terms0_ref,
    n_terms1_ref,
    start_ref,
    b_linear_ref,
    b_constant_ref,
    amp_current_ref,
    inv_next_ref,
    _zero_real_ref,
    _zero_imag_ref,
    out_real_ref,
    out_imag_ref,
    *,
    L,
    ntheta,
    block_size,
):
    """flm -> ftm: ftm[t, m] = sum_l flm[l, m] d^l_{m,-s}(theta_t).

    Fully branchless: degrees outside a program's band contribute exact zeros
    through multiplicative masks instead of control flow.
    """


    local_m = pl.program_id(0)
    chunk = pl.program_id(1)
    theta = chunk * block_size + jnp.arange(block_size)
    valid = theta < ntheta
    x = plt.load(cos_theta_ref.at[theta], mask=valid, other=0.0)
    cosine_half = plt.load(cos_half_ref.at[theta], mask=valid, other=1.0)
    sine_half = plt.load(sin_half_ref.at[theta], mask=valid, other=1.0)

    l_start = plt.load(start_ref.at[local_m])

    u2 = _evaluate_seed_tracked(
        seed_pow_c0_ref,
        seed_pow_s0_ref,
        seed_log_abs0_ref,
        seed_sign0_ref,
        n_terms0_ref,
        local_m,
        cosine_half,
        sine_half,
    )
    off_active = lax.convert_element_type(l_start + 1 < L, jnp.float64)
    u1 = off_active * _evaluate_seed_tracked(
        seed_pow_c1_ref,
        seed_pow_s1_ref,
        seed_log_abs1_ref,
        seed_sign1_ref,
        n_terms1_ref,
        local_m,
        cosine_half,
        sine_half,
    )

    def lane_mask(flag):
        return lax.convert_element_type(flag, jnp.float64)

    first_mask = l_start < L
    alm_real_first = plt.load(
        alm_real_ref.at[local_m, l_start], mask=first_mask, other=0.0
    )
    alm_imag_first = plt.load(
        alm_imag_ref.at[local_m, l_start], mask=first_mask, other=0.0
    )
    acc_real = u2 * alm_real_first
    acc_imag = u2 * alm_imag_first
    alm_real_off = plt.load(
        alm_real_ref.at[local_m, l_start + 1],
        mask=l_start + 1 < L,
        other=0.0,
    )
    alm_imag_off = plt.load(
        alm_imag_ref.at[local_m, l_start + 1],
        mask=l_start + 1 < L,
        other=0.0,
    )
    acc_real += u1 * alm_real_off
    acc_imag += u1 * alm_imag_off

    def degree_step(ell, state):
        um2, um1, acc_r, acc_i = state
        active = lane_mask((ell > l_start + 1) & (ell < L))
        b_value = plt.load(b_linear_ref.at[local_m, ell]) * x - plt.load(
            b_constant_ref.at[local_m, ell]
        )
        current = (
            b_value * um1 - plt.load(amp_current_ref.at[local_m, ell]) * um2
        ) * plt.load(inv_next_ref.at[local_m, ell]) * active
        alm_real_d = plt.load(alm_real_ref.at[local_m, ell])
        alm_imag_d = plt.load(alm_imag_ref.at[local_m, ell])
        acc_r += current * alm_real_d
        acc_i += current * alm_imag_d
        next_um1 = um1 * (1.0 - active) + current * active
        next_um2 = um2 * (1.0 - active) + um1 * active
        return next_um2, next_um1, acc_r, acc_i

    _, _, acc_real, acc_imag = jax.lax.fori_loop(
        0, L, degree_step, (u2, u1, acc_real, acc_imag)
    )
    plt.store(out_real_ref.at[theta, local_m], acc_real, mask=valid)
    plt.store(out_imag_ref.at[theta, local_m], acc_imag, mask=valid)


def _spin_analysis_kernel(
    ftm_real_ref,
    ftm_imag_ref,
    cos_theta_ref,
    cos_half_ref,
    sin_half_ref,
    seed_pow_c0_ref,
    seed_pow_s0_ref,
    seed_log_abs0_ref,
    seed_sign0_ref,
    seed_pow_c1_ref,
    seed_pow_s1_ref,
    seed_log_abs1_ref,
    seed_sign1_ref,
    n_terms0_ref,
    n_terms1_ref,
    start_ref,
    b_linear_ref,
    b_constant_ref,
    amp_current_ref,
    inv_next_ref,
    _zero_real_ref,
    _zero_imag_ref,
    out_real_ref,
    out_imag_ref,
    *,
    L,
    ntheta,
    block_size,
):
    """Adjoint of _spin_synthesis_kernel: weighted ftm -> flm."""


    local_m = pl.program_id(0)
    chunk = pl.program_id(1)
    theta = chunk * block_size + jnp.arange(block_size)
    valid = theta < ntheta
    x = plt.load(cos_theta_ref.at[theta], mask=valid, other=0.0)
    cosine_half = plt.load(cos_half_ref.at[theta], mask=valid, other=1.0)
    sine_half = plt.load(sin_half_ref.at[theta], mask=valid, other=1.0)

    l_start = plt.load(start_ref.at[local_m])

    u2 = _evaluate_seed_tracked(
        seed_pow_c0_ref,
        seed_pow_s0_ref,
        seed_log_abs0_ref,
        seed_sign0_ref,
        n_terms0_ref,
        local_m,
        cosine_half,
        sine_half,
    )
    off_active = lax.convert_element_type(l_start + 1 < L, jnp.float64)
    u1 = off_active * _evaluate_seed_tracked(
        seed_pow_c1_ref,
        seed_pow_s1_ref,
        seed_log_abs1_ref,
        seed_sign1_ref,
        n_terms1_ref,
        local_m,
        cosine_half,
        sine_half,
    )

    ftm_real_lane = plt.load(
        ftm_real_ref.at[theta, local_m], mask=valid, other=0.0
    )
    ftm_imag_lane = plt.load(
        ftm_imag_ref.at[theta, local_m], mask=valid, other=0.0
    )

    def add_coefficient(ell, values):
        plt.store(
            out_real_ref.at[local_m, ell],
            plt.load(out_real_ref.at[local_m, ell])
            + jnp.sum(values * ftm_real_lane),
        )
        plt.store(
            out_imag_ref.at[local_m, ell],
            plt.load(out_imag_ref.at[local_m, ell])
            + jnp.sum(values * ftm_imag_lane),
        )

    first_mask = l_start < L
    add_first = lax.convert_element_type(first_mask, jnp.float64)
    if True:
        contribution_r = jnp.sum(u2 * add_first * ftm_real_lane)
        contribution_i = jnp.sum(u2 * add_first * ftm_imag_lane)
        plt.store(
            out_real_ref.at[local_m, l_start],
            plt.load(out_real_ref.at[local_m, l_start], mask=first_mask, other=0.0)
            + contribution_r,
            mask=first_mask,
        )
        plt.store(
            out_imag_ref.at[local_m, l_start],
            plt.load(out_imag_ref.at[local_m, l_start], mask=first_mask, other=0.0)
            + contribution_i,
            mask=first_mask,
        )
        off_mask = l_start + 1 < L
        add_off = off_active
        contribution_or = jnp.sum(u1 * ftm_real_lane)
        contribution_oi = jnp.sum(u1 * ftm_imag_lane)
        plt.store(
            out_real_ref.at[local_m, l_start + 1],
            plt.load(out_real_ref.at[local_m, l_start + 1], mask=off_mask, other=0.0)
            + contribution_or,
            mask=off_mask,
        )
        plt.store(
            out_imag_ref.at[local_m, l_start + 1],
            plt.load(out_imag_ref.at[local_m, l_start + 1], mask=off_mask, other=0.0)
            + contribution_oi,
            mask=off_mask,
        )

    def degree_step(ell, state):
        um2, um1 = state
        active = lax.convert_element_type(
            (ell > l_start + 1) & (ell < L), jnp.float64
        )
        b_value = plt.load(b_linear_ref.at[local_m, ell]) * x - plt.load(
            b_constant_ref.at[local_m, ell]
        )
        current = (
            b_value * um1 - plt.load(amp_current_ref.at[local_m, ell]) * um2
        ) * plt.load(inv_next_ref.at[local_m, ell]) * active
        contribution_r = jnp.sum(current * ftm_real_lane)
        contribution_i = jnp.sum(current * ftm_imag_lane)
        plt.store(
            out_real_ref.at[local_m, ell],
            plt.load(out_real_ref.at[local_m, ell]) + contribution_r,
        )
        plt.store(
            out_imag_ref.at[local_m, ell],
            plt.load(out_imag_ref.at[local_m, ell]) + contribution_i,
        )
        next_um1 = um1 * (1.0 - active) + current * active
        next_um2 = um2 * (1.0 - active) + um1 * active
        return next_um2, next_um1

    _, _ = jax.lax.fori_loop(0, L, degree_step, (u2, u1))


@partial(jax.jit, static_argnames=("L", "spin", "block_size"))
def _spin_forward_latitudinal(weighted_ftm, theta, *, L, spin, block_size):
    """JIT wrapper: centered (ntheta, orders) weighted ftm -> transposed flm."""
    ntheta = len(theta)
    orders = weighted_ftm.shape[1]
    sine_half = jnp.sin(theta / 2.0)
    cosine_half = jnp.cos(theta / 2.0)
    cos_theta = jnp.cos(theta)
    zeros = jnp.zeros((orders, L), dtype=jnp.float64)
    shape = jax.ShapeDtypeStruct((orders, L), jnp.float64)
    real, imag = pl.pallas_call(
        partial(
            _spin_analysis_kernel,
            L=L,
            ntheta=ntheta,
            block_size=block_size,
        ),
        out_shape=(shape, shape),
        grid=(orders, pl.cdiv(ntheta, block_size)),
        input_output_aliases={20: 0, 21: 1},
        compiler_params=plt.CompilerParams(num_warps=_NUM_WARPS),
        name="gmaster_spin_analysis",
    )(
        jnp.real(weighted_ftm),
        jnp.imag(weighted_ftm),
        cos_theta,
        cosine_half,
        sine_half,
        *_spin_tables(L, spin, orders),
        zeros,
        zeros,
    )
    return real + 1j * imag


@partial(jax.jit, static_argnames=("L", "spin", "block_size"))
def _scalar_spin_synthesis_latitudinal(
    positive_alm, theta, *, L, spin, block_size
):
    """JIT wrapper: (orders, L) flm grid -> centered (ntheta, orders) ftm."""
    ntheta = len(theta)
    orders = positive_alm.shape[0]
    sine_half = jnp.sin(theta / 2.0)
    cosine_half = jnp.cos(theta / 2.0)
    cos_theta = jnp.cos(theta)
    zeros = jnp.zeros((ntheta, orders), dtype=jnp.float64)
    shape = jax.ShapeDtypeStruct((ntheta, orders), jnp.float64)
    real, imag = pl.pallas_call(
        partial(
            _spin_synthesis_kernel,
            L=L,
            ntheta=ntheta,
            block_size=block_size,
        ),
        out_shape=(shape, shape),
        grid=(orders, pl.cdiv(ntheta, block_size)),
        input_output_aliases={20: 0, 21: 1},
        compiler_params=plt.CompilerParams(num_warps=_NUM_WARPS),
        name="gmaster_spin_synthesis",
    )(
        jnp.real(positive_alm),
        jnp.imag(positive_alm),
        cos_theta,
        cosine_half,
        sine_half,
        *_spin_tables(L, spin, orders),
        zeros,
        zeros,
    )
    return real + 1j * imag




@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def _scalar_forward_adjoint(
    positive_ftm, theta, weights, phase, L, block_size, m_start
):
    return _scalar_forward_latitudinal_impl(
        positive_ftm, theta, weights, phase, L, block_size, m_start
    )


def _scalar_forward_fwd(
    positive_ftm, theta, weights, phase, L, block_size, m_start
):
    result = _scalar_forward_latitudinal_impl(
        positive_ftm, theta, weights, phase, L, block_size, m_start
    )
    return result, (theta, weights, phase)


def _scalar_forward_bwd(L, block_size, m_start, residual, cotangent):
    theta, weights, phase = residual
    return (
        _scalar_inverse_latitudinal_impl(
            cotangent, theta, weights, phase, L, block_size, m_start
        ),
        jnp.zeros_like(theta),
        jnp.zeros_like(weights),
        jnp.zeros_like(phase),
    )


_scalar_forward_adjoint.defvjp(_scalar_forward_fwd, _scalar_forward_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def _scalar_inverse_adjoint(
    positive_alm, theta, weights, phase, L, block_size, m_start
):
    return _scalar_inverse_latitudinal_impl(
        positive_alm, theta, weights, phase, L, block_size, m_start
    )


def _scalar_inverse_fwd(
    positive_alm, theta, weights, phase, L, block_size, m_start
):
    result = _scalar_inverse_latitudinal_impl(
        positive_alm, theta, weights, phase, L, block_size, m_start
    )
    return result, (theta, weights, phase)


def _scalar_inverse_bwd(L, block_size, m_start, residual, cotangent):
    theta, weights, phase = residual
    return (
        _scalar_forward_latitudinal_impl(
            cotangent, theta, weights, phase, L, block_size, m_start
        ),
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
    weights = jnp.ones_like(theta) if weights is None else weights
    phase = jnp.zeros_like(theta) if phase is None else phase
    return _scalar_forward_adjoint(
        positive_ftm, theta, weights, phase, L, block_size, m_start
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
    weights = jnp.ones_like(theta) if weights is None else weights
    phase = jnp.zeros_like(theta) if phase is None else phase
    return _scalar_inverse_adjoint(
        positive_alm, theta, weights, phase, L, block_size, m_start
    )
