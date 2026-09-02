"""Double-fp32 Pallas kernels for the scalar latitudinal SHT.

Recurrence state (q_{ell-2}, q_{ell-1}) and the c1/c2/cos coefficients are
kept as three-limb f32 tuples (~72-bit significand); the per-theta data
factors (parity-selected ring-FFT combinations) stay two-limb (~48-bit),
which is sufficient because their errors are not amplified by the
recurrence.  All hot-loop arithmetic is f32 (FMA-based); only the block
reduction into the output and the seed/scale setup use f64.

A 72-bit state is required: with a two-limb state the recurrence's slowly
growing mode amplifies per-step roundoff and the m=0 state drifts by
~1e-7 over 190 steps at Nside 64, while a three-limb state keeps the drift
at ~1e-13 (numpy emulation vs the closed-form Legendre solution).

The value leaking into the data path is scaled by the power-of-two renorm
factor in f32: (h*fac, m*fac + l*fac), so no f64 op sits in the hot loop.
"""

from functools import partial

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
import jax.numpy as jnp
import numpy as np

from . import _sht_pallas


# --- f32 primitives -------------------------------------------------------


def _fma(a, b, c):
    """Native PTX FFMA (f32); this JAX build exposes no jnp.fma/lax.fma.

    ``c`` is always the widest operand at the call sites, so broadcast the
    other two to its shape (inline-asm requires identical shapes).
    """
    if a.shape != c.shape:
        a = jnp.broadcast_to(a, c.shape)
    if b.shape != c.shape:
        b = jnp.broadcast_to(b, c.shape)
    (result,) = plt.elementwise_inline_asm(
        "fma.rn.f32 $0, $1, $2, $3;",
        args=(a, b, c),
        constraints="=f,f,f,f",
        pack=1,
        result_shape_dtypes=(jax.ShapeDtypeStruct(c.shape, c.dtype),),
    )
    return result


def _ep(a, b):
    """Error-free product of two f32 values: a*b == p + e exactly."""
    p = jnp.multiply(a, b)
    e = _fma(a, b, -p)
    return p, e


def _two_sum(x, y):
    """Error-free f32 sum: x + y == s + e exactly (Dekker two-sum)."""
    s = x + y
    b = s - x
    e = (x - (s - b)) + (y - b)
    return s, e


def _dd_mul(a0, a1, b0, b1):
    """Double-fp32 product of (a0 + a1) * (b0 + b1) -> (p, s).

    |p + s - (a0+a1)*(b0+b1)| <= ~2^-47 * |(a0+a1)*(b0+b1)|
    (a1*b1 term is dropped, ~2^-48 of the product).
    """
    p, e = _ep(a0, b0)
    s = _fma(a0, b1, _fma(a1, b0, e))
    return p, s


def _mul3(a0, a1, a2, b0, b1, b2):
    """Three-limb f32 product of (a0+a1+a2)*(b0+b1+b2).

    Keeps every cross term with i+j <= 2; dropped terms are O(2^-72)
    relative to the leading limb even when one operand's limbs are out of
    magnitude order (the state limbs can swap order after cancellation),
    as long as each dropped term contains at least one sub-leading limb of
    the *ordered* operand (true here: coefficients are always ordered).
    """
    p00, e00 = _ep(a0, b0)
    p01, e01 = _ep(a0, b1)
    p10, e10 = _ep(a1, b0)
    p02, _ = _ep(a0, b2)
    p11, _ = _ep(a1, b1)
    p20, _ = _ep(a2, b0)
    s, r = _two_sum(p01, p10)
    s, r2 = _two_sum(s, e00)
    p1 = s
    p2 = r + r2 + e01 + e10 + p02 + p11 + p20
    return p00, p1, p2


def _sub3(a0, a1, a2, b0, b1, b2):
    """Three-limb f32 difference of (a0+a1+a2) - (b0+b1+b2).

    Dekker two-sum on the hi limbs (exact under cancellation), then the
    lower limbs folded through two-sum chains.  Exact to ~2^-96 relative
    to the leading limb.
    """
    s0, r0 = _two_sum(a0, -b0)
    t, e1 = _two_sum(r0, a1)
    t, e2 = _two_sum(t, -b1)
    t, e3 = _two_sum(t, a2)
    t, e4 = _two_sum(t, -b2)
    return s0, t, e1 + e2 + e3 + e4


def _split48(x):
    """Split an f64 value into a two-limb f32 pair (h + l ~ x, 48 bits)."""
    h = x.astype(jnp.float32)
    l = (x - h.astype(jnp.float64)).astype(jnp.float32)
    return h, l


def _split72(x):
    """Split an f64 value into a three-limb f32 tuple (~72 bits)."""
    h = x.astype(jnp.float32)
    r1 = x - h.astype(jnp.float64)
    m = r1.astype(jnp.float32)
    l = (r1 - m.astype(jnp.float64)).astype(jnp.float32)
    return h, m, l


# --- analysis (ftm -> alm) kernel -----------------------------------------


def _dfp32_analysis_kernel(
    ftm_real_ref,
    ftm_imag_ref,
    sine_ref,
    cosine_ref,
    cos_hi_ref,
    cos_mid_ref,
    cos_lo_ref,
    c1_hi_ref,
    c1_mid_ref,
    c1_lo_ref,
    c2_hi_ref,
    c2_mid_ref,
    c2_lo_ref,
    diagonal_ref,
    weight_ref,
    phase_ref,
    out_real_ref,
    out_imag_ref,
    *,
    L,
    ntheta,
    block_size,
    m_start,
):
    # Each theta-chunk writes its partial sums to its own output slab
    # [chunk, m, :]; the wrapper adds the slabs across the chunk axis in f64.
    local_m = pl.program_id(0)
    m = local_m + m_start
    m_float = m.astype(jnp.float64)
    north_count = (ntheta + 1) // 2

    def latitude_chunk(chunk, _):
        offsets = jnp.arange(block_size)
        theta = chunk * block_size + offsets
        valid = theta < north_count
        south_theta = ntheta - 1 - theta
        has_pair = valid & (south_theta != theta)

        # --- data path (f64), exactly as the fp64 kernel ---
        weight_n = plt.load(weight_ref.at[theta], mask=valid, other=0.0)
        angle_n = m_float * plt.load(phase_ref.at[theta], mask=valid, other=0.0)
        cr_n, ci_n = jnp.cos(angle_n), jnp.sin(angle_n)
        Fnr = plt.load(
            ftm_real_ref.at[local_m, theta], mask=valid, other=0.0
        )
        Fni = plt.load(
            ftm_imag_ref.at[local_m, theta], mask=valid, other=0.0
        )
        Gnr = weight_n * (cr_n * Fnr - ci_n * Fni)
        Gni = weight_n * (ci_n * Fnr + cr_n * Fni)

        weight_s = plt.load(
            weight_ref.at[south_theta], mask=has_pair, other=0.0
        )
        angle_s = m_float * plt.load(
            phase_ref.at[south_theta], mask=has_pair, other=0.0
        )
        cr_s, ci_s = jnp.cos(angle_s), jnp.sin(angle_s)
        Fsr = plt.load(
            ftm_real_ref.at[local_m, south_theta], mask=has_pair, other=0.0
        )
        Fsi = plt.load(
            ftm_imag_ref.at[local_m, south_theta], mask=has_pair, other=0.0
        )
        Gsr = weight_s * (cr_s * Fsr - ci_s * Fsi)
        Gsi = weight_s * (ci_s * Fsr + cr_s * Fsi)

        # 48-bit limbs of the parity-selected data factors (Gn +/- Gs)
        Gp_re, Gp_im = Gnr + Gsr, Gni + Gsi
        Gm_re, Gm_im = Gnr - Gsr, Gni - Gsi
        p_re_h, p_re_l = _split48(Gp_re)
        p_im_h, p_im_l = _split48(Gp_im)
        m_re_h, m_re_l = _split48(Gm_re)
        m_im_h, m_im_l = _split48(Gm_im)

        # --- seed / scale (f64), as in the fp64 kernel ---
        sine_n = plt.load(sine_ref.at[theta], mask=valid, other=1.0)
        diagonal = plt.load(diagonal_ref.at[local_m])
        log2_scale = (
            jnp.log2(jnp.abs(diagonal)) + m_float * jnp.log2(sine_n)
        )
        scale_exponent = jnp.floor(log2_scale).astype(jnp.int32)
        qmm_f64 = jnp.where(
            diagonal < 0, -jnp.ones_like(sine_n), jnp.ones_like(sine_n)
        ) * jnp.exp2(log2_scale - scale_exponent)
        # Power-of-two scale factor kept in f64.  For high m the true values
        # shrink like sin(theta)^m * diagonal, so scale_exponent reaches far
        # below -149 where an f32 exp2 underflows to 0 (silently zeroing the
        # pole contributions); the f64 kernel carries it down to -1022.  The
        # state limbs stay in [2^-90, 2^90] via the renorm, so only `fac`
        # holds the extreme magnitude -- exactly the factor f32 cannot.
        fac = jnp.exp2(scale_exponent)

        qmm_h, qmm_m, qmm_l = _split72(qmm_f64)
        cosine_n = plt.load(cosine_ref.at[theta], mask=valid, other=0.0)
        qm1_f64 = jnp.sqrt(2.0 * m_float + 3.0) * cosine_n * qmm_f64
        qm1_h, qm1_m, qm1_l = _split72(qm1_f64)
        cos_h = plt.load(cos_hi_ref.at[theta], mask=valid, other=0.0)
        cos_m = plt.load(cos_mid_ref.at[theta], mask=valid, other=0.0)
        cos_l = plt.load(cos_lo_ref.at[theta], mask=valid, other=0.0)

        # The 48-bit state/data product is computed in f32; the power-of-two
        # scale `fac` (f64) is applied only at the f64 reduction step, so it
        # never underflows even for the extreme high-m values.  `fac` is
        # passed per call because the loop-carried copy is rescaled by the
        # every-step renormalisation and must not be captured stale.
        def add_coefficient(ell, vh, vl, fac):
            par_plus = jnp.bitwise_and(ell + m, 1) == 0
            g_re_h = jnp.where(par_plus, p_re_h, m_re_h)
            g_re_l = jnp.where(par_plus, p_re_l, m_re_l)
            g_im_h = jnp.where(par_plus, p_im_h, m_im_h)
            g_im_l = jnp.where(par_plus, p_im_l, m_im_l)
            re_h, re_l = _dd_mul(vh, vl, g_re_h, g_re_l)
            im_h, im_l = _dd_mul(vh, vl, g_im_h, g_im_l)
            # fold each per-element 48-bit product to f64, apply the scale
            # factor, then a native f64 Pallas block reduction
            re_f64 = jnp.sum(
                (re_h.astype(jnp.float64) + re_l.astype(jnp.float64)) * fac
            )
            im_f64 = jnp.sum(
                (im_h.astype(jnp.float64) + im_l.astype(jnp.float64)) * fac
            )
            plt.store(out_real_ref.at[chunk, local_m, ell], re_f64)
            plt.store(out_imag_ref.at[chunk, local_m, ell], im_f64)

        # ell = m
        add_coefficient(m, qmm_h, qmm_m + qmm_l, fac)

        @pl.when(m + 1 < L)
        def add_first_off_diagonal():
            add_coefficient(m + 1, qm1_h, qm1_m + qm1_l, fac)

        state = (qmm_h, qmm_m, qmm_l, qm1_h, qm1_m, qm1_l, fac)

        def degree_step(ell, state):
            qm2_h, qm2_m, qm2_l, qm1_h, qm1_m, qm1_l, fac = state
            c1_h = plt.load(c1_hi_ref.at[local_m, ell])
            c1_m = plt.load(c1_mid_ref.at[local_m, ell])
            c1_l = plt.load(c1_lo_ref.at[local_m, ell])
            c2_h = plt.load(c2_hi_ref.at[local_m, ell])
            c2_m = plt.load(c2_mid_ref.at[local_m, ell])
            c2_l = plt.load(c2_lo_ref.at[local_m, ell])
            # t = c1 * cos ; a = t * q_{ell-1} ; b = c2 * q_{ell-2} ; cur = a - b
            t_h, t_m, t_l = _mul3(c1_h, c1_m, c1_l, cos_h, cos_m, cos_l)
            a_h, a_m, a_l = _mul3(t_h, t_m, t_l, qm1_h, qm1_m, qm1_l)
            b_h, b_m, b_l = _mul3(c2_h, c2_m, c2_l, qm2_h, qm2_m, qm2_l)
            cur_h, cur_m, cur_l = _sub3(a_h, a_m, a_l, b_h, b_m, b_l)

            # fac is applied in f64 inside add_coefficient at the reduction
            add_coefficient(ell, cur_h, cur_m + cur_l, fac)

            # every-step power-of-2 renormalisation (f32-safe window).
            # Track the magnitude over ALL state limbs: after cancellation
            # the hi limb can be tiny while the middle limb dominates.
            largest = jnp.maximum(
                jnp.maximum(jnp.abs(qm1_h), jnp.abs(qm1_m)), jnp.abs(qm1_l)
            )
            largest = jnp.maximum(
                largest,
                jnp.maximum(
                    jnp.maximum(jnp.abs(cur_h), jnp.abs(cur_m)),
                    jnp.abs(cur_l),
                ),
            )
            large = largest > 2.0**90
            small = (largest < 2.0**-90) & (largest > 0.0)
            mult = jnp.where(
                large, 2.0**-100.0, jnp.where(small, 2.0**100.0, 1.0)
            )
            # values = state * fac must stay invariant: state <- state*mult,
            # fac <- fac/mult (exact power-of-2 rescale; underflow to 0 is
            # negligible, matching the oracle's f32 fac semantics)
            fac = fac / mult
            # state slots are (qm2, qm1): next qm2 <- old qm1 (q_{ell-1}),
            # next qm1 <- cur (q_ell).  Mirrors the fp64 kernel's
            # `return qm1, current` convention.
            return (
                qm1_h * mult,
                qm1_m * mult,
                qm1_l * mult,
                cur_h * mult,
                cur_m * mult,
                cur_l * mult,
                fac,
            )

        lax.fori_loop(m + 2, L, degree_step, state)

    lax.fori_loop(
        0,
        pl.cdiv(north_count, block_size),
        latitude_chunk,
        None,
    )


def _split_np4(x):
    x = np.asarray(x, dtype=np.float64)
    h = x.astype(np.float32)
    l = (x - h.astype(np.float64)).astype(np.float32)
    return h, l


def _split_np7(x):
    x = np.asarray(x, dtype=np.float64)
    h = x.astype(np.float32)
    r1 = x - h.astype(np.float64)
    m = r1.astype(np.float32)
    l = (r1 - m.astype(np.float64)).astype(np.float32)
    return h, m, l


@partial(
    jax.jit,
    static_argnames=("L", "block_size", "m_start", "num_warps"),
)
def _dfp32_forward_latitudinal(
    ftm_real, ftm_imag, sine, cosine,
    cos_hi, cos_mid, cos_lo,
    weights, phase, *,
    L, block_size, m_start=0, num_warps=2,
):
    m_count = ftm_real.shape[0]
    ntheta = ftm_real.shape[1]
    diagonal = jnp.asarray(_sht_pallas._diagonal_normalization(L))
    c1, c2 = _sht_pallas._normalized_coefficients_numpy(
        L, m_start, m_count
    )
    c1_hi, c1_mid, c1_lo = _split_np7(c1)
    c2_hi, c2_mid, c2_lo = _split_np7(c2)
    # Each theta-chunk writes its partial sums to its own output slab
    # [chunk, m, :]; the chunk axis is reduced host-side in f64.
    north_count = (ntheta + 1) // 2
    n_chunks = (north_count + block_size - 1) // block_size
    shape = jax.ShapeDtypeStruct((n_chunks, m_count, L), jnp.float64)
    real, imag = pl.pallas_call(
        partial(
            _dfp32_analysis_kernel,
            L=L,
            ntheta=ntheta,
            block_size=block_size,
            m_start=m_start,
        ),
        out_shape=(shape, shape),
        grid=(m_count,),
        compiler_params=plt.CompilerParams(num_warps=num_warps),
        name="gmaster_scalar_analysis_dfp32_v3",
    )(
        ftm_real,
        ftm_imag,
        sine,
        cosine,
        jnp.asarray(cos_hi),
        jnp.asarray(cos_mid),
        jnp.asarray(cos_lo),
        jnp.asarray(c1_hi),
        jnp.asarray(c1_mid),
        jnp.asarray(c1_lo),
        jnp.asarray(c2_hi),
        jnp.asarray(c2_mid),
        jnp.asarray(c2_lo),
        diagonal,
        weights,
        phase,
    )
    if n_chunks == 1:
        real, imag = real[0], imag[0]
    else:
        real, imag = jnp.sum(real, axis=0), jnp.sum(imag, axis=0)
    # The kernel only writes cells with ell >= m; the out_shape buffer is not
    # zero-initialised, so explicitly mask the never-written ell < m cells.
    m_idx = jnp.arange(m_count) + m_start
    valid_ell = jnp.arange(L)[None, :] >= m_idx[:, None]  # (m_count, L)
    real = jnp.where(valid_ell, real, 0.0)
    imag = jnp.where(valid_ell, imag, 0.0)
    return (real + 1j * imag).T


def scalar_forward_latitudinal_dfp32(
    positive, theta, weights=None, phase=None, *, L, block_size=256, m_start=0,
    num_warps=2,
):
    weights = jnp.ones_like(theta) if weights is None else weights
    phase = jnp.zeros_like(theta) if phase is None else phase
    # All Pallas inputs are computed EAGERLY here on concrete arrays, then
    # passed to the jitted kernel as direct (root) operands. Two JAX 0.10
    # Pallas pitfalls are worked around:
    #   1. pallas_call lowers incorrectly when an input operand is an
    #      intermediate computed inside the enclosing trace (e.g.
    #      jnp.real(pos.T), jnp.sin(theta) computed inside the jit) -- it
    #      must be a direct trace argument. Hence the ftm/sine/cosine are
    #      built here, outside the jit.
    #   2. the cos low limbs `(cos - cos_hi.astype(f64)).astype(f32)` are
    #      off by ~2^-25 when the f32->f64 upcast is traced (corrupting the
    #      coefficients feeding the recurrence, ~1e-2..1e-7 errors).
    #      Computing them on the concrete array is exact.
    transposed = jnp.asarray(positive).T
    cosine = jnp.cos(theta)
    cos_hi = cosine.astype(jnp.float32)
    r1 = cosine - cos_hi.astype(jnp.float64)
    cos_mid = r1.astype(jnp.float32)
    cos_lo = (r1 - cos_mid.astype(jnp.float64)).astype(jnp.float32)
    return _dfp32_forward_latitudinal(
        jnp.real(transposed), jnp.imag(transposed),
        jnp.sin(theta), cosine,
        cos_hi, cos_mid, cos_lo,
        weights, phase,
        L=L, block_size=block_size, m_start=m_start, num_warps=num_warps,
    )
