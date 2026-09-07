"""Table-free spin-2 latitudinal step: the Wigner-d row is marched, not stored.

Why this exists.  At Nside 1024 the polar Wigner-d slice is 73.48 GiB (`triangle_bytes`),
`slabs_for` declines it against the 56 GiB budget, and the pipeline pays a 13.27 s/pass generic
loop while ducc0 does the *whole* spin-2 forward transform in 115 ms (HANDOFF, session 21).
Generating the row instead of storing it removes the table: one Pallas program owns one
(m-window row, theta-tile) pair, marches `ell` in registers, and contracts into the four real
polarised channels inside the same loop -- so the table route's 1:4 load-to-arithmetic ratio
becomes 1:4 FMA and the right-hand side is read once per degree instead of once per table entry.

The closed form (fit against `_spin_slice._build`, which is itself validated end to end against
pymaster; `.qwen/tmp/spin2_lowrow_fit.py`):

    d^l_{m,-s}(theta) = (-1)^m 2^(eps_m LG2N) (sin t/2)^alpha (cos t/2)^beta P^{(a,b)}_n(cos t)
    alpha = m + s,  beta = |m - s|,  n = l - max(m, s),
    LG2N = log2 sqrt((l+m)!(l-m)! / ((l-s)!(l+s)!)),  eps_m = -1 for m < s else +1.

`n` counts from `max(m, s)`, not `m`: a row below the spin starts its polynomial at ell = s.  And
DLMF 14.9.16's `(cos t/2)^(m1+m2)` admits the (m, -s) substitution only for m >= s; below the spin
the pair-swapped branch inverts the factorial ratio, which is the `eps_m` flip.  Measured against
the shipped slice, missing them costs rel 1.0 on row m=0 and 2.6e-02 on m=1 (`eps_m` recovers
5.4e-12 and 5.2e-12), while every row m >= s is unaffected -- so a march that skips this is quietly
wrong in exactly the two rows nothing else checks.

Numerics.  The march carries a float32 (value, limb) pair per theta lane and renormalises each lane
against a 2^+-24 band every degree; the products go through inline PTX because XLA has no float32
FMA here and `add(neg(mul(a,b)), mul(a,b))` folds to exactly zero residuals inside Pallas.  Both
compensations are needed -- coefficient limbs alone give 1.04e-05, state limbs alone the same,
together 5.73e-08 at Nside 256 and 1.21e-07 at 1024 against an fp64 march of the same shape
(`.qwen/tmp/spin2_lowrow_cert_256.log`, `spin2_lowrow_cert_1024.log`).  Tile partials leave in
float64 and are summed across tiles, the same discipline as `_spin_slice._reduce_channels`.

Everything the kernel reads is computed inside the trace from `cos(theta)` and static (L, nside):
no table is materialised on the host, on the device between calls, or in the module.  Selected by
`GMASTER_SPIN2_MARCH=1` at the seam in :func:`gmaster.utils._forward_latitudinal`; off by default,
and any spin other than +2 (or a non-NVIDIA device) keeps the shipped route.
"""
from __future__ import annotations

import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
from jax.scipy.special import gammaln
from s2fft.sampling import s2_samples

SPIN = 2
NC = 4                       # direct re/im, mirror re/im -- the same split as `_rhs_forward`
BLO, BHI, JUMP = 2.0 ** -24, 2.0 ** 24, 24
EX_PAD = -(10 ** 7)          # pad theta lanes: never win `emax`, never flush the real lanes
_TILE = int(os.environ.get("GMASTER_SPIN2_MARCH_TILE", "256"))
_WARPS = int(os.environ.get("GMASTER_SPIN2_MARCH_WARPS", "1"))

_CALLS: dict = {}


def _on_nvidia() -> bool:
    return any(d.platform == "gpu" and "NVIDIA" in d.device_kind.upper()
               for d in jax.devices())


def slice_declined(L, nside) -> bool:
    """True when no Wigner-d slice can be resident for this geometry, at any storage precision.

    `slabs_for` returns `(None, None)` above `_SLAB_BUDGET` and the caller drops to the generic
    scatter loop, which for spin 2 at Nside 1024 is a 13.2 s latitudinal step
    (`.qwen/tmp/score_s2_routes_1024.log`: 13 255 ms against ducc0's 112.9 ms, 0.01x).  fp32 tables
    are the cheapest the layout can ever be, so testing them is what "cannot exist" means.
    """
    from gmaster import _spin_slice as ss
    from gmaster import utils

    return ss.triangle_bytes(nside, L, utils.table_dtype()) > ss._SLAB_BUDGET


def march_requested(spin, *, L=None, nside=None) -> bool:
    """True when the analysis march should serve this call.

    `GMASTER_SPIN2_MARCH` decides outright if it is set.  Left unset the march takes the call only
    where there is no alternative -- a geometry too large for any slice (`slice_declined`) -- and
    the shipped table route keeps everything it can.  That default is the measured one: with a
    slice resident the fused table route runs Nside 512 spin 2 `map2alm` in 9.1-9.4 ms, while the
    march at that geometry takes 24.0 ms (`.qwen/tmp/twosum_check.log` section 4 -- measured with the
    table route declined, so the two numbers are not from one allocator setting; they are not to be
    read as a ratio, only as "the march is not the route to prefer when a slice exists").  With no
    slice the march runs 106.6-109.0 ms against the 13.2 s scatter loop, which is the whole 0.01x to
    parity at Nside 1024 (107.3 ms against ducc0's 105.9, `.qwen/tmp/score_default_pool.log`).  The
    cost is accuracy -- 3.7e-05 against ducc0 marched, 4.0e-07 on the scatter loop -- so
    `GMASTER_SPIN2_MARCH=0` keeps the exact route at any size.

    The seam only sees a spin-2 call when `_spin_slabs` returns `(None, None)`, which is `slabs_for`
    declining -- the same test `slice_declined` makes, so "the seam was reached" and "the march is
    the only route" are the same condition by construction. At Nside 512 the slab route owns the call
    and neither flag changes anything there: counting calls from the E/B entry points returns zero
    with `GMASTER_SPIN2_MARCH=1` set (`.qwen/tmp/route_probe_512.log`).
    """
    if int(spin) != SPIN or not _on_nvidia():
        return False
    flag = os.environ.get("GMASTER_SPIN2_MARCH")
    if flag is not None:
        return flag == "1"
    return L is not None and nside is not None and slice_declined(L, nside)


def synth_requested(spin, *, L=None, nside=None) -> bool:
    """True for the synthesis march: its own flag, otherwise the no-slice rule.

    The flag is separate from `GMASTER_SPIN2_MARCH` because the two directions have different
    economics -- analysis is ahead of ducc0 without a table while synthesis is behind it -- but the
    *default* is the same rule: march wherever `slice_declined` says no Wigner-d slice can exist, and
    keep the exact route everywhere else.  `GMASTER_SPIN2_MARCH_SYNTH=0` restores the exact route at
    any size; `=1` forces this one.

    Measured at Nside 1024 against ducc0's own map, with alms taken from ducc0's analysis (the input
    a pipeline actually produces): the marched map is **1.968e-04 max / 6.370e-06 rms** off ducc0 with
    the fitted convention factor `|s| = 1.00`, where the generic scatter loop it replaces scores
    1.095e-09 (`.qwen/tmp/synth_vs_ducc_1024.log`).  That is roughly the polar-lane row error
    (2.47e-04 at `ell = 1528`) arriving at one pole-most pixel; the map rms is 6e-06.  The cost side:
    122.7 ms against ducc0's 105.7 (0.86x) and against **13.2 s** for the scatter loop, i.e. this
    turns a 0.01x cell into ~0.8x.  It is still behind ducc0, so this is not a win, it is the removal
    of a two-order-of-magnitude default.

    One convention difference to know about.  The march's recurrence starts at `ell = max(m, spin)`,
    so it contributes nothing from `ell < |spin|`, whereas s2fft's `flm_to_ftm` and ducc0 both do use
    such terms if handed them (`.qwen/tmp/ducc_ell_below_spin.log`: injecting `|alm| = 1e3` at
    `ell < 2` moves ducc0's map by 0.14 of its maximum).  Inputs that come out of any spin-2 analysis
    have essentially nothing there -- ducc0 returns exactly 0.0 at `ell = 0` and noise level at
    `ell = 1` -- which is why this never shows up in a pipeline, but a hand-built `alm` with
    sub-spin power will silently lose it.  The number that used to be quoted for this route, rms
    0.93 (`.qwen/tmp/synth_map_err_1024.log`), was exactly that: a probe feeding a red random `alm`
    whose largest coefficients sit at `ell = 0, 1`, so only m = -1, 0, +1 columns disagreed
    (`.qwen/tmp/synth_which_columns_512.log`) while every `ell >= 2` term matched to 1.75e-06
    (`.qwen/tmp/synth_per_m_ell_256.log`).

    What is not settled is why the synthesis lane pair delivers roughly 29 bits at the extreme rows
    instead of the ~48 it costs.  A float64 march of the same recurrence holds 7.5e-13 at
    ell = 1526 (`.qwen/tmp/march_fp64_scan.log`), so it is arithmetic and not the algorithm.  Three
    suspects have been measured and eliminated: the seed (a float32-rounded seed is amplified by
    only 0.1-0.4 ulp across the whole band, `.qwen/tmp/seed_amp_probe.log`, which is why pairing it
    was inert); the primitives (`_two_prod`, `_fma` and `_two_sum` are each bit-exact on this stack,
    `.qwen/tmp/twoprod_probe.log`, `twosum_probe.log`); and the fast-form TwoSum residual, which
    *is* inexact here (~1 ulp, rms 8.2e-09 with |uh| < |th| and 2.0e-08 with |uh| > |th| against
    0.000e+00 for the branch-free 2Sum, `.qwen/tmp/twosum_order_probe.log`) but replacing it changes
    the row error by nothing -- 28 of 30 (m, ring, ell) cells of `.qwen/tmp/polar_lane_dump.py` are
    bit-identical before and after, and `ell=1528` in the pole lane is 2.47e-04 either way
    (`.qwen/tmp/polar_lane_256_fast2sum.log`, `polar_lane_256_2sum.log`).  That dump also shows the
    defect is a lane-growth law rather than a single bad ring: 1.3e-07 at the equator against 6.5e-05
    at the pole for m=0 at `ell=760`, north and south agreeing to the digit.
    """
    if int(spin) != SPIN or not _on_nvidia():
        return False
    flag = os.environ.get("GMASTER_SPIN2_MARCH_SYNTH")
    if flag is not None:
        return flag == "1"
    return L is not None and nside is not None and slice_declined(L, nside)


# --------------------------------------------------------------------------------------- kernel
def _two_prod(a, b):
    """(a*b, exact residual): the only exact-FMA pair available under this XLA/Pallas stack."""
    return plt.elementwise_inline_asm(
        "mul.rn.f32 $0, $2, $3; neg.f32 $1, $0; fma.rn.f32 $1, $2, $3, $1;",
        args=[a, b], constraints="=r,=r,r,r", pack=1,
        result_shape_dtypes=[jax.ShapeDtypeStruct(a.shape, jnp.float32),
                             jax.ShapeDtypeStruct(a.shape, jnp.float32)])


def _fma(a, b, c):
    # One output + three inputs = operands $0..$3 (numbering covers outputs; $4 does not exist).
    return plt.elementwise_inline_asm("fma.rn.f32 $0, $1, $2, $3;", args=[a, b, c],
                                      constraints="=r,r,r,r", pack=1,
                                      result_shape_dtypes=[
                                          jax.ShapeDtypeStruct(a.shape, jnp.float32)])[0]


def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _pow2(d):
    """2**d for an integer block, assembled from the exponent field; `lax.exp2` is inexact on
    integer exponents (measured 1.3e-07) and this rescale has to be exact."""
    biased = jnp.clip(d + 127, 1, 254)
    return jnp.where(d >= -126, jax.lax.bitcast_convert_type(biased << 23, jnp.float32),
                     jnp.float32(0.0))


def _pow2_f64(d):
    """The same exact power of two in float64, for rescaling an accumulator block."""
    biased = jnp.clip(d + 1023, 1, 2046).astype(jnp.int64)
    return jax.lax.bitcast_convert_type(biased << 52, jnp.float64)


def _kern(manr, ex0r, xr, xlr, c1r, c0r, cbr, c1lr, c0lr, cblr, lgnr, sgnr, rr, m0_ref, out_ref,
          *, L, ntheta, chunk):
    row = pl.program_id(0)
    tile = pl.program_id(1)
    m = plt.load(m0_ref.at[0]) + row
    nstart = jnp.maximum(m, SPIN)         # a row below the spin starts its polynomial at ell=spin
    t = tile * chunk + jnp.arange(chunk)
    valid = t < ntheta

    man = plt.load(manr.at[row, t], mask=valid, other=0.0)
    ex0 = plt.load(ex0r.at[row, t])
    ex = ex0
    xv = plt.load(xr.at[t], mask=valid, other=0.0)
    xl = plt.load(xlr.at[t], mask=valid, other=0.0)
    r = plt.load(rr.at[row, t, slice(0, NC)])
    mf = m.astype(jnp.float32)
    # P_1^(alpha,beta) = ((alpha+beta+2)/2) cos t + (alpha-beta)/2.  Not `m*cos t + 2`: that seed
    # leaves every ell >= m+1 a few percent off, growing with ell.
    alpha = mf + jnp.float32(SPIN)
    beta = jnp.abs(mf - jnp.float32(SPIN))
    p1f = ((alpha + beta + 2.0) / 2.0) * xv + ((alpha - beta) / 2.0)
    sgn = plt.load(sgnr.at[row]).astype(jnp.float64)

    def emit(cur, ex, ell):
        # One reduction for all four channels: `cur[:, None] * r` is a (chunk, NC) block, so theta
        # folds through a single Triton tree and the channels share its latency instead of running
        # four dependent ones; the partial leaves as one (NC,) float64 store.
        emax = jnp.max(ex)
        val = cur * _pow2((ex - emax).astype(jnp.int32))
        sc = jnp.exp2(emax.astype(jnp.float64) + plt.load(lgnr.at[row, ell]).astype(jnp.float64))
        parts = jnp.sum(val[:, None] * r, axis=0) * (sc * sgn)
        plt.store(out_ref.at[row, tile, ell, slice(0, NC)], parts.astype(jnp.float64))

    def degree(ell, st):
        ph, pl_, ch, cl, ex = st
        c1 = jnp.broadcast_to(plt.load(c1r.at[row, ell]).astype(jnp.float32), xv.shape)
        c0 = jnp.broadcast_to(plt.load(c0r.at[row, ell]).astype(jnp.float32), xv.shape)
        cb = jnp.broadcast_to(plt.load(cbr.at[row, ell]).astype(jnp.float32), xv.shape)
        # (c1*x + c0) to ~2^-48: the product's own residual, cos's limb, the coefficient limb.
        ah, al = _two_prod(c1, xv)
        al = _fma(c1, xl, al)
        al = _fma(jnp.broadcast_to(plt.load(c1lr.at[row, ell]), xv.shape), xv, al)
        ah, e2 = _two_sum(ah, c0)
        al = al + jnp.broadcast_to(plt.load(c0lr.at[row, ell]), xv.shape) + e2
        th, tl = _two_prod(ah, ch)
        tl = _fma(ah, cl, tl)
        tl = _fma(al, ch, tl)
        uh, ul = _two_prod(cb, ph)
        ul = _fma(cb, pl_, ul)
        ul = _fma(jnp.broadcast_to(plt.load(cblr.at[row, ell]), xv.shape), ph, ul)
        # The residual of `th - uh` comes from the branch-free 2Sum, never the fast form
        # `(th - nxt) - uh`: on this stack the fast form is off by ~1 ulp of `nxt` (rms 8.2e-09 when
        # |uh| < |th|, 2.0e-08 when |uh| > |th|) against 0.000e+00 for `_two_sum` in both orderings
        # (`.qwen/tmp/twosum_order_probe.log`).  It fixes the lane state, not the polar defect --
        # the row error at ell=1528 is 2.47e-04 either way (`.qwen/tmp/polar_lane_256_fast2sum.log`
        # against `.qwen/tmp/polar_lane_256_2sum.log`).
        nxt, e3 = _two_sum(th, -uh)
        nxtl = (tl - ul) + e3
        hh = nxt + nxtl                       # keep |nxtl| < |nxt| so the limb stays a limb
        nxtl = nxtl - (hh - nxt)
        nxt = hh
        at0 = ell == nstart
        nxt = jnp.where(at0, man, jnp.where(ell == nstart + 1, man * p1f, nxt))
        nxtl = jnp.where(at0 | (ell == nstart + 1), jnp.float32(0.0), nxtl)
        ex = jnp.where(at0, ex0, ex)
        nxt = jnp.where(valid, nxt, jnp.float32(0.0))
        nxtl = jnp.where(valid, nxtl, jnp.float32(0.0))
        big = jnp.maximum(jnp.abs(nxt), jnp.abs(ch))
        large = big > BHI
        small = (big < BLO) & (big > 0)
        mult = jnp.where(large, jnp.float32(2.0 ** -JUMP),
                         jnp.where(small, jnp.float32(2.0 ** JUMP), jnp.float32(1.0)))
        ex = ex + jnp.where(large, JUMP, jnp.where(small, -JUMP, 0)).astype(jnp.int32)
        nxt = nxt * mult
        nxtl = nxtl * mult
        emit(nxt, ex, ell)
        return (jnp.where(valid, ch * mult, jnp.float32(0.0)),
                jnp.where(valid, cl * mult, jnp.float32(0.0)), nxt, nxtl, ex)

    zb = jnp.zeros_like(man)
    lax.fori_loop(nstart, L, degree, (zb, zb, man, zb, ex))


def _call(L, ntheta, ntile, mb):
    key = (int(L), int(ntheta), int(ntile), int(mb), _TILE, _WARPS)
    call = _CALLS.get(key)
    if call is None:
        call = jax.jit(pl.pallas_call(
            partial(_kern, L=L, ntheta=ntheta, chunk=_TILE),
            out_shape=jax.ShapeDtypeStruct((mb, ntile, L, NC), jnp.float64),
            grid=(mb, ntile),
            compiler_params=plt.CompilerParams(num_warps=_WARPS),
            name="gmaster_spin2_march"))
        _CALLS[key] = call
    return call


# --------------------------------------------------------------------------------- geometry, in
def _log2_norm(m0, mb, L):
    """`log2 sqrt((l+m)!(l-m)! / ((l-s)!(l+s)!))` for one m-window, with the below-spin flip.

    Shared by the analysis emit and the synthesis coefficient split: the pair-swapped branch of
    DLMF 14.9.16 inverts the factorial ratio below the spin (module docstring), so the whole row
    `m < s` carries the negated exponent.
    """
    ms = m0 + jnp.arange(mb, dtype=jnp.float64)
    m_ = ms[:, None]
    ell_ = jnp.arange(L, dtype=jnp.float64)[None, :]
    lg2n = 0.5 * (gammaln(ell_ + m_ + 1) + gammaln(ell_ - m_ + 1)
                  - gammaln(ell_ - SPIN + 1) - gammaln(ell_ + SPIN + 1)) / np.log(2.0)
    return lg2n * jnp.where(ms >= SPIN, 1.0, -1.0)[:, None]


def _window_geometry(m0, mb, x, sh, ch, L, npad):
    """March coefficients and half-angle weights for one m-window, computed inside the trace.

    Returned float32 (high, low) so the kernel reads exactly the bits it multiplies: the limbs are
    the part the storage width throws away, and the arm needs both to hold 1e-7 at Nside 1024.
    Theta-axis arrays are padded to a whole tile here, with the pad lanes' exponent pinned to
    `EX_PAD` -- a zero there would win `emax` against real lanes near 2^-1400 and flush them.
    """
    ms = m0 + jnp.arange(mb, dtype=jnp.float64)
    m_ = ms[:, None]
    ell_ = jnp.arange(L, dtype=jnp.float64)[None, :]
    alpha = m_ + SPIN
    beta = jnp.abs(m_ - SPIN)
    ns = jnp.maximum(m_, SPIN)
    n_ = jnp.where(ell_ >= ns, ell_ - ns, 0.0)
    s_ = 2.0 * n_ + alpha + beta
    den = jnp.where(n_ >= 2, 2.0 * n_ * (n_ + alpha + beta) * (s_ - 2.0), 1.0)
    a1 = (s_ - 1.0) * s_ * (s_ - 2.0) / den
    a0 = (s_ - 1.0) * (4.0 * m_ * SPIN) / den
    bb = 2.0 * (n_ + alpha - 1.0) * (n_ + beta - 1.0) * s_ / den
    log2s = alpha * jnp.log2(sh)[None, :] + beta * jnp.log2(ch)[None, :]
    ex0 = jnp.floor(log2s)
    mant = jnp.exp2(log2s - ex0)
    lgs = _log2_norm(m0, mb, L)
    sgn = (-1.0) ** ms

    def hi_lo(a):
        h = a.astype(jnp.float32)
        return h, (a - h.astype(jnp.float64)).astype(jnp.float32)

    def pad(a, fill=0.0):
        if a.shape[-1] == npad:
            return a
        return jnp.full(a.shape[:-1] + (npad,), fill, dtype=a.dtype).at[..., :a.shape[-1]].set(a)

    a1h, a1l = hi_lo(a1)
    a0h, a0l = hi_lo(a0)
    bhh, bhl = hi_lo(bb)
    xh, xl = hi_lo(x)
    return (pad(mant.astype(jnp.float32)), pad(ex0.astype(jnp.int32), EX_PAD),
            pad(xh), pad(xl),
            a1h, a0h, bhh, a1l, a0l, bhl, lgs, sgn.astype(jnp.float64))


# --------------------------------------------------------------------------------------- driver
@partial(jax.jit, static_argnames=("L", "spin", "nside"))
def _forward_impl(ftm, *, L, spin, nside):
    from gmaster import _spin_slice as ss

    ftm = jnp.asarray(ftm)
    theta = (jnp.asarray(s2_samples.thetas(L, "healpix", nside), dtype=jnp.float64)
             + 8 * jnp.finfo(jnp.float64).eps)      # the same grid `utils._stable_thetas` gives
    ntheta = theta.shape[0]
    ntile = -(-ntheta // _TILE)
    npad = ntile * _TILE
    x = jnp.cos(theta)
    sh, ch = jnp.sin(theta / 2.0), jnp.cos(theta / 2.0)
    off = L - 1
    rev = ftm[::-1]                                  # ring i of rev is the ring at pi - theta_i
    sign = ss._sign(L, SPIN)
    ell = jnp.arange(L)[:, None]
    out = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
    for (m0, m1, lo) in ss._windows(L):
        mb = m1 - m0
        g = _window_geometry(m0, mb, x, sh, ch, L, npad)
        direct = ftm[:, L + m0:L + m1]
        mirror = rev[:, L - m1 + 1:L - m0 + 1][:, ::-1]
        chan = jnp.stack([direct.real.T, direct.imag.T, mirror.real.T, mirror.imag.T], axis=-1)
        rhs = jnp.zeros((mb, npad, NC), dtype=jnp.float32).at[:, :ntheta].set(
            lax.convert_element_type(chan, jnp.float32))
        parts = _call(L, ntheta, ntile, mb)(*g, rhs, jnp.asarray([m0], jnp.int32)).sum(1)
        parts = parts.transpose(1, 0, 2)             # (m, ell, channel) -> (ell, m, channel)
        # Each row starts emitting at ell = max(m, spin) and never writes below it, so whatever the
        # allocator handed back under that has to be zeroed before it reaches the assembly.
        acc = jnp.where(
            (ell >= jnp.maximum(jnp.arange(mb)[None, :] + m0, SPIN))[..., None], parts, 0.0)
        # `acc` is already (ell, m, channel); the shipped kernel returns (m, ell, channel) and
        # transposes here, which is the only difference in this assembly.
        out = out.at[lo:, off + m0:off + m1].set(acc[lo:, :, 0] + 1j * acc[lo:, :, 1])
        mir = sign[lo:, None] * (acc[lo:, :, 2] + 1j * acc[lo:, :, 3])
        if m0:
            out = out.at[lo:, off - m1 + 1:off - m0 + 1].set(mir[:, ::-1])
        else:
            # Column off is the direct channel alone: m = 0 is its own mirror and does not
            # satisfy the theta -> pi-theta relation.
            out = out.at[lo:, off - m1 + 1:off].set(mir[:, 1:][:, ::-1])
    return out


def forward_latitudinal(ftm, *, L, spin, nside):
    """Analysis latitudinal step with no Wigner-d table.

    Same contract as :func:`gmaster._spin_slice.forward_latitudinal`: ``(L, 2L-1)`` complex with
    ``flm[ell, L-1+m]``, negative orders through the pi-theta symmetry, zeros below
    ``ell = max(|m|, spin)``.  The quadrature weight and ring phase shifts are already in ``ftm``
    and the ``sqrt((2l+1)/4pi)`` / ``(-1)**|spin|`` factors come later in
    ``_finish_forward_s2fft``; neither belongs here.
    """
    if int(spin) != SPIN:
        raise ValueError(f"march route implements spin=+{SPIN}, got spin={spin}")
    return _forward_impl(ftm, L=L, spin=spin, nside=nside)


# ------------------------------------------------------------------- synthesis: the same row, summed
_ST = int(os.environ.get("GMASTER_SPIN2_MARCH_SYNTH_TILE", "512"))
_SW = int(os.environ.get("GMASTER_SPIN2_MARCH_SYNTH_WARPS", "4"))
# `fast` accumulates the contraction in plain float32, `comp` in (value, limb) float32 pairs; see
# `_accumulate_fast` for the measured cost and error of the difference.
_ACC = os.environ.get("GMASTER_SPIN2_MARCH_SYNTH_ACC", "fast")
_ACC_FAST = _ACC != "comp"


def _add_pair(hh, lo, ph, pl):
    """(hh, lo) += (ph, pl) for two compensated float32 pairs, result renormalised."""
    sh, se = _two_sum(hh, ph)
    sl = lo + (pl + se)
    nh = sh + sl
    return nh, sl - (nh - sh)


def _accumulate(nd, ndl, tex, aex, at0, rh, rl, ih, il, a_r, a_l, b_r, b_l):
    """Add one degree's term to a per-lane (value, limb) accumulator pair; return the new state.

    `tex` is the term's exponent and `aex` the accumulator's, and they meet through the exact power
    `2**(tex - aex)`: no `2**+-1400` is ever formed, and the alignment costs one multiply on the
    value and one on its limb.

    The invariant is `value * 2**aex`, so a block that moves has to carry the stored value with it.
    A term that dwarfs the accumulator lifts the block and flushes the residue; a term it dwarfs
    simply underflows in -- the block never moves down, because that would grow the stored value
    toward overflow.  Clamping the drift in both directions without rescaling, as the first version
    did, silently multiplies whatever had accumulated by `2**drift`.  That is what put the synthesis
    step at 7.08e-04 against the shipped route at Nside 1024 while the identical march gives
    1.46e-07 in analysis (`.qwen/tmp/march_knob_1024.log`): analysis reduces over theta, so a lane
    whose residue got relabelled is diluted, while synthesis emits that lane as a pixel.
    """
    aex = jnp.where(at0, tex, aex)
    lift = jnp.maximum(tex - aex - 90, 0).astype(jnp.int32)
    aex = aex + lift
    res = _pow2(-lift)
    a_r, a_l, b_r, b_l = a_r * res, a_l * res, b_r * res, b_l * res
    ds = _pow2((tex - aex).astype(jnp.int32))
    nd, ndl = nd * ds, ndl * ds
    pr, pl = _two_prod(nd, rh)
    pl = _fma(nd, rl, pl)
    pl = _fma(ndl, rh, pl)
    pi, pli = _two_prod(nd, ih)
    pli = _fma(nd, il, pli)
    pli = _fma(ndl, ih, pli)
    a_r, a_l = _add_pair(a_r, a_l, pr, pl)
    b_r, b_l = _add_pair(b_r, b_l, pi, pli)
    amax = jnp.maximum(jnp.abs(a_r), jnp.abs(b_r))
    abig = amax > BHI
    asmall = (amax < BLO) & (amax > 0)
    amult = jnp.where(abig, jnp.float32(2.0 ** -JUMP),
                      jnp.where(asmall, jnp.float32(2.0 ** JUMP), jnp.float32(1.0)))
    aex = aex + jnp.where(abig, JUMP, jnp.where(asmall, -JUMP, 0)).astype(jnp.int32)
    return a_r * amult, a_l * amult, b_r * amult, b_l * amult, aex


def _accumulate_fast(nd, tex, aex, at0, rh, ih, a_r, b_r):
    """Plain-float32 accumulation of one degree's term: same range guards, no limbs.

    The range half of `_accumulate` (the `lift`/`ds` alignment and the 2^+-24 accumulator guard) is
    what keeps a 6000-term sum inside float32's exponent range and is kept verbatim.  What is
    dropped is the *precision* half: the limb operands (neither the marched limb nor the
    coefficient limbs are read at all, so the kernel does not even load them) and the
    accumulator limb.  At 4 elements/thread that is 61 of the step's ~165 vector operations, and the
    step is warp-issue-bound at ~100% of the machine's issue rate, so the saving is the op count
    itself: 83.41 -> 51.96 ms over all 48 windows at Nside 1024 (1.605x), with max|diff| 1.260e-07
    against the compensated arm and 2.15e-06 relative to the window's own maximum
    (`.qwen/tmp/synth_acc_cost_1024.log`).
    """
    aex = jnp.where(at0, tex, aex)
    lift = jnp.maximum(tex - aex - 90, 0).astype(jnp.int32)
    aex = aex + lift
    res = _pow2(-lift)
    a_r, b_r = a_r * res, b_r * res
    ds = _pow2((tex - aex).astype(jnp.int32))
    nd = nd * ds
    a_r = a_r + nd * rh
    b_r = b_r + nd * ih
    amax = jnp.maximum(jnp.abs(a_r), jnp.abs(b_r))
    abig = amax > BHI
    asmall = (amax < BLO) & (amax > 0)
    amult = jnp.where(abig, jnp.float32(2.0 ** -JUMP),
                      jnp.where(asmall, jnp.float32(2.0 ** JUMP), jnp.float32(1.0)))
    aex = aex + jnp.where(abig, JUMP, jnp.where(asmall, -JUMP, 0)).astype(jnp.int32)
    return a_r * amult, b_r * amult, aex


def _kern_synth(manr, ex0r, xr, xlr, c1r, c0r, cbr, c1lr, c0lr, cblr,
                lexdr, drh, drl, dih, dil, lexmr, mrh, mrl, mih, mil,
                m0_ref, out_ref, *, L, ntheta, chunk):
    """One (m-window row, theta tile): march the row once, sum it against both order signs.

    The synthesis contraction is the transpose of the analysis sum, so it keeps a per-lane
    accumulator over `ell` rather than reducing theta at every degree.  Positive and negative
    orders need the *same* row: the negative side is ``R_m(pi - theta)``, and because the HEALPix
    theta grid is symmetric (``theta[ntheta-1-i] == pi - theta[i]``) that is the same marched lane
    read at the mirrored ring.  So one march feeds both accumulators and only the store index
    differs -- before this the negative orders ran a second, identical march on a `-cos` geometry,
    which is why synthesis cost 197.7 ms against analysis' 111.6 at Nside 1024.

    The accumulation is plain float32 by default; `GMASTER_SPIN2_MARCH_SYNTH_ACC=comp` restores the
    (value, limb) pairs and `_accumulate_fast` records what the two cost.  Float64 accumulators are
    not an option on this GPU, which runs float64 at 1/64 rate -- an fp64 accumulator alone cost
    793.9 ms for this step (`.qwen/tmp/sht_vs_ducc_s29_ab2_1024_2.log`).  Output channel 0/1 is the
    direct (positive m) sum, 2/3 the mirror (negative m) sum, which the caller stores at the
    reversed theta axis.
    """
    row = pl.program_id(0)
    tile = pl.program_id(1)
    m = plt.load(m0_ref.at[0]) + row
    nstart = jnp.maximum(m, SPIN)
    t = tile * chunk + jnp.arange(chunk)
    valid = t < ntheta

    man = plt.load(manr.at[row, t], mask=valid, other=0.0)
    ex0 = plt.load(ex0r.at[row, t])
    ex = ex0
    xv = plt.load(xr.at[t], mask=valid, other=0.0)
    xl = plt.load(xlr.at[t], mask=valid, other=0.0)
    mf = m.astype(jnp.float32)
    alpha = mf + jnp.float32(SPIN)
    beta = jnp.abs(mf - jnp.float32(SPIN))
    p1f = ((alpha + beta + 2.0) / 2.0) * xv + ((alpha - beta) / 2.0)

    def degree(ell, st):
        if _ACC_FAST:
            (ph, pl_, ch, cl, ex, adr, adi, aexd, amr, ami, aexm) = st
        else:
            (ph, pl_, ch, cl, ex, adr, adl, adi, adli, aexd,
             amr, aml, ami, amli, aexm) = st
        c1 = jnp.broadcast_to(plt.load(c1r.at[row, ell]).astype(jnp.float32), xv.shape)
        c0 = jnp.broadcast_to(plt.load(c0r.at[row, ell]).astype(jnp.float32), xv.shape)
        cb = jnp.broadcast_to(plt.load(cbr.at[row, ell]).astype(jnp.float32), xv.shape)
        ah, al = _two_prod(c1, xv)
        al = _fma(c1, xl, al)
        al = _fma(jnp.broadcast_to(plt.load(c1lr.at[row, ell]), xv.shape), xv, al)
        ah, e2 = _two_sum(ah, c0)
        al = al + jnp.broadcast_to(plt.load(c0lr.at[row, ell]), xv.shape) + e2
        th, tl = _two_prod(ah, ch)
        tl = _fma(ah, cl, tl)
        tl = _fma(al, ch, tl)
        uh, ul = _two_prod(cb, ph)
        ul = _fma(cb, pl_, ul)
        ul = _fma(jnp.broadcast_to(plt.load(cblr.at[row, ell]), xv.shape), ph, ul)
        # The residual of `th - uh` comes from the branch-free 2Sum, never the fast form
        # `(th - nxt) - uh`: on this stack the fast form is off by ~1 ulp of `nxt` (rms 8.2e-09 when
        # |uh| < |th|, 2.0e-08 when |uh| > |th|) against 0.000e+00 for `_two_sum` in both orderings
        # (`.qwen/tmp/twosum_order_probe.log`).  It fixes the lane state, not the polar defect --
        # the row error at ell=1528 is 2.47e-04 either way (`.qwen/tmp/polar_lane_256_fast2sum.log`
        # against `.qwen/tmp/polar_lane_256_2sum.log`).
        nxt, e3 = _two_sum(th, -uh)
        nxtl = (tl - ul) + e3
        hh = nxt + nxtl
        nxtl = nxtl - (hh - nxt)
        nxt = hh
        at0 = ell == nstart
        nxt = jnp.where(at0, man, jnp.where(ell == nstart + 1, man * p1f, nxt))
        nxtl = jnp.where(at0 | (ell == nstart + 1), jnp.float32(0.0), nxtl)
        ex = jnp.where(at0, ex0, ex)
        nxt = jnp.where(valid, nxt, jnp.float32(0.0))
        nxtl = jnp.where(valid, nxtl, jnp.float32(0.0))
        big = jnp.maximum(jnp.abs(nxt), jnp.abs(ch))
        large = big > BHI
        small = (big < BLO) & (big > 0)
        dmult = jnp.where(large, jnp.float32(2.0 ** -JUMP),
                          jnp.where(small, jnp.float32(2.0 ** JUMP), jnp.float32(1.0)))
        ex = ex + jnp.where(large, JUMP, jnp.where(small, -JUMP, 0)).astype(jnp.int32)
        nxt = nxt * dmult
        nxtl = nxtl * dmult

        # Both sums consume the same marched value; only the coefficient set and the block
        # exponent differ.  Pad lanes track their own accumulator so `EX_PAD` cannot poison it.
        lex = jnp.broadcast_to(plt.load(lexdr.at[row, ell]), xv.shape)
        chs = jnp.where(valid, ch * dmult, jnp.float32(0.0))
        cls = jnp.where(valid, cl * dmult, jnp.float32(0.0))
        if _ACC_FAST:
            tex = jnp.where(valid, ex + lex, aexd)
            adr, adi, aexd = _accumulate_fast(
                nxt, tex, aexd, at0,
                jnp.broadcast_to(plt.load(drh.at[row, ell]), xv.shape),
                jnp.broadcast_to(plt.load(dih.at[row, ell]), xv.shape),
                adr, adi)
            lexm = jnp.broadcast_to(plt.load(lexmr.at[row, ell]), xv.shape)
            texm = jnp.where(valid, ex + lexm, aexm)
            amr, ami, aexm = _accumulate_fast(
                nxt, texm, aexm, at0,
                jnp.broadcast_to(plt.load(mrh.at[row, ell]), xv.shape),
                jnp.broadcast_to(plt.load(mih.at[row, ell]), xv.shape),
                amr, ami)
            return chs, cls, nxt, nxtl, ex, adr, adi, aexd, amr, ami, aexm
        tex = jnp.where(valid, ex + lex, aexd)
        adr, adl, adi, adli, aexd = _accumulate(
            nxt, nxtl, tex, aexd, at0,
            jnp.broadcast_to(plt.load(drh.at[row, ell]), xv.shape),
            jnp.broadcast_to(plt.load(drl.at[row, ell]), xv.shape),
            jnp.broadcast_to(plt.load(dih.at[row, ell]), xv.shape),
            jnp.broadcast_to(plt.load(dil.at[row, ell]), xv.shape),
            adr, adl, adi, adli)
        lexm = jnp.broadcast_to(plt.load(lexmr.at[row, ell]), xv.shape)
        texm = jnp.where(valid, ex + lexm, aexm)
        amr, aml, ami, amli, aexm = _accumulate(
            nxt, nxtl, texm, aexm, at0,
            jnp.broadcast_to(plt.load(mrh.at[row, ell]), xv.shape),
            jnp.broadcast_to(plt.load(mrl.at[row, ell]), xv.shape),
            jnp.broadcast_to(plt.load(mih.at[row, ell]), xv.shape),
            jnp.broadcast_to(plt.load(mil.at[row, ell]), xv.shape),
            amr, aml, ami, amli)
        return (chs, cls, nxt, nxtl, ex,
                adr, adl, adi, adli, aexd, amr, aml, ami, amli, aexm)

    zb = jnp.zeros_like(man)
    zi = jnp.zeros(xv.shape, jnp.int32)
    if _ACC_FAST:
        (_, _, _, _, _, adr, adi, aexd,
         amr, ami, aexm) = lax.fori_loop(
            nstart, L, degree, (zb, zb, man, zb, ex, zb, zb, zi, zb, zb, zi))
        sd = _pow2_f64(aexd)
        sm_ = _pow2_f64(aexm)
        plt.store(out_ref.at[row, tile, slice(0, chunk), slice(0, 2)],
                  jnp.stack([adr.astype(jnp.float64) * sd,
                             adi.astype(jnp.float64) * sd], axis=-1))
        plt.store(out_ref.at[row, tile, slice(0, chunk), slice(2, 4)],
                  jnp.stack([amr.astype(jnp.float64) * sm_,
                             ami.astype(jnp.float64) * sm_], axis=-1))
        return
    (_, _, _, _, _, adr, adl, adi, adli, aexd,
     amr, aml, ami, amli, aexm) = lax.fori_loop(
        nstart, L, degree, (zb, zb, man, zb, ex, zb, zb, zb, zb, zi,
                            zb, zb, zb, zb, zi))
    sd = _pow2_f64(aexd)
    sm_ = _pow2_f64(aexm)
    # Two stores, not one 4-channel `stack`: Pallas lowers concatenate only in arity 2.
    plt.store(out_ref.at[row, tile, slice(0, chunk), slice(0, 2)],
              jnp.stack([(adr.astype(jnp.float64) + adl.astype(jnp.float64)) * sd,
                         (adi.astype(jnp.float64) + adli.astype(jnp.float64)) * sd], axis=-1))
    plt.store(out_ref.at[row, tile, slice(0, chunk), slice(2, 4)],
              jnp.stack([(amr.astype(jnp.float64) + aml.astype(jnp.float64)) * sm_,
                         (ami.astype(jnp.float64) + amli.astype(jnp.float64)) * sm_], axis=-1))


def _call_synth(L, ntheta, ntile, mb):
    key = ("synth", int(L), int(ntheta), int(ntile), int(mb), _ST, _SW)
    call = _CALLS.get(key)
    if call is None:
        call = jax.jit(pl.pallas_call(
            partial(_kern_synth, L=L, ntheta=ntheta, chunk=_ST),
            out_shape=jax.ShapeDtypeStruct((mb, ntile, _ST, 4), jnp.float64),
            grid=(mb, ntile),
            compiler_params=plt.CompilerParams(num_warps=_SW),
            name="gmaster_spin2_synth"))
        _CALLS[key] = call
    return call


def _synth_coeff(flm, m0, mb, L, *, mirror):
    """Coefficient sequence and exponent split for one window of orders.

    ``lgs`` is split into ``lex`` (integer, folded into the term exponent) and a ``[1, 2)`` factor
    that rides in the float32 coefficient together with the closed form's ``(-1)**m``; the mirror
    half carries the slice's own ``(-1)**(ell + |spin|)``, which the delta probe
    (``.qwen/tmp/spin2_synth_negm_64.log``) shows is exactly the ``theta -> pi - theta`` factor the
    shipped synthesis applies to negative orders.

    Split into a float32 high part plus limb: the contraction inside the kernel is float32
    (float64 is 1/64-rate here), so the coefficient keeps its precision only as a pair -- a
    float32 coefficient alone is re-spent at every degree of the same lane and its error grows
    like L * 2**-24 (measured 5.9e-06 at Nside 32).
    """
    from gmaster import _spin_slice as ss

    ms = m0 + jnp.arange(mb)
    cols = L - 1 + ms if not mirror else L - 1 - ms
    c = flm[:, cols]                                   # (L, mb) complex
    if mirror:
        c = c * ss._sign(L, SPIN)[:, None]
    lgs = _log2_norm(m0, mb, L)                        # (mb, L) float64
    lex = jnp.floor(lgs)
    sc = jnp.exp2(lgs - lex) * ((-1.0) ** ms)[:, None]
    out = []
    for part in (c.real.T * sc, c.imag.T * sc):
        hi = part.astype(jnp.float32)
        out.append(hi)
        out.append((part - hi.astype(jnp.float64)).astype(jnp.float32))
    drh, drl, dih, dil = out
    return lex.astype(jnp.int32), drh, drl, dih, dil


@partial(jax.jit, static_argnames=("L", "spin", "nside"))
def _inverse_impl(flm, *, L, spin, nside):
    from gmaster import _spin_slice as ss

    flm = jnp.asarray(flm)
    theta = (jnp.asarray(s2_samples.thetas(L, "healpix", nside), dtype=jnp.float64)
             + 8 * jnp.finfo(jnp.float64).eps)
    ntheta = theta.shape[0]
    ntile = -(-ntheta // _ST)
    npad = ntile * _ST
    x = jnp.cos(theta)
    sh, ch = jnp.sin(theta / 2.0), jnp.cos(theta / 2.0)
    # The window column ranges are disjoint and tile the result, so the result is assembled by one
    # concatenation instead of `out.at[...].set` per window.  XLA scatter is out-of-place, so the
    # per-window form copied the whole (ntheta, 2L) complex128 buffer 48 times at Nside 1024 and 96
    # times at 2048 -- 1.50 GiB per copy at the latter -- which was 65% and 81% of this step:
    # 138.75 -> 47.29 ms (2.93x) and 1805.60 -> 346.99 ms (5.20x), bit-identical output
    # (`.qwen/tmp/synth_assembly_ab.log`).
    dirs, mirs = [], []
    for (m0, m1, lo) in ss._windows(L):
        mb = m1 - m0
        g = _window_geometry(m0, mb, x, sh, ch, L, npad)
        dc = _synth_coeff(flm, m0, mb, L, mirror=False)
        mc = _synth_coeff(flm, m0, mb, L, mirror=True)
        if m0 == 0:
            # m = 0 is its own mirror and has no negative column, so its mirror accumulator is
            # simply never fed; every other row of this window is a genuine negative order.
            keep = (jnp.arange(mb) != 0)[:, None]
            mc = tuple(jnp.where(keep, a, jnp.zeros_like(a)) for a in mc)
        v = _call_synth(L, ntheta, ntile, mb)(*g[:10], *dc, *mc,
                                              jnp.asarray([m0], jnp.int32))
        v = v.reshape(mb, npad, 4)[:, :ntheta]
        dirs.append(v[:, :, 0].T + 1j * v[:, :, 1].T)             # columns L+m0 .. L+m1-1
        # `R_m(pi - theta_i)` is the marched lane at the mirrored ring, so the mirror accumulator
        # lands at the reversed theta axis; its rows are descending orders by the column layout.
        n0 = max(int(m0), 1)
        if n0 < m1:
            re = v[n0 - m0:, :, 2][::-1, ::-1].T
            im = v[n0 - m0:, :, 3][::-1, ::-1].T
            mirs.append(re + 1j * im)                             # columns L-m1+1 .. L-n0
    direct = jnp.concatenate(dirs, axis=1)                        # (ntheta, L)
    # Ascending windows cover descending columns, so the mirror half goes in reverse window order;
    # column 0 is never written by either half and stays the zero the contract asks for.
    mir = jnp.concatenate(mirs[::-1], axis=1)                     # (ntheta, L-1)
    return jnp.concatenate([jnp.zeros((ntheta, 1), dtype=direct.dtype), mir, direct], axis=1)



def inverse_latitudinal(flm, *, L, spin, nside):
    """Synthesis latitudinal step with no Wigner-d table.

    ``(L, 2L-1)`` complex in (``flm[ell, L-1+m]``) to ``(4*nside-1, 2L)`` complex, the contract
    `utils._inverse_latitudinal` has today; the kernel is the same certified ``d^l_(m,-2)`` the
    analysis march uses, measured against the shipped route at rel 1e-14 with unit scalar
    (``.qwen/tmp/spin2_synth_row2_64.log``).
    """
    if int(spin) != SPIN:
        raise ValueError(f"march route implements spin=+{SPIN}, got spin={spin}")
    return _inverse_impl(flm, L=L, spin=spin, nside=nside)


def clear_cache():
    """Drop the compiled kernels (tests, or to free the device)."""
    _CALLS.clear()
