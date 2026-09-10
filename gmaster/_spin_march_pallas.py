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
# Ceiling on programs per march launch for the wide m-window (see :func:`_march_windows`).
_MARCH_GRID_CAP = int(os.environ.get("GMASTER_MARCH_GRID_CAP", "2048"))

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
          *, L, ntheta, chunk, spin=SPIN, nc=NC):
    row = pl.program_id(0)
    tile = pl.program_id(1)
    m0 = plt.load(m0_ref.at[0])
    m = m0 + row
    nstart = jnp.maximum(m, spin)          # a row below the spin starts its polynomial at ell=spin
    t = tile * chunk + jnp.arange(chunk)
    valid = t < ntheta

    # The slab's `ell` axis is the window's own range (`L - m0`): this row writes `L - nstart` lanes
    # at index `ell - m0`, and its head of `nstart - m0 <= mb - 1` lanes is the only part of the slab
    # the march itself never reaches, so the program zeroes it rather than leaving it uninitialized.
    # One lane per iteration, matching `emit`'s store width: the Triton lowering requires every
    # operation's array to be a power of two in size, and a ragged last window makes `mb` something
    # like 96 (nside 32, L = 96), so a `(mb, NC)` head block fails to compile on those geometries.
    def zero_head(i, carry):
        plt.store(out_ref.at[row, tile, i, slice(0, nc)],
                  jnp.zeros((nc,), dtype=jnp.float64))
        return carry

    lax.fori_loop(0, jnp.maximum(nstart - m0, 0), zero_head, ())

    man = plt.load(manr.at[row, t], mask=valid, other=0.0)
    ex0 = plt.load(ex0r.at[row, t])
    ex = ex0
    xv = plt.load(xr.at[t], mask=valid, other=0.0)
    xl = plt.load(xlr.at[t], mask=valid, other=0.0)
    r = plt.load(rr.at[row, t, slice(0, nc)])
    mf = m.astype(jnp.float32)
    # P_1^(alpha,beta) = ((alpha+beta+2)/2) cos t + (alpha-beta)/2.  Not `m*cos t + 2`: that seed
    # leaves every ell >= m+1 a few percent off, growing with ell.
    alpha = mf + jnp.float32(spin)
    beta = jnp.abs(mf - jnp.float32(spin))
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
        plt.store(out_ref.at[row, tile, ell - m0, slice(0, nc)], parts.astype(jnp.float64))

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


def _call(L, ntheta, ntile, mb, spin=SPIN, Lm=None, nc=NC):
    # `Lm` is the slab's ell extent: the window's own range `L - m0`, which is what the row writes
    # (`L - nstart` lanes at index `ell - m0`) plus the head it zeroes.
    Lm = L if Lm is None else Lm
    key = (int(L), int(ntheta), int(ntile), int(mb), int(Lm), _TILE, _WARPS, int(spin), int(nc))
    call = _CALLS.get(key)
    if call is None:
        call = jax.jit(pl.pallas_call(
            partial(_kern, L=L, ntheta=ntheta, chunk=_TILE, spin=spin, nc=nc),
            out_shape=jax.ShapeDtypeStruct((mb, ntile, Lm, nc), jnp.float64),
            grid=(mb, ntile),
            compiler_params=plt.CompilerParams(num_warps=_WARPS),
            name=f"gmaster_spin{int(spin)}_march"))
        _CALLS[key] = call
    return call


# --------------------------------------------------------------------------------- geometry, in
def _log2_norm(m0, mb, L, spin=SPIN):
    """`log2 sqrt((l+m)!(l-m)! / ((l-s)!(l+s)!))` for one m-window, with the below-spin flip.

    Shared by the analysis emit and the synthesis coefficient split: the pair-swapped branch of
    DLMF 14.9.16 inverts the factorial ratio below the spin (module docstring), so the whole row
    `m < s` carries the negated exponent.  At `spin = 0` the flip never fires (`m >= 0` always) and
    the ratio collapses to `sqrt((l+m)!(l-m)!)/l!`, which is the `d^l_{m,0}` normalisation the
    folded route marches.
    """
    ms = m0 + jnp.arange(mb, dtype=jnp.float64)
    m_ = ms[:, None]
    ell_ = jnp.arange(L, dtype=jnp.float64)[None, :]
    lg2n = 0.5 * (gammaln(ell_ + m_ + 1) + gammaln(ell_ - m_ + 1)
                  - gammaln(ell_ - spin + 1) - gammaln(ell_ + spin + 1)) / np.log(2.0)
    return lg2n * jnp.where(ms >= spin, 1.0, -1.0)[:, None]


def _window_geometry(m0, mb, x, sh, ch, L, npad, spin=SPIN):
    """March coefficients and half-angle weights for one m-window, computed inside the trace.

    Returned float32 (high, low) so the kernel reads exactly the bits it multiplies: the limbs are
    the part the storage width throws away, and the arm needs both to hold 1e-7 at Nside 1024.
    Theta-axis arrays are padded to a whole tile here, with the pad lanes' exponent pinned to
    `EX_PAD` -- a zero there would win `emax` against real lanes near 2^-1400 and flush them.

    `spin` enters through alpha = m + s, beta = |m - s| and the three Jacobi coefficients, so the
    same builder serves the folded spin-0 march: there `a0 = (s-1) * 4ms / den` is identically zero
    and the recurrence reduces to the plain symmetric-Jacobi (Legendre at m = 0) three-term.
    """
    ms = m0 + jnp.arange(mb, dtype=jnp.float64)
    m_ = ms[:, None]
    ell_ = jnp.arange(L, dtype=jnp.float64)[None, :]
    alpha = m_ + spin
    beta = jnp.abs(m_ - spin)
    ns = jnp.maximum(m_, spin)
    n_ = jnp.where(ell_ >= ns, ell_ - ns, 0.0)
    s_ = 2.0 * n_ + alpha + beta
    den = jnp.where(n_ >= 2, 2.0 * n_ * (n_ + alpha + beta) * (s_ - 2.0), 1.0)
    a1 = (s_ - 1.0) * s_ * (s_ - 2.0) / den
    a0 = (s_ - 1.0) * (4.0 * m_ * spin) / den
    bb = 2.0 * (n_ + alpha - 1.0) * (n_ + beta - 1.0) * s_ / den
    log2s = alpha * jnp.log2(sh)[None, :] + beta * jnp.log2(ch)[None, :]
    ex0 = jnp.floor(log2s)
    mant = jnp.exp2(log2s - ex0)
    lgs = _log2_norm(m0, mb, L, spin)
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
def _march_windows(L, ntile, spin):
    """The m-windows of one march launch of *ntile* theta tiles at *spin*.

    One program covers one ``(m, theta tile)`` pair, so the window width changes neither the total
    work nor the total program count -- only how it is split: ``L/mb`` launches of ``mb*ntile``
    programs.  With no slab to skip bytes in (the whole point of the marched routes) nothing argues
    against a wide window, so 128 is taken -- until a launch passes ``_MARCH_GRID_CAP`` programs,
    where it starts to lose.  Measured 128 over 64, arms alternating in one process
    (`.qwen/tmp/march_mblock_ab.log`) and one size per process for the rest:

    | cell | programs per launch, 64 -> 128 | 64 | 128 |
    |---|---|---|---|
    | spin 2 analysis, nside 1024 | 1024 -> 2048 | 107.14 ms | **78.57** (**1.36x**) |
    | spin-0 fold analysis, nside 2048 | 2048 -> 2048 | 419.42 ms | **312.04** (**1.34x**) |
    | spin 2 synthesis, nside 2048 | 1024 -> 2048 | 525.70 ms | **473.49** (**1.11x**) |
    | spin-0 fold synthesis, nside 2048 | 1024 -> 2048 | 325.03 ms | **303.59** (**1.07x**) |
    | spin 0, nside 1024, both directions | <= 1024 | 28.14 / 31.14 ms | 28.13 / 30.54 (parity) |
    | spin 2 analysis, nside 2048 | 2048 -> **4096** | **610.69 ms** | 634.22 (**0.96x**) |
    | spin-0 fold analysis, nside 4096 | 2048 -> **4096** | **2253.8 ms** | 2386.7 (**0.94x**) |

    Every win is at a launch of 2048 programs or fewer and both losses at 4096.  The alms are
    bit-identical except at nside 1024 spin 2, where they move by 4.3e-19 against
    ``|alm|max 2.7e-03`` -- different summation grouping, last-bit.  Against ducc0 the two flips
    that matter are spin-0 2048 ``map2alm`` 0.75x -> **1.01x** and spin-2 1024 ``map2alm``
    1.04x -> **1.45x** (`.qwen/tmp/score_s28.log`; cross-process repetitions of the two losses are
    588.5 -> 611.9 ms in `.qwen/tmp/score_mb_ab.log`).

    **Session 28 withheld it from spin 0 only.**  With the polarised routes wide, one build in about
    twenty-four returned an alm that was entirely NaN (9,440,250 of 9,440,256 entries) and took the
    coupled cell with it, while ~60 builds at 64 never did and 24 builds with `jax_debug_nans`
    enabled never fired.  It was always the first build of the process and only the first, it happened
    with the analysis wide and the synthesis narrow as well as the reverse, the two widths' results
    agreed to 4.3e-19 when both completed, and no read of the drivers found a slot the kernel leaves
    unwritten -- the analysis masks its `ell < max(m, spin)` wedge and both synthesis kernels store
    every ``(row, tile, lane)`` once.  That combination is uninitialized device memory, not the
    arithmetic, so the width was withheld from spin 2 until it was root-caught; the spin-0 folded
    routes, where no such event has been seen, kept the win.  Reproducers:
    `.qwen/tmp/nan_where.py`, `.qwen/tmp/nan_dir.py`, `.qwen/tmp/nan_debug.py`; details in HANDOFF
    session 28.

    **The wedge is root-caught and spin 2 takes the width.**  Session 28's pre-registered experiment
    was "make the kernel write the wedge and see whether the poisoned wide build stops NaNing"; it did.
    The analysis slab is now the window's own range (`L - m0`), each row stores at `ell - m0`, and the
    program zeroes its own `<= mb - 1` lane head instead of leaving allocator bytes there, so every
    lane the reduce reads is written.  With that, `GMASTER_M_BLOCK=128` gave `max|dCl| = 4.17e-12`,
    `rel = 3.69e-06` -- digit-for-digit the narrow value -- in every pipeline process tried, against
    2/2 `nan` at the same cell before the fix (`.qwen/tmp/pipe_trim_wide.log`, HANDOFF session 29).
    """
    from gmaster import _spin_slice as ss

    mb = ss._MARCH_M_BLOCK
    if mb * ntile > _MARCH_GRID_CAP:
        mb = ss._M_BLOCK
    return ss._windows(L, mb)


def _synth_windows(L, ntile, spin):
    """The m-windows of one synthesis launch: as wide as the program ceiling still allows.

    The law measured for `_march_windows` is about programs per launch, not about the window itself
    (every win was at <= 2048 programs, every loss at 4096), and this route is the one marched route
    with room: `_ST0` leaves it 4 theta tiles at nside 2048 and 8 at 4096, where the analysis has 16
    and 32.  So the window is grown to fill the ceiling (`cap // ntile`, rounded down to a power of
    two, capped at `_MARCH_M_SYNTH0_MAX`) instead of stopping at 128.  Measured against ducc0, spin-0
    `alm2map`, fp32, 5 reps, one geometry per process (`.qwen/tmp/swinf_s29.log`, against the
    pre-change `BOARD` stage of `.qwen/tmp/s29z.log`): **2048 215.6 ms / 1.40x against 246.3 ms /
    1.24x** and **4096 1649.2 ms / 1.19x against 1740.0 ms / 1.14x**, with `rel alm` digit-for-digit
    unchanged (6.9e-05 / 2.1e-04) because the window only groups independent m-lanes into one launch.
    Reproduced in the width sweep of `.qwen/tmp/swin_s29.log` (2048: 217.2 ms with the ceiling at 512,
    249.6 ms with it at 256; 4096: 1670.6 ms) and at 4096 again in `.qwen/tmp/s29y.log` (1650.3 ms).
    Nside 1024 does not move (30.2 ms either way, 24 launches of 128 orders against 6 of 512), and the
    four-tile gate exists because Nside 512 -- two theta tiles, so the rule would pick 256 or 512 there
    -- measured 4.9 and 5.0 ms against the shipped 4.4 ms in that same pair of arms.  The analysis
    route keeps `_march_windows`: growing its window past the ceiling does not widen anything, it
    trips the fallback to `_M_BLOCK` and costs 1.37x at 2048.
    """
    from gmaster import _spin_slice as ss

    if int(spin) != 0 or int(ntile) < 4:
        return _march_windows(L, ntile, spin)
    mb = min(ss._MARCH_M_SYNTH0_MAX, max(ss._MARCH_M_BLOCK, _MARCH_GRID_CAP // max(int(ntile), 1)))
    mb = 1 << (int(mb).bit_length() - 1)          # the Triton lowering wants powers of two
    if mb * ntile > _MARCH_GRID_CAP:
        mb = ss._M_BLOCK
    return ss._windows(L, mb)


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
    out = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
    for (m0, m1, lo) in _march_windows(L, ntile, SPIN):
        mb = m1 - m0
        g = _window_geometry(m0, mb, x, sh, ch, L, npad)
        direct = ftm[:, L + m0:L + m1]
        mirror = rev[:, L - m1 + 1:L - m0 + 1][:, ::-1]
        chan = jnp.stack([direct.real.T, direct.imag.T, mirror.real.T, mirror.imag.T], axis=-1)
        rhs = jnp.zeros((mb, npad, NC), dtype=jnp.float32).at[:, :ntheta].set(
            lax.convert_element_type(chan, jnp.float32))
        parts = _call(L, ntheta, ntile, mb, Lm=L - m0)(*g, rhs, jnp.asarray([m0], jnp.int32)).sum(1)
        parts = parts.transpose(1, 0, 2)             # (m, ell, channel) -> (ell, m, channel)
        # Slab index `i` is ell = m0 + i, and row `i_m` emits from max(m0 + i_m, spin); the kernel
        # zeroed the head, so this mask only pins the contract down outside the kernel.
        acc = jnp.where(
            (jnp.arange(L - m0)[:, None]
             >= jnp.maximum(jnp.arange(mb)[None, :], SPIN - m0))[..., None], parts, 0.0)
        # `acc` is already (ell, m, channel); the shipped kernel returns (m, ell, channel) and
        # transposes here, which is the only difference in this assembly.
        out = out.at[lo:, off + m0:off + m1].set(acc[:, :, 0] + 1j * acc[:, :, 1])
        mir = sign[lo:, None] * (acc[:, :, 2] + 1j * acc[:, :, 3])
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


# ------------------------------------------------------- spin 0: the antipodal fold halves the grid
def fold_requested(nside, L) -> bool:
    """True when the folded spin-0 march should serve a latitudinal call.

    Left unset the march takes the call only where the Legendre band is refused for *size*, because
    a resident table beats it: Nside 1024 reads its 36.75 GiB band in 26.83 ms at 1471 GB/s, which
    is the card's streaming rate and about half what a march over the same folded lane count costs
    (`.qwen/tmp/spin0_stage_split.log`).  Where the band cannot exist the alternative is the fp64
    on-the-fly scalar kernel, and that is the worst cell in the repo: Nside 2048 `map2alm` 949.2 ms
    (0.33x) and Nside 4096 7516.9 ms (0.27x) against ducc0's 315.6 and 2014.4
    (`.qwen/tmp/score_n2048_spin0_s26.log`, `.qwen/tmp/score_n4096_spin0.log`).  The fold halves the
    lane count relative to the spin-2 march, and theta tiles are the kernel's second program axis,
    so the program count halves with it rather than just the work per program.

    `GMASTER_SPIN0_MARCH=1` serves every scalar call, including the geometries where a band exists,
    which is how the two routes are compared against each other; `=0` restores the fp64 kernel
    everywhere.  The accuracy cost is the march's own -- float32 lanes and a 2^+-24 renormalisation
    band, measured against ducc0 in the scoring runs below, against 3.4e-07 for the kernel it
    replaces.
    """
    from gmaster import utils

    on_gpu = _on_nvidia()
    flag = os.environ.get("GMASTER_SPIN0_MARCH")
    if flag is not None:
        return flag == "1" and on_gpu
    return on_gpu and not utils._prefer_theta_band(nside, L, 0)


def fold_synth_requested(nside, L) -> bool:
    """True when the folded spin-0 march should serve a synthesis latitudinal call.

    Mirrors `fold_requested` for the inverse direction -- the synthesis band is refused by the same
    size test plus its own 16 GiB reserve, so Nside 2048/4096 fall all the way to the fp64 on-the-fly
    kernel, which is the worst cell on the board (`2048 0 alm2map 305.8 1267.1 0.24x`,
    `4096 0 alm2map 2022.3 10050.4 0.20x`).  Its own flag so the two directions can be A/B'd apart;
    unset falls back to `GMASTER_SPIN0_MARCH` so one switch still flips the whole spin-0 route.
    """
    from gmaster import utils

    on_gpu = _on_nvidia()
    flag = os.environ.get("GMASTER_SPIN0_MARCH_SYNTH")
    if flag is None:
        flag = os.environ.get("GMASTER_SPIN0_MARCH")
    if flag is not None:
        return flag == "1" and on_gpu
    return on_gpu and not utils._prefer_theta_band(nside, L, 0)


def _fold_analyze(positives, weights, phase, *, L, nside):
    """One folded march contracting any number of scalar right-hand sides.

    The expensive part of a marched row is the recurrence, which is a function of the `(m, theta)`
    triple alone; the right-hand side enters only in the emit, where `NC` channels already fold
    through a single Triton reduction tree.  Extra maps therefore join that block as channels
    `4k:4k+4` and add a contraction to a row that is being computed anyway, instead of a second
    march over the same lane count.
    """
    from gmaster import utils
    from gmaster import _spin_slice as ss

    positives = [jnp.asarray(p) for p in positives]
    weights = jnp.asarray(weights)
    phase = jnp.asarray(phase)
    theta = jnp.asarray(utils._stable_thetas(L, nside), dtype=jnp.float64)
    ntheta = theta.shape[0]
    north = (ntheta + 1) // 2
    ntile = -(-north // _TILE)
    npad = ntile * _TILE
    tn = theta[:north]
    x = jnp.cos(tn)
    sh, ch = jnp.sin(tn / 2.0), jnp.cos(tn / 2.0)
    # Ring `ntheta - 1 - i` sits at `pi - theta_i`, shares its ring weight and its phi with ring i,
    # and `d^l_{m,0}(pi - theta) = (-1)**(l+m) d^l_{m,0}(theta)`.  So the latitudinal sum over the
    # whole sphere is, per northern lane, one row contracted against two right-hand sides,
    #     A_lm = sum_north d [G_i + (-1)**(l+m) G_i'],
    # and the parity factor -- which is not constant along a marched row, the way it is in the
    # parity-split band -- is applied outside the kernel to the two fp64 partials rather than
    # selected inside it per degree.  This is the same algebra `_theta_matrix._transform` performs;
    # there the row split makes `(-1)**(l+m)` a per-column sign, here the row is marched whole.
    m = jnp.arange(L, dtype=jnp.float64)[:, None]
    p2phi = jnp.exp(1j * (m * phase[None, :]))
    jn = jnp.arange(north)
    partner = ntheta - 1 - jn

    # The four channels of the rhs are interleaved on the last axis, so building them per window is
    # a strided store per window: measured 126.9 ms of the 442 ms Nside 2048 `map2alm`, against
    # ~2 ms for the same bytes streamed once (`.qwen/tmp/spin0_fold_rhs_cost_2048b.log` -- neither
    # bandwidth nor the `zeros().at[].set` form is the cost, a concatenate is 1.04x).  So the whole
    # (L, npad, nc) block is built once and each window slices its own rows out of it.
    chans = []
    for positive in positives:
        folded = positive.T * weights[None, :] * p2phi
        g_north = folded[:, jn]
        # The equator lane is its own partner and is counted once, matching both the band and the
        # fused kernel, which skips the south half there.
        g_part = jnp.where(partner == jn, 0.0, folded[:, partner])
        chans.append(jnp.stack([g_north.real, g_north.imag,
                                g_part.real, g_part.imag], axis=-1))
    nc = NC * len(positives)
    chan_all = chans[0] if len(chans) == 1 else jnp.concatenate(chans, axis=-1)
    rhs_all = lax.convert_element_type(chan_all, jnp.float32)
    if npad != north:
        rhs_all = jnp.concatenate(
            [rhs_all, jnp.zeros((L, npad - north, nc), dtype=jnp.float32)], axis=1)

    cols = [[] for _ in positives]
    norm = jnp.sqrt((2.0 * jnp.arange(L, dtype=jnp.float64) + 1.0) / (4.0 * jnp.pi))
    for (m0, m1, lo) in _march_windows(L, ntile, 0):
        mb = m1 - m0
        g = _window_geometry(m0, mb, x, sh, ch, L, npad, spin=0)
        rhs = rhs_all[m0:m1]
        parts = _call(L, north, ntile, mb, spin=0, Lm=L - m0, nc=nc)(
            *g, rhs, jnp.asarray([m0], jnp.int32)).sum(1)      # (mb, L - m0, nc)
        ms = m0 + jnp.arange(mb)
        el = m0 + jnp.arange(L - m0)
        # Rows emit from `ell = m` and never below it; the kernel zeroed that head and the mask keeps
        # the assembly independent of it.
        keep = (el[None, :] >= ms[:, None])[..., None]
        sgn = 1.0 - 2.0 * ((ms[:, None] + el[None, :]) % 2)
        for k in range(len(positives)):
            c = 4 * k
            p = jnp.where(keep, parts[..., c:c + NC], 0.0)
            # The scalar route's rows are `sqrt((2l+1)/4pi) * d^l_{m0}`, not the bare Wigner row:
            # the band's `_diagonal_normalization` seed carries the degree factor (measured as a
            # constant ratio of exactly that over every row and lane, 5.2e-08 at Nside 32), and so
            # does the fused kernel that shares the seed.  The march closes the form on the bare
            # row, so the degree factor is applied here rather than inside the kernel, where it
            # would also land on the spin-2 route -- whose contract deliberately leaves it to
            # `_finish_forward_s2fft`.
            block = ((p[..., 0] + 1j * p[..., 1])
                     + sgn * (p[..., 2] + 1j * p[..., 3])) * norm[m0:][None, :]
            cols[k].append(jnp.zeros((L, mb), dtype=block.dtype).at[m0:, :].set(block.T))
    # Window column ranges are disjoint and tile the positive-m block, so one concatenation assembles
    # it; `out.at[...].set` per window would copy the whole buffer once per window (see
    # `_inverse_impl`).
    return [jnp.concatenate(c, axis=1) for c in cols]


@partial(jax.jit, static_argnames=("L", "nside"))
def _forward_fold_impl(positive, weights, phase, *, L, nside):
    return _fold_analyze([positive], weights, phase, L=L, nside=nside)[0]


def forward_latitudinal_positive(positive, weights, phase, *, L, nside):
    """Folded table-free spin-0 analysis latitudinal step.

    Same contract as :func:`gmaster._theta_matrix.positive_latitudinal`: the ``(4*nside-1, L)``
    positive-m block of the ring-FFT map in, the ``(L, L)`` complex ``(ell, m)`` block out, with the
    quadrature weight and ring phase inside the contraction and the ``sqrt((2l+1)/4pi)`` factor left
    to the caller.  Negative orders never appear here: for spin 0 they follow from the real-map
    symmetry in ``_finish_forward_s2fft``, which is why this route needs no mirror channel.
    """
    return _forward_fold_impl(positive, weights, phase, L=L, nside=nside)


@partial(jax.jit, static_argnames=("L", "nside"))
def _forward_fold_pair_impl(positive_a, positive_b, weights, phase, *, L, nside):
    return _fold_analyze([positive_a, positive_b], weights, phase, L=L, nside=nside)


def forward_latitudinal_positive_pair(positive_a, positive_b, weights, phase, *, L, nside):
    """Two folded scalar analyses, one march.

    Same contract as `forward_latitudinal_positive` applied to each map; the two rows are the same
    rows, so the second map costs its emit contraction and its fp64 partials and not its recurrence.
    """
    return _forward_fold_pair_impl(positive_a, positive_b, weights, phase, L=L, nside=nside)


# ------------------------------------------------------------------- synthesis: the same row, summed
_ST = int(os.environ.get("GMASTER_SPIN2_MARCH_SYNTH_TILE", "512"))
# Spin 0's synthesis march wants a wider theta tile than spin 2's, and the fold is why: it hands the
# kernel `north = (ntheta+1)//2` rows instead of `ntheta`, so at the spin-2 width a 2048-grid launch
# is 8 tiles of 512 northern rows and each block re-does its per-window prologue 8 times over half
# the rows.  Measured against ducc0 in one geometry per process, `GM_PREC=fp32`, 5 reps
# (`.qwen/tmp/st2048_s29.log`, `.qwen/tmp/st_spin0_s29.log`, `.qwen/tmp/default_s29.log`), spin-0
# `alm2map`:
#   Nside 2048  512 (shipped) 287.4 ms 0.99x | 256 270.0 ms 1.04x | 1024 243.7 ms 1.17x | 2048 401.2 0.71x
#   Nside 4096  512          1885.1 ms 1.02x | 1024 1745.8 ms 1.10x | 2048 3186.5 ms 0.61x
#   Nside 1024  512            31.0 ms 1.73x | 1024  30.7 ms 1.70x | 2048  30.7 ms 1.59x  (flat)
# Spin 2 at the same geometry goes the other way -- 453.7 ms 1.25x at 512, 508.1 at 256, 524.6 at
# 1024 -- so the width is per-spin, not global.  Applied only where `north` can still fill four
# tiles at the wide width, which is exactly the regime measured above; below it (Nside <= 1024,
# where the widths are indistinguishable) the shipped geometry stands.
_ST0 = int(os.environ.get("GMASTER_SPIN0_SYNTH_TILE", "1024"))
# Warps per synthesis block.  Re-measured at the shipped tile widths, Nside 2048 `alm2map`
# (`.qwen/tmp/s29z.log` stage `SW`; the 4-warp row is the default arm of stage `BOARD`): spin 0 gives
# 1522.1 ms (1 warp), 383.0 (2), **246.3 (4)**, 254.2 (8); spin 2 gives 526.2 (2), **457.7 (4)**.
# The one-warp arm is 6x the four-warp one -- a 1024-lane tile over 32 lanes is an occupancy cliff,
# not a slope -- so this knob and `_ST`/`_ST0` are one decision and must be swept together.
_SW = int(os.environ.get("GMASTER_SPIN2_MARCH_SYNTH_WARPS", "4"))
# `fast` accumulates the contraction in plain float32, `comp` in (value, limb) float32 pairs; see
# `_accumulate_fast` for the measured cost and error of the difference.
_ACC = os.environ.get("GMASTER_SPIN2_MARCH_SYNTH_ACC", "fast")
_ACC_FAST = _ACC != "comp"


def _synth_tile(spin, ntheta) -> int:
    """Theta tile for one synthesis launch; see the `_ST0` note above for why it is per-spin."""
    if int(spin) == 0 and int(ntheta) >= 4 * _ST0:
        return _ST0
    return _ST


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
                m0_ref, out_ref, *, L, ntheta, chunk, spin=SPIN):
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
    nstart = jnp.maximum(m, spin)
    t = tile * chunk + jnp.arange(chunk)
    valid = t < ntheta

    man = plt.load(manr.at[row, t], mask=valid, other=0.0)
    ex0 = plt.load(ex0r.at[row, t])
    ex = ex0
    xv = plt.load(xr.at[t], mask=valid, other=0.0)
    xl = plt.load(xlr.at[t], mask=valid, other=0.0)
    mf = m.astype(jnp.float32)
    alpha = mf + jnp.float32(spin)
    beta = jnp.abs(mf - jnp.float32(spin))
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


def _call_synth(L, ntheta, ntile, mb, spin=SPIN):
    st = _synth_tile(spin, ntheta)
    key = ("synth", int(L), int(ntheta), int(ntile), int(mb), st, _SW, int(spin))
    call = _CALLS.get(key)
    if call is None:
        call = jax.jit(pl.pallas_call(
            partial(_kern_synth, L=L, ntheta=ntheta, chunk=st, spin=spin),
            out_shape=jax.ShapeDtypeStruct((mb, ntile, st, 4), jnp.float64),
            grid=(mb, ntile),
            compiler_params=plt.CompilerParams(num_warps=_SW),
            name=f"gmaster_spin{int(spin)}_synth"))
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


def _synth_coeff0(alm, m0, mb, L, *, mirror):
    """Spin-0 synthesis coefficients for one m-window, read from the positive-m block.

    Same split as `_synth_coeff`, with two differences forced by the fold: the alm input is the
    `(ell, m)` positive block (so a window reads columns `m0:m0+mb` directly, not the centred
    `L-1+m` layout), and the "mirror" set is not a negative order at all -- it is the *southern*
    half of the same order, `(-1)^(ell+m)` applied so the northern marched row can serve the
    partner ring, exactly like `_theta_matrix._inverse`'s `acc_e - acc_o` block sign.  Because the
    parity is `+-1` it leaves `lex` untouched, so both sets share one exponent array.

    `alm` arrives already carrying `sqrt((2l+1)/4pi)`; the marched row is the bare `d^l_(m,0)`, so
    the degree factor has to ride in the coefficient here (the analysis fold applies it outside the
    kernel instead, where the row is what gets scaled).
    """
    ms = m0 + jnp.arange(mb)
    c = alm[:, m0:m0 + mb]                             # (L, mb) complex
    lgs = _log2_norm(m0, mb, L, 0)                     # (mb, L) float64
    lex = jnp.floor(lgs)
    sc = jnp.exp2(lgs - lex) * ((-1.0) ** ms)[:, None]
    if mirror:
        parity = 1.0 - 2.0 * ((ms[:, None] + jnp.arange(L)[None, :]) % 2)
    else:
        parity = 1.0
    out = []
    for part in (c.real.T * sc * parity, c.imag.T * sc * parity):
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
    # times at 2048 -- 1.50 GiB per copy at the latter.  Re-measured end to end in one process with
    # both arms on identical inputs: 114.36 -> 64.35 ms (**1.78x**) at 1024 and 1611.28 -> 482.72 ms
    # (**3.34x**) at 2048, `max|diff| 0.000e+00` (bit-identical), `.qwen/tmp/synth_assembly_ab2.log`.
    # The numbers this comment used to quote (2.93x / 5.20x) came from the same A/B under different
    # conditions and the log it cited had since been overwritten by an unrelated probe; the direction
    # was right, the magnitudes were not.  Analysis assembly does *not* benefit (0.98-0.99x,
    # `.qwen/tmp/analysis_assembly_ab.log`) -- do not port this back and forth.
    dirs, mirs = [], []
    for (m0, m1, lo) in _march_windows(L, ntile, SPIN):
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



@partial(jax.jit, static_argnames=("L", "nside"))
def _inverse_fold_impl(positive, phase, *, L, nside):
    """Spin-0 synthesis latitudinal step on the northern grid, table-free.

    The transpose of `_forward_fold_impl`: one march over `theta[:north]` feeds two accumulators
    that differ only in the coefficient set, the direct one for the northern rings and a
    `(-1)^(ell+m)`-signed one for the southern partners of those same lanes.  The partner's own
    ring phase is applied outside the kernel (synthesis carries no quadrature weight, so `weights`
    is 1 by contract), and the south half is stored at the reversed theta axis -- the same assembly
    `_theta_matrix._inverse` uses, including dropping the equator lane, which is its own partner and
    contributes only through the direct accumulator.
    """
    from gmaster import _spin_slice as ss
    from gmaster import utils

    positive = jnp.asarray(positive)
    theta = jnp.asarray(utils._stable_thetas(L, nside), dtype=jnp.float64)
    ntheta = theta.shape[0]
    north = (ntheta + 1) // 2
    st0 = _synth_tile(0, north)
    ntile = -(-north // st0)
    npad = ntile * st0
    tn = theta[:north]
    x = jnp.cos(tn)
    sh, ch = jnp.sin(tn / 2.0), jnp.cos(tn / 2.0)
    norm = jnp.sqrt((2.0 * jnp.arange(L, dtype=jnp.float64) + 1.0) / (4.0 * jnp.pi))
    alm = positive * norm[:, None]
    dirs, mirs = [], []
    for (m0, m1, lo) in _synth_windows(L, ntile, 0):
        mb = m1 - m0
        ms = m0 + jnp.arange(mb)
        g = _window_geometry(m0, mb, x, sh, ch, L, npad, spin=0)
        dc = _synth_coeff0(alm, m0, mb, L, mirror=False)
        mc = _synth_coeff0(alm, m0, mb, L, mirror=True)
        v = _call_synth(L, north, ntile, mb, spin=0)(*g[:10], *dc, *mc,
                                                     jnp.asarray([m0], jnp.int32))
        v = v.reshape(mb, npad, 4)[:, :north]
        nf = jnp.exp(1j * (ms[:, None] * phase[:north][None, :]))
        sf = jnp.exp(1j * (ms[:, None] * jnp.flip(phase)[: north - 1][None, :]))
        dirs.append((v[:, :, 0] + 1j * v[:, :, 1]) * nf)              # (mb, north)
        mirs.append((v[:, :, 2] + 1j * v[:, :, 3])[:, :north - 1] * sf)
    north_vals = jnp.concatenate(dirs, axis=0)                        # (L, north)
    south_vals = jnp.concatenate(mirs, axis=0)                        # (L, north-1)
    return jnp.transpose(jnp.concatenate(
        [north_vals, jnp.flip(south_vals, axis=1)], axis=1))          # (ntheta, L)


def inverse_latitudinal_positive(positive, phase, *, L, nside):
    """Folded spin-0 synthesis: `(L, L)` positive-m block in, `(4*nside-1, L)` complex out.

    Same contract as `_theta_matrix.inverse_latitudinal` / `utils.scalar_inverse_latitudinal`
    (ring phi phase applied, `sqrt((2l+1)/4pi)` baked in), without the Legendre band.
    """
    return _inverse_fold_impl(positive, phase, L=L, nside=nside)


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
