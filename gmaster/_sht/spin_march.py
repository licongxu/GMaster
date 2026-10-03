"""Table-free latitudinal step: the Wigner-d row is marched in registers, not stored.

A Pallas/Triton difference-form march for the latitudinal (theta) step of the spin-2 and folded
spin-0 transforms.  One program owns one (m-window row, theta tile) pair, marches `ell` in
registers and contracts each degree into the real channels of the right-hand side inside the same
loop, so no Wigner-d table is materialised on the host or device.  This is the route for
geometries whose Wigner-d slice exceeds the memory budget (e.g. 73 GiB for spin 2 at Nside 1024).
When the CUDA v2 march (:mod:`gmaster._sht.march_v2`) is enabled for `L`, every public entry
point delegates to it instead.

Closed form marched (validated against `_spin_slice._build`, itself checked against pymaster):

    d^l_{m,-s}(theta) = (-1)^m 2^(eps_m LG2N) (sin t/2)^alpha (cos t/2)^beta P^{(a,b)}_n(cos t)
    alpha = m + s,  beta = |m - s|,  n = l - max(m, s),
    LG2N = log2 sqrt((l+m)!(l-m)! / ((l-s)!(l+s)!)),  eps_m = -1 for m < s else +1.

`n` counts from `max(m, s)`, not `m`: a row below the spin starts its polynomial at ell = s.
DLMF 14.9.16's `(cos t/2)^(m1+m2)` admits the (m, -s) substitution only for m >= s; below the spin
the pair-swapped branch inverts the factorial ratio, which is the `eps_m` flip.  Omitting either
leaves rows m >= s correct but makes rows m = 0 and m = 1 wrong (O(1) and O(1e-2) relative).

Numerics.  The march carries a float32 (value, limb) pair per theta lane and renormalises each
lane into a 2^+-24 band every degree.  Products go through inline PTX FMA because XLA offers no
float32 FMA here and `add(neg(mul(a,b)), mul(a,b))` folds to an exactly zero residual inside
Pallas.  Both compensations (coefficient limbs and state limbs) are needed: either alone gives
~1e-5, together ~1e-7 against a float64 march of the same recurrence up to Nside 1024.  Tile
partials are summed across tiles in float64, as in `_spin_slice._reduce_channels`.

Every kernel input is computed inside the trace from `cos(theta)` and static (L, nside).  Route
selection is by :func:`march_requested`, :func:`synth_requested`, :func:`fold_requested` and
:func:`fold_synth_requested` (see their docstrings for the `GMASTER_*` flags).
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

from gmaster._sht import march_v2 as _march_v2
from gmaster._sht.cuda_gpu import on_cuda_gpu

SPIN = 2
NC = 4                       # direct re/im, mirror re/im -- the same split as `_rhs_forward`
BLO, BHI, JUMP = 2.0 ** -24, 2.0 ** 24, 24
EX_PAD = -(10 ** 7)          # pad theta lanes: never win `emax`, never flush the real lanes
_TILE = int(os.environ.get("GMASTER_SPIN2_MARCH_TILE", "256"))
_WARPS = int(os.environ.get("GMASTER_SPIN2_MARCH_WARPS", "1"))
# Ceiling on programs per march launch for the wide m-window (see :func:`_march_windows`).
_MARCH_GRID_CAP = int(os.environ.get("GMASTER_MARCH_GRID_CAP", "2048"))
# Degrees per loop iteration of the analysis march (see `_kern`).  8 is fastest at Nside 1024;
# 16 spills registers and is slower than 4.
_UNROLL = int(os.environ.get("GMASTER_MARCH_UNROLL", "8"))
# `GMASTER_MARCH_POLAR_SKIP=0` marches every (m, theta tile) program, including the ones above the
# ring's `mlim` (see `_polar_skip`); on by default.
_POLAR_SKIP = os.environ.get("GMASTER_MARCH_POLAR_SKIP", "1") == "1"

_CALLS: dict = {}


def _on_nvidia() -> bool:
    """True on a CUDA GPU. Colab reports ``Tesla T4`` without ``NVIDIA``."""
    return on_cuda_gpu()


def slice_declined(L, nside) -> bool:
    """True when no Wigner-d slice can be resident for this geometry, at any storage precision.

    Mirrors `slabs_for`, which returns `(None, None)` above `_SLAB_BUDGET`; the only alternative
    route is then the generic scatter loop (~13 s for spin 2 at Nside 1024, ~100x slower than
    the march).  The size is tested at the configured table dtype, the smallest layout available.
    """
    from gmaster._config import table_dtype
    from gmaster._sht import spin_slice as ss

    return ss.triangle_bytes(nside, L, table_dtype()) > ss._SLAB_BUDGET


def march_requested(spin, *, L=None, nside=None) -> bool:
    """True when the analysis march should serve this call.

    `GMASTER_SPIN2_MARCH=1` forces the march (on NVIDIA, or wherever the v2 march is enabled);
    `=0` keeps the table route at any size.  Unset, the march serves the call when the v2 march is
    enabled for `L`, or on NVIDIA when no Wigner-d slice fits (`slice_declined`).  With a slice
    resident the table route is faster (e.g. Nside 512: ~9 ms against ~24 ms) and more accurate;
    without one the march is ~100x faster than the scatter loop.  Accuracy cost of the Pallas
    march: ~4e-05 relative against ducc0, versus ~4e-07 for the scatter loop.

    The caller only reaches this seam when `_spin_slabs` returns `(None, None)`, which is the same
    test `slice_declined` makes, so where a slab exists the flag has no effect.
    """
    if int(spin) != SPIN:
        return int(spin) in _march_v2.SPINS and _march_v2.spin_enabled(spin, L)
    flag = os.environ.get("GMASTER_SPIN2_MARCH")
    if flag is not None:
        return flag == "1" and (_on_nvidia() or _march_v2.enabled(L))
    if _march_v2.enabled(L):
        return True
    if not _on_nvidia():
        return False
    return L is not None and nside is not None and slice_declined(L, nside)


def synth_requested(spin, *, L=None, nside=None) -> bool:
    """True for the synthesis march: its own flag, otherwise the no-slice rule.

    `GMASTER_SPIN2_MARCH_SYNTH` is separate from `GMASTER_SPIN2_MARCH` because the two directions
    have different cost/accuracy tradeoffs; the default rule is the same (march only where
    `slice_declined`, or wherever the v2 march is enabled).  `=0` keeps the exact route at any
    size; `=1` forces the march.

    Accuracy at Nside 1024 against ducc0, alms from a spin-2 analysis: ~2e-04 max, ~6e-06 rms
    relative, against ~1e-09 for the scatter loop.  The max error comes from the pole-most rings,
    where the float32 lane pair delivers ~29 bits at high `ell` (e.g. 2.5e-04 at `ell = 1528`) and
    the error grows toward the pole; a float64 march of the same recurrence holds ~1e-12, so the
    loss is float32 arithmetic, not the algorithm.  Cost is ~10x below the scatter loop, still
    somewhat above ducc0.

    Convention: the recurrence starts at `ell = max(m, spin)`, so coefficients with
    `ell < |spin|` contribute nothing.  s2fft's `flm_to_ftm` and ducc0 do use such terms if given
    them.  Alms from any spin-2 analysis are (numerically) zero there, but a hand-built `alm` with
    sub-spin power will silently lose it.
    """
    if int(spin) != SPIN:
        return int(spin) in _march_v2.SPINS and _march_v2.spin_enabled(spin, L)
    flag = os.environ.get("GMASTER_SPIN2_MARCH_SYNTH")
    if flag is not None:
        return flag == "1" and (_on_nvidia() or _march_v2.enabled(L))
    if _march_v2.enabled(L):
        return True
    if not _on_nvidia():
        return False
    return L is not None and nside is not None and slice_declined(L, nside)


# --------------------------------------------------------------------------------------- kernel
def _two_prod(a, b):
    """(a*b, exact residual) via PTX `mul`/`fma`; XLA exposes no float32 FMA inside Pallas."""
    return plt.elementwise_inline_asm(
        "mul.rn.f32 $0, $2, $3; neg.f32 $1, $0; fma.rn.f32 $1, $2, $3, $1;",
        args=[a, b], constraints="=r,=r,r,r", pack=1,
        result_shape_dtypes=[jax.ShapeDtypeStruct(a.shape, jnp.float32),
                             jax.ShapeDtypeStruct(a.shape, jnp.float32)])


def _fma(a, b, c):
    # Single-rounded float32 a*b + c.  Operands $0..$3 are one output then three inputs.
    return plt.elementwise_inline_asm("fma.rn.f32 $0, $1, $2, $3;", args=[a, b, c],
                                      constraints="=r,r,r,r", pack=1,
                                      result_shape_dtypes=[
                                          jax.ShapeDtypeStruct(a.shape, jnp.float32)])[0]


def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _pow2(d):
    """Exact float32 2**d for integer `d`, assembled from the exponent field (0 below 2**-126).

    `lax.exp2` is not exact on integer exponents (~1e-7 relative) and the rescale must be."""
    biased = jnp.clip(d + 127, 1, 254)
    return jnp.where(d >= -126, jax.lax.bitcast_convert_type(biased << 23, jnp.float32),
                     jnp.float32(0.0))


def _pow2_f64(d):
    """The same exact power of two in float64, for rescaling an accumulator block."""
    biased = jnp.clip(d + 1023, 1, 2046).astype(jnp.int64)
    return jax.lax.bitcast_convert_type(biased << 52, jnp.float64)


def _polar_skip(m, xv, valid, *, L, spin):
    """True when no harmonic of order `m` is non-negligible on any ring of this tile.

    ducc0's `sharp_get_mlim` (libsharp2 `sharp_ylmgen_c`): a ring at colatitude theta carries no
    order above `mlim(theta) = s|cos theta| + sqrt((lmax sin theta + ofs)^2 - s^2 sin^2 theta)`,
    `ofs = max(100, 0.01 lmax)`, and the reference skips the ring for every higher order.  Beyond
    that order every `d^ell_{m,-s}(theta)` with `ell <= lmax` sits in its forbidden region by at
    least `ofs` in `m`, where the WKB action `(2 sqrt 2 / 3) ofs^1.5 / sqrt(lmax sin theta) / cos theta`
    puts it below ~1e-8 (docs/latitudinal_march_maths.md §8).  The tile is skipped when its
    largest `mlim` is below `m`, which removes 14-19 % of the analysis march at Nside 1024-4096.
    `xv` is the float32 cosine; its rounding in `1 - x^2` at the pole moves `mlim` by ~1 order,
    well inside the margin of 100.  Disabled by `GMASTER_MARCH_POLAR_SKIP=0`.
    """
    if not _POLAR_SKIP:
        return jnp.zeros((), jnp.bool_)
    sth = jnp.sqrt(jnp.maximum(1.0 - xv * xv, 0.0))
    lmax = float(L - 1)
    t1 = lmax * sth + max(100.0, 0.01 * lmax)
    ml = spin * jnp.abs(xv) + jnp.sqrt(jnp.maximum(t1 * t1 - (spin * sth) ** 2, 0.0))
    return m.astype(jnp.float32) > jnp.max(jnp.where(valid, ml, 0.0))


def _kern(manr, ex0r, xr, xlr, c1r, c0r, cbr, c1lr, c0lr, cblr, mantr, lexr, sgnr, rr, m0_ref,
          out_ref, oex_ref, *, L, ntheta, chunk, unroll, spin=SPIN, nc=NC):
    """One (m-window row, theta tile) of the analysis march; see the module docstring.

    The emit is float32 end to end; float64 is applied only outside the kernel, because the GPU
    issues float64 at a small fraction of the float32 rate.  Each degree stores its four channel
    partials as float32 scaled by the normalisation's *fractional* binade (`mantr`, in [1, 2)),
    and in `oex_ref` the integer binade the partial is missing: the tile exponent `emax` plus
    `lexr = floor(log2 N(ell, m))`.  The driver applies `2**oex` in float64 and sums the tiles
    (`_sum_tiles`).  This matches a float64 emit to ~1e-7 relative.

    `unroll` degrees are marched per loop iteration so that the independent per-degree reduction
    trees overlap the following degrees' recurrence.

    Output layout: `out_ref` is `(mb, ntile, Lm, nc)` float32 and `oex_ref` `(mb, ntile, Lm)`
    int32, indexed by `ell - m0`.
    """
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
    xv = plt.load(xr.at[t], mask=valid, other=0.0)
    skip = _polar_skip(m, xv, valid, L=L, spin=spin)

    def zero_head(i, carry):
        plt.store(out_ref.at[row, tile, i, slice(0, nc)],
                  jnp.zeros((nc,), dtype=jnp.float32))
        plt.store(oex_ref.at[row, tile, i], jnp.int32(0))
        return carry

    # A skipped tile zeroes its whole row (`out_ref.shape[2]` lanes) instead of marching it.
    lax.fori_loop(0, jnp.where(skip, out_ref.shape[2], jnp.maximum(nstart - m0, 0)),
                  zero_head, ())

    man = plt.load(manr.at[row, t], mask=valid, other=0.0)
    ex0 = plt.load(ex0r.at[row, t])
    ex = ex0
    xl = plt.load(xlr.at[t], mask=valid, other=0.0)
    r = plt.load(rr.at[row, t, slice(0, nc)])
    mf = m.astype(jnp.float32)
    # P_1^(alpha,beta) = ((alpha+beta+2)/2) cos t + (alpha-beta)/2.  Not `m*cos t + 2`: that seed
    # leaves every ell >= m+1 a few percent off, growing with ell.
    alpha = mf + jnp.float32(spin)
    beta = jnp.abs(mf - jnp.float32(spin))
    p1f = ((alpha + beta + 2.0) / 2.0) * xv + ((alpha - beta) / 2.0)
    sgn = plt.load(sgnr.at[row]).astype(jnp.float32)

    def emit(cur, ex, ell, ok):
        # One reduction for all four channels: `cur[:, None] * r` is a (chunk, NC) block, so theta
        # folds through a single Triton tree and the channels share its latency instead of running
        # four dependent ones; the partial leaves as one (NC,) float32 store plus its binade.
        emax = jnp.max(ex)
        val = cur * _pow2((ex - emax).astype(jnp.int32))
        parts = jnp.sum(val[:, None] * r, axis=0) * (plt.load(mantr.at[row, ell]) * sgn)
        plt.store(out_ref.at[row, tile, ell - m0, slice(0, nc)], parts,
                  mask=jnp.broadcast_to(ok, (nc,)))
        plt.store(oex_ref.at[row, tile, ell - m0], emax + plt.load(lexr.at[row, ell]), mask=ok)

    def degree(ell, st):
        ph, pl_, ch, cl, ex = st
        # Degrees past `L` only occur in the tail of an unrolled block; they are marched on the
        # last coefficient and their emit is masked off.
        ell = jnp.minimum(ell, L - 1)
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
        # The residual of `th - uh` uses the branch-free 2Sum, not the fast form
        # `(th - nxt) - uh`, which is off by ~1 ulp on this stack in either magnitude ordering.
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
        return (jnp.where(valid, ch * mult, jnp.float32(0.0)),
                jnp.where(valid, cl * mult, jnp.float32(0.0)), nxt, nxtl, ex)

    def block(it, st):
        # `unroll` degrees per iteration: the recurrence is one serial chain, but the emits of
        # consecutive degrees are independent of it and of each other, so the scheduler can overlap
        # their reduction trees with the following degrees' arithmetic.
        base = nstart + it * unroll
        stage = []
        for k in range(unroll):
            st = degree(base + k, st)
            stage.append((st[2], st[4]))
        for k, (cur, exk) in enumerate(stage):
            emit(cur, exk, base + k, base + k < L)
        return st

    zb = jnp.zeros_like(man)
    lax.fori_loop(0, jnp.where(skip, 0, (L - nstart + unroll - 1) // unroll), block,
                  (zb, zb, man, zb, ex))


def _call(L, ntheta, ntile, mb, spin=SPIN, Lm=None, nc=NC):
    # `Lm` is the slab's ell extent: the window's own range `L - m0`, which is what the row writes
    # (`L - nstart` lanes at index `ell - m0`) plus the head it zeroes.
    Lm = L if Lm is None else Lm
    key = (int(L), int(ntheta), int(ntile), int(mb), int(Lm), _TILE, _WARPS, _UNROLL, int(spin),
           int(nc))
    call = _CALLS.get(key)
    if call is None:
        call = jax.jit(pl.pallas_call(
            partial(_kern, L=L, ntheta=ntheta, chunk=_TILE, unroll=_UNROLL, spin=spin, nc=nc),
            out_shape=(jax.ShapeDtypeStruct((mb, ntile, Lm, nc), jnp.float32),
                       jax.ShapeDtypeStruct((mb, ntile, Lm), jnp.int32)),
            grid=(mb, ntile),
            compiler_params=plt.CompilerParams(num_warps=_WARPS),
            name=f"gmaster_spin{int(spin)}_march"))
        _CALLS[key] = call
    return call


def _analysis_inputs(g):
    """Kernel operands from a `_window_geometry` tuple: the normalisation split into its
    fractional binade (float32, in [1, 2)) and its integer binade (int32)."""
    lgs = g[10]
    lex = jnp.floor(lgs)
    return (*g[:10], jnp.exp2(lgs - lex).astype(jnp.float32), lex.astype(jnp.int32), g[11])


def _pow2_f64_xla(e):
    """Exact `2.0**e` for an integer array, assembled from the exponent field; binades the
    partial cannot represent in float64 (below 2**-1022) are the contributions that vanish."""
    biased = (jnp.clip(e, -1022, 1023) + 1023).astype(jnp.int64)
    return jnp.where(e < -1022, 0.0, lax.bitcast_convert_type(biased << 52, jnp.float64))


def _sum_tiles(parts, oex):
    """`(mb, ntile, Lm, nc)` float32 partials and their binades to `(mb, Lm, nc)` float64."""
    return jnp.sum(parts.astype(jnp.float64) * _pow2_f64_xla(oex)[..., None], axis=1)


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
    work nor the total program count, only how it is split: ``L/mb`` launches of ``mb*ntile``
    programs.  Wider windows (``_MARCH_M_BLOCK``, 128) mean fewer launches and are 1.1-1.4x faster
    as long as a launch stays at or below ``_MARCH_GRID_CAP`` (2048) programs; at 4096 programs per
    launch they are ~5 % slower, so the narrower ``_M_BLOCK`` is used there.  The width only
    regroups independent m-rows, so results agree to the last bit.

    The analysis kernel writes every lane of its slab, including the ``ell < max(m, spin)`` head it
    does not march (see `_kern`).  This matters for wide windows: an unwritten head left
    uninitialised device memory in the slab, which occasionally surfaced as an all-NaN alm.
    """
    from gmaster._sht import spin_slice as ss

    mb = ss._MARCH_M_BLOCK
    if mb * ntile > _MARCH_GRID_CAP:
        mb = ss._M_BLOCK
    return ss._windows(L, mb)


def _synth_windows(L, ntile, spin):
    """The m-windows of one synthesis launch: as wide as the program ceiling still allows.

    The rule in `_march_windows` concerns programs per launch, not the window itself.  The spin-0
    synthesis route uses a wider theta tile (`_ST0`), so it has few tiles (4 at Nside 2048, 8 at
    4096) and room under ``_MARCH_GRID_CAP``: the window is grown to ``cap // ntile``, rounded down
    to a power of two and capped at ``_MARCH_M_SYNTH0_MAX``.  This is 1.05-1.15x faster at Nside
    2048-4096 with identical results.  Below four theta tiles (Nside <= 512) a wide window is
    slightly slower, so those geometries and all spin-2 calls keep `_march_windows`.  The analysis
    route cannot do the same: its window is already at the cap, and growing it would trip the
    fallback to ``_M_BLOCK``.
    """
    from gmaster._sht import spin_slice as ss

    if int(spin) != 0 or int(ntile) < 4:
        return _march_windows(L, ntile, spin)
    mb = min(ss._MARCH_M_SYNTH0_MAX, max(ss._MARCH_M_BLOCK, _MARCH_GRID_CAP // max(int(ntile), 1)))
    mb = 1 << (int(mb).bit_length() - 1)          # the Triton lowering wants powers of two
    if mb * ntile > _MARCH_GRID_CAP:
        mb = ss._M_BLOCK
    return ss._windows(L, mb)


@partial(jax.jit, static_argnames=("L", "spin", "nside"))
def _forward_impl(ftm, *, L, spin, nside):
    from gmaster._sht import spin_slice as ss

    ftm = jnp.asarray(ftm)
    theta = (jnp.asarray(s2_samples.thetas(L, "healpix", nside), dtype=jnp.float64)
             + 8 * jnp.finfo(jnp.float64).eps)      # the same grid `healpix._stable_thetas` gives
    ntheta = theta.shape[0]
    ntile = -(-ntheta // _TILE)
    npad = ntile * _TILE
    x = jnp.cos(theta)
    sh, ch = jnp.sin(theta / 2.0), jnp.cos(theta / 2.0)
    rev = ftm[::-1]                                  # ring i of rev is the ring at pi - theta_i
    sign = ss._sign(L, SPIN)
    # Assembled by one concatenation of per-window column blocks, like `_inverse_impl`.  A
    # per-window `out.at[...].set` keeps a full (L, 2L-1) complex128 buffer (4.8 GiB at Nside 4096)
    # live across the unrolled window loop, which exhausts the device pool at that size.
    dirs, mirs = [], []
    for (m0, m1, lo) in _march_windows(L, ntile, SPIN):
        mb = m1 - m0
        g = _window_geometry(m0, mb, x, sh, ch, L, npad)
        direct = ftm[:, L + m0:L + m1]
        mirror = rev[:, L - m1 + 1:L - m0 + 1][:, ::-1]
        chan = jnp.stack([direct.real.T, direct.imag.T, mirror.real.T, mirror.imag.T], axis=-1)
        rhs = jnp.zeros((mb, npad, NC), dtype=jnp.float32).at[:, :ntheta].set(
            lax.convert_element_type(chan, jnp.float32))
        parts = _sum_tiles(*_call(L, ntheta, ntile, mb, Lm=L - m0)(
            *_analysis_inputs(g), rhs, jnp.asarray([m0], jnp.int32)))
        parts = parts.transpose(1, 0, 2)             # (m, ell, channel) -> (ell, m, channel)
        # Slab index `i` is ell = m0 + i, and row `i_m` emits from max(m0 + i_m, spin); the kernel
        # zeroed the head, so this mask only pins the contract down outside the kernel.
        acc = jnp.where(
            (jnp.arange(L - m0)[:, None]
             >= jnp.maximum(jnp.arange(mb)[None, :], SPIN - m0))[..., None], parts, 0.0)
        head = jnp.zeros((lo, mb), dtype=jnp.complex128)
        dirs.append(jnp.concatenate([head, acc[:, :, 0] + 1j * acc[:, :, 1]], axis=0))
        mir = sign[lo:, None] * (acc[:, :, 2] + 1j * acc[:, :, 3])
        if m0:
            mirs.append(jnp.concatenate([head, mir[:, ::-1]], axis=0))     # columns off-m1+1 .. off-m0
        else:
            # Column off is the direct channel alone: m = 0 is its own mirror and does not
            # satisfy the theta -> pi-theta relation.
            mirs.append(jnp.concatenate([head[:, 1:], mir[:, 1:][:, ::-1]], axis=0))
    # Ascending windows cover descending mirror columns: highest window first, window 0 last.
    return jnp.concatenate(mirs[::-1] + dirs, axis=1)


def forward_latitudinal(ftm, *, L, spin, nside):
    """Analysis latitudinal step with no Wigner-d table.

    Same contract as :func:`gmaster._spin_slice.forward_latitudinal`: ``(L, 2L-1)`` complex with
    ``flm[ell, L-1+m]``, negative orders through the pi-theta symmetry, zeros below
    ``ell = max(|m|, spin)``.  The quadrature weight and ring phase shifts are already in ``ftm``
    and the ``sqrt((2l+1)/4pi)`` / ``(-1)**|spin|`` factors come later in
    ``_finish_forward_s2fft``; neither belongs here.
    """
    if int(spin) != SPIN and not _march_v2.spin_enabled(spin, L):
        raise ValueError(f"march route implements spin=+{SPIN}, got spin={spin}")
    if _march_v2.spin_enabled(spin, L):
        return _march_v2.forward_latitudinal(ftm, L=L, spin=spin, nside=nside)
    return _forward_impl(ftm, L=L, spin=spin, nside=nside)


# ------------------------------------------------------- spin 0: the antipodal fold halves the grid
def fold_requested(nside, L) -> bool:
    """True when the folded spin-0 march should serve a latitudinal call.

    Unset, the march serves the call when the v2 march is enabled for `L`, or on NVIDIA where the
    Legendre band is refused for size (`healpix._prefer_theta_band`).  A resident band is faster:
    it is read at streaming bandwidth, about twice as fast as marching it.  Without a band the
    alternative is the float64 on-the-fly scalar kernel, which is 3-4x slower than ducc0 at
    Nside 2048-4096.  Folding north/south halves the theta lanes, and hence the program count.

    `GMASTER_SPIN0_MARCH=1` serves every scalar call, including the geometries where a band exists,
    which is how the two routes are compared against each other; `=0` restores the fp64 kernel
    everywhere.  The accuracy cost is the march's own (float32 lanes, 2^+-24 renormalisation
    band); the float64 kernel it replaces is accurate to ~3e-07 against ducc0.
    """
    from gmaster._sht import healpix

    on_gpu = _on_nvidia()
    flag = os.environ.get("GMASTER_SPIN0_MARCH")
    if flag is not None:
        return flag == "1" and (on_gpu or _march_v2.enabled(L))
    if _march_v2.enabled(L):
        return True
    return on_gpu and not healpix._prefer_theta_band(nside, L, 0)


# Safety factor on `fold_pair_bytes` against the device pool (see `fold_pair_fits`).  Calibrated so
# that Nside 2048 pairs and Nside 4096 does not on a ~71 GiB pool: the paired fold runs at 2048
# but ran out of memory at 4096, where two separate calls still fit.
_PAIR_POOL_FACTOR = 4


def fold_pair_bytes(nside, L) -> int:
    """Analytic live set of one folded march over two right-hand sides, in bytes.

    Each term is a temporary `_fold_analyze` builds: the shared `p2phi` plane, the per-map
    fold (`folded`, its two gathered halves and the stacked fp32 channels), the concatenated
    rhs block, the widest kernel slab (`min(128, L)` m-rows, the first window being the long
    one), the fp64 per-window partials the assembly consumes, and the two assembled alms.
    It is a scale estimate for a gate, not a scheduler's answer -- see `fold_pair_fits`.
    The fold planes are counted at 16 bytes even under `set_ring_precision("fp32")`, because
    the gate was calibrated with the azimuthal stage in float64; with float32 rings it is
    therefore conservative.
    """
    ntheta = 4 * nside - 2
    north = (ntheta + 1) // 2
    ntile = -(-north // _TILE)
    nc = NC * 2
    phase = L * ntheta * 16
    per_map = L * ntheta * 16 + 2 * (L * north * 16) + L * north * 4 * NC
    rhs = L * (ntile * _TILE) * 4 * nc
    slab = min(128, L) * ntile * L * nc * 8
    partials = nc * 8 * L * L // 2
    alms = 2 * L * L * 16
    return phase + 2 * per_map + rhs + slab + partials + alms


def fold_pair_fits(nside, L) -> bool:
    """True when this device's pool can be asked to hold the paired fold.

    Pairing puts both maps' azimuthal stages, both rhs blocks and one unrolled window loop
    into a single XLA program, so the allocator's peak is well above the analytic live set
    (more than twice it at Nside 4096).  Requiring `_PAIR_POOL_FACTOR` times the estimate
    scales with the pool instead of hard-coding an Nside.  A device with no pool accounting
    (`memory_stats()` returns None, as on CPU) is treated as unlimited.
    """
    stats = jax.devices()[0].memory_stats()
    limit = stats.get("bytes_limit") if stats else None
    if not limit:
        return True
    if _march_v2.enabled(L):
        # The v2 pair carries no Wigner-d slab, so its own gate decides (memory *and* the band
        # limit above which the paired kernel's tile partials cost more than the march it shares).
        return _march_v2.pair_fits(L, nside)
    return fold_pair_bytes(nside, L) * _PAIR_POOL_FACTOR <= limit


def fold_synth_requested(nside, L) -> bool:
    """True when the folded spin-0 march should serve a synthesis latitudinal call.

    Mirrors `fold_requested` for the inverse direction.  Without the march, Nside 2048/4096 fall
    back to the float64 on-the-fly kernel (4-5x slower than ducc0).  `GMASTER_SPIN0_MARCH_SYNTH`
    controls this direction alone; unset, it falls back to `GMASTER_SPIN0_MARCH`, so one switch
    flips the whole spin-0 route.
    """
    from gmaster._sht import healpix

    on_gpu = _on_nvidia()
    flag = os.environ.get("GMASTER_SPIN0_MARCH_SYNTH")
    if flag is None:
        flag = os.environ.get("GMASTER_SPIN0_MARCH")
    if flag is not None:
        return flag == "1" and (on_gpu or _march_v2.enabled(L))
    if _march_v2.enabled(L):
        return True
    return on_gpu and not healpix._prefer_theta_band(nside, L, 0)


def _fold_analyze(positives, weights, phase, *, L, nside):
    """One folded march contracting any number of scalar right-hand sides.

    The expensive part of a marched row is the recurrence, which is a function of the `(m, theta)`
    triple alone; the right-hand side enters only in the emit, where `NC` channels already fold
    through a single Triton reduction tree.  Extra maps therefore join that block as channels
    `4k:4k+4` and add a contraction to a row that is being computed anyway, instead of a second
    march over the same lane count.
    """
    from gmaster._sht import healpix

    positives = [jnp.asarray(p) for p in positives]
    weights = jnp.asarray(weights)
    phase = jnp.asarray(phase)
    theta = jnp.asarray(healpix._stable_thetas(L, nside), dtype=jnp.float64)
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

    # The rhs channels are interleaved on the last axis, so building them per window would be a
    # strided store per window (a large share of the Nside 2048 runtime).  The whole
    # (L, npad, nc) block is built once and each window slices its own rows.
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
        parts = _sum_tiles(*_call(L, north, ntile, mb, spin=0, Lm=L - m0, nc=nc)(
            *_analysis_inputs(g), rhs, jnp.asarray([m0], jnp.int32)))   # (mb, L - m0, nc)
        ms = m0 + jnp.arange(mb)
        el = m0 + jnp.arange(L - m0)
        # Rows emit from `ell = m` and never below it; the kernel zeroed that head and the mask keeps
        # the assembly independent of it.
        keep = (el[None, :] >= ms[:, None])[..., None]
        sgn = 1.0 - 2.0 * ((ms[:, None] + el[None, :]) % 2)
        for k in range(len(positives)):
            c = 4 * k
            p = jnp.where(keep, parts[..., c:c + NC], 0.0)
            # The scalar route's rows are `sqrt((2l+1)/4pi) * d^l_{m0}` (the band's
            # `_diagonal_normalization` seed carries the degree factor), while the march produces
            # the bare row.  The factor is applied here, not in the shared kernel, because the
            # spin-2 contract leaves it to `_finish_forward_s2fft`.
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
    if _march_v2.enabled(L):
        return _march_v2.forward_latitudinal_positive(positive, weights, phase, L=L, nside=nside)
    return _forward_fold_impl(positive, weights, phase, L=L, nside=nside)


@partial(jax.jit, static_argnames=("L", "nside"))
def _forward_fold_pair_impl(positive_a, positive_b, weights, phase, *, L, nside):
    return _fold_analyze([positive_a, positive_b], weights, phase, L=L, nside=nside)


def forward_latitudinal_positive_pair(positive_a, positive_b, weights, phase, *, L, nside):
    """Two folded scalar analyses, one march.

    Same contract as `forward_latitudinal_positive` applied to each map; the two rows are the same
    rows, so the second map costs its emit contraction and its fp64 partials and not its recurrence.
    """
    if _march_v2.enabled(L):
        return _march_v2.forward_latitudinal_positive_pair(
            positive_a, positive_b, weights, phase, L=L, nside=nside)
    return _forward_fold_pair_impl(positive_a, positive_b, weights, phase, L=L, nside=nside)


# ------------------------------------------------------------------- synthesis: the same row, summed
_ST = int(os.environ.get("GMASTER_SPIN2_MARCH_SYNTH_TILE", "512"))
# Spin 0's synthesis march uses a wider theta tile than spin 2's.  The fold hands the kernel
# `north = (ntheta+1)//2` rows, so at the spin-2 width each program repeats its per-window prologue
# over half as many rows.  1024 is 1.07-1.17x faster than 512 at Nside 2048-4096; 2048 is much
# slower, and at Nside 1024 the widths are indistinguishable.  Spin 2 prefers 512, so the width is
# per-spin.  `_synth_tile` applies `_ST0` only where `north` fills at least four wide tiles.
_ST0 = int(os.environ.get("GMASTER_SPIN0_SYNTH_TILE", "1024"))
# Warps per synthesis block; 4 is fastest for both spins at the default tile widths.  Too few warps
# for a wide tile is an occupancy cliff (1 warp on a 1024-lane tile is ~6x slower), so this knob and
# `_ST`/`_ST0` must be tuned together.
_SW = int(os.environ.get("GMASTER_SPIN2_MARCH_SYNTH_WARPS", "4"))
# `fast` accumulates the contraction in plain float32, `comp` in (value, limb) float32 pairs; see
# `_accumulate_fast` for the cost/accuracy tradeoff.
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
    toward overflow.  Moving the block without rescaling the stored value would silently multiply
    the accumulated sum by `2**drift`; synthesis is sensitive to this because each lane is emitted
    as a pixel rather than reduced over theta.
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


def _accumulate_fixed(nd, tex, rh, ih, mrh, mih, a_r, b_r, c_r, d_r):
    """Plain-float32 accumulation of one degree's term into both order signs at a fixed binade.

    The coefficients arrive prescaled to `[1, 2)` at their maximum (`_window_prescale`) and the
    term `nd * 2**tex` is the bare `|d| / 2**frac <= 1`, so the accumulators stay below
    `2**(2 + log2 L)` and need no block exponent: the alignment is one exact power of two shared by
    the direct and mirror sums (their `lex` are the same array), then four FMAs.  Compared with
    `_accumulate_fast` this drops the per-lane range bookkeeping and is bit-identical except for
    terms below `2**-126` of the window's scale, which are flushed.
    """
    nd = nd * _pow2(tex)
    return a_r + nd * rh, b_r + nd * ih, c_r + nd * mrh, d_r + nd * mih


def _accumulate_fast(nd, tex, aex, at0, rh, ih, a_r, b_r):
    """Plain-float32 accumulation of one degree's term: same range guards, no limbs.

    The range half of `_accumulate` (the `lift`/`ds` alignment and the 2^+-24 accumulator guard) is
    what keeps a 6000-term sum inside float32's exponent range and is kept verbatim.  What is
    dropped is the *precision* half: the limb operands (neither the marched limb nor the
    coefficient limbs are read at all, so the kernel does not even load them) and the
    accumulator limb.  The step is instruction-issue bound, so removing ~40 % of its vector
    operations makes it ~1.6x faster, at ~2e-06 error relative to the window's maximum against the
    compensated form (Nside 1024).
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
    differs.

    The accumulation is plain float32 by default; `GMASTER_SPIN2_MARCH_SYNTH_ACC=comp` selects
    (value, limb) pairs (see `_accumulate_fast`).  Float64 accumulators are avoided because
    many GPUs issue float64 at a small fraction (e.g. 1/64) of the float32 rate.  Output
    channel 0/1 is the direct (positive m) sum, 2/3 the mirror (negative m) sum, which the caller stores at the
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
    # A tile above its `mlim` marches no degree and stores its zero accumulators.
    stop = jnp.where(_polar_skip(m, xv, valid, L=L, spin=spin), nstart, L)

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
        # The residual of `th - uh` uses the branch-free 2Sum, not the fast form
        # `(th - nxt) - uh`, which is off by ~1 ulp on this stack in either magnitude ordering.
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
            # Pad lanes carry `EX_PAD` and flush to zero through `_pow2`; `lexmr` is the same
            # array as `lexdr` (the mirror sign is +-1), so one alignment serves both sums.
            adr, adi, amr, ami = _accumulate_fixed(
                nxt, ex + lex,
                jnp.broadcast_to(plt.load(drh.at[row, ell]), xv.shape),
                jnp.broadcast_to(plt.load(dih.at[row, ell]), xv.shape),
                jnp.broadcast_to(plt.load(mrh.at[row, ell]), xv.shape),
                jnp.broadcast_to(plt.load(mih.at[row, ell]), xv.shape),
                adr, adi, amr, ami)
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
            nstart, stop, degree, (zb, zb, man, zb, ex, zb, zb, zi, zb, zb, zi))
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
        nstart, stop, degree, (zb, zb, man, zb, ex, zb, zb, zb, zb, zi,
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


def _window_prescale(c):
    """Divide a window's coefficients by the power of two at their maximum, returning the scale.

    With `max(|Re c|, |Im c|) < 2` inside the kernel, `|d^ell_{m,-s}| <= 1` and at most `L` terms,
    every synthesis accumulator is below `2^(2 + log2 L)` and no per-lane block exponent is needed
    (`_kern_synth`): the accumulated value is the true one, and the caller multiplies the result by
    `scale`.  Powers of two are exact, so this is bit-identical to accumulating at the caller's
    scale except for terms below `2^-126` of it, which the fixed binade flushes.
    """
    amax = jnp.max(jnp.maximum(jnp.abs(c.real), jnp.abs(c.imag)))
    e = jnp.where(amax > 0, jnp.floor(jnp.log2(jnp.where(amax > 0, amax, 1.0))), 0.0)
    scale = jnp.exp2(e)
    return c / scale, scale


def _synth_coeff(flm, m0, mb, L, *, mirror):
    """Coefficient sequence and exponent split for one window of orders.

    ``lgs`` is split into ``lex`` (integer, folded into the term exponent) and a ``[1, 2)`` factor
    that rides in the float32 coefficient together with the closed form's ``(-1)**m``; the mirror
    half carries the slice's own ``(-1)**(ell + |spin|)``, the ``theta -> pi - theta`` factor the
    table-based synthesis applies to negative orders.

    Split into a float32 high part plus limb: the contraction inside the kernel is float32
    (float64 is 1/64-rate here), so the coefficient keeps its precision only as a pair -- a
    float32 coefficient alone is re-spent at every degree of the same lane and its error grows
    like L * 2**-24.
    """
    from gmaster._sht import spin_slice as ss

    ms = m0 + jnp.arange(mb)
    cols = L - 1 + ms if not mirror else L - 1 - ms
    c = flm[:, cols]                                   # (L, mb) complex
    if mirror:
        c = c * ss._sign(L, SPIN)[:, None]
    lgs = _log2_norm(m0, mb, L)                        # (mb, L) float64
    lex = jnp.floor(lgs)
    sc = jnp.exp2(lgs - lex) * ((-1.0) ** ms)[:, None]
    c, scale = _window_prescale(c)
    out = []
    for part in (c.real.T * sc, c.imag.T * sc):
        hi = part.astype(jnp.float32)
        out.append(hi)
        out.append((part - hi.astype(jnp.float64)).astype(jnp.float32))
    drh, drl, dih, dil = out
    return lex.astype(jnp.int32), drh, drl, dih, dil, scale


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
    c, scale = _window_prescale(c)
    out = []
    for part in (c.real.T * sc * parity, c.imag.T * sc * parity):
        hi = part.astype(jnp.float32)
        out.append(hi)
        out.append((part - hi.astype(jnp.float64)).astype(jnp.float32))
    drh, drl, dih, dil = out
    return lex.astype(jnp.int32), drh, drl, dih, dil, scale


@partial(jax.jit, static_argnames=("L", "spin", "nside"))
def _inverse_impl(flm, *, L, spin, nside):

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
    # per-window form would copy the whole (ntheta, 2L) complex128 buffer once per window (1.5 GiB
    # per copy at Nside 2048); the concatenation is 1.8-3.3x faster and bit-identical.
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
            mc = tuple(jnp.where(keep, a, jnp.zeros_like(a)) for a in mc[:5]) + mc[5:]
        v = _call_synth(L, ntheta, ntile, mb)(*g[:10], *dc[:5], *mc[:5],
                                              jnp.asarray([m0], jnp.int32))
        v = v.reshape(mb, npad, 4)[:, :ntheta]
        v = v * jnp.stack([dc[5], dc[5], mc[5], mc[5]])
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
    from gmaster._sht import healpix

    positive = jnp.asarray(positive)
    theta = jnp.asarray(healpix._stable_thetas(L, nside), dtype=jnp.float64)
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
        v = _call_synth(L, north, ntile, mb, spin=0)(*g[:10], *dc[:5], *mc[:5],
                                                     jnp.asarray([m0], jnp.int32))
        v = v.reshape(mb, npad, 4)[:, :north]
        v = v * jnp.stack([dc[5], dc[5], mc[5], mc[5]])
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

    Same contract as `_theta_matrix.inverse_latitudinal` / `sht_pallas.scalar_inverse_latitudinal`
    (ring phi phase applied, `sqrt((2l+1)/4pi)` baked in), without the Legendre band.
    """
    if _march_v2.enabled(L):
        return _march_v2.inverse_latitudinal_positive(positive, phase, L=L, nside=nside)
    return _inverse_fold_impl(positive, phase, L=L, nside=nside)


def inverse_latitudinal(flm, *, L, spin, nside):
    """Synthesis latitudinal step with no Wigner-d table.

    ``(L, 2L-1)`` complex in (``flm[ell, L-1+m]``) to ``(4*nside-1, 2L)`` complex, the contract
    `healpix._inverse_latitudinal` expects.  The kernel marches the same ``d^l_(m,-2)`` rows as the
    analysis; see :func:`synth_requested` for accuracy.
    """
    if int(spin) != SPIN and not _march_v2.spin_enabled(spin, L):
        raise ValueError(f"march route implements spin=+{SPIN}, got spin={spin}")
    if _march_v2.spin_enabled(spin, L):
        return _march_v2.inverse_latitudinal(flm, L=L, spin=spin, nside=nside)
    return _inverse_impl(flm, L=L, spin=spin, nside=nside)


def clear_cache():
    """Drop the compiled kernels (tests, or to free the device)."""
    _CALLS.clear()
