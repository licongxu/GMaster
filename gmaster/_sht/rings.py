"""Azimuthal (ring) stage of the HEALPix transforms.

A HEALPix ring with `nphi` pixels is Fourier-transformed in longitude.  Equatorial-belt rings
all have `nphi = 4 nside >= L` and take one batched FFT.  Polar-cap rings are shorter than
the band limit, so their order-`m` coefficients alias; those rows use a chirp-Z (Bluestein)
transform, whose three constant factors depend on the geometry only and are cached here
(`_ring_analysis_tables`, `_spin_ring_*_tables`).  Above `_RING_FACTORS_KEPT_MAX_L`
(Nside 8192) only the ring lengths are kept and each row block builds its own factors.

Everything here is private to `gmaster._sht.healpix`.
"""

import gc
import os
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np

from .._config import ring_dtype


def _next_fast_len_pow2(size):
    return 1 << max(1, int(size) - 1).bit_length()


@lru_cache(maxsize=32)
def _ring_czt_constants_numpy(L, nside):
    ntheta = 4 * nside - 1
    npix = 12 * nside**2
    ring = np.arange(ntheta)
    nphi = 4 * np.minimum(np.minimum(ring + 1, nside), ntheta - ring)
    start = np.empty(ntheta, dtype=np.int64)
    north = np.arange(nside - 1)
    start[: nside - 1] = 2 * north * (north + 1)
    start[nside - 1 : 3 * nside] = 2 * nside * (nside - 1) + 4 * nside * np.arange(
        2 * nside + 1
    )
    k = np.arange(1, nside)
    start[4 * nside - 1 - k] = npix - 2 * k * (k + 1)
    width = 4 * nside
    offsets = np.arange(width, dtype=np.int64)[None, :]
    valid = offsets < nphi[:, None]
    gather = np.where(valid, start[:, None] + offsets, 0)
    return nphi, start, gather, valid.astype(bool), width


def _ring_czt_constants(L, nside):
    """`(nphi, start, None, None, width)`: the per-ring vectors only.

    The `(4 nside - 1, 4 nside)` gather / valid grids are not uploaded: at Nside 4096 the int64
    gather alone is 2.1 GB, and it was copied to the device by every table build and embedded as a
    constant in every jitted forward ring transform.  `_cap_pixels` forms the cap rows' pixel
    indices from `start + j, j < nphi` instead.
    """
    nphi, start, _, _, width = _ring_czt_constants_numpy(L, nside)
    return jnp.asarray(nphi), jnp.asarray(start), None, None, width


def _cap_pixels(pixels, caps, L, nside):
    """The polar-cap rows of the ring grid, zero past each ring's `nphi` pixels."""
    nphi, start, _, _, width = _ring_czt_constants_numpy(L, nside)
    off = jnp.arange(width, dtype=jnp.int32)[None, :]
    nrow = jnp.asarray(nphi[caps].astype(np.int32))[:, None]
    valid = off < nrow
    index = jnp.where(valid, jnp.asarray(start[caps])[:, None] + off, 0)
    return jnp.where(valid, pixels[index], 0.0)


@lru_cache(maxsize=32)
def _ring_inverse_layout_numpy(L, nside):
    """Per-pixel source slot of the (4*nside-1, 4*nside) ring grid, on the host.

    The inverse ring transform used to write the map with an index-add scatter,
    which needs atomics: the padded slots of every short ring all point at
    pixel 0. The HEALPix rings partition the map, so the same write is a pure
    gather of exactly one slot per pixel.
    """
    ntheta = 4 * nside - 1
    npix = 12 * nside**2
    width = 4 * nside
    _, _, gather, valid, _ = _ring_czt_constants_numpy(L, nside)
    slots = np.arange(ntheta * width, dtype=np.int64).reshape(ntheta, width)
    source = np.zeros(npix, dtype=np.int64)
    used = np.zeros(npix, dtype=bool)
    source[gather[valid]] = slots[valid]
    used[gather[valid]] = True
    return source, used


def _ring_inverse_layout(L, nside):
    # Built per call, like `_ring_czt_constants`: caching the device arrays here
    # would keep tracers from whichever trace first asked.
    source, used = _ring_inverse_layout_numpy(L, nside)
    return jnp.asarray(source), jnp.asarray(used)


# Ring chirp-Z tables are rebuilt when large instead of kept: in the node-space refinement the ring
# FFT runs once per map, and at Nside 4096 the kept tables were 6.5 GiB (spin) + 3.75 GiB (scalar).
_RING_TABLE_CACHE_BYTES = int(float(os.environ.get("GMASTER_RING_TABLE_CACHE_BYTES", str(2 ** 30))))


class _CacheInfo:
    """The `currsize` field of `functools.lru_cache`'s `cache_info()`."""

    def __init__(self, currsize):
        self.currsize = currsize


def _size_cached(fn):
    """`lru_cache(maxsize=2)` for results under `_RING_TABLE_CACHE_BYTES`; larger ones are rebuilt."""
    cache = {}

    def wrapper(*args):
        hit = cache.get(args)
        if hit is not None:
            return hit
        out = fn(*args)
        if sum(int(t.size) * t.dtype.itemsize for t in jax.tree.leaves(out)) <= _RING_TABLE_CACHE_BYTES:
            if len(cache) >= 2:
                cache.pop(next(iter(cache)))
            cache[args] = out
        return out

    wrapper.cache_clear = cache.clear
    wrapper.cache_info = lambda: _CacheInfo(len(cache))
    wrapper.__wrapped__ = fn
    wrapper.__doc__ = fn.__doc__
    return wrapper


def _chirp(index, two_nphi, sign, L, wide=None):
    """exp(sign i pi q^2 / nphi) in the ring stage's dtype.

    The angle is reduced exactly in integers as in `_chirp_angle`; for a complex64 ring stage the
    reduction runs as ((q mod m)^2) mod m (m = 2 nphi) in uint32 while m <= 65536 -- the square of a
    residue then fits -- and in int64 above (Nside > 8192); the angle is scaled in float64 and
    evaluated with float32 sincos, all in one fused program.  (An int32 square overflowed at Nside
    8192, where polar rings reach m = 65528: O(1) errors in every polar-cap ring transform.)  Built
    in complex128 (int64 squares, float64 sincos at 1/64 rate, complex128 kernel FFTs) these tables
    took 287 ms at Nside 4096 spin 2 -- too slow to rebuild per field, so 6.5 GiB stayed resident.
    """
    if jnp.dtype(ring_dtype(L)) == jnp.complex64:
        two_nphi = jnp.asarray(two_nphi)
        if wide is None:
            wide = int(np.max(np.asarray(two_nphi))) > 65536
        return _chirp_c64(jnp.asarray(index), two_nphi, sign=float(sign), wide=wide)
    reduced = (index.astype(jnp.int64) ** 2) % two_nphi
    return jnp.exp(sign * 1j * reduced * (jnp.pi / two_nphi) * 2.0)


@partial(jax.jit, static_argnames=("sign", "wide"))
def _chirp_c64(index, two_nphi, *, sign, wide=False):
    if wide:                                   # m > 65536: the residue's square needs 64 bits
        m = two_nphi.astype(jnp.int64)
        r = jnp.mod(index.astype(jnp.int64), m)
        reduced = jnp.mod(r * r, m)
    else:                                      # r < m <= 65536: r * r < 2^32 in uint32, exactly
        m = two_nphi.astype(jnp.uint32)
        r = jnp.mod(index.astype(jnp.int64), two_nphi.astype(jnp.int64)).astype(jnp.uint32)
        reduced = jnp.mod(r * r, m)
    ang = (reduced.astype(jnp.float64) * (2.0 * jnp.pi / m.astype(jnp.float64))).astype(jnp.float32)
    return jax.lax.complex(jnp.cos(ang), sign * jnp.sin(ang))


def _chirp_angle(index, two_nphi, inverse):
    """exp(+-i*pi*q^2/nphi) angles with exact integer modular reduction."""
    reduced = (index.astype(jnp.int64) ** 2) % two_nphi
    angle = reduced * (jnp.pi / two_nphi) * 2.0
    return jnp.where(inverse, -angle, angle)


@lru_cache(maxsize=32)
def _ring_split_numpy(L, nside):
    """Split the ring axis into the equatorial belt and the two polar caps.

    Rings `nside-1 .. 3*nside-1` all have `nphi = 4*nside`: a power-of-two FFT length that
    is also wider than `L = 3*nside-1`, and in RING order their pixels are one contiguous
    run. Their `m` block is therefore a plain transform of a reshape — no input chirp, no
    zero-pad, no kernel multiply, no second transform — while `2*(nside-1)` polar rings
    (`nphi = 4j`, not an FFT-fast length) keep the chirp-Z. That belt is `2*nside+1` of
    `4*nside-1` rings and 0.667 of the pixels (checked for 64/256/1024).

    The shortcut needs `L <= 4*nside`, because a length-`4*nside` transform returns exactly
    the band `m < 4*nside`. Beyond that the ring sums repeat with period `nphi` (measured
    3.5e-16, `.qwen/tmp/ring_alias_check.py`), so an overset band would have to be tiled
    back in; rather than pay a tile/alias pass for a band limit nobody fits anyway, `L >
    4*nside` puts every ring back in the chirp-Z and returns an empty belt.
    """
    ntheta = 4 * nside - 1
    if L > 4 * nside:
        return 0, 0, 0, np.arange(ntheta)
    belt_lo = nside - 1
    belt_hi = 3 * nside                      # exclusive
    caps = np.concatenate((np.arange(belt_lo), np.arange(belt_hi, ntheta)))
    belt_start = 2 * nside * (nside - 1)     # first belt pixel in RING order
    return belt_lo, belt_hi, belt_start, caps


@_size_cached
def _ring_analysis_tables(L, nside, device=None):
    """The constant factors of the analysis ring chirp-Z, built once per geometry.

    The convolution kernel and both chirp ramps depend only on (L, nside), but spelled
    inline they are recomputed *inside* the jitted transform on every call, and XLA does
    not fold them: a jitted program containing nothing but `fft(kernel)` measures **2.03 ms
    at Nside 1024** (5.02 ms with its own `exp`), against a 13.57 ms ring stage, and the
    chirp exponentials cost another 1.65 + 1.26 ms at 163/213 GB/s because `exp(i*angle)`
    over a (4n-1, N) array is transcendental-bound, not bandwidth-bound. A pipeline
    evaluation runs the ring stage 4-8 times, so this is paid repeatedly for values that
    never change. Returned as device arrays to be passed as jit *arguments* — closing over
    them would just move them into the jaxpr as literals.

    They cover the polar rings only (`_ring_split_numpy`): the equatorial belt needs no
    chirp-Z, so shipping belt rows here would be 2/3 of the table bytes for nothing.

    `device` is the device of the map being transformed (the multi-GPU path runs the ring
    stage on a non-default device, and jit arguments must be committed there).
    `ensure_compile_time_eval` makes the builder trace-safe: the small-`L` cores below jit
    the whole transform, so a cached tracer from the first trace would leak into the next
    one. With it the cached values are always concrete arrays.
    """
    with jax.ensure_compile_time_eval(), jax.default_device(device):
        nphi, _, _, _, width = _ring_czt_constants(L, nside)
        caps = _ring_split_numpy(L, nside)[-1]
        two_nphi = (2 * nphi[caps])[:, None] if caps.size else (2 * nphi[:1])[:, None]
        # Above `_RING_FACTORS_KEPT_MAX_L` only the ring lengths are kept, as for the polarised
        # tables: at Nside 8192 the three factors are 15 GiB and the one-shot cap transform
        # another ~32 GiB, which on top of a polarised field failed the mask's cuFFT plan.
        tables = ((jnp.asarray(two_nphi),) if L > _RING_FACTORS_KEPT_MAX_L
                  else _scalar_czt_factors(two_nphi, L, nside))
    return tables if device is None else tuple(
        jax.device_put(t, device) for t in tables
    )


def _scalar_czt_factors(two_nphi, L, nside):
    """The three chirp-Z factors of the spin-0 analysis for polar rows of ring lengths
    `two_nphi / 2`.  `wide` is taken from every polar ring of the geometry, as the whole-table
    build takes it, so a row block gets the same factors the table would hold."""
    width = 4 * nside
    transform_size = _next_fast_len_pow2(L + width)
    nphi, _, _, _, _ = _ring_czt_constants_numpy(L, nside)
    caps = _ring_split_numpy(L, nside)[-1]
    wide = int(2 * nphi[caps].max()) > 65536 if caps.size else int(2 * nphi[0]) > 65536
    factors = (
        _chirp(jnp.arange(width, dtype=jnp.int64), two_nphi, -1.0, L, wide),
        jnp.fft.fft(_chirp(jnp.arange(transform_size, dtype=jnp.int64) - (width - 1),
                           two_nphi, 1.0, L, wide), axis=-1),
        _chirp(jnp.arange(L, dtype=jnp.int64), two_nphi, -1.0, L, wide),
    )
    cplx = ring_dtype(L)
    return tuple(t.astype(cplx) for t in factors)


@partial(jax.jit, static_argnames=("L", "nside"))
def _forward_ring_fft_positive(map_flat, tables, *, L, nside):
    """Every ring's `m in [0, L)` block: plain FFT on the belt, chirp-Z on the caps.

    `_forward_healpix_fft` additionally fills the Hermitian mirror into a
    `(4*nside-1, 2L)` window because s2fft's latitudinal primitive takes that layout. The
    HEALPix pipeline asks for it and then slices it straight back off again
    (`ftm[:, L_work:]`), so a zeroed 2L-wide buffer, two update-slices and a flip/conj
    gather — 100 MB-scale at Nside 512, 400 MB-scale at 1024 — buy nothing there. Same
    numbers, fewer passes: `_forward_ring_fft` is this plus the fill.

    `tables` is `_ring_analysis_tables(L, nside)` (polar rows only). The belt is
    `pixels[belt_start : belt_start + (2*nside+1)*4*nside]` reshaped to
    `(2*nside+1, 4*nside)` and transformed by one `fft`, whose first `L` outputs are the
    same sums the chirp-Z would have produced — `nphi = 4*nside > L` there, so nothing is
    aliased and nothing is padded.
    """
    tables = tuple(tables)
    belt_lo, belt_hi, belt_start, caps = _ring_split_numpy(L, nside)
    width = 4 * nside
    cplx = ring_dtype(L) if len(tables) == 1 else tables[0].dtype
    pixels = jnp.reshape(jnp.asarray(map_flat), (-1,)).astype(jnp.zeros((), cplx).real.dtype)

    cap_out = None
    if caps.size:
        cap_pixels = _cap_pixels(pixels, caps, L, nside)
        transform_size = _next_fast_len_pow2(L + width)

        def czt(px, *factors):
            if len(factors) == 1:
                factors = _scalar_czt_factors(factors[0], L, nside)
            chirp_in, kernel_spec, chirp_out = factors
            embedded = jnp.pad(px * chirp_in, ((0, 0), (0, transform_size - width)))
            convolution = jnp.fft.ifft(
                jnp.fft.fft(embedded, axis=-1) * kernel_spec, axis=-1
            )
            return chirp_out * convolution[:, width - 1 : width - 1 + L]

        if len(tables) == 1:
            cap_out = _chunked_rows_seq(_CAP_CHUNK_ROWS, czt, cap_pixels, tables[0])
        else:
            cap_out = czt(cap_pixels, *tables)

    belt_rows = belt_hi - belt_lo
    if belt_rows == 0:
        return cap_out
    belt = jnp.reshape(pixels[belt_start : belt_start + belt_rows * width],
                       (belt_rows, width))
    belt_out = jnp.fft.fft(belt, axis=-1)[:, :L]
    if cap_out is None:
        return belt_out
    return jnp.concatenate((cap_out[:belt_lo], belt_out, cap_out[belt_lo:]), axis=0)


@partial(jax.jit, static_argnames=("L", "nside"))
def _forward_ring_fft(map_flat, tables, *, L, nside):
    """Exact HEALPix ring FFT for every ring as one batched chirp-Z transform."""
    positive = _forward_ring_fft_positive(map_flat, tables, L=L, nside=nside)
    ftm = jnp.zeros((4 * nside - 1, 2 * L), dtype=positive.dtype)
    ftm = ftm.at[:, L:].set(positive)
    ftm = ftm.at[:, 1:L].set(jnp.flip(jnp.conj(positive[:, 1:L]), axis=-1))
    return ftm


@partial(jax.jit, static_argnames=("L", "nside"))
def _inverse_ring_fft(ftm_positive, *, L, nside):
    """Exact HEALPix ring inverse FFT as one batched chirp-Z transform.

    Kept as the baseline `_inverse_ring_fft_herm` is checked against (identical to
    1.3e-15 at Nside 512, 2.07x slower); nothing in the shipped path calls it.
    """
    nphi, _, _, _, width = _ring_czt_constants(L, nside)
    ntheta = 4 * nside - 1
    positive = jnp.asarray(ftm_positive)
    full = jnp.zeros((ntheta, 2 * L), dtype=positive.dtype)
    full = full.at[:, L:].set(positive)
    full = full.at[:, 1:L].set(jnp.flip(jnp.conj(positive[:, 1:L]), axis=-1))
    two_nphi = (2 * nphi)[:, None]
    transform_size = _next_fast_len_pow2(2 * L - 1 + width)

    c_index = jnp.arange(2 * L, dtype=jnp.int64)
    embedded = full * jnp.exp(1j * _chirp_angle(c_index, two_nphi, False))
    embedded = jnp.pad(embedded, ((0, 0), (0, transform_size - 2 * L)))
    shift = jnp.arange(transform_size, dtype=jnp.int64) - (2 * L - 1)
    kernel = jnp.exp(-1j * _chirp_angle(shift, two_nphi, False))
    convolution = jnp.fft.ifft(
        jnp.fft.fft(embedded, axis=-1) * jnp.fft.fft(kernel, axis=-1), axis=-1
    )

    p_index = jnp.arange(width, dtype=jnp.int64)
    result = jnp.exp(1j * _chirp_angle(p_index, two_nphi, False)) * convolution[
        :, 2 * L - 1 : 2 * L - 1 + width
    ]
    wrap_phase = ((L % nphi)[:, None] * p_index[None, :]) % nphi[:, None]
    result *= jnp.exp(-1j * wrap_phase * (2 * jnp.pi / nphi)[:, None])

    source, used = _ring_inverse_layout(L, nside)
    slots = jnp.reshape(result.real, (-1,))
    return jnp.where(used, slots[source], 0.0)


@_size_cached
def _ring_synthesis_tables(L, nside, device=None):
    """Constant factors of the synthesis ring chirp-Z; see `_ring_analysis_tables`.

    Polar rows only — the equatorial belt inverts with one plain inverse FFT.
    """
    with jax.ensure_compile_time_eval(), jax.default_device(device):
        nphi, _, _, _, width = _ring_czt_constants(L, nside)
        caps = _ring_split_numpy(L, nside)[-1]
        two_nphi = (2 * nphi[caps])[:, None] if caps.size else (2 * nphi[:1])[:, None]
        transform_size = _next_fast_len_pow2(L + width - 1)
        m_index = jnp.arange(L, dtype=jnp.int64)
        p_index = jnp.arange(width, dtype=jnp.int64)
        shift = jnp.arange(transform_size, dtype=jnp.int64) - (L - 1)
        tables = (
            _chirp(m_index, two_nphi, 1.0, L),
            jnp.fft.fft(_chirp(shift, two_nphi, -1.0, L), axis=-1),
            _chirp(p_index, two_nphi, 1.0, L),
        )
        cplx = ring_dtype(L)
        tables = tuple(t.astype(cplx) for t in tables)
    return tables if device is None else tuple(
        jax.device_put(t, device) for t in tables
    )


@partial(jax.jit, static_argnames=("L", "width", "transform_size"))
def _cap_inverse_rows(cap_pos, chirp_m, kernel_spec, chirp_p, *, L, width, transform_size):
    embedded = jnp.pad(cap_pos * chirp_m, ((0, 0), (0, transform_size - L)))
    convolution = jnp.fft.ifft(
        jnp.fft.fft(embedded, axis=-1) * kernel_spec, axis=-1
    )
    return 2.0 * (chirp_p * convolution[:, L - 1 : L - 1 + width]) - cap_pos[:, :1]


@partial(jax.jit, static_argnames=("L", "width"))
def _belt_inverse_rows(rows, *, L, width):
    belt_pad = jnp.pad(rows, ((0, 0), (0, width - L)))
    return 2.0 * (jnp.fft.ifft(belt_pad, axis=-1) * width) - rows[:, :1]


def _pack_ring_rows(part, nphi_rows):
    """Valid pixels of one ring batch, in ring order. Short rings drop the pad."""
    real = jnp.real(part)
    counts = np.asarray(nphi_rows)
    if int(counts.min()) == real.shape[1]:
        packed = real.reshape(-1)
    else:
        cols = jnp.arange(real.shape[1])
        mask = cols[None, :] < jnp.asarray(counts)[:, None]
        packed = real[mask]
    packed = packed.astype(jnp.float64)
    packed.block_until_ready()
    return packed


def _inverse_ring_fft_herm_chunked(ftm_positive, *, L, nside):
    """Same ring inverse as `_inverse_ring_fft_herm`, one row-batch at a time.

    The batched form is jitted, so a Python loop inside it is one program and
    still asks for the full cuFFT work area. This loop runs outside jit. Each
    batch is packed to its real pixels; the full map is concatenated only
    after the spectrum is dropped.
    """
    belt_lo, belt_hi, _, caps = _ring_split_numpy(L, nside)
    width = 4 * nside
    device = getattr(ftm_positive, "device", None)
    if ftm_positive.dtype == ring_dtype(L):
        positive = ftm_positive
    else:
        positive = ftm_positive.astype(ring_dtype(L))
    positive.block_until_ready()
    chunk = 1024
    transform_size = _next_fast_len_pow2(L + width - 1)
    nphi, start, _, _, _ = _ring_czt_constants_numpy(L, nside)
    pieces = []

    def take_caps(lo, hi):
        rings = caps[lo:hi]
        if int(rings[-1]) - int(rings[0]) + 1 != hi - lo:
            raise RuntimeError("polar chunk is not one contiguous ring range")
        with jax.ensure_compile_time_eval(), jax.default_device(device):
            two_nphi = (2 * jnp.asarray(nphi)[rings])[:, None]
            chirp_m = _chirp(jnp.arange(L, dtype=jnp.int64), two_nphi, 1.0, L)
            kernel_spec = jnp.fft.fft(
                _chirp(
                    jnp.arange(transform_size, dtype=jnp.int64) - (L - 1),
                    two_nphi, -1.0, L,
                ),
                axis=-1,
            )
            chirp_p = _chirp(jnp.arange(width, dtype=jnp.int64), two_nphi, 1.0, L)
            kernel_spec.block_until_ready()
        part = _cap_inverse_rows(
            positive[jnp.asarray(rings)],
            chirp_m, kernel_spec, chirp_p,
            L=L, width=width, transform_size=transform_size,
        )
        part.block_until_ready()
        packed = _pack_ring_rows(part, nphi[rings])
        expect = int(start[int(rings[-1])] + nphi[int(rings[-1])] - start[int(rings[0])])
        if int(packed.shape[0]) != expect:
            raise RuntimeError("polar chunk packed the wrong number of pixels")
        return packed

    for lo in range(0, belt_lo, chunk):
        pieces.append(take_caps(lo, min(lo + chunk, belt_lo)))
        gc.collect()
    belt_pos = positive[belt_lo:belt_hi]
    for lo in range(0, belt_hi - belt_lo, chunk):
        part = _belt_inverse_rows(belt_pos[lo : lo + chunk], L=L, width=width)
        part.block_until_ready()
        rows = nphi[belt_lo + lo : belt_lo + lo + part.shape[0]]
        pieces.append(_pack_ring_rows(part, rows))
        gc.collect()
    for lo in range(belt_lo, int(caps.size), chunk):
        pieces.append(take_caps(lo, min(lo + chunk, int(caps.size))))
        gc.collect()
    positive = None
    belt_pos = None
    gc.collect()
    out = jnp.concatenate(pieces)
    out.block_until_ready()
    if int(out.shape[0]) != 12 * nside * nside:
        raise RuntimeError("ring inverse did not cover the map")
    return out


@partial(jax.jit, static_argnames=("L", "nside"))
def _inverse_ring_fft_herm(ftm_positive, tables, *, L, nside):
    """Real HEALPix synthesis ring transform, from the positive-m half only.

    `_inverse_ring_fft` mirrors the block into a centred window of 2L coefficients and
    chirp-Z transforms all of it, so the convolution length bound (`2L - 1 + width`)
    forces a transform of 8192 at Nside 512 and 16384 at 1024.  The mirror is exact by
    construction — `full[L - m] == conj(full[L + m])` — so the ring sum collapses to

        sum_{m=-L+1}^{L-1} F_m e^{2i pi p m / nphi} = 2 P(p) - F_0,
        P(p) = sum_{m=0}^{L-1} F_m e^{2i pi p m / nphi},

    which needs only L coefficients and a bound of `L + width - 1`: 4096 at Nside 512,
    8192 at 1024.  Same numbers, N log N cheaper.  The centred form's `wrap_phase`
    correction is the bookkeeping for the L-sample shift of the coefficient index and
    disappears here, because this form never shifts it.

    `tables` is `_ring_synthesis_tables(L, nside)` (polar rows only). On the equatorial
    belt `nphi = 4*nside > L`, so `P` is just `width * ifft(zero-pad(F))`: one transform,
    no chirp and no kernel.
    """
    chirp_m, kernel_spec, chirp_p = tables
    belt_lo, belt_hi, _, caps = _ring_split_numpy(L, nside)
    width = 4 * nside
    positive = jnp.asarray(ftm_positive).astype(chirp_m.dtype)

    cap_res = None
    if caps.size:
        cap_pos = positive[jnp.asarray(caps)]
        transform_size = _next_fast_len_pow2(L + width - 1)
        embedded = jnp.pad(cap_pos * chirp_m, ((0, 0), (0, transform_size - L)))
        convolution = jnp.fft.ifft(
            jnp.fft.fft(embedded, axis=-1) * kernel_spec, axis=-1
        )
        cap_res = 2.0 * (chirp_p * convolution[:, L - 1 : L - 1 + width]) - cap_pos[:, :1]

    belt_rows = belt_hi - belt_lo
    if belt_rows == 0:
        result = cap_res
    else:
        belt_pos = positive[belt_lo:belt_hi]
        belt_pad = jnp.pad(belt_pos, ((0, 0), (0, width - L)))
        belt_res = 2.0 * (jnp.fft.ifft(belt_pad, axis=-1) * width) - belt_pos[:, :1]
        result = belt_res if cap_res is None else jnp.concatenate(
            (cap_res[:belt_lo], belt_res, cap_res[belt_lo:]), axis=0
        )

    source, used = _ring_inverse_layout(L, nside)
    slots = jnp.reshape(result.real, (-1,))
    return jnp.where(used, slots[source], 0.0)


# Largest bandlimit whose polar-cap chirp-Z factors are kept as tables.  Above it (Nside 8192) the
# analysis kernel spectrum alone is (16382, 131072) complex64 = 16 GiB and the three synthesis
# factors 28 GiB, which with the fused transform's working set does not fit the card
# (`Failed to create cuFFT batched plan`); there the tables are each row's `2 nphi` and every
# row block builds its own factors -- elementwise chirps and one extra FFT per block.
_RING_FACTORS_KEPT_MAX_L = int(os.environ.get("GMASTER_RING_FACTORS_KEPT_MAX_L", "12288"))


def _spin_czt_factors(two_nphi, L, nside, *, synthesis):
    """The three chirp-Z factors of the polar rows with ring lengths `two_nphi / 2`.

    Analysis (pixels -> centred window m in [-(L-1), L)): the kernel is shifted by
    `(width-1) + (L-1)` to bring order `m = -(L-1)` under the output window and the convolution
    bound is `width + 2L - 1`.  Synthesis is the transpose.  `wide` is fixed from the geometry
    (every polar ring has `2 nphi < 8 nside`) so this also runs on traced row blocks.
    """
    width = 4 * nside
    wide = 8 * nside > 65536
    if synthesis:
        size = _next_fast_len_pow2(2 * L - 1 + width)
        first, last = jnp.arange(2 * L - 1, dtype=jnp.int64), jnp.arange(width, dtype=jnp.int64)
        shift, sign = jnp.arange(size, dtype=jnp.int64) - (2 * L - 2), 1.0
    else:
        size = _next_fast_len_pow2(width + 2 * L)
        first, last = jnp.arange(width, dtype=jnp.int64), jnp.arange(-(L - 1), L, dtype=jnp.int64)
        shift, sign = jnp.arange(size, dtype=jnp.int64) - (width - 1) - (L - 1), -1.0
    factors = (
        _chirp(first, two_nphi, sign, L, wide),
        jnp.fft.fft(_chirp(shift, two_nphi, -sign, L, wide), axis=-1),
        _chirp(last, two_nphi, sign, L, wide),
    )
    cplx = ring_dtype(L)
    return tuple(t.astype(cplx) for t in factors)


@_size_cached
def _spin_ring_analysis_tables(L, nside, device=None):
    """Constant factors of the polarised analysis ring chirp-Z.

    Polar rows only, like `_ring_analysis_tables`: the equatorial belt is served by one plain
    FFT (see `_forward_ring_fft_full`).  The centred window is `m in [-(L-1), L)`, so the
    kernel is shifted by `(width-1) + (L-1)` to bring order `m = -(L-1)` under the output
    window and the convolution bound is `width + 2L - 1`.
    """
    with jax.ensure_compile_time_eval(), jax.default_device(device):
        nphi, _, _, _, width = _ring_czt_constants(L, nside)
        caps = _ring_split_numpy(L, nside)[-1]
        two_nphi = (2 * nphi[caps])[:, None] if caps.size else (2 * nphi[:1])[:, None]
        tables = ((jnp.asarray(two_nphi),) if L > _RING_FACTORS_KEPT_MAX_L
                  else _spin_czt_factors(two_nphi, L, nside, synthesis=False))
    return tables if device is None else tuple(
        jax.device_put(t, device) for t in tables
    )


# Polar-cap rows per chirp-Z batch in the polarised ring stages.  All cap rows at once is
# `2*(nside-1)` rows of a `next_pow2(width + 2L)`-point transform: at Nside 4096 that is
# `(8190, 65536)` complex128 = 8.6 GiB *per intermediate* (pad, FFT, product, inverse FFT), which
# with the 14 GiB of resident tables is what exhausted the 71 GiB pool for a single spin-2 pass
# (`.qwen/tmp/chain_s36e.log`, session 36).  Rows are independent, so the stage runs in
# `fori_loop` chunks and the peak is one chunk's intermediates.  Chunks past the end are clamped
# to the last full window and rewrite identical rows.
_CAP_CHUNK_ROWS = int(os.environ.get("GMASTER_CAP_CHUNK_ROWS", "1024"))


def _chunked_rows(nrows, chunk, body):
    """`body(lo, hi)` over static row blocks `[lo, hi)`, concatenated.

    Static slices rather than a `fori_loop`: XLA's copy insertion duplicates every jit
    parameter that enters a while loop, and here those parameters are the chirp-Z tables
    (8.6 GiB each at Nside 4096); a Python loop of static slices lets XLA fuse each slice into
    its FFT chain instead.
    """
    return jnp.concatenate(
        [body(lo, min(lo + chunk, nrows)) for lo in range(0, nrows, chunk)], axis=0)


def _chunked_rows_seq(chunk, body, *rows):
    """`body(*blocks)` once per `chunk` rows, so only one block's FFT is live.

    Used when the chirp-Z factors are built inside the block (Nside 8192). Unrolling those
    blocks makes one program of every FFT at once: 64 GiB, then a 30 GiB allocation on a
    pool that has already reserved the whole 96 GB card.
    """
    nrows = rows[0].shape[0]
    if nrows <= chunk:
        return body(*rows)
    pad = (-nrows) % chunk
    padded = [jnp.pad(r, [(0, pad)] + [(0, 0)] * (r.ndim - 1)) for r in rows]
    nchunk = padded[0].shape[0] // chunk

    def step(i):
        blocks = []
        for r in padded:
            tail = r.shape[1:]
            blocks.append(jax.lax.dynamic_slice(
                r, (i * chunk,) + (0,) * (r.ndim - 1), (chunk,) + tail))
        return body(*blocks)

    parts = jax.lax.map(step, jnp.arange(nchunk))
    out_tail = parts.shape[2:]
    return jnp.reshape(parts, (nchunk * chunk,) + out_tail)[:nrows]


@partial(jax.jit, static_argnames=("L", "nside"))
def _forward_ring_fft_full(signal, tables, *, L, nside):
    """Complex ring FFT returning the full centered window m in [-(L-1), L).

    The ring sums are periodic in the order with period `nphi`, and a negative order is just
    a residue: `F_-m = F_(nphi-m)`.  On the equatorial belt `nphi = 4*nside`, so one plain
    FFT of the contiguous reshape holds the whole centred window and it comes out as two
    contiguous slices of that result — `F[4n-L+1 : 4n]` for `m < 0`, `F[0 : L]` for
    `m >= 0`.  That replaces a `next_pow2(width + 2L)` transform pair (8192 at Nside 512,
    16384 at 1024) with one length-`4*nside` transform on `2*nside+1` of the `4*nside-1`
    rings, with no chirp, pad or kernel multiply there at all.  It agrees with the chirp-Z
    form to 4e-16 relative for `L` up to the `4*nside` gate (`.qwen/tmp/spin2_ring_ab.py`).

    The polar rows keep the chirp-Z; `tables` is `_spin_ring_analysis_tables(L, nside)` so
    its ramps and kernel are built once per geometry instead of inside every trace.
    """
    tables = tuple(tables)
    belt_lo, belt_hi, belt_start, caps = _ring_split_numpy(L, nside)
    width = 4 * nside
    # The signal here is a helicity combination `Q -/+ iU`, so it takes the tables'
    # *complex* type: casting to their real type would silently drop the imaginary part.
    cplx = ring_dtype(L) if len(tables) == 1 else tables[0].dtype
    pixels = jnp.reshape(jnp.asarray(signal), (-1,)).astype(cplx)

    cap_out = None
    if caps.size:
        cap_pixels = _cap_pixels(pixels, caps, L, nside)
        transform_size = _next_fast_len_pow2(width + 2 * L)

        def czt(px, *factors):
            if len(factors) == 1:
                factors = _spin_czt_factors(factors[0], L, nside, synthesis=False)
            c_in, k_spec, c_out = factors
            embedded = jnp.pad(px * c_in, ((0, 0), (0, transform_size - width)))
            convolution = jnp.fft.ifft(jnp.fft.fft(embedded, axis=-1) * k_spec, axis=-1)
            return c_out * convolution[:, width - 1 : width - 1 + 2 * L - 1]

        nrows = int(caps.size)
        if nrows <= _CAP_CHUNK_ROWS:
            cap_out = czt(cap_pixels, *tables)
        elif len(tables) == 1:
            cap_out = _chunked_rows_seq(
                _CAP_CHUNK_ROWS, lambda px, two: czt(px, two), cap_pixels, tables[0])
        else:
            cap_out = _chunked_rows(
                nrows, _CAP_CHUNK_ROWS,
                lambda lo, hi: czt(*(a[lo:hi] for a in (cap_pixels, *tables))))

    belt_rows = belt_hi - belt_lo
    if belt_rows == 0:
        return cap_out
    belt = jnp.reshape(pixels[belt_start : belt_start + belt_rows * width],
                       (belt_rows, width))
    full = jnp.fft.fft(belt, axis=-1)
    belt_out = jnp.concatenate((full[:, width - L + 1 :], full[:, :L]), axis=1)
    if cap_out is None:
        return belt_out
    return jnp.concatenate((cap_out[:belt_lo], belt_out, cap_out[belt_lo:]), axis=0)


@_size_cached
def _spin_ring_synthesis_tables(L, nside, device=None):
    """Constant factors of the polarised synthesis ring chirp-Z; polar rows only."""
    with jax.ensure_compile_time_eval(), jax.default_device(device):
        nphi, _, _, _, width = _ring_czt_constants(L, nside)
        caps = _ring_split_numpy(L, nside)[-1]
        two_nphi = (2 * nphi[caps])[:, None] if caps.size else (2 * nphi[:1])[:, None]
        tables = ((jnp.asarray(two_nphi),) if L > _RING_FACTORS_KEPT_MAX_L
                  else _spin_czt_factors(two_nphi, L, nside, synthesis=True))
    return tables if device is None else tuple(
        jax.device_put(t, device) for t in tables
    )


@partial(jax.jit, static_argnames=("L", "nside"))
def _inverse_ring_fft_complex(centered, tables, *, L, nside):
    """Complex inverse ring FFT of a centered (ntheta, 2L-1) spectrum.

    The residue identity that serves the forward direction has a reverse: summing
    `F_m e^{2 pi i p m / nphi}` over a window wider than one period `nphi` only requires
    adding the coefficients that share a residue and transforming once.  On the belt that
    means placing `m in [0, L)` in slots `[0, L)` and `m in [-(L-1), 0)` in slots
    `[4n-L+1, 4n)` and doing one length-`4*nside` inverse transform instead of the chirp-Z
    pair at `next_pow2(2L-1 + width)`.  For `L > 2*nside` the two slot ranges overlap, so
    placement is the *sum* of two padded arrays rather than a concatenation — still slice
    arithmetic, no scatter.  Checked against the chirp-Z form at 8e-16 relative, including
    the overlapping `L = 3*nside` case.

    The polar rows keep the chirp-Z with `tables` = `_spin_ring_synthesis_tables(L, nside)`
    and the `wrap_phase` correction that accounts for the shift of their coefficient index.
    """
    tables = tuple(tables)
    belt_lo, belt_hi, _, caps = _ring_split_numpy(L, nside)
    nphi, _, _, _, width = _ring_czt_constants(L, nside)
    grid = jnp.asarray(centered).astype(ring_dtype(L) if len(tables) == 1 else tables[0].dtype)

    cap_res = None
    if caps.size:
        cap_grid = grid[jnp.asarray(caps)]
        transform_size = _next_fast_len_pow2(2 * L - 1 + width)
        p_index = jnp.arange(width, dtype=jnp.int64)
        caps_nphi = nphi[jnp.asarray(caps)]

        def czt(g, rows_nphi, *factors):
            if len(factors) == 1:
                factors = _spin_czt_factors(factors[0], L, nside, synthesis=True)
            c_c, k_spec, c_p = factors
            embedded = jnp.pad(g * c_c, ((0, 0), (0, transform_size - (2 * L - 1))))
            convolution = jnp.fft.ifft(jnp.fft.fft(embedded, axis=-1) * k_spec, axis=-1)
            res = c_p * convolution[:, 2 * L - 2 : 2 * L - 2 + width]
            rows_nphi = rows_nphi[:, None]
            wrap_phase = ((L - 1) % rows_nphi) * p_index[None, :] % rows_nphi
            return res * jnp.exp(-1j * wrap_phase * (2 * jnp.pi / rows_nphi))

        nrows = int(caps.size)
        if nrows <= _CAP_CHUNK_ROWS:
            cap_res = czt(cap_grid, caps_nphi, *tables)
        elif len(tables) == 1:
            cap_res = _chunked_rows_seq(
                _CAP_CHUNK_ROWS,
                lambda g, nphi, two: czt(g, nphi[:, 0], two),
                cap_grid, caps_nphi[:, None], tables[0])
        else:
            cap_res = _chunked_rows(
                nrows, _CAP_CHUNK_ROWS,
                lambda lo, hi: czt(*(a[lo:hi] for a in (cap_grid, caps_nphi, *tables))))

    belt_rows = belt_hi - belt_lo
    if belt_rows == 0:
        result = cap_res
    else:
        belt_grid = grid[belt_lo:belt_hi]
        zeros_left = jnp.zeros((belt_rows, width - L + 1), dtype=grid.dtype)
        zeros_right = jnp.zeros((belt_rows, width - L), dtype=grid.dtype)
        slot = (jnp.concatenate((belt_grid[:, L - 1 :], zeros_right), axis=1)
                + jnp.concatenate((zeros_left, belt_grid[:, :L - 1]), axis=1))
        belt_res = jnp.fft.ifft(slot, axis=-1) * width
        result = belt_res if cap_res is None else jnp.concatenate(
            (cap_res[:belt_lo], belt_res, cap_res[belt_lo:]), axis=0
        )

    source, used = _ring_inverse_layout(L, nside)
    slots = jnp.reshape(result, (-1,))
    return jnp.where(used, slots[source], 0.0)


def drop_ring_tables():
    """Evict the cached ring chirp-Z tables (they are rebuilt on the next transform)."""
    _ring_analysis_tables.cache_clear()
    _ring_synthesis_tables.cache_clear()
    _spin_ring_analysis_tables.cache_clear()
    _spin_ring_synthesis_tables.cache_clear()
