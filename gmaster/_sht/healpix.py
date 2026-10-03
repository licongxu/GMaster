"""HEALPix spherical-harmonic transforms: routing, refinement and the latitudinal stage.

A HEALPix transform is two separable stages:

    maps --[ring FFT, `rings`]--> ftm(theta, m) --[latitudinal, engines]--> alm(ell, m)

The latitudinal stage dominates the cost, and which engine runs it depends on the band limit
`L = lmax + 1`, the spin, the hardware and the memory available:

* ``march_v2``     CUDA difference-form Wigner-d march (the default on NVIDIA GPUs).
* ``dc``           divide-and-conquer engine, O(L^2 log L) per transform, for L in
                   [GMASTER_DC_MIN_L, GMASTER_DC_MAX_L] (host-built plan, cached on disk).
* ``theta_matrix`` / ``spin_slice``   precomputed Legendre / Wigner-d tables (small L).
* ``sht_pallas`` / ``spin_march``     Pallas kernels (portable fallback).

`map2alm` adds NaMaster's Jacobi refinement (`n_iter`): each iteration synthesises the current
alms, takes the residual against the input maps, and analyses it.  On the fold routes the
residual is formed in ring-Fourier space (`ring_fold_residual`) so no ring FFT runs inside the
loop.  Above `_RING_FACTORS_KEPT_MAX_L` (Nside 8192) the polarised transform runs as separate
programs with complex64 spectra so that one 96 GB card can hold it.
"""

import gc
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from s2fft.sampling import s2_samples
from s2fft.transforms import _ftm_flm_primitive
from s2fft.utils import healpix_ffts, quadrature_jax

from .._config import nmt_params, ring_dtype, table_dtype
from . import spin_march as _spin_march
from . import spin_slice as _spin_slice
from .cuda_gpu import on_cuda_gpu as _on_cuda_gpu
from .dfp32 import scalar_forward_latitudinal_dfp32
from .sht_pallas import (
    scalar_forward_latitudinal,
    scalar_inverse_latitudinal,
)
from .theta_matrix import band_bytes as _theta_band_bytes
from .theta_matrix import inverse_latitudinal as _theta_matrix_inverse_latitudinal
from .theta_matrix import inverse_latitudinal_pair as _theta_matrix_inverse_latitudinal_pair
from .theta_matrix import positive_latitudinal as _theta_matrix_latitudinal
from .theta_matrix import positive_latitudinal_pair as _theta_matrix_latitudinal_pair
from .theta_matrix import synth_pair_ready as _theta_matrix_synth_pair_ready
from .rings import (
    _forward_ring_fft,
    _forward_ring_fft_full,
    _forward_ring_fft_positive,
    _inverse_ring_fft_complex,
    _inverse_ring_fft_herm,
    _inverse_ring_fft_herm_chunked,
    _ring_analysis_tables,
    _RING_FACTORS_KEPT_MAX_L,
    _ring_synthesis_tables,
    _spin_ring_analysis_tables,
    _spin_ring_synthesis_tables,
)


def _stable_thetas(L, nside):
    theta = jnp.asarray(s2_samples.thetas(L, "healpix", nside))
    # Nudge theta off exact zero: s2fft's Price-McEwen recurrence drops subsequent modes when
    # its renormalisation hits an exact zero.  Remove once s2fft handles that case.
    return theta + 8 * jnp.finfo(theta.dtype).eps


@partial(jax.jit, static_argnames=("L", "nside", "reality"))
def _forward_s2fft_ftm(maps, tables, *, L, nside, reality):
    m_start = L - 1 if reality else 0
    if reality:
        ftm = healpix_ffts.healpix_fft(maps, L, nside, "jax", reality)
    else:
        # One batched ring FFT instead of s2fft's per-ring unroll, which issues one tiny FFT
        # per ring and is launch-bound.  Agrees with s2fft to ~1e-15.
        centered = _forward_ring_fft_full(maps, tables, L=L, nside=nside)
        ftm = jnp.concatenate(
            (jnp.zeros((centered.shape[0], 1), dtype=centered.dtype), centered),
            axis=1,
        )
    ftm = jnp.einsum(
        "tm,t->tm",
        ftm,
        quadrature_jax.quad_weights_transform(L, "healpix", nside),
        optimize=True,
    )
    ftm = ftm.at[:, m_start + 1 :].multiply(
        healpix_ffts.ring_phase_shifts_hp_jax(L, nside, True, reality)
    )
    return ftm


def _forward_latitudinal(ftm, *, L, spin, nside, reality, L_lower):
    if not reality and _spin_march.march_requested(spin, L=L, nside=nside):
        # Generate the Wigner-d rows inside the kernel instead of reading a precomputed slice
        # (too large above Nside ~512) or falling back to s2fft's slow generic scatter loop.
        # When a slice is resident the table route keeps the call; see `march_requested`.
        return _spin_march.forward_latitudinal(ftm, L=L, spin=spin, nside=nside)
    return _ftm_flm_primitive.ftm_to_flm(
        ftm,
        _stable_thetas(L, nside),
        L=L,
        spin=spin,
        nside=nside,
        sampling="healpix",
        reality=reality,
        spmd=False,
        L_lower=L_lower,
        precomps=None,
    )


@partial(jax.jit, static_argnames=("L", "spin", "reality"))
def _finish_forward_s2fft(flm, *, L, spin, reality):
    """Apply the degree normalisation, spin sign and ell < |spin| zeroing in one pass.

    All three factors are diagonal in `ell`, so they combine into one `(L,)` vector and the
    finish is a single read and write of the `(L, 2L-1)` block (1.2 GiB complex128 at
    Nside 2048 spin 2), rather than three memory-bound passes.
    """
    m_start = L - 1 if reality else 0
    ell = jnp.arange(L)
    factor = (
        jnp.sqrt((2 * ell + 1) / (4 * jnp.pi))
        * jnp.where(ell < abs(spin), 0.0, 1.0)
        * (-1.0) ** abs(spin)
    )
    if reality:
        # The negative-m half is mirrored from the positive half, so the degree scaling
        # must be applied before the mirror.
        flm = flm * jnp.sqrt((2 * ell + 1) / (4 * jnp.pi))[:, None]
        flm = flm.at[:, :m_start].set(
            jnp.flip(
                (-1) ** (jnp.arange(1, L) % 2) * jnp.conj(flm[:, m_start + 1 :]),
                axis=-1,
            )
        )
        return flm * (jnp.where(ell < abs(spin), 0.0, 1.0) * (-1.0) ** abs(spin))[:, None]
    return flm * factor[:, None]


def _forward_s2fft(maps, *, L, spin, nside, reality):
    """Not a jit boundary: the ring constants cross to the trace as arguments."""
    return _forward_s2fft_impl(
        maps,
        () if reality else _spin_ring_analysis_tables(
            L, nside, getattr(maps, "device", None)),
        L=L, spin=spin, nside=nside, reality=reality,
    )


@partial(jax.jit, static_argnames=("L", "spin", "nside", "reality"))
def _forward_s2fft_impl(maps, tables, *, L, spin, nside, reality):
    ftm = _forward_s2fft_ftm(maps, tables, L=L, nside=nside, reality=reality)
    flm = _forward_latitudinal(
        ftm, L=L, spin=spin, nside=nside, reality=reality, L_lower=0
    )
    return _finish_forward_s2fft(flm, L=L, spin=spin, reality=reality)


@partial(jax.jit, static_argnames=("L",))
def _prepare_inverse_s2fft(flm, *, L):
    return jnp.einsum(
        "lm,l->lm",
        flm,
        jnp.sqrt((2 * jnp.arange(L) + 1) / (4 * jnp.pi)),
        optimize=True,
    )


def _inverse_latitudinal(flm, theta, *, L, spin, nside, reality):
    if not reality and _spin_march.synth_requested(spin, L=L, nside=nside):
        # Table-free row march in the synthesis direction (sum over ell per theta lane), far
        # faster than the generic scatter loop.  `GMASTER_SPIN2_MARCH_SYNTH=0` restores the
        # exact s2fft route: this march agrees with ducc0 to ~2e-4 max (~6e-6 rms) at Nside
        # 1024, against ~1e-9 for the scatter loop.
        return _spin_march.inverse_latitudinal(flm, L=L, spin=spin, nside=nside)
    return _ftm_flm_primitive.flm_to_ftm(
        flm,
        theta,
        L=L,
        spin=spin,
        nside=nside,
        sampling="healpix",
        reality=reality,
        spmd=False,
        L_lower=0,
        precomps=None,
    )


@partial(jax.jit, static_argnames=("L", "spin", "nside", "reality"))
def _finish_inverse_s2fft(ftm, tables, *, L, spin, nside, reality):
    m_start = L - 1 if reality else 0
    ftm = ftm.at[:, m_start + 1 :].multiply(
        healpix_ffts.ring_phase_shifts_hp_jax(L, nside, False, reality)
    )
    ftm *= (-1) ** abs(spin)
    if reality:
        ftm = ftm.at[:, 1:L].set(jnp.flip(jnp.conj(ftm[:, L + 1 :]), axis=-1))
        return healpix_ffts.healpix_ifft(ftm, L, nside, "jax", reality)
    return _inverse_ring_fft_complex(ftm[:, 1:], tables, L=L, nside=nside)


def _inverse_s2fft(flm, *, L, spin, nside, reality):
    """Not a jit boundary: the ring constants cross to the trace as arguments."""
    return _inverse_s2fft_impl(
        flm,
        () if reality else _spin_ring_synthesis_tables(
            L, nside, getattr(flm, "device", None)),
        L=L, spin=spin, nside=nside, reality=reality,
    )


@partial(jax.jit, static_argnames=("L", "spin", "nside", "reality"))
def _inverse_s2fft_impl(flm, tables, *, L, spin, nside, reality):
    flm = _prepare_inverse_s2fft(flm, L=L)
    ftm = _inverse_latitudinal(
        flm,
        _stable_thetas(L, nside),
        L=L,
        spin=spin,
        nside=nside,
        reality=reality,
    )
    return _finish_inverse_s2fft(
        ftm, tables, L=L, spin=spin, nside=nside, reality=reality
    )


@lru_cache(maxsize=1)
def _gpu_devices():
    return tuple(device for device in jax.devices() if device.platform == "gpu")


def _copy_to_device(array, target):
    source = getattr(array, "device", None)
    if source == target:
        return array
    if hasattr(array, "block_until_ready"):
        array.block_until_ready()
    copied = jax.device_put(array, target)
    copied.block_until_ready()
    return copied


def _run_blocking(transform, array):
    result = transform(array)
    result.block_until_ready()
    return result


def _use_multi_gpu_sht(L):
    calculator = nmt_params.sht_calculator
    if calculator in ("jax-single", "jax-generic") or len(_gpu_devices()) < 2:
        return False
    return calculator == "jax-mgpu" or L >= 768


_SPIN_PALLAS_MAX_L = 768

# Size cap on the precomputed Legendre band, which is O(L^2 * nside) in the storage
# precision: 9.4 GiB at Nside 512 and 73.5 GiB at Nside 1024 in float64, half that with
# `set_table_precision("fp32")`.  Larger bands fall back to the fused kernel.  Synthesis
# uses a re-laid-out second copy only when the pool has room (`_theta_matrix._synth_band`)
# and otherwise reads the analysis layout, so this one number gates both directions.
# 40 GiB admits the float32 Nside 1024 band on an ~80 GB card; float64 Nside 1024 and
# Nside 2048 decline.  This only decides whether a build is attempted: the builders also
# decline on `RESOURCE_EXHAUSTED`, and `_spin_slice` checks live pool headroom.
_MATRIX_BAND_BUDGET = 40 * 1024**3

# Largest Wigner-d slab that may be passed into a fused analysis program.  Fusion pays off
# at small Nside (the slab is 2.44 GiB at Nside 256); at Nside 512 the slab is 18.7 GiB and a
# program holding it alongside the maps doubles the layout, so the split route is used.
_SLAB_FUSE_MAX_BYTES = 4 * 1024**3


def _slab_bytes(slab):
    return sum(int(a.nbytes) for a in slab) if isinstance(slab, tuple) else int(slab.nbytes)


def _spin_slabs(L_work, spin, *, nside):
    """Cached (analysis, synthesis) Wigner-d slabs for a polarised transform.

    ``(None, None)`` means keep the generic s2fft scatter loop: either the
    calculator asked for it explicitly, or the pair would not fit in memory.
    """
    if nmt_params.sht_calculator in ("jax-generic", "jax-mgpu"):
        return None, None
    if _spin_march._march_v2.enabled(L_work):
        # The v2 CUDA march is faster than the resident slab and needs none of its memory
        # (the slab route peaks at ~44 GiB at Nside 512 spin 2).
        return None, None
    return _spin_slice.slabs_for(
        _stable_thetas(L_work, nside), L=L_work, spin=spin, nside=nside
    )


def _use_pallas_sht(L, spin):
    if nmt_params.sht_calculator not in (
        "jax", "jax-single", "jax-mgpu", "jax-dfp32", "jax-matrix"
    ):
        return False
    # No lower bound on L: the generic s2fft latitudinal step is a scatter loop whose cost
    # barely falls with map size, so the fused scalar path wins even at Nside 16.
    # Spin-weighted fused kernels are not used: their closed-form Wigner-d seeds lose
    # relative precision through cancellation as |m| approaches l.
    if spin != 0:
        return False
    return _on_cuda_gpu()


def _pallas_block_size(nside):
    # 512 lanes per program: larger tiles under-utilise the SMs (synthesis degrades sharply
    # at 1024-2048), smaller ones repeat degree-loop work in analysis.  2*nside covers small maps.
    return min(512, 2 * nside)


# Above this working band limit the refinement loop runs op-by-op instead of as one traced
# program.  Up to 768 (Nside 256) the single program is bit-identical and ~9% faster; at
# Nside 512 it is slower, because the program then carries a 4.7 GiB band.
_PALLAS_TRACED_MAX_L = 768
# The same limit for the march routes (`GMASTER_MARCH_TRACED_MAX_L`).  Tracing the whole
# refinement loop at large L compiles for many minutes on small GPUs, so larger sizes run
# the already-jitted per-transform programs.
_MARCH_TRACED_MAX_L = int(os.environ.get("GMASTER_MARCH_TRACED_MAX_L", "768"))


def _trace_route_ready(nside, L_work):
    """Whether the refinement loop may run as one traced program at this size.

    The Legendre band can only be built outside a trace (`_theta_matrix._band` declines
    under one), so it is built here, at top level, before the route is chosen.  Otherwise
    the traced program would silently fall back to the slow on-the-fly kernel, or carry a
    table build that XLA re-runs on every call.
    """
    if not _prefer_theta_band(nside, L_work, 0):
        return L_work <= _MARCH_TRACED_MAX_L
    if L_work > _PALLAS_TRACED_MAX_L:
        return False
    from . import theta_matrix as _theta_matrix

    return _theta_matrix.warm(nside, L_work)


def _use_multi_gpu_pallas(L, values):
    if len(_gpu_devices()) < 2 or isinstance(values, jax.core.Tracer):
        return False
    calculator = nmt_params.sht_calculator
    return calculator == "jax-mgpu" or (calculator == "jax" and L >= 2048)


def _prefer_theta_band(nside, L, m_start):
    """Whether to use the precomputed Legendre band instead of the kernel's recurrence.

    The band holds the d^l_{m,0}(theta_j) values the fused kernel regenerates on every call,
    turning the latitudinal stage into a memory-bound contraction (alms agree to ~1e-14).
    Declined for an m-split, when the v2 CUDA march is enabled, or when the band exceeds
    `_MATRIX_BAND_BUDGET`.
    """
    if m_start != 0:
        return False
    if nmt_params.sht_calculator not in ("jax", "jax-matrix"):
        return False
    if _spin_march._march_v2.enabled(L):
        # The v2 CUDA march is faster than the resident band at every size.
        return False
    return _theta_band_bytes(nside, L, dtype=table_dtype()) <= _MATRIX_BAND_BUDGET


def _fused_forward_sht(positive, theta, weights, phase, *, L, block_size,
                       m_start=0):
    """Scalar forward latitudinal stage, dispatched on the configured calculator.

    Order of preference: the folded spin-0 march (`fold_requested`), the precomputed
    Legendre band while it fits (`_prefer_theta_band`; it cannot be built under a trace,
    e.g. on `jax.grad` paths), then the fp64 Pallas kernel.  The double-fp32 (DFP32) kernel
    replaces the fp64 one when selected; it is analysis-only and slightly less precise.
    """
    nside = (len(theta) + 1) // 4
    # The folded march builds phases for all m = 0..L-1, so it cannot serve an m-split slice.
    if (m_start == 0 and positive.shape[-1] == L
            and nmt_params.sht_calculator in ("jax", "jax-matrix")
            and _spin_march.fold_requested(nside, L)):
        return _spin_march.forward_latitudinal_positive(
            positive, weights, phase, L=L, nside=nside
        )
    if _prefer_theta_band(nside, L, m_start):
        positive_alm = _theta_matrix_latitudinal(
            positive, L=L, nside=nside, weights=weights, phase=phase
        )
        if positive_alm is not None:
            return positive_alm
    if nmt_params.sht_calculator == "jax-dfp32":
        return scalar_forward_latitudinal_dfp32(
            positive, theta, weights=weights, phase=phase,
            L=L, block_size=block_size, m_start=m_start,
        )
    return scalar_forward_latitudinal(
        positive, theta, weights=weights, phase=phase,
        L=L, block_size=block_size, m_start=m_start,
    )


def _fused_inverse_sht(positive, theta, phase, *, L, nside, block_size):
    """Scalar synthesis latitudinal stage, dispatched like `_fused_forward_sht`.

    HEALPix synthesis carries no quadrature weight (the ring transform has it), so the
    routes get a unit weight vector.  The band gate is the same single-band test as
    analysis: `_theta_matrix._synth_band` adds a re-laid-out copy only if the pool has room.
    """
    weights = jnp.ones_like(theta)
    if (nmt_params.sht_calculator in ("jax", "jax-matrix")
            and _spin_march.fold_synth_requested(nside, L)):
        return _spin_march.inverse_latitudinal_positive(positive, phase, L=L, nside=nside)
    if _prefer_theta_band(nside, L, 0):
        ftm = _theta_matrix_inverse_latitudinal(
            positive, L=L, nside=nside, weights=weights, phase=phase)
        if ftm is not None:
            return ftm
    return scalar_inverse_latitudinal(
        positive, theta, weights=weights, phase=phase,
        L=L, block_size=block_size,
    )


@lru_cache(maxsize=16)
def _pallas_order_split(L):
    approximate = round(L * (1 - 1 / np.sqrt(2)))
    candidates = range(max(1, approximate - 2), min(L, approximate + 3))
    total = L * (L + 1) // 2
    return min(
        candidates,
        key=lambda split: abs(split * (2 * L - split + 1) - total),
    )


def _pallas_fft_method():
    extension = getattr(healpix_ffts, "_s2fft", None)
    return (
        "cuda"
        if extension is not None
        and getattr(extension, "COMPILED_WITH_CUDA", False)
        else "jax"
    )


def _forward_healpix_fft(maps, *, L, nside, reality):
    """HEALPix ring FFT in the centred `(4*nside-1, 2L)` layout s2fft's primitive wants."""
    # Not a jit boundary: the ring constants are fetched outside the trace and passed as
    # arguments, so they are not rebuilt or baked in as constants.
    return _forward_ring_fft(
        maps, _ring_analysis_tables(L, nside, getattr(maps, "device", None)),
        L=L, nside=nside,
    )


def _finish_inverse_pallas(ftm_positive, *, L, nside):
    # At Nside 8192 the full polar-cap kernel is (16382, 65536) complex64 = 8 GiB, which
    # does not fit beside the spectrum, so the rings are transformed in chunks.
    if nside >= 8192:
        return _inverse_ring_fft_herm_chunked(ftm_positive, L=L, nside=nside)
    return _inverse_ring_fft_herm(
        ftm_positive,
        _ring_synthesis_tables(L, nside, getattr(ftm_positive, "device", None)),
        L=L, nside=nside,
    )


@lru_cache(maxsize=32)
def _forward_latitudinal_device(L, spin, nside, reality, L_lower, device_index):
    def transform(ftm):
        return _forward_latitudinal(
            ftm,
            L=L,
            spin=spin,
            nside=nside,
            reality=reality,
            L_lower=L_lower,
        )

    return jax.jit(transform)


@lru_cache(maxsize=32)
def _inverse_latitudinal_device(L, spin, nside, reality, half, device_index):
    def transform(flm):
        theta = _stable_thetas(L, nside)
        cut = (len(theta) + 1) // 2
        theta = theta[:cut] if half == 0 else theta[cut:]
        return _inverse_latitudinal(
            flm,
            theta,
            L=L,
            spin=spin,
            nside=nside,
            reality=reality,
        )

    return jax.jit(transform)


def _forward_s2fft_multi_gpu(maps, *, L, spin, nside, reality):
    primary, secondary = _gpu_devices()[:2]
    if not reality:
        # The ring FFT workspace (~16 GiB) does not fit beside the map, so run the ring
        # FFT and the march on the second GPU.
        maps = _copy_to_device(maps, secondary)
        ftm = _forward_s2fft_ftm(
            maps,
            _spin_ring_analysis_tables(L, nside, secondary),
            L=L, nside=nside, reality=reality,
        )
        ftm.block_until_ready()
        transform = _forward_latitudinal_device(L, spin, nside, reality, 0, 1)
        flm = _run_blocking(transform, ftm)
        ftm = None
        # Keep the spectrum on this GPU: copying it back leaves the primary card no room
        # for the alm gather's temporary.
        flm = _finish_forward_s2fft(flm, L=L, spin=spin, reality=reality)
        flm.block_until_ready()
        return flm
    maps = _copy_to_device(maps, primary)
    ftm = _forward_s2fft_ftm(
        maps, (), L=L, nside=nside, reality=reality,
    )
    split = max(abs(spin) + 1, 2 * L // 3)
    if split >= L:
        return _forward_s2fft(maps, L=L, spin=spin, nside=nside, reality=reality)

    low_input = _copy_to_device(ftm[:, L - split : L + split], secondary)
    low_transform = _forward_latitudinal_device(
        split, spin, nside, reality, 0, 1
    )
    high_transform = _forward_latitudinal_device(
        L, spin, nside, reality, split, 0
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        low_future = executor.submit(_run_blocking, low_transform, low_input)
        high_future = executor.submit(_run_blocking, high_transform, ftm)
        low, high = low_future.result(), high_future.result()
    low = _copy_to_device(low, primary)
    flm = high.at[:split, L - split : L + split - 1].set(low)
    return _finish_forward_s2fft(flm, L=L, spin=spin, reality=reality)


def _inverse_s2fft_multi_gpu(flm, *, L, spin, nside, reality):
    primary, secondary = _gpu_devices()[:2]
    flm = _copy_to_device(flm, primary)
    flm = _prepare_inverse_s2fft(flm, L=L)
    if not reality:
        # Spin synthesis ignores the theta half (a north/south split would be two full
        # marches), so run a single march on the second GPU.
        flm = _copy_to_device(flm, secondary)
        transform = _inverse_latitudinal_device(L, spin, nside, reality, 0, 1)
        ftm = _run_blocking(transform, flm)
        return _finish_inverse_s2fft(
            ftm,
            _spin_ring_synthesis_tables(L, nside, secondary),
            L=L, spin=spin, nside=nside, reality=reality,
        )
    other = _copy_to_device(flm, secondary)
    north_transform = _inverse_latitudinal_device(L, spin, nside, reality, 0, 0)
    south_transform = _inverse_latitudinal_device(L, spin, nside, reality, 1, 1)
    with ThreadPoolExecutor(max_workers=2) as executor:
        north_future = executor.submit(_run_blocking, north_transform, flm)
        south_future = executor.submit(_run_blocking, south_transform, other)
        north, south = north_future.result(), south_future.result()
    south = _copy_to_device(south, primary)
    ftm = jnp.concatenate((north, south))
    return _finish_inverse_s2fft(
        ftm,
        () if reality else _spin_ring_synthesis_tables(L, nside, primary),
        L=L, spin=spin, nside=nside, reality=reality,
    )


def _packed_triangle(alm, L):
    ell = jnp.arange(L)[:, None]
    order = jnp.arange(L)[None, :]
    packed_index = order * (2 * L - 1 - order) // 2 + ell
    return jnp.where(ell >= order, alm[packed_index], 0)


@partial(jax.jit, static_argnums=(1, 2))
def _unpack_real(alm, L, L_work):
    positive = _packed_triangle(alm, L)
    positive = jnp.pad(positive, ((0, L_work - L), (0, L_work - L)))
    negative = jnp.conj(positive[:, 1:L][:, ::-1])
    negative *= (-1) ** jnp.arange(L - 1, 0, -1)[None, :]
    negative = jnp.pad(negative, ((0, 0), (L_work - L, 0)))
    return jnp.concatenate((negative, positive), axis=1)


@partial(jax.jit, static_argnums=(1, 2))
def _unpack_spin(alm, L, L_work):
    e_positive = _packed_triangle(alm[0], L)
    b_positive = _packed_triangle(alm[1], L)
    positive = -(e_positive + 1j * b_positive)
    positive = jnp.pad(positive, ((0, L_work - L), (0, L_work - L)))
    negative = -(
        jnp.conj(e_positive[:, 1:L][:, ::-1])
        + 1j * jnp.conj(b_positive[:, 1:L][:, ::-1])
    )
    negative *= (-1) ** jnp.arange(L - 1, 0, -1)[None, :]
    negative = jnp.pad(negative, ((0, L_work - L), (L_work - L, 0)))
    return jnp.concatenate((negative, positive), axis=1)


def _polarised_ring_tables(spin, L_work, nside, like, *, synthesis):
    """Ring chirp-Z constants for a polarised transform, fetched outside any trace.

    They must enter the jitted cores as arguments: built inside a trace they would be baked
    into the program as HLO constants, copied to the host at lowering and re-embedded per
    executable.  At Nside 4096 the analysis table alone is `(8190, 65536)` complex64 =
    4 GiB, enough to exhaust device memory.  Spin 0 needs no tables and gets `()`.
    """
    if spin == 0:
        return ()
    device = getattr(like, "device", None)
    if synthesis:
        return _spin_ring_synthesis_tables(L_work, nside, device)
    return _spin_ring_analysis_tables(L_work, nside, device)


def _dc_spin(L_work, spin):
    """The divide-and-conquer engine for a polarised transform at this band limit, or None.

    Its spin-s calls are composed eagerly (ring FFT, latitudinal step, finish, each its own
    program) so the plan's arrays enter as jit arguments; traced inside the fused programs
    below they would be captured as constants (~20 GiB of HLO at Nside 4096).
    """
    return _spin_march._march_v2._dc(L_work) if spin == 2 else None


@partial(jax.jit, static_argnames=("L", "spin", "nside"))
def _march_forward_latitudinal_spin(ftm, *, L, spin, nside):
    return _forward_latitudinal(ftm, L=L, spin=spin, nside=nside, reality=False, L_lower=0)


@partial(jax.jit, static_argnames=("L", "spin", "nside"))
def _march_inverse_latitudinal_spin(flm, *, L, spin, nside):
    return _inverse_latitudinal(flm, _stable_thetas(L, nside), L=L, spin=spin, nside=nside,
                                reality=False)


class _MarchStages:
    forward_latitudinal_spin = staticmethod(_march_forward_latitudinal_spin)
    inverse_latitudinal_spin = staticmethod(_march_inverse_latitudinal_spin)


# At Nside 8192 the staged march passes its spectra between programs in complex64.  The v2
# kernels compute in float32, so no precision is lost; in complex128 the synthesis program alone
# needs ~74 GiB of inputs, outputs and temporaries, which does not fit a 96 GB card beside the
# resident maps and alms.
@partial(jax.jit, static_argnums=(1, 2))
def _prepare_spin_c64(alm, L, L_work):
    return _prepare_inverse_s2fft(_unpack_spin(alm, L, L_work), L=L_work).astype(jnp.complex64)


@partial(jax.jit, static_argnames=("L", "nside"))
def _forward_s2fft_ftm_c64(maps, tables, *, L, nside):
    return _forward_s2fft_ftm(maps[0] + 1j * maps[1], tables, L=L, nside=nside,
                              reality=False).astype(jnp.complex64)


@partial(jax.jit, static_argnames=("L", "nside"))
def _residual_ftm_c64(synth, maps, tables, *, L, nside):
    """Ring spectrum of `synth - maps`, with the difference formed inside the program.

    Materialising the difference would add a third full map (12 GiB at Nside 8192).
    """
    return _forward_s2fft_ftm_c64(synth - maps, tables, L=L, nside=nside)


def _march_c64_analysis(ftm_box, ell, order, *, L_work, spin, nside):
    """March analysis of the complex64 ring spectrum held in `ftm_box`.

    `ftm_box` is a one-element list that is emptied here, so the spectrum can be freed as
    soon as the march has read it.
    """
    flm = _spin_march._march_v2.forward_latitudinal(ftm_box.pop(), L=L_work, spin=spin,
                                                    nside=nside, cdtype=jnp.complex64)
    return _finish_pack_spin(flm, ell, order, L_work=L_work, spin=spin)


@partial(jax.jit, static_argnames=("L_work", "spin"))
def _finish_pack_spin(flm, ell, order, *, L_work, spin):
    """`_spin_pack_plus(_finish_forward_s2fft(flm))` without an intermediate `(L, 2L-1)` block.

    The finish factor is diagonal in `ell`, so it is applied to the gathered entries.
    """
    factor = (jnp.sqrt((2 * ell + 1) / (4 * jnp.pi)) * jnp.where(ell < abs(spin), 0.0, 1.0)
              * (-1.0) ** abs(spin))
    plus_m = flm[ell, L_work - 1 + order].astype(jnp.complex128) * factor
    minus_m = (-1) ** order * jnp.conj(flm[ell, L_work - 1 - order].astype(jnp.complex128) * factor)
    return jnp.stack([-(plus_m + minus_m) / 2, 0.5j * (plus_m - minus_m)])


@partial(jax.jit, static_argnames=("L", "spin", "nside"))
def _finish_inverse_c64(ftm, tables, *, L, spin, nside):
    """`_finish_inverse_s2fft` on a complex64 march output (ring phase applied in complex128)."""
    return _finish_inverse_s2fft(ftm.astype(jnp.complex128), tables, L=L, spin=spin, nside=nside,
                                 reality=False)


def _march_c64_ready(L_work, spin):
    return _spin_march._march_v2.enabled(L_work) and _spin_march.march_requested(
        spin, L=L_work, nside=None) and _spin_march.synth_requested(spin, L=L_work, nside=None)


def _staged_spin(L_work, spin):
    """The latitudinal engine that runs a polarised transform as separate programs, or None.

    The D&C engine where it serves; above `_RING_FACTORS_KEPT_MAX_L` (Nside 8192) also the
    march, whose fused synthesis program would need a ~72 GiB temporary.  Split into stages,
    each program's working set is that of one stage.
    """
    dc = _dc_spin(L_work, spin)
    if dc is None and spin != 0 and L_work > _RING_FACTORS_KEPT_MAX_L:
        return _MarchStages
    return dc


@partial(jax.jit, static_argnames=("L_work",))
def _spin_pack_plus(plus, ell, order, *, L_work):
    plus_m = plus[ell, L_work - 1 + order]
    minus_m = (-1) ** order * jnp.conj(plus[ell, L_work - 1 - order])
    return jnp.stack([-(plus_m + minus_m) / 2, 0.5j * (plus_m - minus_m)])


@jax.jit
def _complex_to_qu(maps):
    return jnp.stack([jnp.real(maps), jnp.imag(maps)])


def _alm2map_core(alm, ell, order, *, spin, nside, L, L_work):
    tables = _polarised_ring_tables(spin, L_work, nside, alm, synthesis=True)
    dc = _staged_spin(L_work, spin)
    if dc is _MarchStages and _march_c64_ready(L_work, spin):
        flm = _prepare_spin_c64(alm, L, L_work)
        ftm = _spin_march._march_v2.inverse_latitudinal(flm, L=L_work, spin=spin, nside=nside,
                                            cdtype=jnp.complex64)
        del flm
        maps = _finish_inverse_c64(ftm, tables, L=L_work, spin=spin, nside=nside)
        return _complex_to_qu(maps)
    if dc is not None:
        flm = _prepare_inverse_s2fft(_unpack_spin(alm, L, L_work), L=L_work)
        ftm = dc.inverse_latitudinal_spin(flm, L=L_work, spin=spin, nside=nside)
        maps = _finish_inverse_s2fft(ftm, tables, L=L_work, spin=spin, nside=nside, reality=False)
        return _complex_to_qu(maps)
    return _alm2map_core_impl(alm, tables, ell, order, spin=spin, nside=nside, L=L,
                              L_work=L_work)


@partial(
    jax.jit,
    static_argnames=("spin", "nside", "L", "L_work"),
)
def _alm2map_core_impl(alm, tables, ell, order, *, spin, nside, L, L_work):
    if spin == 0:
        elm = _unpack_real(alm[0], L, L_work)
        maps = _inverse_s2fft_impl(elm, (), L=L_work, spin=0, nside=nside, reality=True)
        return jnp.real(maps)[None, :]
    maps = _inverse_s2fft_impl(
        _unpack_spin(alm, L, L_work),
        tables,
        L=L_work,
        spin=spin,
        nside=nside,
        reality=False,
    )
    return jnp.stack([jnp.real(maps), jnp.imag(maps)])


def _map2alm_once(maps, ell, order, *, spin, nside, L, L_work):
    tables = _polarised_ring_tables(spin, L_work, nside, maps, synthesis=False)
    dc = _staged_spin(L_work, spin)
    if dc is _MarchStages and _march_c64_ready(L_work, spin):
        return _march_c64_analysis([_forward_s2fft_ftm_c64(maps, tables, L=L_work, nside=nside)],
                                   ell, order, L_work=L_work, spin=spin, nside=nside)
    if dc is not None:
        ftm = _forward_s2fft_ftm(maps[0] + 1j * maps[1], tables, L=L_work, nside=nside, reality=False)
        flm = dc.forward_latitudinal_spin(ftm, L=L_work, spin=spin, nside=nside)
        plus = _finish_forward_s2fft(flm, L=L_work, spin=spin, reality=False)
        return _spin_pack_plus(plus, ell, order, L_work=L_work)
    return _map2alm_once_impl(maps, tables, ell, order, spin=spin, nside=nside, L=L,
                              L_work=L_work)


@partial(
    jax.jit,
    static_argnames=("spin", "nside", "L", "L_work"),
)
def _map2alm_once_impl(maps, tables, ell, order, *, spin, nside, L, L_work):
    if spin == 0:
        flm = _forward_s2fft_impl(maps[0], (), L=L_work, spin=0, nside=nside, reality=True)
        return flm[ell, L_work - 1 + order][None, :]

    plus = _forward_s2fft_impl(
        maps[0] + 1j * maps[1],
        tables,
        L=L_work,
        spin=spin,
        nside=nside,
        reality=False,
    )
    plus_m = plus[ell, L_work - 1 + order]
    minus_m = (-1) ** order * jnp.conj(plus[ell, L_work - 1 - order])
    return jnp.stack([-(plus_m + minus_m) / 2, 0.5j * (plus_m - minus_m)])


# Above this ring-spectrum size one refinement iteration runs as two programs (synthesis, then
# analysis of the residual) instead of one.  XLA gives each program a single temporary buffer,
# and for the fused iteration it reaches ~26 GiB at Nside 4096 spin 2.
_ITERATION_SPLIT_BYTES = 4 * 1024 ** 3


def _map2alm_iteration(alm, maps, ell, order, *, spin, nside, L, L_work):
    if _staged_spin(L_work, spin) is _MarchStages and _march_c64_ready(L_work, spin):
        synth = _alm2map_core(alm, ell, order, spin=spin, nside=nside, L=L, L_work=L_work)
        box = [_residual_ftm_c64(
            synth, maps, _polarised_ring_tables(spin, L_work, nside, maps, synthesis=False),
            L=L_work, nside=nside)]
        del synth
        return alm - _march_c64_analysis(box, ell, order, L_work=L_work, spin=spin, nside=nside)
    if _staged_spin(L_work, spin) is not None:
        residual = _alm2map_core(alm, ell, order, spin=spin, nside=nside, L=L, L_work=L_work) - maps
        return alm - _map2alm_once(residual, ell, order, spin=spin, nside=nside, L=L, L_work=L_work)
    a_tables = _polarised_ring_tables(spin, L_work, nside, maps, synthesis=False)
    s_tables = _polarised_ring_tables(spin, L_work, nside, alm, synthesis=True)
    if (4 * nside - 1) * 2 * L_work * 16 > _ITERATION_SPLIT_BYTES:
        residual = _alm2map_core_impl(alm, s_tables, ell, order, spin=spin, nside=nside, L=L,
                                      L_work=L_work) - maps
        return alm - _map2alm_once_impl(residual, a_tables, ell, order, spin=spin, nside=nside,
                                        L=L, L_work=L_work)
    return _map2alm_iteration_impl(alm, maps, a_tables, s_tables, ell, order, spin=spin,
                                   nside=nside, L=L, L_work=L_work)


@partial(jax.jit, static_argnames=("spin", "nside", "L", "L_work"))
def _map2alm_iteration_impl(alm, maps, a_tables, s_tables, ell, order, *, spin, nside, L,
                            L_work):
    residual = (
        _alm2map_core_impl(
            alm, s_tables, ell, order, spin=spin, nside=nside, L=L, L_work=L_work
        )
        - maps
    )
    return alm - _map2alm_once_impl(
        residual, a_tables, ell, order, spin=spin, nside=nside, L=L, L_work=L_work
    )


@partial(jax.jit, static_argnames=("L",))
def _forward_latitudinal_slab(ftm, slab, *, L):
    """The slab contraction alone; see :func:`_inverse_latitudinal_slab`."""
    return _spin_slice.forward_latitudinal(ftm, slab, L=L)


@partial(jax.jit, static_argnames=("L",))
def _inverse_latitudinal_slab(flm, slab, *, L):
    """The slab contraction alone, as its own jit.

    A slab is tens of GiB from Nside 512 up.  A jit boundary whose arguments carry both
    slab layouts makes XLA count them twice against the pool and copy them on every call,
    so only the latitudinal step is jitted with the slab; the surrounding s2fft stages are
    jitted separately and never see it.
    """
    return _spin_slice.inverse_latitudinal(flm, slab, L=L)


def _map2alm_once_slab_body(maps, tables, ell, order, *, spin, nside, L, L_work, slab):
    ftm = _forward_s2fft_ftm(maps[0] + 1j * maps[1], tables,
                             L=L_work, nside=nside, reality=False)
    plus = _finish_forward_s2fft(
        _forward_latitudinal_slab(ftm, slab, L=L_work),
        L=L_work,
        spin=spin,
        reality=False,
    )
    plus_m = plus[ell, L_work - 1 + order]
    minus_m = (-1) ** order * jnp.conj(plus[ell, L_work - 1 - order])
    return jnp.stack([-(plus_m + minus_m) / 2, 0.5j * (plus_m - minus_m)])


_map2alm_once_slab_fused = jax.jit(
    _map2alm_once_slab_body, static_argnames=("spin", "nside", "L", "L_work")
)


def _map2alm_once_slab(maps, ell, order, *, spin, nside, L, L_work, slab):
    """Polarised analysis with the latitudinal step done as a slab contraction.

    While the slab is at most `_SLAB_FUSE_MAX_BYTES`, the ring FFT, contraction, finish and
    E/B combination run as one program; at small Nside this removes the per-op dispatch
    overhead of the E/B gather, which otherwise dominates.  Larger slabs run the stages
    separately, which is faster there (bit-identical results either way).
    """
    tables = _spin_ring_analysis_tables(
        L_work, nside,
        getattr(maps, "device", None) or getattr(maps[0], "device", None))
    if _slab_bytes(slab) <= _SLAB_FUSE_MAX_BYTES:
        return _map2alm_once_slab_fused(
            maps, tables, ell, order, spin=spin, nside=nside, L=L, L_work=L_work,
            slab=slab)
    return _map2alm_once_slab_body(
        maps, tables, ell, order, spin=spin, nside=nside, L=L, L_work=L_work,
        slab=slab)


def _alm2map_core_slab_body(alm, slab, tables, *, spin, nside, L, L_work):
    flm = _prepare_inverse_s2fft(_unpack_spin(alm, L, L_work), L=L_work)
    ftm = _inverse_latitudinal_slab(flm, slab, L=L_work)
    maps = _finish_inverse_s2fft(
        ftm, tables,
        L=L_work, spin=spin, nside=nside, reality=False,
    )
    return jnp.stack([jnp.real(maps), jnp.imag(maps)])


_alm2map_core_slab_fused = jax.jit(
    _alm2map_core_slab_body, static_argnames=("spin", "nside", "L", "L_work")
)


def _alm2map_core_slab(alm, *, spin, nside, L, L_work, slab):
    return _alm2map_core_slab_fused(
        alm, slab,
        _spin_ring_synthesis_tables(L_work, nside, getattr(alm, "device", None)),
        spin=spin, nside=nside, L=L, L_work=L_work)


# Largest working band limit whose whole polarised slab refinement loop is traced into one
# program (1536 is Nside 512; the slab route does not reach Nside 1024).  Tracing is
# bit-identical to the eager loop, ~6-14% faster on map2alm at Nside 256-512, and adds no
# device memory beyond the slabs themselves.
_SPIN_SLAB_TRACED_MAX_L = 1536


def _spin_slab_trace_ready(maps, L_work):
    """Whether the polarised slab refinement loop may run as one program at this size.

    Under an outer trace (e.g. `jax.grad`) the loop stays eager: one graph over `n_iter`
    passes would keep every iteration's residual alive for the transpose, a memory cost
    not validated for this route.
    """
    if isinstance(maps, jax.core.Tracer):
        return False
    return L_work <= _SPIN_SLAB_TRACED_MAX_L


@partial(jax.jit, static_argnames=("spin", "nside", "L", "L_work", "n_iter"))
def _map2alm_core_slab_traced(maps, ell, order, analysis_tables, synthesis_tables,
                              analysis_slab, synthesis_slab, *, spin, nside, L,
                              L_work, n_iter):
    """Polarised slab analysis with the refinement loop inside one XLA program.

    Ring tables and slabs are passed as arguments: a traced program may read tables but must
    never build them, since XLA would re-run the build on every call.
    """
    alm = _map2alm_once_slab_body(
        maps, analysis_tables, ell, order, spin=spin, nside=nside, L=L,
        L_work=L_work, slab=analysis_slab)
    for _ in range(n_iter):
        residual = _alm2map_core_slab_body(
            alm, synthesis_slab, synthesis_tables, spin=spin, nside=nside, L=L,
            L_work=L_work) - maps
        alm = alm - _map2alm_once_slab_body(
            residual, analysis_tables, ell, order, spin=spin, nside=nside, L=L,
            L_work=L_work, slab=analysis_slab)
    return alm


def _map2alm_iteration_slab(alm, maps, ell, order, *, spin, nside, L, L_work,
                            analysis_slab, synthesis_slab):
    residual = (
        _alm2map_core_slab(alm, spin=spin, nside=nside, L=L, L_work=L_work,
                           slab=synthesis_slab)
        - maps
    )
    return alm - _map2alm_once_slab(
        residual, ell, order, spin=spin, nside=nside, L=L, L_work=L_work,
        slab=analysis_slab,
    )


def _map2alm_core_slab(maps, ell, order, *, spin, nside, L, L_work, n_iter,
                       analysis_slab, synthesis_slab):
    if _spin_slab_trace_ready(maps, L_work):
        device = getattr(maps, "device", None) or getattr(maps[0], "device", None)
        return _map2alm_core_slab_traced(
            maps, ell, order,
            _spin_ring_analysis_tables(L_work, nside, device),
            _spin_ring_synthesis_tables(L_work, nside, device),
            analysis_slab, synthesis_slab,
            spin=spin, nside=nside, L=L, L_work=L_work, n_iter=n_iter)
    alm = _map2alm_once_slab(maps, ell, order, spin=spin, nside=nside, L=L,
                             L_work=L_work, slab=analysis_slab)
    for _ in range(n_iter):
        alm = _map2alm_iteration_slab(
            alm, maps, ell, order, spin=spin, nside=nside, L=L, L_work=L_work,
            analysis_slab=analysis_slab, synthesis_slab=synthesis_slab,
        )
    return alm


@partial(jax.jit, static_argnames=("L", "nside"))
def _spin_analysis_weigh(centered, *, L, nside):
    """Raw centred ring spectrum -> the weighted, phase-shifted ``(ntheta, 2L)`` analysis input."""
    ftm = jnp.concatenate((jnp.zeros((centered.shape[0], 1), centered.dtype), centered), axis=1)
    ftm = ftm * quadrature_jax.quad_weights_transform(L, "healpix", nside)[:, None]
    return ftm.at[:, 1:].multiply(healpix_ffts.ring_phase_shifts_hp_jax(L, nside, True, False))


@partial(jax.jit, static_argnames=("L", "nside", "spin"))
def _spin_synthesis_raw(ftm, *, L, nside, spin):
    """Latitudinal ``(ntheta, 2L)`` synthesis output -> the raw centred spectrum the ring IFFT takes."""
    return ftm[:, 1:] * healpix_ffts.ring_phase_shifts_hp_jax(L, nside, False, False) * (-1) ** abs(spin)


def _map2alm_core_dc_spin(maps, ell, order, *, spin, nside, L_work, n_iter, dc):
    """Spin-s Jacobi refinement in the D&C engine's node space.

    The engine's tree transforms are orthogonal (V^T V = I), so they cancel between
    iterations and are applied only once, after the loop.  Residuals are formed in
    ring-Fourier space by `ring_fold_residual_complex`.
    """
    tables = _polarised_ring_tables(spin, L_work, nside, maps, synthesis=False)
    fmap = _forward_ring_fft_full(maps[0] + 1j * maps[1], tables, L=L_work, nside=nside)
    fmap = fmap.astype(jnp.complex64)
    weights = quadrature_jax.quad_weights_transform(L_work, "healpix", nside)
    phi = healpix_ffts.p2phi_rings_jax(jnp.arange(4 * nside - 1), nside)
    w = dc.cd_forward_spin_raw(fmap, weights, phi, L=L_work, spin=spin, nside=nside)
    for _ in range(n_iter):
        raw = dc.cd_inverse_spin_raw(w, phi, L=L_work, spin=spin, nside=nside)
        resid = _spin_march._march_v2.ring_fold_residual_complex(raw, fmap, nside=nside)
        w = w - dc.cd_forward_spin_raw(resid, weights, phi, L=L_work, spin=spin, nside=nside)
    flm = dc.tree_finish_spin(w, L=L_work, spin=spin, nside=nside)
    plus = _finish_forward_s2fft(flm, L=L_work, spin=spin, reality=False)
    return _spin_pack_plus(plus, ell, order, L_work=L_work)


def _map2alm_core(maps, ell, order, *, spin, nside, L, L_work, n_iter):
    """Run Jacobi refinement without retaining every iteration in one XLA graph."""
    dc = _dc_spin(L_work, spin)
    if dc is not None and n_iter and L == L_work and _NODE_SPACE and _RING_FOLD \
            and _spin_march._march_v2.fold_available():
        return _map2alm_core_dc_spin(maps, ell, order, spin=spin, nside=nside, L_work=L_work,
                                     n_iter=n_iter, dc=dc)
    alm = _map2alm_once(
        maps, ell, order, spin=spin, nside=nside, L=L, L_work=L_work
    )
    for _ in range(n_iter):
        alm = _map2alm_iteration(
            alm,
            maps,
            ell,
            order,
            spin=spin,
            nside=nside,
            L=L,
            L_work=L_work,
        )
    return alm


@partial(jax.jit, static_argnames=("L", "L_work"))
def _positive_alm(alm, *, L, L_work):
    positive = _packed_triangle(alm, L)
    return jnp.pad(positive, ((0, L_work - L), (0, L_work - L)))


def _alm2map_core_pallas_eager(alm, *, nside, L, L_work, spin=0):
    if spin != 0:
        # Spin-weighted transforms never take the fused Pallas route (`_use_pallas_sht`).
        raise NotImplementedError("the fused Pallas synthesis is spin-0 only")
    positive = _positive_alm(alm[0], L=L, L_work=L_work)
    theta = _stable_thetas(L_work, nside)
    phase = healpix_ffts.p2phi_rings_jax(jnp.arange(len(theta)), nside)
    ftm_positive = _fused_inverse_sht(
        positive,
        theta,
        phase,
        L=L_work,
        nside=nside,
        block_size=_pallas_block_size(nside),
    )
    maps = _finish_inverse_pallas(ftm_positive, L=L_work, nside=nside)
    return jnp.real(maps)[None, :]


_alm2map_core_pallas_traced = jax.jit(
    _alm2map_core_pallas_eager, static_argnames=("nside", "L", "L_work", "spin")
)


def _alm2map_core_pallas(alm, *, nside, L, L_work, spin=0):
    """Synthesis, traced whole below `_PALLAS_TRACED_MAX_L` (see `_map2alm_core_pallas`)."""
    core = (_alm2map_core_pallas_traced if _trace_route_ready(nside, L_work)
            else _alm2map_core_pallas_eager)
    return core(alm, nside=nside, L=L, L_work=L_work, spin=spin)


def _shared_band_route(nside, L_work):
    """Which stages of two same-geometry spin-0 transforms may share one Legendre band.

    Returns ``(analysis, synthesis)``.  The MASTER pipeline runs such a pair: a field's
    `n_iter` refinement and, in `compute_coupling_matrix`, its mask's `n_iter_mask`
    refinement, both spin 0 when `lmax_mask == lmax`.  Both stream the same resident band,
    so one pass serves both at roughly half the cost of two, bit-identically.

    Requires that no march serves these geometries (e.g. forced by
    `GMASTER_SPIN0_MARCH=1`), that the band fits, and that it is concrete, so `warm`
    builds it here at top level.  Synthesis is reported separately because pairing only
    pays with the contiguous re-laid-out band copy (`_theta_matrix.inverse_latitudinal_pair`).
    """
    if not _prefer_theta_band(nside, L_work, 0):
        return (False, False)
    if (_spin_march.fold_requested(nside, L_work)
            or _spin_march.fold_synth_requested(nside, L_work)):
        return (False, False)
    from . import theta_matrix as _theta_matrix

    if not _theta_matrix.warm(nside, L_work):
        return (False, False)
    return (True, _theta_matrix_synth_pair_ready(nside, L_work))


def _march_pair_route(nside, L_work):
    """Whether the folded spin-0 march can serve a pair of same-geometry analyses.

    This is the pair route where no band exists (Nside 2048 and above).  The Legendre
    recurrence depends only on `(ell, m, theta)`, so two maps share it and only the emit
    doubles; a paired step costs ~0.5-0.75 of two separate calls.  Widening the emit block
    changes the reduction order, so results differ from the single march at ~1e-7 relative,
    with the same accuracy against the fp64 band.  Synthesis is not paired on this route:
    its per-lane accumulators scale with the number of maps, so nothing is shared.

    `fold_pair_fits` is a memory limit: at Nside 4096 the paired program does not fit
    (~8 GiB allocation fails), and the caller then runs two separate calls.
    """
    return (_spin_march.fold_requested(nside, L_work)
            and _spin_march.fold_pair_fits(nside, L_work))


def _map2alm_pair_once_pallas(maps_a, maps_b, ell, order, *, nside, L_work, march_pair,
                              return_ftm=False):
    """Two scalar analyses: one ring FFT per map, one shared latitudinal sweep.

    Only the latitudinal contraction is shared, since that is where the cost lies (band
    reads or march recurrence).  With ``return_ftm`` the two ring spectra are also
    returned, for reuse by the ring-fold refinement loop.
    """
    ftm_a = _forward_ring_fft_positive(
        maps_a[0],
        _ring_analysis_tables(
            L_work, nside,
            getattr(maps_a, "device", None) or getattr(maps_a[0], "device", None)),
        L=L_work, nside=nside,
    )
    ftm_b = _forward_ring_fft_positive(
        maps_b[0],
        _ring_analysis_tables(
            L_work, nside,
            getattr(maps_b, "device", None) or getattr(maps_b[0], "device", None)),
        L=L_work, nside=nside,
    )
    alm_a, alm_b = _pair_latitudinal_analysis(ftm_a, ftm_b, ell, order, nside=nside,
                                              L_work=L_work, march_pair=march_pair)
    if return_ftm:
        return alm_a, alm_b, ftm_a, ftm_b
    return alm_a, alm_b


def _pair_latitudinal_analysis(ftm_a, ftm_b, ell, order, *, nside, L_work, march_pair):
    """The latitudinal half of `_map2alm_pair_once_pallas`, from two ring spectra."""
    theta = _stable_thetas(L_work, nside)
    weights = quadrature_jax.quad_weights_transform(L_work, "healpix", nside)
    phase = -healpix_ffts.p2phi_rings_jax(jnp.arange(len(theta)), nside)
    if _spin_march._march_v2._dc(L_work, ftm_a) is not None:
        # The D&C engine pairs by sharing its plan traversal and has no pair memory gate.
        positive_a, positive_b = _spin_march._march_v2.forward_latitudinal_positive_pair(
            ftm_a, ftm_b, weights, phase, L=L_work, nside=nside)
    elif march_pair:
        positive_a, positive_b = _spin_march.forward_latitudinal_positive_pair(
            ftm_a, ftm_b, weights, phase, L=L_work, nside=nside)
    elif _spin_march._march_v2.enabled(L_work) and not _prefer_theta_band(nside, L_work, 0):
        # Pair route taken for the synthesis only: above `_march_v2._PAIR_MAX_L` the paired
        # analysis kernel is slower (it needs a half-size theta tile and writes 4x the tile
        # partials), so the two analyses run as separate marched calls.
        positive_a = _spin_march.forward_latitudinal_positive(
            ftm_a, weights, phase, L=L_work, nside=nside)
        positive_b = _spin_march.forward_latitudinal_positive(
            ftm_b, weights, phase, L=L_work, nside=nside)
    else:
        positive_a, positive_b = _theta_matrix_latitudinal_pair(
            ftm_a, ftm_b, L=L_work, nside=nside, weights=weights, phase=phase)
    return positive_a[ell, order][None, :], positive_b[ell, order][None, :]


def _pair_latitudinal_synthesis(alm_a, alm_b, *, nside, L, L_work):
    """The latitudinal half of `_alm2map_core_pallas_pair_eager`: two positive ring spectra."""
    theta = _stable_thetas(L_work, nside)
    phase = healpix_ffts.p2phi_rings_jax(jnp.arange(len(theta)), nside)
    positive_a = _positive_alm(alm_a[0], L=L, L_work=L_work)
    positive_b = _positive_alm(alm_b[0], L=L, L_work=L_work)
    if _spin_march._march_v2.enabled(L_work):
        return _spin_march._march_v2.inverse_latitudinal_positive_pair(
            positive_a, positive_b, phase, L=L_work, nside=nside)
    return _theta_matrix_inverse_latitudinal_pair(
        positive_a, positive_b,
        L=L_work, nside=nside, weights=jnp.ones_like(theta), phase=phase)


def _alm2map_core_pallas_pair_eager(alm_a, alm_b, *, nside, L, L_work):
    """Two scalar syntheses over one sweep of the row source (band or v2 march)."""
    ftm_a, ftm_b = _pair_latitudinal_synthesis(alm_a, alm_b, nside=nside, L=L, L_work=L_work)
    return (
        jnp.real(_finish_inverse_pallas(ftm_a, L=L_work, nside=nside))[None, :],
        jnp.real(_finish_inverse_pallas(ftm_b, L=L_work, nside=nside))[None, :],
    )


# The refinement residual `FFT(IFFT(F) - map)` of a HEALPix ring equals the n_phi-periodic fold
# of F minus the map's own spectrum (`_march_v2.ring_fold_residual`), so the refinement loop keeps
# the spectra from its first analysis and runs no further ring FFTs.  `GMASTER_RING_FOLD=0`
# restores synthesis to pixels.
_RING_FOLD = os.environ.get("GMASTER_RING_FOLD", "1") != "0"
# Refinement in the D&C engine's node space (V^T V = I, so the trees cancel between iterations).
# `GMASTER_DC_NODE_SPACE=0` disables it.
_NODE_SPACE = os.environ.get("GMASTER_DC_NODE_SPACE", "1") != "0"


def _ring_fold_ready(L_work):
    return _RING_FOLD and _spin_march._march_v2.fold_available()


def _single_latitudinal_synthesis(alm, *, nside, L, L_work):
    """The latitudinal half of the scalar `_alm2map_core_pallas_eager`: one positive ring spectrum."""
    theta = _stable_thetas(L_work, nside)
    phase = healpix_ffts.p2phi_rings_jax(jnp.arange(len(theta)), nside)
    positive = _positive_alm(alm[0], L=L, L_work=L_work)
    if (nmt_params.sht_calculator in ("jax", "jax-matrix")
            and _spin_march.fold_synth_requested(nside, L_work)
            and _spin_march._march_v2.enabled(L_work)):
        # Every caller passes this to `ring_fold_residual`, which reads complex64; emitting
        # complex64 directly avoids a complex128 copy (12 GiB at Nside 8192).
        return _spin_march._march_v2.inverse_latitudinal_positive(
            positive, phase, L=L_work, nside=nside, cdtype=jnp.complex64)
    return _fused_inverse_sht(positive, theta, phase,
                              L=L_work, nside=nside, block_size=_pallas_block_size(nside))


def _single_latitudinal_analysis(ftm, ell, order, *, nside, L_work):
    """The latitudinal half of the scalar `_map2alm_once_pallas`, from a ring spectrum."""
    theta = _stable_thetas(L_work, nside)
    weights = quadrature_jax.quad_weights_transform(L_work, "healpix", nside)
    phase = -healpix_ffts.p2phi_rings_jax(jnp.arange(len(theta)), nside)
    positive = _fused_forward_sht(ftm, theta, weights, phase, L=L_work,
                                  block_size=_pallas_block_size(nside))
    return positive[ell, order][None, :]


def _map2alm_core_pallas_pair_eager(
    maps_a, maps_b, ell, order, *, nside, L, L_work, n_iter, pair_synth, march_pair
):
    """Two independent Jacobi refinements stepping in lockstep over one row source.

    Each map refines its own alms; they share only the latitudinal passes.  `pair_synth`
    and `march_pair` are static: the route is chosen before any trace (a traced program must
    not branch on a table cache), and they select different kernels.
    """
    dc = _spin_march._march_v2._dc(L_work)
    if n_iter and dc is not None and L == L_work and _ring_fold_ready(L_work):
        # The D&C engine serves both latitudinal stages: the refinement runs in its packed alm
        # layout (fp64 accumulation) and converts to the output packing once at the end.
        ftms = [_forward_ring_fft_positive(
                    m[0],
                    _ring_analysis_tables(
                        L_work, nside, getattr(m, "device", None) or getattr(m[0], "device", None)),
                    L=L_work, nside=nside)
                for m in (maps_a, maps_b)]
        weights = quadrature_jax.quad_weights_transform(L_work, "healpix", nside)
        phi = healpix_ffts.p2phi_rings_jax(jnp.arange(4 * nside - 1), nside)
        if _NODE_SPACE:
            # Node-space refinement: the tree transforms cancel between iterations.
            return dc.refine_s0(ftms, weights, phi, ell, order, L=L_work, nside=nside, n_iter=n_iter)
        else:
            acc = dc.forward_packed(ftms, weights, -phi, L=L_work, nside=nside).astype(jnp.complex128)
            for _ in range(n_iter):
                rings = dc.inverse_packed(acc, phi, L=L_work, nside=nside)
                resid = [_spin_march._march_v2.ring_fold_residual(r, f, nside=nside)
                         for r, f in zip(rings, ftms)]
                acc = acc - dc.forward_packed(resid, weights, -phi, L=L_work, nside=nside)
        return tuple(dc.packed_to_alm(acc[:, k], ell, order, L=L_work, nside=nside)[None, :]
                     for k in range(2))
    if n_iter and _ring_fold_ready(L_work):
        alm_a, alm_b, ftm_a, ftm_b = _map2alm_pair_once_pallas(
            maps_a, maps_b, ell, order, nside=nside, L_work=L_work, march_pair=march_pair,
            return_ftm=True)
        for _ in range(n_iter):
            if pair_synth:
                syn_a, syn_b = _pair_latitudinal_synthesis(alm_a, alm_b, nside=nside, L=L,
                                                           L_work=L_work)
            else:
                syn_a = _single_latitudinal_synthesis(alm_a, nside=nside, L=L, L_work=L_work)
                syn_b = _single_latitudinal_synthesis(alm_b, nside=nside, L=L, L_work=L_work)
            delta_a, delta_b = _pair_latitudinal_analysis(
                _spin_march._march_v2.ring_fold_residual(syn_a, ftm_a, nside=nside),
                _spin_march._march_v2.ring_fold_residual(syn_b, ftm_b, nside=nside),
                ell, order, nside=nside, L_work=L_work, march_pair=march_pair)
            alm_a -= delta_a
            alm_b -= delta_b
        return alm_a, alm_b
    alm_a, alm_b = _map2alm_pair_once_pallas(
        maps_a, maps_b, ell, order, nside=nside, L_work=L_work, march_pair=march_pair
    )
    for _ in range(n_iter):
        if pair_synth:
            synth_a, synth_b = _alm2map_core_pallas_pair(
                alm_a, alm_b, nside=nside, L=L, L_work=L_work
            )
        else:
            synth_a = _alm2map_core_pallas(
                alm_a, nside=nside, L=L, L_work=L_work, spin=0
            )
            synth_b = _alm2map_core_pallas(
                alm_b, nside=nside, L=L, L_work=L_work, spin=0
            )
        delta_a, delta_b = _map2alm_pair_once_pallas(
            synth_a - maps_a, synth_b - maps_b, ell, order,
            nside=nside, L_work=L_work, march_pair=march_pair,
        )
        alm_a -= delta_a
        alm_b -= delta_b
    return alm_a, alm_b


_alm2map_core_pallas_pair_traced = jax.jit(
    _alm2map_core_pallas_pair_eager, static_argnames=("nside", "L", "L_work")
)

_map2alm_core_pallas_pair_traced = jax.jit(
    _map2alm_core_pallas_pair_eager,
    static_argnames=("nside", "L", "L_work", "n_iter", "pair_synth", "march_pair"),
)


def _alm2map_core_pallas_pair(alm_a, alm_b, *, nside, L, L_work):
    """Paired synthesis, traced on the same condition as the paired analysis."""
    core = (_alm2map_core_pallas_pair_traced if _trace_route_ready(nside, L_work)
            else _alm2map_core_pallas_pair_eager)
    return core(alm_a, alm_b, nside=nside, L=L, L_work=L_work)


def _map2alm_core_pallas_pair(
    maps_a, maps_b, ell, order, *, nside, L, L_work, n_iter, pair_synth, march_pair
):
    """Paired analysis, traced as one program on the same condition as the single route.

    See `_map2alm_core_pallas`.  The march pairing only serves sizes above
    `_PALLAS_TRACED_MAX_L`, so in practice it runs op-by-op.
    """
    core = (_map2alm_core_pallas_pair_traced if _trace_route_ready(nside, L_work)
            else _map2alm_core_pallas_pair_eager)
    return core(maps_a, maps_b, ell, order, nside=nside, L=L, L_work=L_work,
                n_iter=n_iter, pair_synth=pair_synth, march_pair=march_pair)


@lru_cache(maxsize=32)
def _ell_order_arrays(lmax):
    ell = np.arange(lmax + 1)[None, :]
    m = np.arange(lmax + 1)[:, None]
    mask = m <= ell
    ells = np.broadcast_to(ell, (lmax + 1, lmax + 1))[mask.T]
    ms = np.broadcast_to(m, (lmax + 1, lmax + 1))[mask.T]
    return jnp.asarray(ells), jnp.asarray(ms)


def _map2alm_once_pallas(maps, ell, order, *, nside, L_work, spin=0):
    if spin != 0:
        # Spin-weighted transforms never take the fused Pallas route (`_use_pallas_sht`).
        raise NotImplementedError("the fused Pallas analysis is spin-0 only")
    ftm = _forward_ring_fft_positive(
        maps[0],
        _ring_analysis_tables(
            L_work, nside,
            getattr(maps, "device", None) or getattr(maps[0], "device", None)),
        L=L_work, nside=nside,
    )
    theta = _stable_thetas(L_work, nside)
    weights = quadrature_jax.quad_weights_transform(
        L_work, "healpix", nside
    )
    phase = -healpix_ffts.p2phi_rings_jax(jnp.arange(len(theta)), nside)
    positive = _fused_forward_sht(
        ftm,
        theta,
        weights,
        phase,
        L=L_work,
        block_size=_pallas_block_size(nside),
    )
    return positive[ell, order][None, :]


def _map2alm_core_pallas(
    maps, ell, order, *, nside, L, L_work, n_iter, spin=0
):
    """Pallas-route analysis with refinement, traced as one program or run op-by-op.

    At small band limit the per-primitive host dispatch dominates (at Nside 128 the device
    work is about a third of the op-by-op wall time), so tracing the refinement loop into one
    program is 2-3x faster with bit-identical output.  `_trace_route_ready` decides, building
    the Legendre band first; see `_PALLAS_TRACED_MAX_L` for the size limit.
    """
    core = (_map2alm_core_pallas_traced if _trace_route_ready(nside, L_work)
            else _map2alm_core_pallas_eager)
    return core(maps, ell, order, nside=nside, L=L, L_work=L_work,
                n_iter=n_iter, spin=spin)


def _map2alm_core_pallas_eager(
    maps, ell, order, *, nside, L, L_work, n_iter, spin=0
):
    dc = _spin_march._march_v2._dc(L_work) if spin == 0 else None
    if dc is not None and n_iter and L == L_work and _NODE_SPACE and _ring_fold_ready(L_work):
        # One map refined in the D&C engine's node space (the paired loop with one right-hand side).
        ftm = _forward_ring_fft_positive(
            maps[0],
            _ring_analysis_tables(
                L_work, nside,
                getattr(maps, "device", None) or getattr(maps[0], "device", None)),
            L=L_work, nside=nside,
        )
        weights = quadrature_jax.quad_weights_transform(L_work, "healpix", nside)
        phi = healpix_ffts.p2phi_rings_jax(jnp.arange(4 * nside - 1), nside)
        return dc.refine_s0([ftm], weights, phi, ell, order, L=L_work, nside=nside, n_iter=n_iter)[0]
    if spin == 0 and n_iter and _ring_fold_ready(L_work):
        ftm = _forward_ring_fft_positive(
            maps[0],
            _ring_analysis_tables(
                L_work, nside,
                getattr(maps, "device", None) or getattr(maps[0], "device", None)),
            L=L_work, nside=nside,
        )
        alm = _single_latitudinal_analysis(ftm, ell, order, nside=nside, L_work=L_work)
        for _ in range(n_iter):
            syn = _single_latitudinal_synthesis(alm, nside=nside, L=L, L_work=L_work)
            alm -= _single_latitudinal_analysis(
                _spin_march._march_v2.ring_fold_residual(syn, ftm, nside=nside),
                ell, order, nside=nside, L_work=L_work)
        return alm
    alm = _map2alm_once_pallas(
        maps, ell, order, nside=nside, L_work=L_work, spin=spin
    )
    for _ in range(n_iter):
        residual = (
            _alm2map_core_pallas(
                alm, nside=nside, L=L, L_work=L_work, spin=spin
            )
            - maps
        )
        alm -= _map2alm_once_pallas(
            residual, ell, order, nside=nside, L_work=L_work, spin=spin
        )
    return alm


_map2alm_core_pallas_traced = jax.jit(
    _map2alm_core_pallas_eager,
    static_argnames=("nside", "L", "L_work", "n_iter", "spin"),
)


@partial(jax.jit, static_argnames=("split",))
def _pack_pallas_parts(low, high, ell, order, *, split):
    low_order = jnp.minimum(order, split - 1)
    high_order = jnp.clip(order - split, 0, high.shape[1] - 1)
    return jnp.where(
        order < split,
        low[ell, low_order],
        high[ell, high_order],
    )


def _pallas_parameters(L, nside):
    theta = _stable_thetas(L, nside)
    weights = quadrature_jax.quad_weights_transform(L, "healpix", nside)
    phase = healpix_ffts.p2phi_rings_jax(jnp.arange(len(theta)), nside)
    return theta, weights, phase


def _map2alm_once_pallas_multi_gpu(maps, ell, order, *, nside, L_work):
    primary, secondary = _gpu_devices()[:2]
    # A map already on the second GPU stays there: copying its high-m slice from the first
    # GPU would cost an extra ~8 GiB while the spin maps are still resident.
    home = secondary if getattr(maps, "device", None) == secondary else primary
    maps = _copy_to_device(maps, home)
    ftm = _forward_healpix_fft(maps[0], L=L_work, nside=nside, reality=True)
    positive = ftm[:, L_work:]
    split = _pallas_order_split(L_work)
    high_input = positive[:, split:]
    if home != secondary:
        high_input = _copy_to_device(high_input, secondary)
    theta, weights, phase = _pallas_parameters(L_work, nside)
    if home == secondary:
        theta = _copy_to_device(theta, secondary)
        weights = _copy_to_device(weights, secondary)
        phase = _copy_to_device(phase, secondary)
        other_theta, other_weights, other_phase = theta, weights, phase
    else:
        other_theta = _copy_to_device(theta, secondary)
        other_weights = _copy_to_device(weights, secondary)
        other_phase = _copy_to_device(phase, secondary)
    block_size = _pallas_block_size(nside)

    def low_transform(values):
        return _fused_forward_sht(
            values,
            theta,
            weights,
            -phase,
            L=L_work,
            block_size=block_size,
        )

    def high_transform(values):
        return _fused_forward_sht(
            values,
            other_theta,
            other_weights,
            -other_phase,
            L=L_work,
            block_size=block_size,
            m_start=split,
        )

    if home == secondary:
        low = _run_blocking(low_transform, positive[:, :split])
        high = _run_blocking(high_transform, high_input)
    else:
        with ThreadPoolExecutor(max_workers=2) as executor:
            low_future = executor.submit(
                _run_blocking, low_transform, positive[:, :split]
            )
            high_future = executor.submit(_run_blocking, high_transform, high_input)
            low, high = low_future.result(), high_future.result()
        high = _copy_to_device(high, primary)
    packed = _pack_pallas_parts(low, high, ell, order, split=split)[None, :]
    packed.block_until_ready()
    # The gather owns its buffer; drop the ring spectra before the next iteration.
    ftm = None
    positive = None
    low = None
    high = None
    high_input = None
    gc.collect()
    return packed


def _alm2map_core_pallas_multi_gpu(alm, *, nside, L, L_work):
    primary, secondary = _gpu_devices()[:2]
    home = secondary if getattr(alm, "device", None) == secondary else primary
    alm = _copy_to_device(alm, home)
    positive = _positive_alm(alm[0], L=L, L_work=L_work)
    split = _pallas_order_split(L_work)
    high_input = positive[:, split:]
    if home != secondary:
        high_input = _copy_to_device(high_input, secondary)
    theta = _stable_thetas(L_work, nside)
    phase = healpix_ffts.p2phi_rings_jax(jnp.arange(len(theta)), nside)
    if home == secondary:
        theta = _copy_to_device(theta, secondary)
        phase = _copy_to_device(phase, secondary)
        other_theta, other_phase = theta, phase
    else:
        other_theta = _copy_to_device(theta, secondary)
        other_phase = _copy_to_device(phase, secondary)
    block_size = _pallas_block_size(nside)

    def low_transform(values):
        return scalar_inverse_latitudinal(
            values,
            theta,
            phase=phase,
            L=L_work,
            block_size=block_size,
        )

    def high_transform(values):
        return scalar_inverse_latitudinal(
            values,
            other_theta,
            phase=other_phase,
            L=L_work,
            block_size=block_size,
            m_start=split,
        )

    if home == secondary:
        low = _run_blocking(low_transform, positive[:, :split])
        high = _run_blocking(high_transform, high_input)
    else:
        with ThreadPoolExecutor(max_workers=2) as executor:
            low_future = executor.submit(
                _run_blocking, low_transform, positive[:, :split]
            )
            high_future = executor.submit(_run_blocking, high_transform, high_input)
            low, high = low_future.result(), high_future.result()
        high = _copy_to_device(high, primary)
    ftm_positive = jnp.concatenate((low, high), axis=1)
    ftm_positive.block_until_ready()
    low = None
    high = None
    gc.collect()
    if nside >= 8192:
        cast = ftm_positive.astype(ring_dtype(L_work))
        cast.block_until_ready()
        ftm_positive = None
        gc.collect()
        out = _inverse_ring_fft_herm_chunked(cast, L=L_work, nside=nside)[None, :]
    else:
        out = jnp.real(_finish_inverse_pallas(ftm_positive, L=L_work, nside=nside))[None, :]
    out.block_until_ready()
    return out


def _map2alm_core_pallas_multi_gpu(
    maps, ell, order, *, nside, L, L_work, n_iter
):
    alm = _map2alm_once_pallas_multi_gpu(
        maps, ell, order, nside=nside, L_work=L_work
    )
    for _ in range(n_iter):
        synthed = _alm2map_core_pallas_multi_gpu(
            alm, nside=nside, L=L, L_work=L_work
        )
        residual = synthed - maps
        residual.block_until_ready()
        synthed = None
        alm = alm - _map2alm_once_pallas_multi_gpu(
            residual, ell, order, nside=nside, L_work=L_work
        )
        residual = None
    return alm


def _alm2map_core_multi_gpu(alm, ell, order, *, spin, nside, L, L_work):
    if spin == 0:
        elm = _unpack_real(alm[0], L, L_work)
        maps = _inverse_s2fft_multi_gpu(
            elm, L=L_work, spin=0, nside=nside, reality=True
        )
        return jnp.real(maps)[None, :]
    maps = _inverse_s2fft_multi_gpu(
        _unpack_spin(alm, L, L_work),
        L=L_work,
        spin=spin,
        nside=nside,
        reality=False,
    )
    return jnp.stack([jnp.real(maps), jnp.imag(maps)])


def _map2alm_once_multi_gpu(maps, ell, order, *, spin, nside, L, L_work):
    if spin == 0:
        flm = _forward_s2fft_multi_gpu(
            maps[0], L=L_work, spin=0, nside=nside, reality=True
        )
        return flm[ell, L_work - 1 + order][None, :]
    plus = _forward_s2fft_multi_gpu(
        maps[0] + 1j * maps[1],
        L=L_work,
        spin=spin,
        nside=nside,
        reality=False,
    )
    plus_m = plus[ell, L_work - 1 + order]
    minus_m = (-1) ** order * jnp.conj(plus[ell, L_work - 1 - order])
    alm = jnp.stack([-(plus_m + minus_m) / 2, 0.5j * (plus_m - minus_m)])
    # A slice would keep the full spectrum alive; at Nside 8192 it and the next pass's copy
    # (~16 GiB) do not fit together, so copy the packed alms into their own buffer.
    alm.block_until_ready()
    owned = jax.device_put(np.array(alm), alm.device)
    owned.block_until_ready()
    return owned


def _map2alm_core_multi_gpu(
    maps, ell, order, *, spin, nside, L, L_work, n_iter
):
    primary = _gpu_devices()[0]
    maps = _copy_to_device(maps, primary)
    alm = _map2alm_once_multi_gpu(
        maps, ell, order, spin=spin, nside=nside, L=L, L_work=L_work
    )
    # The spin march fills the second GPU; keep the packed alms and the residual on the
    # first so the next march starts on an empty card.
    if spin:
        alm = _copy_to_device(alm, primary)
    for _ in range(n_iter):
        synthed = _alm2map_core_multi_gpu(
            alm,
            ell,
            order,
            spin=spin,
            nside=nside,
            L=L,
            L_work=L_work,
        )
        if spin:
            synthed = _copy_to_device(synthed, primary)
        residual = synthed - maps
        synthed = None
        delta = _map2alm_once_multi_gpu(
            residual,
            ell,
            order,
            spin=spin,
            nside=nside,
            L=L,
            L_work=L_work,
        )
        if spin:
            delta = _copy_to_device(delta, primary)
        alm = alm - delta
        delta = None
    return alm


# ------------------------------------------------------------------------------------------------
# Entry points.  `gmaster.utils.map2alm` / `alm2map` / `map2alm_pair` validate their arguments and
# hand HEALPix geometries to these, which choose the route for the band limit and spin.
# ------------------------------------------------------------------------------------------------
def map2alm(maps, spin, map_info, alm_info, *, n_iter):
    """HEALPix analysis: `(nmaps, npix)` maps to Healpy-packed `(nmaps, nalm)` alms.

    Parameters
    ----------
    maps : array, shape (nmaps, npix)
        One map for spin 0, `(Q, U)` for spin > 0.
    spin : int
    map_info, alm_info
        HEALPix geometry (`nside`) and alm layout (`lmax`, packed `_ell`, `_m` indices).
    n_iter : int
        Number of Jacobi refinement iterations, as in NaMaster.

    Returns
    -------
    array, shape (nmaps, nalm)
        Alms (`E, B` for spin > 0) in Healpy (m-major) packing.
    """
    L = alm_info.lmax + 1
    # The Pallas route resolves m up to L-1 directly from the rings and is accurate for
    # L < 2*nside (NaMaster likewise only integrates m <= lmax), so it works at L_work = L.
    L_work = L
    if _use_pallas_sht(L_work, spin):
        if _use_multi_gpu_pallas(L_work, maps) and spin == 0:
            return _map2alm_core_pallas_multi_gpu(
                maps,
                alm_info._ell,
                alm_info._m,
                nside=map_info.nside,
                L=L,
                L_work=L_work,
                n_iter=int(n_iter),
            )
        return _map2alm_core_pallas(
            maps,
            alm_info._ell,
            alm_info._m,
            nside=map_info.nside,
            L=L,
            L_work=L_work,
            n_iter=int(n_iter),
            spin=int(spin),
        )
    # The other routes use s2fft's ring FFT, which concatenates the polar and equatorial ring
    # regions and requires L >= 2*nside, so lift the working band limit.
    L_work = max(L_work, 2 * map_info.nside)
    if spin != 0:
        analysis_slab, synthesis_slab = _spin_slabs(
            L_work, int(spin), nside=map_info.nside
        )
        if analysis_slab is not None:
            return _map2alm_core_slab(
                maps,
                alm_info._ell,
                alm_info._m,
                spin=int(spin),
                nside=map_info.nside,
                L=L,
                L_work=L_work,
                n_iter=int(n_iter),
                analysis_slab=analysis_slab,
                synthesis_slab=synthesis_slab,
            )
    if _use_multi_gpu_sht(L_work):
        return _map2alm_core_multi_gpu(
            maps,
            alm_info._ell,
            alm_info._m,
            spin=int(spin),
            nside=map_info.nside,
            L=L,
            L_work=L_work,
            n_iter=int(n_iter),
        )
    return _map2alm_core(
        maps,
        alm_info._ell,
        alm_info._m,
        spin=int(spin),
        nside=map_info.nside,
        L=L,
        L_work=L_work,
        n_iter=int(n_iter),
    )


def map2alm_pair(maps_a, maps_b, map_info, alm_info, *, n_iter):
    """Two spin-0 analyses that share one latitudinal pass, or None.

    The two transforms are independent, but the latitudinal step dominates each, so one
    pass can serve both: with a Legendre band the band is read once (`_shared_band_route`);
    without one (Nside 2048 and above) the folded march shares its recurrence
    (`_march_pair_route`).  A pair costs roughly 0.5-0.75 of two separate calls.

    The band route is bit-identical to two separate transforms.  The march route is not
    (the wider emit block reorders the theta sum) but is as accurate against the fp64 band
    as the single march.

    Parameters
    ----------
    maps_a, maps_b : array, shape (1, npix)
        Two spin-0 maps on the same geometry.
    map_info, alm_info
        HEALPix geometry and alm layout shared by both.
    n_iter : int
        Number of Jacobi refinement iterations for each map.

    Returns
    -------
    tuple of two arrays of shape (1, nalm), or None
        None when this geometry cannot share a pass; the caller then makes two
        ordinary `map2alm` calls.
    """

    L = alm_info.lmax + 1
    L_work = L
    if not _use_pallas_sht(L_work, 0) or _use_multi_gpu_pallas(L_work, maps_a):
        return None
    analysis, pair_synth = _shared_band_route(map_info.nside, L_work)
    march_pair = False
    if not analysis:
        march_pair = _march_pair_route(map_info.nside, L_work)
        # The v2 march also pairs the synthesis: the second map uses the accumulators the
        # spin-2 kernel reserves for its second helicity, so it is nearly free and bit-identical
        # to two launches.  This holds at every size, so the pair route is taken even when the
        # analyses cannot be paired.
        pair_synth = _spin_march._march_v2.enabled(L_work)
        analysis = march_pair or pair_synth
    if not analysis:
        return None
    return _map2alm_core_pallas_pair(
        maps_a, maps_b, alm_info._ell, alm_info._m,
        nside=map_info.nside, L=L, L_work=L_work, n_iter=int(n_iter),
        pair_synth=pair_synth, march_pair=march_pair,
    )


def alm2map(alm, spin, map_info, alm_info):
    """HEALPix synthesis: Healpy-packed `(nmaps, nalm)` alms to `(nmaps, npix)` maps.

    `alm` holds one set of alms for spin 0 or `(E, B)` for spin > 0; the output is one map
    or `(Q, U)`.  `map_info` and `alm_info` give the HEALPix geometry and alm layout.
    """
    L = alm_info.lmax + 1
    L_work = L
    if _use_pallas_sht(L_work, spin):
        return _alm2map_core_pallas(
            alm,
            nside=map_info.nside,
            L=L,
            L_work=L_work,
            spin=int(spin),
        )
    # The other routes use s2fft's ring FFT, which requires L >= 2*nside.
    L_work = max(L_work, 2 * map_info.nside)
    if spin != 0:
        _, synthesis_slab = _spin_slabs(L_work, int(spin), nside=map_info.nside)
        if synthesis_slab is not None:
            return _alm2map_core_slab(
                alm,
                spin=int(spin),
                nside=map_info.nside,
                L=L,
                L_work=L_work,
                slab=synthesis_slab,
            )
    if _use_multi_gpu_sht(L_work):
        alm = _copy_to_device(alm, _gpu_devices()[0])
        return _alm2map_core_multi_gpu(
            alm,
            alm_info._ell,
            alm_info._m,
            spin=int(spin),
            nside=map_info.nside,
            L=L,
            L_work=L_work,
        )
    return _alm2map_core(
        alm,
        alm_info._ell,
        alm_info._m,
        spin=int(spin),
        nside=map_info.nside,
        L=L,
        L_work=L_work,
    )
