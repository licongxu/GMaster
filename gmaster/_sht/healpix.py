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
    _scalar_spin_synthesis_latitudinal,
    _spin_forward_latitudinal,
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
    # ponytail: remove this perturbation when s2fft's Price-McEwen recurrence
    # handles exact-zero renormalisation without dropping subsequent modes.
    return theta + 8 * jnp.finfo(theta.dtype).eps


@partial(jax.jit, static_argnames=("L", "nside", "reality"))
def _forward_s2fft_ftm(maps, tables, *, L, nside, reality):
    m_start = L - 1 if reality else 0
    if reality:
        ftm = healpix_ffts.healpix_fft(maps, L, nside, "jax", reality)
    else:
        # One batched transform instead of s2fft's per-ring unroll: the
        # latter issues one tiny FFT per ring (1023 of them at Nside 256) and
        # runs launch-bound at ~4 GB/s.  Matches it to 1.6e-15.
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
        # March the Wigner-d row inside the kernel instead of building the 73.48 GiB slice that
        # `slabs_for` declines at Nside 1024, or the 13.2 s generic scatter loop that replaces it.
        # With a slice resident the table route wins and keeps the call; see
        # :func:`gmaster._spin_march_pallas.march_requested`.
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
    """Degree normalisation, the spin sign and the sub-spin zeroing, in one elementwise pass.

    Written as an einsum, a `where` and a multiply this was three separate passes over the
    `(L, 2L-1)` complex128 block -- 1.2 GiB at Nside 2048 spin 2, measured 72.7 ms of a 155 ms
    analysis pass, against 58 ms for the march that produced it
    (`.qwen/tmp/s2split_s37.py`).  The three factors are diagonal in `ell`, so they multiply into
    one `(L,)` vector and the whole finish is one read and one write.
    """
    m_start = L - 1 if reality else 0
    ell = jnp.arange(L)
    factor = (
        jnp.sqrt((2 * ell + 1) / (4 * jnp.pi))
        * jnp.where(ell < abs(spin), 0.0, 1.0)
        * (-1.0) ** abs(spin)
    )
    if reality:
        # The negative-order half is filled from the positive one, so the scaling has to land
        # before the mirror: keep the two steps, but each is still a single pass.
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
        # The same table-free row march as the analysis seam, in the synthesis direction (sum over
        # ell per theta lane): 122.7 ms at Nside 1024 against the 13.2 s generic loop the declined
        # slice falls back to.  It takes the same no-slice default as analysis but has its own flag,
        # `GMASTER_SPIN2_MARCH_SYNTH=0` to restore the exact route, because it is 0.86x against
        # ducc0 rather than ahead (105.7 ms) and its map is 1.968e-04 max / 6.370e-06 rms off ducc0
        # where the scatter loop is 1.095e-09 (`.qwen/tmp/synth_vs_ducc_1024.log`).
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

# The precomputed Legendre band is O(L^2 * nside) in the *storage* precision:
# 9.4 GiB at Nside 512 and 73.5 GiB at 1024 in float64, half that with
# `set_table_precision("fp32")` (36.7 GiB at Nside 1024). Anything above this falls
# back to the fused kernel rather than trading a faster theta stage for an OOM.
# Synthesis prefers a second, re-laid-out copy and takes it only while the pool has
# room (`_theta_matrix._synth_band`); with one resident band it reduces the analysis
# layout strided, so this single number gates both directions.
# 40 GiB is what makes the float32 Nside 1024 analysis band engage on this box's
# 71.2 GiB pool; the float64 Nside 1024 band (73.5 GiB) and both Nside 2048 sizes
# still decline. This gate only decides whether a build is attempted — the builders
# also decline on `RESOURCE_EXHAUSTED`, and `_spin_slice` checks live pool headroom.
_MATRIX_BAND_BUDGET = 40 * 1024**3

# Largest Wigner-d slab that may ride a fused analysis boundary.  The fusion is a 7.9x / 2.5x win at
# Nside 64 / 128 (`.qwen/tmp/slab_fuse2.log`) and 1.05x at 256, where the slab is 2.44 GiB; at Nside
# 512 it is 18.7 GiB and a boundary carrying both it and the maps is the layout-doubling that
# `_inverse_latitudinal_slab` exists to avoid, so the split route keeps the call there.
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
        # The v2 CUDA march beats the resident slab and needs none of its memory: at Nside 512
        # spin 2 the slab route is 41.5 ms per pass with a 43.9 GiB device peak, the march 2.7 ms.
        return None, None
    return _spin_slice.slabs_for(
        _stable_thetas(L_work, nside), L=L_work, spin=spin, nside=nside
    )


def _use_pallas_sht(L, spin):
    if nmt_params.sht_calculator not in (
        "jax", "jax-single", "jax-mgpu", "jax-dfp32", "jax-matrix"
    ):
        return False
    # No lower bound on L: interleaved against the generic s2fft path with the
    # clocks forced up, the fused scalar path is 1.8x faster at Nside 16 and
    # 4.5x at Nside 32, because the generic latitudinal step is a scatter loop
    # whose cost barely falls with map size. Alms agree to 2.9e-13.
    # Fused spin-weighted kernels remain experimental: their closed-form
    # Wigner-d seeds lose relative precision through catastrophic cancellation
    # once |m| approaches l. Spin transforms use the generic path until the
    # kernels adopt a renormalized sideways recursion (Turok-Bucher class).
    if spin != 0:
        return False
    return _on_cuda_gpu()


def _pallas_block_size(nside):
    # 512 lanes per program is the throughput sweet spot: larger tiles
    # under-utilize the SMs (the synthesis kernel degrades ~1.5-1.6x at
    # 1024 and cliffs hard at 2048), while smaller tiles add redundant
    # degree-loop work in analysis. 2*nside keeps the small-Nside case.
    return min(512, 2 * nside)


# Above this working bandlimit the refinement loop runs op-by-op instead of as
# one traced program.  768 is Nside 256: with the band hoisted out of the trace by
# `_trace_route_ready`, one program over the refinement loop is bit-identical
# (rel 0.000e+00, `.qwen/tmp/traceidentity_s32.py`) and 9.4 % faster end to end at
# that size -- 13.737 -> 12.442 ms, field 7.243 -> 6.378 ms
# (`.qwen/tmp/tracepipe_s32.log`).  It does not extend: the same arm at Nside 512
# (gate 1536) is 17 % *slower*, 68.407 -> 80.283 ms, with a 4.69 GiB band inside the
# program instead of the 0.61 GiB one it carries here.  Before the hoist the gate
# could not move at all -- the band build inside an outer trace raised
# `AttributeError: 'block_until_ready' is not available on traced array
# float32[160, 64, 512]` (`.qwen/tmp/s24_256_0.log`).
_PALLAS_TRACED_MAX_L = 768
# Tracing n_iter at L=3072 (Nside 1024) compiled for ~20 min on a Colab T4 and
# the warmed field was still minutes; the already-jitted per-transform march
# programs are the T4 route.  768 is Nside 256, where one program over the
# refinement loop was measured 9.4 % faster.
_MARCH_TRACED_MAX_L = int(os.environ.get("GMASTER_MARCH_TRACED_MAX_L", "768"))


def _trace_route_ready(nside, L_work):
    """Whether the single-program refinement route can see its tables at this size.

    The Legendre band is built outside a trace or not at all (`_theta_matrix._band`
    drains its build caches with `block_until_ready` and declines under one), so a
    geometry first touched inside a traced program silently takes the fused fp64
    on-the-fly kernel inside the very program that was traced to be fast.  The band
    is therefore built here, at top level, before the route is chosen -- so a
    program never carries a table build, which XLA would then re-run on every call.
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
    """Use the precomputed Legendre band instead of the kernel's recurrence.

    The band holds the same fp64 d^l_{m,0}(theta_j) values the fused kernel
    regenerates on every call, so the latitudinal stage becomes a memory-bound
    contraction. Measured against the kernel with the two interleaved and the
    clocks forced up: map2alm + alm2map is 1.12x faster at Nside 64, 1.23x at
    128, 1.33x at 256, with the alms agreeing to 1.8e-14. It is the default for
    the scalar transform; `jax` still means "band if it fits", and an
    insufficient budget or an m-split falls through to the kernel.
    """
    if m_start != 0:
        return False
    if nmt_params.sht_calculator not in ("jax", "jax-matrix"):
        return False
    if _spin_march._march_v2.enabled(L):
        # The v2 CUDA march beats the resident band at every size (2.8x at Nside 1024).
        return False
    return _theta_band_bytes(nside, L, dtype=table_dtype()) <= _MATRIX_BAND_BUDGET


def _fused_forward_sht(positive, theta, weights, phase, *, L, block_size,
                       m_start=0):
    """Forward latitudinal SHT, dispatched on the configured calculator.

    The double-fp32 (DFP32) kernel is analysis-only and slightly different
    in precision; every other calculator uses the fp64 Pallas kernel.
    The scalar path prefers the precomputed Legendre band while it fits (see
    `_prefer_theta_band`) and falls back to the kernel when it does not or
    cannot be materialized concretely, which is what happens on `jax.grad`
    paths.  Where the band is refused for *size* -- Nside 2048 and above at
    float32 -- the folded spin-0 march serves the call instead of the fp64
    on-the-fly kernel, because that kernel is the worst cell in the repo
    (0.33x at 2048, 0.27x at 4096); see
    :func:`gmaster._spin_march_pallas.fold_requested`.
    """
    nside = (len(theta) + 1) // 4
    # The folded march builds phases for m = 0..L-1. An m-split hands it only
    # one slice (Nside 8192 low half is 7198 columns, not 24576).
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

    HEALPix synthesis carries no quadrature weight of its own (the ring
    transform has it), so both routes get a unit weight vector.  The synthesis
    band prefers a copy re-laid out so the reduction runs over a contiguous axis,
    and `_theta_matrix._synth_band` makes that copy only while the pool has room
    for it; with one band resident it reduces the analysis layout strided instead.
    So the gate here is the same single-band test as the analysis path, and the
    doubling that used to be written here is what silently handed Nside 1024
    synthesis back to the fused kernel.
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
    # Not a jit boundary itself: the ring constants have to be fetched outside the trace
    # so they cross as arguments instead of being rebuilt (or baked) inside it.
    return _forward_ring_fft(
        maps, _ring_analysis_tables(L, nside, getattr(maps, "device", None)),
        L=L, nside=nside,
    )


def _finish_inverse_pallas(ftm_positive, *, L, nside):
    # The full polar-cap kernel at Nside 8192 is (16382, 65536) complex64,
    # exactly the 8 GiB cuFFT buffer that does not fit beside the spectrum.
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
        # The ring FFT's 16 GiB workspace does not fit on the card that already
        # holds the map. Build it on the other GPU and march there too.
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
        # Keep the spectrum on this GPU. Copying it back fills the other card,
        # and the alm gather then has no room for its 4.5 GiB temporary.
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
        # Spin synthesis ignores the theta half, so the north/south split is
        # two full marches. Keep the single march on the second GPU.
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
    """The ring chirp-Z constants a polarised core needs, fetched *outside* its trace.

    `_map2alm_once`, `_alm2map_core` and `_map2alm_iteration` used to be the jit boundaries
    themselves and called the non-boundary `_forward_s2fft`/`_inverse_s2fft` inside their trace;
    those build the tables under `ensure_compile_time_eval`, so the concrete arrays were baked
    into the outer program as HLO constants -- copied to the host at lowering
    (`_array_mlir_constant_handler`) and re-embedded per executable.  At Nside 4096 the analysis
    kernel table alone is `(8190, 65536)` complex64 = 4.00 GiB and the polarised `map2alm` died
    with `RESOURCE_EXHAUSTED ... 4.00GiB` before any transform ran
    (`.qwen/tmp/validate_s36c.log`, session 36).  Spin 0 passes `()` and never saw it.
    """
    if spin == 0:
        return ()
    device = getattr(like, "device", None)
    if synthesis:
        return _spin_ring_synthesis_tables(L_work, nside, device)
    return _spin_ring_analysis_tables(L_work, nside, device)


def _dc_spin(L_work, spin):
    """The divide-and-conquer engine for a polarised transform at this bandlimit, if it serves it.

    Its spin-s calls are composed eagerly here (ring FFT, latitudinal step, finish -- each its own
    program) so that the plan's arrays enter as jit arguments; traced inside the fused programs
    below they would be captured as constants (~20 GiB of HLO at Nside 4096).
    """
    return _spin_march._march_v2._dc(L_work) if spin != 0 else None


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


# The staged march at Nside 8192 carries its spectra in complex64 between programs.  The v2
# kernels read and write float32, so the values the kernel sees and emits are the same; complex128
# made the synthesis program 18 GiB in + 24 GiB out + 32 GiB temporaries, which with the maps and
# alms resident was past the 96 GB card at the first refinement iteration.
@partial(jax.jit, static_argnums=(1, 2))
def _prepare_spin_c64(alm, L, L_work):
    return _prepare_inverse_s2fft(_unpack_spin(alm, L, L_work), L=L_work).astype(jnp.complex64)


@partial(jax.jit, static_argnames=("L", "nside"))
def _forward_s2fft_ftm_c64(maps, tables, *, L, nside):
    return _forward_s2fft_ftm(maps[0] + 1j * maps[1], tables, L=L, nside=nside,
                              reality=False).astype(jnp.complex64)


@partial(jax.jit, static_argnames=("L", "nside"))
def _residual_ftm_c64(synth, maps, tables, *, L, nside):
    """Ring spectrum of `synth - maps` with the difference formed inside the program: held as
    its own array it was a third 12 GiB map beside the input and the march's temporaries."""
    return _forward_s2fft_ftm_c64(synth - maps, tables, L=L, nside=nside)


def _march_c64_analysis(ftm_box, ell, order, *, L_work, spin, nside):
    """March analysis of the complex64 ring spectrum in `ftm_box` (a one-element list, emptied so
    the spectrum is freed as soon as the march has read it)."""
    flm = _spin_march._march_v2.forward_latitudinal(ftm_box.pop(), L=L_work, spin=spin,
                                                    nside=nside, cdtype=jnp.complex64)
    return _finish_pack_spin(flm, ell, order, L_work=L_work, spin=spin)


@partial(jax.jit, static_argnames=("L_work", "spin"))
def _finish_pack_spin(flm, ell, order, *, L_work, spin):
    """`_spin_pack_plus(_finish_forward_s2fft(flm))` with no `(L, 2L-1)` block between them.

    The finish factor is diagonal in `ell`, so it is applied to the gathered entries instead.
    """
    factor = (jnp.sqrt((2 * ell + 1) / (4 * jnp.pi)) * jnp.where(ell < abs(spin), 0.0, 1.0)
              * (-1.0) ** abs(spin))
    plus_m = flm[ell, L_work - 1 + order].astype(jnp.complex128) * factor
    minus_m = (-1) ** order * jnp.conj(flm[ell, L_work - 1 - order].astype(jnp.complex128) * factor)
    return jnp.stack([-(plus_m + minus_m) / 2, 0.5j * (plus_m - minus_m)])


@partial(jax.jit, static_argnames=("L", "spin", "nside"))
def _finish_inverse_c64(ftm, tables, *, L, spin, nside):
    """`_finish_inverse_s2fft` on a complex64 march output: the ring phase is applied in complex128
    as before, and the ring transform takes its complex64 input as it always did."""
    return _finish_inverse_s2fft(ftm.astype(jnp.complex128), tables, L=L, spin=spin, nside=nside,
                                 reality=False)


def _march_c64_ready(L_work, spin):
    return _spin_march._march_v2.enabled(L_work) and _spin_march.march_requested(
        spin, L=L_work, nside=None) and _spin_march.synth_requested(spin, L=L_work, nside=None)


def _staged_spin(L_work, spin):
    """The latitudinal engine a polarised transform runs as separate programs, if any.

    The D&C engine where it serves; above `_RING_FACTORS_KEPT_MAX_L` (Nside 8192) also the march,
    whose fused synthesis program needed one 72 GiB temporary there -- split, each program's
    working set is one stage's.
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


# Above this ring-spectrum size one refinement iteration runs as its two programs (synthesis,
# then analysis of the residual) instead of one: XLA hands a program a single temporary buffer,
# and the fused iteration's was 26 GiB at Nside 4096 spin 2 (`MaxAllocSize` in
# `.qwen/tmp/chain_s36s2.log`), the largest allocation of the whole pipeline.
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
    """The slice contraction on its own: see :func:`_inverse_latitudinal_slab`."""
    return _spin_slice.forward_latitudinal(ftm, slab, L=L)


@partial(jax.jit, static_argnames=("L",))
def _inverse_latitudinal_slab(flm, slab, *, L):
    """The slice contraction, and nothing else.

    A block set is tens of GiB from Nside 512 up, and a ``jax.jit`` boundary whose
    argument list carries both layouts makes XLA count them twice against the pool
    (``The byte size of input/output arguments ... exceeds the base limit``) and
    schedule multi-gibibyte copies of them on every call.  So only the latitudinal
    step is jitted here; the s2fft stages around it are each already jitted
    individually and never see the table.
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
    """Polarised analysis with the latitudinal step replaced by a slab contraction.

    The ring FFT, the contraction, the epilogue and the E/B combination are one program while the slab
    is small enough to ride the boundary.  Run as three jits with an eager gather afterwards the call
    cost 1.700 / 1.944 ms at Nside 64 / 128 whatever the map size, because `plus[ell, L_work-1+order]`
    and its parity conjugate are dispatched op by op; fused it is 0.215 and 0.765 ms (7.91x, 2.54x)
    with bit-identical output (`.qwen/tmp/slab_fuse2.log`).  Above the gate fusion is not merely
    impossible, it is unwanted: forcing it on an 18.74 GiB slab at Nside 512 costs 34.414 -> 38.175 ms
    per analysis call and 5.075 -> 5.420 ms on the 2.44 GiB slab at Nside 256, with the result unchanged
    to every printed digit (`.qwen/tmp/s35_slabgate.log`).  The gate therefore keeps the split path for
    throughput, not because the boundary would refuse.
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


# Largest working bandlimit whose whole polarised refinement loop is traced into one program.
# 1536 is Nside 512, and the slab route does not reach past it anyway (`_spin_slabs` returns no
# pair at 1024), so the cap is the measured range rather than a boundary the trace would refuse.
# With the default float64 tables map2alm goes 36.949 -> 34.479 ms at Nside 256 and 239.320 ->
# 225.025 ms at 512; with `set_table_precision("fp32")` 8.966 -> 7.900 ms at 256 and 60.955 ->
# 52.372 ms at 512, every arm bit-identical to the eager loop (`max|d|=0.000e+00`).  End to end
# that is 56 -> 52 ms at Nside 256 and 340 -> 333 ms at 512 spin 2, with `dCl` unchanged to every
# printed digit and `bytes_in_use` unchanged too (57.5 -> 57.6 GiB, the slabs, not this program).
# Nside 128 with float64 tables is the one core-call reversal, 5.829/5.830/5.835 ->
# 5.999/5.996/5.974 ms (1.03x, three repeats), and nine-repeat harness runs put both arms at
# 10-11 ms, so it is left inside a single upper gate rather than carved out (`.qwen/tmp/
# s35_b3_fp64.log`, `.qwen/tmp/s35_b3_repeat.log`, `.qwen/tmp/s35_b3_256.log`,
# `.qwen/tmp/s35_b3_512.log`, `.qwen/tmp/s35_b3_pipe_before.log`, `.qwen/tmp/s35_b3_pipe_after.log`,
# `.qwen/tmp/s35_b3_n128ab.log`).
_SPIN_SLAB_TRACED_MAX_L = 1536


def _spin_slab_trace_ready(maps, L_work):
    """Whether the refinement loop may become one program at this size.

    The loop keeps its eager form under an outer trace: one graph over `n_iter` refinement passes
    keeps every iteration's residual alive for the transpose, which is a memory cost this route has
    never been measured against, and the repository's AD tests reach only the scalar route.  A
    gradient of a polarised analysis therefore continues to see the boundaries it had before.
    """
    if isinstance(maps, jax.core.Tracer):
        return False
    return L_work <= _SPIN_SLAB_TRACED_MAX_L


@partial(jax.jit, static_argnames=("spin", "nside", "L", "L_work", "n_iter"))
def _map2alm_core_slab_traced(maps, ell, order, analysis_tables, synthesis_tables,
                              analysis_slab, synthesis_slab, *, spin, nside, L,
                              L_work, n_iter):
    """Polarised analysis with the refinement loop inside one XLA program.

    The tables ride in as arguments because a traced program may read them but must never build
    them (`_theta_matrix`'s rule, and the reason `_trace_route_ready` warms the scalar band
    first): a build inside this program would be re-run by XLA on every call.
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
    """Spin-s refinement in the D&C engine's node space (see `_dc_lat`): no tree inside the loop."""
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
        # Two-helicity synthesis: a+ = E + iB drives the mp=+s ladder of
        # f+ = Q - iU; a- = E - iB drives the mp=-s ladder of f- = Q + iU.
        e_alms, b_alms = jnp.asarray(alm[0]), jnp.asarray(alm[1])
        ell_idx, m_idx = _ell_order_arrays(L - 1)
        a_plus = jnp.zeros((L_work, 2 * L_work - 1), dtype=jnp.complex128)
        a_minus = jnp.zeros((L_work, 2 * L_work - 1), dtype=jnp.complex128)
        plus_vals = e_alms + 1j * b_alms
        minus_vals = e_alms - 1j * b_alms
        parity = (-1.0) ** m_idx
        a_plus = a_plus.at[ell_idx, L_work - 1 + m_idx].set(plus_vals)
        a_plus = a_plus.at[ell_idx, L_work - 1 - m_idx].set(parity * jnp.conj(plus_vals))
        a_minus = a_minus.at[ell_idx, L_work - 1 + m_idx].set(minus_vals)
        a_minus = a_minus.at[ell_idx, L_work - 1 - m_idx].set(parity * jnp.conj(minus_vals))
        norm_l = jnp.sqrt((2 * jnp.arange(L) + 1) / (4 * jnp.pi))
        a_plus = a_plus * norm_l[:, None]
        a_minus = a_minus * norm_l[:, None]
        theta = _stable_thetas(L_work, nside)
        block = min(256, 2 * nside)
        f_plus = _scalar_spin_synthesis_latitudinal(
            jnp.asarray(a_plus).T, theta, L=L_work, spin=int(spin), block_size=block
        )
        f_minus = _scalar_spin_synthesis_latitudinal(
            jnp.asarray(a_minus).T, theta, L=L_work, spin=-int(spin), block_size=block
        )
        shifts = healpix_ffts.ring_phase_shifts_hp_jax(L_work, nside, False, False)
        synth_tables = _spin_ring_synthesis_tables(
            L_work, nside, getattr(f_plus, "device", None))

        def to_map(centered):
            full = jnp.concatenate(
                (jnp.zeros((centered.shape[0], 1), dtype=centered.dtype), centered),
                axis=1,
            )
            full = full.at[:, 1:].multiply(shifts)
            return _inverse_ring_fft_complex(full[:, 1:], synth_tables,
                                             L=L_work, nside=nside)
        q_map = 0.5 * (to_map(f_plus) + jnp.conj(to_map(f_minus)))
        u_map = -0.5j * (to_map(f_plus) - jnp.conj(to_map(f_minus)))
        return jnp.stack([q_map, u_map])[None, :]
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
    """Which stages of two same-geometry spin-0 transforms may share one band.

    Returns ``(analysis, synthesis)``.  The MASTER pipeline runs exactly such a
    pair -- a field's `n_iter` Richardson passes and, inside
    `compute_coupling_matrix`, its mask's `n_iter_mask` passes, both spin 0 at
    `lmax_mask == lmax` -- and at Nside 256 that pair is 11.918 ms of a 12.442 ms
    pipeline (`.qwen/tmp/tracepipe_s32.log`).  Both passes stream the same
    resident band, so one program can serve both: measured 0.55x/0.56x/0.53x of
    two separate calls at Nside 256/512/1024 for the analysis and 0.68x/0.63x for
    the synthesis, outputs bit-identical (`.qwen/tmp/pairsettle_s33.log`,
    `.qwen/tmp/pairsettle_s33_big.log`).

    The gate is the band's own: no march may be serving these geometries (a
    forced `GMASTER_SPIN0_MARCH=1` would otherwise silently get the band here and
    the march on the single route), the band must fit, and it must be concrete --
    so `warm` builds it here, at top level, rather than inside a trace.  The
    synthesis half answers separately because the contiguous re-layout is what
    makes a second right-hand side cheap; with the copy refused the pair costs
    4.19x instead (`_theta_matrix.inverse_latitudinal_pair`).
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
    """True when the folded spin-0 march can serve a pair of same-geometry analyses.

    This is the pair route for the sizes where no band exists -- Nside 2048 and above.  A marched
    row's cost is the Legendre recurrence, which depends on the `(m, theta)` triple alone; the map
    enters only in the emit, where four channels already fold through one Triton reduction tree.
    Two maps therefore share the recurrence, though less completely than the band pairing does,
    because the emit's fp64 partials and rhs also double: measured against two separate calls,
    one paired latitudinal step costs **0.489** at Nside 512 (13.10 ms against 26.81 ms, where a
    single march is 13.90 ms), 0.504 at 1024 (55.0 against 109.2 ms) and **0.752** at 2048 (435.0
    against 578.4 ms), the trend being the store and rhs volume growing into the recurrence
    (`.qwen/tmp/marchpair_512.log`, `.qwen/tmp/marchpair_1024.log`, `.qwen/tmp/marchpair_2048.log`).
    The pairing cannot be bit-identical -- widening the emit block from 4 to 8 channels changes the
    reduction tree, which shows up as 1.2-1.3e-07 relative against the single march -- so it is
    checked against the fp64 band instead: at Nside 512 `|paired - band| = 2.03e-04` against
    `|single - band| = 2.03e-04`, max abs 8.406e-08 against 8.405e-08, i.e. the paired route is
    exactly as close to the accurate contraction as the march it replaces.  The synthesis half is
    not paired here: the marched synthesis kernel keeps per-lane accumulators and per-degree
    coefficient loads, both of which scale with the number of maps, so it has no shared reduction
    tree to widen and a second map would double the part that costs.

    The last clause is a memory limit, not a speed one.  At Nside 4096 the paired program dies in
    the allocator -- `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 8.15GiB` inside the
    71.2 GiB pool -- where the same geometry run as two separate calls completes in 33.62 s against
    NaMaster's 74.11 s (`rel dCl` 1.42e-06) with a 16.3 GiB GPU peak
    (`.qwen/tmp/pairrun_4096_s34.log`).  `fold_pair_fits` refuses it, and the two calls the caller
    then makes are the arm that works.
    """
    return (_spin_march.fold_requested(nside, L_work)
            and _spin_march.fold_pair_fits(nside, L_work))


def _map2alm_pair_once_pallas(maps_a, maps_b, ell, order, *, nside, L_work, march_pair,
                              return_ftm=False):
    """Two scalar analyses, one ring FFT per map and one sweep of the row source.

    The azimuthal stage is per-map work and is done twice; only the latitudinal contraction is
    shared, which is where the bytes (band) or the recurrence (march) are.  ``return_ftm`` also
    hands back the two ring spectra, which the folded refinement loop reuses.
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
        # The divide-and-conquer engine pairs by sharing its plan traversal; it has no march-style
        # memory gate, so both maps always go through one call.
        positive_a, positive_b = _spin_march._march_v2.forward_latitudinal_positive_pair(
            ftm_a, ftm_b, weights, phase, L=L_work, nside=nside)
    elif march_pair:
        positive_a, positive_b = _spin_march.forward_latitudinal_positive_pair(
            ftm_a, ftm_b, weights, phase, L=L_work, nside=nside)
    elif _spin_march._march_v2.enabled(L_work) and not _prefer_theta_band(nside, L_work, 0):
        # The pair route was taken for the *synthesis* alone: above `_march_v2._PAIR_MAX_L` the
        # paired analysis kernel loses (it needs the half-size theta tile, so it writes four times
        # a single launch's tile partials), while the paired synthesis is free and bit-identical.
        # The two analyses then run as two ordinary marched calls.
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


# The Richardson residual `FFT(IFFT(F) - map)` of a HEALPix ring is the n_phi-periodic fold of F
# minus the map's own spectrum (`_march_v2.ring_fold_residual`), so the refinement loop keeps the
# spectra from its first analysis and never runs a ring FFT again.  `GMASTER_RING_FOLD=0` restores
# the synthesise-to-pixels form.
_RING_FOLD = os.environ.get("GMASTER_RING_FOLD", "1") != "0"
# Refinement in the D&C engine's node space (V^T V = I: the trees cancel between iterations).
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
        # Every caller hands this to `ring_fold_residual`, which reads complex64: emitted as such
        # it is the same numbers without the complex128 copy (12 GiB at Nside 8192).
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
    """Two Richardson recursions stepping in lockstep over one row source.

    The recursions are independent -- each refines its own alms against its own
    map -- so they can share a pass even though neither can be batched internally.
    `pair_synth` and `march_pair` are static because which source serves this
    geometry is decided before any trace, and a traced program must not branch on
    a table cache; the two also select different kernels, so neither can become a
    runtime value.
    """
    dc = _spin_march._march_v2._dc(L_work)
    if n_iter and dc is not None and L == L_work and _ring_fold_ready(L_work):
        # The divide-and-conquer engine serves both latitudinal stages: the refinement runs in its
        # packed alm layout (fp64 accumulation) and converts to the output packing once.
        ftms = [_forward_ring_fft_positive(
                    m[0],
                    _ring_analysis_tables(
                        L_work, nside, getattr(m, "device", None) or getattr(m[0], "device", None)),
                    L=L_work, nside=nside)
                for m in (maps_a, maps_b)]
        weights = quadrature_jax.quad_weights_transform(L_work, "healpix", nside)
        phi = healpix_ffts.p2phi_rings_jax(jnp.arange(4 * nside - 1), nside)
        if _NODE_SPACE:
            # Node-space refinement (`_dc_lat` notes): the trees cancel between iterations.
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
    """Paired analysis, traced whole below `_PALLAS_TRACED_MAX_L`.

    Same condition as the single route, for the same reason: tracing removes host
    dispatch, which pays up to `_PALLAS_TRACED_MAX_L` and costs 17 % above it.
    The march pairing only ever serves sizes far above that gate, so in practice it
    runs op-by-op and the flag is there to keep one code path.
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


def _map2alm_once_pallas_spin(maps, ell, order, *, nside, L_work, spin):
    """Two-helicity spin-2 analysis.

    a+_lm = sum_rings w e^{-im phi} (Q - iU)_ring d^l_{m,+s}
    a-_lm = sum_rings w e^{-im phi} (Q + iU)_ring d^l_{m,-s}
    E = (a+ + a-)/2, B = (a+ - a-)/(2i), packed in Healpy ordering with the
    sqrt((2l+1)/4pi) normalization applied per degree.
    """
    theta = _stable_thetas(L_work, nside)
    weights = quadrature_jax.quad_weights_transform(L_work, "healpix", nside)
    phase = -healpix_ffts.p2phi_rings_jax(jnp.arange(len(theta)), nside)
    m_centered = jnp.arange(-(L_work - 1), L_work)
    window = weights[:, None] * jnp.exp(1j * (phase[:, None] * m_centered[None, :]))
    block = min(256, 2 * nside)

    signal_plus = maps[0] - 1j * maps[1]   # helicity +s
    signal_minus = maps[0] + 1j * maps[1]  # helicity -s
    analysis_tables = _spin_ring_analysis_tables(
        L_work, nside,
        getattr(maps, "device", None) or getattr(signal_plus, "device", None))
    rings_p = _forward_ring_fft_full(signal_plus, analysis_tables,
                                     L=L_work, nside=nside)
    rings_m = _forward_ring_fft_full(signal_minus, analysis_tables,
                                     L=L_work, nside=nside)

    a_plus_grid = _spin_forward_latitudinal(
        rings_p * window, theta, L=L_work, spin=int(spin), block_size=block
    )
    a_minus_grid = _spin_forward_latitudinal(
        rings_m * window, theta, L=L_work, spin=-int(spin), block_size=block
    )

    norm_l = jnp.sqrt((2 * jnp.arange(L_work) + 1) / (4 * jnp.pi))
    a_plus = a_plus_grid.T * norm_l[:, None]
    a_minus = a_minus_grid.T * norm_l[:, None]

    plus_col = a_plus[ell, L_work - 1 + order]
    minus_col = a_minus[ell, L_work - 1 + order]
    e_vals = 0.5 * (plus_col + minus_col)
    b_vals = (plus_col - minus_col) / (2j)
    return jnp.stack([e_vals, b_vals])


def _map2alm_once_pallas(maps, ell, order, *, nside, L_work, spin=0):
    if spin != 0:
        return _map2alm_once_pallas_spin(
            maps, ell, order, nside=nside, L_work=L_work, spin=spin
        )
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
    """Analytic pseudo-Cl analysis, traced whole or run op-by-op.

    Tracing the refinement loop as one program removes the per-primitive host dispatch, which at small
    bandlimit is the whole cost: at Nside 128 the op-by-op route takes 1.058 ms for a 196608-pixel map
    whose device work is 0.333 ms, so one program over the same body is 3.17x faster with bit-identical
    output -- 1.96x on the inverse, and 2.18x / 1.77x with two refinement iterations
    (`.qwen/tmp/spin0_traced.log`, `.qwen/tmp/spin0_traced_it2.log`).  The gate used to stop at Nside 128
    because the Legendre band cannot be built inside a trace; now that `_trace_route_ready` builds it
    first, Nside 256 takes the single program too (9.4 % on the pipeline, bit-identical alms) and Nside
    512 is measurably worse without it -- see `_PALLAS_TRACED_MAX_L`.
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
        # One map in the D&C engine's node space (the paired route's loop with one right-hand side).
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
    # A mask placed on the second GPU stays there. Copying its high-m slice
    # off the first GPU is an extra 8 GiB while the spin maps are still resident.
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
    # The gather owns its buffer. Drop the ring FFT before the next iteration.
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
    # A slice keeps the full ring spectrum alive. At Nside 8192 that spectrum
    # plus the next pass's copy is the 16 GiB alloc that does not fit.
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
    # The spin march fills the second GPU. Keep the packed spectrum and the
    # residual on the first so the next march starts on an empty card.
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
    """HEALPix analysis: `(nmaps, npix)` maps to Healpy-packed `(nmaps, nalm)` alms."""
    L = alm_info.lmax + 1
    # The Pallas latitudinal SHT resolves m up to L-1 from the HEALPix rings
    # directly and is accurate for L < 2*nside (NaMaster likewise only
    # integrates m <= lmax), so let the working order track L there.  The
    # generic s2fft reference ring FFT cannot concatenate its ring regions
    # when L < 2*nside, so it still needs the working order lifted to 2*nside.
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
    # Non-Pallas paths use the s2fft reference ring FFT, whose JAX kernel
    # concatenates the polar/equatorial/south ring regions and only works for
    # L >= 2*nside. Lift the working order so those paths stay valid; the
    # Pallas path above already returned with L_work == L.
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

    `map_a` and `map_b` are both ``(1, npix)`` and are analysed against the same
    `alm_info` with the same `n_iter`.  The two transforms are independent; the
    only reason to run them together is that the latitudinal step dominates either
    one, so one pass can serve both.  Where a Legendre band exists it is the largest
    thing either transform touches and pairing it reads it once (0.53-0.68x of two
    calls, outputs bit-identical, `_shared_band_route`).  Where none exists -- Nside
    2048 and above -- the folded march generates its row inside the kernel, so the
    two maps share the recurrence itself: 0.752 of two calls there, 0.489 at
    Nside 512 (`_march_pair_route`).  Returns `(alm_a, alm_b)`, or None when this
    geometry cannot share -- the caller then makes two ordinary `map2alm` calls,
    which is exactly what a None here preserves.

    Both halves come out eagerly, so a caller that needs only one of them pays for
    both; that second transform costs about a tenth of the first here against the
    full price of a separate call, and it is the trade `NmtField` makes on behalf
    of the pipeline that always needs both.  The band route is bit-identical to two
    separate transforms; the march route is not, because widening the emit block
    reassociates the theta sum, and its error against the fp64 band is measured
    unchanged from the shipped march's.
    """

    L = alm_info.lmax + 1
    L_work = L
    if not _use_pallas_sht(L_work, 0) or _use_multi_gpu_pallas(L_work, maps_a):
        return None
    analysis, pair_synth = _shared_band_route(map_info.nside, L_work)
    march_pair = False
    if not analysis:
        march_pair = _march_pair_route(map_info.nside, L_work)
        # The v2 march pairs the synthesis too: its spin-0 kernel leaves the accumulators the
        # spin-2 kernel uses for its second helicity idle, so the second map is free of registers
        # and the paired launch is bit-identical to two separate ones (1.25x at Nside 1024).  That
        # holds at every size, so above the paired *analysis* limit the route is still worth
        # taking with the analyses unpaired -- a field and its mask then share one refinement.
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
    """HEALPix synthesis: Healpy-packed `(nmaps, nalm)` alms to `(nmaps, npix)` maps."""
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
    # Non-Pallas paths use the s2fft reference ring FFT, which requires
    # L >= 2*nside; the Pallas path above already returned with L_work == L.
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
