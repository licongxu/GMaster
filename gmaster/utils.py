"""Map metadata and JAX spherical-harmonic transforms."""

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial

import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer
from s2fft.sampling import s2_samples
from s2fft.recursions import turok_jax
from s2fft.transforms import _ftm_flm_primitive
from s2fft.utils import healpix_ffts, quadrature_jax

from ._cuda_gpu import on_cuda_gpu as _on_cuda_gpu
from ._sht_pallas import (
    scalar_forward_latitudinal,
    scalar_inverse_latitudinal,
    _scalar_spin_synthesis_latitudinal,
    _spin_forward_latitudinal,
)
from ._sht_dfp32 import scalar_forward_latitudinal_dfp32
from ._theta_matrix import band_bytes as _theta_band_bytes
from ._theta_matrix import inverse_latitudinal as _theta_matrix_inverse_latitudinal
from ._theta_matrix import (
    inverse_latitudinal_pair as _theta_matrix_inverse_latitudinal_pair,
)
from ._theta_matrix import positive_latitudinal as _theta_matrix_latitudinal
from ._theta_matrix import positive_latitudinal_pair as _theta_matrix_latitudinal_pair
from ._theta_matrix import synth_pair_ready as _theta_matrix_synth_pair_ready
from . import _spin_march_pallas as _spin_march
from . import _spin_slice


class NmtParams:
    def __init__(self):
        self.sht_calculator = "jax"
        self.n_iter_default = 3
        self.n_iter_mask_default = 3
        self.tol_pinv_default = 1e-10
        # Storage precision of the precomputed transform tables.  Every
        # contraction accumulates in float64 whatever this is.
        self.table_dtype = "fp64"
        # Element type of the azimuthal transforms.  "auto" is complex64 where the v2
        # march serves the latitudinal stage and complex128 below it; "follow" keeps the
        # historical coupling to the table precision; "fp64"/"fp32" pin it independently.
        self.ring_precision = "auto"


nmt_params = NmtParams()

_TABLE_DTYPES = {"fp64": jnp.float64, "fp32": jnp.float32}
_RING_DTYPES = {"follow": None, "auto": None, "fp64": jnp.complex128,
                "fp32": jnp.complex64}


def table_dtype():
    """jnp dtype the precomputed transform tables are stored in."""
    return _TABLE_DTYPES[nmt_params.table_dtype]


def ring_dtype(L=None):
    """Element type of the azimuthal (ring) transforms.

    The ring stage is a batched FFT, and on this card a double-precision FFT is
    compute-bound at roughly the fp64 FMA rate while the fp32 one runs on the tensor-core
    path: the same length transform is ~4.5x cheaper in `complex64` (`.qwen/tmp/ring_fp32_ab.py`:
    0.91 -> 0.21 ms at Nside 512, 4.08 -> 0.83 ms at 1024).

    The shipped default is ``"auto"``: complex64 exactly where the v2 march serves the
    latitudinal stage, complex128 below it.  ``"follow"`` tracks `set_table_precision`, which
    is what every pre-v2 published number used.  The cast is not
    confined to the chirp tables: `_forward_ring_fft_positive` casts the *pixels* to the
    chirp's real dtype too, so ``"follow"`` with fp32 tables analyzes the map itself in
    float32.  ``set_ring_precision("fp64")`` keeps the azimuthal stage exact under fp32
    tables; ``"fp32"`` buys the ~4.5x regardless of the table choice.
    """
    forced = _RING_DTYPES[nmt_params.ring_precision]
    if forced is not None:
        return forced
    if nmt_params.ring_precision == "auto" and L is not None:
        # The shipped default: complex64 exactly where the v2 march serves the latitudinal stage
        # (every band limit with a CUDA build).  There the pass already carries the march's
        # float32-class 1e-6, the ring stage is 55 % of it (10.4 of 18.7 ms at Nside 1024 spin 2)
        # and complex64 halves that; `GMASTER_MARCH_V2=0` keeps the exact transform.
        from . import _march_v2

        if _march_v2.enabled(L):
            return jnp.complex64
    return jnp.complex64 if table_dtype() == jnp.float32 else jnp.complex128


def set_ring_precision(name):
    """Choose the element type of the azimuthal transforms independently of the tables.

    The two precisions are bought for different reasons: halved *table* bytes change which
    theta route a geometry dispatches to, while the *ring* precision changes the FFT that
    the map pixels run in.  Following the tables couples them; this breaks the coupling so
    an fp32 table route can keep an exact azimuthal transform.

    Clears the ring table caches for the same reason `set_table_precision` does.
    """
    if name not in _RING_DTYPES:
        raise KeyError(
            "GMaster ring precision must be 'auto', 'follow', 'fp64' or 'fp32'")
    if name == nmt_params.ring_precision:
        return
    nmt_params.ring_precision = name
    drop_ring_tables()


def drop_ring_tables():
    """Evict the cached ring chirp-Z tables (they are rebuilt on the next transform)."""
    _ring_analysis_tables.cache_clear()
    _ring_synthesis_tables.cache_clear()
    _spin_ring_analysis_tables.cache_clear()
    _spin_ring_synthesis_tables.cache_clear()


_ROOM_HOOKS = []       # extra callables that free device caches (registered by workspaces)
_ROOM_HOOKS_LAST = []  # freed only when the steps above did not make room (v2 march tables)


def make_room(nbytes):
    """Drop the ring-table caches when the device pool cannot hold `nbytes` more.

    The polarised ring tables are 30 GiB of complex128 at Nside 4096 and live in `lru_cache`s
    for the process; the coupling matrix never uses them, and at Nside 4096 spin 2 its assembly
    (two `(ncls (lmax+1))^2` float64 copies, 18 GiB each) failed with them resident at 50.6 GiB
    in use (`.qwen/tmp/chain_s36j.log`, session 36).  A device without allocator statistics
    reports nothing and nothing is dropped.
    """
    def short():
        stats = jax.devices()[0].memory_stats() or {}
        limit, in_use = stats.get("bytes_limit"), stats.get("bytes_in_use")
        return bool(limit) and in_use is not None and limit - in_use < nbytes

    if not short():
        return
    # Cheapest first: the Wigner-d quadrature cache (9 GiB at Nside 4096, seconds to rebuild),
    # then the ring tables (30 GiB, which the next transform rebuilds), and only then the v2
    # march's window tables.  Those are last because dropping them costs the most: the Nside 4096
    # spin-0 `NmtField` is 2.00 s with them resident and 3.26 s without
    # (`.qwen/tmp/field_s37.py`), and the benchmark's repeated field builds were freeing them on
    # every call, which is the whole difference between that stage measuring 2.0 s and 2.6 s.
    for hook in _ROOM_HOOKS:
        hook()
    if not short():
        return
    drop_ring_tables()
    if not short():
        return
    for hook in _ROOM_HOOKS_LAST:
        hook()


def set_table_precision(name):
    """Choose the device storage precision of the precomputed tables.

    `"fp64"` (default) is what every published GMaster number used: the tables
    hold exactly the values the fused kernel recomputes, so a table transform and
    a kernel transform agree to ~1e-16.

    `"fp32"` halves the device bytes of those tables.  The largest geometries are
    dispatched by a fit test, not by speed, so what this buys is *engagement*:
    a geometry that declined the tables and took the recurrence-bound fused
    kernel can take the memory-bound contraction instead.  The recurrence that
    generates the values stays float64 and every contraction still accumulates in
    float64; the price is the table's own representation error (~1e-7 relative on
    the coupling matrix), which is why it is opt-in and never inferred.

    Cached tables are keyed by dtype, and the caches are process-global, so
    switching clears them rather than handing a caller the other precision's
    bytes.
    """
    if name not in _TABLE_DTYPES:
        raise KeyError("GMaster table precision must be 'fp64' or 'fp32'")
    if name == nmt_params.table_dtype:
        return
    nmt_params.table_dtype = name
    from . import _spin_slice
    from . import _theta_matrix

    _theta_matrix.release()
    _spin_slice.clear_cache()
    _ring_analysis_tables.cache_clear()
    _ring_synthesis_tables.cache_clear()
    _spin_ring_analysis_tables.cache_clear()
    _spin_ring_synthesis_tables.cache_clear()


def set_sht_calculator(calc_name):
    if calc_name not in (
        "jax",
        "jax-single",
        "jax-mgpu",
        "jax-generic",
        "jax-dfp32",
        "jax-matrix",
    ):
        raise KeyError(
            "GMaster's SHT calculator must be 'jax', 'jax-single', "
            "'jax-mgpu', 'jax-generic', 'jax-dfp32', or 'jax-matrix'"
        )
    nmt_params.sht_calculator = calc_name


def set_n_iter_default(n_iter, mask=False):
    if n_iter < 0:
        raise ValueError("n_iter must be positive")
    attribute = "n_iter_mask_default" if mask else "n_iter_default"
    setattr(nmt_params, attribute, int(n_iter))


def set_tol_pinv_default(tol_pinv):
    if not 0 <= tol_pinv <= 1:
        raise ValueError("tol_pinv must be between 0 and 1")
    nmt_params.tol_pinv_default = float(tol_pinv)


def get_default_params():
    return {
        name: getattr(nmt_params, name)
        for name in (
            "sht_calculator",
            "n_iter_default",
            "n_iter_mask_default",
            "tol_pinv_default",
        )
    }


def mask_apodization(mask_in, aposize, apotype="C1"):
    """Apodize a HEALPix mask using NaMaster's C1, C2, or Smooth rule."""
    if apotype not in ("C1", "C2", "Smooth"):
        raise ValueError(
            f"Apodization type {apotype} unknown. Choose from ['C1', 'C2', 'Smooth']"
        )
    if aposize < 0:
        raise ValueError("Apodization scale must be a positive number")
    mask = np.asarray(mask_in, dtype=np.float64)
    if mask.ndim != 1:
        raise ValueError("Mask must be a one-dimensional HEALPix map")
    if aposize == 0:
        return mask.copy()

    import healpy as hp
    from scipy.spatial import cKDTree

    nside = hp.npix2nside(len(mask))
    radius = np.radians(aposize)
    scale = 2.5 if apotype == "Smooth" else 1.2
    if int(4 * len(mask) * (1 - np.cos(scale * radius))) < 2:
        raise ValueError("Your apodization scale is too small for this pixel size")
    masked = np.flatnonzero(mask <= 0)
    if not len(masked):
        return mask.copy()
    vectors = np.asarray(hp.pix2vec(nside, np.arange(len(mask)))).T

    if apotype == "Smooth":
        binary = mask.copy()
        for pixel in masked:
            binary[
                hp.query_disc(
                    nside,
                    vectors[pixel],
                    2.5 * radius,
                    inclusive=True,
                    fact=1,
                )
            ] = 0
        map_info = NmtMapInfo(None, binary.shape)
        alm_info = NmtAlmInfo(map_info.get_lmax())
        alm = map2alm(binary[None, :], 0, map_info, alm_info, n_iter=3)
        multipoles = jnp.arange(alm_info.lmax + 1)
        beam = jnp.exp(-0.5 * multipoles * (multipoles + 1) * radius**2)
        smoothed = alm2map(alm * beam[alm_info._ell], 0, map_info, alm_info)[0]
        return smoothed * jnp.asarray(mask)

    distances, _ = cKDTree(vectors[masked]).query(vectors)
    distance_measure = 0.5 * distances**2
    threshold = 1 - np.cos(radius)
    affected = (mask > 0) & (distance_measure < threshold)
    x = np.sqrt(np.maximum(distance_measure[affected], 0) / threshold)
    if apotype == "C1":
        factor = x - np.sin(2 * np.pi * x) / (2 * np.pi)
    else:
        factor = 0.5 * (1 - np.cos(np.pi * x))
    output = mask.copy()
    output[affected] *= factor
    return output


def mask_apodization_flat(mask_in, lx, ly, aposize, apotype="C1"):
    """Apodize a 2-D flat-sky mask using NaMaster's conventions."""
    if apotype not in ("C1", "C2", "Smooth"):
        raise ValueError(
            f'Unknown apodization type {apotype}. Allowed: "Smooth", "C1", "C2"'
        )
    if aposize < 0:
        raise ValueError("Apodization scale must be a positive number")
    mask = np.asarray(mask_in, dtype=np.float64)
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D array")
    if aposize == 0 or not np.any(mask <= 0):
        return mask.copy()

    from scipy.ndimage import distance_transform_edt

    from .field_flat import _flat_alm2map, _flat_map2alm, _wavevectors

    radius = np.radians(aposize)
    pixel_size = lx / mask.shape[1]
    distance = distance_transform_edt(mask > 0, sampling=pixel_size)
    if apotype == "Smooth":
        eroded = np.where(distance <= 2.5 * radius, 0, mask)
        alms = _flat_map2alm(jnp.asarray(eroded)[None], 0, lx, ly)
        kx, ky = _wavevectors(mask.shape[1], mask.shape[0], lx, ly)
        ell = jnp.sqrt(kx**2 + ky**2)
        sigma = 0.00012352884853326381 * aposize * 60 * 2.355
        sample_ells = jnp.arange(640) / (128 * sigma)
        sample_beam = jnp.exp(-0.5 * (sample_ells * sigma) ** 2)
        beam = jnp.where(
            ell >= sample_ells[-1],
            0,
            jnp.interp(ell, sample_ells, sample_beam, left=1),
        )
        smoothed = _flat_alm2map(
            alms * beam, 0, lx, ly, mask.shape[1]
        )[0]
        return smoothed * jnp.asarray(mask)

    affected = (mask > 0) & (distance < radius)
    x = distance[affected] / radius
    if apotype == "C1":
        factor = x - np.sin(2 * np.pi * x) / (2 * np.pi)
    else:
        factor = 0.5 * (1 - np.cos(np.pi * x))
    output = mask.copy()
    output[affected] *= factor
    return output


class _SHTInfo:
    def __init__(self, nside):
        self.nside = nside
        self.nring = 4 * nside - 1
        rings = np.arange(1, 4 * nside)
        north = np.where(rings > 2 * nside, 4 * nside - rings, rings)
        cap = north < nside
        npix = 12 * nside**2

        self.theta = np.empty(self.nring)
        self.phi0 = np.empty(self.nring)
        self.nphi = np.empty(self.nring, dtype=np.uint64)
        self.theta[cap] = 2 * np.arcsin(north[cap] / (np.sqrt(6) * nside))
        self.phi0[cap] = np.pi / (4 * north[cap])
        self.nphi[cap] = 4 * north[cap]
        self.theta[~cap] = np.arccos((2 * nside - north[~cap]) * (8 * nside / npix))
        self.phi0[~cap] = np.pi / (4 * nside) * (((north[~cap] - nside) & 1) == 0)
        self.nphi[~cap] = 4 * nside
        south = north != rings
        self.theta[south] = np.pi - self.theta[south]
        self.offsets = np.concatenate([[0], np.cumsum(self.nphi)[:-1]]).astype(
            np.uint64
        )
        self.weight = 4 * np.pi / npix
        self.is_CAR = False
        self.nx_short = -1
        self.nx_full = -1

    def pad_map(self, maps):
        return maps

    def unpad_map(self, maps):
        return maps

    def times_weight(self, maps):
        return maps * self.weight

    def map_integral(self, maps):
        return jnp.sum(self.times_weight(maps))

    def dot_map(self, map1, map2):
        return self.map_integral(map1 * map2)


def _clenshaw_curtis_weights(size):
    intervals = size - 1
    theta = np.pi * np.arange(size) / intervals
    weights = np.zeros(size)
    interior = np.arange(1, intervals)
    values = np.ones(intervals - 1)
    if intervals % 2 == 0:
        weights[[0, -1]] = 1 / (intervals**2 - 1)
        for order in range(1, intervals // 2):
            values -= 2 * np.cos(2 * order * theta[interior]) / (4 * order**2 - 1)
        values -= np.cos(intervals * theta[interior]) / (intervals**2 - 1)
    else:
        weights[[0, -1]] = 1 / intervals**2
        for order in range(1, (intervals + 1) // 2):
            values -= 2 * np.cos(2 * order * theta[interior]) / (4 * order**2 - 1)
    weights[interior] = 2 * values / intervals
    return weights


class _CARSHTInfo:
    def __init__(self, n_theta, theta_min, d_theta, n_phi, d_phi, phi0):
        self.nring = int(n_theta)
        self.theta = theta_min + np.arange(n_theta) * d_theta
        self.phi0 = np.full(n_theta, phi0)
        self.nx_short = int(n_phi)
        self.nx_full = int(2 * np.pi / d_phi + 0.5)
        self.nphi = np.full(n_theta, self.nx_full, dtype=np.uint64)
        self.offsets = (np.arange(n_theta) * self.nx_full).astype(np.uint64)
        full_theta = int(np.pi / d_theta + 0.5) + 1
        theta_start = int(theta_min / d_theta + 0.5)
        self.weight = (
            _clenshaw_curtis_weights(full_theta)[theta_start : theta_start + n_theta]
            * d_phi
        )
        self.is_CAR = True
        self.nside = -1
        theta_grid = np.repeat(self.theta, n_phi)
        phi_grid = np.tile(phi0 + np.arange(n_phi) * d_phi, n_theta)
        self.positions = jnp.asarray(
            np.stack((theta_grid, np.mod(phi_grid, 2 * np.pi)))
        )

    def pad_map(self, maps):
        shape = maps.shape[:-1] + (self.nring, self.nx_short)
        padded = jnp.zeros(maps.shape[:-1] + (self.nring, self.nx_full))
        return padded.at[..., : self.nx_short].set(maps.reshape(shape)).reshape(
            maps.shape[:-1] + (self.nring * self.nx_full,)
        )

    def unpad_map(self, maps):
        return maps.reshape(maps.shape[:-1] + (self.nring, self.nx_full))[
            ..., : self.nx_short
        ].reshape(maps.shape[:-1] + (self.nring * self.nx_short,))

    def times_weight(self, maps):
        shape = maps.shape[:-1] + (self.nring, self.nx_short)
        reshaped = maps.reshape(shape)
        weight_shape = (1,) * (reshaped.ndim - 2) + (self.nring, 1)
        return (reshaped * jnp.asarray(self.weight).reshape(weight_shape)).reshape(maps.shape)

    def map_integral(self, maps):
        return jnp.sum(self.times_weight(maps))

    def dot_map(self, map1, map2):
        return self.map_integral(map1 * map2)


class NmtMapInfo:
    """Description of a curved-sky map pixelization."""

    def __init__(self, wcs, axes):
        if wcs is not None:
            try:
                ny, nx = axes
            except ValueError as error:
                raise ValueError("Input maps must be 2D if not HEALPix") from error
            d_ra, d_dec = wcs.wcs.cdelt[:2]
            _, dec0 = wcs.wcs.crval[:2]
            ctype_ra, ctype_dec = wcs.wcs.ctype[0], wcs.wcs.ctype[1]
            if not (ctype_ra[-3:] == "CAR" and ctype_dec[-3:] == "CAR"):
                raise ValueError("Maps must have CAR pixelization")
            if abs(dec0) > 1e-3:
                raise ValueError("Reference pixel must be at the equator")
            d_theta, d_phi = abs(np.radians(d_dec)), abs(np.radians(d_ra))
            if (
                abs(round(2 * np.pi / d_phi) - 2 * np.pi / d_phi) > 0.01
                or abs(round(np.pi / d_theta) - np.pi / d_theta) > 0.01
            ):
                raise ValueError("The pixels should divide the sphere exactly")
            self.flip_th = d_dec > 0
            self.flip_ph = d_ra < 0
            points = np.zeros((1, len(wcs.wcs.crpix)))
            other = points.copy()
            other[0, 1] = ny - 1
            edges = np.array([
                wcs.wcs_pix2world(points, 0)[0, 1],
                wcs.wcs_pix2world(other, 0)[0, 1],
            ])
            self.theta_min = np.radians(90 - np.max(edges))
            self.theta_max = np.radians(90 - np.min(edges))
            azimuth = np.zeros((1, len(wcs.wcs.crpix)))
            if self.flip_ph:
                azimuth[0, 0] = nx - 1
            phi0 = wcs.wcs_pix2world(azimuth, 0)[0, 0]
            if np.isnan(phi0):
                raise ValueError("There is something wrong with the azimuths")
            self.phi0 = np.radians(phi0)
            if (
                self.theta_min < 0
                or self.theta_max > np.pi
                or np.isnan(edges).any()
            ):
                raise ValueError("The colatitude map edges are outside the sphere")
            if abs(nx * d_ra) > 360 + 0.1 * abs(d_ra):
                raise ValueError("Seems like you're wrapping the sphere more than once")
            self.is_healpix = False
            self.nside = -1
            self.npix = int(nx * ny)
            self.nx, self.ny = int(nx), int(ny)
            self.d_theta, self.d_phi = d_theta, d_phi
            self.si = _CARSHTInfo(
                ny, self.theta_min, d_theta, nx, d_phi, self.phi0
            )
            return
        nside = 2
        while 12 * nside**2 != axes[0]:
            nside *= 2
            if nside > 65536:
                raise ValueError("Something is wrong with your input arrays")

        self.is_healpix = True
        self.nside = nside
        self.npix = 12 * nside**2
        self.nx = self.ny = -1
        self.flip_th = self.flip_ph = False
        self.theta_min = self.theta_max = -1
        self.d_theta = self.d_phi = self.phi0 = -1
        self.si = _SHTInfo(nside)

    def __eq__(self, other):
        if not isinstance(other, NmtMapInfo) or self.is_healpix != other.is_healpix:
            return False
        if self.is_healpix:
            return self.nside == other.nside
        return all(
            getattr(self, name) == getattr(other, name)
            for name in (
                "npix", "nx", "ny", "theta_min", "theta_max",
                "phi0", "d_theta", "d_phi",
            )
        )

    def reform_map(self, maps):
        if not self._map_compatible(maps):
            raise ValueError("Incompatible map!")
        if self.is_healpix:
            return maps
        if self.flip_th:
            maps = maps[..., ::-1, :]
        if self.flip_ph:
            maps = maps[..., :, ::-1]
        return maps.reshape(maps.shape[:-2] + (self.npix,))

    def _map_compatible(self, maps):
        if self.is_healpix:
            return maps.shape[-1] == self.npix
        return maps.shape[-2:] == (self.ny, self.nx)

    def get_lmax(self):
        if self.is_healpix:
            return 3 * self.nside - 1
        return int(np.pi / min(self.d_theta, self.d_phi))


# `ell`/`m` are a pure function of lmax, but building them costs lmax+1 numpy.arange calls
# plus a device_put per construction, and `NmtField` builds two per field: 1.15 ms of the
# 1.66 ms constructor at Nside 64.  The budget is in elements because at Nside 2048 a single
# pair is 38M of them.
_ALM_INDEX_BUDGET = 64_000_000
_ALM_INDEX_CACHE: dict = {}
_ALM_INDEX_ELEMENTS = 0


def _alm_index_arrays(lmax, m):
    global _ALM_INDEX_ELEMENTS
    cached = _ALM_INDEX_CACHE.get(lmax)
    if cached is not None:
        return cached
    ell = np.concatenate([np.arange(mm, lmax + 1) for mm in m])
    order = np.repeat(m, lmax + 1 - m)
    cached = (jnp.asarray(ell), jnp.asarray(order))
    if isinstance(cached[0], Tracer) or isinstance(cached[1], Tracer):
        # Several helpers build an `NmtAlmInfo` inside a trace; a tracer in a
        # module-level cache surfaces later as an UnexpectedTracerError.
        return cached
    if _ALM_INDEX_ELEMENTS + 2 * len(ell) > _ALM_INDEX_BUDGET:
        _ALM_INDEX_CACHE.clear()
        _ALM_INDEX_ELEMENTS = 0
    _ALM_INDEX_CACHE[lmax] = cached
    _ALM_INDEX_ELEMENTS += 2 * len(ell)
    return cached


class NmtAlmInfo:
    """Description of Healpy-packed spherical-harmonic coefficients."""

    def __init__(self, lmax):
        self.lmax = int(lmax)
        self.mmax = self.lmax
        m = np.arange(self.mmax + 1)
        self.mstart = (m * (2 * self.lmax + 1 - m) // 2).astype(np.uint64)
        self.nelem = (self.lmax + 1) * (self.lmax + 2) // 2
        self._ell, self._m = _alm_index_arrays(self.lmax, m)

    def __eq__(self, other):
        return isinstance(other, NmtAlmInfo) and self.lmax == other.lmax


def moore_penrose_pinvh(mat, tol_pinv):
    """Hermitian pseudo-inverse using NaMaster's relative eigenvalue cutoff."""
    matrix = jnp.asarray(mat)
    if tol_pinv is None or tol_pinv <= 0:
        return jnp.linalg.inv(matrix)
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    inverse = jnp.where(
        eigenvalues >= tol_pinv * jnp.max(eigenvalues), 1 / eigenvalues, 0
    )
    return (eigenvectors * inverse) @ jnp.conj(eigenvectors.T)


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
    if source is not None:
        array = np.asarray(array)
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
    from . import _theta_matrix

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
    if (m_start == 0 and nmt_params.sht_calculator in ("jax", "jax-matrix")
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


def _chirp(index, two_nphi, sign, L):
    """exp(sign i pi q^2 / nphi) in the ring stage's dtype.

    The angle is reduced exactly in integers as in `_chirp_angle`; for a complex64 ring stage the
    reduction runs in int32 as ((q mod m)^2) mod m (m = 2 nphi <= 32768, so the square fits), the
    angle is scaled in float64 and evaluated with float32 sincos, all in one fused program.  Built
    in complex128 (int64 squares, float64 sincos at 1/64 rate, complex128 kernel FFTs) these tables
    took 287 ms at Nside 4096 spin 2 -- too slow to rebuild per field, so 6.5 GiB stayed resident.
    """
    if jnp.dtype(ring_dtype(L)) == jnp.complex64:
        return _chirp_c64(jnp.asarray(index), jnp.asarray(two_nphi), sign=float(sign))
    reduced = (index.astype(jnp.int64) ** 2) % two_nphi
    return jnp.exp(sign * 1j * reduced * (jnp.pi / two_nphi) * 2.0)


@partial(jax.jit, static_argnames=("sign",))
def _chirp_c64(index, two_nphi, *, sign):
    m = two_nphi.astype(jnp.int32)
    r = jnp.mod(index.astype(jnp.int32), m)
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
        transform_size = _next_fast_len_pow2(L + width)
        n_index = jnp.arange(width, dtype=jnp.int64)
        m_index = jnp.arange(L, dtype=jnp.int64)
        shift = jnp.arange(transform_size, dtype=jnp.int64) - (width - 1)
        tables = (
            _chirp(n_index, two_nphi, -1.0, L),
            jnp.fft.fft(_chirp(shift, two_nphi, 1.0, L), axis=-1),
            _chirp(m_index, two_nphi, -1.0, L),
        )
        cplx = ring_dtype(L)
        tables = tuple(t.astype(cplx) for t in tables)
    return tables if device is None else tuple(
        jax.device_put(t, device) for t in tables
    )


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
    chirp_in, kernel_spec, chirp_out = tables
    belt_lo, belt_hi, belt_start, caps = _ring_split_numpy(L, nside)
    width = 4 * nside
    pixels = jnp.reshape(jnp.asarray(map_flat), (-1,)).astype(chirp_in.real.dtype)

    cap_out = None
    if caps.size:
        cap_pixels = _cap_pixels(pixels, caps, L, nside)
        transform_size = _next_fast_len_pow2(L + width)
        embedded = jnp.pad(cap_pixels * chirp_in, ((0, 0), (0, transform_size - width)))
        convolution = jnp.fft.ifft(
            jnp.fft.fft(embedded, axis=-1) * kernel_spec, axis=-1
        )
        cap_out = chirp_out * convolution[:, width - 1 : width - 1 + L]

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
        transform_size = _next_fast_len_pow2(width + 2 * L)
        n_index = jnp.arange(width, dtype=jnp.int64)
        m_index = jnp.arange(-(L - 1), L, dtype=jnp.int64)
        shift = jnp.arange(transform_size, dtype=jnp.int64) - (width - 1) - (L - 1)
        tables = (
            _chirp(n_index, two_nphi, -1.0, L),
            jnp.fft.fft(_chirp(shift, two_nphi, 1.0, L), axis=-1),
            _chirp(m_index, two_nphi, -1.0, L),
        )
        cplx = ring_dtype(L)
        tables = tuple(t.astype(cplx) for t in tables)
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
    chirp_in, kernel_spec, chirp_out = tables
    belt_lo, belt_hi, belt_start, caps = _ring_split_numpy(L, nside)
    width = 4 * nside
    # The signal here is a helicity combination `Q -/+ iU`, so it takes the tables'
    # *complex* type: casting to their real type would silently drop the imaginary part.
    pixels = jnp.reshape(jnp.asarray(signal), (-1,)).astype(chirp_in.dtype)

    cap_out = None
    if caps.size:
        cap_pixels = _cap_pixels(pixels, caps, L, nside)
        transform_size = _next_fast_len_pow2(width + 2 * L)

        def czt(px, c_in, k_spec, c_out):
            embedded = jnp.pad(px * c_in, ((0, 0), (0, transform_size - width)))
            convolution = jnp.fft.ifft(jnp.fft.fft(embedded, axis=-1) * k_spec, axis=-1)
            return c_out * convolution[:, width - 1 : width - 1 + 2 * L - 1]

        nrows = int(caps.size)
        if nrows <= _CAP_CHUNK_ROWS:
            cap_out = czt(cap_pixels, chirp_in, kernel_spec, chirp_out)
        else:
            cap_out = _chunked_rows(
                nrows, _CAP_CHUNK_ROWS,
                lambda lo, hi: czt(*(a[lo:hi] for a in (cap_pixels, chirp_in, kernel_spec,
                                                         chirp_out))))

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
        transform_size = _next_fast_len_pow2(2 * L - 1 + width)
        c_index = jnp.arange(2 * L - 1, dtype=jnp.int64)
        p_index = jnp.arange(width, dtype=jnp.int64)
        shift = jnp.arange(transform_size, dtype=jnp.int64) - (2 * L - 2)
        tables = (
            _chirp(c_index, two_nphi, 1.0, L),
            jnp.fft.fft(_chirp(shift, two_nphi, -1.0, L), axis=-1),
            _chirp(p_index, two_nphi, 1.0, L),
        )
        cplx = ring_dtype(L)
        tables = tuple(t.astype(cplx) for t in tables)
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
    chirp_c, kernel_spec, chirp_p = tables
    belt_lo, belt_hi, _, caps = _ring_split_numpy(L, nside)
    nphi, _, _, _, width = _ring_czt_constants(L, nside)
    grid = jnp.asarray(centered).astype(chirp_c.dtype)

    cap_res = None
    if caps.size:
        cap_grid = grid[jnp.asarray(caps)]
        transform_size = _next_fast_len_pow2(2 * L - 1 + width)
        p_index = jnp.arange(width, dtype=jnp.int64)
        caps_nphi = nphi[jnp.asarray(caps)]

        def czt(g, c_c, k_spec, c_p, rows_nphi):
            embedded = jnp.pad(g * c_c, ((0, 0), (0, transform_size - (2 * L - 1))))
            convolution = jnp.fft.ifft(jnp.fft.fft(embedded, axis=-1) * k_spec, axis=-1)
            res = c_p * convolution[:, 2 * L - 2 : 2 * L - 2 + width]
            rows_nphi = rows_nphi[:, None]
            wrap_phase = ((L - 1) % rows_nphi) * p_index[None, :] % rows_nphi
            return res * jnp.exp(-1j * wrap_phase * (2 * jnp.pi / rows_nphi))

        nrows = int(caps.size)
        if nrows <= _CAP_CHUNK_ROWS:
            cap_res = czt(cap_grid, chirp_c, kernel_spec, chirp_p, caps_nphi)
        else:
            cap_res = _chunked_rows(
                nrows, _CAP_CHUNK_ROWS,
                lambda lo, hi: czt(*(a[lo:hi] for a in (cap_grid, chirp_c, kernel_spec, chirp_p,
                                                         caps_nphi))))

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
    maps = _copy_to_device(maps, primary)
    ftm = _forward_s2fft_ftm(
        maps,
        () if reality else _spin_ring_analysis_tables(L, nside, primary),
        L=L, nside=nside, reality=reality,
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
    dc = _dc_spin(L_work, spin)
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
    dc = _dc_spin(L_work, spin)
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
    if _dc_spin(L_work, spin) is not None:
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
    from . import _theta_matrix

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
    return _fused_inverse_sht(_positive_alm(alm[0], L=L, L_work=L_work), theta, phase,
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
    maps = _copy_to_device(maps, primary)
    ftm = _forward_healpix_fft(maps[0], L=L_work, nside=nside, reality=True)
    positive = ftm[:, L_work:]
    split = _pallas_order_split(L_work)
    high_input = _copy_to_device(positive[:, split:], secondary)
    theta, weights, phase = _pallas_parameters(L_work, nside)
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

    with ThreadPoolExecutor(max_workers=2) as executor:
        low_future = executor.submit(
            _run_blocking, low_transform, positive[:, :split]
        )
        high_future = executor.submit(_run_blocking, high_transform, high_input)
        low, high = low_future.result(), high_future.result()
    high = _copy_to_device(high, primary)
    packed = _pack_pallas_parts(low, high, ell, order, split=split)
    return packed[None, :]


def _alm2map_core_pallas_multi_gpu(alm, *, nside, L, L_work):
    primary, secondary = _gpu_devices()[:2]
    alm = _copy_to_device(alm, primary)
    positive = _positive_alm(alm[0], L=L, L_work=L_work)
    split = _pallas_order_split(L_work)
    high_input = _copy_to_device(positive[:, split:], secondary)
    theta = _stable_thetas(L_work, nside)
    phase = healpix_ffts.p2phi_rings_jax(jnp.arange(len(theta)), nside)
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

    with ThreadPoolExecutor(max_workers=2) as executor:
        low_future = executor.submit(
            _run_blocking, low_transform, positive[:, :split]
        )
        high_future = executor.submit(_run_blocking, high_transform, high_input)
        low, high = low_future.result(), high_future.result()
    high = _copy_to_device(high, primary)
    ftm_positive = jnp.concatenate((low, high), axis=1)
    maps = _finish_inverse_pallas(ftm_positive, L=L_work, nside=nside)
    return jnp.real(maps)[None, :]


def _map2alm_core_pallas_multi_gpu(
    maps, ell, order, *, nside, L, L_work, n_iter
):
    alm = _map2alm_once_pallas_multi_gpu(
        maps, ell, order, nside=nside, L_work=L_work
    )
    for _ in range(n_iter):
        residual = (
            _alm2map_core_pallas_multi_gpu(
                alm, nside=nside, L=L, L_work=L_work
            )
            - maps
        )
        alm -= _map2alm_once_pallas_multi_gpu(
            residual, ell, order, nside=nside, L_work=L_work
        )
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
    return jnp.stack([-(plus_m + minus_m) / 2, 0.5j * (plus_m - minus_m)])


def _map2alm_core_multi_gpu(
    maps, ell, order, *, spin, nside, L, L_work, n_iter
):
    maps = _copy_to_device(maps, _gpu_devices()[0])
    alm = _map2alm_once_multi_gpu(
        maps, ell, order, spin=spin, nside=nside, L=L, L_work=L_work
    )
    for _ in range(n_iter):
        residual = (
            _alm2map_core_multi_gpu(
                alm,
                ell,
                order,
                spin=spin,
                nside=nside,
                L=L,
                L_work=L_work,
            )
            - maps
        )
        alm -= _map2alm_once_multi_gpu(
            residual,
            ell,
            order,
            spin=spin,
            nside=nside,
            L=L,
            L_work=L_work,
        )
    return alm


def map2alm(map, spin, map_info, alm_info, *, n_iter):
    """Transform HEALPix maps to NaMaster/Healpy-packed E/B coefficients."""
    maps = jnp.asarray(map)
    nmaps = 1 if spin == 0 else 2
    if maps.ndim != 2 or maps.shape != (nmaps, map_info.npix):
        raise ValueError("Input map has wrong shape")
    if spin < 0 or spin > alm_info.lmax:
        raise ValueError("spin must satisfy 0 <= spin <= lmax")
    if not map_info.is_healpix:
        # ponytail: direct CAR synthesis is O(npix*lmax^2); replace it with a
        # separable theta/FFT transform when production CAR maps exceed memory.
        weighted = map_info.si.times_weight(maps)
        alm = catalog2alm(
            weighted, map_info.si.positions, spin, alm_info.lmax
        )
        for _ in range(int(n_iter)):
            residual = alm2catalog(
                alm, map_info.si.positions, spin, alm_info.lmax
            ) - maps
            alm -= catalog2alm(
                map_info.si.times_weight(residual),
                map_info.si.positions,
                spin,
                alm_info.lmax,
            )
        return alm
    L = alm_info.lmax + 1
    # The Pallas latitudinal SHT resolves m up to L-1 from the HEALPix rings
    # directly and is accurate for L < 2*nside (NaMaster likewise only
    # integrates m <= lmax), so let the working order track L there.  The
    # generic s2fft reference ring FFT cannot concatenate its ring regions
    # when L < 2*nside, so it still needs the working order lifted to 2*nside.
    L_work = L
    if _use_pallas_sht(L_work, spin):
        if False:
            return _map2alm_core_pallas_multi_gpu(
                maps,
                alm_info._ell,
                alm_info._m,
                nside=map_info.nside,
                L=L,
                L_work=L_work,
                n_iter=int(n_iter),
            )
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


def map2alm_pair(map_a, map_b, map_info, alm_info, *, n_iter):
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
    maps_a = jnp.asarray(map_a)
    maps_b = jnp.asarray(map_b)
    if (maps_a.ndim != 2 or maps_a.shape != (1, map_info.npix)
            or maps_b.shape != (1, map_info.npix)):
        raise ValueError("shared-band pair expects two (1, npix) spin-0 maps")
    if not map_info.is_healpix:
        return None
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
    """Transform NaMaster/Healpy-packed E/B coefficients to HEALPix maps."""
    alm = jnp.asarray(alm)
    nmaps = 1 if spin == 0 else 2
    if alm.ndim != 2 or alm.shape != (nmaps, alm_info.nelem):
        raise ValueError("Input alm has wrong shape")
    if spin < 0 or spin > alm_info.lmax:
        raise ValueError("spin must satisfy 0 <= spin <= lmax")
    if not map_info.is_healpix:
        return alm2catalog(alm, map_info.si.positions, spin, alm_info.lmax)
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


@partial(jax.jit, static_argnames=("spin", "lmax"))
def _catalog2alm_core(values, positions, *, spin, lmax):
    """Adjoint arbitrary-position spin transform in Healpy alm ordering."""
    L = lmax + 1
    theta, phi = positions
    signal = values[0] if spin == 0 else values[0] + 1j * values[1]
    full = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
    # ponytail: O(n_source*lmax^2); block sources or add a catalog NUFFT when
    # retained catalogs no longer fit device memory at survey scale.
    for ell in range(abs(spin), L):
        d_slice = jax.vmap(
            lambda angle: turok_jax.compute_slice(angle, ell, L, -spin)
        )(theta)
        orders = jnp.arange(-ell, ell + 1)
        coefficients = (
            (-1) ** abs(spin)
            * jnp.sqrt((2 * ell + 1) / (4 * jnp.pi))
            * jnp.sum(
                signal[:, None]
                * d_slice[:, orders + L - 1]
                * jnp.exp(-1j * phi[:, None] * orders),
                axis=0,
            )
        )
        full = full.at[ell, orders + L - 1].set(coefficients)

    alm_info = NmtAlmInfo(lmax)
    plus = full[alm_info._ell, L - 1 + alm_info._m]
    if spin == 0:
        return plus[None]
    minus = (-1) ** alm_info._m * jnp.conj(
        full[alm_info._ell, L - 1 - alm_info._m]
    )
    return jnp.stack([-(plus + minus) / 2, 0.5j * (plus - minus)])


def catalog2alm(values, positions, spin, lmax):
    """Transform weighted catalog samples to Healpy-packed E/B coefficients."""
    positions = jnp.asarray(positions, dtype=jnp.float64)
    values = jnp.atleast_2d(jnp.asarray(values, dtype=jnp.float64))
    nmaps = 1 if spin == 0 else 2
    if positions.ndim != 2 or positions.shape[0] != 2:
        raise ValueError("positions must have shape (2, n_source)")
    if values.shape != (nmaps, positions.shape[1]):
        raise ValueError(f"values must have shape {(nmaps, positions.shape[1])}")
    if spin < 0 or spin > lmax:
        raise ValueError("spin must satisfy 0 <= spin <= lmax")
    return _catalog2alm_core(values, positions, spin=int(spin), lmax=int(lmax))


@partial(jax.jit, static_argnames=("spin", "lmax"))
def _alm2catalog_core(alms, positions, *, spin, lmax):
    L = lmax + 1
    theta, phi = positions
    alm_info = NmtAlmInfo(lmax)
    full = (
        _unpack_real(alms[0], L, L)
        if spin == 0
        else _unpack_spin(alms, L, L)
    )
    signal = jnp.zeros(len(theta), dtype=jnp.complex128)
    for ell in range(abs(spin), L):
        d_slice = jax.vmap(
            lambda angle: turok_jax.compute_slice(angle, ell, L, -spin)
        )(theta)
        orders = jnp.arange(-ell, ell + 1)
        signal += (
            (-1) ** abs(spin)
            * jnp.sqrt((2 * ell + 1) / (4 * jnp.pi))
            * jnp.sum(
                d_slice[:, orders + L - 1]
                * jnp.exp(1j * phi[:, None] * orders)
                * full[ell, orders + L - 1],
                axis=1,
            )
        )
    return jnp.real(signal)[None] if spin == 0 else jnp.stack(
        [jnp.real(signal), jnp.imag(signal)]
    )


def alm2catalog(alms, positions, spin, lmax):
    """Evaluate Healpy-packed E/B coefficients at catalog positions."""
    positions = jnp.asarray(positions, dtype=jnp.float64)
    alms = jnp.asarray(alms)
    nmaps = 1 if spin == 0 else 2
    expected = NmtAlmInfo(lmax).nelem
    if positions.ndim != 2 or positions.shape[0] != 2:
        raise ValueError("positions must have shape (2, n_source)")
    if alms.shape != (nmaps, expected):
        raise ValueError(f"alms must have shape {(nmaps, expected)}")
    return _alm2catalog_core(alms, positions, spin=int(spin), lmax=int(lmax))


@jax.jit
def _gaussian_alms(key, covariance_root, ell, order):
    real_key, imaginary_key = jax.random.split(key)
    shape = (len(ell), covariance_root.shape[-1])
    real = jax.random.normal(real_key, shape, dtype=covariance_root.dtype)
    imaginary = jax.random.normal(
        imaginary_key, shape, dtype=covariance_root.dtype
    )
    root = covariance_root[ell]
    real = jnp.einsum("aij,aj->ai", root, real)
    imaginary = jnp.einsum("aij,aj->ai", root, imaginary) / jnp.sqrt(2)
    real = jnp.where(order[:, None] == 0, real, real / jnp.sqrt(2))
    imaginary = jnp.where(order[:, None] == 0, 0, imaginary)
    return (real + 1j * imaginary).T


def synfast_spherical(nside, cls, spin_arr, beam=None, seed=-1, wcs=None, lmax=None):
    """Generate correlated Gaussian HEALPix fields on the active JAX device."""
    if wcs is None and int(nside) <= 0:
        raise ValueError("nside must be positive")
    spins = np.asarray(spin_arr, dtype=np.int32)
    if spins.ndim != 1 or np.any(spins < 0):
        raise ValueError("Spins must be a one-dimensional array of positive values")
    components = np.asarray([1 if spin == 0 else 2 for spin in spins])
    first = np.concatenate([[0], np.cumsum(components)[:-1]])
    nmaps = int(np.sum(components))
    spectra = np.asarray(cls, dtype=np.float64)
    expected = nmaps * (nmaps + 1) // 2
    if spectra.ndim != 2 or len(spectra) != expected:
        raise ValueError(f"Must provide all Cls necessary to simulate all fields ({expected})")

    if wcs is None:
        map_info = NmtMapInfo(None, (12 * int(nside) ** 2,))
    else:
        n_ra = int(round(360 / abs(wcs.wcs.cdelt[0])))
        n_dec = int(round(180 / abs(wcs.wcs.cdelt[1]))) + 1
        map_info = NmtMapInfo(wcs, (n_dec, n_ra))
        if abs(map_info.theta_min) > 1e-4 or abs(map_info.theta_max - np.pi) > 1e-4:
            raise ValueError("Given your WCS, the map wouldn't cover the whole sphere exactly")
    lmax = map_info.get_lmax() if lmax is None else int(lmax)
    lmax = min(lmax, spectra.shape[1] - 1)
    alm_info = NmtAlmInfo(lmax)
    covariance = np.zeros((lmax + 1, nmaps, nmaps))
    index = 0
    for row in range(nmaps):
        for column in range(row, nmaps):
            covariance[:, row, column] = spectra[index, : lmax + 1]
            covariance[:, column, row] = spectra[index, : lmax + 1]
            index += 1
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = np.max(np.abs(eigenvalues), axis=1, keepdims=True)
    if np.any(eigenvalues < -1e-12 * np.maximum(scale, 1)):
        raise ValueError("Input power spectra do not form positive-semidefinite matrices")
    covariance_root = jnp.asarray(
        eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))[:, None, :]
    )
    if seed < 0:
        seed = np.random.randint(50_000_000)
    alms = _gaussian_alms(
        jax.random.key(int(seed)), covariance_root, alm_info._ell, alm_info._m
    )

    if beam is not None:
        beam = np.asarray(beam)
        if beam.ndim != 2 or len(beam) != len(spins):
            raise ValueError("Must provide one beam per field")
        if beam.shape[1] < lmax + 1:
            raise ValueError(f"The beam should be provided to ell = {lmax}")
        beam_per_component = np.repeat(beam[:, : lmax + 1], components, axis=0)
        alms *= jnp.asarray(beam_per_component)[:, alm_info._ell]

    maps = jnp.concatenate(
        [
            alm2map(
                alms[start : start + count],
                int(spin),
                map_info,
                alm_info,
            )
            for start, count, spin in zip(first, components, spins)
        ]
    )
    if map_info.is_healpix:
        return maps
    maps = maps.reshape((nmaps, map_info.ny, map_info.nx))
    if map_info.flip_th:
        maps = maps[:, ::-1]
    if map_info.flip_ph:
        maps = maps[:, :, ::-1]
    return maps


def synfast_flat(nx, ny, lx, ly, cls, spin_arr, beam=None, seed=-1):
    """Generate correlated Gaussian flat-sky fields on the active JAX device."""
    from .field_flat import _flat_alm2map, _wavevectors

    spins = np.asarray(spin_arr, dtype=np.int32)
    if spins.ndim != 1 or np.any(spins < 0):
        raise ValueError("Spins must be positive")
    components = np.asarray([1 if spin == 0 else 2 for spin in spins])
    nmaps = int(np.sum(components))
    spectra = np.asarray(cls, dtype=np.float64)
    expected = nmaps * (nmaps + 1) // 2
    if spectra.ndim != 2 or len(spectra) != expected:
        raise ValueError(f"Must provide all Cls necessary to simulate all fields ({expected}).")
    nell = spectra.shape[1]

    kx, ky = _wavevectors(int(nx), int(ny), float(lx), float(ly))
    ell = jnp.sqrt(kx**2 + ky**2)
    covariance = jnp.zeros((*ell.shape, nmaps, nmaps), dtype=jnp.float64)
    index = 0
    for row in range(nmaps):
        for column in range(row, nmaps):
            values = jnp.where(
                ell >= nell - 1,
                0,
                jnp.interp(ell, jnp.arange(nell), jnp.asarray(spectra[index])),
            )
            covariance = covariance.at[..., row, column].set(values)
            covariance = covariance.at[..., column, row].set(values)
            index += 1
    eigenvalues, eigenvectors = jnp.linalg.eigh(covariance)
    scale = jnp.max(jnp.abs(eigenvalues), axis=-1, keepdims=True)
    if bool(jnp.any(eigenvalues < -1e-12 * jnp.maximum(scale, 1))):
        raise ValueError("Input power spectra do not form positive-semidefinite matrices")
    root = eigenvectors * jnp.sqrt(jnp.maximum(eigenvalues, 0))[..., None, :]

    if seed < 0:
        seed = np.random.randint(50_000_000)
    white = jax.random.normal(
        jax.random.key(int(seed)), (nmaps, int(ny), int(nx)), dtype=jnp.float64
    )
    independent = jnp.fft.rfft2(white) / jnp.sqrt(nx * ny)
    alms = (
        jnp.einsum("yxij,jyx->iyx", root, independent)
        * jnp.sqrt(lx * ly)
        / (2 * jnp.pi)
    )

    if beam is not None:
        beam = np.asarray(beam, dtype=np.float64)
        if beam.ndim != 2 or len(beam) != len(spins):
            raise ValueError("Must provide one beam per field")
        if beam.shape[1] != nell:
            raise ValueError(
                "The beam should have as many multipoles as the power spectrum"
            )
        beam_components = np.repeat(beam, components, axis=0)
        windows = jnp.stack(
            [
                jnp.where(
                    ell >= nell - 1,
                    0,
                    jnp.interp(ell, jnp.arange(nell), jnp.asarray(window)),
                )
                for window in beam_components
            ]
        )
        alms *= windows

    maps = []
    start = 0
    for spin, count in zip(spins, components):
        maps.append(
            _flat_alm2map(
                alms[start : start + count], int(spin), lx, ly, int(nx)
            )
        )
        start += count
    return jnp.concatenate(maps)
