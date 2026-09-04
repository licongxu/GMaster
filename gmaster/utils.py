"""Map metadata and JAX spherical-harmonic transforms."""

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from s2fft.sampling import s2_samples
from s2fft.recursions import turok_jax
from s2fft.transforms import _ftm_flm_primitive
from s2fft.utils import healpix_ffts, quadrature_jax

from ._sht_pallas import (
    scalar_forward_latitudinal,
    scalar_inverse_latitudinal,
    _scalar_spin_synthesis_latitudinal,
    _spin_forward_latitudinal,
)
from ._sht_dfp32 import scalar_forward_latitudinal_dfp32
from ._theta_matrix import band_bytes as _theta_band_bytes
from ._theta_matrix import inverse_latitudinal as _theta_matrix_inverse_latitudinal
from ._theta_matrix import positive_latitudinal as _theta_matrix_latitudinal
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


nmt_params = NmtParams()

_TABLE_DTYPES = {"fp64": jnp.float64, "fp32": jnp.float32}


def table_dtype():
    """jnp dtype the precomputed transform tables are stored in."""
    return _TABLE_DTYPES[nmt_params.table_dtype]


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


class NmtAlmInfo:
    """Description of Healpy-packed spherical-harmonic coefficients."""

    def __init__(self, lmax):
        self.lmax = int(lmax)
        self.mmax = self.lmax
        m = np.arange(self.mmax + 1)
        self.mstart = (m * (2 * self.lmax + 1 - m) // 2).astype(np.uint64)
        self.nelem = (self.lmax + 1) * (self.lmax + 2) // 2
        ell = np.concatenate([np.arange(mm, self.lmax + 1) for mm in m])
        order = np.repeat(m, self.lmax + 1 - m)
        self._ell = jnp.asarray(ell)
        self._m = jnp.asarray(order)

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
def _forward_s2fft_ftm(maps, *, L, nside, reality):
    m_start = L - 1 if reality else 0
    if reality:
        ftm = healpix_ffts.healpix_fft(maps, L, nside, "jax", reality)
    else:
        # One batched chirp-Z transform instead of s2fft's per-ring unroll: the
        # latter issues one tiny FFT per ring (1023 of them at Nside 256) and
        # runs launch-bound at ~4 GB/s.  Matches it to 1.6e-15.
        centered = _forward_ring_fft_full(maps, L=L, nside=nside)
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
    m_start = L - 1 if reality else 0
    flm = jnp.einsum(
        "lm,l->lm",
        flm,
        jnp.sqrt((2 * jnp.arange(L) + 1) / (4 * jnp.pi)),
        optimize=True,
    )
    if reality:
        flm = flm.at[:, :m_start].set(
            jnp.flip(
                (-1) ** (jnp.arange(1, L) % 2) * jnp.conj(flm[:, m_start + 1 :]),
                axis=-1,
            )
        )
    flm = jnp.where(jnp.arange(L)[:, None] < abs(spin), 0, flm)
    return flm * (-1) ** abs(spin)


@partial(jax.jit, static_argnames=("L", "spin", "nside", "reality"))
def _forward_s2fft(maps, *, L, spin, nside, reality):
    ftm = _forward_s2fft_ftm(maps, L=L, nside=nside, reality=reality)
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
def _finish_inverse_s2fft(ftm, *, L, spin, nside, reality):
    m_start = L - 1 if reality else 0
    ftm = ftm.at[:, m_start + 1 :].multiply(
        healpix_ffts.ring_phase_shifts_hp_jax(L, nside, False, reality)
    )
    ftm *= (-1) ** abs(spin)
    if reality:
        ftm = ftm.at[:, 1:L].set(jnp.flip(jnp.conj(ftm[:, L + 1 :]), axis=-1))
        return healpix_ffts.healpix_ifft(ftm, L, nside, "jax", reality)
    return _inverse_ring_fft_complex(ftm[:, 1:], L=L, nside=nside)


@partial(jax.jit, static_argnames=("L", "spin", "nside", "reality"))
def _inverse_s2fft(flm, *, L, spin, nside, reality):
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
        ftm, L=L, spin=spin, nside=nside, reality=reality
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


def _spin_slabs(L_work, spin, *, nside):
    """Cached (analysis, synthesis) Wigner-d slabs for a polarised transform.

    ``(None, None)`` means keep the generic s2fft scatter loop: either the
    calculator asked for it explicitly, or the pair would not fit in memory.
    """
    if nmt_params.sht_calculator in ("jax-generic", "jax-mgpu"):
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
    return any(
        device.platform == "gpu" and "NVIDIA" in device.device_kind.upper()
        for device in jax.devices()
    )


def _pallas_block_size(nside):
    # 512 lanes per program is the throughput sweet spot: larger tiles
    # under-utilize the SMs (the synthesis kernel degrades ~1.5-1.6x at
    # 1024 and cliffs hard at 2048), while smaller tiles add redundant
    # degree-loop work in analysis. 2*nside keeps the small-Nside case.
    return min(512, 2 * nside)


# Above this working bandlimit the refinement loop runs op-by-op instead of as
# one traced program; see `_map2alm_core_pallas` for the measured crossover.
_PALLAS_TRACED_MAX_L = 256


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
    return _theta_band_bytes(nside, L, dtype=table_dtype()) <= _MATRIX_BAND_BUDGET


def _fused_forward_sht(positive, theta, weights, phase, *, L, block_size,
                       m_start=0):
    """Forward latitudinal SHT, dispatched on the configured calculator.

    The double-fp32 (DFP32) kernel is analysis-only and slightly different
    in precision; every other calculator uses the fp64 Pallas kernel.
    The scalar path prefers the precomputed Legendre band while it fits (see
    `_prefer_theta_band`) and falls back to the kernel when it does not or
    cannot be materialized concretely, which is what happens on `jax.grad`
    paths.
    """
    if _prefer_theta_band((len(theta) + 1) // 4, L, m_start):
        positive_alm = _theta_matrix_latitudinal(
            positive, L=L, nside=(len(theta) + 1) // 4, weights=weights,
            phase=phase,
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
    nphi, start, gather, valid, width = _ring_czt_constants_numpy(L, nside)
    return (
        jnp.asarray(nphi),
        jnp.asarray(start),
        jnp.asarray(gather),
        jnp.asarray(valid),
        width,
    )


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


def _chirp_angle(index, two_nphi, inverse):
    """exp(+-i*pi*q^2/nphi) angles with exact integer modular reduction."""
    reduced = (index.astype(jnp.int64) ** 2) % two_nphi
    angle = reduced * (jnp.pi / two_nphi) * 2.0
    return jnp.where(inverse, -angle, angle)


@partial(jax.jit, static_argnames=("L", "nside"))
def _forward_ring_fft_positive(map_flat, *, L, nside):
    """Chirp-Z ring analysis for every ring, returning the `m in [0, L)` block.

    `_forward_healpix_fft` additionally fills the Hermitian mirror into a
    `(4*nside-1, 2L)` window because s2fft's latitudinal primitive takes that layout. The
    HEALPix pipeline asks for it and then slices it straight back off again
    (`ftm[:, L_work:]`), so a zeroed 2L-wide buffer, two update-slices and a flip/conj
    gather — 100 MB-scale at Nside 512, 400 MB-scale at 1024 — buy nothing there. Same
    numbers, fewer passes: `_forward_ring_fft` is this plus the fill.
    """
    nphi, _, gather, valid, width = _ring_czt_constants(L, nside)
    pixels = jnp.reshape(jnp.asarray(map_flat), (-1,))
    two_nphi = (2 * nphi)[:, None]
    rows = jnp.where(valid, pixels[gather], 0.0)
    transform_size = _next_fast_len_pow2(L + width)

    n_index = jnp.arange(width, dtype=jnp.int64)
    embedded = rows * jnp.exp(-1j * _chirp_angle(n_index, two_nphi, False))
    embedded = jnp.pad(embedded, ((0, 0), (0, transform_size - width)))
    shift = jnp.arange(transform_size, dtype=jnp.int64) - (width - 1)
    kernel = jnp.exp(1j * _chirp_angle(shift, two_nphi, False))
    convolution = jnp.fft.ifft(
        jnp.fft.fft(embedded, axis=-1) * jnp.fft.fft(kernel, axis=-1), axis=-1
    )

    m_index = jnp.arange(L, dtype=jnp.int64)
    return jnp.exp(-1j * _chirp_angle(m_index, two_nphi, False)) * convolution[
        :, width - 1 : width - 1 + L
    ]


@partial(jax.jit, static_argnames=("L", "nside"))
def _forward_ring_fft(map_flat, *, L, nside):
    """Exact HEALPix ring FFT for every ring as one batched chirp-Z transform."""
    positive = _forward_ring_fft_positive(map_flat, L=L, nside=nside)
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


@partial(jax.jit, static_argnames=("L", "nside"))
def _inverse_ring_fft_herm(ftm_positive, *, L, nside):
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
    """
    nphi, _, _, _, width = _ring_czt_constants(L, nside)
    positive = jnp.asarray(ftm_positive)
    two_nphi = (2 * nphi)[:, None]
    transform_size = _next_fast_len_pow2(L + width - 1)

    m_index = jnp.arange(L, dtype=jnp.int64)
    embedded = positive * jnp.exp(1j * _chirp_angle(m_index, two_nphi, False))
    embedded = jnp.pad(embedded, ((0, 0), (0, transform_size - L)))
    shift = jnp.arange(transform_size, dtype=jnp.int64) - (L - 1)
    kernel = jnp.exp(-1j * _chirp_angle(shift, two_nphi, False))
    convolution = jnp.fft.ifft(
        jnp.fft.fft(embedded, axis=-1) * jnp.fft.fft(kernel, axis=-1), axis=-1
    )

    p_index = jnp.arange(width, dtype=jnp.int64)
    positive_half = jnp.exp(1j * _chirp_angle(p_index, two_nphi, False)) * convolution[
        :, L - 1 : L - 1 + width
    ]
    result = 2.0 * positive_half - positive[:, :1]

    source, used = _ring_inverse_layout(L, nside)
    slots = jnp.reshape(result.real, (-1,))
    return jnp.where(used, slots[source], 0.0)


@partial(jax.jit, static_argnames=("L", "nside"))
def _forward_ring_fft_full(signal, *, L, nside):
    """Complex ring FFT returning the full centered window m in [-(L-1), L)."""
    nphi, _, gather, valid, width = _ring_czt_constants(L, nside)
    pixels = jnp.reshape(jnp.asarray(signal), (-1,))
    two_nphi = (2 * nphi)[:, None]
    rows = jnp.where(valid, pixels[gather], 0.0)
    transform_size = _next_fast_len_pow2(width + 2 * L)

    n_index = jnp.arange(width, dtype=jnp.int64)
    embedded = rows * jnp.exp(-1j * _chirp_angle(n_index, two_nphi, False))
    embedded = jnp.pad(embedded, ((0, 0), (0, transform_size - width)))
    shift = jnp.arange(transform_size, dtype=jnp.int64) - (width - 1) - (L - 1)
    kernel = jnp.exp(1j * _chirp_angle(shift, two_nphi, False))
    convolution = jnp.fft.ifft(
        jnp.fft.fft(embedded, axis=-1) * jnp.fft.fft(kernel, axis=-1), axis=-1
    )

    m_index = jnp.arange(-(L - 1), L, dtype=jnp.int64)
    positive = jnp.exp(-1j * _chirp_angle(m_index, two_nphi, False)) * (
        convolution[:, width - 1 : width - 1 + m_index.shape[0]]
    )
    return positive


@partial(jax.jit, static_argnames=("L", "nside"))
def _inverse_ring_fft_complex(centered, *, L, nside):
    """Complex inverse ring FFT of a centered (ntheta, 2L-1) spectrum."""
    nphi, _, _, _, width = _ring_czt_constants(L, nside)
    grid = jnp.asarray(centered)
    ntheta = grid.shape[0]
    two_nphi = (2 * nphi)[:, None]
    transform_size = _next_fast_len_pow2(2 * L - 1 + width)

    c_index = jnp.arange(2 * L - 1, dtype=jnp.int64)
    embedded = grid * jnp.exp(1j * _chirp_angle(c_index, two_nphi, False))
    embedded = jnp.pad(embedded, ((0, 0), (0, transform_size - (2 * L - 1))))
    shift = jnp.arange(transform_size, dtype=jnp.int64) - (2 * L - 2)
    kernel = jnp.exp(-1j * _chirp_angle(shift, two_nphi, False))
    convolution = jnp.fft.ifft(
        jnp.fft.fft(embedded, axis=-1) * jnp.fft.fft(kernel, axis=-1), axis=-1
    )

    p_index = jnp.arange(width, dtype=jnp.int64)
    result = jnp.exp(1j * _chirp_angle(p_index, two_nphi, False)) * convolution[
        :, 2 * L - 2 : 2 * L - 2 + width
    ]
    wrap_phase = (((L - 1) % nphi)[:, None] * p_index[None, :]) % nphi[:, None]
    result *= jnp.exp(-1j * wrap_phase * (2 * jnp.pi / nphi)[:, None])

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


@partial(jax.jit, static_argnames=("L", "nside", "reality"))
def _forward_healpix_fft(maps, *, L, nside, reality):
    return _forward_ring_fft(maps, L=L, nside=nside)


@partial(jax.jit, static_argnames=("L", "nside"))
def _finish_inverse_pallas(ftm_positive, *, L, nside):
    return _inverse_ring_fft_herm(ftm_positive, L=L, nside=nside)


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
    ftm = _forward_s2fft_ftm(maps, L=L, nside=nside, reality=reality)
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
        ftm, L=L, spin=spin, nside=nside, reality=reality
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


@partial(
    jax.jit,
    static_argnames=("spin", "nside", "L", "L_work"),
)
def _alm2map_core(alm, ell, order, *, spin, nside, L, L_work):
    if spin == 0:
        elm = _unpack_real(alm[0], L, L_work)
        maps = _inverse_s2fft(elm, L=L_work, spin=0, nside=nside, reality=True)
        return jnp.real(maps)[None, :]
    maps = _inverse_s2fft(
        _unpack_spin(alm, L, L_work),
        L=L_work,
        spin=spin,
        nside=nside,
        reality=False,
    )
    return jnp.stack([jnp.real(maps), jnp.imag(maps)])


@partial(
    jax.jit,
    static_argnames=("spin", "nside", "L", "L_work"),
)
def _map2alm_once(maps, ell, order, *, spin, nside, L, L_work):
    if spin == 0:
        flm = _forward_s2fft(maps[0], L=L_work, spin=0, nside=nside, reality=True)
        return flm[ell, L_work - 1 + order][None, :]

    plus = _forward_s2fft(
        maps[0] + 1j * maps[1],
        L=L_work,
        spin=spin,
        nside=nside,
        reality=False,
    )
    plus_m = plus[ell, L_work - 1 + order]
    minus_m = (-1) ** order * jnp.conj(plus[ell, L_work - 1 - order])
    return jnp.stack([-(plus_m + minus_m) / 2, 0.5j * (plus_m - minus_m)])


@partial(jax.jit, static_argnames=("spin", "nside", "L", "L_work"))
def _map2alm_iteration(alm, maps, ell, order, *, spin, nside, L, L_work):
    residual = (
        _alm2map_core(
            alm, ell, order, spin=spin, nside=nside, L=L, L_work=L_work
        )
        - maps
    )
    return alm - _map2alm_once(
        residual, ell, order, spin=spin, nside=nside, L=L, L_work=L_work
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


def _map2alm_once_slab(maps, ell, order, *, spin, nside, L, L_work, slab):
    """Polarised analysis with the latitudinal step replaced by a slab contraction."""
    ftm = _forward_s2fft_ftm(
        maps[0] + 1j * maps[1], L=L_work, nside=nside, reality=False
    )
    plus = _finish_forward_s2fft(
        _forward_latitudinal_slab(ftm, slab, L=L_work),
        L=L_work,
        spin=spin,
        reality=False,
    )
    plus_m = plus[ell, L_work - 1 + order]
    minus_m = (-1) ** order * jnp.conj(plus[ell, L_work - 1 - order])
    return jnp.stack([-(plus_m + minus_m) / 2, 0.5j * (plus_m - minus_m)])


def _alm2map_core_slab(alm, *, spin, nside, L, L_work, slab):
    flm = _prepare_inverse_s2fft(_unpack_spin(alm, L, L_work), L=L_work)
    ftm = _inverse_latitudinal_slab(flm, slab, L=L_work)
    maps = _finish_inverse_s2fft(
        ftm, L=L_work, spin=spin, nside=nside, reality=False
    )
    return jnp.stack([jnp.real(maps), jnp.imag(maps)])


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
    alm = _map2alm_once_slab(maps, ell, order, spin=spin, nside=nside, L=L,
                             L_work=L_work, slab=analysis_slab)
    for _ in range(n_iter):
        alm = _map2alm_iteration_slab(
            alm, maps, ell, order, spin=spin, nside=nside, L=L, L_work=L_work,
            analysis_slab=analysis_slab, synthesis_slab=synthesis_slab,
        )
    return alm


def _map2alm_core(maps, ell, order, *, spin, nside, L, L_work, n_iter):
    """Run Jacobi refinement without retaining every iteration in one XLA graph."""
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
        def to_map(centered):
            full = jnp.concatenate(
                (jnp.zeros((centered.shape[0], 1), dtype=centered.dtype), centered),
                axis=1,
            )
            full = full.at[:, 1:].multiply(shifts)
            return _inverse_ring_fft_complex(full[:, 1:], L=L_work, nside=nside)
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
    core = (_alm2map_core_pallas_traced if L_work <= _PALLAS_TRACED_MAX_L
            else _alm2map_core_pallas_eager)
    return core(alm, nside=nside, L=L, L_work=L_work, spin=spin)


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
    rings_p = _forward_ring_fft_full(signal_plus, L=L_work, nside=nside)
    rings_m = _forward_ring_fft_full(signal_minus, L=L_work, nside=nside)

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
    ftm = _forward_ring_fft_positive(maps[0], L=L_work, nside=nside)
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

    Tracing the refinement loop as one program removes the per-primitive host
    dispatch, which at small bandlimit is the whole cost (Nside 32-64 are
    launch-bound: the device idles between kernels).  Above the crossover the
    monolithic program is the slower of the two: Nside 128 goes 15 -> 18 ms and
    Nside 256 58 -> 82 ms, because one large program holds every intermediate
    live at once and loses the allocator's reuse between boundaries.
    """
    core = (_map2alm_core_pallas_traced if L_work <= _PALLAS_TRACED_MAX_L
            else _map2alm_core_pallas_eager)
    return core(maps, ell, order, nside=nside, L=L, L_work=L_work,
                n_iter=n_iter, spin=spin)


def _map2alm_core_pallas_eager(
    maps, ell, order, *, nside, L, L_work, n_iter, spin=0
):
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
