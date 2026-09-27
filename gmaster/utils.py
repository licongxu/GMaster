"""Map and harmonic-space utilities (the counterpart of ``pymaster.utils``).

Geometry
    `NmtMapInfo` (HEALPix or CAR pixelisation) and `NmtAlmInfo` (Healpy-packed
    ``a_lm`` layout).

Transforms
    `map2alm` / `alm2map` for spin-0 and spin-s fields.  HEALPix maps use the GPU
    transforms in `gmaster._sht`; CAR maps and arbitrary positions use the direct
    sums `catalog2alm` / `alm2catalog`.

Masks and simulations
    `mask_apodization` (C1, C2, Smooth), `mask_apodization_flat`,
    `synfast_spherical`, `synfast_flat`, and `moore_penrose_pinvh`.

Settings
    Re-exported from `gmaster._config`: `nmt_params`, `set_n_iter_default`,
    `set_tol_pinv_default`, `set_sht_calculator`, and the GMaster-specific
    `set_table_precision`, `set_ring_precision` and `set_latitudinal_method`.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer
from s2fft.recursions import turok_jax

from ._config import (
    NmtParams,
    get_default_params,
    latitudinal_method,
    nmt_params,
    ring_dtype,
    set_latitudinal_method,
    set_n_iter_default,
    set_ring_precision,
    set_sht_calculator,
    set_table_precision,
    set_tol_pinv_default,
    table_dtype,
)
from ._sht import healpix as _healpix

__all__ = [
    # geometry
    "NmtAlmInfo", "NmtMapInfo",
    # transforms
    "alm2catalog", "alm2map", "catalog2alm", "map2alm", "map2alm_pair",
    # masks, simulations, linear algebra
    "mask_apodization", "mask_apodization_flat", "moore_penrose_pinvh",
    "synfast_flat", "synfast_spherical",
    # settings (defined in gmaster._config)
    "NmtParams", "get_default_params", "latitudinal_method", "nmt_params", "ring_dtype",
    "set_latitudinal_method", "set_n_iter_default", "set_ring_precision", "set_sht_calculator",
    "set_table_precision", "set_tol_pinv_default", "table_dtype",
]


def mask_apodization(mask_in, aposize, apotype="C1"):
    """Apodize a HEALPix mask.

    A pixel counts as masked where the mask is zero (or negative).  With ``theta`` the
    angular distance from a pixel to the nearest masked pixel, ``theta_*`` the
    apodization scale and ``x = sqrt((1 - cos theta) / (1 - cos theta_*))``, each pixel
    is multiplied by

    - ``"C1"``: ``x - sin(2 pi x) / (2 pi)`` for ``x < 1``, else 1;
    - ``"C2"``: ``(1 - cos(pi x)) / 2`` for ``x < 1``, else 1;
    - ``"Smooth"``: all pixels within ``2.5 theta_*`` of a masked pixel are zeroed, the
      result is smoothed with a Gaussian of standard deviation ``theta_*``, and then
      multiplied by the original mask.

    As in NaMaster, the "C1"/"C2" names are swapped relative to Grain et al. (2009).

    Parameters
    ----------
    mask_in : array_like, shape (npix,)
        HEALPix mask in RING ordering.
    aposize : float
        Apodization scale ``theta_*`` in degrees.
    apotype : {"C1", "C2", "Smooth"}, optional
        Apodization type.

    Returns
    -------
    ndarray or jax.Array, shape (npix,)
        Apodized mask (a NumPy array for "C1"/"C2", a JAX array for "Smooth").
    """
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
    """Apodize a flat-sky (rectangular) mask.

    Same apodization types as `mask_apodization`, with Euclidean distances on the
    flat patch.

    Parameters
    ----------
    mask_in : array_like, shape (ny, nx)
        Input mask.
    lx, ly : float
        Patch size along x and y, in radians.
    aposize : float
        Apodization scale in degrees.
    apotype : {"C1", "C2", "Smooth"}, optional
        Apodization type.

    Returns
    -------
    ndarray or jax.Array, shape (ny, nx)
        Apodized mask.
    """
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
    """HEALPix ring geometry: colatitudes, first-pixel azimuths, pixels per ring, weights."""

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
    """Clenshaw-Curtis quadrature weights for `size` equispaced colatitudes on [0, pi]."""
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
    """CAR ring geometry, with the pixel positions used by the direct transforms."""

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
    """Description of a curved-sky map pixelization (HEALPix or CAR).

    Instances can be compared with ``==`` to check that two fields share a
    pixelization.

    Parameters
    ----------
    wcs : astropy.wcs.WCS or None
        WCS of a CAR map.  If None, HEALPix is assumed and `axes` must be a
        one-element sequence holding the number of pixels.
    axes : sequence of int
        Map shape: ``(npix,)`` for HEALPix, ``(ny, nx)`` for CAR.
    """

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
        """Flatten maps to ``(..., npix)`` in the internal pixel order.

        CAR maps are flipped so that colatitude and azimuth increase along the axes;
        HEALPix maps are returned unchanged.
        """
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
        """Return the maximum multipole supported by the pixelization.

        ``3 * nside - 1`` for HEALPix, ``pi / min(d_theta, d_phi)`` for CAR.
        """
        if self.is_healpix:
            return 3 * self.nside - 1
        return int(np.pi / min(self.d_theta, self.d_phi))


# Cache of the per-coefficient `ell` / `m` index arrays, keyed by lmax.  They depend only on
# lmax, but rebuilding them (lmax + 1 numpy.arange calls and a host-to-device copy) dominated
# the cost of a small `NmtField`.  The budget counts elements because a single pair is ~38M
# elements at Nside 2048.
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
        # Do not cache under a JAX trace: a tracer stored in a module-level cache
        # would later raise UnexpectedTracerError.
        return cached
    if _ALM_INDEX_ELEMENTS + 2 * len(ell) > _ALM_INDEX_BUDGET:
        _ALM_INDEX_CACHE.clear()
        _ALM_INDEX_ELEMENTS = 0
    _ALM_INDEX_CACHE[lmax] = cached
    _ALM_INDEX_ELEMENTS += 2 * len(ell)
    return cached


class NmtAlmInfo:
    """Layout of healpy-packed spherical-harmonic coefficients.

    Coefficients are stored m-major for ``0 <= m <= l <= lmax`` (``mmax = lmax``),
    as in healpy.

    Parameters
    ----------
    lmax : int
        Maximum multipole.

    Attributes
    ----------
    lmax, mmax : int
        Maximum multipole and azimuthal order.
    mstart : ndarray of uint64, shape (mmax + 1,)
        Offset of the first coefficient of each m, so that ``a_lm`` sits at
        ``mstart[m] + l``.
    nelem : int
        Number of coefficients, ``(lmax + 1) (lmax + 2) / 2``.
    """

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
    """Moore-Penrose pseudo-inverse of a Hermitian matrix.

    The matrix is diagonalised and the inverse of every eigenvalue smaller than
    ``tol_pinv`` times the largest one is set to zero.

    Parameters
    ----------
    mat : array_like, shape (n, n)
        Hermitian matrix.
    tol_pinv : float or None
        Relative eigenvalue threshold.  If None or ``<= 0``, the ordinary inverse is
        returned.

    Returns
    -------
    jax.Array, shape (n, n)
        Pseudo-inverse of `mat`.
    """
    matrix = jnp.asarray(mat)
    if tol_pinv is None or tol_pinv <= 0:
        return jnp.linalg.inv(matrix)
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    inverse = jnp.where(
        eigenvalues >= tol_pinv * jnp.max(eigenvalues), 1 / eigenvalues, 0
    )
    return (eigenvectors * inverse) @ jnp.conj(eigenvectors.T)


def map2alm(map, spin, map_info, alm_info, *, n_iter):
    """Spherical-harmonic analysis of a spin-0 or spin-s map.

    For spin ``s > 0`` the two input components are (Q, U)-like and the output rows are
    the E and B coefficients, following NaMaster's conventions.

    Parameters
    ----------
    map : array_like, shape (nmaps, npix)
        Input map(s); ``nmaps`` is 1 for spin 0 and 2 otherwise.  CAR maps must already
        be flattened with `NmtMapInfo.reform_map`.
    spin : int
        Spin of the field.
    map_info : NmtMapInfo
        Pixelization of the map.
    alm_info : NmtAlmInfo
        Layout of the output coefficients.
    n_iter : int
        Number of Jacobi iterations used to refine the transform (keyword only).

    Returns
    -------
    jax.Array, shape (nmaps, alm_info.nelem)
        Healpy-packed coefficients.
    """
    maps = jnp.asarray(map)
    nmaps = 1 if spin == 0 else 2
    if maps.ndim != 2 or maps.shape != (nmaps, map_info.npix):
        raise ValueError("Input map has wrong shape")
    if spin < 0 or spin > alm_info.lmax:
        raise ValueError("spin must satisfy 0 <= spin <= lmax")
    if not map_info.is_healpix:
        # CAR maps use the direct sum, which costs O(npix lmax^2); adequate for
        # moderate resolutions.  Large CAR maps would need a separable ring transform.
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
    return _healpix.map2alm(maps, spin, map_info, alm_info, n_iter=n_iter)


def map2alm_pair(map_a, map_b, map_info, alm_info, *, n_iter):
    """Analyse two spin-0 HEALPix maps in one shared latitudinal pass, if possible.

    The latitudinal stage dominates the cost of a transform, so two maps on the same
    grid (for example a field and its mask) can share it: the precomputed Legendre band
    is read once, or the on-the-fly Wigner-d recurrence is generated once for both.
    The pair costs roughly 0.5-0.75 of two separate calls.  The band route is
    bit-identical to two `map2alm` calls; the march route agrees to float32-class
    accuracy (the summation order changes).

    Parameters
    ----------
    map_a, map_b : array_like, shape (1, npix)
        Spin-0 HEALPix maps.
    map_info : NmtMapInfo
        Pixelization shared by both maps.
    alm_info : NmtAlmInfo
        Layout of the output coefficients.
    n_iter : int
        Number of Jacobi iterations (keyword only).

    Returns
    -------
    tuple of jax.Array or None
        ``(alm_a, alm_b)``, each of shape ``(1, alm_info.nelem)``, or None when this
        geometry has no shared route (the caller should then make two `map2alm` calls).
    """
    maps_a = jnp.asarray(map_a)
    maps_b = jnp.asarray(map_b)
    if (maps_a.ndim != 2 or maps_a.shape != (1, map_info.npix)
            or maps_b.shape != (1, map_info.npix)):
        raise ValueError("shared-band pair expects two (1, npix) spin-0 maps")
    if not map_info.is_healpix:
        return None
    return _healpix.map2alm_pair(maps_a, maps_b, map_info, alm_info, n_iter=n_iter)


def alm2map(alm, spin, map_info, alm_info):
    """Spherical-harmonic synthesis of a spin-0 or spin-s field.

    Parameters
    ----------
    alm : array_like, shape (nmaps, alm_info.nelem)
        Healpy-packed coefficients (E and B for spin ``s > 0``).
    spin : int
        Spin of the field.
    map_info : NmtMapInfo
        Output pixelization.
    alm_info : NmtAlmInfo
        Layout of `alm`.

    Returns
    -------
    jax.Array, shape (nmaps, npix)
        Map(s); CAR maps are returned flattened in the internal pixel order.
    """
    alm = jnp.asarray(alm)
    nmaps = 1 if spin == 0 else 2
    if alm.ndim != 2 or alm.shape != (nmaps, alm_info.nelem):
        raise ValueError("Input alm has wrong shape")
    if spin < 0 or spin > alm_info.lmax:
        raise ValueError("spin must satisfy 0 <= spin <= lmax")
    if not map_info.is_healpix:
        return alm2catalog(alm, map_info.si.positions, spin, alm_info.lmax)
    return _healpix.alm2map(alm, spin, map_info, alm_info)


@partial(jax.jit, static_argnames=("spin", "lmax"))
def _catalog2alm_core(values, positions, *, spin, lmax):
    """Adjoint spin-s transform from arbitrary positions, in healpy alm ordering."""
    L = lmax + 1
    theta, phi = positions
    signal = values[0] if spin == 0 else values[0] + 1j * values[1]
    full = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
    # Direct sum, O(n_source lmax^2); `gmaster.nusht` provides fast non-uniform
    # transforms for large catalogues.
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
    """Direct adjoint spherical-harmonic transform of values at arbitrary positions.

    Computes ``a_lm = sum_i v_i Y*_lm(theta_i, phi_i)`` (spin-weighted for ``spin > 0``,
    returning E/B), with no quadrature weight beyond what `values` already carries.

    Parameters
    ----------
    values : array_like, shape (nmaps, n_source)
        Values (already multiplied by any weights); ``nmaps`` is 1 for spin 0, else 2.
    positions : array_like, shape (2, n_source)
        Colatitude ``theta`` and azimuth ``phi`` of each source, in radians.
    spin : int
        Spin of the field.
    lmax : int
        Maximum multipole.

    Returns
    -------
    jax.Array, shape (nmaps, (lmax + 1) (lmax + 2) / 2)
        Healpy-packed coefficients.
    """
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
    """Spin-s synthesis at arbitrary positions from healpy-ordered alms."""
    L = lmax + 1
    theta, phi = positions
    full = (
        _healpix._unpack_real(alms[0], L, L)
        if spin == 0
        else _healpix._unpack_spin(alms, L, L)
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
    """Evaluate a spherical-harmonic expansion at arbitrary positions (direct sum).

    Parameters
    ----------
    alms : array_like, shape (nmaps, (lmax + 1) (lmax + 2) / 2)
        Healpy-packed coefficients (E and B for ``spin > 0``).
    positions : array_like, shape (2, n_source)
        Colatitude and azimuth of each position, in radians.
    spin : int
        Spin of the field.
    lmax : int
        Maximum multipole.

    Returns
    -------
    jax.Array, shape (nmaps, n_source)
        Field values at the positions.
    """
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
    """Draw correlated Gaussian alms given a per-ell square root of the covariance."""
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
    """Generate full-sky correlated Gaussian random fields.

    Produces outputs statistically equivalent to healpy's ``synfast``, generated on the
    default JAX device.  Random numbers come from JAX, so a given `seed` does not
    reproduce pymaster's maps.

    Parameters
    ----------
    nside : int
        HEALPix resolution.  Ignored when `wcs` is given.
    cls : array_like, shape (n_cls, n_ell)
        Power spectra of all map pairs, ``n_cls = nmaps (nmaps + 1) / 2`` with ``nmaps``
        counting 1 map per spin-0 field and 2 per spin-s field.  Only the upper triangle
        of the spectrum matrix is given, in row-major order (for 3 maps:
        ``[11, 12, 13, 22, 23, 33]``).
    spin_arr : array_like of int, shape (nfields,)
        Spin of each field.
    beam : array_like, shape (nfields, n_ell), optional
        Beam window function of each field, applied to the harmonic coefficients.
    seed : int, optional
        Random seed; a negative value draws a random seed.
    wcs : astropy.wcs.WCS, optional
        WCS of a full-sky CAR map, used instead of HEALPix.
    lmax : int, optional
        Maximum multipole of the generated coefficients.  Defaults to the maximum
        supported by the pixelization (and never exceeds ``n_ell - 1``).

    Returns
    -------
    jax.Array
        Maps of shape ``(nmaps, npix)`` for HEALPix or ``(nmaps, ny, nx)`` for CAR.
    """
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
    """Generate correlated Gaussian random fields on a flat rectangular patch.

    Parameters
    ----------
    nx, ny : int
        Number of pixels along x and y.
    lx, ly : float
        Patch size along x and y, in radians.
    cls : array_like, shape (n_cls, n_ell)
        Power spectra, ordered as in `synfast_spherical`, sampled at integer multipoles
        ``0 .. n_ell - 1`` and interpolated to the 2-D wavevectors.
    spin_arr : array_like of int, shape (nfields,)
        Spin of each field.
    beam : array_like, shape (nfields, n_ell), optional
        Beam window function of each field.
    seed : int, optional
        Random seed; a negative value draws a random seed.

    Returns
    -------
    jax.Array, shape (nmaps, ny, nx)
        Simulated maps (1 per spin-0 field, 2 per spin-s field).
    """
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
