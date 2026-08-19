"""JAX-backed flat-sky fields and Fourier transforms."""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from .utils import moore_penrose_pinvh, nmt_params


def _wavevectors(nx, ny, lx, ly):
    kx = 2 * jnp.pi * jnp.arange(nx // 2 + 1) / lx
    iy = jnp.arange(ny)
    ky = 2 * jnp.pi * jnp.where(2 * iy <= ny, iy, iy - ny) / ly
    return kx[None, :], ky[:, None]


def _rotation(nx, ny, lx, ly, spin, inverse=False):
    kx, ky = _wavevectors(nx, ny, lx, ly)
    phi = jnp.arctan2(ky, kx)
    angle = spin * phi
    sign = -((-1j) ** spin if inverse else (1j) ** spin)
    return sign, jnp.cos(angle), jnp.sin(angle)


@partial(jax.jit, static_argnames="spin")
def _flat_map2alm(maps, spin, lx, ly):
    """Transform maps using NaMaster's flat-sky Fourier normalization."""
    maps = jnp.asarray(maps)
    ny, nx = maps.shape[-2:]
    alms = jnp.fft.rfft2(maps) * (lx * ly / (2 * jnp.pi * nx * ny))
    if spin:
        sign, cosine, sine = _rotation(nx, ny, lx, ly, spin)
        qalm, ualm = alms
        alms = jnp.stack(
            [sign * (qalm * cosine - ualm * sine),
             sign * (qalm * sine + ualm * cosine)]
        )
    return alms


@partial(jax.jit, static_argnames=("spin", "nx"))
def _flat_alm2map(alms, spin, lx, ly, nx):
    """Inverse of :func:`_flat_map2alm`."""
    alms = jnp.asarray(alms)
    ny = alms.shape[-2]
    if spin:
        sign, cosine, sine = _rotation(nx, ny, lx, ly, spin, inverse=True)
        ealm, balm = alms
        alms = jnp.stack(
            [sign * (ealm * cosine + balm * sine),
             sign * (-ealm * sine + balm * cosine)]
        )
    return jnp.fft.irfft2(alms, s=(ny, nx)) * (nx * ny * 2 * jnp.pi / (lx * ly))


class NmtFieldFlat:
    """Masked flat-sky field compatible with NaMaster's public API."""

    def __init__(
        self,
        lx,
        ly,
        mask,
        maps,
        spin=None,
        templates=None,
        beam=None,
        purify_e=False,
        purify_b=False,
        tol_pinv=None,
        masked_on_input=False,
        lite=False,
    ):
        if lx < 0 or ly < 0:
            raise ValueError("Must supply sensible dimensions for flat-sky field")
        mask = jnp.asarray(mask, dtype=jnp.float64)
        if mask.ndim != 2:
            raise ValueError("Mask must be a 2-D array")

        self.lx = float(lx)
        self.ly = float(ly)
        self.ny, self.nx = mask.shape
        self.npix = self.nx * self.ny
        self.mask = mask
        self.pure_e = bool(purify_e)
        self.pure_b = bool(purify_b)
        self.lite = bool(lite)
        self.maps = self.temp = self.alm = self.alm_temp = self.alm_mask = None
        self.n_temp = 0
        self.alphas = self.iM = None

        if maps is None:
            if spin is None:
                raise ValueError("Please supply field spin")
            self.spin = int(spin)
            self.nmaps = 1 if self.spin == 0 else 2
            if (self.pure_e or self.pure_b) and self.spin != 2:
                raise ValueError("Purification only implemented for spin-2 fields")
            self.lite = True
            self._set_beam(beam)
            return

        maps = jnp.asarray(maps, dtype=jnp.float64)
        if maps.ndim != 3 or len(maps) not in (1, 2):
            raise ValueError("Must supply 1 or 2 maps per field")
        if maps.shape[1:] != mask.shape:
            raise ValueError("Mask and maps don't have the same shape")
        if spin is None:
            spin = 0 if len(maps) == 1 else 2
        if (spin == 0) != (len(maps) == 1):
            raise ValueError("Spin-zero fields are associated with a single map")
        self.spin = int(spin)
        self.nmaps = 1 if self.spin == 0 else 2
        if (self.pure_e or self.pure_b) and self.spin != 2:
            raise ValueError("Purification only implemented for spin-2 fields")
        self._set_beam(beam)

        if templates is not None:
            if not isinstance(templates, (list, tuple, np.ndarray, jax.Array)):
                raise ValueError("Input templates can only be an array or None")
            templates = jnp.asarray(templates, dtype=jnp.float64)
            if templates.ndim != 4 or templates.shape[1:] != maps.shape:
                raise ValueError("Each template must have the same shape as the maps")
            self.n_temp = len(templates)

        pure_any = self.pure_e or self.pure_b
        maps_unmasked = templates_unmasked = None
        if pure_any:
            maps_unmasked = maps
            templates_unmasked = templates
            if masked_on_input:
                good = mask > 0
                denominator = jnp.where(good, mask, 1)
                maps_unmasked = jnp.where(
                    good[None], maps / denominator[None], 0
                )
                if templates is not None:
                    templates_unmasked = jnp.where(
                        good[None, None],
                        templates / denominator[None, None],
                        0,
                    )

        if not masked_on_input:
            maps = maps * mask[None]
            if templates is not None:
                templates = templates * mask[None, None]

        if templates is not None:
            pixel_area = self.lx * self.ly / self.npix
            covariance = pixel_area * jnp.einsum("iayx,jayx->ij", templates, templates)
            self.iM = moore_penrose_pinvh(
                covariance,
                nmt_params.tol_pinv_default if tol_pinv is None else tol_pinv,
            )
            products = pixel_area * jnp.einsum("iayx,ayx->i", templates, maps)
            self.alphas = self.iM @ products
            maps = maps - jnp.einsum("i,iayx->ayx", self.alphas, templates)
            if pure_any:
                maps_unmasked = maps_unmasked - jnp.einsum(
                    "i,iayx->ayx", self.alphas, templates_unmasked
                )

        if pure_any:
            self.alm_mask = _flat_map2alm(mask[None], 0, self.lx, self.ly)[0]
            self.alm, maps = self._purify(maps_unmasked)
        else:
            self.alm = _flat_map2alm(maps, self.spin, self.lx, self.ly)

        if not self.lite:
            self.maps = maps
            if templates is not None:
                self.temp = templates
                if pure_any:
                    self.alm_temp = jnp.stack(
                        [self._purify(template)[0] for template in templates_unmasked]
                    )
                else:
                    self.alm_temp = jax.vmap(
                        lambda template: _flat_map2alm(
                            template, self.spin, self.lx, self.ly
                        )
                    )(templates)

    def _set_beam(self, beam):
        if beam is None:
            self.beam = None
            return
        if not isinstance(beam, (list, tuple, np.ndarray, jax.Array)):
            raise ValueError("Input beam can only be an array or None")
        beam = jnp.asarray(beam, dtype=jnp.float64)
        if beam.ndim != 2 or beam.shape[0] != 2:
            raise ValueError("Input beam must have shape (2, nl)")
        self.beam = beam

    def _purify(self, maps_unmasked):
        mask_alm = self.alm_mask
        ix = jnp.arange(self.nx // 2 + 1)
        iy = jnp.arange(self.ny)
        kx = 2 * jnp.pi * ix / self.nx
        ky = 2 * jnp.pi * jnp.where(2 * iy <= self.ny, iy, iy - self.ny) / self.ny
        k = jnp.sqrt(kx[None] ** 2 + ky[:, None] ** 2)
        alms = _flat_map2alm(
            maps_unmasked * self.mask[None], 2, self.lx, self.ly
        )

        walm = jnp.stack([-mask_alm * k, jnp.zeros_like(mask_alm)])
        wmap = _flat_alm2map(walm, 1, self.lx, self.ly, self.nx)
        qmap, umap = maps_unmasked
        products = jnp.stack(
            [wmap[0] * qmap + wmap[1] * umap,
             wmap[0] * umap - wmap[1] * qmap]
        )
        palm = _flat_map2alm(products, 1, self.lx, self.ly)
        factors = jnp.where(k > 0, 2 / k, 0)
        if self.pure_e:
            alms = alms.at[0].add(palm[0] * factors)
        if self.pure_b:
            alms = alms.at[1].add(palm[1] * factors)

        walm = walm.at[0].set(-mask_alm * k**2)
        wmap = _flat_alm2map(walm, 2, self.lx, self.ly, self.nx)
        products = -jnp.stack(
            [wmap[0] * qmap + wmap[1] * umap,
             wmap[0] * umap - wmap[1] * qmap]
        )
        palm = _flat_map2alm(products[:1], 0, self.lx, self.ly)
        palm = jnp.concatenate(
            [palm, _flat_map2alm(products[1:], 0, self.lx, self.ly)]
        )
        factors = jnp.where(k > 0, 1 / k**2, 0)
        if self.pure_e:
            alms = alms.at[0].add(palm[0] * factors)
        if self.pure_b:
            alms = alms.at[1].add(palm[1] * factors)
        return alms, _flat_alm2map(alms, 2, self.lx, self.ly, self.nx)

    def is_compatible(self, other):
        return (
            self.nx == other.nx
            and self.ny == other.ny
            and self.lx == other.lx
            and self.ly == other.ly
        )

    def get_mask(self):
        return self.mask

    def get_maps(self):
        if self.maps is None:
            raise ValueError(
                "Input maps unavailable for lightweight fields. To use this function, "
                "create an `NmtFieldFlat` object with `lite=False`."
            )
        return self.maps

    def get_alms(self):
        if self.alm is None:
            raise ValueError("Mask-only fields have no alms")
        return self.alm

    def get_templates(self):
        if self.temp is None:
            raise ValueError(
                "Input maps unavailable for lightweight fields. To use this function, "
                "create an `NmtFieldFlat` object with `lite=False`."
            )
        return self.temp

    def get_ell_sampling(self):
        dk = min(2 * np.pi / self.lx, 2 * np.pi / self.ly)
        kmax = np.hypot(2 * np.pi / self.lx * (self.nx // 2),
                        2 * np.pi / self.ly * (self.ny // 2))
        return (np.arange(int(np.floor(kmax / dk))) + 0.5) * dk
