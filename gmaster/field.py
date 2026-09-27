"""Curved-sky fields (`NmtField`), the GPU counterpart of ``pymaster.NmtField``."""

import jax
import jax.numpy as jnp
import numpy as np

from . import _config
from ._sht import healpix as _healpix
from .utils import (
    NmtAlmInfo,
    NmtMapInfo,
    alm2map,
    map2alm,
    map2alm_pair,
    moore_penrose_pinvh,
    nmt_params,
)


class NmtField:
    """A masked curved-sky field: mask, maps, contaminant templates and their alms.

    Same interface and conventions as ``pymaster.NmtField``.  Maps, masks and
    coefficients are stored as JAX arrays on the GPU, and all spherical-harmonic
    transforms run there.  Supports HEALPix and CAR maps, spin-0 and spin-s fields,
    linear contaminant deprojection, E/B purification (spin 2) and anisotropic masks.

    Parameters
    ----------
    mask : array_like, shape (npix,) or (ny, nx)
        Mask (weight map), HEALPix in RING ordering or CAR.  With `mask_22` it is the
        11 component of an anisotropic weight matrix.
    maps : array_like, shape (nmaps, npix) or (nmaps, ny, nx), or None
        Observed maps: 1 for spin 0, 2 (e.g. Q/U or gamma_1/gamma_2, HEALPix
        polarisation convention) for spin s > 0.  If None, the field holds only a mask
        and can be used to compute coupling matrices but not power spectra.
    spin : int, optional
        Spin of the field.  Defaults to 0 for one map and 2 for two maps.
    templates : array_like, shape (ntemp, nmaps, npix) or (ntemp, nmaps, ny, nx), optional
        Contaminant templates.  Their best-fit contribution is subtracted from the maps.
    beam : array_like, shape (>= lmax + 1,), optional
        Harmonic transform of the (azimuthally symmetric) beam.  No pixel window is
        applied automatically.
    purify_e, purify_b : bool, optional
        Purify E or B modes (spin 2 only).
    n_iter : int, optional
        Jacobi iterations for the map transforms.  Defaults to the global setting
        (`set_n_iter_default`).
    n_iter_mask : int, optional
        Jacobi iterations for the mask transform.  Defaults to the global setting.
    tol_pinv : float, optional
        Relative eigenvalue threshold when inverting the template covariance; see
        `moore_penrose_pinvh`.  Defaults to the global setting.
    wcs : astropy.wcs.WCS, optional
        WCS of CAR maps.  If None, HEALPix is assumed.
    lmax : int, optional
        Maximum multipole of the field's alms.  Defaults to the maximum supported by
        the pixelization (``3 * nside - 1`` for HEALPix).
    lmax_mask : int, optional
        Maximum multipole of the mask's alms.  Same default as `lmax`.
    masked_on_input : bool, optional
        True if the maps and templates are already multiplied by the mask (not
        advisable with purification).
    lite : bool, optional
        Keep only what is needed for the power spectrum itself (no maps, templates or
        cached mask alms), saving memory; deprojection bias cannot then be computed.
    mask_22, mask_12 : array_like, optional
        The 22 and 12 components of an anisotropic weight matrix (spin > 0 only).
        Must be given together.

    Notes
    -----
    For scalar HEALPix fields the mask alms are computed together with the map alms
    (unless ``lite=True``), because the two transforms can share a single latitudinal
    pass; `get_mask_alms` then returns them without further work.
    """

    def __init__(
        self,
        mask,
        maps,
        *,
        spin=None,
        templates=None,
        beam=None,
        purify_e=False,
        purify_b=False,
        n_iter=None,
        n_iter_mask=None,
        tol_pinv=None,
        wcs=None,
        lmax=None,
        lmax_mask=None,
        masked_on_input=False,
        lite=False,
        mask_22=None,
        mask_12=None,
    ):
        self.lite = bool(lite)
        self.n_iter = nmt_params.n_iter_default if n_iter is None else int(n_iter)
        self.n_iter_mask = (
            nmt_params.n_iter_mask_default if n_iter_mask is None else int(n_iter_mask)
        )
        self.pure_e = bool(purify_e)
        self.pure_b = bool(purify_b)
        self.is_catalog = False
        self.anisotropic_mask = False
        self.mask_a = self.alm_mask_a = None
        self.alm = self.alm_mask = self.alm_temp = None
        self.maps = self.temp = None
        self.n_temp = 0
        self._Nw = self._Nf = 0
        self._alpha = None

        mask = jnp.asarray(mask, dtype=jnp.float64)
        if (mask_22 is None) != (mask_12 is None):
            raise ValueError("Both mask_22 and mask_12 must be passed together")
        if mask_22 is not None:
            mask_22 = jnp.asarray(mask_22, dtype=jnp.float64)
            mask_12 = jnp.asarray(mask_12, dtype=jnp.float64)
            if mask_22.shape != mask.shape or mask_12.shape != mask.shape:
                raise ValueError("All anisotropic mask components must have the same shape")
            mask_scale = jnp.mean(mask + mask_22)
            if bool(jnp.any(mask * mask_22 - mask_12**2 < -1e-5 * mask_scale)):
                raise ValueError("The anisotropic mask does not seem positive-definite")
            mask0 = 0.5 * (mask + mask_22)
            mask_a = np.stack([0.5 * (mask - mask_22), mask_12])
            mask = mask0
            self.anisotropic_mask = True

        self.minfo = NmtMapInfo(wcs, mask.shape)
        self.mask = self.minfo.reform_map(mask)
        if self.anisotropic_mask:
            self.mask_a = self.minfo.reform_map(jnp.asarray(mask_a))
        lmax = self.minfo.get_lmax() if lmax is None else int(lmax)
        lmax_mask = self.minfo.get_lmax() if lmax_mask is None else int(lmax_mask)
        if lmax <= 0 or lmax_mask <= 0:
            raise ValueError("`lmax` and `lmax_mask` must be positive")
        self.ainfo = NmtAlmInfo(lmax)
        self.ainfo_mask = NmtAlmInfo(lmax_mask)

        if beam is None:
            self.beam = jnp.ones(lmax + 1)
        elif isinstance(beam, (list, tuple, np.ndarray, jax.Array)):
            if len(beam) <= lmax:
                raise ValueError(
                    f"Input beam must have at least {lmax + 1} elements "
                    "given the input map resolution"
                )
            self.beam = jnp.asarray(beam)
        else:
            raise ValueError("Input beam can only be an array or None")

        if maps is None:
            if spin is None:
                raise ValueError("Please supply field spin")
            self.spin = int(spin)
            if self.spin == 0 and self.anisotropic_mask:
                raise ValueError("Anisotropic masks can't be used with scalar fields")
            self.nmaps = 1 if self.spin == 0 else 2
            return

        maps = jnp.asarray(maps, dtype=jnp.float64)
        if len(maps) not in (1, 2):
            raise ValueError("Must supply 1 or 2 maps per field")
        maps = self.minfo.reform_map(maps)
        if maps.ndim != 2:
            raise ValueError("Must supply 1 or 2 maps per field")
        if spin is None:
            spin = 0 if len(maps) == 1 else 2
        if (spin == 0) != (len(maps) == 1):
            raise ValueError("Spin-zero fields are associated with a single map")
        self.spin = int(spin)
        self.nmaps = 1 if self.spin == 0 else 2

        pure_any = self.pure_e or self.pure_b
        if pure_any and self.spin != 2:
            raise ValueError("Purification only implemented for spin-2 fields")
        if self.anisotropic_mask:
            if self.spin == 0:
                raise ValueError("Anisotropic masks can't be used with scalar fields")
            if pure_any:
                raise NotImplementedError(
                    "Purification not implemented for anisotropic masks"
                )
            if templates is not None:
                raise NotImplementedError(
                    "Contaminant deprojection not supported for anisotropic masks"
                )
        if pure_any and self.ainfo != self.ainfo_mask:
            raise ValueError("Pure fields require lmax == lmax_mask")

        if templates is not None:
            if not isinstance(templates, (list, tuple, np.ndarray, jax.Array)):
                raise ValueError("Input templates can only be an array or None")
            templates = jnp.asarray(templates, dtype=jnp.float64)
            templates = self.minfo.reform_map(templates)
            if templates.ndim != 3 or templates.shape[1:] != maps.shape:
                raise ValueError("Each template must have the same shape as the maps")
            self.n_temp = len(templates)

        maps_unmasked = templates_unmasked = None
        if pure_any:
            maps_unmasked = maps
            templates_unmasked = templates
            if masked_on_input:
                good = self.mask > 0
                denominator = jnp.where(good, self.mask, 1)
                maps_unmasked = jnp.where(
                    good[None, :], maps / denominator[None, :], maps
                )
                if templates is not None:
                    templates_unmasked = jnp.where(
                        good[None, None, :],
                        templates / denominator[None, None, :],
                        templates,
                    )

        if not masked_on_input:
            if self.anisotropic_mask:
                qmap, umap = maps
                mask1, mask2 = self.mask_a
                maps = maps * self.mask[None, :] + jnp.stack(
                    [qmap * mask1 + umap * mask2, qmap * mask2 - umap * mask1]
                )
            else:
                maps = maps * self.mask[None, :]
            if templates is not None:
                templates = templates * self.mask[None, None, :]

        if templates is not None:
            covariance = jnp.einsum(
                "iap,jap->ij", templates, self.minfo.si.times_weight(templates)
            )
            self.iM = moore_penrose_pinvh(
                covariance,
                nmt_params.tol_pinv_default if tol_pinv is None else tol_pinv,
            )
            products = jnp.einsum(
                "iap,ap->i", templates, self.minfo.si.times_weight(maps)
            )
            self.alphas = self.iM @ products
            maps = maps - jnp.einsum("i,iap->ap", self.alphas, templates)
            if pure_any:
                maps_unmasked = maps_unmasked - jnp.einsum(
                    "i,iap->ap", self.alphas, templates_unmasked
                )

        # The transforms of a large field need tens of GiB of temporaries (a polarised
        # Nside 4096 pass peaks near 40 GiB), so free stale device caches first.
        nside = getattr(self.minfo, "nside", None)
        if nside:
            _config.make_room(6 * (4 * nside - 1) * 2 * (self.ainfo.lmax + 1) * 16)
        # At Nside >= 8192 the mask transform (~47 GiB working set) does not fit next to a
        # polarised field's maps and alms, so compute it first, while only the maps are
        # resident.  Block until it finishes; otherwise asynchronous dispatch would overlap
        # its buffers with those of the spin-2 transform.
        if (nside and nside >= 8192 and self.spin and not self.lite
                and self.mask is not None):
            jax.block_until_ready(self.get_mask_alms())
        if pure_any:
            task = (self.pure_e, self.pure_b)
            self.alm, maps = self._purify(
                self.get_mask_alms(), maps_unmasked, task=task
            )
        else:
            # A scalar field and its mask are two spin-0 transforms with the same lmax
            # and n_iter, so they can share one latitudinal pass (`utils.map2alm_pair`),
            # costing roughly 0.5-0.7 of two separate transforms.  The mask alms are
            # stored in `alm_mask`, exactly what `get_mask_alms` would compute later.
            # `lite` fields keep the lazy route, since they retain no derived state.
            fused = None
            if (self.spin == 0 and not self.lite and self.mask is not None
                    and not self.anisotropic_mask
                    and self.ainfo == self.ainfo_mask
                    and self.n_iter == self.n_iter_mask
                    and not (nside and nside >= 8192)):
                fused = map2alm_pair(
                    maps, self.mask[None, :], self.minfo, self.ainfo,
                    n_iter=self.n_iter,
                )
            if fused is None:
                self.alm = map2alm(
                    maps, self.spin, self.minfo, self.ainfo, n_iter=self.n_iter
                )
            else:
                # `alm` has map2alm's (1, nelem) shape; `alm_mask` has the (nelem,)
                # shape that `get_mask_alms` returns.
                self.alm, self.alm_mask = fused[0], fused[1][0]
        if not self.lite:
            self.maps = maps
            if templates is not None:
                self.temp = templates
                if pure_any:
                    self.alm_temp = jnp.stack(
                        [
                            self._purify(
                                self.get_mask_alms(), template, task=task
                            )[0]
                            for template in templates_unmasked
                        ]
                    )
                else:
                    self.alm_temp = jnp.stack(
                        [
                            map2alm(
                                template,
                                self.spin,
                                self.minfo,
                                self.ainfo,
                                n_iter=self.n_iter,
                            )
                            for template in templates
                        ]
                    )

    def is_compatible(self, other, strict=True):
        """Check whether another field is compatible with this one.

        Parameters
        ----------
        other : NmtField
            Field to compare with.
        strict : bool, optional
            If True, compare both the pixelization and the harmonic resolution;
            otherwise only the harmonic resolution (enough for power spectra).

        Returns
        -------
        bool
        """
        if strict and self.minfo != other.minfo:
            return False
        return self.ainfo_mask == other.ainfo_mask and self.ainfo == other.ainfo

    def get_mask(self):
        """Return the field's mask, shape (npix,) (CAR maps flattened)."""
        if self.mask is None:
            raise ValueError("Input mask unavailable for this field")
        return self.mask

    def get_anisotropic_mask(self):
        """Return the anisotropic mask components, shape (2, npix)."""
        if self.mask_a is None:
            raise ValueError("Input anisotropic mask unavailable for this field")
        return self.mask_a

    def get_mask_alms(self):
        """Return the spherical-harmonic coefficients of the mask.

        Computed on first call if not already available, and stored unless the field
        is `lite`.

        Returns
        -------
        jax.Array, shape (nelem_mask,)
            Healpy-packed mask alms up to `lmax_mask`.
        """
        if self.alm_mask is None:
            # With several GPUs, run the mask transform on the second one so that it
            # does not compete for memory with the field's spin transform on the first.
            mask = self.mask[None, :]
            gpus = _healpix._gpu_devices()
            if len(gpus) > 1:
                mask = jax.device_put(mask, gpus[1])
            alms = map2alm(
                mask,
                0,
                self.minfo,
                self.ainfo_mask,
                n_iter=self.n_iter_mask,
            )[0]
            if not self.lite:
                self.alm_mask = alms
            return alms
        return self.alm_mask

    def get_anisotropic_mask_alms(self):
        """Return the alms of the anisotropic mask components (spin ``2 * spin``).

        Returns
        -------
        jax.Array, shape (2, nelem_mask)
        """
        if self.mask_a is None:
            raise ValueError("This field does not have an anisotropic mask")
        if self.alm_mask_a is None:
            alms = map2alm(
                self.mask_a,
                2 * self.spin,
                self.minfo,
                self.ainfo_mask,
                n_iter=self.n_iter_mask,
            )
            if not self.lite:
                self.alm_mask_a = alms
            return alms
        return self.alm_mask_a

    def get_maps(self):
        """Return the masked, deprojected (and purified, if requested) maps.

        Returns
        -------
        jax.Array, shape (nmaps, npix)
        """
        if self.maps is None:
            raise ValueError("Input maps unavailable for this field")
        return self.maps

    def get_alms(self):
        """Return the field's alms, including masking, deprojection and purification.

        Returns
        -------
        jax.Array, shape (nmaps, nelem)
        """
        if self.alm is None:
            raise ValueError("Mask-only fields have no alms")
        return self.alm

    def get_templates(self):
        """Return the masked contaminant templates.

        Returns
        -------
        jax.Array, shape (ntemp, nmaps, npix)
        """
        if self.temp is None:
            raise ValueError("Input templates unavailable for this field")
        return self.temp

    def _purify(
        self, alm_mask, maps_unmasked, *, task, n_iter=None, return_maps=True
    ):
        """Pure E/B alms of a spin-2 field (Smith 2006; Alonso et al. 2019, Sec. 2.4).

        Adds to the ordinary masked alms the counter-terms built from the spin-1 and
        spin-2 derivatives of the mask.  ``task = (purify_e, purify_b)`` selects which
        component receives them.  Returns the alms, and the corresponding maps if
        `return_maps`.
        """
        n_iter = self.n_iter if n_iter is None else int(n_iter)
        ell_mask = self.ainfo_mask._ell
        ell = self.ainfo._ell
        ls = jnp.arange(self.ainfo_mask.lmax + 1, dtype=self.mask.dtype)

        alms = map2alm(
            maps_unmasked * self.mask[None, :],
            2,
            self.minfo,
            self.ainfo_mask,
            n_iter=n_iter,
        )

        factors = -jnp.sqrt((ls + 1) * ls)
        walm = jnp.stack([alm_mask * factors[ell_mask], jnp.zeros_like(alm_mask)])
        wmap = alm2map(walm, 1, self.minfo, self.ainfo_mask)
        qmap, umap = maps_unmasked
        maps = jnp.stack(
            [wmap[0] * qmap + wmap[1] * umap, wmap[0] * umap - wmap[1] * qmap]
        )
        palm = map2alm(maps, 1, self.minfo, self.ainfo, n_iter=n_iter)
        factors = jnp.where(
            ls >= 2, 2 / jnp.sqrt((ls + 2) * (ls - 1)), 0
        )
        for ipol, purify in enumerate(task):
            if purify:
                alms = alms.at[ipol].add(palm[ipol] * factors[ell])

        factors = jnp.where(ls >= 2, -jnp.sqrt((ls + 2) * (ls - 1)), 0)
        walm = walm.at[0].set(walm[0] * factors[ell_mask])
        wmap = alm2map(walm, 2, self.minfo, self.ainfo_mask)
        maps = jnp.stack(
            [wmap[0] * qmap + wmap[1] * umap, wmap[0] * umap - wmap[1] * qmap]
        )
        palm = jnp.stack(
            [
                map2alm(
                    component[None, :],
                    0,
                    self.minfo,
                    self.ainfo,
                    n_iter=n_iter,
                )[0]
                for component in maps
            ]
        )
        factors = jnp.where(
            ls >= 2,
            1 / jnp.sqrt((ls + 2) * (ls + 1) * ls * (ls - 1)),
            0,
        )
        for ipol, purify in enumerate(task):
            if purify:
                alms = alms.at[ipol].add(palm[ipol] * factors[ell])
        if return_maps:
            return alms, alm2map(alms, 2, self.minfo, self.ainfo)
        return alms

    @property
    def Nw(self):
        """Shot-noise contribution of the mask for catalog-based fields (0 otherwise)."""
        return self._Nw

    @property
    def Nf(self):
        """Shot-noise contribution to the field power spectrum for catalog-based fields (0 otherwise)."""
        return self._Nf

    @property
    def alpha(self):
        """Ratio of data to random sources for clustering catalogs (None otherwise)."""
        return self._alpha
