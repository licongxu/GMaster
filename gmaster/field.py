"""NaMaster-compatible curved-sky fields."""

import jax
import jax.numpy as jnp
import numpy as np

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
    """Masked curved-sky field with device-resident maps and coefficients."""

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

        if pure_any:
            task = (self.pure_e, self.pure_b)
            self.alm, maps = self._purify(
                self.get_mask_alms(), maps_unmasked, task=task
            )
        else:
            # A scalar field and its mask are two independent spin-0 transforms at
            # the same order and the same `n_iter`, streaming the same Legendre
            # band; run together they cost 0.53-0.68x of the price of the two
            # passes (`utils.map2alm_pair`).  The mask alms land in `alm_mask`,
            # which is what `get_mask_alms` would have computed later, so the only
            # behaviour that changes is that they exist now: a field that never
            # reaches a coupling matrix pays for the second transform at ~10 % of
            # the first instead of 100 % of it.  `lite` fields keep the lazy route
            # because they asked not to retain derived state.
            fused = None
            if (self.spin == 0 and not self.lite and self.mask is not None
                    and not self.anisotropic_mask
                    and self.ainfo == self.ainfo_mask
                    and self.n_iter == self.n_iter_mask):
                fused = map2alm_pair(
                    maps, self.mask[None, :], self.minfo, self.ainfo,
                    n_iter=self.n_iter,
                )
            if fused is None:
                self.alm = map2alm(
                    maps, self.spin, self.minfo, self.ainfo, n_iter=self.n_iter
                )
            else:
                # `alm` keeps map2alm's (1, nelem) packing; `alm_mask` keeps
                # get_mask_alms' unpacked (nelem,) row, which is what the pair's
                # second element has to be sliced to for the two routes to be
                # interchangeable.
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
        if strict and self.minfo != other.minfo:
            return False
        return self.ainfo_mask == other.ainfo_mask and self.ainfo == other.ainfo

    def get_mask(self):
        if self.mask is None:
            raise ValueError("Input mask unavailable for this field")
        return self.mask

    def get_anisotropic_mask(self):
        if self.mask_a is None:
            raise ValueError("Input anisotropic mask unavailable for this field")
        return self.mask_a

    def get_mask_alms(self):
        if self.alm_mask is None:
            alms = map2alm(
                self.mask[None, :],
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
        if self.maps is None:
            raise ValueError("Input maps unavailable for this field")
        return self.maps

    def get_alms(self):
        if self.alm is None:
            raise ValueError("Mask-only fields have no alms")
        return self.alm

    def get_templates(self):
        if self.temp is None:
            raise ValueError("Input templates unavailable for this field")
        return self.temp

    def _purify(
        self, alm_mask, maps_unmasked, *, task, n_iter=None, return_maps=True
    ):
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
        return self._Nw

    @property
    def Nf(self):
        return self._Nf

    @property
    def alpha(self):
        return self._alpha
