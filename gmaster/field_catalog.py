"""JAX-backed fields sampled at arbitrary catalog positions."""

import jax.numpy as jnp
import numpy as np

from .field import NmtField
from .utils import (
    NmtAlmInfo,
    NmtMapInfo,
    alm2catalog,
    alm2map,
    catalog2alm,
    moore_penrose_pinvh,
    nmt_params,
)
from .workspaces import _compute_coupled_cell


def _process_positions(positions, weights, lonlat, kind):
    positions = np.asarray(positions, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if positions.shape != (2, len(weights)):
        raise ValueError(f"{kind} positions must have shape {(2, len(weights))}")
    if lonlat:
        positions = np.radians(positions[::-1])
        positions[0] = np.pi / 2 - positions[0]
    if np.any((positions[0] < 0) | (positions[0] > np.pi)):
        raise ValueError("Catalog colatitudes must lie between 0 and pi")
    if np.any((positions[1] < 0) | (positions[1] > 2 * np.pi)):
        raise ValueError("Catalog longitudes must lie between 0 and 2*pi")
    return jnp.asarray(positions), jnp.asarray(weights)


class NmtFieldCatalog(NmtField):
    """Field sampled at discrete spherical positions."""

    def __init__(
        self,
        positions,
        weights,
        field,
        lmax,
        lmax_mask=None,
        spin=None,
        field_is_weighted=False,
        lonlat=False,
        templates=None,
        tol_pinv=None,
        noise_variance=None,
        retain_catalog=False,
        n_iter_mask=None,
        nside_ipd=16,
    ):
        self.lite = not retain_catalog
        self.mask = self.maps = self.temp = self.alm_temp = None
        self.mask_a = self.alm_mask_a = None
        self.minfo = None
        self.beam = jnp.ones(lmax + 1)
        self.n_iter = None
        self.n_iter_mask = nmt_params.n_iter_mask_default if n_iter_mask is None else n_iter_mask
        self.pure_e = self.pure_b = self.anisotropic_mask = False
        self.is_catalog = True
        self.is_clustering = False
        self.ainfo = NmtAlmInfo(lmax)
        self.ainfo_mask = NmtAlmInfo(lmax if lmax_mask is None else lmax_mask)
        self.alm = None
        self.alm_mask = None
        self.n_temp = 0
        self._alpha = None
        self._nl_deproj = None
        self.nside_ipd = int(nside_ipd)
        self.theta_cloud = None
        self.pos = self.pos_r = self.weights = self.weights_r = self.field = None
        self.noise_variance = None

        positions, weights = _process_positions(positions, weights, lonlat, "source")
        self._Nw = jnp.sum(weights**2) / (4 * jnp.pi)
        self.alm_mask = catalog2alm(
            weights, positions, 0, self.ainfo_mask.lmax
        )[0]

        if field is None:
            if spin is None:
                raise ValueError("If field is None, spin needs to be provided")
            if retain_catalog:
                raise ValueError("If field is None, `retain_catalog` must be `False`")
            self.spin = int(spin)
            self.nmaps = 1 if self.spin == 0 else 2
            self._Nf = 0.0
            return

        field = jnp.atleast_2d(jnp.asarray(field, dtype=jnp.float64))
        if spin is None:
            spin = 0 if len(field) == 1 else 2
        self.spin = int(spin)
        self.nmaps = 1 if self.spin == 0 else 2
        expected = (self.nmaps, len(weights))
        if field.shape != expected:
            raise ValueError(f"Field should have shape {expected}")
        if not field_is_weighted:
            field = field * weights[None]

        if templates is not None:
            templates = jnp.asarray(templates, dtype=jnp.float64)
            if templates.ndim != 3 or templates.shape[1:] != expected:
                raise ValueError(
                    f"Templates should have shape (ntemp, {self.nmaps}, {len(weights)})"
                )
            if not field_is_weighted:
                templates = templates * weights[None, None]
            self.n_temp = len(templates)
            covariance = jnp.einsum("iap,jap->ij", templates, templates)
            self.iM = moore_penrose_pinvh(
                covariance,
                nmt_params.tol_pinv_default if tol_pinv is None else tol_pinv,
            )
            products = jnp.einsum("iap,ap->i", templates, field)
            self.alphas = self.iM @ products
            field = field - jnp.einsum("i,iap->ap", self.alphas, templates)

        self.alm = catalog2alm(field, positions, self.spin, lmax)
        self._Nf = jnp.sum(field**2) / (4 * jnp.pi * self.nmaps)
        if retain_catalog:
            self.pos = self.pos_r = positions
            self.weights = self.weights_r = weights
            self.field = field
            self.noise_variance = (
                None if noise_variance is None else jnp.asarray(noise_variance)
            )
            if templates is not None:
                self.temp = templates

    def get_theta_cloud(self):
        if self.lite:
            raise ValueError(
                "Cannot compute inter-particle distance for fields generated "
                "with `retain_catalog = False`"
            )
        if self.theta_cloud is None:
            import healpy as hp

            positions = self.pos if self.pos_r is None else self.pos_r
            weights = self.weights if self.weights_r is None else self.weights_r
            nsource = len(weights)
            power = min(4, max(int(0.5 * np.log2(0.5 * nsource)), 0))
            nside = 2**power
            pixels = hp.ang2pix(nside, *np.asarray(positions))
            density = np.bincount(
                pixels, minlength=12 * nside**2, weights=np.asarray(weights)
            )
            theta_ipd = np.median(
                np.sqrt(4 * np.pi / (12 * nside**2) / density[pixels])
            )
            self.theta_cloud = np.hypot(
                theta_ipd, np.pi / self.ainfo_mask.lmax
            )
        return self.theta_cloud

    def get_cloud_kernel(self, lmax):
        multipoles = np.arange(lmax + 1)
        return np.exp(-0.5 * self.get_theta_cloud() ** 2 * multipoles * (multipoles + 1))

    def get_catalog_variance_alm(self):
        if self.lite:
            raise ValueError("Cannot compute variance map for lightweight fields")
        alms = catalog2alm(
            jnp.sum(self.field**2, axis=0) / self.nmaps,
            self.pos,
            0,
            self.ainfo_mask.lmax,
        )[0]
        if self.mask is None and self.is_clustering:
            alms += catalog2alm(
                self.weights_r**2, self.pos_r, 0, self.ainfo_mask.lmax
            )[0]
        return alms

    def get_catalog_mask_map(self):
        if self.mask is not None:
            return None
        lmax = self.ainfo_mask.lmax
        nside = 2 ** int(np.ceil(np.log2(lmax / 3)))
        map_info = NmtMapInfo(None, (12 * nside**2,))
        kernel = jnp.asarray(self.get_cloud_kernel(lmax))
        smoothed = self.alm_mask * kernel[self.ainfo_mask._ell]
        return alm2map(smoothed[None], 0, map_info, self.ainfo_mask)[0], nside

    def get_catalog_mask_squared_map(self):
        if self.mask is not None:
            return None
        lmax = self.ainfo_mask.lmax
        nside = 2 ** int(np.ceil(np.log2(lmax / 3)))
        theta = self.get_theta_cloud()
        multipoles = jnp.arange(lmax + 1)
        kernel = (
            jnp.exp(-0.25 * theta**2 * multipoles * (multipoles + 1))
            / (4 * jnp.pi * theta**2)
        )
        alms = catalog2alm(
            self.weights_r**2, self.pos_r, 0, lmax
        )[0] * kernel[self.ainfo_mask._ell]
        map_info = NmtMapInfo(None, (12 * nside**2,))
        return alm2map(alms[None], 0, map_info, self.ainfo_mask)[0], nside

    def get_noise_deprojection_bias(self):
        if self.lite:
            raise ValueError(
                "Cannot compute noise deprojection bias for lightweight field"
            )
        shape = (self.nmaps * self.nmaps, self.ainfo.lmax + 1)
        if self._nl_deproj is not None:
            return self._nl_deproj
        if self.temp is None or self.noise_variance is None:
            self._nl_deproj = jnp.zeros(shape)
            return self._nl_deproj

        template_alms = jnp.stack(
            [catalog2alm(template, self.pos, self.spin, self.ainfo.lmax)
             for template in self.temp]
        )
        first = jnp.zeros(
            (self.n_temp, self.n_temp, self.nmaps, self.nmaps,
             self.ainfo.lmax + 1)
        )
        for j, template in enumerate(self.temp):
            weighted = catalog2alm(
                template * self.weights**2 * self.noise_variance,
                self.pos,
                self.spin,
                self.ainfo.lmax,
            )
            for i, template_alm in enumerate(template_alms):
                first = first.at[i, j].set(
                    _compute_coupled_cell(
                        weighted,
                        template_alm,
                        self.ainfo._ell,
                        self.ainfo._m,
                        lmax=self.ainfo.lmax,
                    ).reshape(self.nmaps, self.nmaps, -1)
                )
        bias = -2 * jnp.einsum("ij,ijabk->abk", self.iM, first)

        spectra = jnp.stack(
            [
                jnp.stack(
                    [
                        _compute_coupled_cell(
                            left,
                            right,
                            self.ainfo._ell,
                            self.ainfo._m,
                            lmax=self.ainfo.lmax,
                        ).reshape(self.nmaps, self.nmaps, -1)
                        for right in template_alms
                    ]
                )
                for left in template_alms
            ]
        )
        products = jnp.einsum(
            "p,iap,jap->ij",
            self.weights**2 * self.noise_variance,
            self.temp,
            self.temp,
        )
        prematrix = self.iM @ products @ self.iM
        bias += jnp.einsum("ij,ijabk->abk", prematrix, spectra)
        self._nl_deproj = bias.reshape(shape)
        return self._nl_deproj


def _summed_component_cell(alms1, alms2, alm_info, lmax):
    spectra = _compute_coupled_cell(
        alms1, alms2, alm_info._ell, alm_info._m, lmax=lmax
    ).reshape((len(alms1), len(alms2), -1))
    return jnp.trace(spectra, axis1=0, axis2=1)


class NmtFieldCatalogMomentum(NmtFieldCatalog):
    """Density-weighted catalog field using randoms or a map mask."""

    def __init__(
        self,
        positions,
        weights,
        field,
        positions_rand,
        weights_rand,
        lmax,
        lmax_mask=None,
        spin=None,
        field_is_weighted=False,
        lonlat=False,
        mask=None,
        n_iter_mask=None,
        wcs=None,
        templates=None,
        tol_pinv=None,
        lmax_deproj=None,
        n_iter_temp=None,
        masked_on_input=False,
        noise_variance=None,
        retain_catalog=False,
        nside_ipd=16,
    ):
        self.lite = not retain_catalog
        self.mask = self.maps = self.temp = self.alm_temp = None
        self.mask_a = self.alm_mask_a = None
        self.minfo = None
        self.beam = jnp.ones(lmax + 1)
        self.n_iter = None
        self.n_iter_mask = nmt_params.n_iter_mask_default if n_iter_mask is None else n_iter_mask
        n_iter_temp = nmt_params.n_iter_mask_default if n_iter_temp is None else n_iter_temp
        self.pure_e = self.pure_b = self.anisotropic_mask = False
        self.is_catalog = True
        self.is_clustering = field is None
        self.ainfo = NmtAlmInfo(lmax)
        self.alm = self.alm_mask = None
        self.n_temp = 0
        self._alpha = 0.0
        self._Nw = self._Nf = 0.0
        self._nl_deproj = None
        self.lmax_deproj = None
        self.nside_ipd = int(nside_ipd)
        self.theta_cloud = None
        self.pos = self.pos_r = self.weights = self.weights_r = self.field = None
        self.noise_variance = None

        positions, weights = _process_positions(positions, weights, lonlat, "data")
        if mask is not None:
            mask = jnp.asarray(mask, dtype=jnp.float64)
            self.minfo = NmtMapInfo(wcs, mask.shape)
            self.mask = self.minfo.reform_map(mask)
        if lmax_mask is None:
            lmax_mask = lmax if mask is None else self.minfo.get_lmax()
        self.ainfo_mask = NmtAlmInfo(lmax_mask)

        if mask is None:
            positions_rand, weights_rand = _process_positions(
                positions_rand, weights_rand, lonlat, "random"
            )
            nrand = len(weights_rand)
            self._alpha = jnp.sum(weights) / jnp.sum(weights_rand)
            self._Nw = jnp.sum(weights_rand**2) / (4 * jnp.pi)
            self.alm_mask = catalog2alm(
                weights_rand, positions_rand, 0, lmax_mask
            )[0]
            if self.is_clustering:
                mask_at_lmax = (
                    self.alm_mask
                    if lmax_mask == lmax
                    else catalog2alm(weights_rand, positions_rand, 0, lmax)[0]
                )
        else:
            self._alpha = jnp.sum(weights) / self.minfo.si.map_integral(self.mask)
            from .utils import map2alm

            self.alm_mask = map2alm(
                self.mask[None], 0, self.minfo, self.ainfo_mask,
                n_iter=self.n_iter_mask,
            )[0]
            if self.is_clustering:
                mask_at_lmax = (
                    self.alm_mask
                    if lmax_mask == lmax
                    else map2alm(
                        self.mask[None], 0, self.minfo, self.ainfo,
                        n_iter=self.n_iter_mask,
                    )[0]
                )

        if self.is_clustering:
            field = weights[None] / self._alpha
            field_is_weighted = True
            self.spin = 0
        else:
            field = jnp.atleast_2d(jnp.asarray(field, dtype=jnp.float64)) / self._alpha
            if spin is None:
                spin = 0 if len(field) == 1 else 2
            self.spin = int(spin)
        self.nmaps = 1 if self.spin == 0 else 2
        expected = (self.nmaps, len(weights))
        if field.shape != expected:
            raise ValueError(f"Field should have shape {expected}")
        if not field_is_weighted:
            field = field * weights[None]
        self.alm = catalog2alm(field, positions, self.spin, lmax)
        if self.is_clustering:
            self.alm = self.alm.at[0].add(-mask_at_lmax)

        if templates is not None:
            templates = jnp.asarray(templates, dtype=jnp.float64)
            self.lmax_deproj = lmax if lmax_deproj is None else int(lmax_deproj)
            if self.lmax_deproj > lmax:
                raise ValueError("lmax_deproj shouldn't be larger than lmax")
            self.n_temp = len(templates)
            zero_matrix = jnp.zeros((self.n_temp, self.n_temp))
            zero_products = jnp.zeros(self.n_temp)
            if mask is None:
                expected_templates = (self.n_temp, self.nmaps, nrand)
                if templates.shape != expected_templates:
                    raise ValueError(f"Templates should have shape {expected_templates}")
                if not masked_on_input:
                    templates = templates * weights_rand[None, None]
                template_alms = jnp.stack(
                    [catalog2alm(t, positions_rand, self.spin, lmax) for t in templates]
                )
                zero_matrix = jnp.einsum("iap,jap->ij", templates, templates) / (4 * jnp.pi)
                if self.is_clustering:
                    zero_products = -jnp.einsum(
                        "iap,p->i", templates, weights_rand
                    ) / (4 * jnp.pi)
            else:
                if templates.size != self.n_temp * self.nmaps * self.minfo.npix:
                    raise ValueError("Templates have wrong total size")
                templates = templates.reshape((self.n_temp, self.nmaps, self.minfo.npix))
                if not masked_on_input:
                    templates = templates * self.mask[None, None]
                from .utils import map2alm

                template_alms = jnp.stack(
                    [map2alm(t, self.spin, self.minfo, self.ainfo,
                             n_iter=n_iter_temp) for t in templates]
                )

            cells = jnp.stack(
                [jnp.stack([_summed_component_cell(a, b, self.ainfo, self.lmax_deproj)
                            for b in template_alms]) for a in template_alms]
            )
            multipoles = jnp.arange(self.lmax_deproj + 1)
            covariance = jnp.sum(
                (2 * multipoles + 1)[None, None]
                * (cells - zero_matrix[:, :, None]), axis=-1
            )
            self.iM = moore_penrose_pinvh(
                covariance,
                nmt_params.tol_pinv_default if tol_pinv is None else tol_pinv,
            )
            cross = jnp.stack(
                [_summed_component_cell(self.alm, t, self.ainfo, self.lmax_deproj)
                 for t in template_alms]
            )
            products = jnp.sum(
                (2 * multipoles + 1)[None] *
                (cross - zero_products[:, None]), axis=-1
            )
            self.alphas = self.iM @ products
            self.alm = self.alm - jnp.einsum(
                "i,iak->ak", self.alphas, template_alms
            )

        self._Nf = jnp.sum(field**2) / (4 * jnp.pi * self.nmaps)
        if self.is_clustering:
            self._Nf += self._Nw
        if retain_catalog:
            self.pos, self.weights, self.field = positions, weights, field
            self.pos_r, self.weights_r = positions_rand, weights_rand
            if self.is_clustering:
                if noise_variance is not None:
                    raise ValueError("The noise variance is fixed to the Poisson value")
                self.noise_variance = jnp.ones_like(weights)
            else:
                self.noise_variance = (
                    None if noise_variance is None else jnp.asarray(noise_variance)
                )
            if templates is not None:
                self.temp, self.alm_temp = templates, template_alms

    def get_noise_deprojection_bias(self):
        if self.lite:
            raise ValueError("Cannot compute noise deprojection bias for lightweight field")
        shape = (self.nmaps * self.nmaps, self.ainfo.lmax + 1)
        if self._nl_deproj is not None:
            return self._nl_deproj
        if self.temp is None or self.noise_variance is None:
            self._nl_deproj = jnp.zeros(shape)
            return self._nl_deproj

        keep = (jnp.arange(self.ainfo.lmax + 1) <= self.lmax_deproj)
        filtered = self.alm_temp * keep[self.ainfo._ell][None, None]
        at_data = jnp.stack([
            jnp.stack([alm2catalog(a[None], self.pos, 0, self.ainfo.lmax)[0]
                       for a in alms]) for alms in filtered
        ])
        at_random = None if self.mask is not None else jnp.stack([
            jnp.stack([alm2catalog(a[None], self.pos_r, 0, self.ainfo.lmax)[0]
                       for a in alms]) for alms in filtered
        ])
        first = jnp.zeros((self.n_temp, self.n_temp, self.nmaps, self.nmaps,
                           self.ainfo.lmax + 1))
        for j, samples in enumerate(at_data):
            weighted = catalog2alm(
                (self.weights / self._alpha) ** 2 * samples * self.noise_variance,
                self.pos, self.spin, self.ainfo.lmax,
            )
            if self.is_clustering and self.mask is None:
                weighted += catalog2alm(
                    self.weights_r**2 * at_random[j], self.pos_r,
                    self.spin, self.ainfo.lmax,
                )
            for i, template_alm in enumerate(self.alm_temp):
                first = first.at[i, j].set(
                    _compute_coupled_cell(
                        weighted, template_alm, self.ainfo._ell, self.ainfo._m,
                        lmax=self.ainfo.lmax,
                    ).reshape(self.nmaps, self.nmaps, -1)
                )
        bias = -2 * jnp.einsum("ij,ijabk->abk", self.iM, first)
        spectra = jnp.stack([jnp.stack([
            _compute_coupled_cell(a, b, self.ainfo._ell, self.ainfo._m,
                                  lmax=self.ainfo.lmax).reshape(self.nmaps, self.nmaps, -1)
            for b in self.alm_temp]) for a in self.alm_temp])
        products = jnp.einsum(
            "p,iap,jap->ij",
            (self.weights / self._alpha) ** 2 * self.noise_variance,
            at_data, at_data,
        )
        if self.is_clustering and self.mask is None:
            products += jnp.einsum(
                "p,iap,jap->ij", self.weights_r**2, at_random, at_random
            )
        bias += jnp.einsum(
            "ij,ijabk->abk", self.iM @ products @ self.iM, spectra
        )
        self._nl_deproj = bias.reshape(shape)
        return self._nl_deproj


class NmtFieldCatalogClustering(NmtFieldCatalogMomentum):
    """Catalog overdensity field normalized by randoms or a map mask."""

    def __init__(
        self, positions, weights, positions_rand, weights_rand, lmax,
        lmax_mask=None, lonlat=False, mask=None, n_iter_mask=None, wcs=None,
        templates=None, tol_pinv=None, lmax_deproj=None, n_iter_temp=None,
        masked_on_input=False, retain_catalog=False, nside_ipd=16,
    ):
        super().__init__(
            positions, weights, None, positions_rand, weights_rand, lmax,
            lmax_mask=lmax_mask, lonlat=lonlat, mask=mask,
            n_iter_mask=n_iter_mask, wcs=wcs, templates=templates,
            tol_pinv=tol_pinv, lmax_deproj=lmax_deproj,
            n_iter_temp=n_iter_temp, masked_on_input=masked_on_input,
            retain_catalog=retain_catalog, nside_ipd=nside_ipd,
        )
