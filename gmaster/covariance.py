"""Gaussian pseudo-C_ell covariance workspaces."""

import jax
import jax.numpy as jnp
import numpy as np

from .bins import NmtBin, NmtBinFlat
from .field_catalog import NmtFieldCatalog
from .field_flat import _flat_map2alm
from .utils import NmtMapInfo, alm2map, map2alm
from .workspaces import (
    NmtWorkspace,
    _apply_toeplitz,
    _binning_operators,
    _compute_coupled_cell,
    _coupling_matrix_tt,
    _coupling_matrix_tt_toeplitz,
    _coupling_matrices_spin2,
    _toeplitz_sanity,
    compute_coupled_cell,
)


_KERNEL_NAMES = ("00", "0s", "pp", "mm")


def _is_catalog(field):
    return isinstance(field, NmtFieldCatalog)


def _is_mask_catalog(field):
    return _is_catalog(field) and field.mask is None


def _mask_product_alm(first, second):
    """Harmonics of a mask product, including catalog self-pair removal."""
    if not first.is_compatible(second, strict=False):
        raise ValueError("Fields have incompatible pixelizations.")

    if not _is_mask_catalog(first) and _is_mask_catalog(second):
        first, second = second, first

    if _is_mask_catalog(first):
        mask_a, nside = first.get_catalog_mask_map()
        minfo = NmtMapInfo(None, mask_a.shape)
        if _is_mask_catalog(second):
            mask_b, nside_b = second.get_catalog_mask_map()
            if nside != nside_b:
                raise ValueError("Catalog masks use incompatible resolutions")
            product = mask_a * mask_b
            if first is second:
                mask_squared, _ = first.get_catalog_mask_squared_map()
                product = product - mask_squared
        else:
            mask_b = alm2map(
                second.get_mask_alms()[None], 0, minfo, second.ainfo_mask
            )[0]
            product = mask_a * mask_b
    else:
        minfo = first.minfo
        mask_a = first.get_mask()
        mask_b = (
            second.get_mask()
            if first.minfo == second.minfo
            else alm2map(
                second.get_mask_alms()[None], 0, minfo, second.ainfo_mask
            )[0]
        )
        product = mask_a * mask_b

    return map2alm(
        product[None], 0, minfo, first.ainfo_mask,
        n_iter=first.n_iter_mask,
    )[0]


def _alm_cross_cell(first, second, ainfo, lmax):
    return _compute_coupled_cell(
        first[None], second[None], ainfo._ell, ainfo._m, lmax=lmax
    )[0]


def _covariance_kernels(
    mask_cell,
    spin1,
    spin2,
    lmax,
    l_toeplitz=-1,
    l_exact=-1,
    dl_band=-1,
):
    """Return NaMaster's Xi kernels using GMaster's existing recurrences."""
    padded = jnp.pad(
        jnp.asarray(mask_cell),
        (0, max(0, 2 * lmax + 1 - len(mask_cell))),
    )[: 2 * lmax + 1]
    columns = (2 * jnp.arange(lmax + 1) + 1)[None]
    kernels = {name: None for name in _KERNEL_NAMES}
    if spin1 == 0 and spin2 == 0:
        kernels["00"] = (
            _coupling_matrix_tt_toeplitz(
                padded,
                lmax=lmax,
                l_toeplitz=l_toeplitz,
                l_exact=l_exact,
                dl_band=dl_band,
            )
            if l_toeplitz > 0
            else _coupling_matrix_tt(padded, lmax=lmax)
        ) / columns
    elif (spin1 == 0) != (spin2 == 0):
        mixed, _, _ = _coupling_matrices_spin2(padded, lmax=lmax, need_ee=False)
        kernels["0s"] = mixed / columns
    else:
        _, even, odd = _coupling_matrices_spin2(padded, lmax=lmax, need_te=False)
        kernels["pp"] = even / columns
        kernels["mm"] = odd / columns
    if l_toeplitz > 0:
        for name in ("0s", "pp", "mm"):
            if kernels[name] is not None:
                kernels[name] = _apply_toeplitz(
                    kernels[name], l_toeplitz, l_exact, dl_band
                )
    return kernels


def _signal_index(nmaps1, nmaps2, out1, out2, inp1, inp2):
    if nmaps1 == 1:
        if nmaps2 == 1:
            return "0"
        return "0" if out2 == inp2 else "Z"
    if nmaps2 == 1:
        return "0" if out1 == inp1 else "Z"
    return (
        "+", "--", "-+", "Z",
        "-+", "+", "Z", "-+",
        "--", "Z", "+", "--",
        "Z", "--", "-+", "+",
    )[inp2 + 2 * inp1 + 4 * (out2 + 2 * out1)]


def _noise_index(nmaps1, nmaps2, out1, out2):
    if nmaps1 != nmaps2:
        return "Z"
    if nmaps1 == 1:
        return "0"
    return ("+", "-+", "--", "+")[out2 + 2 * out1]


def _paired_kernel(first, second):
    if first == second == "0":
        return 1, "00"
    if (first, second) in (("0", "+"), ("+", "0")):
        return 1, "0s"
    if first == second == "+":
        return 1, "pp"
    if (first, second) in (("-+", "-+"), ("--", "--")):
        return -1, "mm"
    if (first, second) in (("-+", "--"), ("--", "-+")):
        return 1, "mm"
    return 0, None


def _zero_kernel_set():
    return [{name: None for name in _KERNEL_NAMES} for _ in range(2)]


class NmtCovarianceWorkspace:
    """Coupling coefficients for curved-sky Gaussian covariances."""

    def __init__(
        self,
        fla1=None,
        fla2=None,
        flb1=None,
        flb2=None,
        l_toeplitz=-1,
        l_exact=-1,
        dl_band=-1,
        fname=None,
    ):
        if fname is not None:
            self.read_from(fname)
        elif fla1 is not None or fla2 is not None:
            self.compute_coupling_coefficients(
                fla1, fla2, flb1=flb1, flb2=flb2,
                l_toeplitz=l_toeplitz, l_exact=l_exact, dl_band=dl_band,
            )

    @classmethod
    def from_fields(cls, fla1, fla2, flb1=None, flb2=None, **kwargs):
        return cls(fla1, fla2, flb1, flb2, **kwargs)

    @classmethod
    def from_file(cls, fname):
        return cls(fname=fname)

    def _post_init(self):
        self.nmaps = tuple(1 if spin == 0 else 2 for spin in self.spins)
        self.nclsa = self.nmaps[0] * self.nmaps[1]
        self.nclsb = self.nmaps[2] * self.nmaps[3]

    def _get_covariance_kernels(self, cell_1122=None, cell_1221=None):
        combinations = {
            (0, 0, 0, 0): ((0, 0), (0, 0)),
            (0, 0, 0, 2): ((0, 0), (0, 0)),
            (0, 0, 2, 0): ((0, 0), (0, 0)),
            (0, 0, 2, 2): ((0, 0), (0, 0)),
            (0, 2, 0, 0): ((0, 0), (0, 0)),
            (0, 2, 0, 2): ((0, 2), (0, 0)),
            (0, 2, 2, 0): ((0, 0), (0, 2)),
            (0, 2, 2, 2): ((0, 2), (0, 2)),
            (2, 0, 0, 0): ((0, 0), (0, 0)),
            (2, 0, 0, 2): ((0, 0), (0, 2)),
            (2, 0, 2, 0): ((0, 2), (0, 0)),
            (2, 0, 2, 2): ((0, 2), (0, 2)),
            (2, 2, 0, 0): ((0, 0), (0, 0)),
            (2, 2, 0, 2): ((0, 2), (0, 2)),
            (2, 2, 2, 0): ((0, 2), (0, 2)),
            (2, 2, 2, 2): ((2, 2), (2, 2)),
        }
        pairs = combinations.get(self.spins)
        if pairs is None:
            raise ValueError(f"Invalid combination of spins: {self.spins}")
        kernels = [
            _zero_kernel_set()[0]
            if cell is None
            else _covariance_kernels(
                cell,
                *pair,
                self.lmax,
                self.l_toeplitz,
                self.l_exact,
                self.dl_band,
            )
            for cell, pair in zip((cell_1122, cell_1221), pairs)
        ]
        return kernels

    def compute_coupling_coefficients(
        self,
        fla1,
        fla2,
        flb1=None,
        flb2=None,
        l_toeplitz=-1,
        l_exact=-1,
        dl_band=-1,
    ):
        flb1 = fla1 if flb1 is None else flb1
        flb2 = fla2 if flb2 is None else flb2
        fields = (fla1, fla2, flb1, flb2)
        if any(field.spin not in (0, 2) for field in fields):
            raise ValueError(
                "Covariance matrix estimation is only implemented for "
                "spin-0 and spin-2 fields."
            )
        if any(field.anisotropic_mask for field in fields):
            raise NotImplementedError(
                "Covariance matrix estimation not implemented for anisotropic weights."
            )
        _toeplitz_sanity(
            l_toeplitz, l_exact, dl_band, fla1.ainfo.lmax, fields
        )
        if any(not fla1.is_compatible(field, strict=False) for field in fields[1:]):
            raise ValueError("Fields have incompatible pixelizations.")

        self.spins = tuple(field.spin for field in fields)
        (
            self.spin_a1,
            self.spin_a2,
            self.spin_b1,
            self.spin_b2,
        ) = self.spins
        self.lmax = fla1.ainfo.lmax
        self.lmax_mask = fla1.ainfo_mask.lmax
        self.l_toeplitz = int(l_toeplitz)
        self.l_exact = int(l_exact)
        self.dl_band = int(dl_band)
        self._post_init()

        s11 = _mask_product_alm(fla1, flb1)
        s22 = _mask_product_alm(fla2, flb2)
        s12 = _mask_product_alm(fla1, flb2)
        s21 = _mask_product_alm(fla2, flb1)
        ainfo = fla1.ainfo_mask
        cell_1122 = _alm_cross_cell(s11, s22, ainfo, self.lmax_mask)
        cell_1221 = _alm_cross_cell(s12, s21, ainfo, self.lmax_mask)
        self.xiSS = self._get_covariance_kernels(cell_1122, cell_1221)

        self.has_SN = np.zeros(2, dtype=bool)
        self.has_NS = np.zeros(2, dtype=bool)
        self.has_NN = np.zeros(2, dtype=bool)
        self.xiSN = _zero_kernel_set()
        self.xiNS = _zero_kernel_set()
        self.xiNN = _zero_kernel_set()
        if not any(_is_catalog(field) for field in fields):
            return

        n11 = n22 = None
        if _is_catalog(fla1) and (fla1 is flb1 or fla1 is flb2):
            n11 = fla1.get_catalog_variance_alm()
        if _is_catalog(fla2) and (fla2 is flb1 or fla2 is flb2):
            n22 = n11 if fla2 is fla1 and n11 is not None else fla2.get_catalog_variance_alm()

        sn = [None, None]
        ns = [None, None]
        nn = [None, None]
        if fla1 is flb1 and _is_catalog(fla1):
            self.has_NS[0] = True
            ns[0] = _alm_cross_cell(n11, s22, ainfo, self.lmax_mask)
            if fla2 is flb2 and _is_catalog(fla2):
                self.has_NN[0] = True
                nn[0] = _alm_cross_cell(n11, n22, ainfo, self.lmax_mask)
        if fla2 is flb2 and _is_catalog(fla2):
            self.has_SN[0] = True
            sn[0] = _alm_cross_cell(s11, n22, ainfo, self.lmax_mask)
        if fla1 is flb2 and _is_catalog(fla1):
            self.has_NS[1] = True
            ns[1] = _alm_cross_cell(n11, s21, ainfo, self.lmax_mask)
            if fla2 is flb1 and _is_catalog(fla2):
                self.has_NN[1] = True
                nn[1] = _alm_cross_cell(n11, n22, ainfo, self.lmax_mask)
        if fla2 is flb1 and _is_catalog(fla2):
            self.has_SN[1] = True
            sn[1] = _alm_cross_cell(s12, n22, ainfo, self.lmax_mask)

        if fla1 is fla2 and _is_catalog(fla1) and not fla1.is_clustering:
            correction = jnp.sum(
                (jnp.sum(fla1.field**2, axis=0) / fla1.nmaps) ** 2
            ) / (4 * jnp.pi)
            nn = [None if value is None else value - correction for value in nn]
        if self.has_SN.any():
            self.xiSN = self._get_covariance_kernels(*sn)
        if self.has_NS.any():
            self.xiNS = self._get_covariance_kernels(*ns)
        if self.has_NN.any():
            self.xiNN = self._get_covariance_kernels(*nn)

    def _signal_covariance(self, spectra, xis, outputs, wick):
        na, nb, nc, nd = self.nmaps
        ia, ib, ic, id_ = outputs
        result = jnp.zeros((self.lmax + 1, self.lmax + 1))
        for iap in range(na):
            for ibp in range(nb):
                for icp in range(nc):
                    for idp in range(nd):
                        if wick == 0:
                            first = _signal_index(na, nc, ia, ic, iap, icp)
                            second = _signal_index(nb, nd, ib, id_, ibp, idp)
                            left = icp + nc * iap
                            right = idp + nd * ibp
                        else:
                            first = _signal_index(na, nd, ia, id_, iap, idp)
                            second = _signal_index(nb, nc, ib, ic, ibp, icp)
                            left = idp + nd * iap
                            right = icp + nc * ibp
                        sign, name = _paired_kernel(first, second)
                        if name is not None and xis[name] is not None:
                            result = result + sign * spectra[left, right] * xis[name]
        return result

    def _mixed_covariance(self, spectra, xis, outputs, wick, noise_first):
        na, nb, nc, nd = self.nmaps
        ia, ib, ic, id_ = outputs
        if noise_first:
            signal_out, noise_out = ib, ia
            signal_nm = nb
            other_nm, other_out = (nd, id_) if wick == 0 else (nc, ic)
            noise_other_nm, noise_other_out = (nc, ic) if wick == 0 else (nd, id_)
        else:
            signal_out, noise_out = ia, ib
            signal_nm = na
            other_nm, other_out = (nc, ic) if wick == 0 else (nd, id_)
            noise_other_nm, noise_other_out = (nd, id_) if wick == 0 else (nc, ic)
        result = jnp.zeros((self.lmax + 1, self.lmax + 1))
        noise = _noise_index(
            nb if not noise_first else na,
            noise_other_nm,
            noise_out,
            noise_other_out,
        )
        for first_in in range(signal_nm):
            for other_in in range(other_nm):
                signal = _signal_index(
                    signal_nm, other_nm,
                    signal_out, other_out, first_in, other_in,
                )
                sign, name = _paired_kernel(signal, noise)
                if name is not None and xis[name] is not None:
                    index = other_in + other_nm * first_in
                    result = result + sign * spectra[index] * xis[name]
        return result

    def _noise_covariance(self, xis, outputs, wick):
        na, nb, nc, nd = self.nmaps
        ia, ib, ic, id_ = outputs
        if wick == 0:
            first = _noise_index(na, nc, ia, ic)
            second = _noise_index(nb, nd, ib, id_)
        else:
            first = _noise_index(na, nd, ia, id_)
            second = _noise_index(nb, nc, ib, ic)
        sign, name = _paired_kernel(first, second)
        if name is None or xis[name] is None:
            return 0
        return sign * xis[name]

    def gaussian_covariance(
        self, cla1b1, cla1b2, cla2b1, cla2b2, wa, wb=None, coupled=False
    ):
        wb = wa if wb is None else wb
        na, nb, nc, nd = self.nmaps
        if wa.ncls != na * nb or wb.ncls != nc * nd:
            raise ValueError("Field spins do not match input workspaces")
        cls = tuple(jnp.asarray(cell) for cell in (cla1b1, cla1b2, cla2b1, cla2b2))
        expected_rows = (na * nc, na * nd, nb * nc, nb * nd)
        if any(cell.ndim != 2 or len(cell) != size for cell, size in zip(cls, expected_rows)):
            raise ValueError("Field spins do not match input power spectrum shapes")
        if any(cell.shape[1] < self.lmax + 1 for cell in cls):
            lengths = tuple(cell.shape[1] for cell in cls)
            raise ValueError(
                f"Input C_ls have a weird length. Expected {self.lmax + 1}, "
                f"but got {lengths}."
            )
        if wa.lmax != self.lmax or wb.lmax != self.lmax:
            raise ValueError("Input workspaces have a different lmax than the covariance workspace")

        c11, c12, c21, c22 = (cell[:, : self.lmax + 1] for cell in cls)
        product_1122 = c11[:, None, :, None] * c22[None, :, None, :]
        product_1122 = 0.5 * (product_1122 + product_1122.swapaxes(2, 3))
        product_1221 = c12[:, None, :, None] * c21[None, :, None, :]
        product_1221 = 0.5 * (product_1221 + product_1221.swapaxes(2, 3))
        mixed_sn = (
            0.5 * (c11[:, None] + c11[:, :, None]),
            0.5 * (c12[:, None] + c12[:, :, None]),
        )
        mixed_ns = (
            0.5 * (c22[:, None] + c22[:, :, None]),
            0.5 * (c21[:, None] + c21[:, :, None]),
        )

        blocks = []
        for ia in range(na):
            for ib in range(nb):
                row = []
                for ic in range(nc):
                    for id_ in range(nd):
                        outputs = ia, ib, ic, id_
                        cov = self._signal_covariance(product_1122, self.xiSS[0], outputs, 0)
                        cov += self._signal_covariance(product_1221, self.xiSS[1], outputs, 1)
                        for wick in range(2):
                            if self.has_SN[wick]:
                                cov += self._mixed_covariance(
                                    mixed_sn[wick], self.xiSN[wick], outputs, wick, False
                                )
                            if self.has_NS[wick]:
                                cov += self._mixed_covariance(
                                    mixed_ns[wick], self.xiNS[wick], outputs, wick, True
                                )
                            if self.has_NN[wick]:
                                cov += self._noise_covariance(self.xiNN[wick], outputs, wick)
                        row.append(cov)
                blocks.append(jnp.stack(row))
        covariance = jnp.stack(blocks)

        if not coupled:
            bin_a, _ = _binning_operators(wa.bins)
            bin_b, _ = _binning_operators(wb.bins)
            covariance = jnp.einsum(
                "pl,ablm,qm->abpq", bin_a, covariance, bin_b
            )
        covariance = covariance.reshape(
            (self.nclsa, self.nclsb, covariance.shape[-2], covariance.shape[-1])
        ).transpose((2, 0, 3, 1))
        covariance = covariance.reshape(
            (covariance.shape[0] * self.nclsa, covariance.shape[2] * self.nclsb)
        )
        if not coupled:
            covariance = jnp.linalg.solve(
                wa.mcm_binned,
                jnp.linalg.solve(wb.mcm_binned, covariance.T).T,
            )
        return covariance

    def write_to(self, fname):
        if not hasattr(self, "xiSS"):
            raise RuntimeError("Must initialize workspace before writing")
        import fitsio

        header = {
            "LMAX": self.lmax,
            "LMAX_MASK": self.lmax_mask,
            "SPIN_A1": self.spin_a1,
            "SPIN_A2": self.spin_a2,
            "SPIN_B1": self.spin_b1,
            "SPIN_B2": self.spin_b2,
            "L_TOEPLITZ": self.l_toeplitz,
            "L_EXACT": self.l_exact,
            "DL_BAND": self.dl_band,
        }
        labels = {"00": "00", "0s": "02", "pp": "22P", "mm": "22M"}
        with fitsio.FITS(fname, "rw", clobber=True) as fits:
            fits.write(np.ones((1, 1)), header=header, extname="CWSP_PRIMARY")
            for values, prefix in (
                (self.xiSS, ""), (self.xiSN, "SN"),
                (self.xiNS, "NS"), (self.xiNN, "NN"),
            ):
                for wick, suffix in enumerate(("1122", "1221")):
                    for name in _KERNEL_NAMES:
                        if values[wick][name] is not None:
                            fits.write(
                                np.asarray(values[wick][name]),
                                extname=f"XI{prefix}{labels[name]}_{suffix}",
                            )

    def read_from(self, fname):
        import fitsio

        labels = {"00": "00", "0s": "02", "pp": "22P", "mm": "22M"}
        with fitsio.FITS(fname) as fits:
            header = fits["CWSP_PRIMARY"].read_header()
            self.lmax = int(header["LMAX"])
            self.lmax_mask = int(header.get("LMAX_MASK", self.lmax))
            self.spins = tuple(
                int(header.get(name, 0))
                for name in ("SPIN_A1", "SPIN_A2", "SPIN_B1", "SPIN_B2")
            )
            (
                self.spin_a1, self.spin_a2, self.spin_b1, self.spin_b2
            ) = self.spins
            self.l_toeplitz = int(header.get("L_TOEPLITZ", -1))
            self.l_exact = int(header.get("L_EXACT", -1))
            self.dl_band = int(header.get("DL_BAND", -1))
            for attribute, prefix in (
                ("xiSS", ""), ("xiSN", "SN"),
                ("xiNS", "NS"), ("xiNN", "NN"),
            ):
                values = _zero_kernel_set()
                for wick, suffix in enumerate(("1122", "1221")):
                    for name in _KERNEL_NAMES:
                        extension = f"XI{prefix}{labels[name]}_{suffix}"
                        if extension in fits:
                            array = np.asarray(fits[extension].read(), dtype=np.float64)
                            expected = (self.lmax + 1, self.lmax + 1)
                            if array.shape != expected:
                                raise ValueError(f"{extension} shape does not match expected dimensions")
                            values[wick][name] = jnp.asarray(array)
                setattr(self, attribute, values)
        self.has_SN = np.asarray([
            any(self.xiSN[wick][name] is not None for name in _KERNEL_NAMES)
            for wick in range(2)
        ])
        self.has_NS = np.asarray([
            any(self.xiNS[wick][name] is not None for name in _KERNEL_NAMES)
            for wick in range(2)
        ])
        self.has_NN = np.asarray([
            any(self.xiNN[wick][name] is not None for name in _KERNEL_NAMES)
            for wick in range(2)
        ])
        self._post_init()


@jax.jit
def _flat_covariance_kernels(mask_1122, mask_1221, selectors, cosine, sine, norm):
    def convolution(mask, inputs):
        return jnp.fft.ifft2(
            jnp.fft.fft2(mask)[None] * jnp.fft.fft2(inputs)
        ).real

    def reduce(outputs, convolved):
        return jnp.einsum("piyx,pjyx->ij", outputs, convolved)

    one = jnp.ones_like(cosine)
    outputs = selectors[None] * one[None, None]

    def kernels(mask):
        xi00 = reduce(outputs, convolution(mask, outputs))
        inputs = selectors[None] * jnp.stack((cosine, sine))[:, None]
        xi02 = reduce(
            selectors[None] * jnp.stack((cosine, sine))[:, None],
            convolution(mask, inputs),
        )
        pp_out = jnp.stack((cosine**2, jnp.sqrt(2.0) * cosine * sine, sine**2))
        pp = reduce(
            selectors[None] * pp_out[:, None],
            convolution(mask, selectors[None] * pp_out[:, None]),
        )
        mm_out = jnp.stack((sine**2, jnp.sqrt(2.0) * sine * cosine, cosine**2))
        mm_in = jnp.stack((cosine**2, -jnp.sqrt(2.0) * sine * cosine, sine**2))
        mm = reduce(
            selectors[None] * mm_out[:, None],
            convolution(mask, selectors[None] * mm_in[:, None]),
        )
        return jnp.stack((xi00, xi02, pp, mm)) * norm

    return kernels(mask_1122), kernels(mask_1221)


class NmtCovarianceWorkspaceFlat:
    """GPU-accelerated flat-sky Gaussian covariance workspace."""

    def __init__(
        self, fla1=None, fla2=None, bin_a=None,
        flb1=None, flb2=None, bin_b=None, fname=None,
    ):
        self.wsp = None
        if fname is not None:
            self.read_from(fname)
        elif fla1 is not None or fla2 is not None or bin_a is not None:
            self.compute_coupling_coefficients(
                fla1, fla2, bin_a, flb1=flb1, flb2=flb2, bin_b=bin_b
            )

    @classmethod
    def from_fields(cls, fla1, fla2, bin_a, flb1=None, flb2=None, bin_b=None):
        return cls(fla1, fla2, bin_a, flb1, flb2, bin_b)

    @classmethod
    def from_file(cls, fname):
        return cls(fname=fname)

    def compute_coupling_coefficients(
        self, fla1, fla2, bin_a, flb1=None, flb2=None, bin_b=None
    ):
        flb1 = fla1 if flb1 is None else flb1
        flb2 = fla2 if flb2 is None else flb2
        bin_b = bin_a if bin_b is None else bin_b
        fields = (fla1, fla2, flb1, flb2)
        if any((field.nx, field.ny) != (fla1.nx, fla1.ny) for field in fields):
            raise ValueError("Can't compute covariance for fields with different resolutions")
        if bin_a.n_bands != bin_b.n_bands or not (
            np.array_equal(bin_a.l0, bin_b.l0) and np.array_equal(bin_a.lf, bin_b.lf)
        ):
            raise RuntimeError("Can't compute covariance for different binning schemes")

        self.bins = self.bin = bin_a
        self.nbands = bin_a.n_bands
        self.nx, self.ny = fla1.nx, fla1.ny
        self.lx, self.ly = fla1.lx, fla1.ly
        self.wsp = self
        masks = []
        for left, right in ((fla1, flb1), (fla1, flb2), (fla2, flb1), (fla2, flb2)):
            masks.append(_flat_map2alm((left.mask * right.mask)[None], 0, self.lx, self.ly)[0])
        ix = jnp.arange(self.nx)
        iy = jnp.arange(self.ny)
        kx = 2 * jnp.pi * jnp.where(2 * ix <= self.nx, ix, ix - self.nx) / self.lx
        ky = 2 * jnp.pi * jnp.where(2 * iy <= self.ny, iy, iy - self.ny) / self.ly
        stored_x = jnp.where(2 * ix <= self.nx, ix, self.nx - ix)
        mask_1122 = jnp.real(masks[0][:, stored_x] * jnp.conj(masks[3][:, stored_x]))
        mask_1221 = jnp.real(masks[1][:, stored_x] * jnp.conj(masks[2][:, stored_x]))
        ell = jnp.sqrt(ky[:, None] ** 2 + kx[None] ** 2)
        selectors = jnp.stack([
            (ell >= low) & (ell < high) for low, high in zip(bin_a.l0, bin_a.lf)
        ])
        counts = jnp.sum(selectors, axis=(1, 2))
        phi = jnp.arctan2(ky[:, None], kx[None])
        cosine, sine = jnp.cos(2 * phi), jnp.sin(2 * phi)
        norm = (
            4 * jnp.pi**2 / (self.lx**2 * self.ly**2)
            / jnp.where(counts[:, None] * counts[None] > 0,
                        counts[:, None] * counts[None], 1)
        )
        first, second = _flat_covariance_kernels(
            mask_1122, mask_1221, selectors, cosine, sine, norm
        )
        self.xi = [
            dict(zip(_KERNEL_NAMES, first)),
            dict(zip(_KERNEL_NAMES, second)),
        ]

    def gaussian_covariance(
        self, spin_a1, spin_a2, spin_b1, spin_b2, larr,
        cla1b1, cla1b2, cla2b1, cla2b2, wa, wb=None,
    ):
        wb = wa if wb is None else wb
        nmaps = tuple(1 if spin == 0 else 2 for spin in (
            spin_a1, spin_a2, spin_b1, spin_b2
        ))
        na, nb, nc, nd = nmaps
        if wa.ncls != na * nb or wb.ncls != nc * nd:
            raise ValueError("Input spins do not match input workspaces")
        larr = jnp.asarray(larr)
        cls = tuple(jnp.asarray(cell) for cell in (cla1b1, cla1b2, cla2b1, cla2b2))
        sizes = (na * nc, na * nd, nb * nc, nb * nd)
        if any(cell.ndim != 2 or len(cell) != size for cell, size in zip(cls, sizes)):
            raise ValueError("Input spins do not match input power spectrum shapes")
        if any(cell.shape[1] != len(larr) for cell in cls):
            raise ValueError("Input C_ls have a weird length")
        c11, c12, c21, c22 = (self.bins.bin_cell(larr, cell) for cell in cls)

        covariance = jnp.zeros((self.nbands, na * nb, self.nbands, nc * nd))
        for ia in range(na):
            for ib in range(nb):
                out_a = ib + nb * ia
                for ic in range(nc):
                    for id_ in range(nd):
                        out_b = id_ + nd * ic
                        block = jnp.zeros((self.nbands, self.nbands))
                        for iap in range(na):
                            for ibp in range(nb):
                                for icp in range(nc):
                                    for idp in range(nd):
                                        fac_1122 = 0.5 * (
                                            c11[icp + nc * iap, :, None]
                                            * c22[idp + nd * ibp, None]
                                            + c11[icp + nc * iap, None]
                                            * c22[idp + nd * ibp, :, None]
                                        )
                                        fac_1221 = 0.5 * (
                                            c12[idp + nd * iap, :, None]
                                            * c21[icp + nc * ibp, None]
                                            + c12[idp + nd * iap, None]
                                            * c21[icp + nc * ibp, :, None]
                                        )
                                        pair1 = _paired_kernel(
                                            _signal_index(na, nc, ia, ic, iap, icp),
                                            _signal_index(nb, nd, ib, id_, ibp, idp),
                                        )
                                        pair2 = _paired_kernel(
                                            _signal_index(na, nd, ia, id_, iap, idp),
                                            _signal_index(nb, nc, ib, ic, ibp, icp),
                                        )
                                        for fac, xis, (sign, name) in (
                                            (fac_1122, self.xi[0], pair1),
                                            (fac_1221, self.xi[1], pair2),
                                        ):
                                            if name is not None:
                                                block += sign * xis[name] * fac
                        covariance = covariance.at[:, out_a, :, out_b].set(block)
        covariance = covariance.reshape(
            (self.nbands * na * nb, self.nbands * nc * nd)
        )
        return jnp.linalg.solve(
            wa.mcm_binned,
            jnp.linalg.solve(wb.mcm_binned, covariance.T).T,
        )

    def write_to(self, fname):
        if self.wsp is None:
            raise RuntimeError("Must initialize workspace before writing")
        import fitsio

        labels = {"00": "00", "0s": "02", "pp": "22P", "mm": "22M"}
        with fitsio.FITS(fname, "rw", clobber=True) as fits:
            fits.write(
                [self.bins.l0, self.bins.lf],
                names=["ELL_0", "ELL_F"], extname="BINS_SUMMARY",
            )
            for wick, suffix in enumerate(("1122", "1221")):
                for name in _KERNEL_NAMES:
                    fits.write(
                        np.asarray(self.xi[wick][name]),
                        extname=f"XI{labels[name]}_{suffix}",
                    )

    def read_from(self, fname):
        import fitsio

        labels = {"00": "00", "0s": "02", "pp": "22P", "mm": "22M"}
        with fitsio.FITS(fname) as fits:
            summary = fits["BINS_SUMMARY"].read()
            self.bins = self.bin = NmtBinFlat(summary["ELL_0"], summary["ELL_F"])
            self.nbands = self.bins.n_bands
            self.xi = [
                {
                    name: jnp.asarray(fits[f"XI{labels[name]}_{suffix}"].read())
                    for name in _KERNEL_NAMES
                }
                for suffix in ("1122", "1221")
            ]
        self.wsp = self


def gaussian_covariance(
    cw, spin_a1, spin_a2, spin_b1, spin_b2,
    cla1b1, cla1b2, cla2b1, cla2b2, wa, wb=None, coupled=False,
):
    spins = (spin_a1, spin_a2, spin_b1, spin_b2)
    if spins != cw.spins:
        raise ValueError("Requested spins do not match those used to initialize the workspace")
    return cw.gaussian_covariance(
        cla1b1, cla1b2, cla2b1, cla2b2, wa, wb=wb, coupled=coupled
    )


def gaussian_covariance_flat(
    cw, spin_a1, spin_a2, spin_b1, spin_b2, larr,
    cla1b1, cla1b2, cla2b1, cla2b2, wa, wb=None,
):
    return cw.gaussian_covariance(
        spin_a1, spin_a2, spin_b1, spin_b2, larr,
        cla1b1, cla1b2, cla2b1, cla2b2, wa, wb=wb,
    )


def get_iNKA_cell(fla, flb, cl_guess=None, w=None):
    """Return the improved narrow-kernel input spectrum."""
    if not fla.is_compatible(flb, strict=False):
        raise ValueError("Fields have incompatible pixelizations")
    map_product = (
        fla.mask is not None and flb.mask is not None
        and fla.minfo is not None and fla.minfo == flb.minfo
    )
    if map_product:
        fsky = jnp.mean(fla.get_mask() * flb.get_mask())
    else:
        lmax = fla.ainfo_mask.lmax
        cell = _alm_cross_cell(
            fla.get_mask_alms(), flb.get_mask_alms(), fla.ainfo_mask, lmax
        )
        if _is_catalog(fla) and _is_catalog(flb):
            kernel_a = 1 if fla.mask is not None else fla.get_cloud_kernel(lmax)
            kernel_b = 1 if flb.mask is not None else flb.get_cloud_kernel(lmax)
            if fla is flb:
                cell = cell - fla.Nw
            cell = cell * jnp.asarray(kernel_a) * jnp.asarray(kernel_b)
        ell = jnp.arange(lmax + 1)
        fsky = jnp.sum((2 * ell + 1) * cell) / (4 * jnp.pi)
    if cl_guess is None:
        coupled = compute_coupled_cell(fla, flb)
    else:
        if w is None:
            width = max(fla.ainfo.lmax // 10, 1)
            bins = NmtBin.from_lmax_linear(fla.ainfo.lmax, width)
            w = NmtWorkspace.from_fields(fla, flb, bins)
        coupled = w.couple_cell(cl_guess)
    return coupled / fsky
