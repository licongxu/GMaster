"""Flat-sky MASTER: coupling matrices, decoupling and deprojection bias.

GPU counterpart of ``pymaster.NmtWorkspaceFlat`` and its companion functions.  The
mode-coupling matrix is built as a 2-D convolution of the mask power spectrum with the
spin response of each field, evaluated with FFTs on the device.
"""

import jax.numpy as jnp
import numpy as np

from .bins import NmtBinFlat
from .field_flat import _flat_alm2map, _flat_map2alm
from .workspaces import _compute_coupled_cell_flat


def _geometry(nx, ny, lx, ly):
    """Full-plane wavevectors kx, ky, their modulus ell and angle phi."""
    ix = jnp.arange(nx)
    iy = jnp.arange(ny)
    kx = 2 * jnp.pi * jnp.where(2 * ix <= nx, ix, ix - nx) / lx
    ky = 2 * jnp.pi * jnp.where(2 * iy <= ny, iy, iy - ny) / ly
    ell = jnp.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    phi = jnp.arctan2(ky[:, None], kx[None, :])
    return kx, ky, ell, phi


def _response_terms(spin, pure_e, pure_b, ell, phi):
    """Separable output/input factors of a field's flat-sky spin response.

    Each factor has shape ``(nmaps, nmaps, 2, ny, nx)``: output component, input
    component, and two separable terms whose products sum to the response.  Spin
    fields rotate by ``spin * phi``; purified components replace the rotation by the
    pure-mode weighting.
    """
    shape = ell.shape
    if spin == 0:
        out = jnp.zeros((1, 1, 2, *shape)).at[0, 0, 0].set(1)
        return out, out

    cosine = jnp.cos(spin * phi)
    sine = jnp.sin(spin * phi)
    rotation = jnp.stack(
        [jnp.stack([cosine, -sine]), jnp.stack([sine, cosine])]
    )
    out = jnp.broadcast_to(rotation[:, None], (2, 2, 2, *shape))
    inp = jnp.broadcast_to(rotation[None], (2, 2, 2, *shape))

    inverse_ell2 = jnp.where(ell > 0, 1 / ell**2, 0)
    origin = (ell == 0).astype(ell.dtype)
    for component, pure in enumerate((pure_e, pure_b)):
        if pure:
            out = out.at[component].set(0)
            inp = inp.at[component].set(0)
            out = out.at[component, component, 0].set(inverse_ell2)
            inp = inp.at[component, component, 0].set(ell**2)
            out = out.at[component, component, 1].set(origin)
            inp = inp.at[component, component, 1].set(origin)
    return out, inp


def _beam_values(field, ell):
    """Field beam interpolated onto `ell` (1 without a beam, 0 beyond its last ell)."""
    if field.beam is None:
        return jnp.ones_like(ell)
    sampled = jnp.interp(
        ell, field.beam[0], field.beam[1], left=field.beam[1, 0]
    )
    return jnp.where(ell >= field.beam[0, -1], 0, sampled)


def _factor_batches(response_pairs):
    """Flatten pairs of field responses into (output, input, row, column) term lists."""
    out_factors = []
    in_factors = []
    rows = []
    columns = []
    for out1, inp1, out2, inp2, offset in response_pairs:
        nmap1, nmap2 = out1.shape[0], out2.shape[0]
        for a in range(nmap1):
            for b in range(nmap2):
                for i in range(nmap1):
                    for j in range(nmap2):
                        for term1 in range(2):
                            for term2 in range(2):
                                out_factors.append(
                                    out1[a, i, term1] * out2[b, j, term2]
                                )
                                in_factors.append(
                                    inp1[a, i, term1] * inp2[b, j, term2]
                                )
                                rows.append(offset + a * nmap2 + b)
                                columns.append(offset + i * nmap2 + j)
    return (
        jnp.stack(out_factors),
        jnp.stack(in_factors),
        jnp.asarray(rows),
        jnp.asarray(columns),
    )


def _convolved_matrix(
    mask_power,
    beam,
    output_selectors,
    input_selectors,
    factors,
    ncls,
    normalization,
):
    """Coupling matrix: mask power convolved with each response term, binned in and out.

    Returns shape ``(n_out * ncls, n_in * ncls)``, with the output selectors (bands)
    and input selectors (ell rings or bands) as the slow index.
    """
    out_factors, in_factors, rows, columns = factors
    ny, nx = mask_power.shape
    inputs = (
        in_factors[:, None]
        * input_selectors[None]
        * beam[None, None]
    )
    convolution = jnp.fft.ifft2(
        jnp.fft.fft2(mask_power)[None, None] * jnp.fft.fft2(inputs)
    ).real
    values = jnp.einsum(
        "qyx,qgyx,oyx->qog", out_factors, convolution, output_selectors
    )
    blocks = jnp.zeros(
        (ncls, ncls, len(output_selectors), len(input_selectors)),
        dtype=values.dtype,
    ).at[rows, columns].add(values)
    matrix = blocks.transpose(2, 0, 3, 1)
    matrix *= normalization[:, None, None, None]
    return matrix.reshape(
        (len(output_selectors) * ncls, len(input_selectors) * ncls)
    )


class NmtWorkspaceFlat:
    """Mode-coupling matrix and decoupling operator for a pair of flat-sky fields.

    Same interface as ``pymaster.NmtWorkspaceFlat``.  If fields and bins are given the
    coupling matrix is computed immediately; with `fname` it is read from a file.

    Parameters
    ----------
    fl1, fl2 : NmtFieldFlat, optional
        Fields to correlate (must share the pixel grid).
    bins : NmtBinFlat, optional
        Bandpowers.
    ell_cut_x, ell_cut_y : tuple of float, optional
        Fourier modes with ``ell_cut_x[0] <= k_x <= ell_cut_x[1]`` (or likewise in y)
        are removed.  The default ``(1.0, -1.0)`` is an empty range: no cut.
    is_teb : bool, optional
        If True, `fl1` must be spin 0 and `fl2` spin 2, and the full 7x7 T/E/B coupling
        (TT, TE, TB, EE, EB, BE, BB) is computed together.
    fname : str, optional
        FITS file written by `write_to` to read the workspace from.
    """

    def __init__(
        self,
        fl1=None,
        fl2=None,
        bins=None,
        ell_cut_x=(1.0, -1.0),
        ell_cut_y=(1.0, -1.0),
        is_teb=False,
        fname=None,
    ):
        self.mcm = self.mcm_binned = None
        if fname is not None:
            self.read_from(fname)
        elif fl1 is not None or fl2 is not None or bins is not None:
            self.compute_coupling_matrix(
                fl1,
                fl2,
                bins,
                ell_cut_x=ell_cut_x,
                ell_cut_y=ell_cut_y,
                is_teb=is_teb,
            )

    @classmethod
    def from_fields(
        cls,
        fl1,
        fl2,
        bins,
        ell_cut_x=(1.0, -1.0),
        ell_cut_y=(1.0, -1.0),
        is_teb=False,
    ):
        """Create a workspace and compute its coupling matrix from two fields.

        See the class docstring for the parameters.

        Returns
        -------
        NmtWorkspaceFlat
        """
        return cls(
            fl1,
            fl2,
            bins,
            ell_cut_x=ell_cut_x,
            ell_cut_y=ell_cut_y,
            is_teb=is_teb,
        )

    @classmethod
    def from_file(cls, fname):
        """Read a workspace from a FITS file written by `write_to`.

        Returns
        -------
        NmtWorkspaceFlat
        """
        return cls(fname=fname)

    def compute_coupling_matrix(
        self,
        fl1,
        fl2,
        bins,
        ell_cut_x=(1.0, -1.0),
        ell_cut_y=(1.0, -1.0),
        is_teb=False,
    ):
        """Compute the mode-coupling matrix of two flat-sky fields.

        Stores the matrix with unbinned input (``mcm``, input sampled on rings of width
        ``dell``) and the fully binned matrix used for decoupling (``mcm_binned``).
        Parameters are as in the class docstring.
        """
        if not fl1.is_compatible(fl2):
            raise ValueError("Fields must have same resolution")
        if not isinstance(bins, NmtBinFlat):
            raise TypeError("bins must be an NmtBinFlat")
        if is_teb and (fl1.spin != 0 or fl2.spin == 0):
            raise ValueError(
                "For T-E-B MCM the first input field must be spin-0 and the second spin-!=0"
            )

        self.nx, self.ny = fl1.nx, fl1.ny
        self.lx, self.ly = fl1.lx, fl1.ly
        self.bins = bins
        self.nbands = bins.n_bands
        self.is_teb = bool(is_teb)
        self.ncls = 7 if is_teb else fl1.nmaps * fl2.nmaps
        self.spin1, self.spin2 = fl1.spin, fl2.spin
        self.pure_e1, self.pure_b1 = fl1.pure_e, fl1.pure_b
        self.pure_e2, self.pure_b2 = fl2.pure_e, fl2.pure_b
        self.ell_cut_x = tuple(ell_cut_x)
        self.ell_cut_y = tuple(ell_cut_y)

        kx, ky, ell, phi = _geometry(
            self.nx, self.ny, self.lx, self.ly
        )
        valid = ~(
            ((kx[None] >= ell_cut_x[0]) & (kx[None] <= ell_cut_x[1]))
            | ((ky[:, None] >= ell_cut_y[0]) & (ky[:, None] <= ell_cut_y[1]))
        )
        output_selectors = jnp.stack(
            [
                valid & (ell >= low) & (ell < high)
                for low, high in zip(bins.l0, bins.lf)
            ]
        )
        input_bands = jnp.stack(
            [(ell >= low) & (ell < high) for low, high in zip(bins.l0, bins.lf)]
        )
        self.n_cells = jnp.sum(output_selectors, axis=(1, 2))

        dell = min(2 * np.pi / self.lx, 2 * np.pi / self.ly)
        kmax = np.hypot(
            2 * np.pi / self.lx * (self.nx // 2),
            2 * np.pi / self.ly * (self.ny // 2),
        )
        self.dell = dell
        self.n_ell = int(np.floor(kmax / dell))
        rings = jnp.floor(ell / dell).astype(jnp.int32)
        input_rings = jnp.stack([rings == ring for ring in range(self.n_ell)])
        self.ell_sampling = (np.arange(self.n_ell) + 0.5) * dell

        stored_x = jnp.where(
            2 * jnp.arange(self.nx) <= self.nx,
            jnp.arange(self.nx),
            self.nx - jnp.arange(self.nx),
        )
        mask_alm1 = _flat_map2alm(fl1.mask[None], 0, self.lx, self.ly)[0]
        mask_alm2 = (
            mask_alm1
            if fl2 is fl1
            else _flat_map2alm(fl2.mask[None], 0, self.lx, self.ly)[0]
        )
        mask_power = jnp.real(
            mask_alm1[:, stored_x] * jnp.conj(mask_alm2[:, stored_x])
        )
        beam = _beam_values(fl1, ell) * _beam_values(fl2, ell)

        response1 = _response_terms(
            fl1.spin, fl1.pure_e, fl1.pure_b, ell, phi
        )
        response2 = _response_terms(
            fl2.spin, fl2.pure_e, fl2.pure_b, ell, phi
        )
        if is_teb:
            scalar = _response_terms(0, False, False, ell, phi)
            spin = response2
            response_pairs = [
                (*scalar, *scalar, 0),
                (*scalar, *spin, 1),
                (*spin, *spin, 3),
            ]
        else:
            response_pairs = [(*response1, *response2, 0)]
        factors = _factor_batches(response_pairs)

        normalization = (
            4 * jnp.pi**2 / (self.lx**2 * self.ly**2)
            / jnp.where(self.n_cells > 0, self.n_cells, 1)
        )
        self.mcm = _convolved_matrix(
            mask_power,
            beam,
            output_selectors,
            input_rings,
            factors,
            self.ncls,
            normalization,
        )
        self.mcm_binned = _convolved_matrix(
            mask_power,
            beam,
            output_selectors,
            input_bands,
            factors,
            self.ncls,
            normalization,
        )

    def couple_cell(self, ells, cl_in):
        """Apply the mode-coupling matrix to theory power spectra.

        Parameters
        ----------
        ells : array_like, shape (n_ell,)
            Multipoles at which `cl_in` is sampled (increasing).
        cl_in : array_like, shape (ncls, n_ell)
            Power spectra, interpolated onto the Fourier grid.

        Returns
        -------
        jax.Array, shape (ncls, n_bands)
            Coupled bandpowers, comparable to `compute_coupled_cell_flat`.
        """
        if self.mcm is None:
            raise RuntimeError("Must initialize workspace before coupling")
        ells = jnp.asarray(ells)
        cl_in = jnp.asarray(cl_in)
        if cl_in.shape != (self.ncls, len(ells)):
            raise ValueError(
                f"Input power spectrum has wrong shape. Expected "
                f"({self.ncls}, {len(ells)}), but got {cl_in.shape}."
            )
        _, _, ell, _ = _geometry(self.nx, self.ny, self.lx, self.ly)
        ring = jnp.floor(ell / self.dell).astype(jnp.int32).reshape(-1)
        valid = ring < self.n_ell
        sampled = jnp.stack(
            [
                jnp.where(
                    ell >= ells[-1],
                    0,
                    jnp.interp(ell, ells, cl, left=cl[0]),
                ).reshape(-1)
                for cl in cl_in
            ]
        )
        counts = jnp.bincount(ring[valid], length=self.n_ell)
        ring_cls = jnp.stack(
            [
                jnp.bincount(ring[valid], weights=cl[valid], length=self.n_ell)
                / jnp.where(counts > 0, counts, 1)
                for cl in sampled
            ]
        )
        coupled = self.mcm @ ring_cls.T.reshape(-1)
        return coupled.reshape(self.nbands, self.ncls).T

    def decouple_cell(self, cl_in, cl_bias=None, cl_noise=None):
        """Invert the binned mode-coupling matrix.

        Parameters
        ----------
        cl_in : array_like, shape (ncls, n_bands)
            Coupled bandpowers.
        cl_bias : array_like, shape (ncls, n_bands), optional
            Deprojection bias to subtract first.
        cl_noise : array_like, shape (ncls, n_bands), optional
            Coupled noise bias to subtract first.

        Returns
        -------
        jax.Array, shape (ncls, n_bands)
            Decoupled bandpowers.
        """
        if self.mcm_binned is None:
            raise RuntimeError("Must initialize workspace before decoupling")
        cl_in = jnp.asarray(cl_in)
        expected = (self.ncls, self.nbands)
        if cl_in.shape != expected:
            raise ValueError(
                f"Input power spectrum has wrong shape. Expected {expected}, "
                f"but got {cl_in.shape}"
            )
        rhs = cl_in
        for correction in (cl_noise, cl_bias):
            if correction is not None:
                correction = jnp.asarray(correction)
                if correction.shape != expected:
                    raise ValueError("Input bias power spectrum has wrong shape")
                rhs = rhs - correction
        result = jnp.linalg.solve(self.mcm_binned, rhs.T.reshape(-1))
        return result.reshape(self.nbands, self.ncls).T

    def read_from(self, fname):
        """Read the workspace from a FITS file in NaMaster's flat workspace layout."""
        import fitsio

        with fitsio.FITS(fname) as fits:
            header = fits["WSP_PRIMARY"].read_header()
            self.mcm = jnp.asarray(fits["WSP_PRIMARY"].read())
            fs_header = fits["FS_INFO"].read_header()
            self.nx = int(fs_header["NX"])
            self.ny = int(fs_header["NY"])
            self.lx = float(fs_header["LX"])
            self.ly = float(fs_header["LY"])
            self.dell = float(fs_header["DELL"])
            self.ell_sampling = (
                np.asarray(fits["FS_INFO"]["L_MIN"].read()) + 0.5 * self.dell
            )
            self.n_ell = len(self.ell_sampling)
            self.n_cells = jnp.asarray(
                fits["N_CELLS"]["N_CELLS"].read().astype(np.int32)
            )
            self.mcm_binned = jnp.asarray(fits["MCM_BINNED"].read())
            summary = fits["BINS_SUMMARY"].read()
            self.bins = NmtBinFlat(summary["ELL_0"], summary["ELL_F"])

        self.nbands = self.bins.n_bands
        self.ncls = int(header["NCLS"])
        self.is_teb = bool(header["IS_TEB"])
        self.pure_e1 = bool(header["PURE_E1"])
        self.pure_e2 = bool(header["PURE_E2"])
        self.pure_b1 = bool(header["PURE_B1"])
        self.pure_b2 = bool(header["PURE_B2"])
        self.ell_cut_x = (float(header["ELLCUT_X_I"]), float(header["ELLCUT_X_F"]))
        self.ell_cut_y = (float(header["ELLCUT_Y_I"]), float(header["ELLCUT_Y_F"]))

    def write_to(self, fname):
        """Write the workspace to a FITS file in NaMaster's flat workspace layout.

        The file also carries the LU factorisation of the binned matrix, as NaMaster
        writes it.
        """
        if self.mcm is None:
            raise RuntimeError("Must initialize workspace before writing")
        import fitsio
        from scipy.linalg import lu_factor

        matrix = np.asarray(self.mcm_binned)
        lu, pivots = lu_factor(matrix)
        permutation = np.arange(len(matrix), dtype=np.int32)
        for row, pivot in enumerate(pivots):
            permutation[row], permutation[pivot] = (
                permutation[pivot],
                permutation[row],
            )
        header = {
            "LMAX": float(self.bins.lf[-1]),
            "ELLCUT_X_I": self.ell_cut_x[0],
            "ELLCUT_X_F": self.ell_cut_x[1],
            "ELLCUT_Y_I": self.ell_cut_y[0],
            "ELLCUT_Y_F": self.ell_cut_y[1],
            "PURE_E1": int(self.pure_e1),
            "PURE_E2": int(self.pure_e2),
            "PURE_B1": int(self.pure_b1),
            "PURE_B2": int(self.pure_b2),
            "IS_TEB": int(self.is_teb),
            "NCLS": self.ncls,
        }
        with fitsio.FITS(fname, "rw", clobber=True) as fits:
            fits.write(np.asarray(self.mcm), header=header, extname="WSP_PRIMARY")
            fs_header = {
                "NX": self.nx,
                "NY": self.ny,
                "NPIX": self.nx * self.ny,
                "LX": self.lx,
                "LY": self.ly,
                "PIXSIZE": self.lx * self.ly / (self.nx * self.ny),
                "DELL": self.dell,
                "I_DELL": 1 / self.dell,
            }
            fits.write(
                [self.ell_sampling - 0.5 * self.dell],
                names=["L_MIN"],
                header=fs_header,
                extname="FS_INFO",
            )
            fits.write(
                [np.asarray(self.n_cells, dtype=np.int32)],
                names=["N_CELLS"],
                extname="N_CELLS",
            )
            fits.write(matrix, extname="MCM_BINNED")
            fits.write(lu, extname="MCM_BINNED_GSL")
            fits.write(permutation, extname="MCM_PERM")
            fits.write(
                [self.bins.l0, self.bins.lf],
                names=["ELL_0", "ELL_F"],
                extname="BINS_SUMMARY",
            )


def _interpolate_spectra(ells, cls, ell):
    """Interpolate spectra onto the Fourier-plane `ell` (0 beyond the last multipole)."""
    return jnp.stack(
        [
            jnp.where(
                ell >= ells[-1],
                0,
                jnp.interp(ell, ells, spectrum, left=spectrum[0]),
            )
            for spectrum in cls
        ]
    )


def _filter_alms(alms, cls, ells, field_out):
    """Multiply Fourier coefficients by a matrix of spectra: out_a = sum_b C_ab alm_b."""
    kx = 2 * jnp.pi * jnp.arange(field_out.nx // 2 + 1) / field_out.lx
    iy = jnp.arange(field_out.ny)
    ky = (
        2
        * jnp.pi
        * jnp.where(2 * iy <= field_out.ny, iy, iy - field_out.ny)
        / field_out.ly
    )
    ell = jnp.sqrt(ky[:, None] ** 2 + kx[None] ** 2)
    windows = _interpolate_spectra(ells, cls.reshape((-1, len(ells))), ell)
    windows = windows.reshape((*cls.shape[:2], field_out.ny, field_out.nx // 2 + 1))
    return jnp.einsum("abij,bij->aij", windows, alms)


def _masked_alms(field, maps):
    """Fourier coefficients of `maps` after masking (and purification) as for `field`."""
    if field.pure_e or field.pure_b:
        return field._purify(maps)[0]
    return _flat_map2alm(maps * field.mask[None], field.spin, field.lx, field.ly)


def _raw_coupled(alms1, alms2, field, bins, ell_cut_x, ell_cut_y):
    """Binned coupled power spectrum of two sets of flat Fourier coefficients."""
    return _compute_coupled_cell_flat(
        alms1,
        alms2,
        jnp.asarray(bins.l0),
        jnp.asarray(bins.lf),
        field.lx,
        field.ly,
        jnp.asarray([*ell_cut_x, *ell_cut_y]),
        nx=field.nx,
        ny=field.ny,
    )


def deprojection_bias_flat(
    f1,
    f2,
    b,
    ells,
    cl_guess,
    ell_cut_x=(1.0, -1.0),
    ell_cut_y=(1.0, -1.0),
):
    """Bias to the coupled power spectrum caused by contaminant deprojection (flat sky).

    Parameters
    ----------
    f1, f2 : NmtFieldFlat
        Fields (not `lite`).
    b : NmtBinFlat
        Bandpowers.
    ells : array_like, shape (n_ell,)
        Multipoles at which `cl_guess` is sampled.
    cl_guess : array_like, shape (nmaps1 * nmaps2, n_ell)
        Best guess of the true power spectra.
    ell_cut_x, ell_cut_y : tuple of float, optional
        Fourier modes with ``ell_cut_x[0] <= k_x <= ell_cut_x[1]`` (or likewise in y)
        are removed.  The default ``(1.0, -1.0)`` is an empty range: no cut.

    Returns
    -------
    jax.Array, shape (nmaps1 * nmaps2, n_bands)
        Deprojection bias, to be passed as ``cl_bias`` to `decouple_cell`.
    """
    if f1.lite or f2.lite:
        raise ValueError("No deprojection bias for lightweight fields")
    if not f1.is_compatible(f2):
        raise ValueError("Fields must have same resolution")
    ells = jnp.asarray(ells)
    cl_guess = jnp.asarray(cl_guess)
    expected = (f1.nmaps * f2.nmaps, len(ells))
    if cl_guess.shape != expected:
        raise ValueError("Proposal Cell doesn't match number of maps")
    guess = cl_guess.reshape((f1.nmaps, f2.nmaps, len(ells)))
    bias = jnp.zeros((f1.nmaps * f2.nmaps, b.n_bands), dtype=cl_guess.dtype)

    if f2.n_temp:
        for template_i in range(f2.n_temp):
            for template_j in range(f2.n_temp):
                alms2 = _flat_map2alm(
                    f2.temp[template_j] * f2.mask[None],
                    f2.spin,
                    f2.lx,
                    f2.ly,
                )
                alms1 = _filter_alms(alms2, guess, ells, f1)
                maps1 = _flat_alm2map(alms1, f1.spin, f1.lx, f1.ly, f1.nx)
                alms1 = _masked_alms(f1, maps1)
                bias -= f2.iM[template_i, template_j] * _raw_coupled(
                    alms1,
                    f2.alm_temp[template_i],
                    f1,
                    b,
                    ell_cut_x,
                    ell_cut_y,
                )

    if f1.n_temp:
        for template_i in range(f1.n_temp):
            for template_j in range(f1.n_temp):
                alms1 = _flat_map2alm(
                    f1.temp[template_j] * f1.mask[None],
                    f1.spin,
                    f1.lx,
                    f1.ly,
                )
                alms2 = _filter_alms(
                    alms1, guess.transpose(1, 0, 2), ells, f2
                )
                maps2 = _flat_alm2map(alms2, f2.spin, f2.lx, f2.ly, f2.nx)
                alms2 = _masked_alms(f2, maps2)
                bias -= f1.iM[template_i, template_j] * _raw_coupled(
                    f1.alm_temp[template_i],
                    alms2,
                    f1,
                    b,
                    ell_cut_x,
                    ell_cut_y,
                )

    if f1.n_temp and f2.n_temp:
        products = jnp.zeros((f1.n_temp, f2.n_temp), dtype=cl_guess.dtype)
        pixel_area = f1.lx * f1.ly / f1.npix
        for template_j in range(f1.n_temp):
            for template_q in range(f2.n_temp):
                alms2 = _flat_map2alm(
                    f2.temp[template_q] * f2.mask[None],
                    f2.spin,
                    f2.lx,
                    f2.ly,
                )
                alms1 = _filter_alms(alms2, guess, ells, f1)
                maps1 = _flat_alm2map(alms1, f1.spin, f1.lx, f1.ly, f1.nx)
                product = pixel_area * jnp.sum(
                    maps1 * f1.mask[None] * f1.temp[template_j]
                )
                products = products.at[template_j, template_q].set(product)

        for template_i in range(f1.n_temp):
            for template_p in range(f2.n_temp):
                spectrum = _raw_coupled(
                    f1.alm_temp[template_i],
                    f2.alm_temp[template_p],
                    f1,
                    b,
                    ell_cut_x,
                    ell_cut_y,
                )
                coefficient = jnp.einsum(
                    "j,q,jq->",
                    f1.iM[template_i],
                    f2.iM[template_p],
                    products,
                )
                bias += coefficient * spectrum
    return bias


def compute_full_master_flat(
    f1,
    f2,
    b,
    cl_noise=None,
    cl_guess=None,
    ells_guess=None,
    workspace=None,
    ell_cut_x=(1.0, -1.0),
    ell_cut_y=(1.0, -1.0),
):
    """Full flat-sky MASTER estimate: coupled spectrum, deprojection bias and decoupling.

    Parameters
    ----------
    f1, f2 : NmtFieldFlat
        Fields to correlate.
    b : NmtBinFlat
        Bandpowers.
    cl_noise : array_like, shape (nmaps1 * nmaps2, n_bands), optional
        Coupled noise bias.
    cl_guess : array_like, shape (nmaps1 * nmaps2, n_ell), optional
        Guess of the true spectra for the deprojection bias (zero if None).
    ells_guess : array_like, shape (n_ell,), optional
        Multipoles of `cl_guess`; required if `cl_guess` is given.
    workspace : NmtWorkspaceFlat, optional
        Precomputed workspace; computed from the fields if None.
    ell_cut_x, ell_cut_y : tuple of float, optional
        Fourier modes with ``ell_cut_x[0] <= k_x <= ell_cut_x[1]`` (or likewise in y)
        are removed.  The default ``(1.0, -1.0)`` is an empty range: no cut.

    Returns
    -------
    jax.Array, shape (nmaps1 * nmaps2, n_bands)
        Decoupled bandpowers.
    """
    expected = (f1.nmaps * f2.nmaps, b.n_bands)
    noise = jnp.zeros(expected) if cl_noise is None else jnp.asarray(cl_noise)
    if noise.shape != expected:
        raise ValueError("Wrong length for noise power spectrum")
    if cl_guess is None:
        ells_guess = b.get_effective_ells()
        cl_guess = jnp.zeros(expected)
    elif ells_guess is None:
        raise ValueError("Must provide ell-values for cl_guess")
    coupled = _raw_coupled(
        f1.get_alms(), f2.get_alms(), f1, b, ell_cut_x, ell_cut_y
    )
    bias = deprojection_bias_flat(
        f1,
        f2,
        b,
        jnp.asarray(ells_guess),
        jnp.asarray(cl_guess),
        ell_cut_x,
        ell_cut_y,
    )
    if workspace is None:
        workspace = NmtWorkspaceFlat(
            f1,
            f2,
            b,
            ell_cut_x=ell_cut_x,
            ell_cut_y=ell_cut_y,
        )
    return workspace.decouple_cell(coupled, cl_bias=bias, cl_noise=noise)
