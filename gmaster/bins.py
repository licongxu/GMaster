"""Bandpower binning schemes for curved-sky (`NmtBin`) and flat-sky (`NmtBinFlat`) spectra.

Mirrors ``pymaster.NmtBin`` / ``pymaster.NmtBinFlat``.  Binning and unbinning run on the
JAX device and return JAX arrays; the per-band accessors return NumPy arrays.
"""

from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np


@partial(jax.jit, static_argnames="n_bands")
def _bin_cell(cls, ells, bpws, factors, *, n_bands):
    """Weighted sum of each spectrum over the multipoles of every band."""
    return jax.vmap(
        lambda cl: jax.ops.segment_sum(cl[ells] * factors, bpws, num_segments=n_bands)
    )(cls)


@partial(jax.jit, static_argnames="lmax")
def _unbin_cell(cls, ells, bpws, f_ell, lmax):
    """Spread bandpowers back onto their multipoles, dividing out f_ell."""
    out = jnp.zeros((cls.shape[0], lmax + 1), dtype=cls.dtype)
    return out.at[:, ells].set(cls[:, bpws] / f_ell)


def _linear_bands(lmax, nlb):
    """Bands of `nlb` multipoles starting at ell = 2; ell < 2 and an incomplete last band get -1."""
    ells = np.arange(lmax + 1, dtype=np.int32)
    bpws = ((ells - 2) // nlb).astype(np.int32)
    bpws[:2] = -1
    if np.sum(bpws == bpws[-1]) != nlb:
        bpws[bpws == bpws[-1]] = -1
    return ells, bpws


class NmtBin:
    """Bandpower definition for curved-sky power spectra.

    Each multipole ``ell`` is assigned to a bandpower index ``bpws`` with a weight
    ``w_ell``; the binned spectrum is ``C_b = sum_{ell in b} w_ell f_ell C_ell``, with the
    weights normalised to sum to one in every band.  Multipoles with a negative band
    index, or above `lmax`, are ignored.  Usually built with `from_nside_linear`,
    `from_lmax_linear` or `from_edges`.  As in pymaster, the arguments are keyword-only.

    Parameters
    ----------
    bpws : array_like of int, shape (n_ell,)
        Bandpower index of each entry of `ells`.
    ells : array_like of int, shape (n_ell,)
        Multipoles.
    lmax : int, optional
        Maximum multipole of the spectra to be binned.  Defaults to ``max(ells)``.
    weights : array_like, shape (n_ell,), optional
        Weight of each multipole within its band (normalised internally).  Defaults to
        uniform weights.
    f_ell : array_like, shape (n_ell,), optional
        Multiplicative factor applied before binning, e.g. ``ell (ell + 1) / 2 pi`` to
        bin D_ell.  Non-positive values are replaced by 1.  Defaults to 1.
    """

    def __init__(self, *, bpws, ells, lmax=None, weights=None, f_ell=None):
        ells = np.asarray(ells, dtype=np.int32)
        bpws = np.asarray(bpws, dtype=np.int32)
        if ells.ndim != 1 or bpws.shape != ells.shape or not len(ells):
            raise ValueError("ells and bpws must be non-empty 1-D arrays of equal size")

        self.lmax = int(np.max(ells) if lmax is None else lmax)
        weights = np.ones(len(ells)) if weights is None else np.asarray(weights)
        f_ell = np.ones(len(ells)) if f_ell is None else np.asarray(f_ell)
        if weights.shape != ells.shape or f_ell.shape != ells.shape:
            raise ValueError("weights and f_ell must have the same shape as ells")

        keep = (ells <= self.lmax) & (bpws >= 0)
        ells = ells[keep]
        bpws = bpws[keep]
        weights = np.asarray(weights[keep], dtype=np.float64)
        f_ell = np.asarray(f_ell[keep], dtype=np.float64)
        n_bands = int(np.max(bpws, initial=0)) + 1

        totals = np.bincount(bpws, weights=weights, minlength=n_bands)
        if np.any(totals <= 0):
            bad = int(np.flatnonzero(totals <= 0)[0])
            raise RuntimeError(f"Weights in band {bad} are wrong")
        weights /= totals[bpws]
        f_ell = np.where(f_ell > 0, f_ell, 1.0)

        self.n_bands = n_bands
        self.ell_max = self.lmax
        self.bin = SimpleNamespace(ell_max=self.lmax, n_bands=n_bands)
        self._ells_np = ells
        self._bpws_np = bpws
        self._weights_np = weights
        self._f_ell_np = f_ell
        self._ells = jnp.asarray(ells)
        self._bpws = jnp.asarray(bpws)
        self._f_ell = jnp.asarray(f_ell)
        self._factors = jnp.asarray(weights * f_ell)

    @classmethod
    def from_nside_linear(cls, nside, nlb, is_Dell=False, f_ell=None):
        """Linear bands of constant width up to ``lmax = 3 * nside - 1``.

        Parameters
        ----------
        nside : int
            HEALPix resolution.
        nlb : int
            Number of multipoles per band.  Bands start at ell = 2; an incomplete last
            band is discarded.
        is_Dell : bool, optional
            If True, bin ``D_ell = ell (ell + 1) C_ell / 2 pi`` instead of ``C_ell``.
        f_ell : array_like, shape (lmax + 1,), optional
            Custom multiplicative factor; overrides `is_Dell`.

        Returns
        -------
        NmtBin
        """
        return cls.from_lmax_linear(3 * nside - 1, nlb, is_Dell, f_ell)

    @classmethod
    def from_lmax_linear(cls, lmax, nlb, is_Dell=False, f_ell=None):
        """Linear bands of constant width up to `lmax`.

        Parameters
        ----------
        lmax : int
            Maximum multipole.
        nlb : int
            Number of multipoles per band.  Bands start at ell = 2; an incomplete last
            band is discarded.
        is_Dell : bool, optional
            If True, bin ``D_ell = ell (ell + 1) C_ell / 2 pi`` instead of ``C_ell``.
        f_ell : array_like, shape (lmax + 1,), optional
            Custom multiplicative factor; overrides `is_Dell`.

        Returns
        -------
        NmtBin
        """
        ells, bpws = _linear_bands(lmax, nlb)
        if is_Dell and f_ell is None:
            f_ell = ells * (ells + 1) / (2 * np.pi)
        return cls(bpws=bpws, ells=ells, lmax=lmax, f_ell=f_ell)

    @classmethod
    def from_edges(cls, ell_ini, ell_end, is_Dell=False, f_ell=None):
        """Top-hat bands with the given edges.

        Parameters
        ----------
        ell_ini : array_like of int, shape (n_bands,)
            First multipole of each band (inclusive).
        ell_end : array_like of int, shape (n_bands,)
            Last multipole of each band (exclusive).
        is_Dell : bool, optional
            If True, bin ``D_ell = ell (ell + 1) C_ell / 2 pi`` instead of ``C_ell``.
        f_ell : array_like, optional
            Custom multiplicative factor, one value per binned multipole (in band
            order); overrides `is_Dell`.

        Returns
        -------
        NmtBin
            Uses ``lmax = max(ell_end) - 1``.
        """
        ells = np.concatenate(
            [np.arange(start, end) for start, end in zip(ell_ini, ell_end)]
        )
        bpws = np.concatenate(
            [
                np.full(end - start, band)
                for band, (start, end) in enumerate(zip(ell_ini, ell_end))
            ]
        )
        weights = np.concatenate(
            [
                np.full(end - start, 1 / (end - start))
                for start, end in zip(ell_ini, ell_end)
            ]
        )
        if is_Dell and f_ell is None:
            f_ell = ells * (ells + 1) / (2 * np.pi)
        return cls(
            bpws=bpws, ells=ells, lmax=np.max(ells), weights=weights, f_ell=f_ell
        )

    def get_n_bands(self):
        """Return the number of bandpowers."""
        return self.n_bands

    def get_nell_list(self):
        """Return the number of multipoles in each band, shape (n_bands,)."""
        return np.bincount(self._bpws_np, minlength=self.n_bands)

    def get_ell_min(self, b):
        """Return the lowest multipole in band `b`."""
        return self.get_ell_list(b)[0]

    def get_ell_max(self, b):
        """Return the highest multipole in band `b`."""
        return self.get_ell_list(b)[-1]

    def get_ell_list(self, b):
        """Return the multipoles in band `b` (NumPy array)."""
        return self._ells_np[self._bpws_np == int(b)].copy()

    def get_weight_list(self, b):
        """Return the normalised weights of the multipoles in band `b`."""
        return self._weights_np[self._bpws_np == int(b)].copy()

    def get_fell_list(self, b):
        """Return the f_ell factors of the multipoles in band `b`."""
        return self._f_ell_np[self._bpws_np == int(b)].copy()

    def get_effective_ells(self):
        """Return the weighted mean multipole of each band, shape (n_bands,)."""
        return np.bincount(
            self._bpws_np,
            weights=self._ells_np * self._weights_np,
            minlength=self.n_bands,
        )

    def bin_cell(self, cls_in):
        """Bin power spectra into bandpowers.

        Parameters
        ----------
        cls_in : array_like, shape (n_cls, lmax + 1) or (lmax + 1,)
            Power spectra.

        Returns
        -------
        jax.Array, shape (n_cls, n_bands) or (n_bands,)
            Bandpowers.
        """
        cls_in = jnp.asarray(cls_in)
        oned = cls_in.ndim != 2
        if oned:
            cls_in = cls_in[None]
        if cls_in.ndim > 2 or cls_in.shape[1] != self.lmax + 1:
            raise ValueError("Input Cl has wrong size")
        out = _bin_cell(
            cls_in, self._ells, self._bpws, self._factors, n_bands=self.n_bands
        )
        return out[0] if oned else out

    def unbin_cell(self, cls_in):
        """Expand bandpowers into a piecewise-constant spectrum.

        Every multipole of band ``b`` gets ``C_b / f_ell``; unbinned multipoles are 0.

        Parameters
        ----------
        cls_in : array_like, shape (n_cls, n_bands) or (n_bands,)
            Bandpowers.

        Returns
        -------
        jax.Array, shape (n_cls, lmax + 1) or (lmax + 1,)
        """
        cls_in = jnp.asarray(cls_in)
        oned = cls_in.ndim != 2
        if oned:
            cls_in = cls_in[None]
        if cls_in.ndim > 2 or cls_in.shape[1] != self.n_bands:
            raise ValueError("Input Cl has wrong size")
        out = _unbin_cell(cls_in, self._ells, self._bpws, self._f_ell, self.lmax)
        return out[0] if oned else out

    @classmethod
    def _from_fits_file(cls, fits, extname="BINS"):
        """Read a binning scheme from an open fitsio file (workspace I/O)."""
        hdu = fits[extname]
        header = hdu.read_header()
        return cls(
            lmax=header["ELL_MAX"],
            bpws=hdu["BAND"].read(),
            ells=hdu["ELLS"].read(),
            weights=hdu["WEIGHTS"].read().astype(np.float64),
            f_ell=hdu["F_ELL"].read().astype(np.float64),
        )

    def _to_fits_file(self, fits, extname="BINS"):
        """Write the binning scheme to an open fitsio file (workspace I/O)."""
        ells = np.arange(self.lmax + 1, dtype=np.int32)
        bands = np.full(self.lmax + 1, -1, dtype=np.int32)
        weights = np.zeros(self.lmax + 1)
        f_ell = np.zeros(self.lmax + 1)
        bands[self._ells_np] = self._bpws_np
        weights[self._ells_np] = self._weights_np
        f_ell[self._ells_np] = self._f_ell_np
        fits.write(
            [bands, ells, weights, f_ell],
            names=["BAND", "ELLS", "WEIGHTS", "F_ELL"],
            header={"ELL_MAX": self.lmax, "N_BANDS": self.n_bands},
            extname=extname,
        )


class NmtBinFlat:
    """Top-hat bandpower definition for flat-sky power spectra.

    Parameters
    ----------
    l0 : array_like, shape (n_bands,)
        Lower edge of each band.
    lf : array_like, shape (n_bands,)
        Upper edge of each band.
    """

    def __init__(self, l0, lf):
        self.l0 = np.asarray(l0, dtype=np.float64)
        self.lf = np.asarray(lf, dtype=np.float64)
        if self.l0.ndim != 1 or self.lf.shape != self.l0.shape:
            raise ValueError("l0 and lf must be 1-D arrays of equal size")
        self.n_bands = len(self.l0)
        self.bin = SimpleNamespace(n_bands=self.n_bands)
        self._effective_ells = jnp.asarray(0.5 * (self.l0 + self.lf))

    def get_n_bands(self):
        """Return the number of bandpowers."""
        return self.n_bands

    def get_effective_ells(self):
        """Return the band centres ``(l0 + lf) / 2``, shape (n_bands,)."""
        return np.asarray(self._effective_ells)

    def bin_cell(self, ells, cls_in):
        """Evaluate power spectra at the band centres.

        The spectra are linearly interpolated at the band centres (constant beyond the
        ends of `ells`).

        Parameters
        ----------
        ells : array_like, shape (n_ell,)
            Multipoles at which `cls_in` is sampled (increasing).
        cls_in : array_like, shape (n_cls, n_ell) or (n_ell,)
            Power spectra.

        Returns
        -------
        jax.Array, shape (n_cls, n_bands) or (n_bands,)
        """
        ells = jnp.asarray(ells)
        cls_in = jnp.asarray(cls_in)
        oned = cls_in.ndim != 2
        if oned:
            cls_in = cls_in[None]
        if cls_in.ndim > 2 or cls_in.shape[1] != len(ells):
            raise ValueError("Input Cl has wrong size")
        out = jax.vmap(
            lambda cl: jnp.interp(
                self._effective_ells, ells, cl, left=cl[0], right=cl[-1]
            )
        )(cls_in)
        return out[0] if oned else out

    def unbin_cell(self, cls_in, ells):
        """Evaluate bandpowers at arbitrary multipoles as a step function.

        Parameters
        ----------
        cls_in : array_like, shape (n_cls, n_bands) or (n_bands,)
            Bandpowers.
        ells : array_like, shape (n_ell,)
            Multipoles at which to evaluate.

        Returns
        -------
        jax.Array, shape (n_cls, n_ell) or (n_ell,)
            Value of the band containing each multipole, or -999 outside all bands.
        """
        cls_in = jnp.asarray(cls_in)
        ells = jnp.asarray(ells)
        oned = cls_in.ndim != 2
        if oned:
            cls_in = cls_in[None]
        if cls_in.ndim > 2 or cls_in.shape[1] != self.n_bands:
            raise ValueError("Input Cl has wrong size")
        bands = jnp.sum(ells[:, None] >= jnp.asarray(self.lf)[None, :], axis=1)
        valid = (bands < self.n_bands) & (ells >= self.l0[0])
        values = cls_in[:, jnp.minimum(bands, self.n_bands - 1)]
        out = jnp.where(valid[None, :], values, -999)
        return out[0] if oned else out
