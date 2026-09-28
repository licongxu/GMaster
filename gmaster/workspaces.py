"""Curved-sky and flat-sky MASTER pseudo-C_ell machinery.

This module mirrors ``pymaster.workspaces``. It provides

* coupled pseudo-spectra of two fields (:func:`compute_coupled_cell`,
  :func:`compute_coupled_cell_flat`);
* mode-coupling matrices (MCMs) for any spin pair, including pure-E/B fields
  (:class:`NmtWorkspace`, :func:`get_general_coupling_matrix`), built either by
  the Wigner-3j recurrence or by Gauss-Legendre quadrature of Wigner-d
  products, optionally with the Toeplitz approximation;
* binning and decoupling of pseudo-spectra, and bandpower window functions;
* full-estimator helpers (:func:`compute_full_master`,
  :func:`deprojection_bias`).

Spectra are ordered as in NaMaster: for fields of spins ``(s1, s2)`` there are
``ncls = n1 * n2`` spectra (1, 2 or 4), and an unbinned MCM has shape
``(ncls * (lmax+1), ncls * (lmax+1))`` with index ``l * ncls + c``.
"""

import os
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln
from scipy.special import roots_legendre

from .bins import NmtBin, NmtBinFlat
from . import _config
from . import _coupling_tt_cuda
from .utils import alm2map, map2alm

# Operand precision of the coupling-matrix builders ("fp64", "fp32" or "auto").
# The polarised quadrature is dominated by two large GEMMs that are fp64-bound on
# the GPU; float32 operands make them ~30x faster at a relative matrix error of
# ~2e-6. The scalar recurrence has no GEMM, and float32 per-term products give
# ~8x at ~2e-7. "auto" (default) uses float32 exactly where the float32 transform
# engine (`gmaster._sht.march_v2`, every band limit with a CUDA build) is active;
# its own accuracy is ~1e-6, so the matrix error stays within the pipeline's.
# With GMASTER_MARCH_V2=0 the transforms and this builder stay float64 (tested
# to atol 2e-14). GMASTER_COUPLING_PRECISION or set_coupling_precision() forces a
# single width at every size.
_COUPLING_PRECISION = os.environ.get("GMASTER_COUPLING_PRECISION", "auto")


def _coupling_f32(lmax):
    """Whether this coupling build uses float32 operands."""
    if _COUPLING_PRECISION != "auto":
        return _COUPLING_PRECISION == "fp32"
    from ._sht import march_v2 as _march_v2

    return _march_v2.enabled(int(lmax) + 1)


def set_coupling_precision(name):
    """Choose the operand precision of the coupling-matrix builders.

    Applies to the polarised quadrature (operand width of its two large
    contractions) and to the scalar build (lookup tables and per-term products;
    the log-cumsum table and the offset accumulator stay float64).

    Parameters
    ----------
    name : {"fp64", "fp32", "auto"}
        ``"auto"`` uses float32 where the float32 transform engine is active
        and float64 otherwise.

    Notes
    -----
    The flag is read at trace time, so already-compiled programs keep their
    precision. Call this before the first coupling build, or call
    ``jax.clear_caches()`` afterwards.
    """
    global _COUPLING_PRECISION
    if name not in ("fp64", "fp32", "auto"):
        raise KeyError("GMaster coupling precision must be 'fp64', 'fp32' or 'auto'")
    _COUPLING_PRECISION = name


def coupling_precision():
    """Return the current coupling-precision setting ("fp64", "fp32" or "auto")."""
    return _COUPLING_PRECISION


@partial(jax.jit, static_argnames="lmax")
def _compute_coupled_cell(alm1, alm2, ell, order, *, lmax):
    """Cross-spectra of all component pairs from ``m >= 0`` alms (m > 0 counted twice)."""
    products = jnp.real(alm1[:, None, :] * jnp.conj(alm2[None, :, :]))
    products *= jnp.where(order == 0, 1, 2)
    products = products.reshape((-1, products.shape[-1]))
    cls = jax.vmap(
        lambda product: jax.ops.segment_sum(product, ell, num_segments=lmax + 1)
    )(products)
    return cls / (2 * jnp.arange(lmax + 1) + 1)


def compute_coupled_cell(f1, f2):
    """Compute the coupled pseudo-C_ell of two masked fields.

    Parameters
    ----------
    f1, f2 : NmtField
        Fields to correlate; they must share pixelization and alm layout.

    Returns
    -------
    jax.Array
        ``(n1 * n2, lmax + 1)`` array of coupled spectra, where ``n1, n2`` are
        the numbers of field components and ``lmax`` the smaller of the two
        fields' band limits. For an auto-correlation the field's noise-bias
        offset ``Nf`` is subtracted from the diagonal spectra.
    """
    if not f1.is_compatible(f2, strict=False):
        raise ValueError("You're trying to correlate incompatible fields")
    alm1 = f1.get_alms()
    alm2 = f2.get_alms()
    lmax = min(f1.ainfo.lmax, f2.ainfo.lmax)
    cls = _compute_coupled_cell(alm1, alm2, f1.ainfo._ell, f1.ainfo._m, lmax=lmax)
    if f2 is f1 and f1.Nf != 0:
        cls = cls.reshape((len(alm1), len(alm2), lmax + 1))
        diagonal = jnp.arange(len(alm1))
        cls = cls.at[diagonal, diagonal].add(-f1.Nf)
        cls = cls.reshape((-1, lmax + 1))
    return cls


@partial(jax.jit, static_argnames=("nx", "ny"))
def _compute_coupled_cell_flat(
    alm1, alm2, l0, lf, lx, ly, cuts, *, nx, ny
):
    """Band-averaged flat-sky cross-power of all component pairs, with ell cuts."""
    ix = jnp.arange(nx)
    iy = jnp.arange(ny)
    kx = 2 * jnp.pi * jnp.where(2 * ix <= nx, ix, ix - nx) / lx
    ky = 2 * jnp.pi * jnp.where(2 * iy <= ny, iy, iy - ny) / ly
    stored_x = jnp.where(2 * ix <= nx, ix, nx - ix)
    modes1 = alm1[:, iy[:, None], stored_x[None, :]]
    modes2 = alm2[:, iy[:, None], stored_x[None, :]]
    products = jnp.real(modes1[:, None] * jnp.conj(modes2[None, :]))
    ell = jnp.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    valid = ~(
        ((kx[None, :] >= cuts[0]) & (kx[None, :] <= cuts[1]))
        | ((ky[:, None] >= cuts[2]) & (ky[:, None] <= cuts[3]))
    )
    in_band = (
        valid[None]
        & (ell[None] >= l0[:, None, None])
        & (ell[None] < lf[:, None, None])
    )
    counts = jnp.sum(in_band, axis=(1, 2))
    summed = jnp.einsum("abij,kij->abk", products, in_band)
    return (summed * (4 * jnp.pi**2 / (lx * ly))
            / jnp.where(counts > 0, counts, 1)).reshape((-1, len(l0)))


def compute_coupled_cell_flat(
    f1, f2, b, ell_cut_x=(1.0, -1.0), ell_cut_y=(1.0, -1.0)
):
    """Compute the binned coupled pseudo-C_ell of two flat-sky fields.

    Parameters
    ----------
    f1, f2 : NmtFieldFlat
        Fields to correlate; they must share resolution.
    b : NmtBinFlat
        Bandpowers to average into.
    ell_cut_x, ell_cut_y : tuple of float, optional
        ``(min, max)`` ranges of ``l_x`` and ``l_y`` to remove (no cut when
        ``min > max``).

    Returns
    -------
    jax.Array
        ``(n1 * n2, n_bands)`` binned coupled spectra.
    """
    if not isinstance(b, NmtBinFlat):
        raise TypeError("b must be an NmtBinFlat")
    if not f1.is_compatible(f2):
        raise ValueError("Fields must have same resolution")
    cuts = jnp.asarray([*ell_cut_x, *ell_cut_y])
    return _compute_coupled_cell_flat(
        f1.get_alms(),
        f2.get_alms(),
        jnp.asarray(b.l0),
        jnp.asarray(b.lf),
        f1.lx,
        f1.ly,
        cuts,
        nx=f1.nx,
        ny=f1.ny,
    )


# Offsets per scan step in the scalar recurrence; see `_coupling_matrix_tt_recurrence`.
_OFFSET_CHUNK = 16


# Dense coupling matrices above this size are assembled in pieces (`_assemble_mcm`).
_MCM_PIECED_BYTES = 4 * 1024 ** 3
# Polarised matrices are held as their distinct blocks (`_BlockMCM`); 0 restores the dense form.
_BLOCK_MCM = os.environ.get("GMASTER_BLOCK_MCM", "1") != "0"


@partial(jax.jit, static_argnames=("dtype", "lmax", "ncls", "slots", "signs"))
def _assemble_mcm_program(blocks, *, dtype, lmax, ncls, slots, signs):
    """Place the ``(lmax+1, lmax+1)`` blocks into the dense MCM in one program.

    Tracing the chain of ``.at[:, i, :, j].set`` updates inside a single jit
    lets XLA write the ``(lmax+1, ncls, lmax+1, ncls)`` output in one pass
    instead of copying the whole matrix once per block. Signs are exactly
    ``+-1``, so values are bit-identical to the unfused form.
    """
    matrix = jnp.zeros((lmax + 1, ncls, lmax + 1, ncls), dtype=dtype)
    for block, (row, column), sign in zip(blocks, slots, signs):
        matrix = matrix.at[:, row, :, column].set(block * sign)
    return matrix.reshape((ncls * (lmax + 1), ncls * (lmax + 1)))


@partial(jax.jit, static_argnames=("signs",))
def _mcm_row(pieces, *, signs):
    """One row channel of the MCM, shape ``(l1, l2, c2)`` with the channel axis innermost."""
    return jnp.stack([p * s for p, s in zip(pieces, signs)], axis=-1)


@partial(jax.jit, static_argnames=("ncls", "slots", "signs", "index"))
def _place_blocks(parts, *, ncls, slots, signs, index):
    """`(rows, ncls, cols, ncls)` array with `parts[index[k]] * signs[k]` at slot `slots[k]`."""
    rows, cols = parts[0].shape
    out = jnp.zeros((rows, ncls, cols, ncls), dtype=parts[0].dtype)
    for k, (c1, c2) in enumerate(slots):
        out = out.at[:, c1, :, c2].add(parts[index[k]] * signs[k])
    return out


class _BlockMCM:
    """A polarised MCM held as its distinct ``(lmax+1, lmax+1)`` blocks.

    For spin-2 fields the ``(ncls (lmax+1))^2`` matrix repeats a few blocks
    (``M+`` and ``M-``, Alonso et al. 2019 eqs. 19-20; more with purification)
    in ``ncls^2`` slots with signs ``+-1``. Binning (``W @ M``), matvecs and
    host export all work per distinct block, so the dense matrix (18 GiB at
    Nside 4096 spin 2) is never built on the device, and ``W @ M`` costs one
    ``(nbpw, L) @ (L, L)`` GEMM per block instead of one ``ncls^2`` times
    larger.
    """

    def __init__(self, blocks, slots, signs, ncls):
        uniq, index = [], []
        for b in blocks:
            k = next((i for i, u in enumerate(uniq) if u is b), None)
            if k is None:
                uniq.append(b)
                k = len(uniq) - 1
            index.append(k)
        self.blocks, self.index = tuple(uniq), tuple(index)
        self.slots, self.signs, self.ncls = tuple(slots), tuple(float(x) for x in signs), ncls
        self.lmax1 = uniq[0].shape[0]
        self.n = ncls * self.lmax1
        self.shape = (self.n, self.n)

    def _place(self, parts):
        """Scatter per-block ``parts`` into a ``(rows, ncls, cols, ncls)`` array."""
        return _place_blocks(tuple(parts), ncls=self.ncls, slots=self.slots, signs=self.signs,
                             index=self.index)

    def dense_host(self):
        """The `(n, n)` matrix as a host numpy array."""
        out = np.zeros((self.lmax1, self.ncls, self.lmax1, self.ncls))
        host = [np.asarray(b) for b in self.blocks]
        for k, (c1, c2) in enumerate(self.slots):
            out[:, c1, :, c2] += host[self.index[k]] * self.signs[k]
        return out.reshape(self.shape)

    def __array__(self, dtype=None, copy=None):
        out = self.dense_host()
        return out if dtype is None else out.astype(dtype)

    def dense_device(self):
        """The ``(n, n)`` matrix on the device."""
        return self._place(self.blocks).reshape(self.shape)

    def matvec(self, v):
        """``M @ v`` for a flat ``(n,)`` vector in ``l * ncls + c`` ordering."""
        v2 = v.reshape((self.lmax1, self.ncls))
        out = [jnp.zeros(self.lmax1, dtype=jnp.result_type(v.dtype, self.blocks[0].dtype))
               for _ in range(self.ncls)]
        for k, (c1, c2) in enumerate(self.slots):
            out[c1] = out[c1] + self.signs[k] * (self.blocks[self.index[k]] @ v2[:, c2])
        return jnp.stack(out, axis=1).reshape(-1)

    def banded(self, weights, theory, beam, *, f32):
        """Binned products with ``kron(weights, I)`` on the left and ``kron(theory, I)`` on the right.

        Returns ``(one_sided, binned)`` with ``one_sided = W M diag(beam)``
        (shape ``(nbpw ncls, n)``) and ``binned = one_sided T`` (shape
        ``(nbpw ncls, nbpw ncls)``), computed per distinct block and then
        placed into the channel grid. ``f32`` selects float32 GEMM operands.
        """
        if f32:
            parts = [jnp.matmul(weights.astype(jnp.float32), b.astype(jnp.float32),
                                precision=jax.lax.Precision.HIGHEST).astype(jnp.float64)
                     for b in self.blocks]
        else:
            parts = [weights @ b for b in self.blocks]
        parts = [p * beam[None, :] for p in parts]
        nb = weights.shape[0]
        one_sided = self._place(parts).reshape((nb * self.ncls, self.n))
        binned = self._place([p @ theory for p in parts]).reshape((nb * self.ncls, nb * self.ncls))
        return one_sided, binned


def _assemble_mcm(dtype, blocks, *, lmax, ncls, slots, signs):
    """Build the MCM from its ``(lmax+1, lmax+1)`` blocks.

    Polarised matrices are returned as a :class:`_BlockMCM` unless
    ``GMASTER_BLOCK_MCM=0``. Dense matrices up to ``_MCM_PIECED_BYTES`` are
    built by one program. Larger ones are kept as row-channel pieces
    (:class:`_RowPieces`): a single matrix-sized transpose fusion cannot be
    autotuned within the GPU memory pool at Nside 4096 spin 2 (18 GiB), since
    the autotuner needs extra copies of the output. Each ``(l1, l2, ncls)``
    piece already has the output's innermost layout, so joining them is a
    concatenation, not a transpose.
    """
    if ncls > 1 and _BLOCK_MCM:
        return _BlockMCM(tuple(b.astype(dtype) for b in blocks), slots, signs, ncls)
    n = ncls * (lmax + 1)
    if n * n * jnp.dtype(dtype).itemsize <= _MCM_PIECED_BYTES:
        # Pass static arguments by keyword: a positional static argument drops
        # JAX's C++ dispatch fast path and costs ~1.5 ms of host time per call.
        return _assemble_mcm_program(blocks, dtype=dtype, lmax=lmax, ncls=ncls, slots=slots,
                                     signs=signs)
    placed = {slot: (block, sign) for block, slot, sign in zip(blocks, slots, signs)}
    rows = []
    for c1 in range(ncls):
        pieces, sg = [], []
        for c2 in range(ncls):
            block, sign = placed.get((c1, c2), (blocks[0], 0.0))
            pieces.append(block)
            sg.append(float(sign))
        rows.append(_mcm_row(tuple(pieces), signs=tuple(sg)))             # (l1, l2, ncls)
    return _RowPieces(rows)


# Scalar MCMs with lmax below this use the exact Wigner-3j recurrence (tested to
# 2e-14); at and above it, Gauss-Legendre quadrature of Legendre products
# (d^l_00 = P_l), which reuses the GEMM form of the polarised path and avoids
# the recurrence's O(n^3) serial offset work.
_TT_QUADRATURE_LMAX = int(os.environ.get("GMASTER_TT_QUADRATURE_LMAX", "48"))


def _coupling_matrix_tt(window_cls, *, lmax):
    """Scalar MCM: Wigner-3j recurrence below ``_TT_QUADRATURE_LMAX``, quadrature above.

    ``window_cls`` is the mask power spectrum up to at least ``2 * lmax``; the
    result is the ``(lmax+1, lmax+1)`` matrix ``M[l1, l2]``.
    """
    lmax = int(lmax)
    if lmax >= _TT_QUADRATURE_LMAX:
        return _coupling_matrix_tt_quadrature(window_cls, lmax=lmax)
    return _coupling_matrix_tt_recurrence(window_cls, lmax=lmax)


@partial(jax.jit, static_argnames=("lmax", "lmax_mask"))
def _tt_quadrature_kernel(window_cls, weights, table, *, lmax, lmax_mask):
    """Scalar MCM from the ``x >= 0`` half of the Gauss-Legendre nodes.

    ``table`` holds ``P_l(x_i)`` for ``l <= lmax_mask`` on the non-negative
    nodes. With ``C(x)`` the mask correlation function, the quadrature
    ``sum_i w_i P_l1(x_i) P_l2(x_i) C(x_i)`` over symmetric nodes equals
    ``sum_{x_i >= 0} w_i P_l1 P_l2 [C(x_i) + (-1)^(l1+l2) C(-x_i)]``, because
    ``P_l(-x) = (-1)^l P_l(x)`` (the caller halves the ``x = 0`` weight). So
    even-``l1 + l2`` pairs contract against ``C(x) + C(-x)`` and odd pairs
    against ``C(x) - C(-x)``: three quarter-size GEMMs over half the nodes.
    Operand width follows :func:`set_coupling_precision`.
    """
    mask_ell = jnp.arange(lmax_mask + 1)
    coefficients = (2 * mask_ell + 1) * window_cls[: lmax_mask + 1] / (4 * jnp.pi)
    mask_sign = jnp.where(mask_ell % 2, -1.0, 1.0)
    north = table @ coefficients
    south = table @ (coefficients * mask_sign)
    plus = weights * (north + south)
    minus = weights * (north - south)
    first = table[:, : lmax + 1]
    even, odd = first[:, 0::2], first[:, 1::2]
    use_f32 = _coupling_f32(lmax)

    def gram(left, weight, right):
        left = (left * weight[:, None]).T
        if use_f32:
            # HIGHEST keeps float32 products off the TF32 tensor-core path.
            return jnp.matmul(
                left.astype(jnp.float32), right.astype(jnp.float32),
                precision=jax.lax.Precision.HIGHEST,
            ).astype(jnp.float64)
        return jnp.matmul(left, right, precision=jax.lax.Precision.HIGHEST)

    even_even = gram(even, plus, even)
    odd_odd = gram(odd, plus, odd)
    even_odd = gram(even, minus, odd)
    size = lmax + 1
    matrix = (
        jnp.zeros((size, size), dtype=window_cls.dtype)
        .at[0::2, 0::2].set(even_even)
        .at[1::2, 1::2].set(odd_odd)
        .at[0::2, 1::2].set(even_odd)
        .at[1::2, 0::2].set(even_odd.T)
    )
    return matrix * (2 * jnp.arange(size) + 1)[None] / 2


def _coupling_matrix_tt_quadrature(window_cls, *, lmax):
    """Scalar MCM by Gauss-Legendre quadrature (agrees with the recurrence to ~1e-11).

    Uses enough nodes to integrate the product of two ``P_l`` (``l <= lmax``)
    and the mask correlation function (``l <= 2 lmax``) exactly.
    """
    lmax = int(lmax)
    lmax_mask = 2 * lmax
    order = (2 * lmax + lmax_mask) // 2 + 1
    nodes, weights = _gauss_legendre(order)
    half = order // 2                     # nodes ascend, so nodes[half:] are the x >= 0 half
    nodes, weights = nodes[half:], np.array(weights[half:])
    if order % 2:
        weights[0] *= 0.5                 # x = 0 is its own mirror and is counted twice below
    window = jnp.asarray(window_cls, dtype=jnp.float64)
    window = jnp.pad(window, (0, max(0, lmax_mask + 1 - int(window.shape[0]))))
    window = window[: lmax_mask + 1]
    # d^l_00 = P_l. The table depends only on the nodes, so it shares the polarised tables' cache.
    table = _wigner_d_shared(jnp.arccos(jnp.asarray(nodes)), 0, 0, lmax_mask)
    return _tt_quadrature_kernel(
        window,
        jnp.asarray(weights, dtype=jnp.float64),
        table,
        lmax=lmax,
        lmax_mask=lmax_mask,
    )


@partial(jax.jit, static_argnames="lmax")
def _coupling_matrix_tt_recurrence(window_cls, *, lmax):
    """Exact scalar MCM from the closed-form Wigner-3j sum (threej_cosmo).

    ``M[l1, l2]`` is a sum over an offset ``o`` of products of the tabulated
    ratio ``g(p) = Gamma(p + 1/2) / (sqrt(pi) Gamma(p + 1))`` and the mask
    power; a term contributes only when ``o <= min(l1, l2)``. Offsets are
    processed ``_OFFSET_CHUNK`` at a time in a ``lax.scan`` with a uniform
    ``(chunk, n, n)`` body, so a single compilation serves every step;
    out-of-range lanes are masked. This costs O(n^3) elementwise work and no
    GEMM, so the only speed lever is operand width: in the float32 arm the
    lookup tables and terms are float32 (relative matrix error ~2e-7) while
    the log-cumsum table and the accumulator stay float64.

    On a GPU the float32 arm runs the CUDA kernel in
    :mod:`gmaster._coupling_tt_cuda`, which avoids the ``(chunk, n, n)``
    temporaries; the scan is the CPU / no-nvcc fallback.
    """
    use_f32 = _coupling_f32(lmax)
    element_dtype = jnp.float32 if use_f32 else window_cls.dtype
    # The accumulator stays float64 even in the float32 arm: each entry sums up to
    # lmax terms, and float32 partial sums would add error beyond the per-term rounding.
    accumulator_dtype = jnp.float64 if use_f32 else window_cls.dtype
    table_dtype = jnp.float64 if use_f32 else window_cls.dtype

    n_ell = lmax + 1
    p = jnp.arange(1, 2 * lmax + 1, dtype=table_dtype)
    log_g = jnp.concatenate(
        [jnp.zeros(1, dtype=table_dtype), jnp.cumsum(jnp.log((p - 0.5) / p))]
    )
    g = jnp.exp(log_g).astype(element_dtype)
    mask_power = (
        window_cls * (2 * jnp.arange(2 * lmax + 1) + 1) / (4 * jnp.pi)
    ).astype(element_dtype)

    if use_f32 and _coupling_tt_cuda.enabled():
        return _coupling_tt_cuda.coupling_tt(
            mask_power.astype(jnp.float32),
            g.astype(jnp.float32),
            lmax=lmax,
        )

    n_chunks = (n_ell + _OFFSET_CHUNK - 1) // _OFFSET_CHUNK
    last_offset = n_ell - 1
    multipoles = jnp.arange(n_ell)
    row = multipoles[:, None]
    column = multipoles[None, :]

    # The summand depends on (l1, l2) only through min, max and their difference.
    # An upper-triangle-only variant is not faster: XLA fuses the lookups into
    # one pass over the full rectangle anyway, and the triangle adds a transpose.
    lower = jnp.minimum(row, column)
    upper = jnp.maximum(row, column)
    lane = jnp.arange(_OFFSET_CHUNK)[:, None, None]

    def add_chunk(matrix, chunk):
        # The last chunk is padded to full width so every step has the same shape;
        # offsets past lmax are clamped for the lookup and then masked out.
        offs = chunk * _OFFSET_CHUNK + lane
        in_range = offs <= last_offset
        offs_g = jnp.minimum(offs, last_offset)
        p_total = upper + offs_g
        term = (
            mask_power[jnp.minimum(upper - lower + 2 * offs_g, 2 * lmax)]
            * g[upper - lower + offs_g]
            * g[offs_g]
            * g[jnp.maximum(lower - offs_g, 0)]
            / (g[p_total] * (2 * p_total + 1))
        )
        contrib = jnp.sum(jnp.where(in_range & (offs <= lower), term, 0), axis=0)
        return matrix + contrib, None

    matrix, _ = jax.lax.scan(
        add_chunk,
        jnp.zeros((n_ell, n_ell), dtype=accumulator_dtype),
        jnp.arange(n_chunks),
    )
    return matrix * (2 * column + 1)


@lru_cache(maxsize=32)
def _toeplitz_exact_pairs(size, l_toeplitz, l_exact, dl_band):
    """Indices of the exact entries consumed by NaMaster's Toeplitz fill."""
    pairs = {(ell, ell) for ell in range(size)}
    pairs.update((ell, l_toeplitz) for ell in range(size))
    pairs.update(
        (ell, column)
        for column in range(l_exact + 1)
        for ell in range(size)
    )
    pairs.update(
        (ell, ell + offset)
        for offset in range(dl_band + 1)
        for ell in range(size - offset)
    )
    encoded = np.fromiter(
        (row * size + column for row, column in pairs), dtype=np.int64
    )
    encoded.sort()
    return encoded // size, encoded % size


@partial(
    jax.jit,
    static_argnames=("lmax", "l_toeplitz", "l_exact", "dl_band"),
)
def _coupling_matrix_tt_toeplitz(
    window_cls, *, lmax, l_toeplitz, l_exact, dl_band
):
    """Scalar MCM under the Toeplitz approximation.

    Evaluates exactly (by the Wigner-3j sum) only the entries that NaMaster's
    Toeplitz fill reads (see :func:`_toeplitz_exact_pairs`), then fills the
    rest with :func:`_apply_toeplitz`.
    """
    size = lmax + 1
    rows, columns = _toeplitz_exact_pairs(
        size, l_toeplitz, l_exact, dl_band
    )
    rows = jnp.asarray(rows)
    columns = jnp.asarray(columns)
    lower = jnp.minimum(rows, columns)
    upper = jnp.maximum(rows, columns)

    p = jnp.arange(1, 2 * lmax + 1, dtype=window_cls.dtype)
    log_g = jnp.concatenate(
        [jnp.zeros(1, dtype=window_cls.dtype), jnp.cumsum(jnp.log((p - 0.5) / p))]
    )
    g = jnp.exp(log_g)
    mask_power = window_cls * (2 * jnp.arange(2 * lmax + 1) + 1) / (4 * jnp.pi)

    def add_offset(offset, values):
        mask_ell = upper - lower + 2 * offset
        total = upper + offset
        term = (
            mask_power[jnp.minimum(mask_ell, 2 * lmax)]
            * g[upper - lower + offset]
            * g[offset]
            * g[jnp.maximum(lower - offset, 0)]
            / (g[total] * (2 * total + 1))
        )
        return values + jnp.where(offset <= lower, term, 0)

    values = jax.lax.fori_loop(
        0, size, add_offset, jnp.zeros(len(rows), dtype=window_cls.dtype)
    )
    exact = jnp.zeros((size, size), dtype=window_cls.dtype)
    exact = exact.at[rows, columns].set(values)
    exact = exact.at[columns, rows].set(values)
    correlation = _apply_toeplitz(exact, l_toeplitz, l_exact, dl_band)
    return correlation * (2 * jnp.arange(size) + 1)[None]


@lru_cache(maxsize=16)
def _gauss_legendre(order):
    """Cached Gauss-Legendre nodes (ascending) and weights on [-1, 1]."""
    return roots_legendre(order)


@partial(jax.jit, static_argnames=("m", "n", "lmax"))
def _wigner_d_table(beta, *, m, n, lmax):
    """Wigner d^l_mn(beta) for ``l <= lmax``, shape ``(len(beta), lmax + 1)``.

    Uses the Jacobi-polynomial three-term recurrence in ``l`` starting at
    ``l = max(|m|, |n|)``; entries below that are zero.
    """
    minimum = max(abs(m), abs(n))
    if minimum > lmax:
        return jnp.zeros((len(beta), lmax + 1), dtype=beta.dtype)

    mu = abs(m - n)
    nu = abs(m + n)
    degrees = jnp.arange(lmax - minimum + 1)
    normalisation = jnp.exp(
        0.5
        * (
            gammaln(degrees + 1)
            + gammaln(degrees + mu + nu + 1)
            - gammaln(degrees + mu + 1)
            - gammaln(degrees + nu + 1)
        )
    )
    prefactor = (
        (-1.0) ** ((n - m + mu) // 2)
        * jnp.sin(beta / 2) ** mu
        * jnp.cos(beta / 2) ** nu
    )
    previous = jnp.ones_like(beta)
    if minimum == lmax:
        table = jnp.zeros((len(beta), lmax + 1), dtype=beta.dtype)
        return table.at[:, minimum].set(prefactor * normalisation[0])

    current = ((mu - nu) + (mu + nu + 2) * jnp.cos(beta)) / 2

    def advance(state, degree):
        previous, current = state
        order = jnp.asarray(degree, dtype=beta.dtype)
        total = mu + nu
        following = (
            (2 * order + total - 1)
            * (
                (2 * order + total)
                * (2 * order + total - 2)
                * jnp.cos(beta)
                + mu**2
                - nu**2
            )
            * current
            - 2
            * (order + mu - 1)
            * (order + nu - 1)
            * (2 * order + total)
            * previous
        ) / (
            2 * order * (order + total) * (2 * order + total - 2)
        )
        return (current, following), following

    # The scan output is a second full table; free device memory for it now.
    _config.make_room(int(len(beta)) * int(lmax + 1) * 8 * 4)
    # Stack the scan outputs rather than writing columns into the table: a
    # traced column index would make XLA copy the whole table every step.
    # Each step is small (one value per node), so unroll=8 amortises launches.
    _, rest = jax.lax.scan(
        advance, (previous, current), jnp.arange(2, lmax - minimum + 1),
        unroll=8,
    )
    values = jnp.concatenate([previous[None], current[None], rest], axis=0)
    values = (values * normalisation[:, None]) * prefactor[None, :]
    table = jnp.zeros((len(beta), lmax + 1), dtype=beta.dtype)
    return table.at[:, minimum:].set(values.T)


@partial(jax.jit, static_argnames=("lmax", "lmax_mask"))
def _general_coupling_matrix_quadrature(
    mask_cls,
    weights,
    tables,
    *,
    lmax,
    lmax_mask,
):
    """Even- and odd-parity general coupling matrices by Gauss-Legendre quadrature.

    With ``tables = (d^l1_{n1 n2}, d^l2_{-s1,-s2}, d^L_{s1-n1, s2-n2})`` on the
    nodes, the mask correlation ``xi(x) = sum_L (2L+1)/(4 pi) C^mask_L d^L(x)``
    gives ``M[l1, l2] = (2 l2 + 1)/2 sum_i w_i d^l1(x_i) xi(x_i) d^l2(x_i)``.
    The parity split uses the ``(-1)^L``-signed mask correlation. Returns a
    ``(2, lmax+1, lmax+1)`` array (even, odd).
    """
    first, second, mask = tables
    mask_ell = jnp.arange(lmax_mask + 1)
    coefficients = (
        (2 * mask_ell + 1) * mask_cls[: lmax_mask + 1] / (4 * jnp.pi)
    )
    column_factor = 2 * jnp.arange(lmax + 1) + 1
    use_f32 = _coupling_f32(lmax)
    first_l = first.T.astype(jnp.float32) if use_f32 else first.T
    second_l = second.astype(jnp.float32) if use_f32 else second
    weights_l = weights.astype(jnp.float32) if use_f32 else weights

    def integrate(correlation):
        if use_f32:
            # HIGHEST keeps the products off the TF32 path: DEFAULT precision is
            # faster but raises the matrix error from ~2e-6 to ~5e-4.
            left = first_l * (weights_l * correlation.astype(jnp.float32))
            return (
                jnp.matmul(left, second_l, precision=jax.lax.Precision.HIGHEST)
                .astype(jnp.float64)
                * column_factor[None]
                / 2
            )
        left = first_l * (weights * correlation)
        return left @ second_l * column_factor[None] / 2

    total = integrate(mask @ coefficients)
    mask_sign = jnp.where(mask_ell % 2, -1, 1)
    signed = integrate(mask @ (coefficients * mask_sign))
    multipoles = jnp.arange(lmax + 1)
    pair_sign = jnp.where(
        (multipoles[:, None] + multipoles[None]) % 2, -1, 1
    )
    signed *= pair_sign
    return jnp.stack(((total + signed) / 2, (total - signed) / 2))


_WD_TRIPLE_CACHE = {}
_config._ROOM_HOOKS.append(_WD_TRIPLE_CACHE.clear)   # evictable: ~9 GiB at Nside 4096 spin 2
# Wigner-d quadrature tables depend only on the geometry and are cached between
# workspace builds; rebuilding them (a sequential scan over l) dominates the
# polarised coupling stage (e.g. the 3.6 GiB of tables at Nside 2048 spin 2).
# They are kept while their size is at most max(2 GiB, this fraction of the pool).
_WD_CACHE_KEEP_FRACTION = float(os.environ.get("GMASTER_WD_CACHE_FRACTION", "0.10"))


# The cache is also dropped when free pool memory is less than this multiple of
# its size, so later large allocations (e.g. at Nside 4096 spin 2) still fit.
_WD_POOL_HEADROOM = float(os.environ.get("GMASTER_WD_POOL_HEADROOM", "4.0"))


def _pool_limit_and_free():
    """``(limit, free)`` bytes of the device memory pool (``(0, None)`` if unknown).

    A preallocated pool reports its limit. The non-preallocated ``cuda_async``
    allocator reports 0; then the driver's numbers are used instead (90 % of
    the device as the limit, and the driver's free memory, which is the
    conservative figure on a shared GPU).
    """
    try:
        stats = jax.devices()[0].memory_stats() or {}
    except Exception:  # noqa: BLE001 - a backend without pool accounting
        return 0, None
    limit = stats.get("bytes_limit") or 0
    if limit:
        return limit, limit - (stats.get("bytes_in_use") or 0)
    from ._sht import march_v2 as _march_v2

    info = _march_v2.device_memory_info()
    if info is None:
        return 0, None
    return int(0.9 * info[1]), info[0]


def _pool_is_tight(total):
    """True when free pool memory is below ``_WD_POOL_HEADROOM * total``."""
    limit, free = _pool_limit_and_free()
    return bool(limit) and free is not None and free < _WD_POOL_HEADROOM * total


def _wd_cache_keep_bytes():
    """Largest Wigner-d cache size worth keeping between workspaces."""
    limit, _ = _pool_limit_and_free()
    return max(2 * 1024 ** 3, int(limit * _WD_CACHE_KEEP_FRACTION))


def _drop_large_wigner_cache():
    """Drop cached quadrature tables that are too large to keep between workspaces.

    Otherwise, at large Nside, the tables outlive their workspace and leave no
    room for the next field's transforms.
    """
    total = sum(int(t.size) * int(t.dtype.itemsize) for t in _WD_TRIPLE_CACHE.values())
    if total and (total > _wd_cache_keep_bytes() or _pool_is_tight(total)):
        _WD_TRIPLE_CACHE.clear()


def _wigner_d_shared(beta, m, n, order):
    """Cached :func:`_wigner_d_table`, sharing ``(m, n)`` and ``(-m, -n)``.

    ``d^l_{-m,-n} = (-1)^(m-n) d^l_{m,n}``, so the two are the same table when
    ``m - n`` is even (the case for every pair the couplings use); odd
    differences are not folded. The cache key uses the node count rather than
    the nodes, which is valid because callers always pass the Gauss-Legendre
    nodes of that count.
    """
    if (m - n) % 2 == 0 and (m < 0 or (m == 0 and n < 0)):
        m, n = -m, -n
    key = (m, n, order, len(beta))
    table = _WD_TRIPLE_CACHE.get(key)
    if table is None:
        table = _wigner_d_table(beta, m=m, n=n, lmax=order)
        if getattr(table, "is_fully_addressable", True):
            if len(_WD_TRIPLE_CACHE) > 24:
                _WD_TRIPLE_CACHE.clear()
            _WD_TRIPLE_CACHE[key] = table
    return table


def _wigner_d_triple(beta, *, s1, s2, n1, n2, lmax, lmax_mask):
    """The three Wigner-d tables used by :func:`_general_coupling_matrix_quadrature`."""
    return (
        _wigner_d_shared(beta, n1, n2, lmax),
        _wigner_d_shared(beta, -s1, -s2, lmax),
        _wigner_d_shared(beta, s1 - n1, s2 - n2, lmax_mask),
    )


# Above GMASTER_WD_STREAM_GIB of float64 Wigner-d tables, the quadrature is summed
# over blocks of GMASTER_WD_STREAM_NODES nodes instead of holding whole tables.
# The three tables are 38.6 GiB at Nside 8192 spin 2 (too large to fit next to
# the field) and 9.7 GiB at Nside 4096, which keeps the cached whole-table path.
_WD_STREAM_BYTES = int(float(os.environ.get("GMASTER_WD_STREAM_GIB", "16")) * 1024 ** 3)
_WD_STREAM_NODES = int(os.environ.get("GMASTER_WD_STREAM_NODES", "4096"))


def _folded(m, n):
    """The ``(m, n)`` key under which :func:`_wigner_d_shared` stores a table."""
    if (m - n) % 2 == 0 and (m < 0 or (m == 0 and n < 0)):
        return -m, -n
    return m, n


@partial(jax.jit, static_argnames=("pairs", "lmax", "lmax_mask"), donate_argnames="acc")
def _coupling_quadrature_block(acc, mask_cls, beta, weights, *, pairs, lmax, lmax_mask):
    """Add one node block's contribution to the quadrature into ``acc`` (donated)."""
    built = {}
    tables = []
    for (m, n), order in zip(pairs, (lmax, lmax, lmax_mask)):
        key = (m, n, order)
        if key not in built:
            built[key] = _wigner_d_table(beta, m=m, n=n, lmax=order)
        tables.append(built[key])
    return acc + _general_coupling_matrix_quadrature(
        mask_cls, weights, tuple(tables), lmax=lmax, lmax_mask=lmax_mask)


def _general_coupling_matrix_streamed(mask_cls, nodes, weights, *, pairs, lmax, lmax_mask):
    """Quadrature summed over node blocks, so only one block's tables are live.

    The last block is padded with zero-weight nodes at the equator, which
    contribute exactly zero.
    """
    block = _WD_STREAM_NODES
    total = -(-len(nodes) // block) * block
    beta = np.full(total, np.pi / 2)
    beta[: len(nodes)] = np.arccos(nodes)
    padded = np.zeros(total)
    padded[: len(weights)] = weights
    mask_cls = jnp.asarray(mask_cls)
    acc = jnp.zeros((2, lmax + 1, lmax + 1))
    for lo in range(0, total, block):
        acc = _coupling_quadrature_block(
            acc, mask_cls, jnp.asarray(beta[lo : lo + block]),
            jnp.asarray(padded[lo : lo + block]),
            pairs=pairs, lmax=lmax, lmax_mask=lmax_mask,
        )
    return acc


def _general_coupling_matrix(
    mask_cls, *, s1, s2, n1, n2, lmax, lmax_mask, tables=None
):
    """Even/odd general coupling matrices, ``(2, lmax+1, lmax+1)``.

    Uses enough Gauss-Legendre nodes to integrate the triple product exactly,
    with cached whole tables or, above ``_WD_STREAM_BYTES``, node streaming.
    """
    order = (2 * lmax + lmax_mask) // 2 + 1
    nodes, weights = _gauss_legendre(order)
    if tables is None and order * (2 * lmax + lmax_mask + 3) * 8 > _WD_STREAM_BYTES:
        _WD_TRIPLE_CACHE.clear()
        pairs = (_folded(n1, n2), _folded(-s1, -s2), _folded(s1 - n1, s2 - n2))
        return _general_coupling_matrix_streamed(
            mask_cls, nodes, weights, pairs=pairs, lmax=lmax, lmax_mask=lmax_mask)
    nodes = jnp.asarray(nodes)
    if tables is None:
        tables = _wigner_d_triple(
            jnp.arccos(nodes), s1=s1, s2=s2, n1=n1, n2=n2, lmax=lmax,
            lmax_mask=lmax_mask,
        )
    return _general_coupling_matrix_quadrature(
        mask_cls,
        jnp.asarray(weights),
        tables,
        lmax=lmax,
        lmax_mask=lmax_mask,
    )


def _coupling_matrices_spin2(window_cls, *, lmax, need_te=True, need_ee=True):
    """``(te, even, odd)`` coupling blocks for spin-0/spin-2 fields.

    ``te`` is the spin-0 x spin-2 block; ``even``/``odd`` are ``M+``/``M-`` of
    spin-2 x spin-2. Blocks not requested (``need_te``/``need_ee``) are None,
    since each is a full quadrature.
    """
    mixed = even = odd = None
    if need_te:
        mixed = _general_coupling_matrix(
            window_cls, s1=0, s2=2, n1=0, n2=2, lmax=lmax, lmax_mask=2 * lmax,
        ).sum(axis=0)
    if need_ee:
        even, odd = _general_coupling_matrix(
            window_cls, s1=2, s2=2, n1=2, n2=2, lmax=lmax, lmax_mask=2 * lmax,
        )
    return mixed, even, odd


@partial(jax.jit, static_argnames="lmax")
def _coupling_matrices_pure_quadrature(
    window_cls, nodes, weights, *, lmax
):
    """Standard and pure-E/B spin-2 coupling blocks by Gauss-Legendre quadrature.

    Purification brings in mask derivatives, i.e. mask correlations with spin
    weights 1 and 2 (``c_mn`` / signed ``s_mn`` below) and ``l``-dependent
    factors on the rows. Returns eight ``(lmax+1, lmax+1)`` blocks:
    ``(standard_te, standard_even, standard_odd, pure_te, one_even, one_odd,
    two_even, two_odd)``, where "one" / "two" are the blocks with one and two
    purified fields.
    """
    beta = jnp.arccos(nodes)
    mask_ell = jnp.arange(2 * lmax + 1)
    mask_first = jnp.sqrt(mask_ell * (mask_ell + 1))
    mask_second = jnp.sqrt(
        jnp.maximum(
            (mask_ell + 2)
            * (mask_ell + 1)
            * mask_ell
            * (mask_ell - 1),
            0,
        )
    )
    row = jnp.arange(lmax + 1)
    row_first = jnp.where(
        row > 1, 2 / jnp.sqrt(jnp.maximum((row + 2) * (row - 1), 1)), 0
    )
    row_second = jnp.where(
        row > 1,
        1
        / jnp.sqrt(
            jnp.maximum((row + 2) * (row + 1) * row * (row - 1), 1)
        ),
        0,
    )
    coefficients = (2 * mask_ell + 1) * window_cls / (4 * jnp.pi)
    mask_sign = jnp.where(mask_ell % 2, -1, 1)

    def correlations(m, n, factor=1):
        table = _wigner_d_table(beta, m=m, n=n, lmax=2 * lmax)
        weighted = coefficients * factor
        return table @ weighted, table @ (weighted * mask_sign)

    c00, s00 = correlations(0, 0)
    c10, s10 = correlations(1, 0, mask_first)
    c20, s20 = correlations(2, 0, mask_second)
    c11, s11 = correlations(1, 1, mask_first**2)
    c12, s12 = correlations(1, 2, mask_first * mask_second)
    c22, s22 = correlations(2, 2, mask_second**2)
    c01, _ = correlations(0, 1, mask_first)
    c02, _ = correlations(0, 2, mask_second)

    d00 = _wigner_d_table(beta, m=0, n=0, lmax=lmax)
    d01 = _wigner_d_table(beta, m=0, n=1, lmax=lmax)
    d02 = _wigner_d_table(beta, m=0, n=2, lmax=lmax)
    d10 = _wigner_d_table(beta, m=1, n=0, lmax=lmax)
    d11 = _wigner_d_table(beta, m=1, n=1, lmax=lmax)
    d12 = _wigner_d_table(beta, m=1, n=2, lmax=lmax)
    d22 = _wigner_d_table(beta, m=2, n=2, lmax=lmax)
    right_temperature = _wigner_d_table(beta, m=0, n=-2, lmax=lmax)
    right_spin = _wigner_d_table(beta, m=-2, n=-2, lmax=lmax)
    column_factor = 2 * row + 1

    def integrate(left, right):
        return (left.T * weights) @ right * column_factor[None] / 2

    temperature_left = (
        d02 * c00[:, None]
        + d01 * c01[:, None] * row_first[None]
        + d00 * c02[:, None] * row_second[None]
    )
    temperature = integrate(temperature_left, right_temperature)

    one_left = (
        d22 * c00[:, None]
        + d12 * c10[:, None] * row_first[None]
        + d02 * c20[:, None] * row_second[None]
    )
    one_signed_left = (
        d22 * s00[:, None]
        + d12 * s10[:, None] * row_first[None]
        + d02 * s20[:, None] * row_second[None]
    )
    two_left = (
        d22 * c00[:, None]
        + 2 * d12 * c10[:, None] * row_first[None]
        + 2 * d02 * c20[:, None] * row_second[None]
        + d11 * c11[:, None] * row_first[None] ** 2
        + 2 * d10 * c12[:, None] * row_first[None] * row_second[None]
        + d00 * c22[:, None] * row_second[None] ** 2
    )
    two_signed_left = (
        d22 * s00[:, None]
        + 2 * d12 * s10[:, None] * row_first[None]
        + 2 * d02 * s20[:, None] * row_second[None]
        + d11 * s11[:, None] * row_first[None] ** 2
        + 2 * d10 * s12[:, None] * row_first[None] * row_second[None]
        + d00 * s22[:, None] * row_second[None] ** 2
    )
    pair_sign = jnp.where((row[:, None] + row[None]) % 2, -1, 1)

    def parity(left, signed_left):
        total = integrate(left, right_spin)
        signed = integrate(signed_left, right_spin) * pair_sign
        return (total + signed) / 2, (total - signed) / 2

    standard_temperature = integrate(d02 * c00[:, None], right_temperature)
    standard_even, standard_odd = parity(
        d22 * c00[:, None], d22 * s00[:, None]
    )
    one_even, one_odd = parity(one_left, one_signed_left)
    two_even, two_odd = parity(two_left, two_signed_left)
    return (
        standard_temperature,
        standard_even,
        standard_odd,
        temperature,
        one_even,
        one_odd,
        two_even,
        two_odd,
    )


def _coupling_matrices_spin2_pure(window_cls, *, lmax):
    """All eight blocks of :func:`_coupling_matrices_pure_quadrature`."""
    nodes, weights = _gauss_legendre(2 * lmax + 1)
    return _coupling_matrices_pure_quadrature(
        window_cls,
        jnp.asarray(nodes),
        jnp.asarray(weights),
        lmax=lmax,
    )


def _coupling_matrices_pure(window_cls, *, lmax):
    """The five purified blocks ``(te, one_even, one_odd, two_even, two_odd)``."""
    return _coupling_matrices_spin2_pure(window_cls, lmax=lmax)[3:]


def get_general_coupling_matrix(pcl_mask, s1, s2, n1, n2, parity="all"):
    """Return a general mode-coupling matrix, as in ``pymaster``.

    Computes

        M[l, l'] = sum_l'' (2l' + 1)(2l'' + 1) / (4 pi) C_l'' P_{l+l'+l''}
                   (l l' l''; n1 -s1 s1-n1) (l l' l''; n2 -s2 s2-n2)

    with ``P_L = 1`` for ``parity="all"``, ``(1 + (-1)^L)/2`` for ``"even"``
    and ``(1 - (-1)^L)/2`` for ``"odd"``.

    Parameters
    ----------
    pcl_mask : array_like
        1D mask (cross-)power spectrum ``C_l``; its length ``nl`` sets the
        multipole range ``0 .. nl-1``.
    s1, s2, n1, n2 : int
        Spin indices in the formula above.
    parity : {"all", "even", "odd", "both"}, optional
        Parity selection. ``"both"`` returns the even and odd matrices stacked.

    Returns
    -------
    jax.Array
        ``(nl, nl)`` matrix, or ``(2, nl, nl)`` (even, odd) for ``"both"``.
        Rows and columns below ``max(s1, s2, n1, n2)`` are zero.
    """
    if parity not in ("all", "even", "odd", "both"):
        raise ValueError(
            '`parity` must be "all", "even", "odd", or "both".'
        )
    pcl_mask = jnp.asarray(pcl_mask)
    if pcl_mask.ndim != 1:
        raise ValueError("pcl_mask must be one-dimensional")
    lmax = len(pcl_mask) - 1
    matrices = _general_coupling_matrix(
        pcl_mask,
        s1=int(s1),
        s2=int(s2),
        n1=int(n1),
        n2=int(n2),
        lmax=lmax,
        lmax_mask=lmax,
    )
    lstart = max(int(s1), int(s2), int(n1), int(n2))
    keep = jnp.arange(lmax + 1) >= lstart
    matrices = jnp.where(keep[None, :, None] & keep[None, None, :], matrices, 0)
    if parity == "both":
        return matrices
    if parity == "even":
        return matrices[0]
    if parity == "odd":
        return matrices[1]
    return matrices.sum(axis=0)


def _toeplitz_sanity(l_toeplitz, l_exact, dl_band, lmax, fields=()):
    """Validate Toeplitz parameters as NaMaster does (no-op when ``l_toeplitz <= 0``)."""
    if l_toeplitz <= 0:
        return
    if any(field.pure_e or field.pure_b for field in fields):
        raise ValueError("Can't use Toeplitz approximation with purification.")
    if l_exact <= 0 or dl_band < 0:
        raise ValueError("`l_exact` and `dl_band` must be positive numbers")
    if l_exact > l_toeplitz:
        raise ValueError("`l_exact` must be `<= l_toeplitz")
    if l_toeplitz >= lmax or l_exact >= lmax or dl_band >= lmax:
        raise ValueError(
            "`l_toeplitz`, `l_exact` and `dl_band` must be smaller than `l_max`"
        )


def _apply_toeplitz(matrix, l_toeplitz, l_exact, dl_band):
    """Toeplitz approximation of a scalar MCM (Louis et al. 2020), as in NaMaster.

    The correlation matrix ``R[l, l'] = M[l, l'] / sqrt(M[l, l] M[l', l'])`` is
    taken as Toeplitz and read from column ``l_toeplitz``. Columns up to
    ``l_exact``, the diagonal, and the ``dl_band`` sub-diagonals for columns
    below ``l_toeplitz`` keep their exact values; the result is symmetrised
    from its lower triangle. No-op when ``l_toeplitz <= 0``.
    """
    if l_toeplitz <= 0:
        return matrix
    matrix = jnp.asarray(matrix)
    size = len(matrix)
    diagonal = jnp.abs(jnp.diag(matrix))
    denominator = jnp.sqrt(diagonal * diagonal[l_toeplitz])
    correlation = jnp.where(denominator > 0, matrix[:, l_toeplitz] / denominator, 0)
    indices = (
        jnp.abs(jnp.arange(size)[:, None] - jnp.arange(size)[None])
        + l_toeplitz
    ) % size
    root = jnp.sqrt(diagonal[:, None] * diagonal[None])
    pure_toeplitz = correlation[indices] * root
    result = pure_toeplitz.at[:, : l_exact + 1].set(
        matrix[:, : l_exact + 1]
    )
    for offset in range(dl_band + 1):
        index = jnp.arange(size - offset)
        result = result.at[index + offset, index].set(jnp.diag(matrix, offset))
    result = result.at[l_toeplitz:, l_toeplitz:].set(
        pure_toeplitz[l_toeplitz:, l_toeplitz:]
    )
    diagonal_index = jnp.arange(size)
    result = result.at[diagonal_index, diagonal_index].set(jnp.diag(matrix))
    for offset in range(1, dl_band + 1):
        count = min(size - offset, l_toeplitz + 1)
        index = jnp.arange(count)
        result = result.at[index + offset, index].set(
            jnp.diag(matrix, offset)[:count]
        )
    column_correlation = jnp.where(
        diagonal > 0,
        matrix[:, l_exact] / jnp.sqrt(diagonal * diagonal[l_exact]),
        0,
    )
    source_start = size - 1 - l_toeplitz + l_exact
    for offset in range(l_toeplitz - l_exact + 1):
        column = offset + l_exact
        target_start = size - 1 - l_toeplitz + column
        values = column_correlation[source_start : size - offset]
        result = result.at[target_start:, column].set(
            values * jnp.sqrt(diagonal[column] * diagonal[target_start:])
        )
    lower = jnp.tril(result)
    return lower + jnp.tril(result, -1).T


def get_master_coefficients(
    pcl_mask,
    lmax,
    spin1,
    spin2,
    is_teb=False,
    pure_any=False,
    l_toeplitz=-1,
    l_exact=-1,
    dl_band=-1,
):
    """Return the distinct MASTER coupling blocks for a spin pair.

    Parameters
    ----------
    pcl_mask : array_like
        Mask (cross-)power spectrum, ``(nl_mask,)`` or ``(n_masks, nl_mask)``.
    lmax : int
        Maximum multipole of the blocks.
    spin1, spin2 : int
        Field spins.
    is_teb : bool, optional
        Also build the blocks of a joint (spin-0, spin-s) T/E/B workspace.
    pure_any : bool, optional
        Include the pure-E/B blocks (spin 2 only).
    l_toeplitz, l_exact, dl_band : int, optional
        Toeplitz approximation parameters (disabled when ``l_toeplitz <= 0``).

    Returns
    -------
    dict
        ``"00"`` (spin-0 x spin-0), ``"0s"`` (spin-0 x spin-s), ``"pp"`` and
        ``"mm"`` (the even ``M+`` and odd ``M-`` spin-s x spin-s blocks), each
        None when not applicable. Blocks are divided by ``2 l' + 1``. The
        ``"0s"`` / ``"pp"`` / ``"mm"`` entries carry a leading purification-
        level axis (length 1 without purification; 2 or 3 with it), and a
        mask axis before the last two when ``pcl_mask`` is 2D. Metadata keys
        ``pure_any``, ``toeplitz``, ``spins``, ``lmax`` and ``lmax_mask`` are
        also set.
    """
    spin1, spin2, lmax = int(spin1), int(spin2), int(lmax)
    if is_teb and not (spin1 == 0 and spin2 != 0):
        raise ValueError("is_teb is only valid for spin1=0 and spin2!=0")
    if pure_any and spin1 != 2 and spin2 != 2:
        raise ValueError("pure_any is only valid for spin 2")
    if pure_any and (spin1 not in (0, 2) or spin2 not in (0, 2)):
        raise ValueError("Purification is only implemented for spin 2")
    _toeplitz_sanity(l_toeplitz, l_exact, dl_band, lmax)
    masks = jnp.asarray(pcl_mask)
    one_dimensional = masks.ndim == 1
    if one_dimensional:
        masks = masks[None]
    if masks.ndim != 2:
        raise ValueError("pcl_mask must be one- or two-dimensional")
    lmax_mask = masks.shape[1] - 1
    columns = (2 * jnp.arange(lmax + 1) + 1)[None]

    has_00 = (spin1 == spin2 == 0) or is_teb
    has_0s = ((spin1 != spin2) and (spin1 == 0 or spin2 == 0)) or is_teb
    has_ss = (spin1 != 0 and spin2 != 0) or is_teb

    def standard(mask):
        padded = jnp.pad(mask, (0, max(0, 2 * lmax + 1 - len(mask))))
        padded = padded[: 2 * lmax + 1]
        spin2_pure = (
            _coupling_matrices_spin2_pure(padded, lmax=lmax)
            if pure_any
            else None
        )
        result = {}
        if has_00:
            result["00"] = (
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
        if has_0s:
            if spin1 in (0, 2) and spin2 in (0, 2):
                mixed, _, _ = (
                    spin2_pure[:3]
                    if spin2_pure is not None
                    else _coupling_matrices_spin2(padded, lmax=lmax, need_ee=False)
                )
                result["0s"] = mixed / columns
            else:
                pair = (0, spin2) if spin1 == 0 else (spin1, 0)
                result["0s"] = _general_coupling_matrix(
                    mask,
                    s1=pair[0], s2=pair[1], n1=pair[0], n2=pair[1],
                    lmax=lmax, lmax_mask=lmax_mask,
                ).sum(axis=0) / columns
        if has_ss:
            ss1 = spin2 if is_teb else spin1
            ss2 = spin2 if is_teb else spin2
            if ss1 == ss2 == 2:
                _, even, odd = (
                    spin2_pure[:3]
                    if spin2_pure is not None
                    else _coupling_matrices_spin2(padded, lmax=lmax, need_te=False)
                )
                result["pp"], result["mm"] = even / columns, odd / columns
            else:
                matrices = _general_coupling_matrix(
                    mask,
                    s1=ss1, s2=ss2, n1=ss1, n2=ss2,
                    lmax=lmax, lmax_mask=lmax_mask,
                ) / columns
                result["pp"], result["mm"] = matrices[0], matrices[1]
        if l_toeplitz > 0:
            for name in ("0s", "pp", "mm"):
                if name in result:
                    result[name] = _apply_toeplitz(
                        result[name], l_toeplitz, l_exact, dl_band
                    )
        if pure_any:
            pure = spin2_pure[3:]
            pure = tuple(matrix / columns for matrix in pure)
            if has_0s:
                result["0s"] = jnp.stack((result["0s"], pure[0]))
            if has_ss:
                result["pp"] = jnp.stack((result["pp"], pure[1], pure[3]))
                result["mm"] = jnp.stack((result["mm"], pure[2], pure[4]))
        else:
            if has_0s:
                result["0s"] = result["0s"][None]
            if has_ss:
                result["pp"] = result["pp"][None]
                result["mm"] = result["mm"][None]
        return result

    values = [standard(mask) for mask in masks]
    output = {}
    for name, present in (("00", has_00), ("0s", has_0s), ("pp", has_ss), ("mm", has_ss)):
        if not present:
            output[name] = None
            continue
        arrays = [value[name] for value in values]
        output[name] = arrays[0] if one_dimensional else jnp.stack(arrays, axis=-3)
    output.update(
        pure_any=bool(pure_any),
        toeplitz={
            "l_toeplitz": l_toeplitz,
            "l_exact": l_exact,
            "dl_band": dl_band,
        },
        spins=(spin1, spin2),
        lmax=lmax,
        lmax_mask=lmax_mask,
    )
    return output


def _anisotropic_coupling_matrix(
    spin1,
    spin2,
    aniso1,
    aniso2,
    spectra,
    *,
    lmax,
    lmax_mask,
):
    """Full MCM for fields with anisotropic (spin-weighted) mask components.

    ``spectra`` holds the mask cross-spectra (``"00"``, ``"0e"``, ``"e0"``,
    ``"ee"``, ...) between the isotropic and spin-``s`` mask components; each
    enters a general coupling with the appropriate spin arguments and sign,
    and the results are combined into the ``(ncls (lmax+1))^2`` matrix in
    ``l * ncls + c`` ordering.
    """
    def coupling(spectrum, s1, s2, n1=spin1, n2=spin2):
        return _general_coupling_matrix(
            spectrum,
            s1=s1,
            s2=s2,
            n1=n1,
            n2=n2,
            lmax=lmax,
            lmax_mask=lmax_mask,
        )

    zero = jnp.zeros((2, lmax + 1, lmax + 1), dtype=spectra["00"].dtype)
    m00 = coupling(spectra["00"], spin1, spin2)
    m0e = m0b = me0 = mb0 = mee = meb = mbe = mbb = zero
    if aniso2:
        sign = (-1) ** spin2
        m0e = coupling(spectra["0e"] * sign, spin1, -spin2)
        m0b = coupling(spectra["0b"] * sign, spin1, -spin2)
    if aniso1:
        sign = (-1) ** spin1
        me0 = coupling(spectra["e0"] * sign, -spin1, spin2)
        mb0 = coupling(spectra["b0"] * sign, spin1, -spin2)
    if aniso1 and aniso2:
        sign = (-1) ** (spin1 + spin2)
        mee = coupling(spectra["ee"] * sign, -spin1, -spin2)
        meb = coupling(spectra["eb"] * sign, -spin1, -spin2)
        mbe = coupling(spectra["be"] * sign, -spin1, -spin2)
        mbb = coupling(spectra["bb"] * sign, -spin1, -spin2)

    ncls = (1 if spin1 == 0 else 2) * (1 if spin2 == 0 else 2)
    matrix = jnp.zeros((lmax + 1, ncls, lmax + 1, ncls))
    if spin1 == 0:
        matrix = matrix.at[:, 0, :, 0].set(m00[0] - m0e[0])
        matrix = matrix.at[:, 0, :, 1].set(-m0b[0])
        matrix = matrix.at[:, 1, :, 0].set(-m0b[0])
        matrix = matrix.at[:, 1, :, 1].set(m00[0] + m0e[0])
    elif spin2 == 0:
        matrix = matrix.at[:, 0, :, 0].set(m00[0] - me0[0])
        matrix = matrix.at[:, 0, :, 1].set(-mb0[0])
        matrix = matrix.at[:, 1, :, 0].set(-mb0[0])
        matrix = matrix.at[:, 1, :, 1].set(m00[0] + me0[0])
    else:
        matrix = matrix.at[:, 0, :, 0].set(m00[0] - m0e[0] - me0[0] + mee[0] + mbb[1])
        matrix = matrix.at[:, 0, :, 1].set(-m0b[0] + meb[0] - mb0[1] - mbe[1])
        matrix = matrix.at[:, 0, :, 2].set(-mb0[0] + mbe[0] - m0b[1] - meb[1])
        matrix = matrix.at[:, 0, :, 3].set(mbb[0] + m00[1] + m0e[1] + me0[1] + mee[1])
        matrix = matrix.at[:, 1, :, 0].set(-m0b[0] + meb[0] + mb0[1] - mbe[1])
        matrix = matrix.at[:, 1, :, 1].set(m00[0] + m0e[0] - me0[0] - mee[0] - mbb[1])
        matrix = matrix.at[:, 1, :, 2].set(mbb[0] - m00[1] + m0e[1] - me0[1] + mee[1])
        matrix = matrix.at[:, 1, :, 3].set(-mb0[0] - mbe[0] + m0b[1] + meb[1])
        matrix = matrix.at[:, 2, :, 0].set(-mb0[0] + mbe[0] + m0b[1] - meb[1])
        matrix = matrix.at[:, 2, :, 1].set(mbb[0] - m00[1] - m0e[1] + me0[1] + mee[1])
        matrix = matrix.at[:, 2, :, 2].set(m00[0] - m0e[0] + me0[0] - mee[0] - mbb[1])
        matrix = matrix.at[:, 2, :, 3].set(-m0b[0] - meb[0] + mb0[1] + mbe[1])
        matrix = matrix.at[:, 3, :, 0].set(mbb[0] + m00[1] - m0e[1] - me0[1] + mee[1])
        matrix = matrix.at[:, 3, :, 1].set(-mb0[0] - mbe[0] - m0b[1] + meb[1])
        matrix = matrix.at[:, 3, :, 2].set(-m0b[0] - meb[0] - mb0[1] + mbe[1])
        matrix = matrix.at[:, 3, :, 3].set(m00[0] + m0e[0] + me0[0] + mee[0] + mbb[1])
    return matrix.reshape((ncls * (lmax + 1), ncls * (lmax + 1)))


# The binning operators depend only on the binning scheme and are built with eager
# scatters (~1 ms of host time), so they are cached. The key is built from the
# host copies of the band arrays: exact content equality, no device sync.
_binning_operator_cache: dict = {}


def _binning_key(bins):
    """Hashable key identifying a binning scheme by content."""
    return (
        bins.n_bands,
        bins.lmax,
        bins._bpws_np.tobytes(),
        bins._ells_np.tobytes(),
        bins._weights_np.tobytes(),
        bins._f_ell_np.tobytes(),
    )


def _binning_operators(bins):
    """``(output, theory)`` binning operators, shapes ``(nb, L)`` and ``(L, nb)``.

    ``output`` averages a spectrum into bandpowers with the bin weights and
    ``f_ell`` factors; ``theory`` spreads bandpowers back over ``l`` divided
    by ``f_ell``.
    """
    key = _binning_key(bins)
    cached = _binning_operator_cache.get(key)
    if cached is not None:
        return cached
    output = jnp.zeros((bins.n_bands, bins.lmax + 1), dtype=jnp.float64)
    output = output.at[bins._bpws, bins._ells].set(bins._factors)
    theory = jnp.zeros((bins.lmax + 1, bins.n_bands), dtype=jnp.float64)
    cached = (output, theory.at[bins._ells, bins._bpws].set(1 / bins._f_ell))
    if len(_binning_operator_cache) > 8:
        _binning_operator_cache.clear()
    _binning_operator_cache[key] = cached
    return cached


# Cache of the operators expanded by kron(., eye(ncls)). These are dense (5.1 GiB
# at Nside 8192 spin 2), so the cache is dropped whenever a transform needs room.
_expanded_binning_cache: dict = {}
_config._ROOM_HOOKS.append(_expanded_binning_cache.clear)


def _expanded_binning_operators(bins, ncls):
    """``(kron(output, I_ncls), kron(theory, I_ncls))``, cached."""
    key = _binning_key(bins) + (ncls,)
    cached = _expanded_binning_cache.get(key)
    if cached is not None:
        return cached
    output, theory = _binning_operators(bins)
    identity = jnp.eye(ncls)
    cached = (jnp.kron(output, identity), jnp.kron(theory, identity))
    if len(_expanded_binning_cache) > 4:
        _expanded_binning_cache.clear()
    _expanded_binning_cache[key] = cached
    return cached


class _RowPieces(tuple):
    """A dense MCM held as ``ncls`` row-channel pieces, never assembled on the device.

    Piece ``c1`` has shape ``(lmax+1, lmax+1, ncls)`` with
    ``piece[l1, l2, c2] = M[l1 * ncls + c1, l2 * ncls + c2]``. Used above
    ``_MCM_PIECED_BYTES``, where a second contiguous matrix-sized buffer
    (18 GiB at Nside 4096 spin 2) would not fit next to the pieces; consumers
    contract piece by piece and the dense form exists only on the host.
    """

    @property
    def n(self):
        """Matrix dimension ``ncls * (lmax + 1)``."""
        return self[0].shape[0] * len(self)

    def dense_host(self):
        """The ``(n, n)`` float64 matrix as a host numpy array."""
        n = self.n
        return np.concatenate([np.asarray(p)[:, None] for p in self], axis=1).reshape((n, n))

    def matvec(self, v):
        """``M @ v`` for a flat ``(n,)`` vector, returned flat in the same ordering."""
        lmax1 = self[0].shape[0]
        cols = [p.reshape((lmax1, self.n)) @ v for p in self]          # each (lmax + 1,)
        return jnp.stack(cols, axis=1).reshape(-1)


def _mcm_dense(mcm):
    """Host numpy copy of the MCM, whichever form it is held in."""
    return mcm.dense_host() if isinstance(mcm, (_RowPieces, _BlockMCM)) else np.asarray(mcm)


# Above this size, `_left_contract` contracts `output @ mcm` in row chunks of the matrix.
_LEFT_CONTRACT_CHUNK_BYTES = 4 * 1024 ** 3


def _left_contract(output, mcm, *, f32=False):
    """``output @ mcm``, contracted in row chunks of ``mcm`` when it is large.

    XLA autotunes a GEMM with duplicate operand buffers, which does not fit
    for a matrix of tens of GiB. ``sum_k output[:, rows_k] @ mcm[rows_k, :]``
    over contiguous row blocks is equal up to summation order. ``mcm`` may
    also be a :class:`_RowPieces` tuple. ``f32`` selects float32 operands.
    """
    def dot(a, b):
        if not f32:
            return a @ b
        # HIGHEST keeps float32 products off the TF32 path (~2e-6 error rather than ~5e-4).
        return jnp.matmul(a.astype(jnp.float32), b.astype(jnp.float32),
                          precision=jax.lax.Precision.HIGHEST).astype(jnp.float64)

    if isinstance(mcm, tuple):
        # Row-channel pieces: matrix rows (l1, c1) are piece c1; the matching
        # columns of `output` are c1::ncls.
        ncls = len(mcm)
        lmax1 = mcm[0].shape[0]
        acc = dot(output[:, 0::ncls], mcm[0].reshape((lmax1, lmax1 * ncls)))
        for c1 in range(1, ncls):
            acc = acc + dot(output[:, c1::ncls], mcm[c1].reshape((lmax1, lmax1 * ncls)))
        return acc
    n = mcm.shape[0]
    if mcm.size * jnp.dtype(mcm.dtype).itemsize <= _LEFT_CONTRACT_CHUNK_BYTES:
        return dot(output, mcm)
    # Use the smallest divisor of n at or above the byte target as the chunk
    # count, so no padding (i.e. no copy of the matrix) is needed.
    target = -(-(mcm.size * jnp.dtype(mcm.dtype).itemsize) // _LEFT_CONTRACT_CHUNK_BYTES)
    nchunk = next(k for k in range(target, n + 1) if n % k == 0)
    chunk = n // nchunk
    # Static slices rather than a fori_loop: XLA copies a jit parameter that
    # enters a while loop, which here would duplicate the whole matrix.
    acc = dot(output[:, 0:chunk], mcm[0:chunk, :])
    for k in range(1, nchunk):
        acc = acc + dot(output[:, k * chunk:(k + 1) * chunk], mcm[k * chunk:(k + 1) * chunk, :])
    return acc


@partial(jax.jit, static_argnames=("ncls", "norm_type", "lmax"))
def _banded_operators(mcm, beam1, beam2, output, theory, wawb, *, ncls, norm_type, lmax):
    """Binned MCM and one-sided bandpower operator from a dense MCM.

    Returns ``(mcm_binned, one_sided)`` with ``one_sided = B M diag(beam)``
    and ``mcm_binned = one_sided T`` (or ``wawb * I`` for FKP normalisation),
    where ``B`` and ``T`` are the expanded output/theory binning operators.
    ``B M`` is the largest GEMM of the pipeline (2 TFLOP at Nside 2048
    spin 2), so it follows the coupling precision setting: float32 operands
    add no error beyond what the matrix already carries.
    """
    beam = jnp.repeat(beam1 * beam2, ncls)
    # The beam scales MCM columns, which commutes with the left product, so apply
    # it after the contraction: scaling first would materialise a second copy of
    # the matrix (18 GiB at Nside 4096 spin 2).
    one_sided = _left_contract(output, mcm, f32=_coupling_f32(lmax)) * beam[None, :]
    if norm_type:
        mcm_binned = wawb * jnp.eye(output.shape[0])
    else:
        mcm_binned = one_sided @ theory
    return mcm_binned, one_sided


class NmtWorkspace:
    """Curved-sky MASTER workspace: mode-coupling matrix and bandpower operators.

    Mirrors ``pymaster.NmtWorkspace``. A workspace holds the MCM of a pair of
    fields (which depends only on their masks, spins, beams and purification
    settings), its binned form, and the bandpower window functions, so that
    coupled pseudo-spectra can be decoupled into bandpowers.

    Parameters
    ----------
    fl1, fl2 : NmtField, optional
        Fields whose masks define the coupling. If given with ``bins``, the
        MCM is computed at construction (see :meth:`compute_coupling_matrix`).
    bins : NmtBin, optional
        Binning scheme.
    is_teb : bool, optional
        Build the joint 7x7 T/E/B coupling (``fl1`` spin 0, ``fl2`` spin s).
    l_toeplitz, l_exact, dl_band : int, optional
        Toeplitz approximation parameters; disabled when ``l_toeplitz <= 0``.
    fname : str, optional
        Read a saved workspace from this file instead of computing one.
    normalization : {"MASTER", "FKP"}, optional
        ``"MASTER"`` inverts the full binned MCM; ``"FKP"`` divides by the
        mean product of the masks instead.

    Attributes
    ----------
    mcm : jax.Array or internal block form
        Unbinned MCM, ``(ncls (lmax+1), ncls (lmax+1))``.
    mcm_binned : jax.Array
        Binned MCM, ``(ncls nbands, ncls nbands)``.
    bpws : jax.Array
        Bandpower windows, ``(ncls nbands, ncls (lmax+1))``.
    """

    @property
    def bpws(self):
        """Bandpower windows ``solve(mcm_binned, one_sided)``, computed on first access."""
        if getattr(self, "_bpws", None) is None and getattr(self, "_one_sided", None) is not None:
            self._bpws = jnp.linalg.solve(self.mcm_binned, self._one_sided)
            self._one_sided = None
        return getattr(self, "_bpws", None)

    @bpws.setter
    def bpws(self, value):
        self._bpws, self._one_sided = value, None

    def __init__(
        self,
        fl1=None,
        fl2=None,
        bins=None,
        is_teb=False,
        l_toeplitz=-1,
        l_exact=-1,
        dl_band=-1,
        fname=None,
        normalization="MASTER",
    ):
        self.mcm = self.mcm_binned = self.bpws = None
        if fname is not None:
            self.read_from(fname)
            return
        if fl1 is not None or fl2 is not None or bins is not None:
            self.compute_coupling_matrix(
                fl1,
                fl2,
                bins,
                is_teb=is_teb,
                l_toeplitz=l_toeplitz,
                l_exact=l_exact,
                dl_band=dl_band,
                normalization=normalization,
            )

    @classmethod
    def from_fields(cls, fl1, fl2, bins, **kwargs):
        """Create a workspace and compute its MCM from two fields (see ``__init__``)."""
        return cls(fl1, fl2, bins, **kwargs)

    @classmethod
    def from_file(cls, fname):
        """Create a workspace by reading it from ``fname``."""
        return cls(fname=fname)

    def compute_coupling_matrix(
        self,
        fl1,
        fl2,
        bins,
        is_teb=False,
        l_toeplitz=-1,
        l_exact=-1,
        dl_band=-1,
        normalization="MASTER",
    ):
        """Compute the mode-coupling matrix and bandpower operators of two fields.

        Parameters
        ----------
        fl1, fl2 : NmtField
            Fields to correlate; they must share pixelization, and their band
            limit must equal ``bins.lmax``.
        bins : NmtBin
            Binning scheme.
        is_teb : bool, optional
            Build the joint T/E/B coupling (7 spectra); requires ``fl1`` spin 0
            and ``fl2`` of non-zero spin, and isotropic masks.
        l_toeplitz, l_exact, dl_band : int, optional
            Toeplitz approximation parameters (not allowed with purification);
            disabled when ``l_toeplitz <= 0``.
        normalization : {"MASTER", "FKP"}, optional
            Normalisation of the binned coupling matrix.
        """
        if not fl1.is_compatible(fl2, strict=False):
            raise ValueError("Fields have incompatible pixelizations")
        if fl1.ainfo.lmax != bins.lmax:
            raise ValueError(
                f"Maximum multipoles in bins ({bins.lmax}) and fields "
                f"({fl1.ainfo.lmax}) are not the same"
            )
        if is_teb and (fl1.spin != 0 or fl2.spin == 0):
            raise ValueError("If is_teb=True, fl1 must be spin-0 and fl2 non-zero spin")
        if is_teb and (fl1.anisotropic_mask or fl2.anisotropic_mask):
            raise NotImplementedError(
                "Joint TEB workspaces do not support anisotropic masks"
            )
        _toeplitz_sanity(l_toeplitz, l_exact, dl_band, fl1.ainfo.lmax, (fl1, fl2))
        if normalization not in ("MASTER", "FKP"):
            raise ValueError(
                f"Unknown normalization type {normalization}. "
                "Allowed options are 'MASTER' and 'FKP'"
            )

        self.spin1 = fl1.spin
        self.spin2 = fl2.spin
        self.nmaps1 = 1 if self.spin1 == 0 else 2
        self.nmaps2 = 1 if self.spin2 == 0 else 2
        self.ncls = 7 if is_teb else self.nmaps1 * self.nmaps2
        self.aniso1 = fl1.anisotropic_mask
        self.aniso2 = fl2.anisotropic_mask
        self.pure_e1 = fl1.pure_e
        self.pure_b1 = fl1.pure_b
        self.pure_e2 = fl2.pure_e
        self.pure_b2 = fl2.pure_b
        self.beam1 = jnp.asarray(fl1.beam)
        self.beam2 = jnp.asarray(fl2.beam)
        self.lmax = fl1.ainfo.lmax
        self.lmax_mask = fl1.ainfo_mask.lmax
        self.is_teb = bool(is_teb)
        self.l_toeplitz = int(l_toeplitz)
        self.l_exact = int(l_exact)
        self.dl_band = int(dl_band)
        self.bins = bins
        self.nbands = bins.get_n_bands()
        self.normalization = normalization
        self.norm_type = int(normalization == "FKP")

        # Wait for the field transforms to finish so their temporaries (tens of
        # GiB at Nside 8192) return to the pool before the quadrature tables are
        # allocated. Mask-only fields (catalogs, covariance inputs) have no alms.
        for fl in (fl1,) if fl2 is fl1 else (fl1, fl2):
            if getattr(fl, "alm", None) is not None:
                jax.block_until_ready(fl.alm)
            jax.block_until_ready(fl.get_mask_alms())
        alm1 = fl1.get_mask_alms()[None, :]
        alm2 = alm1 if fl2 is fl1 else fl2.get_mask_alms()[None, :]
        # Reserve room for the matrix twice (blocks and assembled form) plus the
        # quadrature temporaries, evicting cached transform tables if needed.
        nside = getattr(fl1.minfo, "nside", None) or 0
        # Ring-spectrum width is (lmax_mask+1) complex128 per ring for a scalar
        # transform and twice that for a polarised one; over-reserving would
        # needlessly evict the transform engine's cached tables.
        ring_width = 2 * (self.lmax_mask + 1) if self.spin1 or self.spin2 else self.lmax_mask + 1
        _config.make_room(2 * (self.ncls * (self.lmax + 1)) ** 2 * 8
                        + 6 * (4 * nside - 1) * ring_width * 16)
        self.pcl_mask = _compute_coupled_cell(
            alm1,
            alm2,
            fl1.ainfo_mask._ell,
            fl1.ainfo_mask._m,
            lmax=self.lmax_mask,
        )[0]
        if fl2 is fl1:
            self.pcl_mask = self.pcl_mask - fl1.Nw
        window_cls = jnp.pad(
            self.pcl_mask,
            (0, max(0, 2 * self.lmax + 1 - len(self.pcl_mask))),
        )[: 2 * self.lmax + 1]
        pure_any = self.pure_e1 or self.pure_b1 or self.pure_e2 or self.pure_b2
        spin2_pure = (
            _coupling_matrices_spin2_pure(window_cls, lmax=self.lmax)
            if pure_any
            else None
        )
        scalar = None
        if self.ncls in (1, 7):
            scalar = (
                _coupling_matrix_tt_toeplitz(
                    window_cls,
                    lmax=self.lmax,
                    l_toeplitz=l_toeplitz,
                    l_exact=l_exact,
                    dl_band=dl_band,
                )
                if l_toeplitz > 0
                else _coupling_matrix_tt(window_cls, lmax=self.lmax)
            )
        te = even = odd = None
        if self.ncls != 1:
            if self.spin1 in (0, 2) and self.spin2 in (0, 2):
                te, even, odd = (
                    spin2_pure[:3]
                    if spin2_pure is not None
                    else _coupling_matrices_spin2(
                        window_cls, lmax=self.lmax,
                        need_te=self.ncls in (2, 7), need_ee=self.ncls in (4, 7))
                )
            else:
                if self.spin1 == 0 or self.spin2 == 0:
                    te = _general_coupling_matrix(
                        self.pcl_mask,
                        s1=self.spin1, s2=self.spin2,
                        n1=self.spin1, n2=self.spin2,
                        lmax=self.lmax, lmax_mask=self.lmax_mask,
                    ).sum(axis=0)
                else:
                    even, odd = _general_coupling_matrix(
                        self.pcl_mask,
                        s1=self.spin1, s2=self.spin2,
                        n1=self.spin1, n2=self.spin2,
                        lmax=self.lmax, lmax_mask=self.lmax_mask,
                    )
                if self.is_teb:
                    even, odd = _general_coupling_matrix(
                        self.pcl_mask,
                        s1=self.spin2, s2=self.spin2,
                        n1=self.spin2, n2=self.spin2,
                        lmax=self.lmax, lmax_mask=self.lmax_mask,
                    )
        if l_toeplitz > 0:
            column_factor = (2 * jnp.arange(self.lmax + 1) + 1)[None]

            def approximate(value):
                if value is None:
                    return None
                return _apply_toeplitz(
                    value / column_factor, l_toeplitz, l_exact, dl_band
                ) * column_factor

            te, even, odd = map(approximate, (te, even, odd))
        if pure_any:
            pure_te, even_one, odd_one, even_two, odd_two = (
                spin2_pure[3:]
            )
            te_levels = (te, pure_te)
            even_levels = (even, even_one, even_two)
            odd_levels = (odd, odd_one, odd_two)
        # Map distinct blocks to their (row channel, column channel) slots and
        # signs. With purification, the block used for a slot depends on how
        # many of the two fields are purified in that channel (0, 1 or 2). For
        # spin x spin, M+ fills the diagonal (EE, EB, BE, BB) slots and M- the
        # anti-diagonal ones; a TEB workspace puts TT, TE, TB first (offset 3).
        blocks = []
        slots = []
        signs = []
        if self.ncls == 1:
            blocks.append(scalar)
            slots.append((0, 0))
            signs.append(1)
        elif self.ncls == 2:
            sign = (-1) ** (self.spin1 + self.spin2)
            pure_e = self.pure_e1 + self.pure_e2
            pure_b = self.pure_b1 + self.pure_b2
            for index, level in enumerate((pure_e, pure_b)):
                blocks.append(te_levels[level] if pure_any else te)
                slots.append((index, index))
                signs.append(sign)
        else:
            offset = 3 if self.ncls == 7 else 0
            if self.ncls == 7:
                mixed_sign = (-1) ** self.spin2
                blocks.append(scalar)
                slots.append((0, 0))
                signs.append(1)
                for index, level in enumerate((int(self.pure_e2), int(self.pure_b2))):
                    blocks.append(te_levels[level] if pure_any else te)
                    slots.append((1 + index, 1 + index))
                    signs.append(mixed_sign)
                pure_e1 = pure_e2 = self.pure_e2
                pure_b1 = pure_b2 = self.pure_b2
            else:
                pure_e1, pure_b1 = self.pure_e1, self.pure_b1
                pure_e2, pure_b2 = self.pure_e2, self.pure_b2

            spin_sign = 1 if self.ncls == 7 else (-1) ** (self.spin1 + self.spin2)

            levels = (
                pure_e1 + pure_e2,
                pure_e1 + pure_b2,
                pure_b1 + pure_e2,
                pure_b1 + pure_b2,
            )
            for index, level in enumerate(levels):
                blocks.append(even_levels[level] if pure_any else even)
                slots.append((offset + index, offset + index))
                signs.append(spin_sign)
            for index, level in enumerate(levels):
                blocks.append(odd_levels[level] if pure_any else odd)
                slots.append((offset + index, offset + 3 - index))
                signs.append(-spin_sign if index in (1, 2) else spin_sign)
        _drop_large_wigner_cache()
        self.mcm = _assemble_mcm(
            window_cls.dtype,
            tuple(blocks),
            lmax=self.lmax,
            ncls=self.ncls,
            slots=tuple(slots),
            signs=tuple(signs),
        )
        if self.aniso1 or self.aniso2:
            zeros = jnp.zeros_like(self.pcl_mask)
            spectra = {
                name: zeros
                for name in ("0e", "0b", "e0", "b0", "ee", "eb", "be", "bb")
            }
            spectra["00"] = self.pcl_mask
            if self.aniso2:
                cross = _compute_coupled_cell(
                    alm1,
                    fl2.get_anisotropic_mask_alms(),
                    fl1.ainfo_mask._ell,
                    fl1.ainfo_mask._m,
                    lmax=self.lmax_mask,
                )
                spectra["0e"], spectra["0b"] = cross
            if self.aniso1:
                alm1_anisotropic = fl1.get_anisotropic_mask_alms()
                cross = _compute_coupled_cell(
                    alm1_anisotropic,
                    alm2,
                    fl1.ainfo_mask._ell,
                    fl1.ainfo_mask._m,
                    lmax=self.lmax_mask,
                )
                spectra["e0"], spectra["b0"] = cross
                if self.aniso2:
                    cross = _compute_coupled_cell(
                        alm1_anisotropic,
                        fl2.get_anisotropic_mask_alms(),
                        fl1.ainfo_mask._ell,
                        fl1.ainfo_mask._m,
                        lmax=self.lmax_mask,
                    )
                    (
                        spectra["ee"],
                        spectra["eb"],
                        spectra["be"],
                        spectra["bb"],
                    ) = cross
            self.mcm = _anisotropic_coupling_matrix(
                self.spin1,
                self.spin2,
                self.aniso1,
                self.aniso2,
                spectra,
                lmax=self.lmax,
                lmax_mask=self.lmax_mask,
            )

        self.wawb = 0.0
        if self.norm_type:
            if fl1.is_catalog or fl2.is_catalog:
                if fl2 is not fl1:
                    raise ValueError(
                        "Cannot use FKP normalisation for catalog fields unless "
                        "they are the same field"
                    )
                self.wawb = fl1.Nw
            else:
                self.wawb = fl1.minfo.si.dot_map(
                    fl1.get_mask(), fl2.get_mask()
                ) / (4 * jnp.pi)
        self._postprocess()

    def _postprocess(self):
        """Recompute the binned MCM and bandpower operators after any change."""
        output, theory = _expanded_binning_operators(self.bins, self.ncls)
        if isinstance(self.mcm, _BlockMCM):
            weights, theory_raw = _binning_operators(self.bins)
            one_sided, binned = self.mcm.banded(weights, theory_raw, self.beam1 * self.beam2,
                                                f32=_coupling_f32(self.lmax))
            self.mcm_binned = (self.wawb * jnp.eye(one_sided.shape[0]) if self.norm_type
                               else binned)
            # Bandpower windows are formed lazily (see `bpws`): decoupling only
            # needs `mcm_binned`, and the solve with ncls (lmax+1) right-hand
            # sides is expensive at high resolution.
            self._one_sided, self._bpws = one_sided, None
            return
        self.mcm_binned, one_sided = _banded_operators(
            tuple(self.mcm) if isinstance(self.mcm, _RowPieces) else self.mcm,
            self.beam1,
            self.beam2,
            output,
            theory,
            self.wawb,
            ncls=self.ncls,
            norm_type=self.norm_type,
            lmax=self.lmax,
        )
        self.bpws = jnp.linalg.solve(self.mcm_binned, one_sided)

    def get_coupling_matrix(self):
        """Return the unbinned mode-coupling matrix.

        Returns
        -------
        jax.Array or numpy.ndarray
            ``(ncls (lmax+1), ncls (lmax+1))`` matrix in ``l * ncls + c``
            ordering. Matrices larger than ``_MCM_PIECED_BYTES`` are returned
            as a host numpy array rather than a device array.
        """
        if isinstance(self.mcm, _RowPieces):
            return self.mcm.dense_host()
        if isinstance(self.mcm, _BlockMCM):
            if self.mcm.n ** 2 * 8 > _MCM_PIECED_BYTES:
                return self.mcm.dense_host()
            return self.mcm.dense_device()
        return self.mcm

    def update_coupling_matrix(self, new_matrix):
        """Replace the unbinned MCM and recompute the binned operators.

        Parameters
        ----------
        new_matrix : array_like
            ``(ncls (lmax+1), ncls (lmax+1))`` matrix in the same ordering as
            :meth:`get_coupling_matrix`.
        """
        size = self.ncls * (self.lmax + 1)
        expected = (size, size)
        if np.shape(new_matrix) != expected:
            raise ValueError(
                f"Input matrix has an inconsistent shape. Expected {expected}, "
                f"but got {np.shape(new_matrix)}"
            )
        self.mcm = jnp.asarray(new_matrix)
        self._postprocess()

    def update_beams(self, beam1, beam2):
        """Replace the beams of both fields and recompute the binned operators.

        Parameters
        ----------
        beam1, beam2 : array_like
            Beam transfer functions of length ``lmax + 1``.
        """
        if np.shape(beam1) != (self.lmax + 1,) or np.shape(beam2) != (
            self.lmax + 1,
        ):
            raise ValueError(f"The new beams must go up to ell = {self.lmax}")
        self.beam1 = jnp.asarray(beam1)
        self.beam2 = jnp.asarray(beam2)
        self._postprocess()

    def update_bins(self, bins):
        """Replace the binning scheme (same ``lmax``) and recompute the binned operators.

        Parameters
        ----------
        bins : NmtBin
            New binning scheme.
        """
        if bins.lmax != self.lmax:
            raise ValueError(
                "The new binning scheme has a different maximum multipole"
            )
        self.bins = bins
        self.nbands = bins.get_n_bands()
        self._postprocess()

    def couple_cell(self, cl_in):
        """Convolve theory power spectra with the MCM and beams.

        Parameters
        ----------
        cl_in : array_like
            ``(ncls, >= lmax + 1)`` full-sky power spectra.

        Returns
        -------
        jax.Array
            ``(ncls, lmax + 1)`` coupled spectra, the expectation value of
            :func:`compute_coupled_cell` for these theory spectra.
        """
        cl_in = jnp.asarray(cl_in)
        if (
            cl_in.ndim != 2
            or cl_in.shape[0] != self.ncls
            or cl_in.shape[1] < self.lmax + 1
        ):
            raise ValueError(
                f"Input power spectrum has wrong shape. Expected "
                f"({self.ncls}, {self.lmax + 1}), but got {cl_in.shape}"
            )
        theory = cl_in[:, : self.lmax + 1] * (
            self.beam1 * self.beam2
        )[None, :]
        flat = theory.T.reshape(-1)
        coupled = (self.mcm.matvec(flat) if isinstance(self.mcm, (_RowPieces, _BlockMCM))
                   else self.mcm @ flat)
        return coupled.reshape((self.lmax + 1, self.ncls)).T

    def decouple_cell(self, cl_in, cl_bias=None, cl_noise=None):
        """Bin and decouple a coupled pseudo-spectrum into bandpowers.

        Parameters
        ----------
        cl_in : array_like
            ``(ncls, >= lmax + 1)`` coupled pseudo-spectra, e.g. from
            :func:`compute_coupled_cell`.
        cl_bias, cl_noise : array_like, optional
            Coupled deprojection bias and noise bias of the same shape; both
            are subtracted before decoupling.

        Returns
        -------
        jax.Array
            ``(ncls, nbands)`` decoupled bandpowers.
        """
        cl_in = jnp.asarray(cl_in)
        expected = (self.ncls, self.lmax + 1)
        if (
            cl_in.ndim != 2
            or cl_in.shape[0] != self.ncls
            or cl_in.shape[1] < expected[1]
        ):
            raise ValueError(
                f"Input power spectrum has wrong shape. Expected {expected}, "
                f"but got {cl_in.shape}"
            )
        total = cl_in[:, : self.lmax + 1]
        for name, bias in (("bias", cl_bias), ("noise", cl_noise)):
            if bias is not None:
                bias = jnp.asarray(bias)
                if (
                    bias.ndim != 2
                    or bias.shape[0] != self.ncls
                    or bias.shape[1] < expected[1]
                ):
                    raise ValueError(f"Input {name} power spectrum has wrong shape")
                total = total - bias[:, : self.lmax + 1]
        binned = self.bins.bin_cell(total).T.reshape(-1)
        decoupled = jnp.linalg.solve(self.mcm_binned, binned)
        return decoupled.reshape((self.nbands, self.ncls)).T

    def get_bandpower_windows(self):
        """Return the bandpower window functions.

        Returns
        -------
        jax.Array
            ``(ncls, nbands, ncls, lmax + 1)`` array ``W`` such that the
            expected bandpowers are ``sum_{c', l} W[c, b, c', l] C_{c'}(l)``.
        """
        return self.bpws.reshape(
            (self.nbands, self.ncls, self.lmax + 1, self.ncls)
        ).transpose((1, 0, 3, 2))

    def read_from(self, fname):
        """Read a workspace from a NaMaster-compatible FITS file.

        Parameters
        ----------
        fname : str
            Path to the file.
        """
        import fitsio

        with fitsio.FITS(fname) as fits:
            primary = fits["WSP_PRIMARY"]
            header = primary.read_header()
            self.lmax = int(header["LMAX"])
            self.lmax_mask = int(header["LMAX_MASK"])
            self.is_teb = bool(header["IS_TEB"])
            self.ncls = int(header["NCLS"])
            self.norm_type = int(header.get("NORM_TYPE", 0))
            self.normalization = "MASTER" if self.norm_type == 0 else "FKP"
            self.wawb = float(header.get("WAWB", 0))
            self.spin1 = int(header.get("SPIN1", 0 if self.ncls == 1 else 2))
            self.spin2 = int(header.get("SPIN2", 0 if self.ncls == 1 else 2))
            self.aniso1 = bool(header.get("ANISO1", False))
            self.aniso2 = bool(header.get("ANISO2", False))
            self.nmaps1 = 1 if self.spin1 == 0 else 2
            self.nmaps2 = 1 if self.spin2 == 0 else 2
            self.pure_e1 = bool(header.get("PURE_E1", False))
            self.pure_b1 = bool(header.get("PURE_B1", False))
            self.pure_e2 = bool(header.get("PURE_E2", False))
            self.pure_b2 = bool(header.get("PURE_B2", False))
            self.l_toeplitz = int(header.get("L_TOEPLITZ", -1))
            self.l_exact = int(header.get("L_EXACT", -1))
            self.dl_band = int(header.get("DL_BAND", -1))
            self.mcm = jnp.asarray(np.asarray(primary.read(), dtype=np.float64))
            beams = fits["BEAMS"]
            if "BEAMS" in beams.get_colnames():
                common = np.sqrt(beams["BEAMS"].read().astype(np.float64))
                self.beam1 = self.beam2 = jnp.asarray(common)
            else:
                self.beam1 = jnp.asarray(beams["BEAM1"].read().astype(np.float64))
                self.beam2 = jnp.asarray(beams["BEAM2"].read().astype(np.float64))
            self.pcl_mask = jnp.asarray(
                fits["PCL_MASKS"]["PCL_MASKS"].read().astype(np.float64)
            )
            extname = "BINS" if "BINS" in fits else "BANDPOWERS"
            self.bins = NmtBin._from_fits_file(fits, extname=extname)
        self.nbands = self.bins.get_n_bands()
        self._postprocess()

    def write_to(self, fname):
        """Write the workspace to a NaMaster-compatible FITS file.

        Parameters
        ----------
        fname : str
            Output path; an existing file is overwritten.
        """
        import fitsio

        header = {
            "LMAX": self.lmax,
            "LMAX_MASK": self.lmax_mask,
            "IS_TEB": self.is_teb,
            "NCLS": self.ncls,
            "NORM_TYPE": self.norm_type,
            "WAWB": float(self.wawb),
            "SPIN1": self.spin1,
            "SPIN2": self.spin2,
            "ANISO1": self.aniso1,
            "ANISO2": self.aniso2,
            "PURE_E1": self.pure_e1,
            "PURE_B1": self.pure_b1,
            "PURE_E2": self.pure_e2,
            "PURE_B2": self.pure_b2,
            "L_TOEPLITZ": self.l_toeplitz,
            "L_EXACT": self.l_exact,
            "DL_BAND": self.dl_band,
        }
        multipoles = np.arange(self.lmax + 1, dtype=np.int32)
        with fitsio.FITS(fname, "rw", clobber=True) as fits:
            fits.write(
                _mcm_dense(self.mcm), header=header, extname="WSP_PRIMARY"
            )
            fits.write(
                [multipoles, np.asarray(self.beam1), np.asarray(self.beam2)],
                names=["L", "BEAM1", "BEAM2"],
                extname="BEAMS",
            )
            fits.write(
                [multipoles, np.asarray(self.pcl_mask)],
                names=["L", "PCL_MASKS"],
                extname="PCL_MASKS",
            )
            self.bins._to_fits_file(fits, extname="BANDPOWERS")


def _filter_alms(alms, spectra, ell):
    """Multiply each alm component by ``spectra[in, out][l]`` and sum over inputs."""
    return jnp.stack(
        [
            sum(alms[index] * spectra[index, output][ell] for index in range(len(alms)))
            for output in range(spectra.shape[1])
        ]
    )


def deprojection_bias(f1, f2, cl_guess, n_iter=None):
    """Compute the bias to the coupled pseudo-spectrum from template deprojection.

    Parameters
    ----------
    f1, f2 : NmtField
        Fields to correlate; they must share pixelization and must not be
        lightweight (the templates are needed).
    cl_guess : array_like
        ``(n1 * n2, lmax + 1)`` best-guess full-sky power spectrum of the
        signal, used to model the deprojection bias.
    n_iter : int, optional
        Iterations of the spherical-harmonic analysis (defaults to the
        field's ``n_iter``).

    Returns
    -------
    jax.Array
        ``(n1 * n2, lmax + 1)`` coupled deprojection bias, to be passed as
        ``cl_bias`` to :meth:`NmtWorkspace.decouple_cell`.
    """
    if not f1.is_compatible(f2):
        raise ValueError("Fields have incompatible pixelizations")
    expected = (f1.nmaps * f2.nmaps, f1.ainfo.lmax + 1)
    cl_guess = jnp.asarray(cl_guess)
    if cl_guess.shape != expected:
        raise ValueError(f"Guess Cl should have shape {expected}")
    if f1.lite or f2.lite:
        raise ValueError("Can't compute deprojection bias for lightweight fields")
    n_iter = f1.n_iter if n_iter is None else int(n_iter)
    spectra = cl_guess.reshape((f1.nmaps, f2.nmaps, f1.ainfo.lmax + 1))
    ell = f1.ainfo._ell

    def transformed(field, maps):
        if field.pure_e or field.pure_b:
            return field._purify(
                field.get_mask_alms(),
                maps,
                task=(field.pure_e, field.pure_b),
                n_iter=n_iter,
                return_maps=False,
            )
        return map2alm(
            maps * field.mask[None, :],
            field.spin,
            field.minfo,
            field.ainfo,
            n_iter=n_iter,
        )

    def alm_spectra(first, second):
        return _compute_coupled_cell(
            first,
            second,
            f1.ainfo._ell,
            f1.ainfo._m,
            lmax=f1.ainfo.lmax,
        ).reshape((len(first), len(second), f1.ainfo.lmax + 1))

    bias = jnp.zeros((f1.nmaps, f2.nmaps, f1.ainfo.lmax + 1))
    if f1.n_temp:
        ff = []
        for template_j in f1.temp:
            alms = map2alm(
                template_j * f1.mask[None, :],
                f1.spin,
                f1.minfo,
                f1.ainfo,
                n_iter=n_iter,
            )
            alms = _filter_alms(alms, spectra, ell)
            maps = alm2map(alms, f2.spin, f2.minfo, f2.ainfo)
            filtered = transformed(f2, maps)
            ff.append(jnp.stack([alm_spectra(template_i, filtered) for template_i in f1.alm_temp]))
        ff = jnp.stack(ff, axis=1)
        bias -= jnp.einsum("ij,ijklm->klm", f1.iM, ff)

    if f2.n_temp:
        gg = []
        products = []
        for template_j in f2.temp:
            alms = map2alm(
                template_j * f2.mask[None, :],
                f2.spin,
                f2.minfo,
                f2.ainfo,
                n_iter=n_iter,
            )
            alms = _filter_alms(alms, spectra.transpose((1, 0, 2)), ell)
            maps = alm2map(alms, f1.spin, f1.minfo, f1.ainfo)
            if f1.n_temp:
                products.append(
                    jnp.stack(
                        [
                            f1.minfo.si.dot_map(
                                template_i, maps * f1.mask[None, :]
                            )
                            for template_i in f1.temp
                        ]
                    )
                )
            filtered = transformed(f1, maps)
            gg.append(jnp.stack([alm_spectra(filtered, template_i) for template_i in f2.alm_temp]))
        gg = jnp.stack(gg, axis=1)
        bias -= jnp.einsum("ij,ijklm->klm", f2.iM, gg)
        if f1.n_temp:
            products = jnp.stack(products, axis=1)
            fg = jnp.stack(
                [
                    jnp.stack(
                        [alm_spectra(first, second) for second in f2.alm_temp]
                    )
                    for first in f1.alm_temp
                ]
            )
            bias += jnp.einsum(
                "ij,rs,jr,isklm->klm", f1.iM, f2.iM, products, fg
            )
    return bias.reshape(expected)


def uncorr_noise_deprojection_bias(f1, map_var, n_iter=None):
    """Deprojection bias for uncorrelated, inhomogeneous noise.

    Parameters
    ----------
    f1 : NmtField
        Field with templates (not lightweight).
    map_var : array_like
        Per-pixel noise variance map.
    n_iter : int, optional
        Iterations of the spherical-harmonic analysis (defaults to the
        field's ``n_iter``).

    Returns
    -------
    jax.Array
        ``(n1 * n1, lmax + 1)`` coupled bias of the auto-spectrum of ``f1``
        (zero if it has no templates).
    """
    if f1.lite:
        raise ValueError("Can't compute deprojection bias for lightweight fields")
    variance = jnp.asarray(map_var).reshape(-1)
    if len(variance) != f1.minfo.npix:
        raise ValueError("Variance map doesn't match map resolution")
    shape = (f1.nmaps * f1.nmaps, f1.ainfo.lmax + 1)
    if not f1.n_temp:
        return jnp.zeros(shape)
    n_iter = f1.n_iter if n_iter is None else int(n_iter)
    weighted_variance = f1.mask**2 * variance

    def spectra(first, second):
        return _compute_coupled_cell(
            first, second, f1.ainfo._ell, f1.ainfo._m,
            lmax=f1.ainfo.lmax,
        ).reshape((f1.nmaps, f1.nmaps, f1.ainfo.lmax + 1))

    first = []
    for template_j in f1.temp:
        weighted = map2alm(
            template_j * weighted_variance[None],
            f1.spin,
            f1.minfo,
            f1.ainfo,
            n_iter=n_iter,
        )
        first.append(jnp.stack([
            spectra(weighted, template_i) for template_i in f1.alm_temp
        ]))
    first = jnp.stack(first, axis=1)
    bias = -2 * jnp.einsum("ij,ijklm->klm", f1.iM, first)

    template_cells = jnp.stack([
        jnp.stack([spectra(left, right) for right in f1.alm_temp])
        for left in f1.alm_temp
    ])
    products = jnp.stack([
        jnp.stack([
            f1.minfo.si.dot_map(left, right * weighted_variance[None])
            for right in f1.temp
        ])
        for left in f1.temp
    ])
    bias += jnp.einsum(
        "ij,rs,jr,isklm->klm", f1.iM, f1.iM, products, template_cells
    )
    return bias.reshape(shape)


def compute_full_master(
    f1,
    f2,
    b=None,
    cl_noise=None,
    cl_guess=None,
    workspace=None,
    l_toeplitz=-1,
    l_exact=-1,
    dl_band=-1,
    normalization="MASTER",
):
    """Run the full MASTER estimator for two fields.

    Computes the coupled pseudo-spectrum, subtracts the deprojection bias
    (from ``cl_guess``) and the noise bias, and decouples into bandpowers.

    Parameters
    ----------
    f1, f2 : NmtField
        Fields to correlate.
    b : NmtBin, optional
        Binning scheme; required if ``workspace`` is not given.
    cl_noise : array_like, optional
        ``(n1 * n2, lmax + 1)`` coupled noise bias.
    cl_guess : array_like, optional
        ``(n1 * n2, lmax + 1)`` best-guess signal spectrum for the
        deprojection bias.
    workspace : NmtWorkspace, optional
        Precomputed workspace; if None one is built from ``f1``, ``f2``, ``b``.
    l_toeplitz, l_exact, dl_band : int, optional
        Toeplitz approximation parameters used when building the workspace.
    normalization : {"MASTER", "FKP"}, optional
        Normalisation used when building the workspace.

    Returns
    -------
    jax.Array
        ``(n1 * n2, nbands)`` decoupled bandpowers.
    """
    if b is None and workspace is None:
        raise SyntaxError("Must supply either workspace or bins")
    if not f1.is_compatible(f2, strict=False):
        raise ValueError("Fields have incompatible pixelizations")
    shape = (f1.nmaps * f2.nmaps, f1.ainfo.lmax + 1)
    noise = jnp.zeros(shape) if cl_noise is None else jnp.asarray(cl_noise)
    guess = jnp.zeros(shape) if cl_guess is None else jnp.asarray(cl_guess)
    if noise.shape != shape:
        raise ValueError(f"Noise Cl should have shape {shape}")
    if guess.shape != shape:
        raise ValueError(f"Guess Cl should have shape {shape}")
    bias = deprojection_bias(f1, f2, guess)
    coupled = compute_coupled_cell(f1, f2)
    if workspace is None:
        workspace = NmtWorkspace.from_fields(
            f1,
            f2,
            b,
            l_toeplitz=l_toeplitz,
            l_exact=l_exact,
            dl_band=dl_band,
            normalization=normalization,
        )
    return workspace.decouple_cell(coupled - bias - noise)
