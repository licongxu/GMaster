"""Pseudo-spectrum operations."""

import os
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln
from scipy.special import roots_legendre

from .bins import NmtBin, NmtBinFlat
from . import utils
from .utils import alm2map, map2alm

# Operand precision of the coupling-matrix builders.  The polarised quadrature is two
# contractions of 115.9 GFLOP each at lmax 3071, and the float64 form runs at 1.76 TFLOP/s, i.e.
# 94 % of this card's fp64 roofline, so float32 operands are 34.3x there
# (`.qwen/tmp/coupling_prec_1024c_s29.log`) at a cost of rel 2.1e-06 of the matrix.  The scalar
# builder `_coupling_matrix_tt` has no dot at all -- five table lookups and a few elementwise ops
# over ~n^3/3 elements -- and is worth 7.5-9.4x for rel 1.9e-07
# (`.qwen/tmp/ttknob_s30.log`), since only its per-term products are rounded.
#
# The default is "auto": float32 operands exactly where the v2 march already serves the
# transforms (`lmax + 1 >= _march_v2._MIN_L`, i.e. Nside >= 256), float64 below that.  Above the
# gate the pipeline's own accuracy is the march's float32-class 1e-6 and the coupling matrix's
# 2e-06 is inside it; below it the transforms are exact and `tests/test_workspaces.py` holds the
# matrix to `atol=2e-14`, which only float64 meets.  `GMASTER_COUPLING_PRECISION=fp64|fp32`
# (or :func:`set_coupling_precision`) forces one width at every size.
_COUPLING_PRECISION = os.environ.get("GMASTER_COUPLING_PRECISION", "auto")


def _coupling_f32(lmax):
    """Whether this coupling build uses float32 operands."""
    if _COUPLING_PRECISION != "auto":
        return _COUPLING_PRECISION == "fp32"
    from . import _march_v2

    return _march_v2.enabled(int(lmax) + 1)


def set_coupling_precision(name):
    """Choose the operand precision of the coupling-matrix builders.

    Covers both the polarised quadrature (`_general_coupling_matrix_quadrature`, which sets the
    operand width of its two large contractions) and the scalar build (`_coupling_matrix_tt`,
    which rounds its lookup tables and per-term products while keeping the log-cumsum table and
    the offset accumulator in float64).

    Must be called before the first coupling build: the flag is consulted while tracing, so an
    already-compiled program keeps the precision it was traced with (``jax.clear_caches()`` after
    the call if any coupling matrix has already been built in this process).
    """
    global _COUPLING_PRECISION
    if name not in ("fp64", "fp32", "auto"):
        raise KeyError("GMaster coupling precision must be 'fp64', 'fp32' or 'auto'")
    _COUPLING_PRECISION = name


def coupling_precision():
    return _COUPLING_PRECISION


@partial(jax.jit, static_argnames="lmax")
def _compute_coupled_cell(alm1, alm2, ell, order, *, lmax):
    products = jnp.real(alm1[:, None, :] * jnp.conj(alm2[None, :, :]))
    products *= jnp.where(order == 0, 1, 2)
    products = products.reshape((-1, products.shape[-1]))
    cls = jax.vmap(
        lambda product: jax.ops.segment_sum(product, ell, num_segments=lmax + 1)
    )(products)
    return cls / (2 * jnp.arange(lmax + 1) + 1)


def compute_coupled_cell(f1, f2):
    """Compute coupled pseudo-C_ell spectra from two masked fields."""
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
    """Compute binned flat-sky pseudo-C_ell spectra."""
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


# Offsets per block in the scalar MASTER matrix build; see `_coupling_matrix_tt`.
_OFFSET_CHUNK = 16


# Dense coupling matrices above this size are assembled in pieces (`_assemble_mcm`).
_MCM_PIECED_BYTES = 4 * 1024 ** 3


@partial(jax.jit, static_argnames=("dtype", "lmax", "ncls", "slots", "signs"))
def _assemble_mcm_program(blocks, *, dtype, lmax, ncls, slots, signs):
    """Place the coupling blocks into the mode-coupling matrix in one program.

    The chain of ``matrix.at[:, i, :, j].set(...)`` this replaces was up to eight host
    dispatches, each copying the whole ``(lmax + 1, ncls, lmax + 1, ncls)`` array.  Traced
    inside one program the same unrolled chain lowers to a single pass over the output,
    which takes the `coupling` stage of the pipeline from 6 ms to 3 ms at Nside 32 spin 2
    (``.qwen/tmp/pipe4_s29.log`` against ``.qwen/tmp/pipe_fix32_s29.log``) and from 2674 ms
    to 2578 ms at Nside 2048 spin 2 (``.qwen/tmp/pipe2_s29.log`` against
    ``.qwen/tmp/pipe_fix2048_s29.log``), where each update was copying a 9.7 GB matrix.
    Values are bit-identical because every sign here is exactly ``+-1``: the agreement with
    the reference is unchanged (rel ``dCl`` 1.09e-05 at 2048 spin 2 in both logs).
    """
    matrix = jnp.zeros((lmax + 1, ncls, lmax + 1, ncls), dtype=dtype)
    for block, (row, column), sign in zip(blocks, slots, signs):
        matrix = matrix.at[:, row, :, column].set(block * sign)
    return matrix.reshape((ncls * (lmax + 1), ncls * (lmax + 1)))


@partial(jax.jit, static_argnames=("signs",))
def _mcm_row(pieces, *, signs):
    """One row channel of the matrix: `(l1, l2, c2)` with the channel axis innermost."""
    return jnp.stack([p * s for p, s in zip(pieces, signs)], axis=-1)


def _assemble_mcm(dtype, blocks, *, lmax, ncls, slots, signs):
    """The dense mode-coupling matrix from its `(lmax+1, lmax+1)` blocks.

    Below `_MCM_PIECED_BYTES` this is the single program above.  Above it the one-program
    form is a single XLA transpose fusion the size of the matrix, and at Nside 4096 spin 2
    (`ncls = 4`, 18 GiB) that fusion could not be autotuned inside the pool even with the
    ring tables evicted and 22.6 GiB in use (`.qwen/tmp/chain_s36l.log`: the autotuner
    wants the output plus reference copies).  So the matrix is built one row channel at a
    time -- each `(l1, l2, ncls)` piece is a transpose of `ncls` blocks, a quarter of the
    matrix -- and the pieces are joined along the row-channel axis, whose innermost axis
    already matches the output: a plain concatenate, not a transpose.  Values are unchanged.
    """
    n = ncls * (lmax + 1)
    if n * n * jnp.dtype(dtype).itemsize <= _MCM_PIECED_BYTES:
        # `dtype` is passed by keyword: a static argument given *positionally* drops jax's C++
        # fast path, and this call then costs 1.54 ms of host dispatch at Nside 128 spin 0 --
        # a third of that pipeline's whole 4.09 ms enqueue (`.qwen/tmp/jitcount_s37.py`).
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


@partial(jax.jit, static_argnames="lmax")
def _coupling_matrix_tt(window_cls, *, lmax):
    """Exact scalar MASTER matrix using the threej_cosmo recurrence.

    Offsets are accumulated in blocks rather than one at a time.  A term with a
    given offset vanishes unless ``offset <= min(l1, l2)``, so a shrinking
    ``[o0:, o0:]`` slice visits ~n^3/3 elements.  That form is a Python loop of
    distinct shapes, which XLA unrolls into one kernel per chunk: at lmax 3071
    that is 192 compiles, and on a Colab T4 with ~12 GB host RAM the unrolled
    graph thrashes (first call ~30 min, warmed ~13 min vs NaMaster ~3 min).
    The body keeps a uniform ``(chunk, n, n)`` shape and a ``fori_loop``, so one
    compile serves every chunk.  Invalid lanes (offset past ``lmax``, or
    ``offset > min(l1, l2)``) are masked; kept entries match the shrinking form.

    Chunking was 3.0-3.6x over the per-offset loop at lmax 383/767/1535
    (10.05 -> 3.23 ms at lmax 767), values agreeing to 1e-16.

    The block body is five table lookups and a few elementwise ops over the
    chunk, so it has no `dot` to move onto tensor units; its cost is the
    elementwise pass itself, and the only lever on that is the operand width.
    Under :func:`set_coupling_precision` ("fp32") the lookup tables and the term
    are float32 while the log-cumsum table and the offset accumulator stay
    float64, which is worth 9.42x at lmax 1535 (23.48 -> 2.49 ms) and 7.51x at
    lmax 3071 (159.05 -> 21.19 ms) for rel 1.885e-07 of the matrix
    (`.qwen/tmp/ttknob_s30.log`).  Chunk width is not a lever: 8 and 32 both
    measured slower than the shipped 16 at lmax 1535 (27.03/27.42 ms,
    `.qwen/tmp/tt1535_s30.log`).
    """
    use_f32 = _coupling_f32(lmax)
    element_dtype = jnp.float32 if use_f32 else window_cls.dtype
    # The accumulator keeps float64 even in the float32 arm: a row of a block sums
    # up to `lmax` terms, and rounding that partial in float32 would put the error
    # in the sum rather than in the products.
    accumulator_dtype = jnp.float64 if use_f32 else window_cls.dtype
    table_dtype = jnp.float64 if use_f32 else window_cls.dtype

    n_ell = lmax + 1
    n_chunks = (n_ell + _OFFSET_CHUNK - 1) // _OFFSET_CHUNK
    last_offset = n_ell - 1
    multipoles = jnp.arange(n_ell)
    row = multipoles[:, None]
    column = multipoles[None, :]
    p = jnp.arange(1, 2 * lmax + 1, dtype=table_dtype)
    log_g = jnp.concatenate(
        [jnp.zeros(1, dtype=table_dtype), jnp.cumsum(jnp.log((p - 0.5) / p))]
    )
    g = jnp.exp(log_g).astype(element_dtype)
    mask_power = (
        window_cls * (2 * jnp.arange(2 * lmax + 1) + 1) / (4 * jnp.pi)
    ).astype(element_dtype)

    # The summand is symmetric in `(l1, l2)` and depends on them only through `min`, `max` and the
    # difference, so an upper-triangle-only form exists in which three of the four per-element
    # gathers become row/column broadcasts.  It was written and measured and is NOT shipped: at
    # lmax 6143 it is 223 ms against 200 ms here and at lmax 12287 1895 ms against 1761 ms.  The
    # pass is not gather-issue bound the way the operand count suggests -- `lower`/`upper` are
    # hoisted out of the offset loop and XLA fuses the four lookups into one pass -- and the
    # triangular form pays a transpose and a diagonal without removing any elements, because the
    # discarded half is still inside the rectangle XLA evaluates.
    lower = jnp.minimum(row, column)
    upper = jnp.maximum(row, column)
    lane = jnp.arange(_OFFSET_CHUNK)[:, None, None]

    def add_chunk(matrix, chunk):
        # Pad the last chunk to width 16 so every iteration has the same gather
        # shape; indices past lmax are clamped for the lookup and then masked.
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
    """Scalar MASTER matrix without evaluating entries Toeplitz will replace."""
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
    return roots_legendre(order)


@partial(jax.jit, static_argnames=("m", "n", "lmax"))
def _wigner_d_table(beta, *, m, n, lmax):
    """Return unnormalised d^ell_mn(beta) for all ell."""
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

    # The recurrence is stacked instead of written column by column: a traced
    # column index makes XLA copy the whole (nodes, ell) table on every degree.
    # It is also launch-bound -- at lmax 768 with 1535 nodes each step is only
    # ~1535 elements, so unrolling eight degrees per kernel is what actually
    # buys the speedup (31.5 ms -> 4.0 ms for the three tables of one call).
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
            # `HIGHEST` keeps the products off the tf32 units: at DEFAULT the same call is another
            # 1.7x faster and the matrix error grows from 2.1e-06 to 5.5e-04.
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
utils._ROOM_HOOKS.append(_WD_TRIPLE_CACHE.clear)   # 9 GiB of device tables at Nside 4096
# The Wigner-d quadrature tables are kept between workspace builds while they are small against
# the device pool.  A fixed 2 GiB threshold dropped them at every geometry that matters: at
# Nside 2048 spin 2 the three `(12287, 12287)` float64 tables are 3.6 GiB, so every
# `compute_coupling_matrix` rebuilt them through a 12287-step scan -- about 650 ms of the 709 ms
# coupling stage, for tables that never change.  The threshold is now the larger of 2 GiB and a
# tenth of the pool, so a 96 GiB card keeps them and a small one still evicts.
_WD_CACHE_KEEP_FRACTION = float(os.environ.get("GMASTER_WD_CACHE_FRACTION", "0.10"))


# The cache is also dropped when the free pool is no longer a comfortable multiple of it: at Nside
# 4096 spin 2 the coupling stage asks for an 11 GiB block with the polarised field resident and
# failed with 55 GiB in use of 71.2 GiB while these tables were kept.  Keeping them where there is
# room is worth most of the stage: at Nside 2048 spin 2 rebuilding the three tables through their
# 12287-step scan was ~600 ms of a 660 ms `compute_coupling_matrix`.
_WD_POOL_HEADROOM = float(os.environ.get("GMASTER_WD_POOL_HEADROOM", "4.0"))


def _pool_is_tight(total):
    try:
        stats = jax.devices()[0].memory_stats() or {}
        limit = stats.get("bytes_limit") or 0
        in_use = stats.get("bytes_in_use") or 0
    except Exception:  # noqa: BLE001 - a backend without pool accounting
        return False
    return bool(limit) and (limit - in_use) < _WD_POOL_HEADROOM * total


def _wd_cache_keep_bytes():
    try:
        stats = jax.devices()[0].memory_stats() or {}
        limit = stats.get("bytes_limit") or 0
    except Exception:  # noqa: BLE001 - a backend without pool accounting
        limit = 0
    return max(2 * 1024 ** 3, int(limit * _WD_CACHE_KEEP_FRACTION))


def _drop_large_wigner_cache():
    """Forget quadrature tables too large to keep between workspaces.

    The cache pays off for repeated small workspaces; at Nside 4096 spin 2 it is 9 GiB of device
    memory that outlived its workspace and pushed the next field's mask transform over the pool
    (`.qwen/tmp/chain_s36u.log`, cuFFT plan allocation aborting the process).
    """
    total = sum(int(t.size) * int(t.dtype.itemsize) for t in _WD_TRIPLE_CACHE.values())
    if total and (total > _wd_cache_keep_bytes() or _pool_is_tight(total)):
        _WD_TRIPLE_CACHE.clear()


def _wigner_d_shared(beta, m, n, order):
    """``_wigner_d_table`` with the ``(-m, -n)`` redundancy folded away.

    ``d^l_{-m,-n}(beta) == d^l_{m,n}(beta)`` when ``m - n`` is even (checked
    bit-for-bit against the builder); when ``m - n`` is odd the identity carries
    a minus sign, so those tables are left alone rather than shared with the
    wrong sign.  Even differences are exactly what the temperature and
    polarised couplings ask for -- ``(0,2)``, ``(2,2)``, ``(0,0)``.  Each table
    is a *sequential* scan over degree, which is what the polarised coupling
    build spends its time on -- at lmax 95 it costs the same 5 ms as at lmax
    767 -- so handing the same array back twice is a latency win, not just a
    memory one.  The key carries the node count rather than the nodes: only
    :func:`_general_coupling_matrix` asks for tables, and it always passes the
    Gauss-Legendre nodes of that order.
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
    """The three Wigner-d tables one quadrature call integrates against."""
    return (
        _wigner_d_shared(beta, n1, n2, lmax),
        _wigner_d_shared(beta, -s1, -s2, lmax),
        _wigner_d_shared(beta, s1 - n1, s2 - n2, lmax_mask),
    )


def _general_coupling_matrix(
    mask_cls, *, s1, s2, n1, n2, lmax, lmax_mask, tables=None
):
    order = (2 * lmax + lmax_mask) // 2 + 1
    nodes, weights = _gauss_legendre(order)
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


def _coupling_matrices_spin2(window_cls, *, lmax):
    mixed = _general_coupling_matrix(
        window_cls,
        s1=0,
        s2=2,
        n1=0,
        n2=2,
        lmax=lmax,
        lmax_mask=2 * lmax,
    ).sum(axis=0)
    even, odd = _general_coupling_matrix(
        window_cls,
        s1=2,
        s2=2,
        n1=2,
        n2=2,
        lmax=lmax,
        lmax_mask=2 * lmax,
    )
    return mixed, even, odd


@partial(jax.jit, static_argnames="lmax")
def _coupling_matrices_pure_quadrature(
    window_cls, nodes, weights, *, lmax
):
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
    nodes, weights = _gauss_legendre(2 * lmax + 1)
    return _coupling_matrices_pure_quadrature(
        window_cls,
        jnp.asarray(nodes),
        jnp.asarray(weights),
        lmax=lmax,
    )


def _coupling_matrices_pure(window_cls, *, lmax):
    return _coupling_matrices_spin2_pure(window_cls, lmax=lmax)[3:]


def get_general_coupling_matrix(pcl_mask, s1, s2, n1, n2, parity="all"):
    """Return a general spin coupling matrix with optional parity selection."""
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
    """Return the scalar, mixed-spin, and parity MASTER kernels."""
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
                    else _coupling_matrices_spin2(padded, lmax=lmax)
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
                    else _coupling_matrices_spin2(padded, lmax=lmax)
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


# The two binning operators are a pure function of the binning scheme, and both are
# assembled with eager indexed scatters — the slowest way to build an array in XLA (two
# scatters cost ~1.2ms of host time, which is the whole coupling-matrix call at small
# Nsides).  The key is the host-side copies of the band arrays, so it is exact content
# equality and needs no device synchronisation.
_binning_operator_cache: dict = {}


def _binning_key(bins):
    return (
        bins.n_bands,
        bins.lmax,
        bins._bpws_np.tobytes(),
        bins._ells_np.tobytes(),
        bins._weights_np.tobytes(),
        bins._f_ell_np.tobytes(),
    )


def _binning_operators(bins):
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


# `_postprocess` expands both operators by eye(ncls) before touching them; the expansion
# and the two projections below are six more eager dispatches (~0.5ms) whose only varying
# input is the coupling matrix itself.
_expanded_binning_cache: dict = {}


def _expanded_binning_operators(bins, ncls):
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
    """The mode-coupling matrix held as `ncls` row-channel pieces, never assembled on device.

    Piece `c1` is `(lmax + 1, lmax + 1, ncls)` with `piece[l1, l2, c2] = M[(l1, c1), (l2, c2)]`,
    i.e. the rows of channel `c1` in the `l * ncls + c` ordering.  Above `_MCM_PIECED_BYTES` the
    dense matrix is a single contiguous buffer the size of the whole set (18 GiB at Nside 4096
    spin 2) that the allocator could not place next to the pieces themselves
    (`.qwen/tmp/chain_s36p.log`: 19.3 GiB in use, a further 18 GiB refused with 52 GiB free), so
    the consumers contract from the pieces and the dense form exists only on the host.
    """

    @property
    def n(self):
        return self[0].shape[0] * len(self)

    def dense_host(self):
        """The `(n, n)` float64 matrix as a numpy array (host memory)."""
        n = self.n
        return np.concatenate([np.asarray(p)[:, None] for p in self], axis=1).reshape((n, n))

    def matvec(self, v):
        """`M @ v` for a flat `(n,)` vector, returned flat in the same ordering."""
        lmax1 = self[0].shape[0]
        cols = [p.reshape((lmax1, self.n)) @ v for p in self]          # each (lmax + 1,)
        return jnp.stack(cols, axis=1).reshape(-1)


def _mcm_dense(mcm):
    """Host numpy copy of the matrix whichever form it is held in."""
    return mcm.dense_host() if isinstance(mcm, _RowPieces) else np.asarray(mcm)


# Contract `output @ mcm` in row chunks of the matrix above this size (`_left_contract`).
_LEFT_CONTRACT_CHUNK_BYTES = 4 * 1024 ** 3


def _left_contract(output, mcm, *, f32=False):
    """`output @ mcm`, in row chunks of `mcm` when the matrix is large.

    XLA autotunes the GEMM with duplicate operand buffers; at Nside 4096 spin 2 the 18 GiB
    matrix could not be duplicated inside the pool after the field, so the contraction ran out
    of memory at the first sync after `compute_coupling_matrix` (`.qwen/tmp/chain_s36m2.log`).
    `sum_k output[:, rows_k] @ mcm[rows_k, :]` over contiguous row blocks has the same value up
    to summation order; below `_LEFT_CONTRACT_CHUNK_BYTES` the single GEMM is kept.
    """
    def dot(a, b):
        if not f32:
            return a @ b
        # `HIGHEST` keeps the products off the tf32 units, as in the quadrature: the binned
        # operator inherits the matrix's own ~2e-06, and tf32 would put 5e-04 there instead.
        return jnp.matmul(a.astype(jnp.float32), b.astype(jnp.float32),
                          precision=jax.lax.Precision.HIGHEST).astype(jnp.float64)

    if isinstance(mcm, tuple):
        # Row-channel pieces (`_RowPieces`): rows `(l1, c1)` of the matrix are piece `c1`,
        # and the matching columns of `output` are `c1::ncls`.
        ncls = len(mcm)
        lmax1 = mcm[0].shape[0]
        acc = dot(output[:, 0::ncls], mcm[0].reshape((lmax1, lmax1 * ncls)))
        for c1 in range(1, ncls):
            acc = acc + dot(output[:, c1::ncls], mcm[c1].reshape((lmax1, lmax1 * ncls)))
        return acc
    n = mcm.shape[0]
    if mcm.size * jnp.dtype(mcm.dtype).itemsize <= _LEFT_CONTRACT_CHUNK_BYTES:
        return dot(output, mcm)
    # The chunk count is the smallest divisor of `n` at or above the byte target, so no row is
    # padded: padding would copy the whole matrix, which is the allocation this avoids.
    target = -(-(mcm.size * jnp.dtype(mcm.dtype).itemsize) // _LEFT_CONTRACT_CHUNK_BYTES)
    nchunk = next(k for k in range(target, n + 1) if n % k == 0)
    chunk = n // nchunk
    # Static slices, not a `fori_loop`: XLA's copy insertion duplicates a jit parameter that
    # enters a while loop, and for this operand that duplicate is the matrix itself
    # (`.qwen/tmp/chain_s36n2.log`: a second 18 GiB request with the matrix already resident).
    acc = dot(output[:, 0:chunk], mcm[0:chunk, :])
    for k in range(1, nchunk):
        acc = acc + dot(output[:, k * chunk:(k + 1) * chunk], mcm[k * chunk:(k + 1) * chunk, :])
    return acc


@partial(jax.jit, static_argnames=("ncls", "norm_type", "lmax"))
def _banded_operators(mcm, beam1, beam2, output, theory, wawb, *, ncls, norm_type, lmax):
    """Bandpower operators from the mode-coupling matrix.

    `output @ mcm` is the largest contraction of the whole pipeline: at Nside 2048 spin 2 it is
    `(1636, 24576) @ (24576, 24576)`, 2.0 TFLOP, and in float64 on this card (1.8 TFLOP/s) it was
    **567 ms of the 661 ms coupling stage** while the matrices it contracts cost 58 ms
    (`.qwen/tmp/cprof_coup_s37.py`).  Under the same `auto` rule as the matrix build it runs with
    float32 operands and `Precision.HIGHEST`, which is the accuracy the matrix already carries.
    """
    beam = jnp.repeat(beam1 * beam2, ncls)
    # The beam scales columns of the mode-coupling matrix, and a column scaling commutes past a
    # left matmul, so it is applied after the contraction instead of before it.  Written the other
    # way XLA has to materialise `mcm * beam` before the dot: a second full copy of the largest
    # tensor in the stage (18.0 GiB for the polarised matrix at Nside 4096, where the very next
    # allocation is what fails -- `.qwen/tmp/s2_4096_s31.log`).
    one_sided = _left_contract(output, mcm, f32=_coupling_f32(lmax)) * beam[None, :]
    if norm_type:
        mcm_binned = wawb * jnp.eye(output.shape[0])
    else:
        mcm_binned = one_sided @ theory
    return mcm_binned, one_sided


class NmtWorkspace:
    """Curved-sky MASTER coupling matrix and bandpower operators."""

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
        return cls(fl1, fl2, bins, **kwargs)

    @classmethod
    def from_file(cls, fname):
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

        alm1 = fl1.get_mask_alms()[None, :]
        alm2 = alm1 if fl2 is fl1 else fl2.get_mask_alms()[None, :]
        # The stage needs room for the matrix twice over (its blocks and the assembled form) plus
        # the quadrature's own temporaries; with a polarised field at Nside 4096 both ring-table
        # sets (30 GiB) are still cached and the quadrature's Wigner-d transpose could not even be
        # autotuned (`.qwen/tmp/chain_s36q.log`).  Evict them up front if the pool is short.
        nside = getattr(fl1.minfo, "nside", None) or 0
        # The mask transform is spin 0 whatever the field is, so its ring spectrum is
        # `(ntheta, lmax_mask+1)` complex128 -- the `2 (lmax_mask+1)` width belongs to the
        # polarised transform alone.  Asking for twice what the stage needs made `make_room`
        # evict the march's window tables during a scalar Nside 4096 pipeline, and rebuilding
        # them is 1.2 s a time.
        ring_width = 2 * (self.lmax_mask + 1) if self.spin1 or self.spin2 else self.lmax_mask + 1
        utils.make_room(2 * (self.ncls * (self.lmax + 1)) ** 2 * 8
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
                    else _coupling_matrices_spin2(window_cls, lmax=self.lmax)
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
        output, theory = _expanded_binning_operators(self.bins, self.ncls)
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
        """The dense `(ncls (lmax+1), ncls (lmax+1))` matrix.

        Above `_MCM_PIECED_BYTES` the matrix is held as row-channel pieces on the device and
        the dense form is returned as a host numpy array (18 GiB at Nside 4096 spin 2).
        """
        if isinstance(self.mcm, _RowPieces):
            return self.mcm.dense_host()
        return self.mcm

    def update_coupling_matrix(self, new_matrix):
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
        if np.shape(beam1) != (self.lmax + 1,) or np.shape(beam2) != (
            self.lmax + 1,
        ):
            raise ValueError(f"The new beams must go up to ell = {self.lmax}")
        self.beam1 = jnp.asarray(beam1)
        self.beam2 = jnp.asarray(beam2)
        self._postprocess()

    def update_bins(self, bins):
        if bins.lmax != self.lmax:
            raise ValueError(
                "The new binning scheme has a different maximum multipole"
            )
        self.bins = bins
        self.nbands = bins.get_n_bands()
        self._postprocess()

    def couple_cell(self, cl_in):
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
        coupled = (self.mcm.matvec(flat) if isinstance(self.mcm, _RowPieces)
                   else self.mcm @ flat)
        return coupled.reshape((self.lmax + 1, self.ncls)).T

    def decouple_cell(self, cl_in, cl_bias=None, cl_noise=None):
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
        return self.bpws.reshape(
            (self.nbands, self.ncls, self.lmax + 1, self.ncls)
        ).transpose((1, 0, 3, 2))

    def read_from(self, fname):
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
    return jnp.stack(
        [
            sum(alms[index] * spectra[index, output][ell] for index in range(len(alms)))
            for output in range(spectra.shape[1])
        ]
    )


def deprojection_bias(f1, f2, cl_guess, n_iter=None):
    """Compute the contaminant-deprojection pseudo-spectrum bias."""
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
    """Bias from deprojecting templates in uncorrelated inhomogeneous noise."""
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
    """Compute coupled spectra, subtract biases, and decouple bandpowers."""
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
