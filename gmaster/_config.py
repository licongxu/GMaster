"""Run-time settings and the device-memory policy shared by every GMaster module.

Settings
    `nmt_params` holds the process-wide knobs that NaMaster exposes through
    `pymaster.utils` (default `n_iter`, pseudo-inverse tolerance, SHT calculator) plus
    GMaster's own: the storage precision of the transform tables, the precision of the
    azimuthal (ring) FFTs, and the latitudinal engine ("auto", "march" or "dc").

Device memory
    Large transforms keep geometry-only tables cached on the device.  `make_room(n)`
    evicts caches, cheapest first, until `n` bytes can be allocated.  Modules register
    their own caches with `_ROOM_HOOKS` (dropped first) or `_ROOM_HOOKS_LAST` (dropped
    only when that was not enough).
"""

import jax
import jax.numpy as jnp


class NmtParams:
    """Process-wide default settings (the object behind `nmt_params`).

    Attributes
    ----------
    sht_calculator : str
        Backend selector for spherical-harmonic transforms (see `set_sht_calculator`).
    n_iter_default, n_iter_mask_default : int
        Default number of Jacobi iterations in `map2alm` for maps and for masks.
    tol_pinv_default : float
        Default relative eigenvalue threshold for pseudo-inverses.
    table_dtype : str
        Storage precision of precomputed transform tables (see `set_table_precision`).
    ring_precision : str
        Precision of the azimuthal ring FFTs (see `set_ring_precision`).
    latitudinal_method : str
        Latitudinal transform engine (see `set_latitudinal_method`).
    """

    def __init__(self):
        self.sht_calculator = "jax"
        self.n_iter_default = 3
        self.n_iter_mask_default = 3
        self.tol_pinv_default = 1e-10
        # Storage precision of the precomputed transform tables.  Contractions
        # accumulate in float64 regardless.
        self.table_dtype = "fp64"
        # Element type of the azimuthal transforms.  "auto": complex64 wherever the CUDA
        # float32 march runs the latitudinal stage, complex128 otherwise; "follow": match
        # the table precision; "fp64"/"fp32": fixed.
        self.ring_precision = "auto"
        # "auto": divide-and-conquer inside its band-limit window, float32 march elsewhere.
        # "march": always the float32 difference-form march.  "dc": divide-and-conquer.
        self.latitudinal_method = "auto"


nmt_params = NmtParams()

_TABLE_DTYPES = {"fp64": jnp.float64, "fp32": jnp.float32}
_RING_DTYPES = {"follow": None, "auto": None, "fp64": jnp.complex128,
                "fp32": jnp.complex64}


def table_dtype():
    """Return the JAX dtype in which precomputed transform tables are stored.

    Returns
    -------
    dtype
        ``jnp.float64`` (default) or ``jnp.float32``, as set by `set_table_precision`.
    """
    return _TABLE_DTYPES[nmt_params.table_dtype]


def ring_dtype(L=None):
    """Return the element type of the azimuthal (ring) FFTs.

    The ring stage is a batched FFT.  On data-centre GPUs a double-precision FFT is limited
    by the fp64 arithmetic rate, so the same transform is roughly 4-5x cheaper in
    ``complex64``.

    With the default ``"auto"`` setting the ring stage runs in complex64 wherever the CUDA
    float32 march serves the latitudinal stage (whose accuracy is already float32-class,
    ~1e-6), and in complex128 otherwise.  ``"follow"`` matches `set_table_precision`.  Note
    that the pixels themselves are cast to the ring dtype, so ``"follow"`` with fp32 tables
    analyses the map in float32; ``set_ring_precision("fp64")`` keeps the azimuthal stage
    exact in that case.

    Parameters
    ----------
    L : int, optional
        Band limit ``lmax + 1`` of the transform.  Needed by the ``"auto"`` setting.

    Returns
    -------
    dtype
        ``jnp.complex64`` or ``jnp.complex128``.
    """
    forced = _RING_DTYPES[nmt_params.ring_precision]
    if forced is not None:
        return forced
    if nmt_params.ring_precision == "auto" and L is not None:
        # complex64 wherever the float32 march serves the latitudinal stage (every band
        # limit once the CUDA library is built).  The transform is then already accurate to
        # ~1e-6 and the ring FFT is about half its cost, so complex64 loses nothing
        # measurable.  `GMASTER_MARCH_V2=0` disables the march and keeps complex128.
        from ._sht import march_v2 as _march_v2

        if _march_v2.enabled(L):
            return jnp.complex64
    return jnp.complex64 if table_dtype() == jnp.float32 else jnp.complex128


def set_ring_precision(name):
    """Choose the precision of the azimuthal (ring) FFTs.

    GMaster-specific setting with no pymaster equivalent.  The ring precision sets the
    FFT the map pixels run through, independently of the table storage precision chosen
    with `set_table_precision` (which mainly decides which tables fit in memory).

    Parameters
    ----------
    name : {"auto", "follow", "fp64", "fp32"}
        ``"auto"`` (default): complex64 where the float32 latitudinal march is used,
        complex128 otherwise.  ``"follow"``: match the table precision.  ``"fp64"`` /
        ``"fp32"``: always complex128 / complex64.

    Notes
    -----
    Changing the setting clears the cached ring tables, which are keyed by dtype.
    """
    if name not in _RING_DTYPES:
        raise KeyError(
            "GMaster ring precision must be 'auto', 'follow', 'fp64' or 'fp32'")
    if name == nmt_params.ring_precision:
        return
    nmt_params.ring_precision = name
    drop_ring_tables()


def drop_ring_tables():
    """Free the cached ring chirp-Z tables; they are rebuilt on the next transform."""
    from ._sht import rings

    rings.drop_ring_tables()


_ROOM_HOOKS = []       # callables that free device caches; run first by `make_room`
_ROOM_HOOKS_LAST = []  # run only if everything else was not enough (march window tables)


def make_room(nbytes):
    """Free cached device tables until `nbytes` more bytes can be allocated.

    Transform tables are cached for the lifetime of the process and can be large (the
    polarised ring tables alone are ~30 GiB at Nside 4096).  Large later allocations, such
    as the Nside 4096 spin-2 coupling matrix, may not fit while they are resident.  Caches
    are released in order of increasing rebuild cost, stopping as soon as there is room.

    Parameters
    ----------
    nbytes : int
        Number of bytes the caller is about to allocate.

    Notes
    -----
    If the device reports neither allocator statistics nor driver memory information,
    nothing is freed.
    """
    def short():
        stats = jax.devices()[0].memory_stats() or {}
        limit, in_use = stats.get("bytes_limit"), stats.get("bytes_in_use")
        if limit and in_use is not None:
            return limit - in_use < nbytes
        # With XLA_PYTHON_CLIENT_PREALLOCATE=false the pool reports bytes_limit 0, so ask
        # the driver instead (same fallback as workspaces._pool_limit_and_free).
        from ._sht import march_v2 as _march_v2
        info = _march_v2.device_memory_info()
        if info is None:
            return False
        free, total = info
        if in_use is not None and int(0.9 * total) - in_use < nbytes:
            return True
        return free < nbytes

    if not short():
        return
    # Cheapest to rebuild first: registered caches such as the Wigner-d quadrature tables
    # (~9 GiB at Nside 4096, seconds to rebuild), then the ring tables (~30 GiB), and only
    # then the march window tables, whose rebuild is the most expensive (at Nside 4096 a
    # spin-0 `NmtField` takes about 1.6x longer without them).
    for hook in _ROOM_HOOKS:
        hook()
    if not short():
        return
    drop_ring_tables()
    if not short():
        return
    for hook in _ROOM_HOOKS_LAST:
        hook()


def set_table_precision(name):
    """Choose the device storage precision of the precomputed transform tables.

    GMaster-specific setting with no pymaster equivalent.

    Parameters
    ----------
    name : {"fp64", "fp32"}
        ``"fp64"`` (default): tables hold exactly the values an on-the-fly kernel would
        compute, so table and kernel transforms agree to ~1e-16.
        ``"fp32"``: halves the table memory, so larger geometries can use the fast
        table contraction instead of the slower on-the-fly recurrence.  The values are
        still generated in float64 and contractions accumulate in float64; the cost is
        the float32 representation error (~1e-7 relative on the coupling matrix), which
        is why this is opt-in.

    Notes
    -----
    Cached tables are process-global and keyed by dtype, so switching clears them.
    """
    if name not in _TABLE_DTYPES:
        raise KeyError("GMaster table precision must be 'fp64' or 'fp32'")
    if name == nmt_params.table_dtype:
        return
    nmt_params.table_dtype = name
    from ._sht import spin_slice, theta_matrix

    theta_matrix.release()
    spin_slice.clear_cache()
    drop_ring_tables()


_LATITUDINAL_METHODS = ("auto", "march", "dc")


def latitudinal_method():
    """Return the current latitudinal transform engine setting.

    Returns
    -------
    str
        ``"auto"``, ``"march"`` or ``"dc"``; see `set_latitudinal_method`.
    """
    return nmt_params.latitudinal_method


def set_latitudinal_method(name):
    """Choose the engine for the latitudinal (theta) stage of the spherical-harmonic transforms.

    GMaster-specific setting with no pymaster equivalent.  Every transform is split into
    an azimuthal FFT per ring and a latitudinal Wigner-d stage; this selects the latter.

    Parameters
    ----------
    name : {"auto", "march", "dc"}
        ``"march"``: the float32 difference-form Wigner-d march at every size.
        ``"dc"``: the divide-and-conquer engine wherever its plan can be built (band
        limit up to ``GMASTER_DC_MAX_L``, default 12288, i.e. Nside 4096); the march is
        used above that.
        ``"auto"`` (default): divide-and-conquer for band limits between
        ``GMASTER_DC_MIN_L`` and ``GMASTER_DC_MAX_L``, the march outside that window.
    """
    if name not in _LATITUDINAL_METHODS:
        raise KeyError(
            "latitudinal method must be 'auto', 'march' (fp32 v2 difference form), "
            "or 'dc' (divide-and-conquer)"
        )
    nmt_params.latitudinal_method = name


def set_sht_calculator(calc_name):
    """Select the spherical-harmonic transform backend.

    Plays the role of ``pymaster.set_sht_calculator`` (which chooses between ``"ducc"``
    and ``"healpy"``); in GMaster every option is a JAX backend.

    Parameters
    ----------
    calc_name : str
        ``"jax"`` (default): automatic choice of the fastest available route, including
        multi-GPU transforms when several GPUs are visible.  ``"jax-single"``: never split
        across GPUs.  ``"jax-mgpu"``: always split across GPUs when more than one is
        available.  ``"jax-generic"``: portable s2fft-style latitudinal loop, without
        precomputed tables.  ``"jax-dfp32"``: double-float32 kernel for scalar analysis.
        ``"jax-matrix"``: the same routes as ``"jax"``, except that the scalar
        on-the-fly kernel is never split across GPUs.
    """
    if calc_name not in (
        "jax",
        "jax-single",
        "jax-mgpu",
        "jax-generic",
        "jax-dfp32",
        "jax-matrix",
    ):
        raise KeyError(
            "GMaster's SHT calculator must be 'jax', 'jax-single', "
            "'jax-mgpu', 'jax-generic', 'jax-dfp32', or 'jax-matrix'"
        )
    nmt_params.sht_calculator = calc_name


def set_n_iter_default(n_iter, mask=False):
    """Set the default number of Jacobi iterations used in `map2alm`.

    Parameters
    ----------
    n_iter : int
        Number of iterations (non-negative).
    mask : bool, optional
        If True, set the default used for mask transforms; otherwise the default used
        for all other transforms.
    """
    if n_iter < 0:
        raise ValueError("n_iter must be positive")
    attribute = "n_iter_mask_default" if mask else "n_iter_default"
    setattr(nmt_params, attribute, int(n_iter))


def set_tol_pinv_default(tol_pinv):
    """Set the default relative eigenvalue threshold for pseudo-inverses.

    Parameters
    ----------
    tol_pinv : float
        Threshold in [0, 1]; see `moore_penrose_pinvh`.
    """
    if not 0 <= tol_pinv <= 1:
        raise ValueError("tol_pinv must be between 0 and 1")
    nmt_params.tol_pinv_default = float(tol_pinv)


def get_default_params():
    """Return the current default settings.

    Returns
    -------
    dict
        Keys ``"sht_calculator"``, ``"n_iter_default"``, ``"n_iter_mask_default"``,
        ``"tol_pinv_default"`` and ``"latitudinal_method"``.
    """
    return {
        name: getattr(nmt_params, name)
        for name in (
            "sht_calculator",
            "n_iter_default",
            "n_iter_mask_default",
            "tol_pinv_default",
            "latitudinal_method",
        )
    }
