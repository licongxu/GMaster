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
    def __init__(self):
        self.sht_calculator = "jax"
        self.n_iter_default = 3
        self.n_iter_mask_default = 3
        self.tol_pinv_default = 1e-10
        # Storage precision of the precomputed transform tables.  Every
        # contraction accumulates in float64 whatever this is.
        self.table_dtype = "fp64"
        # Element type of the azimuthal transforms.  "auto" is complex64 where the v2
        # march serves the latitudinal stage and complex128 below it; "follow" keeps the
        # historical coupling to the table precision; "fp64"/"fp32" pin it independently.
        self.ring_precision = "auto"
        # "auto": divide-and-conquer where its plan exists, fp32 v2 march elsewhere.
        # "march": always the fp32 difference-form march. "dc": divide-and-conquer.
        self.latitudinal_method = "auto"


nmt_params = NmtParams()

_TABLE_DTYPES = {"fp64": jnp.float64, "fp32": jnp.float32}
_RING_DTYPES = {"follow": None, "auto": None, "fp64": jnp.complex128,
                "fp32": jnp.complex64}


def table_dtype():
    """jnp dtype the precomputed transform tables are stored in."""
    return _TABLE_DTYPES[nmt_params.table_dtype]


def ring_dtype(L=None):
    """Element type of the azimuthal (ring) transforms.

    The ring stage is a batched FFT, and on this card a double-precision FFT is
    compute-bound at roughly the fp64 FMA rate while the fp32 one runs on the tensor-core
    path: the same length transform is ~4.5x cheaper in `complex64` (`.qwen/tmp/ring_fp32_ab.py`:
    0.91 -> 0.21 ms at Nside 512, 4.08 -> 0.83 ms at 1024).

    The shipped default is ``"auto"``: complex64 exactly where the v2 march serves the
    latitudinal stage, complex128 below it.  ``"follow"`` tracks `set_table_precision`, which
    is what every pre-v2 published number used.  The cast is not
    confined to the chirp tables: `_forward_ring_fft_positive` casts the *pixels* to the
    chirp's real dtype too, so ``"follow"`` with fp32 tables analyzes the map itself in
    float32.  ``set_ring_precision("fp64")`` keeps the azimuthal stage exact under fp32
    tables; ``"fp32"`` buys the ~4.5x regardless of the table choice.
    """
    forced = _RING_DTYPES[nmt_params.ring_precision]
    if forced is not None:
        return forced
    if nmt_params.ring_precision == "auto" and L is not None:
        # The shipped default: complex64 exactly where the v2 march serves the latitudinal stage
        # (every band limit with a CUDA build).  There the pass already carries the march's
        # float32-class 1e-6, the ring stage is 55 % of it (10.4 of 18.7 ms at Nside 1024 spin 2)
        # and complex64 halves that; `GMASTER_MARCH_V2=0` keeps the exact transform.
        from ._sht import march_v2 as _march_v2

        if _march_v2.enabled(L):
            return jnp.complex64
    return jnp.complex64 if table_dtype() == jnp.float32 else jnp.complex128


def set_ring_precision(name):
    """Choose the element type of the azimuthal transforms independently of the tables.

    The two precisions are bought for different reasons: halved *table* bytes change which
    theta route a geometry dispatches to, while the *ring* precision changes the FFT that
    the map pixels run in.  Following the tables couples them; this breaks the coupling so
    an fp32 table route can keep an exact azimuthal transform.

    Clears the ring table caches for the same reason `set_table_precision` does.
    """
    if name not in _RING_DTYPES:
        raise KeyError(
            "GMaster ring precision must be 'auto', 'follow', 'fp64' or 'fp32'")
    if name == nmt_params.ring_precision:
        return
    nmt_params.ring_precision = name
    drop_ring_tables()


def drop_ring_tables():
    """Evict the cached ring chirp-Z tables (they are rebuilt on the next transform)."""
    from ._sht import rings

    rings.drop_ring_tables()


_ROOM_HOOKS = []       # extra callables that free device caches (registered by workspaces)
_ROOM_HOOKS_LAST = []  # freed only when the steps above did not make room (v2 march tables)


def make_room(nbytes):
    """Drop the ring-table caches when the device pool cannot hold `nbytes` more.

    The polarised ring tables are 30 GiB of complex128 at Nside 4096 and live in `lru_cache`s
    for the process; the coupling matrix never uses them, and at Nside 4096 spin 2 its assembly
    (two `(ncls (lmax+1))^2` float64 copies, 18 GiB each) failed with them resident at 50.6 GiB
    in use (`.qwen/tmp/chain_s36j.log`, session 36).  A device without allocator statistics
    reports nothing and nothing is dropped.
    """
    def short():
        stats = jax.devices()[0].memory_stats() or {}
        limit, in_use = stats.get("bytes_limit"), stats.get("bytes_in_use")
        if limit and in_use is not None:
            return limit - in_use < nbytes
        # PREALLOCATE=false reports bytes_limit 0, which used to skip eviction entirely.
        # Same driver fallback as workspaces._pool_limit_and_free.
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
    # Cheapest first: the Wigner-d quadrature cache (9 GiB at Nside 4096, seconds to rebuild),
    # then the ring tables (30 GiB, which the next transform rebuilds), and only then the v2
    # march's window tables.  Those are last because dropping them costs the most: the Nside 4096
    # spin-0 `NmtField` is 2.00 s with them resident and 3.26 s without
    # (`.qwen/tmp/field_s37.py`), and the benchmark's repeated field builds were freeing them on
    # every call, which is the whole difference between that stage measuring 2.0 s and 2.6 s.
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
    """Choose the device storage precision of the precomputed tables.

    `"fp64"` (default) is what every published GMaster number used: the tables
    hold exactly the values the fused kernel recomputes, so a table transform and
    a kernel transform agree to ~1e-16.

    `"fp32"` halves the device bytes of those tables.  The largest geometries are
    dispatched by a fit test, not by speed, so what this buys is *engagement*:
    a geometry that declined the tables and took the recurrence-bound fused
    kernel can take the memory-bound contraction instead.  The recurrence that
    generates the values stays float64 and every contraction still accumulates in
    float64; the price is the table's own representation error (~1e-7 relative on
    the coupling matrix), which is why it is opt-in and never inferred.

    Cached tables are keyed by dtype, and the caches are process-global, so
    switching clears them rather than handing a caller the other precision's
    bytes.
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
    """``"march"`` (fp32 v2 difference form), ``"dc"`` (divide-and-conquer), or ``"auto"``."""
    return nmt_params.latitudinal_method


def set_latitudinal_method(name):
    """Choose the latitudinal transform.

    ``"march"`` is the fp32 difference-form v2 march at every size.
    ``"dc"`` is the divide-and-conquer engine wherever its plan can be built
    (bandlimit up to ``GMASTER_DC_MAX_L``, default 12288, i.e. ``Nside`` 4096).
    Above that the march is used, because the plan is not built.
    ``"auto"`` uses the divide-and-conquer engine between ``GMASTER_DC_MIN_L`` and
    ``GMASTER_DC_MAX_L`` and the march outside that window.
    """
    if name not in _LATITUDINAL_METHODS:
        raise KeyError(
            "latitudinal method must be 'auto', 'march' (fp32 v2 difference form), "
            "or 'dc' (divide-and-conquer)"
        )
    nmt_params.latitudinal_method = name


def set_sht_calculator(calc_name):
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
    if n_iter < 0:
        raise ValueError("n_iter must be positive")
    attribute = "n_iter_mask_default" if mask else "n_iter_default"
    setattr(nmt_params, attribute, int(n_iter))


def set_tol_pinv_default(tol_pinv):
    if not 0 <= tol_pinv <= 1:
        raise ValueError("tol_pinv must be between 0 and 1")
    nmt_params.tol_pinv_default = float(tol_pinv)


def get_default_params():
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
