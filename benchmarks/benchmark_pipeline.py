"""End-to-end MASTER pipeline benchmark: GMaster (GPU) versus NaMaster (CPU).

Stages per configuration: field construction, coupling matrix, coupled cell,
and decoupled cell. Reports warmed medians and decoupled-spectrum parity.
"""

import argparse
import gc
import os
import sys
import time

if __name__ == "__main__":          # before numpy/ducc0/JAX start their thread pools
    import os as _os
    import sys as _sys

    if "--reference-cpus" in _sys.argv and _os.environ.get("GMASTER_BENCH_PINNED") != "1":
        _n = int(_sys.argv[_sys.argv.index("--reference-cpus") + 1])
        if _n > 0:
            _cpus = sorted(_os.sched_getaffinity(0))[:_n]
            if len(_cpus) < _n:
                raise SystemExit(f"only {len(_cpus)} cores available, asked for {_n}")
            _env = dict(_os.environ)
            _env["GMASTER_BENCH_PINNED"] = "1"
            for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                          "NUMEXPR_NUM_THREADS"):
                _env[_name] = str(_n)
            _os.sched_setaffinity(0, _cpus)
            _os.execve(_sys.executable, [_sys.executable] + _sys.argv, _env)

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pymaster as reference

jax.config.update("jax_enable_x64", True)

import gmaster as nmt
from gmaster._nmt_bin64 import patch_pymaster


def _block(result):
    """Force every device value in ``result`` to materialise.

    Stage callables return workspace/field objects, not arrays, so a
    ``hasattr(result, "block_until_ready")`` check skips them entirely and the
    sample measures only the host-side enqueue. Walking the pytree container is
    what makes the stage numbers add up to the end-to-end total.

    Neither ``NmtField`` nor ``NmtWorkspace`` is a registered pytree, so the walk
    above hands them back as a single childless leaf and their stages silently
    measured enqueue only: Nside 2048 spin 0 reported ``coupling 5525->2ms
    (2350x)`` while the same work inside the blocked ``TOTAL`` was ~3.4 s
    (``.qwen/tmp/board_2048_rings_fp32_s29b.log``). Their arrays are reached
    through ``__dict__`` instead, one level of dict/tuple/list included. The
    reference objects hold numpy arrays, for which this is a no-op.
    """
    jax.tree.map(
        lambda leaf: leaf.block_until_ready()
        if hasattr(leaf, "block_until_ready") else None,
        result,
        is_leaf=lambda x: hasattr(x, "block_until_ready"),
    )
    pending = [getattr(result, "__dict__", None)]
    arrays = []
    while pending:
        for value in (pending.pop() or {}).values():
            if hasattr(value, "block_until_ready"):
                arrays.append(value)
            elif isinstance(value, dict):
                pending.append(value)
            elif isinstance(value, (list, tuple)):
                pending.append({i: v for i, v in enumerate(value)})
    for array in arrays:
        array.block_until_ready()


def _timed(function, repeats):
    result = function()
    _block(result)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        _block(result)
        samples.append(time.perf_counter() - start)
    return result, float(np.median(samples))


def _make_field(module, mask, maps_t, maps_q, maps_u, spin):
    if spin == 0:
        return module.NmtField(mask, [maps_t], n_iter=3)
    return module.NmtField(mask, [maps_q, maps_u], n_iter=3, spin=2)


def _run_pipeline(
    module, nside, spin, nlb, repeats, mask, maps_t, maps_q, maps_u,
    release_after_first=False, verbose=False,
):
    lmax = 3 * nside - 1
    bins = module.NmtBin.from_lmax_linear(lmax, nlb)
    label = getattr(module, "__name__", type(module).__name__)

    def _note(msg):
        if verbose:
            print(f"  [{label}] {msg}", flush=True)

    def full_pipeline():
        ff = _make_field(module, mask, maps_t, maps_q, maps_u, spin)
        ww = module.NmtWorkspace()
        ww.compute_coupling_matrix(ff, ff, bins)
        cc = module.compute_coupled_cell(ff, ff)
        return ww.decouple_cell(cc)

    t0 = time.perf_counter()
    _note("field")
    f = _make_field(module, mask, maps_t, maps_q, maps_u, spin)
    _note(f"field done {time.perf_counter()-t0:.1f}s; coupling")
    w = module.NmtWorkspace()
    w.compute_coupling_matrix(f, f, bins)
    _note(f"coupling done {time.perf_counter()-t0:.1f}s; coupled_cell")
    cl_coupled = module.compute_coupled_cell(f, f)
    cl_decoupled = w.decouple_cell(cl_coupled)
    _note(f"first pipeline done {time.perf_counter()-t0:.1f}s")
    result_host = np.asarray(cl_decoupled)

    # Nside 4096 spin 2 cannot hold the first field+workspace and a second field
    # at once (8 GiB chirp-Z request with 70.9/71.2 GiB in use).  Release the
    # compile-time objects and time only a fresh full pipeline.
    if release_after_first:
        del f, w, cl_coupled, cl_decoupled
        gc.collect()
        try:
            nmt.utils.drop_ring_tables()
            nmt.utils.make_room(16 * 1024 ** 3)
        except Exception:
            pass
        _, t_total = _timed(full_pipeline, repeats)
        times = {
            "field": float("nan"),
            "mask": float("nan"),
            "coupling": float("nan"),
            "coupled_cell": float("nan"),
            "decouple": float("nan"),
            "total": t_total,
        }
        return result_host, times

    t_field = _timed(
        lambda: _make_field(module, mask, maps_t, maps_q, maps_u, spin),
        repeats,
    )[1]

    # A field's mask alms are lazy in both codes (NaMaster's own `get_mask_alms` docstring says
    # "in most cases ... are not computed when generating the field ... which may be a slow
    # operation"). `lmax_mask` defaults to `minfo.get_lmax()` in both libraries and the workspace
    # does not enlarge it (`.qwen/tmp/lmaxmask_s31.log`), so this is a second spin-0 transform at
    # the same order and the same `n_iter` as the science stage -- which is why it costs 1.00x of
    # it. Timing a fresh field and subtracting the construction measures its marginal cost; on a
    # shared field it is cached and every column would report zero. At Nside 2048 spin 0 this
    # marginal is 1855 ms against a 4210 ms TOTAL (`.qwen/tmp/maskstage_s30.log`).
    t_mask = _timed(
        lambda: _make_field(
            module, mask, maps_t, maps_q, maps_u, spin
        ).get_mask_alms(),
        repeats,
    )[1] - t_field

    def fresh_coupling():
        ww = module.NmtWorkspace()
        ww.compute_coupling_matrix(f, f, bins)
        return ww

    _, t_coupling = _timed(fresh_coupling, repeats)
    _, t_coupled_cell = _timed(lambda: module.compute_coupled_cell(f, f), repeats)
    _, t_decouple = _timed(lambda: w.decouple_cell(cl_coupled), repeats)

    # At Nside 4096 spin 0 this TOTAL is 7.46 s while the same pipeline on its own
    # (`--skip-reference`, which releases the compile-time field) is 4.90 s.  The difference is
    # the repeats themselves: each `full_pipeline` builds a field and a workspace while the
    # previous one is still bound, and under that pressure the march's window tables are evicted
    # and rebuilt.  A `gc.collect()` here was measured and does not help (7.459 -> 7.456 s), so
    # the harness is left alone and both numbers are reported.
    _, t_total = _timed(full_pipeline, repeats)
    times = {
        "field": t_field,
        "mask": max(t_mask, 0.0),
        "coupling": t_coupling,
        "coupled_cell": t_coupled_cell,
        "decouple": t_decouple,
        "total": t_total,
    }
    # recompute decoupled once more for parity reporting
    final = full_pipeline() if module is reference else None
    return (final if final is not None else cl_decoupled), times


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nside", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--spins", type=str, default="0,2")
    parser.add_argument(
        "--precision",
        type=str,
        default="fp64",
        help="GMaster table precision; the shipped default is fp64.",
    )
    parser.add_argument(
        "--ring-precision",
        type=str,
        default="auto",
        choices=("auto", "follow", "fp64", "fp32"),
        help="azimuthal transform precision independently of the tables.  'auto' is the "
        "shipped default (complex64 where the v2 march serves the latitudinal stage); "
        "'follow' tracks the tables and is what the pre-v2 rows used; 'fp64' keeps the map "
        "exact at every size.",
    )
    parser.add_argument(
        "--reference-cpus",
        type=int,
        default=0,
        help="Pin the whole benchmark to this many CPU cores (affinity + OMP_NUM_THREADS) "
        "and re-exec once.  The published board is the unpinned 96-core host; `--reference-cpus "
        "32` is the smaller-host comparison, where NaMaster gets 32 cores and GMaster's host "
        "side is limited the same way.",
    )
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Time GMaster only. Default is both codes; Nside=4096 spin 2 uses "
        "gmaster._nmt_bin64 so pymaster can bin the MCM past INT_MAX.",
    )
    args = parser.parse_args()
    nmt.set_table_precision(args.precision)
    nmt.set_ring_precision(args.ring_precision)
    patch_pymaster()

    nside = args.nside
    npix = 12 * nside**2
    rng = np.random.default_rng(5)
    theta = hp.pix2ang(nside, np.arange(npix))[0]
    mask_np = np.clip((np.cos(theta) + 0.35) / 0.7, 0, 1) ** 2

    map_t = rng.normal(size=npix)
    map_q = rng.normal(size=npix)
    map_u = rng.normal(size=npix)

    print(
        f"nside={nside} precision={args.precision} ring={args.ring_precision} "
        f"coupling={nmt.coupling_precision()} cpus={len(os.sched_getaffinity(0))} "
        f"omp={os.environ.get('OMP_NUM_THREADS', 'unset')} "
        f"devices={[str(d) for d in jax.devices()]}"
    )
    stages = ("field", "mask", "coupling", "coupled_cell", "decouple")
    import resource
    for spin in [int(s) for s in args.spins.split(',') if s.strip()]:
        rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        stats = jax.devices()[0].memory_stats() or {}
        gpu_peak = stats.get("peak_bytes_in_use", stats.get("bytes_in_use", 0)) / 2**30
        gm_maps = (
            jnp.asarray(mask_np), jnp.asarray(map_t),
            jnp.asarray(map_q), jnp.asarray(map_u),
        )
        large_spin2 = nside >= 4096 and spin == 2
        if args.skip_reference:
            gm_out, gm_times = _run_pipeline(
                nmt, nside, spin, 30, args.repeats, *gm_maps,
                release_after_first=True, verbose=large_spin2,
            )
            rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
            stats = jax.devices()[0].memory_stats() or {}
            gpu_peak = stats.get("peak_bytes_in_use", stats.get("bytes_in_use", 0)) / 2**30
            gm_arr = np.asarray(gm_out)
            finite = bool(np.isfinite(gm_arr).all())
            def _ms(name):
                t = gm_times[name]
                return f"{name} n/a" if t != t else f"{name} {t*1e3:.0f}ms"
            line = "  ".join(_ms(s) for s in stages)
            print(
                f"spin={spin}: GMaster-only TOTAL {gm_times['total']*1e3:.0f}ms | "
                f"{line} | finite={finite} shape={tuple(gm_arr.shape)} | "
                f"peakRSS={rss_gb:.1f}GB GPUpeak={gpu_peak:.1f}GiB"
            )
            continue
        ref_out, ref_times = _run_pipeline(
            reference, nside, spin, 30, args.repeats,
            mask_np, map_t, map_q, map_u,
            release_after_first=large_spin2, verbose=large_spin2,
        )
        if large_spin2:
            gc.collect()
            try:
                nmt.utils.drop_ring_tables()
                nmt.utils.make_room(16 * 1024 ** 3)
            except Exception:
                pass
        gm_out, gm_times = _run_pipeline(
            nmt, nside, spin, 30, args.repeats, *gm_maps,
            release_after_first=large_spin2, verbose=large_spin2,
        )
        rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        stats = jax.devices()[0].memory_stats() or {}
        gpu_peak = stats.get("peak_bytes_in_use", stats.get("bytes_in_use", 0)) / 2**30
        diff = float(np.max(np.abs(np.asarray(gm_out) - np.asarray(ref_out))))
        scale = float(np.max(np.abs(ref_out))) or 1.0
        def _pair(s):
            a, b = ref_times[s], gm_times[s]
            if a != a or b != b:
                return f"{s} n/a"
            return f"{s} {a*1e3:.0f}->{b*1e3:.0f}ms ({a/max(b, 1e-12):.0f}x)"
        line = "  ".join(_pair(s) for s in stages)
        print(
            f"spin={spin}: TOTAL {ref_times['total']*1e3:.0f}->"
            f"{gm_times['total']*1e3:.0f}ms "
            f"({ref_times['total']/gm_times['total']:.1f}x) | {line} | "
            f"max|dCl|={diff:.2e} rel={diff/scale:.2e} | "
            f"peakRSS={rss_gb:.1f}GB GPUpeak={gpu_peak:.1f}GiB"
        )
