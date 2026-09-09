"""End-to-end MASTER pipeline benchmark: GMaster (GPU) versus NaMaster (CPU).

Stages per configuration: field construction, coupling matrix, coupled cell,
and decoupled cell. Reports warmed medians and decoupled-spectrum parity.
"""

import argparse
import time

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pymaster as reference

jax.config.update("jax_enable_x64", True)

import gmaster as nmt


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


def _run_pipeline(module, nside, spin, nlb, repeats, mask, maps_t, maps_q, maps_u):
    lmax = 3 * nside - 1
    bins = module.NmtBin.from_lmax_linear(lmax, nlb)
    f = _make_field(module, mask, maps_t, maps_q, maps_u, spin)
    w = module.NmtWorkspace()
    w.compute_coupling_matrix(f, f, bins)
    cl_coupled = module.compute_coupled_cell(f, f)
    cl_decoupled = w.decouple_cell(cl_coupled)

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

    def full_pipeline():
        ff = _make_field(module, mask, maps_t, maps_q, maps_u, spin)
        ww = module.NmtWorkspace()
        ww.compute_coupling_matrix(ff, ff, bins)
        cc = module.compute_coupled_cell(ff, ff)
        return ww.decouple_cell(cc)

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
        default="follow",
        choices=("follow", "fp64", "fp32"),
        help="azimuthal transform precision independently of the tables.  'follow' is the "
        "shipped coupling and what every published row used; 'fp64' keeps the map exact "
        "under fp32 tables, which is the route that passes the suite.",
    )
    args = parser.parse_args()
    nmt.set_table_precision(args.precision)
    nmt.set_ring_precision(args.ring_precision)

    nside = args.nside
    npix = 12 * nside**2
    rng = np.random.default_rng(5)
    theta = hp.pix2ang(nside, np.arange(npix))[0]
    mask_np = np.clip((np.cos(theta) + 0.35) / 0.7, 0, 1) ** 2

    map_t = rng.normal(size=npix)
    map_q = rng.normal(size=npix)
    map_u = rng.normal(size=npix)

    print(
        f"nside={nside} precision={args.precision} ring={nmt.ring_dtype().__name__} "
        f"devices={[str(d) for d in jax.devices()]}"
    )
    for spin in [int(s) for s in args.spins.split(',') if s.strip()]:
        ref_out, ref_times = _run_pipeline(
            reference, nside, spin, 30, args.repeats,
            mask_np, map_t, map_q, map_u,
        )
        gm_out, gm_times = _run_pipeline(
            nmt, nside, spin, 30, args.repeats,
            jnp.asarray(mask_np), jnp.asarray(map_t),
            jnp.asarray(map_q), jnp.asarray(map_u),
        )
        diff = float(np.max(np.abs(np.asarray(gm_out) - np.asarray(ref_out))))
        scale = float(np.max(np.abs(ref_out))) or 1.0
        stages = ("field", "mask", "coupling", "coupled_cell", "decouple")
        line = "  ".join(
            f"{s} {ref_times[s]*1e3:.0f}->{gm_times[s]*1e3:.0f}ms "
            f"({ref_times[s]/max(gm_times[s], 1e-12):.0f}x)"
            for s in stages
        )
        import resource
        rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        gmem = jax.devices()[0].memory_stats()
        print(
            f"spin={spin}: TOTAL {ref_times['total']*1e3:.0f}->"
            f"{gm_times['total']*1e3:.0f}ms "
            f"({ref_times['total']/gm_times['total']:.1f}x) | {line} | "
            f"max|dCl|={diff:.2e} rel={diff/scale:.2e} | "
            f"peakRSS={rss_gb:.1f}GB GPUpeak={gmem.get('bytes_in_use', 0)/2**30:.1f}GiB"
        )
