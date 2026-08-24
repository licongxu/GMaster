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


def _timed(function, repeats):
    result = function()
    if hasattr(result, "block_until_ready"):
        result.block_until_ready()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
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
    args = parser.parse_args()

    nside = args.nside
    npix = 12 * nside**2
    rng = np.random.default_rng(5)
    theta = hp.pix2ang(nside, np.arange(npix))[0]
    mask_np = np.clip((np.cos(theta) + 0.35) / 0.7, 0, 1) ** 2

    map_t = rng.normal(size=npix)
    map_q = rng.normal(size=npix)
    map_u = rng.normal(size=npix)

    print(f"nside={nside} devices={[str(d) for d in jax.devices()]}")
    for spin in (0, 2):
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
        stages = ("field", "coupling", "coupled_cell", "decouple")
        line = "  ".join(
            f"{s} {ref_times[s]*1e3:.0f}->{gm_times[s]*1e3:.0f}ms "
            f"({ref_times[s]/max(gm_times[s], 1e-12):.0f}x)"
            for s in stages
        )
        print(
            f"spin={spin}: TOTAL {ref_times['total']*1e3:.0f}->"
            f"{gm_times['total']*1e3:.0f}ms "
            f"({ref_times['total']/gm_times['total']:.1f}x) | {line} | "
            f"max|dCl|={diff:.2e} rel={diff/scale:.2e}"
        )
