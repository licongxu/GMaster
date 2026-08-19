"""Benchmark warmed HEALPix transforms against NaMaster and across GPUs."""

import argparse
import time

import jax
import numpy as np
import pymaster as reference

jax.config.update("jax_enable_x64", True)

import gmaster as nmt


def _timed(function, repeats):
    start = time.perf_counter()
    result = function()
    if hasattr(result, "block_until_ready"):
        result.block_until_ready()
    compile_seconds = time.perf_counter() - start
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        samples.append(time.perf_counter() - start)
    return result, compile_seconds, float(np.median(samples))


def _random_alms(rng, nmaps, alm_info, spin):
    alms = rng.normal(size=(nmaps, alm_info.nelem)) + 1j * rng.normal(
        size=(nmaps, alm_info.nelem)
    )
    alms[:, alm_info._m == 0] = alms[:, alm_info._m == 0].real
    alms[:, alm_info._ell < spin] = 0
    return alms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nside", type=int, default=512)
    parser.add_argument("--spin", type=int, default=0)
    parser.add_argument("--n-iter", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    nside = args.nside
    lmax = 3 * nside - 1
    npix = 12 * nside**2
    nmaps = 1 if args.spin == 0 else 2
    map_info = nmt.NmtMapInfo(None, (npix,))
    alm_info = nmt.NmtAlmInfo(lmax)
    ref_map_info = reference.NmtMapInfo(None, (npix,))
    ref_alm_info = reference.NmtAlmInfo(lmax)
    rng = np.random.default_rng(7)
    host_maps = rng.normal(size=(nmaps, npix))
    host_alms = _random_alms(rng, nmaps, alm_info, args.spin)
    maps = jax.device_put(host_maps, jax.devices()[0])
    alms = jax.device_put(host_alms, jax.devices()[0])

    ref_analysis, _, ref_analysis_s = _timed(
        lambda: reference.map2alm(
            host_maps,
            args.spin,
            ref_map_info,
            ref_alm_info,
            n_iter=args.n_iter,
        ),
        args.repeats,
    )
    ref_synthesis, _, ref_synthesis_s = _timed(
        lambda: reference.alm2map(
            host_alms, args.spin, ref_map_info, ref_alm_info
        ),
        args.repeats,
    )

    rows = []
    calculators = ["jax-single"]
    if len([device for device in jax.devices() if device.platform == "gpu"]) >= 2:
        calculators.append("jax-mgpu")
    for calculator in calculators:
        nmt.set_sht_calculator(calculator)
        analysis, analysis_compile_s, analysis_s = _timed(
            lambda: nmt.map2alm(
                maps, args.spin, map_info, alm_info, n_iter=args.n_iter
            ),
            args.repeats,
        )
        synthesis, synthesis_compile_s, synthesis_s = _timed(
            lambda: nmt.alm2map(alms, args.spin, map_info, alm_info),
            args.repeats,
        )
        analysis_diff = float(np.max(np.abs(np.asarray(analysis) - ref_analysis)))
        synthesis_diff = float(
            np.max(np.abs(np.asarray(synthesis) - ref_synthesis))
        )
        rows.append(
            (
                calculator,
                analysis_compile_s,
                analysis_s,
                ref_analysis_s / analysis_s,
                analysis_diff,
                analysis_diff / float(np.max(np.abs(ref_analysis))),
                synthesis_compile_s,
                synthesis_s,
                ref_synthesis_s / synthesis_s,
                synthesis_diff,
                synthesis_diff / float(np.max(np.abs(ref_synthesis))),
            )
        )

    print(f"devices: {jax.devices()}")
    print(f"Nside={nside}, lmax={lmax}, spin={args.spin}, n_iter={args.n_iter}")
    print(
        f"NaMaster warmed: analysis={ref_analysis_s:.6f}s, "
        f"synthesis={ref_synthesis_s:.6f}s"
    )
    print(
        "backend     analysis compile/warm  vs NMT  max/rel |dalm|       "
        "synthesis compile/warm  vs NMT  max/rel |dmap|"
    )
    for row in rows:
        name, ac, analysis_s, asp, adiff, arel, sc, synthesis_s, ssp, sdiff, srel = row
        print(
            f"{name:10s} {ac:8.3f}/{analysis_s:8.4f}s "
            f"{asp:7.3f}x {adiff:9.2e}/{arel:9.2e}   "
            f"{sc:8.3f}/{synthesis_s:8.4f}s "
            f"{ssp:7.3f}x {sdiff:9.2e}/{srel:9.2e}"
        )
