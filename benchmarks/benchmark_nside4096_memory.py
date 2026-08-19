"""Compile a small SHT and extrapolate its XLA buffers to a target Nside."""

import argparse

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from gmaster.utils import (
    NmtAlmInfo,
    NmtMapInfo,
    _map2alm_iteration,
    _map2alm_once,
)


def _gib(value):
    return value / 2**30


def compiled_memory(nside, spin):
    lmax = 3 * nside - 1
    info = NmtMapInfo(None, (12 * nside**2,))
    alms = NmtAlmInfo(lmax)
    nmaps = 1 if spin == 0 else 2
    maps = jnp.zeros((nmaps, info.npix), dtype=jnp.float64)
    alm = jnp.zeros((nmaps, alms.nelem), dtype=jnp.complex128)
    options = dict(spin=spin, nside=nside, L=lmax + 1, L_work=lmax + 1)
    initial = _map2alm_once.lower(
        maps, alms._ell, alms._m, **options
    ).compile().memory_analysis()
    iteration = _map2alm_iteration.lower(
        alm, maps, alms._ell, alms._m, **options
    ).compile().memory_analysis()
    return initial, iteration


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-nside", type=int, default=4096)
    parser.add_argument("--probe-nside", type=int, default=16)
    args = parser.parse_args()
    target = args.target_nside
    scale = (target / args.probe_nside) ** 2
    npix = 12 * target**2
    lmax = 3 * target - 1
    nelem = (lmax + 1) * (lmax + 2) // 2

    print(f"JAX device: {jax.devices()[0]}")
    print(f"target Nside={target}, npix={npix:,}, lmax={lmax:,}")
    print(f"one float64 map: {_gib(npix * 8):.3f} GiB")
    print(f"one packed complex128 alm: {_gib(nelem * 16):.3f} GiB")
    print("spin  initial temp  Jacobi temp  estimated peak per compiled step")
    for spin in (0, 2):
        initial, iteration = compiled_memory(args.probe_nside, spin)
        initial_temp = initial.temp_size_in_bytes * scale
        iteration_temp = iteration.temp_size_in_bytes * scale
        nmaps = 1 if spin == 0 else 2
        arguments = nmaps * (npix * 8 + nelem * 16)
        peak = max(initial_temp, iteration_temp + arguments)
        print(
            f"{spin:4d}  {_gib(initial_temp):12.2f}  "
            f"{_gib(iteration_temp):11.2f}  {_gib(peak):31.2f} GiB"
        )
    print("Estimates scale the probe graph quadratically; confirm with hardware telemetry.")
