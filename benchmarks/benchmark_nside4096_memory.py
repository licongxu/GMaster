"""Compile a small SHT and extrapolate its XLA buffers to a target Nside."""

import argparse

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from gmaster.utils import (
    NmtAlmInfo,
    NmtMapInfo,
    _copy_to_device,
    _forward_latitudinal_device,
    _forward_s2fft_ftm,
    _gpu_devices,
    _inverse_latitudinal_device,
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


def compiled_multi_gpu_memory(nside, spin):
    devices = _gpu_devices()
    if len(devices) < 2:
        return None
    L = 3 * nside
    reality = spin == 0
    dtype = jnp.float64 if reality else jnp.complex128
    maps = jax.device_put(jnp.zeros(12 * nside**2, dtype=dtype), devices[0])
    ftm = _forward_s2fft_ftm(maps, L=L, nside=nside, reality=reality)
    ftm.block_until_ready()
    split = 2 * L // 3
    low_ftm = _copy_to_device(ftm[:, L - split : L + split], devices[1])
    forward = (
        _forward_latitudinal_device(
            L, spin, nside, reality, split, 0
        ).lower(ftm).compile().memory_analysis(),
        _forward_latitudinal_device(
            split, spin, nside, reality, 0, 1
        ).lower(low_ftm).compile().memory_analysis(),
    )
    flm = jax.device_put(
        jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128), devices[0]
    )
    other_flm = _copy_to_device(flm, devices[1])
    inverse = (
        _inverse_latitudinal_device(
            L, spin, nside, reality, 0, 0
        ).lower(flm).compile().memory_analysis(),
        _inverse_latitudinal_device(
            L, spin, nside, reality, 1, 1
        ).lower(other_flm).compile().memory_analysis(),
    )
    return forward, inverse


def _compiled_peak(analysis):
    return (
        analysis.argument_size_in_bytes
        + analysis.output_size_in_bytes
        + analysis.temp_size_in_bytes
        - analysis.alias_size_in_bytes
    )


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

    if len(_gpu_devices()) >= 2:
        print("two-GPU staged recurrence peak per device")
        print("spin  forward GPU0/GPU1       inverse GPU0/GPU1")
        for spin in (0, 2):
            forward, inverse = compiled_multi_gpu_memory(args.probe_nside, spin)
            forward = [_gib(_compiled_peak(item) * scale) for item in forward]
            inverse = [_gib(_compiled_peak(item) * scale) for item in inverse]
            print(
                f"{spin:4d}  {forward[0]:7.2f}/{forward[1]:7.2f} GiB  "
                f"{inverse[0]:7.2f}/{inverse[1]:7.2f} GiB"
            )
        print(
            "Staged figures include each recurrence call's arguments, output, and "
            "temporaries; allow additional space for caller-retained maps and alms."
        )
