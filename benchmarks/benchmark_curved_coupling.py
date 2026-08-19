"""Warmed GMaster GPU versus NaMaster CPU curved coupling benchmark."""

import argparse
import time

import jax
import numpy as np
import pymaster as reference

jax.config.update("jax_enable_x64", True)

import gmaster


def timed(call, result):
    result(call()).block_until_ready()
    start = time.perf_counter()
    value = call()
    result(value).block_until_ready()
    return value, time.perf_counter() - start


def run(lmax):
    general_mask = np.exp(-np.arange(lmax + 1) / max(lmax / 10, 1))
    pure_mask = np.exp(-np.arange(2 * lmax + 1) / max(lmax / 10, 1))

    general, general_gpu = timed(
        lambda: gmaster.get_general_coupling_matrix(
            general_mask, 1, 3, 1, 3, parity="both"
        ),
        lambda value: value,
    )
    start = time.perf_counter()
    general_reference = reference.get_general_coupling_matrix(
        general_mask, 1, 3, 1, 3, parity="both"
    )
    general_cpu = time.perf_counter() - start

    pure, pure_gpu = timed(
        lambda: gmaster.get_master_coefficients(
            pure_mask, lmax, 0, 2, pure_any=True
        ),
        lambda value: value["0s"],
    )
    start = time.perf_counter()
    pure_reference = reference.get_master_coefficients(
        pure_mask, lmax, 0, 2, pure_any=True
    )
    pure_cpu = time.perf_counter() - start

    general_error = np.max(np.abs(np.asarray(general) - general_reference))
    pure_error = np.max(
        np.abs(np.asarray(pure["0s"][1]) - pure_reference["0s"][1])
    )
    return general_gpu, general_cpu, general_error, pure_gpu, pure_cpu, pure_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmax", type=int, nargs="+", default=[511, 1023, 2047])
    args = parser.parse_args()
    print(f"JAX device: {jax.devices()[0]}")
    print("lmax  kernel       GMaster (s)  NaMaster (s)  speedup    max |delta|")
    for lmax in args.lmax:
        general_gpu, general_cpu, general_error, pure_gpu, pure_cpu, pure_error = (
            run(lmax)
        )
        print(
            f"{lmax:4d}  general-spin {general_gpu:11.4f} {general_cpu:13.4f} "
            f"{general_cpu / general_gpu:8.2f}x {general_error:13.3e}"
        )
        print(
            f"{lmax:4d}  pure 0x2     {pure_gpu:11.4f} {pure_cpu:13.4f} "
            f"{pure_cpu / pure_gpu:8.2f}x {pure_error:13.3e}"
        )
