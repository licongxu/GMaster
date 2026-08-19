"""Warmed GMaster GPU versus NaMaster CPU flat-covariance benchmark."""

import argparse
import time

import jax
import numpy as np
import pymaster as reference

jax.config.update("jax_enable_x64", True)

import gmaster


def run(size):
    lx = ly = 0.2
    y, x = np.mgrid[:size, :size]
    mask = (np.sin(np.pi * (x + 0.5) / size) *
            np.sin(np.pi * (y + 0.5) / size)) ** 2
    maps = np.zeros((1, size, size))
    ell_max = np.hypot(np.pi * size / lx, np.pi * size / ly)
    edges = np.linspace(0, 0.95 * ell_max, 9)
    bins = gmaster.NmtBinFlat(edges[:-1], edges[1:])
    reference_bins = reference.NmtBinFlat(edges[:-1], edges[1:])
    field = gmaster.NmtFieldFlat(lx, ly, mask, maps)
    reference_field = reference.NmtFieldFlat(lx, ly, mask, maps)

    covariance = gmaster.NmtCovarianceWorkspaceFlat.from_fields(
        field, field, bins
    )
    covariance.xi[0]["00"].block_until_ready()
    start = time.perf_counter()
    covariance = gmaster.NmtCovarianceWorkspaceFlat.from_fields(
        field, field, bins
    )
    covariance.xi[0]["00"].block_until_ready()
    gpu_seconds = time.perf_counter() - start

    start = time.perf_counter()
    reference.NmtCovarianceWorkspaceFlat.from_fields(
        reference_field, reference_field, reference_bins
    )
    cpu_seconds = time.perf_counter() - start
    return gpu_seconds, cpu_seconds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512])
    args = parser.parse_args()
    print(f"JAX device: {jax.devices()[0]}")
    print("pixels    GMaster (s)    NaMaster (s)    speedup")
    for size in args.sizes:
        gpu, cpu = run(size)
        print(f"{size:4d}x{size:<4d} {gpu:12.4f} {cpu:15.4f} {cpu / gpu:10.2f}x")
