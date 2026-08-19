"""Warmed selective-Toeplitz GPU benchmark against NaMaster's exact kernel."""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import pymaster as reference

jax.config.update("jax_enable_x64", True)

import gmaster


def _block(coefficients):
    coefficients["00"].block_until_ready()


def _timed(call, *, warm=False):
    if warm:
        _block(call())
    start = time.perf_counter()
    value = call()
    if isinstance(value, dict) and isinstance(value["00"], jax.Array):
        _block(value)
    return value, time.perf_counter() - start


def run(lmax, l_toeplitz, l_exact, dl_band):
    options = {
        "l_toeplitz": l_toeplitz,
        "l_exact": l_exact,
        "dl_band": dl_band,
    }
    window = np.exp(-np.arange(2 * lmax + 1, dtype=np.float64) / 100)
    device_window = jnp.asarray(window)
    call = lambda: gmaster.get_master_coefficients(
        device_window, lmax, 0, 0, **options
    )
    gpu, gpu_time = _timed(call, warm=True)
    exact, exact_time = _timed(
        lambda: reference.get_master_coefficients(window, lmax, 0, 0)
    )
    approximate, approximate_time = _timed(
        lambda: reference.get_master_coefficients(
            window, lmax, 0, 0, **options
        )
    )
    error = np.max(np.abs(np.asarray(gpu["00"]) - approximate["00"]))
    return gpu_time, exact_time, approximate_time, error


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmax", type=int, nargs="+", default=[2047, 4095])
    parser.add_argument("--l-toeplitz", type=int, default=600)
    parser.add_argument("--l-exact", type=int, default=100)
    parser.add_argument("--dl-band", type=int, default=50)
    args = parser.parse_args()
    print(f"JAX device: {jax.devices()[0]}")
    print(
        "lmax  GMaster (s)  NaMaster exact (s)  speedup  "
        "NaMaster Toeplitz (s)  max |delta|"
    )
    for lmax in args.lmax:
        gpu, exact, approximate, error = run(
            lmax, args.l_toeplitz, args.l_exact, args.dl_band
        )
        print(
            f"{lmax:4d}  {gpu:11.4f}  {exact:18.4f}  {exact / gpu:7.2f}x  "
            f"{approximate:20.4f}  {error:11.3e}"
        )
