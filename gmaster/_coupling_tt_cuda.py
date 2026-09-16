"""CUDA threej scalar MASTER matrix: one kernel, O(n^2) device memory.

The JAX path in :func:`gmaster.workspaces._coupling_matrix_tt` scans padded
``(16, n, n)`` gather chunks.  At lmax 3071 that is ~576 MiB of temps per
iteration and on a Colab T4 the warmed estimator stayed at ~13 min against
NaMaster's ~3 min.  This kernel evaluates the same float32 products and float64
offset accumulator, writing only the ``(n, n)`` matrix.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

_CU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cuda", "coupling_tt.cu")
_NAME = "gm_coupling_tt"
_LIB = None
_LIB_ERROR = None


def _nvcc():
    cand = [os.environ.get("GMASTER_NVCC", ""), "/usr/local/cuda/bin/nvcc", "nvcc"]
    for c in cand:
        if not c:
            continue
        try:
            out = subprocess.run([c, "--version"], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if out.returncode == 0 and "release 1" in out.stdout:
            return c
    return None


def _build():
    """Compile the CUDA source into a per-source-hash shared library."""
    global _LIB, _LIB_ERROR
    if _LIB is not None or _LIB_ERROR is not None:
        return _LIB
    try:
        if os.environ.get("GMASTER_COUPLING_CUDA", "1") != "1":
            raise RuntimeError("GMASTER_COUPLING_CUDA=0")
        devs = [d for d in jax.devices() if d.platform == "gpu"]
        if not devs:
            raise RuntimeError("no GPU device")
        cc = str(getattr(devs[0], "compute_capability", "")).replace(".", "")
        if not cc:
            raise RuntimeError("unknown compute capability")
        src = open(_CU).read()
        digest = hashlib.sha1((src + cc + jax.__version__).encode()).hexdigest()[:12]
        cache = os.environ.get(
            "GMASTER_CUDA_CACHE",
            os.path.join(os.path.expanduser("~"), ".cache", "gmaster"),
        )
        os.makedirs(cache, exist_ok=True)
        so = os.path.join(cache, f"libgm_coupling_tt_{digest}.so")
        if not os.path.exists(so):
            nvcc = _nvcc()
            if nvcc is None:
                raise RuntimeError("nvcc not found (set GMASTER_NVCC)")
            inc = jax.ffi.include_dir()
            tmp = so + f".{os.getpid()}.tmp"
            cmd = [
                nvcc, "-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC",
                f"-arch=sm_{cc}", "-diag-suppress", "940,2473",
                "-I", inc, "-o", tmp, _CU,
            ]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            if out.returncode != 0:
                raise RuntimeError("nvcc failed:\n" + out.stderr[-4000:])
            os.replace(tmp, so)
        lib = ctypes.CDLL(so)
        jax.ffi.register_ffi_target(
            _NAME, jax.ffi.pycapsule(getattr(lib, _NAME)), platform="CUDA",
        )
        _LIB = lib
    except Exception as exc:  # noqa: BLE001 - any failure means "route unavailable"
        _LIB_ERROR = exc
        if os.environ.get("GMASTER_COUPLING_CUDA_VERBOSE"):
            print(f"[gmaster] CUDA TT coupling unavailable: {exc}")
    return _LIB


def enabled() -> bool:
    """True when the CUDA threej kernel can serve a scalar coupling build."""
    return _build() is not None


def unavailable_reason():
    return _LIB_ERROR


@partial(jax.jit, static_argnames="lmax")
def coupling_tt(mask_power, g, *, lmax):
    """``(lmax+1, lmax+1)`` float64 matrix from float32 ``mask_power`` and ``g`` tables."""
    if _build() is None:
        raise RuntimeError(f"CUDA TT coupling unavailable: {_LIB_ERROR}")
    n = lmax + 1
    out_t = jax.ShapeDtypeStruct((n, n), jnp.float64)
    return jax.ffi.ffi_call(_NAME, out_t, vmap_method="broadcast_all")(
        mask_power, g, lmax=np.int64(lmax),
    )
