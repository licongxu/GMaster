"""CUDA kernel for the scalar (spin-0 x spin-0) mode-coupling matrix.

Builds the same ``(lmax+1, lmax+1)`` matrix as
:func:`gmaster.workspaces._coupling_matrix_tt` (Wigner-3j products in float32,
offset accumulation in float64) in a single kernel that needs only O(n^2)
device memory for the output. The JAX path instead materialises padded
``(16, n, n)`` gather chunks, about 576 MiB of temporaries per step at
lmax 3071, which makes it memory-bound on small GPUs.

The source ``_native/cuda/coupling_tt.cu`` is compiled with ``nvcc`` on first
use and cached per source hash, compute capability and JAX version.

Environment variables
---------------------
GMASTER_COUPLING_CUDA
    Set to anything other than ``"1"`` to disable this kernel (default ``"1"``).
GMASTER_NVCC
    Path to ``nvcc``; otherwise ``/usr/local/cuda/bin/nvcc`` and ``nvcc`` on PATH.
GMASTER_CUDA_CACHE
    Directory for compiled libraries (default ``~/.cache/gmaster``).
GMASTER_COUPLING_CUDA_VERBOSE
    If set, print why the kernel is unavailable.
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

_CU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_native", "cuda", "coupling_tt.cu")
_NAME = "gm_coupling_tt"
_LIB = None
_LIB_ERROR = None


def _nvcc():
    """Return a working CUDA ``nvcc`` executable, or None."""
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
    """Compile and register the FFI target once; return the library or None.

    Any failure (no GPU, no nvcc, compile error, disabled by environment) is
    stored in ``_LIB_ERROR`` and makes the caller fall back to the JAX path.
    """
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
    """Exception explaining why the kernel is unavailable, or None."""
    return _LIB_ERROR


@partial(jax.jit, static_argnames="lmax")
def coupling_tt(mask_power, g, *, lmax):
    """Scalar mode-coupling matrix on the GPU.

    Parameters
    ----------
    mask_power : jax.Array, float32
        Mask (cross-)power spectrum table, as prepared by the JAX path.
    g : jax.Array, float32
        Wigner-3j helper table, as prepared by the JAX path.
    lmax : int
        Maximum multipole (static).

    Returns
    -------
    jax.Array
        ``(lmax+1, lmax+1)`` float64 coupling matrix.
    """
    if _build() is None:
        raise RuntimeError(f"CUDA TT coupling unavailable: {_LIB_ERROR}")
    n = lmax + 1
    out_t = jax.ShapeDtypeStruct((n, n), jnp.float64)
    return jax.ffi.ffi_call(_NAME, out_t, vmap_method="broadcast_all")(
        mask_power, g, lmax=np.int64(lmax),
    )
