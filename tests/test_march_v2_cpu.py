"""CPU OpenMP v2 march: same recurrence as CUDA, served when JAX is on CPU."""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.march_v2
def test_cpu_march_library_builds():
    from gmaster import _march_v2

    lib = _march_v2._build_cpu()
    assert lib is not None, _march_v2._CPU_LIB_ERROR


@pytest.mark.march_v2
def test_cpu_v2_map2alm_agrees_with_namaster():
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["GMASTER_MARCH_V2"] = "1"
    env["JAX_ENABLE_X64"] = "1"
    env.pop("GMASTER_MARCH_V2_CPU", None)
    code = r"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pymaster as ref
import gmaster as nmt
from gmaster import _march_v2
assert jax.default_backend() == "cpu", jax.devices()
assert _march_v2.enabled(96), _march_v2.unavailable_reason()
nside, spin = 32, 0
lmax = 3 * nside - 1
npix = 12 * nside ** 2
rng = np.random.default_rng(19)
maps = rng.normal(size=(1, npix))
got = np.asarray(nmt.map2alm(jnp.asarray(maps), spin,
                             nmt.NmtMapInfo(None, (npix,)), nmt.NmtAlmInfo(lmax), n_iter=0))
want = ref.map2alm(maps, spin, ref.NmtMapInfo(None, (npix,)), ref.NmtAlmInfo(lmax), n_iter=0)
scale = float(np.max(np.abs(want)))
err = float(np.max(np.abs(got - want))) / scale
print("rel", err)
assert err < 1e-5, err
"""
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + "\n" + out.stderr
