"""CPU AVX-512 fp32 (v, D) march (gmaster.cpu_dform): the plan/README.md experiment kernel."""
from __future__ import annotations

import shutil

import numpy as np
import pytest

pytest.importorskip("ducc0")
ref = pytest.importorskip("pymaster")


def _avx512():
    try:
        return "avx512f" in open("/proc/cpuinfo").read()
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not (_avx512() and shutil.which("gcc")),
                                reason="needs gcc and an AVX-512 host")


@pytest.mark.parametrize("nside", [32, 64])
def test_cpu_dform_map2alm_matches_namaster(nside):
    from gmaster import cpu_dform as cd

    L = 3 * nside
    npix = 12 * nside ** 2
    maps = np.random.default_rng(7).normal(size=npix)
    want = ref.map2alm(maps[None, :], 0, ref.NmtMapInfo(None, (npix,)), ref.NmtAlmInfo(L - 1),
                       n_iter=0)[0]
    got_many = cd.to_packed(cd.as_alm(cd.map2alm_spin0(maps, nside, 8)), L)
    got_one = cd.to_packed(cd.as_alm(cd.map2alm_spin0(maps, nside, 1)), L)
    assert np.isfinite(got_many).all()
    scale = np.max(np.abs(want))
    assert np.max(np.abs(got_many - want)) / scale < 1e-5
    assert np.max(np.abs(got_many - got_one)) / scale < 1e-5
