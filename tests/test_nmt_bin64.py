"""64-bit NaMaster MCM binning: match C at small nside, trip i64 at 4096 spin 2."""

import numpy as np
import pytest

from gmaster._nmt_bin64 import bin_mcm_i64, mcm_needs_i64, patch_pymaster


def test_mcm_needs_i64_only_past_signed_32bit():
    # nside 2048 spin 2: ncls=4, nls=6144, 16*6144^2 = 6.04e8 < INT_MAX
    assert not mcm_needs_i64(3 * 2048, 4)
    # nside 4096 spin 0 / 0x2: ncls=1 or 2 still fit
    assert not mcm_needs_i64(3 * 4096, 1)
    assert not mcm_needs_i64(3 * 4096, 2)
    # nside 4096 spin 2: ncls=4, nls=12288, 16*12288^2 = 2.416e9 > INT_MAX
    assert mcm_needs_i64(3 * 4096, 4)
    assert mcm_needs_i64(3 * 4096, 7)


@pytest.mark.parametrize("ncls", [1, 2, 4])
@pytest.mark.parametrize("oneside", [False, True])
@pytest.mark.parametrize("is_dell", [False, True])
def test_bin_mcm_i64_matches_namaster_c(ncls, oneside, is_dell):
    reference = pytest.importorskip("pymaster")
    bins = reference.NmtBin.from_lmax_linear(64, 7, is_Dell=is_dell)
    rng = np.random.default_rng(3)
    nls = bins.lmax + 1
    mcm = rng.normal(size=(nls * ncls, nls * ncls))
    beam1 = rng.normal(size=nls)
    beam2 = rng.normal(size=nls)
    got = bin_mcm_i64(mcm, bins, ncls, beam1, beam2, oneside)
    expected = bins._bin_mcm(mcm, 0, 0.0, beam1, beam2, oneside)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_bin_mcm_i64_fkp_matches_namaster_c():
    reference = pytest.importorskip("pymaster")
    bins = reference.NmtBin.from_lmax_linear(48, 5)
    rng = np.random.default_rng(4)
    ncls = 4
    nls = bins.lmax + 1
    mcm = rng.normal(size=(nls * ncls, nls * ncls))
    beam1 = rng.normal(size=nls)
    beam2 = rng.normal(size=nls)
    got = bin_mcm_i64(mcm, bins, ncls, beam1, beam2, False, norm_type=1, w2=2.25)
    expected = bins._bin_mcm(mcm, 1, 2.25, beam1, beam2, False)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_forced_i64_workspace_matches_c(monkeypatch):
    """End-to-end: force the numpy path at a size C can still bin."""
    reference = pytest.importorskip("pymaster")
    import healpy as hp
    import gmaster._nmt_bin64 as b64

    nside = 16
    npix = 12 * nside**2
    rng = np.random.default_rng(8)
    theta = hp.pix2ang(nside, np.arange(npix))[0]
    mask = np.clip((np.cos(theta) + 0.35) / 0.7, 0, 1) ** 2
    q = rng.normal(size=npix)
    u = rng.normal(size=npix)
    bins = reference.NmtBin.from_lmax_linear(3 * nside - 1, 4)
    f = reference.NmtField(mask, [q, u], n_iter=0, spin=2)
    w_c = reference.NmtWorkspace()
    w_c.compute_coupling_matrix(f, f, bins)
    cl_c = w_c.decouple_cell(reference.compute_coupled_cell(f, f))

    monkeypatch.setattr(b64, "mcm_needs_i64", lambda nls, ncls: True)
    patch_pymaster()
    w_i = reference.NmtWorkspace()
    w_i.compute_coupling_matrix(f, f, bins)
    cl_i = w_i.decouple_cell(reference.compute_coupled_cell(f, f))
    np.testing.assert_allclose(cl_i, cl_c, rtol=1e-10, atol=1e-12)
