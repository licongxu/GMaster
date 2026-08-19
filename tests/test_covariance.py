import os
import tempfile

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import gmaster as nmt


def _nmaps(spin):
    return 1 if spin == 0 else 2


@pytest.mark.parametrize(
    "spins", [(0, 0, 0, 0), (0, 2, 0, 2), (0, 0, 2, 2), (2, 2, 2, 2)]
)
def test_curved_covariance_matches_namaster(spins):
    reference = pytest.importorskip("pymaster")
    rng = np.random.default_rng(70)
    lmax = 7
    mask = rng.uniform(0.2, 1, 12 * 4**2)
    got_fields = [
        nmt.NmtField(
            mask, None, spin=spin, lmax=lmax, lmax_mask=lmax,
            n_iter=0, n_iter_mask=0,
        )
        for spin in spins
    ]
    ref_fields = [
        reference.NmtField(
            mask, None, spin=spin, lmax=lmax, lmax_mask=lmax,
            n_iter=0, n_iter_mask=0,
        )
        for spin in spins
    ]
    got_cw = nmt.NmtCovarianceWorkspace.from_fields(*got_fields)
    ref_cw = reference.NmtCovarianceWorkspace.from_fields(*ref_fields)
    got_bin = nmt.NmtBin.from_lmax_linear(lmax, 2)
    ref_bin = reference.NmtBin.from_lmax_linear(lmax, 2)
    got_wa = nmt.NmtWorkspace.from_fields(got_fields[0], got_fields[1], got_bin)
    got_wb = nmt.NmtWorkspace.from_fields(got_fields[2], got_fields[3], got_bin)
    ref_wa = reference.NmtWorkspace.from_fields(ref_fields[0], ref_fields[1], ref_bin)
    ref_wb = reference.NmtWorkspace.from_fields(ref_fields[2], ref_fields[3], ref_bin)
    base = 1 / (10 + np.arange(lmax + 1))
    cls = [
        np.tile(base, (_nmaps(spins[a]) * _nmaps(spins[b]), 1))
        for a, b in ((0, 2), (0, 3), (1, 2), (1, 3))
    ]

    for coupled in (True, False):
        got = got_cw.gaussian_covariance(
            *cls, got_wa, wb=got_wb, coupled=coupled
        )
        expected = ref_cw.gaussian_covariance(
            *cls, ref_wa, wb=ref_wb, coupled=coupled
        )
        np.testing.assert_allclose(got, expected, rtol=2e-12, atol=2e-14)


@pytest.mark.parametrize(
    "spins", [(0, 0, 0, 0), (0, 2, 0, 2), (0, 0, 2, 2), (2, 2, 2, 2)]
)
def test_flat_covariance_matches_namaster(spins):
    reference = pytest.importorskip("pymaster")
    rng = np.random.default_rng(71)
    ny, nx = 8, 10
    lx, ly = np.radians(8), np.radians(6)
    mask = rng.uniform(0.2, 1, (ny, nx))
    got_fields = [
        nmt.NmtFieldFlat(lx, ly, mask, np.zeros((_nmaps(spin), ny, nx)), spin=spin)
        for spin in spins
    ]
    ref_fields = [
        reference.NmtFieldFlat(
            lx, ly, mask, np.zeros((_nmaps(spin), ny, nx)), spin=spin
        )
        for spin in spins
    ]
    got_bin = nmt.NmtBinFlat([20, 80], [80, 160])
    ref_bin = reference.NmtBinFlat([20, 80], [80, 160])
    got_cw = nmt.NmtCovarianceWorkspaceFlat.from_fields(
        got_fields[0], got_fields[1], got_bin, got_fields[2], got_fields[3], got_bin
    )
    ref_cw = reference.NmtCovarianceWorkspaceFlat.from_fields(
        ref_fields[0], ref_fields[1], ref_bin, ref_fields[2], ref_fields[3], ref_bin
    )
    got_wa = nmt.NmtWorkspaceFlat.from_fields(got_fields[0], got_fields[1], got_bin)
    got_wb = nmt.NmtWorkspaceFlat.from_fields(got_fields[2], got_fields[3], got_bin)
    ref_wa = reference.NmtWorkspaceFlat.from_fields(ref_fields[0], ref_fields[1], ref_bin)
    ref_wb = reference.NmtWorkspaceFlat.from_fields(ref_fields[2], ref_fields[3], ref_bin)
    ells = np.linspace(0, 300, 100)
    base = 1 / (10 + ells)
    cls = [
        np.tile(base, (_nmaps(spins[a]) * _nmaps(spins[b]), 1))
        for a, b in ((0, 2), (0, 3), (1, 2), (1, 3))
    ]

    got = got_cw.gaussian_covariance(*spins, ells, *cls, got_wa, wb=got_wb)
    expected = ref_cw.gaussian_covariance(*spins, ells, *cls, ref_wa, wb=ref_wb)
    np.testing.assert_allclose(got, expected, rtol=3e-12, atol=2e-18)


def test_catalog_covariance_and_inka_match_namaster():
    reference = pytest.importorskip("pymaster")
    rng = np.random.default_rng(72)
    lmax = 6
    size = 200
    positions = np.stack(
        [np.arccos(rng.uniform(-1, 1, size)), rng.uniform(0, 2 * np.pi, size)]
    )
    weights = rng.uniform(0.5, 1.5, size)
    values = rng.normal(size=size)
    got_field = nmt.NmtFieldCatalog(
        positions, weights, values, lmax=lmax, retain_catalog=True
    )
    ref_field = reference.NmtFieldCatalog(
        positions, weights, values, lmax=lmax, retain_catalog=True
    )
    got_bin = nmt.NmtBin.from_lmax_linear(lmax, 2)
    ref_bin = reference.NmtBin.from_lmax_linear(lmax, 2)
    got_w = nmt.NmtWorkspace.from_fields(got_field, got_field, got_bin)
    ref_w = reference.NmtWorkspace.from_fields(ref_field, ref_field, ref_bin)
    got_cw = nmt.NmtCovarianceWorkspace.from_fields(got_field, got_field)
    ref_cw = reference.NmtCovarianceWorkspace.from_fields(ref_field, ref_field)
    guess = np.atleast_2d(1 / (10 + np.arange(lmax + 1)))

    got_inka = nmt.get_iNKA_cell(got_field, got_field, guess, got_w)
    ref_inka = reference.get_iNKA_cell(ref_field, ref_field, guess, ref_w)
    np.testing.assert_allclose(got_inka, ref_inka, rtol=2e-5, atol=2e-7)
    got = got_cw.gaussian_covariance(got_inka, got_inka, got_inka, got_inka, got_w)
    expected = ref_cw.gaussian_covariance(ref_inka, ref_inka, ref_inka, ref_inka, ref_w)
    np.testing.assert_allclose(got, expected, rtol=3e-5, atol=2e-10)


def test_covariance_io_and_deprecated_wrappers():
    reference = pytest.importorskip("pymaster")
    rng = np.random.default_rng(73)
    lmax = 7
    mask = rng.uniform(0.2, 1, 12 * 4**2)
    field = nmt.NmtField(
        mask, None, spin=0, lmax=lmax, lmax_mask=lmax,
        n_iter=0, n_iter_mask=0,
    )
    cw = nmt.NmtCovarianceWorkspace.from_fields(field, field)
    bins = nmt.NmtBin.from_lmax_linear(lmax, 2)
    workspace = nmt.NmtWorkspace.from_fields(field, field, bins)
    cell = np.atleast_2d(1 / (10 + np.arange(lmax + 1)))

    handle, path = tempfile.mkstemp(suffix=".fits")
    os.close(handle)
    os.unlink(path)
    try:
        cw.write_to(path)
        loaded = nmt.NmtCovarianceWorkspace.from_file(path)
        reference.NmtCovarianceWorkspace.from_file(path)
        expected = cw.gaussian_covariance(cell, cell, cell, cell, workspace)
        np.testing.assert_allclose(
            loaded.gaussian_covariance(cell, cell, cell, cell, workspace), expected
        )
        np.testing.assert_allclose(
            nmt.gaussian_covariance(
                loaded, 0, 0, 0, 0, cell, cell, cell, cell, workspace
            ),
            expected,
        )
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_covariance_validation():
    mask = np.ones(12 * 4**2)
    field = nmt.NmtField(mask, None, spin=0, lmax=7, lmax_mask=7)
    spin3 = nmt.NmtField(mask, None, spin=3, lmax=7, lmax_mask=7)
    with pytest.raises(ValueError, match="spin-0 and spin-2"):
        nmt.NmtCovarianceWorkspace.from_fields(field, spin3)
    with pytest.raises(ValueError, match="positive"):
        nmt.NmtCovarianceWorkspace.from_fields(field, field, l_toeplitz=4)


def test_toeplitz_covariance_matches_namaster():
    reference = pytest.importorskip("pymaster")
    rng = np.random.default_rng(74)
    lmax = 12
    mask = rng.uniform(0.2, 1, 12 * 8**2)
    options = dict(
        spin=2, lmax=lmax, lmax_mask=lmax, n_iter=0, n_iter_mask=0
    )
    got_field = nmt.NmtField(mask, None, **options)
    ref_field = reference.NmtField(mask, None, **options)
    toeplitz = dict(l_toeplitz=6, l_exact=3, dl_band=2)
    got = nmt.NmtCovarianceWorkspace.from_fields(
        got_field, got_field, **toeplitz
    )
    expected = reference.NmtCovarianceWorkspace.from_fields(
        ref_field, ref_field, **toeplitz
    )
    for wick in range(2):
        for name in ("pp", "mm"):
            np.testing.assert_allclose(
                got.xiSS[wick][name], expected.xiSS[wick][name], atol=3e-14
            )
