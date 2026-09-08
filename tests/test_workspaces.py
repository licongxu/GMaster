import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import gmaster as nmt
from gmaster.workspaces import (
    _apply_toeplitz,
    _binning_operators,
    _coupling_matrix_tt,
    _coupling_matrix_tt_toeplitz,
    _expanded_binning_operators,
)


def test_public_coupling_helpers_match_namaster():
    reference = pytest.importorskip("pymaster")
    rng = np.random.default_rng(12)
    mask_cell = rng.uniform(size=8)
    for parameters in ((0, 0, 0, 0), (0, 2, 0, 2), (2, 2, 2, 2), (3, 2, -1, 1)):
        for parity in ("all", "even", "odd", "both"):
            np.testing.assert_allclose(
                nmt.get_general_coupling_matrix(
                    mask_cell, *parameters, parity=parity
                ),
                reference.get_general_coupling_matrix(
                    mask_cell, *parameters, parity=parity
                ),
                atol=8e-14,
            )
    with pytest.raises(ValueError, match="parity"):
        nmt.get_general_coupling_matrix(mask_cell, 0, 0, 0, 0, parity="bad")


@pytest.mark.parametrize(
    "spin1,spin2,is_teb,pure_any",
    [(0, 0, False, False), (0, 2, False, False), (2, 2, False, False),
     (0, 2, True, False), (0, 2, True, True), (1, 3, False, False)],
)
def test_public_master_coefficients_match_namaster(
    spin1, spin2, is_teb, pure_any
):
    reference = pytest.importorskip("pymaster")
    rng = np.random.default_rng(11)
    mask_cells = rng.uniform(size=(2, 8))
    got = nmt.get_master_coefficients(
        mask_cells, 7, spin1, spin2, is_teb=is_teb, pure_any=pure_any
    )
    expected = reference.get_master_coefficients(
        mask_cells, 7, spin1, spin2, is_teb=is_teb, pure_any=pure_any
    )
    for name in ("00", "0s", "pp", "mm"):
        if expected[name] is None:
            assert got[name] is None
        else:
            np.testing.assert_allclose(got[name], expected[name], atol=2e-13)
    assert got["spins"] == expected["spins"]
    assert got["lmax"] == expected["lmax"]


def test_high_l_spin2_coefficients_match_namaster():
    reference = pytest.importorskip("pymaster")
    lmax = 255
    mask_cell = np.exp(-np.arange(2 * lmax + 1) / 40)
    got = nmt.get_master_coefficients(
        mask_cell, lmax, 0, 2, pure_any=True
    )["0s"]
    expected = reference.get_master_coefficients(
        mask_cell, lmax, 0, 2, pure_any=True
    )["0s"]
    np.testing.assert_allclose(got, expected, rtol=2e-9, atol=2e-9)


@pytest.mark.parametrize(
    "spin1,spin2,is_teb", [(0, 0, False), (0, 2, False), (2, 2, False),
                            (0, 2, True), (1, 3, False)]
)
def test_toeplitz_master_coefficients_match_namaster(spin1, spin2, is_teb):
    reference = pytest.importorskip("pymaster")
    rng = np.random.default_rng(18)
    mask_cell = rng.uniform(size=13)
    options = dict(l_toeplitz=6, l_exact=3, dl_band=2, is_teb=is_teb)
    got = nmt.get_master_coefficients(mask_cell, 12, spin1, spin2, **options)
    expected = reference.get_master_coefficients(
        mask_cell, 12, spin1, spin2, **options
    )
    for name in ("00", "0s", "pp", "mm"):
        if expected[name] is not None:
            np.testing.assert_allclose(got[name], expected[name], atol=2e-13)


def test_selective_scalar_toeplitz_matches_full_kernel():
    lmax = 31
    options = dict(l_toeplitz=16, l_exact=4, dl_band=3)
    window = np.random.default_rng(19).uniform(size=2 * lmax + 1)
    columns = 2 * np.arange(lmax + 1) + 1
    expected = _apply_toeplitz(
        _coupling_matrix_tt(window, lmax=lmax) / columns[None],
        **options,
    ) * columns[None]
    got = _coupling_matrix_tt_toeplitz(window, lmax=lmax, **options)
    np.testing.assert_allclose(got, expected, atol=2e-14)


def test_uncorrelated_noise_deprojection_bias_matches_namaster():
    reference = pytest.importorskip("pymaster")
    rng = np.random.default_rng(10)
    nside = 4
    npix = 12 * nside**2
    lmax = 7
    mask = rng.uniform(0.2, 1, npix)
    maps = rng.normal(size=(2, npix))
    templates = rng.normal(size=(2, 2, npix))
    variance = rng.uniform(0.1, 1, npix)
    options = dict(
        spin=2, templates=templates, lmax=lmax, lmax_mask=lmax, n_iter=0
    )
    got_field = nmt.NmtField(mask, maps, **options)
    ref_field = reference.NmtField(mask, maps, **options)
    np.testing.assert_allclose(
        nmt.uncorr_noise_deprojection_bias(got_field, variance, n_iter=0),
        reference.uncorr_noise_deprojection_bias(ref_field, variance, n_iter=0),
        rtol=2e-12,
        atol=2e-14,
    )


@pytest.mark.parametrize("normalization", ["MASTER", "FKP"])
def test_scalar_workspace_matches_namaster(normalization):
    reference = pytest.importorskip("pymaster")
    rng = np.random.default_rng(13)
    nside = 4
    lmax = 7
    npix = 12 * nside**2
    mask = rng.uniform(0.2, 1, npix)
    maps = rng.normal(size=(1, npix))
    beam = np.exp(-np.arange(lmax + 1) ** 2 / 200)
    got_field = nmt.NmtField(
        mask, maps, beam=beam, lmax=lmax, lmax_mask=lmax
    )
    ref_field = reference.NmtField(
        mask, maps, beam=beam, lmax=lmax, lmax_mask=lmax
    )
    got_bins = nmt.NmtBin.from_lmax_linear(lmax, 2)
    ref_bins = reference.NmtBin.from_lmax_linear(lmax, 2)
    got = nmt.NmtWorkspace.from_fields(
        got_field, got_field, got_bins, normalization=normalization
    )
    ref = reference.NmtWorkspace.from_fields(
        ref_field, ref_field, ref_bins, normalization=normalization
    )

    np.testing.assert_allclose(
        got.get_coupling_matrix(), ref.get_coupling_matrix(), atol=2e-14
    )
    np.testing.assert_allclose(got.mcm_binned, ref.mcm_binned, atol=2e-14)
    theory = rng.uniform(size=(1, lmax + 1))
    np.testing.assert_allclose(
        got.couple_cell(theory), ref.couple_cell(theory), atol=2e-14
    )
    np.testing.assert_allclose(
        got.decouple_cell(got.couple_cell(theory)),
        ref.decouple_cell(ref.couple_cell(theory)),
        atol=2e-13,
    )
    np.testing.assert_allclose(
        got.get_bandpower_windows(), ref.get_bandpower_windows(), atol=2e-13
    )


def test_scalar_workspace_updates():
    npix = 12 * 4**2
    lmax = 7
    field = nmt.NmtField(
        np.ones(npix), np.ones((1, npix)), lmax=lmax, lmax_mask=lmax
    )
    bins = nmt.NmtBin.from_lmax_linear(lmax, 2)
    workspace = nmt.NmtWorkspace.from_fields(field, field, bins)
    matrix = workspace.get_coupling_matrix()
    workspace.update_beams(np.ones(lmax + 1), np.ones(lmax + 1))
    workspace.update_bins(nmt.NmtBin.from_edges([2, 5], [5, 8]))
    workspace.update_coupling_matrix(matrix)

    with pytest.raises(ValueError, match="inconsistent shape"):
        workspace.update_coupling_matrix(np.eye(lmax))
    with pytest.raises(ValueError, match="wrong shape"):
        workspace.couple_cell(np.ones((2, lmax + 1)))


def test_binning_caches_are_keyed_on_band_content():
    lmax = 7
    wide = nmt.NmtBin.from_lmax_linear(lmax, 2)
    equal = nmt.NmtBin.from_lmax_linear(lmax, 2)
    narrow = nmt.NmtBin.from_edges([2, 5], [5, 8])

    operators = _binning_operators(wide)
    assert _binning_operators(equal) is operators
    assert _binning_operators(narrow) is not operators
    assert _expanded_binning_operators(wide, 1) is _expanded_binning_operators(wide, 1)
    assert _expanded_binning_operators(wide, 3) is not _expanded_binning_operators(
        wide, 1
    )

    output, theory = operators
    expected = np.zeros((wide.n_bands, lmax + 1))
    expected[wide._bpws_np, wide._ells_np] = wide._weights_np * wide._f_ell_np
    np.testing.assert_array_equal(np.asarray(output), expected)
    expected_t = np.zeros((lmax + 1, wide.n_bands))
    expected_t[wide._ells_np, wide._bpws_np] = 1.0 / wide._f_ell_np
    np.testing.assert_array_equal(np.asarray(theory), expected_t)

    identity = np.eye(3)
    expanded = _expanded_binning_operators(wide, 3)
    np.testing.assert_array_equal(np.asarray(expanded[0]), np.kron(expected, identity))
    np.testing.assert_array_equal(
        np.asarray(expanded[1]), np.kron(expected_t, identity)
    )


def test_bandpower_operators_are_rebuilt_when_bins_change():
    npix = 12 * 4**2
    lmax = 7
    field = nmt.NmtField(
        np.ones(npix), np.ones((1, npix)), lmax=lmax, lmax_mask=lmax
    )
    workspace = nmt.NmtWorkspace.from_fields(
        field, field, nmt.NmtBin.from_lmax_linear(lmax, 2)
    )
    before = np.asarray(workspace.get_bandpower_windows())

    new_bins = nmt.NmtBin.from_lmax_linear(lmax, 3)
    workspace.update_bins(new_bins)
    after = np.asarray(workspace.get_bandpower_windows())
    assert after.shape[1] != before.shape[1]

    fresh = nmt.NmtWorkspace.from_fields(field, field, new_bins)
    np.testing.assert_allclose(
        after, np.asarray(fresh.get_bandpower_windows()), rtol=1e-12, atol=1e-14
    )


@pytest.mark.parametrize(
    "spins,is_teb", [((0, 1), False), ((1, 0), False), ((1, 3), False),
                      ((3, 3), False), ((0, 3), True)]
)
def test_arbitrary_spin_workspaces_match_namaster(spins, is_teb):
    reference = pytest.importorskip("pymaster")
    rng = np.random.default_rng(17)
    lmax = 7
    npix = 12 * 4**2
    masks = [rng.uniform(0.2, 1, npix), rng.uniform(0.2, 1, npix)]
    got_fields = []
    ref_fields = []
    for mask, spin in zip(masks, spins):
        maps = rng.normal(size=(1 if spin == 0 else 2, npix))
        options = dict(
            spin=spin, lmax=lmax, lmax_mask=lmax, n_iter=0, n_iter_mask=0
        )
        got_fields.append(nmt.NmtField(mask, maps, **options))
        ref_fields.append(reference.NmtField(mask, maps, **options))
    got = nmt.NmtWorkspace.from_fields(
        *got_fields, nmt.NmtBin.from_lmax_linear(lmax, 2), is_teb=is_teb
    )
    expected = reference.NmtWorkspace.from_fields(
        *ref_fields, reference.NmtBin.from_lmax_linear(lmax, 2), is_teb=is_teb
    )
    np.testing.assert_allclose(got.mcm, expected.mcm, atol=4e-14)
    np.testing.assert_allclose(got.mcm_binned, expected.mcm_binned, atol=4e-14)


@pytest.mark.parametrize(
    "spins,is_teb", [((0, 2), False), ((2, 0), False), ((2, 2), False), ((0, 2), True)]
)
def test_spin2_workspaces_match_namaster(spins, is_teb):
    reference = pytest.importorskip("pymaster")
    rng = np.random.default_rng(14)
    nside = 4
    lmax = 7
    npix = 12 * nside**2
    masks = [rng.uniform(0.2, 1, npix), rng.uniform(0.3, 1, npix)]
    maps = [rng.normal(size=(1, npix)), rng.normal(size=(2, npix))]
    got_fields = {
        spin: nmt.NmtField(
            masks[spin // 2], maps[spin // 2], lmax=lmax, lmax_mask=lmax
        )
        for spin in (0, 2)
    }
    ref_fields = {
        spin: reference.NmtField(
            masks[spin // 2], maps[spin // 2], lmax=lmax, lmax_mask=lmax
        )
        for spin in (0, 2)
    }
    got_bins = nmt.NmtBin.from_lmax_linear(lmax, 2)
    ref_bins = reference.NmtBin.from_lmax_linear(lmax, 2)
    got = nmt.NmtWorkspace.from_fields(
        got_fields[spins[0]], got_fields[spins[1]], got_bins, is_teb=is_teb
    )
    ref = reference.NmtWorkspace.from_fields(
        ref_fields[spins[0]], ref_fields[spins[1]], ref_bins, is_teb=is_teb
    )

    np.testing.assert_allclose(got.mcm, ref.mcm, atol=2e-14)
    np.testing.assert_allclose(got.mcm_binned, ref.mcm_binned, atol=2e-14)
    np.testing.assert_allclose(
        got.get_bandpower_windows(), ref.get_bandpower_windows(), atol=2e-13
    )
    theory = rng.normal(size=(got.ncls, lmax + 1))
    np.testing.assert_allclose(
        got.couple_cell(theory), ref.couple_cell(theory), atol=2e-14
    )
    np.testing.assert_allclose(
        got.decouple_cell(got.couple_cell(theory)),
        ref.decouple_cell(ref.couple_cell(theory)),
        atol=3e-13,
    )


@pytest.mark.parametrize(
    "pure1,pure2", [((True, False), (False, True)), ((True, True), (True, True))]
)
def test_pure_spin2_workspaces_match_namaster(pure1, pure2):
    reference = pytest.importorskip("pymaster")
    healpy = pytest.importorskip("healpy")
    rng = np.random.default_rng(15)
    nside = 4
    lmax = 7
    npix = 12 * nside**2
    theta, _ = healpy.pix2ang(nside, np.arange(npix))
    mask = np.sin(theta) ** 4
    maps = rng.normal(size=(2, npix))
    options1 = dict(
        purify_e=pure1[0],
        purify_b=pure1[1],
        lmax=lmax,
        lmax_mask=lmax,
    )
    options2 = dict(
        purify_e=pure2[0],
        purify_b=pure2[1],
        lmax=lmax,
        lmax_mask=lmax,
    )
    got1 = nmt.NmtField(mask, maps, **options1)
    got2 = nmt.NmtField(mask, maps, **options2)
    ref1 = reference.NmtField(mask, maps, **options1)
    ref2 = reference.NmtField(mask, maps, **options2)
    got = nmt.NmtWorkspace.from_fields(
        got1, got2, nmt.NmtBin.from_lmax_linear(lmax, 2)
    )
    ref = reference.NmtWorkspace.from_fields(
        ref1, ref2, reference.NmtBin.from_lmax_linear(lmax, 2)
    )

    np.testing.assert_allclose(got.mcm, ref.mcm, atol=3e-13)
    np.testing.assert_allclose(got.mcm_binned, ref.mcm_binned, atol=3e-13)
    np.testing.assert_allclose(
        got.get_bandpower_windows(), ref.get_bandpower_windows(), atol=5e-12
    )


@pytest.mark.parametrize("spins", [(0, 2), (2, 0), (2, 2)])
def test_anisotropic_workspaces_match_namaster(spins):
    reference = pytest.importorskip("pymaster")
    rng = np.random.default_rng(16)
    nside = 4
    lmax = 7
    npix = 12 * nside**2
    mask11 = rng.uniform(0.5, 1, npix)
    mask22 = rng.uniform(0.5, 1, npix)
    mask12 = 0.1 * np.sqrt(mask11 * mask22)
    scalar_mask = rng.uniform(0.2, 1, npix)
    scalar_map = rng.normal(size=(1, npix))
    spin_map = rng.normal(size=(2, npix))
    got_fields = {
        0: nmt.NmtField(
            scalar_mask, scalar_map, lmax=lmax, lmax_mask=lmax
        ),
        2: nmt.NmtField(
            mask11,
            spin_map,
            mask_22=mask22,
            mask_12=mask12,
            lmax=lmax,
            lmax_mask=lmax,
        ),
    }
    ref_fields = {
        0: reference.NmtField(
            scalar_mask, scalar_map, lmax=lmax, lmax_mask=lmax
        ),
        2: reference.NmtField(
            mask11,
            spin_map,
            mask_22=mask22,
            mask_12=mask12,
            lmax=lmax,
            lmax_mask=lmax,
        ),
    }
    got = nmt.NmtWorkspace.from_fields(
        got_fields[spins[0]],
        got_fields[spins[1]],
        nmt.NmtBin.from_lmax_linear(lmax, 2),
    )
    ref = reference.NmtWorkspace.from_fields(
        ref_fields[spins[0]],
        ref_fields[spins[1]],
        reference.NmtBin.from_lmax_linear(lmax, 2),
    )

    np.testing.assert_allclose(got.mcm, ref.mcm, atol=3e-13)
    np.testing.assert_allclose(got.mcm_binned, ref.mcm_binned, atol=3e-13)
    np.testing.assert_allclose(
        got.get_bandpower_windows(), ref.get_bandpower_windows(), atol=5e-12
    )


def test_deprojection_bias_and_full_master_match_namaster():
    reference = pytest.importorskip("pymaster")
    rng = np.random.default_rng(17)
    nside = 4
    lmax = 7
    npix = 12 * nside**2
    mask = rng.uniform(0.3, 1, npix)
    template1 = rng.normal(size=(1, 1, npix))
    template2 = rng.normal(size=(1, 1, npix))
    map1 = rng.normal(size=(1, npix))
    map2 = rng.normal(size=(1, npix))
    got1 = nmt.NmtField(
        mask, map1, templates=template1, lmax=lmax, lmax_mask=lmax
    )
    got2 = nmt.NmtField(
        mask, map2, templates=template2, lmax=lmax, lmax_mask=lmax
    )
    ref1 = reference.NmtField(
        mask, map1, templates=template1, lmax=lmax, lmax_mask=lmax
    )
    ref2 = reference.NmtField(
        mask, map2, templates=template2, lmax=lmax, lmax_mask=lmax
    )
    guess = rng.uniform(size=(1, lmax + 1))
    np.testing.assert_allclose(
        nmt.deprojection_bias(got1, got2, guess),
        reference.deprojection_bias(ref1, ref2, guess),
        atol=2e-13,
    )
    np.testing.assert_allclose(
        nmt.compute_full_master(
            got1, got2, nmt.NmtBin.from_lmax_linear(lmax, 2), cl_guess=guess
        ),
        reference.compute_full_master(
            ref1,
            ref2,
            reference.NmtBin.from_lmax_linear(lmax, 2),
            cl_guess=guess,
        ),
        atol=3e-13,
    )


def test_pure_spin2_deprojection_bias_matches_namaster():
    reference = pytest.importorskip("pymaster")
    healpy = pytest.importorskip("healpy")
    rng = np.random.default_rng(18)
    nside = 4
    lmax = 7
    npix = 12 * nside**2
    theta, _ = healpy.pix2ang(nside, np.arange(npix))
    mask = np.sin(theta) ** 4
    template = rng.normal(size=(1, 2, npix))
    maps = rng.normal(size=(2, npix))
    options = dict(
        templates=template,
        purify_b=True,
        lmax=lmax,
        lmax_mask=lmax,
    )
    got = nmt.NmtField(mask, maps, **options)
    ref = reference.NmtField(mask, maps, **options)
    guess = rng.uniform(size=(4, lmax + 1))
    np.testing.assert_allclose(
        nmt.deprojection_bias(got, got, guess),
        reference.deprojection_bias(ref, ref, guess),
        atol=3e-13,
    )


def test_workspace_fits_interoperability(tmp_path):
    reference = pytest.importorskip("pymaster")
    npix = 12 * 4**2
    lmax = 7
    maps = np.ones((2, npix))
    got_field = nmt.NmtField(
        np.ones(npix), maps, lmax=lmax, lmax_mask=lmax
    )
    ref_field = reference.NmtField(
        np.ones(npix), maps, lmax=lmax, lmax_mask=lmax
    )
    got = nmt.NmtWorkspace.from_fields(
        got_field, got_field, nmt.NmtBin.from_lmax_linear(lmax, 2)
    )
    ref = reference.NmtWorkspace.from_fields(
        ref_field, ref_field, reference.NmtBin.from_lmax_linear(lmax, 2)
    )

    got_path = tmp_path / "gmaster.fits"
    got.write_to(got_path)
    np.testing.assert_allclose(
        nmt.NmtWorkspace.from_file(got_path).mcm, got.mcm, atol=0
    )
    np.testing.assert_allclose(
        reference.NmtWorkspace.from_file(got_path).mcm, got.mcm, atol=0
    )

    ref_path = tmp_path / "namaster.fits"
    ref.write_to(ref_path)
    loaded = nmt.NmtWorkspace.from_file(ref_path)
    np.testing.assert_allclose(loaded.mcm, ref.mcm, atol=0)
    np.testing.assert_allclose(loaded.bpws, ref.bpws, atol=2e-14)
