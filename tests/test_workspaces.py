import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import gmaster as nmt
from gmaster import workspaces as ws
from gmaster.workspaces import (
    _apply_toeplitz,
    _binning_operators,
    _coupling_matrix_tt,
    _coupling_matrix_tt_toeplitz,
    _expanded_binning_operators,
    _left_contract,
    _RowPieces,
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


def _shrinking_offset_tt(window_cls, lmax):
    """Per-offset threej recurrence the chunked kernel must match."""
    n_ell = lmax + 1
    p = np.arange(1, 2 * lmax + 1, dtype=np.float64)
    log_g = np.concatenate([[0.0], np.cumsum(np.log((p - 0.5) / p))])
    g = np.exp(log_g)
    mask_power = window_cls * (2 * np.arange(2 * lmax + 1) + 1) / (4 * np.pi)
    row = np.arange(n_ell)[:, None]
    column = np.arange(n_ell)[None, :]
    lower = np.minimum(row, column)
    upper = np.maximum(row, column)
    matrix = np.zeros((n_ell, n_ell), dtype=np.float64)
    for offset in range(n_ell):
        p_total = upper + offset
        term = (
            mask_power[np.minimum(upper - lower + 2 * offset, 2 * lmax)]
            * g[upper - lower + offset]
            * g[offset]
            * g[np.maximum(lower - offset, 0)]
            / (g[p_total] * (2 * p_total + 1))
        )
        matrix += np.where(offset <= lower, term, 0)
    return matrix * (2 * column + 1)


@pytest.mark.parametrize("lmax", [15, 16, 17, 31])
def test_scalar_coupling_matches_per_offset_recurrence(lmax):
    window = np.random.default_rng(20 + lmax).uniform(size=2 * lmax + 1)
    nmt.set_coupling_precision("fp64")
    jax.clear_caches()
    try:
        got = np.asarray(_coupling_matrix_tt(window, lmax=lmax))
        np.testing.assert_allclose(got, _shrinking_offset_tt(window, lmax), atol=2e-14)
    finally:
        nmt.set_coupling_precision("auto")
        jax.clear_caches()


def test_scalar_coupling_jaxpr_does_not_grow_with_lmax():
    """A Python chunk loop unrolls with n_ell; the fori_loop HLO must not."""

    def lowered(lmax):
        window = jax.numpy.ones(2 * lmax + 1)
        return _coupling_matrix_tt.lower(window, lmax=lmax).as_text()

    small = lowered(47)
    large = lowered(95)
    assert ("while" in large) or ("scan" in large.lower())
    assert len(large) < 1.4 * len(small), (len(small), len(large))


def test_scalar_quadrature_tt_matches_recurrence():
    lmax = 31
    window = np.random.default_rng(22).uniform(size=2 * lmax + 1)
    nmt.set_coupling_precision("fp64")
    jax.clear_caches()
    try:
        rec = np.asarray(_coupling_matrix_tt(window, lmax=lmax))
        quad = np.asarray(
            ws._general_coupling_matrix(
                window, s1=0, s2=0, n1=0, n2=0, lmax=lmax, lmax_mask=2 * lmax
            )[0]
        )
        np.testing.assert_allclose(quad, rec, atol=1e-11, rtol=1e-11)
    finally:
        nmt.set_coupling_precision("auto")
        jax.clear_caches()


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


def test_spin2_pipeline_survives_releasing_the_first_field():
    """The Nside-4096 spin-2 bench must drop the compile-time field before the warm run.

    Holding the first field+workspace while building a second one is what OOM'd the
    overlapping stage timings (8 GiB chirp-Z with 70.9/71.2 GiB in use).  Two
    sequential pipelines after deleting the first must both finish finite.
    """
    rng = np.random.default_rng(21)
    nside, lmax = 8, 16
    npix = 12 * nside ** 2
    mask = rng.uniform(0.3, 1, npix)
    maps = rng.normal(size=(2, npix))
    bins = nmt.NmtBin.from_lmax_linear(lmax, 4)

    def once():
        field = nmt.NmtField(mask, maps, lmax=lmax, lmax_mask=lmax, n_iter=1, spin=2)
        w = nmt.NmtWorkspace()
        w.compute_coupling_matrix(field, field, bins)
        dec = np.asarray(w.decouple_cell(nmt.compute_coupled_cell(field, field)))
        return field, w, dec

    f1, w1, d1 = once()
    assert np.isfinite(d1).all()
    del f1, w1
    f2, w2, d2 = once()
    assert np.isfinite(d2).all()
    np.testing.assert_allclose(d2, d1, atol=1e-10)


def test_nside4096_spin2_matrix_exceeds_the_pieced_threshold():
    """The Nside-4096 spin-2 operator is past `_MCM_PIECED_BYTES`; small cells are not.

    `ncls=4`, `lmax=12287` is `5.4e9` elements, past `2**31` (the NaMaster segfault)
    and past the 4 GiB pieced-assembly gate.  A nside-4 auto-spectrum is not.
    """
    ncls, lmax = 4, 3 * 4096 - 1
    n = ncls * (lmax + 1)
    assert n * n * 8 > ws._MCM_PIECED_BYTES
    n_small = 4 * (7 + 1)
    assert n_small * n_small * 8 <= ws._MCM_PIECED_BYTES


def test_pieced_coupling_matrix_matches_dense_and_decouples(monkeypatch):
    """Forcing the Nside-4096 `_RowPieces` path at nside=4 keeps NaMaster agreement.

    `_assemble_mcm` is Python-gated on `_MCM_PIECED_BYTES`.  Zeroing the gate takes
    the shipped piece assembly, `couple_cell`, `decouple_cell` and
    `get_coupling_matrix` at a geometry the suite already compares to pymaster.
    """
    rng = np.random.default_rng(14)
    nside, lmax = 4, 7
    npix = 12 * nside ** 2
    mask = rng.uniform(0.3, 1, npix)
    maps = rng.normal(size=(2, npix))
    field = nmt.NmtField(mask, maps, lmax=lmax, lmax_mask=lmax, n_iter=0)
    bins = nmt.NmtBin.from_lmax_linear(lmax, 2)
    dense = nmt.NmtWorkspace()
    dense.compute_coupling_matrix(field, field, bins)
    assert not isinstance(dense.mcm, _RowPieces)

    monkeypatch.setattr(ws, "_MCM_PIECED_BYTES", 0)
    pieced = nmt.NmtWorkspace()
    pieced.compute_coupling_matrix(field, field, bins)
    assert isinstance(pieced.mcm, _RowPieces)
    np.testing.assert_allclose(
        np.asarray(pieced.get_coupling_matrix()),
        np.asarray(dense.get_coupling_matrix()),
        atol=1e-14,
    )
    cl = nmt.compute_coupled_cell(field, field)
    d_dense = np.asarray(dense.decouple_cell(cl))
    d_pieced = np.asarray(pieced.decouple_cell(cl))
    assert np.isfinite(d_pieced).all()
    np.testing.assert_allclose(d_pieced, d_dense, atol=1e-13)
    theory = rng.normal(size=(pieced.ncls, lmax + 1))
    np.testing.assert_allclose(
        np.asarray(pieced.couple_cell(theory)),
        np.asarray(dense.couple_cell(theory)),
        atol=1e-13,
    )


def test_chunked_left_contract_matches_single_gemm(monkeypatch):
    """The 4 GiB GEMM split is the same product as `output @ mcm`, up to summation order.

    `n=32` has a divisor at every power of two, so a threshold that asks for
    three chunks still finds `nchunk=4` rather than walking off `range(target, n+1)`.
    """
    rng = np.random.default_rng(0)
    n, n_out = 32, 5
    mcm = jax.numpy.asarray(rng.normal(size=(n, n)))
    output = jax.numpy.asarray(rng.normal(size=(n_out, n)))
    one = np.asarray(_left_contract(output, mcm))
    bytes_ = n * n * 8
    monkeypatch.setattr(ws, "_LEFT_CONTRACT_CHUNK_BYTES", bytes_ // 3)
    assert mcm.size * 8 > ws._LEFT_CONTRACT_CHUNK_BYTES
    chunked = np.asarray(_left_contract(output, mcm))
    np.testing.assert_allclose(chunked, one, atol=1e-14)


def test_large_wigner_cache_is_dropped_above_keep_bytes(monkeypatch):
    """A quadrature cache past the keep threshold is forgotten after the workspace."""
    monkeypatch.setattr(ws, "_wd_cache_keep_bytes", lambda: 1)
    ws._WD_TRIPLE_CACHE[("probe",)] = jax.numpy.zeros(8)
    ws._drop_large_wigner_cache()
    assert len(ws._WD_TRIPLE_CACHE) == 0
