import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import gmaster as nmt
from gmaster.field_flat import _flat_alm2map, _flat_map2alm


@pytest.mark.parametrize("shape", [(7, 8), (8, 9)])
@pytest.mark.parametrize("spin", [0, 1, 2, 3])
def test_flat_transform_roundtrip(shape, spin):
    rng = np.random.default_rng(20 + spin)
    nmaps = 1 if spin == 0 else 2
    maps = rng.normal(size=(nmaps, *shape))
    alms = _flat_map2alm(maps, spin, 0.3, 0.2)
    recovered = _flat_alm2map(alms, spin, 0.3, 0.2, shape[1])
    np.testing.assert_allclose(recovered, maps, atol=3e-14)


def test_flat_fields_and_coupled_spectra_match_namaster():
    reference = pytest.importorskip("pymaster")
    rng = np.random.default_rng(31)
    ny, nx = 8, 10
    lx, ly = 0.25, 0.18
    mask = rng.uniform(0.2, 1, (ny, nx))
    scalar = rng.normal(size=(1, ny, nx))
    spin2 = rng.normal(size=(2, ny, nx))
    b = nmt.NmtBinFlat([0, 60, 120], [60, 120, 220])
    rb = reference.NmtBinFlat([0, 60, 120], [60, 120, 220])

    got0 = nmt.NmtFieldFlat(lx, ly, mask, scalar)
    got2 = nmt.NmtFieldFlat(lx, ly, mask, spin2)
    ref0 = reference.NmtFieldFlat(lx, ly, mask, scalar)
    ref2 = reference.NmtFieldFlat(lx, ly, mask, spin2)

    np.testing.assert_allclose(got0.get_maps(), ref0.get_maps(), atol=2e-14)
    np.testing.assert_allclose(got2.get_maps(), ref2.get_maps(), atol=2e-14)
    np.testing.assert_allclose(got0.get_ell_sampling(), ref0.get_ell_sampling())
    for gf1, gf2, rf1, rf2 in (
        (got0, got0, ref0, ref0),
        (got0, got2, ref0, ref2),
        (got2, got2, ref2, ref2),
    ):
        np.testing.assert_allclose(
            nmt.compute_coupled_cell_flat(gf1, gf2, b),
            reference.compute_coupled_cell_flat(rf1, rf2, rb),
            atol=2e-14,
        )


def test_flat_templates_and_purification_match_namaster():
    reference = pytest.importorskip("pymaster")
    rng = np.random.default_rng(32)
    ny, nx = 9, 8
    lx, ly = 0.2, 0.16
    y, x = np.mgrid[:ny, :nx]
    mask = np.sin(np.pi * (x + 0.5) / nx) ** 2 * np.sin(
        np.pi * (y + 0.5) / ny
    ) ** 2
    template = rng.normal(size=(1, 2, ny, nx))
    maps = 0.4 * template[0] + rng.normal(size=(2, ny, nx))
    b = nmt.NmtBinFlat([0, 70, 140], [70, 140, 260])
    rb = reference.NmtBinFlat([0, 70, 140], [70, 140, 260])
    options = dict(templates=template, purify_e=True, purify_b=True)
    got = nmt.NmtFieldFlat(lx, ly, mask, maps, **options)
    ref = reference.NmtFieldFlat(lx, ly, mask, maps, **options)

    np.testing.assert_allclose(got.get_maps(), ref.get_maps(), rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(got.get_templates(), ref.get_templates(), atol=2e-14)
    np.testing.assert_allclose(
        nmt.compute_coupled_cell_flat(got, got, b),
        reference.compute_coupled_cell_flat(ref, ref, rb),
        rtol=3e-12,
        atol=3e-12,
    )


def test_flat_mask_only_and_validation():
    mask = np.ones((4, 5))
    field = nmt.NmtFieldFlat(0.2, 0.3, mask, None, spin=2)
    with pytest.raises(ValueError, match="no alms"):
        field.get_alms()
    with pytest.raises(ValueError, match="supply field spin"):
        nmt.NmtFieldFlat(0.2, 0.3, mask, None)
    with pytest.raises(ValueError, match="same shape"):
        nmt.NmtFieldFlat(0.2, 0.3, mask, np.ones((1, 3, 5)))
    with pytest.raises(ValueError, match="spin-2"):
        nmt.NmtFieldFlat(0.2, 0.3, mask, np.ones((1, 4, 5)), purify_e=True)


@pytest.mark.parametrize(
    "spins,purities,is_teb",
    [
        ((0, 0), ((False, False), (False, False)), False),
        ((0, 2), ((False, False), (True, False)), False),
        ((2, 2), ((True, False), (False, True)), False),
        ((0, 2), ((False, False), (True, True)), True),
    ],
)
def test_flat_workspace_matches_namaster(spins, purities, is_teb):
    reference = pytest.importorskip("pymaster")
    rng = np.random.default_rng(41)
    ny, nx = 8, 10
    lx, ly = 0.25, 0.18
    y, x = np.mgrid[:ny, :nx]
    mask1 = np.sin(np.pi * (x + 0.5) / nx) ** 2 * np.sin(
        np.pi * (y + 0.5) / ny
    ) ** 2
    mask2 = mask1 * (0.8 + 0.1 * np.cos(2 * np.pi * x / nx))
    maps = [
        rng.normal(size=(1 if spin == 0 else 2, ny, nx)) for spin in spins
    ]
    options = [
        dict(spin=spin, purify_e=pure[0], purify_b=pure[1])
        for spin, pure in zip(spins, purities)
    ]
    fields = [
        nmt.NmtFieldFlat(lx, ly, mask, field_maps, **field_options)
        for mask, field_maps, field_options in zip(
            (mask1, mask2), maps, options
        )
    ]
    reference_fields = [
        reference.NmtFieldFlat(lx, ly, mask, field_maps, **field_options)
        for mask, field_maps, field_options in zip(
            (mask1, mask2), maps, options
        )
    ]
    bins = nmt.NmtBinFlat([0, 60, 120], [60, 120, 220])
    reference_bins = reference.NmtBinFlat([0, 60, 120], [60, 120, 220])
    got = nmt.NmtWorkspaceFlat(*fields, bins, is_teb=is_teb)
    expected = reference.NmtWorkspaceFlat(
        *reference_fields, reference_bins, is_teb=is_teb
    )
    ncls, nbands, nells = got.ncls, got.nbands, got.n_ell
    expected_unbinned = reference.nmtlib.wsp_flat_get_mcm(
        expected.wsp, 1, 0, ncls**2 * nbands * nells
    ).reshape(ncls * nbands, ncls * nells)
    expected_binned = reference.nmtlib.wsp_flat_get_mcm(
        expected.wsp, 0, 0, (ncls * nbands) ** 2
    ).reshape(ncls * nbands, ncls * nbands)
    np.testing.assert_allclose(got.mcm, expected_unbinned, atol=1e-14)
    np.testing.assert_allclose(got.mcm_binned, expected_binned, atol=1e-14)

    ells = np.linspace(0, 300, 51)
    theory = rng.normal(size=(ncls, len(ells)))
    np.testing.assert_allclose(
        got.couple_cell(ells, theory), expected.couple_cell(ells, theory), atol=1e-13
    )
    if not is_teb:
        coupled = nmt.compute_coupled_cell_flat(fields[0], fields[1], bins)
        reference_coupled = reference.compute_coupled_cell_flat(
            reference_fields[0], reference_fields[1], reference_bins
        )
        np.testing.assert_allclose(
            got.decouple_cell(coupled),
            expected.decouple_cell(reference_coupled),
            atol=2e-13,
        )


def test_flat_workspace_fits_interoperability(tmp_path):
    reference = pytest.importorskip("pymaster")
    rng = np.random.default_rng(42)
    mask = rng.uniform(0.3, 1, (6, 8))
    maps = rng.normal(size=(2, 6, 8))
    field = nmt.NmtFieldFlat(0.2, 0.15, mask, maps)
    bins = nmt.NmtBinFlat([0, 80], [80, 180])
    workspace = nmt.NmtWorkspaceFlat(field, field, bins)
    path = tmp_path / "flat_workspace.fits"
    workspace.write_to(path)

    restored = nmt.NmtWorkspaceFlat.from_file(path)
    np.testing.assert_allclose(restored.mcm, workspace.mcm)
    np.testing.assert_allclose(restored.mcm_binned, workspace.mcm_binned)
    reference_workspace = reference.NmtWorkspaceFlat.from_file(path)
    spectra = rng.normal(size=(4, 2))
    np.testing.assert_allclose(
        restored.decouple_cell(spectra),
        reference_workspace.decouple_cell(spectra),
        atol=2e-13,
    )


@pytest.mark.parametrize("pure", [False, True])
def test_flat_deprojection_bias_and_full_master_match_namaster(pure):
    reference = pytest.importorskip("pymaster")
    rng = np.random.default_rng(43)
    ny, nx = 7, 8
    lx, ly = 0.2, 0.16
    y, x = np.mgrid[:ny, :nx]
    mask = np.sin(np.pi * (x + 0.5) / nx) ** 2 * np.sin(
        np.pi * (y + 0.5) / ny
    ) ** 2
    maps1 = rng.normal(size=(2, ny, nx))
    maps2 = rng.normal(size=(2, ny, nx))
    templates1 = rng.normal(size=(1, 2, ny, nx))
    templates2 = rng.normal(size=(1, 2, ny, nx))
    options1 = dict(
        templates=templates1, purify_e=pure, purify_b=False
    )
    options2 = dict(
        templates=templates2, purify_e=False, purify_b=pure
    )
    got_fields = (
        nmt.NmtFieldFlat(lx, ly, mask, maps1, **options1),
        nmt.NmtFieldFlat(lx, ly, mask, maps2, **options2),
    )
    reference_fields = (
        reference.NmtFieldFlat(lx, ly, mask, maps1, **options1),
        reference.NmtFieldFlat(lx, ly, mask, maps2, **options2),
    )
    bins = nmt.NmtBinFlat([0, 80], [80, 180])
    reference_bins = reference.NmtBinFlat([0, 80], [80, 180])
    ells = np.linspace(0, 250, 30)
    guess = rng.uniform(0.1, 1, (4, len(ells)))

    got_bias = nmt.deprojection_bias_flat(*got_fields, bins, ells, guess)
    expected_bias = reference.deprojection_bias_flat(
        *reference_fields, reference_bins, ells, guess
    )
    np.testing.assert_allclose(got_bias, expected_bias, atol=1e-13)
    workspace = nmt.NmtWorkspaceFlat(*got_fields, bins)
    reference_workspace = reference.NmtWorkspaceFlat(
        *reference_fields, reference_bins
    )
    np.testing.assert_allclose(
        nmt.compute_full_master_flat(
            *got_fields,
            bins,
            cl_guess=guess,
            ells_guess=ells,
            workspace=workspace,
        ),
        reference.compute_full_master_flat(
            *reference_fields,
            reference_bins,
            cl_guess=guess,
            ells_guess=ells,
            workspace=reference_workspace,
        ),
        atol=2e-13,
    )


@pytest.mark.parametrize("apotype", ["C1", "C2", "Smooth"])
def test_flat_mask_apodization_matches_namaster(apotype):
    reference = pytest.importorskip("pymaster")
    ny, nx = 21, 25
    lx, ly = 0.2, 0.17
    y, x = np.mgrid[:ny, :nx]
    mask = np.ones((ny, nx))
    mask[(x - 12) ** 2 + (y - 10) ** 2 < 16] = 0
    mask[:2] = 0
    np.testing.assert_allclose(
        nmt.mask_apodization_flat(mask, lx, ly, 1.2, apotype),
        reference.mask_apodization_flat(mask, lx, ly, 1.2, apotype),
        atol=5e-15,
    )


def test_synfast_flat_is_reproducible_and_has_requested_covariance():
    covariance = np.array(
        [[1.0, 0.2, -0.1], [0.2, 2.0, 0.3], [-0.1, 0.3, 1.5]]
    )
    spectra = np.stack(
        [covariance[row, column] * np.ones(180)
         for row in range(3) for column in range(row, 3)]
    )
    maps = nmt.synfast_flat(64, 64, 2.0, 2.0, spectra, [0, 2], seed=7)
    np.testing.assert_array_equal(
        maps, nmt.synfast_flat(64, 64, 2.0, 2.0, spectra, [0, 2], seed=7)
    )
    mask = np.ones((64, 64))
    scalar = nmt.NmtFieldFlat(2.0, 2.0, mask, maps[:1])
    spin2 = nmt.NmtFieldFlat(2.0, 2.0, mask, maps[1:])
    bins = nmt.NmtBinFlat([10], [100])
    measured = np.concatenate(
        [
            np.asarray(nmt.compute_coupled_cell_flat(scalar, scalar, bins))[:, 0],
            np.asarray(nmt.compute_coupled_cell_flat(scalar, spin2, bins))[:, 0],
            np.asarray(nmt.compute_coupled_cell_flat(spin2, spin2, bins))[:, 0],
        ]
    )
    expected = [1.0, 0.2, -0.1, 2.0, 0.3, 0.3, 1.5]
    np.testing.assert_allclose(measured, expected, atol=0.15)
