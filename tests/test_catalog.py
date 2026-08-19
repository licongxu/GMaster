import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import gmaster as nmt
from gmaster.utils import alm2catalog, catalog2alm


@pytest.mark.parametrize("spin", [0, 1, 2, 3])
def test_catalog_transforms_match_namaster(spin):
    reference = pytest.importorskip("pymaster")
    rng = np.random.default_rng(51 + spin)
    positions = np.array(
        [rng.uniform(0.05, np.pi - 0.05, 30), rng.uniform(0, 2 * np.pi, 30)]
    )
    values = rng.normal(size=(1 if spin == 0 else 2, 30))
    got = catalog2alm(values, positions, spin, 7)
    expected = reference.utils._catalog2alm_ducc0(values, positions, spin, 7)
    np.testing.assert_allclose(got, expected, atol=2e-6)
    np.testing.assert_allclose(
        alm2catalog(got, positions, spin, 7),
        reference.utils._alm2catalog_ducc0(np.asarray(got), positions, spin, 7),
        atol=2e-6,
    )


@pytest.mark.parametrize("spin", [0, 2])
def test_catalog_field_deprojection_and_workspace_match_namaster(spin):
    reference = pytest.importorskip("pymaster")
    rng = np.random.default_rng(60 + spin)
    nsource = 80
    positions = np.array(
        [
            rng.uniform(0.05, np.pi - 0.05, nsource),
            rng.uniform(0, 2 * np.pi, nsource),
        ]
    )
    weights = rng.uniform(0.5, 1.5, nsource)
    nmaps = 1 if spin == 0 else 2
    field = rng.normal(size=(nmaps, nsource))
    templates = rng.normal(size=(2, nmaps, nsource))
    variance = rng.uniform(0.1, 0.3, nsource)
    options = dict(
        spin=spin,
        templates=templates,
        noise_variance=variance,
        retain_catalog=True,
        lmax_mask=12,
    )
    got = nmt.NmtFieldCatalog(positions, weights, field, 6, **options)
    expected = reference.NmtFieldCatalog(
        positions, weights, field, 6, **options
    )
    np.testing.assert_allclose(got.alm_mask, expected.alm_mask, atol=2e-5)
    np.testing.assert_allclose(got.alm, expected.alm, atol=2e-6)
    np.testing.assert_allclose(got.alphas, expected.alphas, atol=1e-13)
    np.testing.assert_allclose(got.Nw, expected.Nw, atol=1e-13)
    np.testing.assert_allclose(got.Nf, expected.Nf, atol=1e-13)
    np.testing.assert_allclose(
        nmt.compute_coupled_cell(got, got),
        reference.compute_coupled_cell(expected, expected),
        atol=2e-6,
    )
    np.testing.assert_allclose(
        got.get_noise_deprojection_bias(),
        expected.get_noise_deprojection_bias(),
        atol=2e-8,
    )

    bins = nmt.NmtBin.from_lmax_linear(6, 2)
    reference_bins = reference.NmtBin.from_lmax_linear(6, 2)
    workspace = nmt.NmtWorkspace(got, got, bins)
    reference_workspace = reference.NmtWorkspace(expected, expected, reference_bins)
    np.testing.assert_allclose(
        workspace.mcm, reference_workspace.get_coupling_matrix(), atol=1e-5
    )


def test_catalog_retained_helpers_and_validation():
    reference = pytest.importorskip("pymaster")
    rng = np.random.default_rng(70)
    positions = np.array(
        [rng.uniform(0.1, 3, 100), rng.uniform(0, 2 * np.pi, 100)]
    )
    weights = rng.uniform(0.5, 1.5, 100)
    field = rng.normal(size=(1, 100))
    got = nmt.NmtFieldCatalog(
        positions, weights, field, 7, lmax_mask=7, retain_catalog=True
    )
    expected = reference.NmtFieldCatalog(
        positions, weights, field, 7, lmax_mask=7, retain_catalog=True
    )
    np.testing.assert_allclose(
        got.get_cloud_kernel(7), expected.get_cloud_kernel(7), atol=1e-14
    )
    np.testing.assert_allclose(
        got.get_catalog_variance_alm(),
        expected.get_catalog_variance_alm(),
        atol=5e-6,
    )
    got_map, got_nside = got.get_catalog_mask_map()
    expected_map, expected_nside = expected.get_catalog_mask_map()
    assert got_nside == expected_nside
    np.testing.assert_allclose(got_map, expected_map, atol=2e-5)

    with pytest.raises(ValueError, match="spin needs"):
        nmt.NmtFieldCatalog(positions, weights, None, 7)
    lite = nmt.NmtFieldCatalog(positions, weights, None, 7, spin=0)
    with pytest.raises(ValueError, match="no alms"):
        lite.get_alms()


@pytest.mark.parametrize("clustering", [False, True])
def test_momentum_and_clustering_random_catalogs_match_namaster(clustering):
    reference = pytest.importorskip("pymaster")
    rng = np.random.default_rng(80 + clustering)
    ndata, nrandom = 50, 150
    data_positions = np.array(
        [rng.uniform(0.1, 3, ndata), rng.uniform(0, 2 * np.pi, ndata)]
    )
    random_positions = np.array(
        [rng.uniform(0.1, 3, nrandom), rng.uniform(0, 2 * np.pi, nrandom)]
    )
    data_weights = rng.uniform(0.5, 1.5, ndata)
    random_weights = rng.uniform(0.5, 1.5, nrandom)
    if clustering:
        got = nmt.NmtFieldCatalogClustering(
            data_positions, data_weights, random_positions, random_weights,
            6, lmax_mask=10, retain_catalog=True,
        )
        expected = reference.NmtFieldCatalogClustering(
            data_positions, data_weights, random_positions, random_weights,
            6, lmax_mask=10, retain_catalog=True,
        )
    else:
        values = rng.normal(size=(2, ndata))
        got = nmt.NmtFieldCatalogMomentum(
            data_positions, data_weights, values, random_positions,
            random_weights, 6, lmax_mask=10, spin=2, retain_catalog=True,
        )
        expected = reference.NmtFieldCatalogMomentum(
            data_positions, data_weights, values, random_positions,
            random_weights, 6, lmax_mask=10, spin=2, retain_catalog=True,
        )
    np.testing.assert_allclose(got.alpha, expected.alpha, atol=1e-14)
    np.testing.assert_allclose(got.Nw, expected.Nw, atol=1e-13)
    np.testing.assert_allclose(got.Nf, expected.Nf, atol=1e-13)
    np.testing.assert_allclose(got.alm_mask, expected.alm_mask, atol=1e-5)
    np.testing.assert_allclose(got.alm, expected.alm, atol=2e-6)


@pytest.mark.parametrize("clustering", [False, True])
def test_momentum_and_clustering_map_templates_match_namaster(clustering):
    reference = pytest.importorskip("pymaster")
    rng = np.random.default_rng(90 + clustering)
    nsource = 80
    positions = np.array(
        [rng.uniform(0.1, 3, nsource), rng.uniform(0, 2 * np.pi, nsource)]
    )
    weights = rng.uniform(0.5, 1.5, nsource)
    mask = rng.uniform(0.2, 1, 12 * 4**2)
    templates = rng.normal(size=(1, 1, len(mask)))
    common = dict(
        mask=mask, lmax_mask=8, templates=templates, retain_catalog=True
    )
    if clustering:
        got = nmt.NmtFieldCatalogClustering(
            positions, weights, None, None, 6, **common
        )
        expected = reference.NmtFieldCatalogClustering(
            positions, weights, None, None, 6, **common
        )
    else:
        values = rng.normal(size=(1, nsource))
        variance = np.full(nsource, 0.2)
        got = nmt.NmtFieldCatalogMomentum(
            positions, weights, values, None, None, 6,
            noise_variance=variance, **common,
        )
        expected = reference.NmtFieldCatalogMomentum(
            positions, weights, values, None, None, 6,
            noise_variance=variance, **common,
        )
    np.testing.assert_allclose(got.alpha, expected.alpha, atol=1e-14)
    np.testing.assert_allclose(got.alm_mask, expected.alm_mask, atol=2e-13)
    np.testing.assert_allclose(got.alphas, expected.alphas, atol=2e-7)
    np.testing.assert_allclose(got.alm, expected.alm, atol=2e-6)
    np.testing.assert_allclose(
        got.get_noise_deprojection_bias(),
        expected.get_noise_deprojection_bias(),
        atol=2e-8,
    )
