import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import gmaster as nmt


def _wcs(resolution=30.0):
    WCS = pytest.importorskip("astropy.wcs").WCS
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---CAR", "DEC--CAR"]
    wcs.wcs.cdelt = [resolution, -resolution]
    wcs.wcs.crval = [0, 0]
    wcs.wcs.crpix = [1, 90 / resolution + 1]
    return wcs


@pytest.mark.parametrize("spin", [0, 1, 2, 3])
def test_car_geometry_and_transforms_match_namaster(spin):
    reference = pytest.importorskip("pymaster")
    wcs = _wcs()
    shape = (7, 12)
    got_info = nmt.NmtMapInfo(wcs, shape)
    ref_info = reference.NmtMapInfo(wcs, shape)
    for name in (
        "npix", "nx", "ny", "theta_min", "theta_max", "phi0",
        "d_theta", "d_phi",
    ):
        np.testing.assert_allclose(getattr(got_info, name), getattr(ref_info, name))
    np.testing.assert_allclose(got_info.si.weight, ref_info.si.weight, atol=3e-17)

    rng = np.random.default_rng(80 + spin)
    nmaps = 1 if spin == 0 else 2
    maps = rng.normal(size=(nmaps, *shape))
    got_ainfo = nmt.NmtAlmInfo(6)
    ref_ainfo = reference.NmtAlmInfo(6)
    got_maps = got_info.reform_map(maps)
    ref_maps = ref_info.reform_map(maps)
    got = nmt.map2alm(got_maps, spin, got_info, got_ainfo, n_iter=1)
    expected = reference.utils.map2alm(
        ref_maps, spin, ref_info, ref_ainfo, n_iter=1
    )
    np.testing.assert_allclose(got, expected, atol=4e-14)
    np.testing.assert_allclose(
        nmt.alm2map(got, spin, got_info, got_ainfo),
        reference.utils.alm2map(expected, spin, ref_info, ref_ainfo),
        atol=2e-13,
    )


@pytest.mark.parametrize("spins", [(0, 0), (0, 2), (2, 2)])
def test_car_fields_and_workspaces_match_namaster(spins):
    reference = pytest.importorskip("pymaster")
    wcs = _wcs()
    shape = (7, 12)
    lmax = 6
    rng = np.random.default_rng(90)
    mask = rng.uniform(0.2, 1, shape)
    got_fields = []
    ref_fields = []
    for spin in spins:
        maps = rng.normal(size=(1 if spin == 0 else 2, *shape))
        options = dict(
            spin=spin, wcs=wcs, lmax=lmax, lmax_mask=lmax,
            n_iter=0, n_iter_mask=0,
        )
        got_fields.append(nmt.NmtField(mask, maps, **options))
        ref_fields.append(reference.NmtField(mask, maps, **options))
    got = nmt.NmtWorkspace.from_fields(
        *got_fields, nmt.NmtBin.from_lmax_linear(lmax, 2)
    )
    expected = reference.NmtWorkspace.from_fields(
        *ref_fields, reference.NmtBin.from_lmax_linear(lmax, 2)
    )
    np.testing.assert_allclose(got.mcm, expected.get_coupling_matrix(), atol=2e-14)


def test_car_templates_purification_and_synfast():
    reference = pytest.importorskip("pymaster")
    wcs = _wcs()
    shape = (7, 12)
    lmax = 6
    rng = np.random.default_rng(91)
    theta = np.linspace(0, np.pi, shape[0])
    mask = np.sin(theta)[:, None] ** 2 * np.ones(shape)
    maps = rng.normal(size=(2, *shape))
    templates = rng.normal(size=(1, 2, *shape))
    for extra in (
        {"templates": templates},
        {"purify_e": True, "purify_b": True},
    ):
        options = dict(
            wcs=wcs, lmax=lmax, lmax_mask=lmax,
            n_iter=0, n_iter_mask=0, **extra,
        )
        got = nmt.NmtField(mask, maps, **options)
        expected = reference.NmtField(mask, maps, **options)
        np.testing.assert_allclose(got.get_alms(), expected.get_alms(), atol=3e-13)

    cells = np.ones((1, lmax + 1))
    first = nmt.synfast_spherical(1, cells, [0], wcs=wcs, lmax=lmax, seed=4)
    second = nmt.synfast_spherical(1, cells, [0], wcs=wcs, lmax=lmax, seed=4)
    assert first.shape == (1, *shape)
    np.testing.assert_array_equal(first, second)


def test_car_validation():
    wcs = _wcs()
    with pytest.raises(ValueError, match="2D"):
        nmt.NmtMapInfo(wcs, (84,))
    wrong = wcs.deepcopy()
    wrong.wcs.ctype[0] = "RA---TAN"
    with pytest.raises(ValueError, match="CAR"):
        nmt.NmtMapInfo(wrong, (7, 12))
