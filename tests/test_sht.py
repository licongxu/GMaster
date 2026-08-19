import healpy as hp
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import gmaster as nmt


def _random_alms(rng, nmaps, ainfo, spin):
    alms = rng.normal(size=(nmaps, ainfo.nelem)) + 1j * rng.normal(
        size=(nmaps, ainfo.nelem)
    )
    for m in range(ainfo.lmax + 1):
        for ell in range(m, ainfo.lmax + 1):
            index = hp.Alm.getidx(ainfo.lmax, ell, m)
            if m == 0:
                alms[:, index] = alms[:, index].real
            if ell < spin:
                alms[:, index] = 0
    return alms


@pytest.mark.parametrize("spin", [0, 2])
def test_healpix_transforms_match_namaster(spin):
    reference = pytest.importorskip("pymaster")
    nside = 8
    lmax = 10
    npix = 12 * nside**2
    minfo = nmt.NmtMapInfo(None, (npix,))
    ainfo = nmt.NmtAlmInfo(lmax)
    ref_minfo = reference.NmtMapInfo(None, (npix,))
    ref_ainfo = reference.NmtAlmInfo(lmax)
    alms = _random_alms(np.random.default_rng(3), 1 if spin == 0 else 2, ainfo, spin)

    got_maps = nmt.alm2map(alms, spin, minfo, ainfo)
    ref_maps = reference.alm2map(alms, spin, ref_minfo, ref_ainfo)
    np.testing.assert_allclose(got_maps, ref_maps, atol=2e-13)

    for n_iter in (0, 3):
        got_alms = nmt.map2alm(got_maps, spin, minfo, ainfo, n_iter=n_iter)
        ref_alms = reference.map2alm(
            np.asarray(got_maps), spin, ref_minfo, ref_ainfo, n_iter=n_iter
        )
        np.testing.assert_allclose(got_alms, ref_alms, atol=3e-13)


def test_map_and_alm_metadata_match_namaster():
    reference = pytest.importorskip("pymaster")
    minfo = nmt.NmtMapInfo(None, (12 * 16**2,))
    ainfo = nmt.NmtAlmInfo(47)
    ref_ainfo = reference.NmtAlmInfo(47)

    assert minfo.get_lmax() == 47
    assert minfo == nmt.NmtMapInfo(None, (12 * 16**2,))
    assert minfo != nmt.NmtMapInfo(None, (12 * 8**2,))
    assert ainfo.nelem == ref_ainfo.nelem
    np.testing.assert_array_equal(ainfo.mstart, ref_ainfo.mstart)
    assert ainfo == nmt.NmtAlmInfo(47)


def test_transform_keyword_names_match_namaster():
    minfo = nmt.NmtMapInfo(None, (48,))
    ainfo = nmt.NmtAlmInfo(1)
    result = nmt.map2alm(
        map=np.ones((1, 48)),
        spin=0,
        map_info=minfo,
        alm_info=ainfo,
        n_iter=0,
    )
    assert result.shape == (1, ainfo.nelem)
    np.testing.assert_allclose(
        nmt.moore_penrose_pinvh(mat=np.eye(2), tol_pinv=1e-10), np.eye(2)
    )


def test_spin_transform_at_default_healpix_bandlimit():
    reference = pytest.importorskip("pymaster")
    nside = 4
    lmax = 3 * nside - 1
    npix = 12 * nside**2
    minfo = nmt.NmtMapInfo(None, (npix,))
    ainfo = nmt.NmtAlmInfo(lmax)
    ref_minfo = reference.NmtMapInfo(None, (npix,))
    ref_ainfo = reference.NmtAlmInfo(lmax)
    alms = _random_alms(np.random.default_rng(7), 2, ainfo, spin=2)
    maps = nmt.alm2map(alms, 2, minfo, ainfo)

    np.testing.assert_allclose(
        maps,
        reference.alm2map(alms, 2, ref_minfo, ref_ainfo),
        atol=2e-13,
    )
    np.testing.assert_allclose(
        nmt.map2alm(maps, 2, minfo, ainfo, n_iter=3),
        reference.map2alm(np.asarray(maps), 2, ref_minfo, ref_ainfo, n_iter=3),
        atol=4e-13,
    )


def test_transform_shape_validation():
    minfo = nmt.NmtMapInfo(None, (12 * 4**2,))
    ainfo = nmt.NmtAlmInfo(7)
    with pytest.raises(ValueError, match="wrong shape"):
        nmt.map2alm(np.ones((2, minfo.npix)), 0, minfo, ainfo, n_iter=0)
    with pytest.raises(ValueError, match="wrong shape"):
        nmt.alm2map(np.ones((1, ainfo.nelem)), 2, minfo, ainfo)
