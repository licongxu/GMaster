import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import gmaster as nmt
from gmaster import utils


_HAS_NVIDIA_GPU = any(
    device.platform == "gpu" and "NVIDIA" in device.device_kind.upper()
    for device in jax.devices()
)


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


@pytest.mark.parametrize("L,L_work", [(1, 1), (4, 4), (4, 7)])
def test_gathered_alm_unpack_matches_healpy_layout(L, L_work):
    ainfo = nmt.NmtAlmInfo(L - 1)
    rng = np.random.default_rng(L_work)
    alm = rng.normal(size=ainfo.nelem) + 1j * rng.normal(size=ainfo.nelem)
    expected = np.zeros((L_work, 2 * L_work - 1), dtype=np.complex128)
    expected[ainfo._ell, L_work - 1 + ainfo._m] = alm
    expected[ainfo._ell, L_work - 1 - ainfo._m] = np.where(
        ainfo._m > 0, (-1) ** ainfo._m * np.conj(alm), alm
    )
    np.testing.assert_array_equal(utils._unpack_real(alm, L, L_work), expected)
    b_alm = rng.normal(size=ainfo.nelem) + 1j * rng.normal(size=ainfo.nelem)
    b_expected = np.zeros_like(expected)
    b_expected[ainfo._ell, L_work - 1 + ainfo._m] = b_alm
    b_expected[ainfo._ell, L_work - 1 - ainfo._m] = np.where(
        ainfo._m > 0, (-1) ** ainfo._m * np.conj(b_alm), b_alm
    )
    np.testing.assert_array_equal(
        utils._unpack_spin(np.stack((alm, b_alm)), L, L_work),
        -(expected + 1j * b_expected),
    )


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
def test_fused_scalar_transforms_match_namaster_and_generic_jax():
    reference = pytest.importorskip("pymaster")
    nside = 64
    lmax = 95
    npix = 12 * nside**2
    minfo = nmt.NmtMapInfo(None, (npix,))
    ainfo = nmt.NmtAlmInfo(lmax)
    ref_minfo = reference.NmtMapInfo(None, (npix,))
    ref_ainfo = reference.NmtAlmInfo(lmax)
    alms = _random_alms(np.random.default_rng(41), 1, ainfo, spin=0)
    original = nmt.nmt_params.sht_calculator
    try:
        nmt.set_sht_calculator("jax")
        fused_map = nmt.alm2map(alms, 0, minfo, ainfo)
        fused_alms = [
            nmt.map2alm(fused_map, 0, minfo, ainfo, n_iter=n_iter)
            for n_iter in (0, 1)
        ]

        nmt.set_sht_calculator("jax-generic")
        generic_map = nmt.alm2map(alms, 0, minfo, ainfo)
        generic_alms = [
            nmt.map2alm(fused_map, 0, minfo, ainfo, n_iter=n_iter)
            for n_iter in (0, 1)
        ]

        reference_map = reference.alm2map(alms, 0, ref_minfo, ref_ainfo)
        np.testing.assert_allclose(fused_map, reference_map, atol=3e-11)
        np.testing.assert_allclose(fused_map, generic_map, atol=3e-11)
        for n_iter, fused_alm, generic_alm in zip(
            (0, 1), fused_alms, generic_alms
        ):
            reference_alm = reference.map2alm(
                np.asarray(fused_map),
                0,
                ref_minfo,
                ref_ainfo,
                n_iter=n_iter,
            )
            np.testing.assert_allclose(fused_alm, reference_alm, atol=3e-11)
            np.testing.assert_allclose(fused_alm, generic_alm, atol=3e-11)
    finally:
        nmt.set_sht_calculator(original)


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
def test_dfp32_scalar_analysis_matches_namaster():
    # jax-dfp32 runs the double-fp32 latitudinal SHT (analysis only; the
    # iterative-refinement residual synthesis stays fp64). The Pallas path is
    # active at L >= 128; double-fp32 retains the 3e-11 gate up to L ~ 192
    # at nside 64, so test at L = 160 with a large margin.
    reference = pytest.importorskip("pymaster")
    nside = 64
    lmax = 159
    npix = 12 * nside**2
    minfo = nmt.NmtMapInfo(None, (npix,))
    ainfo = nmt.NmtAlmInfo(lmax)
    ref_minfo = reference.NmtMapInfo(None, (npix,))
    ref_ainfo = reference.NmtAlmInfo(lmax)
    alms = _random_alms(np.random.default_rng(41), 1, ainfo, spin=0)
    original = nmt.nmt_params.sht_calculator
    try:
        nmt.set_sht_calculator("jax")
        fused_map = nmt.alm2map(alms, 0, minfo, ainfo)
        fused_alms = [
            nmt.map2alm(fused_map, 0, minfo, ainfo, n_iter=n_iter)
            for n_iter in (0, 1)
        ]
        reference_map = reference.alm2map(alms, 0, ref_minfo, ref_ainfo)

        nmt.set_sht_calculator("jax-dfp32")
        for n_iter in (0, 1):
            dfp32_alm = nmt.map2alm(fused_map, 0, minfo, ainfo, n_iter=n_iter)
            np.testing.assert_allclose(
                dfp32_alm, fused_alms[n_iter], atol=3e-11
            )
            np.testing.assert_allclose(
                dfp32_alm,
                reference.map2alm(
                    np.asarray(fused_map),
                    0,
                    ref_minfo,
                    ref_ainfo,
                    n_iter=n_iter,
                ),
                atol=3e-11,
            )
        # the synthesis direction is unaffected by the calculator choice
        dfp32_map = nmt.alm2map(alms, 0, minfo, ainfo)
        np.testing.assert_allclose(dfp32_map, reference_map, atol=3e-11)
    finally:
        nmt.set_sht_calculator(original)


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
def test_fused_scalar_transform_gradients_match_generic_jax():
    nside = 16
    L = 3 * nside
    minfo = nmt.NmtMapInfo(None, (12 * nside**2,))
    ainfo = nmt.NmtAlmInfo(L - 1)
    key = jax.random.key(42)
    maps = jax.random.normal(key, (1, minfo.npix), dtype=jnp.float64)
    alms = jnp.asarray(_random_alms(np.random.default_rng(42), 1, ainfo, 0))

    def fused_analysis_loss(values):
        transformed = utils._map2alm_core_pallas(
            values,
            ainfo._ell,
            ainfo._m,
            nside=nside,
            L=L,
            L_work=L,
            n_iter=0,
        )
        return jnp.real(jnp.vdot(transformed, transformed))

    def fused_synthesis_loss(values):
        transformed = utils._alm2map_core_pallas(
            values, nside=nside, L=L, L_work=L
        )
        return jnp.sum(transformed**2)

    original = nmt.nmt_params.sht_calculator
    try:
        fused_analysis_grad = jax.jit(jax.grad(fused_analysis_loss))(maps)
        fused_synthesis_grad = jax.jit(jax.grad(fused_synthesis_loss))(alms)

        nmt.set_sht_calculator("jax-generic")

        def generic_analysis_loss(values):
            transformed = nmt.map2alm(
                values, 0, minfo, ainfo, n_iter=0
            )
            return jnp.real(jnp.vdot(transformed, transformed))

        def generic_synthesis_loss(values):
            transformed = nmt.alm2map(values, 0, minfo, ainfo)
            return jnp.sum(transformed**2)

        generic_analysis_grad = jax.jit(jax.grad(generic_analysis_loss))(maps)
        generic_synthesis_grad = jax.jit(jax.grad(generic_synthesis_loss))(
            alms
        )
        np.testing.assert_allclose(
            fused_analysis_grad, generic_analysis_grad, rtol=2e-10, atol=2e-12
        )
        np.testing.assert_allclose(
            fused_synthesis_grad,
            generic_synthesis_grad,
            rtol=2e-10,
            atol=2e-10,
        )
    finally:
        nmt.set_sht_calculator(original)


@pytest.mark.skipif(
    len([device for device in jax.devices() if device.platform == "gpu"]) < 2,
    reason="requires two GPUs",
)
@pytest.mark.parametrize("spin", [0, 2])
def test_multi_gpu_transforms_match_single_gpu(spin):
    nside = 4
    ainfo = nmt.NmtAlmInfo(3 * nside - 1)
    minfo = nmt.NmtMapInfo(None, (12 * nside**2,))
    nmaps = 1 if spin == 0 else 2
    maps = np.random.default_rng(spin + 10).normal(size=(nmaps, minfo.npix))
    original = nmt.nmt_params.sht_calculator
    try:
        nmt.set_sht_calculator("jax-single")
        single_alm = nmt.map2alm(maps, spin, minfo, ainfo, n_iter=1)
        single_map = nmt.alm2map(single_alm, spin, minfo, ainfo)
        nmt.set_sht_calculator("jax-mgpu")
        multi_alm = nmt.map2alm(maps, spin, minfo, ainfo, n_iter=1)
        multi_map = nmt.alm2map(multi_alm, spin, minfo, ainfo)
        np.testing.assert_allclose(multi_alm, single_alm, atol=1e-12)
        np.testing.assert_allclose(multi_map, single_map, atol=1e-12)
    finally:
        nmt.set_sht_calculator(original)


@pytest.mark.skipif(
    len([device for device in jax.devices() if device.platform == "gpu"]) < 2,
    reason="requires two GPUs",
)
def test_fused_multi_gpu_scalar_transforms_match_single_gpu():
    nside = 64
    ainfo = nmt.NmtAlmInfo(3 * nside - 1)
    minfo = nmt.NmtMapInfo(None, (12 * nside**2,))
    rng = np.random.default_rng(64)
    maps = rng.normal(size=(1, minfo.npix))
    alms = _random_alms(rng, 1, ainfo, spin=0)
    original = nmt.nmt_params.sht_calculator
    try:
        nmt.set_sht_calculator("jax")
        single_alm = nmt.map2alm(maps, 0, minfo, ainfo, n_iter=1)
        single_map = nmt.alm2map(alms, 0, minfo, ainfo)
        nmt.set_sht_calculator("jax-mgpu")
        multi_alm = nmt.map2alm(maps, 0, minfo, ainfo, n_iter=1)
        multi_map = nmt.alm2map(alms, 0, minfo, ainfo)
        np.testing.assert_allclose(multi_alm, single_alm, atol=2e-13)
        np.testing.assert_allclose(multi_map, single_map, atol=2e-12)
    finally:
        nmt.set_sht_calculator(original)
