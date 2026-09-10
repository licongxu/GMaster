import gc

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from s2fft.utils import healpix_ffts, quadrature_jax

jax.config.update("jax_enable_x64", True)

import gmaster as nmt
from gmaster import _theta_matrix, utils


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


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
def test_matrix_theta_stage_matches_fused_kernel():
    # jax-matrix swaps the in-kernel Legendre recurrence for a cached m-banded
    # matrix. The values are generated by the same recurrence, so the analysis
    # result must match the fused kernel, NaMaster, and the ell >= m support.
    reference = pytest.importorskip("pymaster")
    nside = 64
    lmax = 3 * nside - 1
    npix = 12 * nside**2
    minfo = nmt.NmtMapInfo(None, (npix,))
    ainfo = nmt.NmtAlmInfo(lmax)
    ref_minfo = reference.NmtMapInfo(None, (npix,))
    ref_ainfo = reference.NmtAlmInfo(lmax)
    rng = np.random.default_rng(77)
    alms = _random_alms(rng, 1, ainfo, spin=0)
    reference_map = reference.alm2map(alms, 0, ref_minfo, ref_ainfo)
    original = nmt.nmt_params.sht_calculator
    try:
        nmt.set_sht_calculator("jax")
        fused_map = nmt.alm2map(alms, 0, minfo, ainfo)
        fused_alms = [
            nmt.map2alm(fused_map, 0, minfo, ainfo, n_iter=n_iter)
            for n_iter in (0, 1)
        ]
        nmt.set_sht_calculator("jax-matrix")
        for n_iter in (0, 1):
            matrix_alm = nmt.map2alm(fused_map, 0, minfo, ainfo, n_iter=n_iter)
            np.testing.assert_allclose(matrix_alm, fused_alms[n_iter], atol=1e-13)
            np.testing.assert_allclose(
                matrix_alm,
                reference.map2alm(
                    np.asarray(fused_map), 0, ref_minfo, ref_ainfo,
                    n_iter=n_iter,
                ),
                atol=1e-13,
            )
        # synthesis is untouched by the analysis calculator
        matrix_map = nmt.alm2map(alms, 0, minfo, ainfo)
        np.testing.assert_allclose(matrix_map, reference_map, atol=1e-12)
        np.testing.assert_allclose(matrix_map, fused_map, atol=1e-14)

        L = lmax + 1
        theta = utils._stable_thetas(L, nside)
        ftm = utils._forward_healpix_fft(jnp.asarray(reference_map), L=L,
                                         nside=nside, reality=True)
        positive = _theta_matrix.forward_latitudinal(
            ftm, L=L, nside=nside, theta=theta,
            weights=quadrature_jax.quad_weights_transform(L, "healpix", nside),
            phase=-healpix_ffts.p2phi_rings_jax(jnp.arange(len(theta)), nside),
        )
        below = jnp.arange(L)[:, None] < jnp.arange(L)[None, :]
        assert int(positive.size) == L * L
        assert float(jnp.max(jnp.abs(positive[below]))) == 0.0
    finally:
        nmt.set_sht_calculator(original)


@pytest.mark.parametrize("nside, L", [(32, 95), (64, 191), (32, 40), (16, 71)])
def test_ring_synthesis_from_positive_half_matches_centred_window(nside, L):
    """The positive-m ring synthesis equals the mirrored-window one it replaces.

    `_inverse_ring_fft` mirrors the block into 2L coefficients and chirp-Z transforms
    all of it; `_inverse_ring_fft_herm` spends the exact mirror as `2 Re P - Re F_0` and
    needs the CZT bound `L + width - 1` instead of `2L - 1 + width`: half the transform
    size for the same ring values, measured 2.07x on the stage at Nside 512. The polar
    rows are read out of a concatenated slot buffer, the belt of one plain inverse FFT.
    `L = 40` at Nside 32 is a band wider than the polar rings' own `nphi`, so the cap
    chirp-Z has to alias there; `L = 71 > 4*nside` declines the belt and must still match.
    """
    positive = (
        jax.random.normal(jax.random.PRNGKey(11), (4 * nside - 1, L))
        + 1j * jax.random.normal(jax.random.PRNGKey(12), (4 * nside - 1, L))
    )
    reference = utils._inverse_ring_fft(positive, L=L, nside=nside)
    got = utils._inverse_ring_fft_herm(
        positive, utils._ring_synthesis_tables(L, nside), L=L, nside=nside
    )
    np.testing.assert_allclose(got, reference, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("nside, L", [(16, 47), (32, 95), (16, 64), (16, 71), (32, 40)])
def test_ring_analysis_matches_direct_dft_ring_by_ring(nside, L):
    """Every ring's m block equals an explicit `sum_p x_p exp(-2i pi p m / nphi)`.

    The azimuthal stage now takes two routes: the equatorial belt (`nphi = 4*nside`, a
    power of two wider than the band) is one plain FFT of a contiguous reshape, the polar
    rings keep the Bluestein chirp-Z at the map-wide width. Both routes are checked here
    against the definition, on cap rings, on the two boundary rings and on belt rings.
    `L = 64` is the gate boundary `L == 4*nside`; `L = 71 > 4*nside` is an overset band,
    for which the belt shortcut is declined and every ring goes back through the
    single-width chirp-Z. `L = 40` at Nside 32 is wider than the polar rings' own `nphi`,
    so their chirp-Z has to alias.
    """
    nphi, start, _, _, _ = utils._ring_czt_constants_numpy(L, nside)
    map_flat = jnp.asarray(
        np.random.default_rng(4).normal(size=12 * nside ** 2)
    )
    got = np.asarray(
        utils._forward_ring_fft_positive(
            map_flat, utils._ring_analysis_tables(L, nside), L=L, nside=nside
        )
    )
    ntheta = 4 * nside - 1
    m_index = np.arange(L)
    for ring in {0, nside // 2, nside - 2, nside - 1, nside,
                 2 * nside, 3 * nside - 1, 3 * nside, ntheta - 1}:
        period = nphi[ring]
        direct = (np.exp(-2j * np.pi * np.outer(m_index, np.arange(period)) / period)
                  @ map_flat[start[ring]:start[ring] + period])
        scale = float(np.max(np.abs(direct)))
        np.testing.assert_allclose(got[ring], direct, rtol=0.0, atol=1e-12 * scale)


@pytest.mark.parametrize("nside, L", [(8, 16), (8, 23), (8, 24), (8, 32), (8, 39),
                                      (16, 32), (16, 47), (16, 71)])
def test_spin_ring_window_matches_direct_dft_both_ways(nside, L):
    """The polarised ring stage's centred window and its inverse are the ring DFT.

    `_forward_ring_fft_full` returns the centred window `m in [-(L-1), L)`. On the
    equatorial belt that window is wider than the band but narrower than one period of the
    ring sum (`nphi = 4*nside >= L`), so it is two contiguous slices of a single plain FFT:
    orders `m < 0` are the residues `F_-m = F_(nphi-m)` at the top of the transform, orders
    `m >= 0` at the bottom. `_inverse_ring_fft_complex` runs the same identity backwards by
    adding coefficients that share a residue into one slot and transforming once; when
    `L > 2*nside` the positive and negative halves land in overlapping slots and must be
    summed rather than concatenated. Cap rings keep the chirp-Z in both directions.

    Every case is checked against the definition on cap, boundary and belt rings, forward
    and back. `L = 2*nside` is the boundary where the two slot ranges just touch,
    `L = 3*nside`/`3*nside-1` overlap them, and `L = 4*nside+7` declines the belt entirely.
    """
    nphi, start, _, _, _ = utils._ring_czt_constants_numpy(L, nside)
    npix = 12 * nside ** 2
    ntheta = 4 * nside - 1
    rng = np.random.default_rng(7)
    signal = np.asarray(rng.normal(size=npix) + 1j * rng.normal(size=npix))
    centered = (jax.random.normal(jax.random.PRNGKey(3), (ntheta, 2 * L - 1))
                + 1j * jax.random.normal(jax.random.PRNGKey(4), (ntheta, 2 * L - 1)))

    forward = np.asarray(utils._forward_ring_fft_full(
        jnp.asarray(signal), utils._spin_ring_analysis_tables(L, nside),
        L=L, nside=nside))
    backward = np.asarray(utils._inverse_ring_fft_complex(
        centered, utils._spin_ring_synthesis_tables(L, nside), L=L, nside=nside))

    m_index = np.arange(-(L - 1), L)
    for ring in {0, nside // 2, nside - 2, nside - 1, nside,
                 2 * nside, 3 * nside - 1, 3 * nside, ntheta - 1}:
        period = nphi[ring]
        row = signal[start[ring]:start[ring] + period]
        direct = (np.exp(-2j * np.pi * np.outer(m_index, np.arange(period)) / period)
                  @ row)
        scale = float(np.max(np.abs(direct)))
        np.testing.assert_allclose(forward[ring], direct, rtol=0.0, atol=1e-12 * scale)

        coefficients = np.asarray(centered[ring])
        synth = (np.exp(2j * np.pi * np.outer(np.arange(period), m_index) / period)
                 @ coefficients)
        scale = float(np.max(np.abs(synth)))
        np.testing.assert_allclose(backward[start[ring]:start[ring] + period], synth,
                                   rtol=0.0, atol=1e-12 * scale)


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
@pytest.mark.parametrize("nside", [16, 48, 64])
def test_spin2_march_synthesis_matches_the_route_it_replaces(nside):
    """The table-free synthesis march against the exact `flm_to_ftm` it replaces.

    The march is the default only where no Wigner-d slice can exist (`slice_declined`), which is far
    above every size a test runs at, so the seam is called directly here. Nside 48 is the interesting
    case: L = 144 makes `_spin_slice._windows` widths 64/64/16, and `_inverse_impl` assembles its
    result by concatenating the per-window blocks -- valid only because the windows tile orders
    0..L-1 with `lo == m0`, so a ragged final window is what would break silently.

    `ell < |spin|` is zeroed because the march's recurrence starts at `ell = max(m, spin)` and
    contributes nothing below the spin while the exact route does use those terms; every analysis
    output already satisfies that, which is why the convention difference never reaches a pipeline.
    Measured here with the default (plain-float32) accumulation and, in brackets, the compensated
    one: max rel 2.54e-06 [2.42e-06] at nside 16, 4.90e-06 [4.93e-06] at 48, 8.52e-06 [8.47e-06] at
    64, and 1.01e-05 for both at 128 (`.qwen/tmp/synth_march_rel.log`) — dropping the accumulation
    limb is not what limits this step.
    """
    from gmaster import _spin_march_pallas as march

    lmax = 3 * nside - 1
    L = lmax + 1
    rng = np.random.default_rng(11)
    flm = rng.normal(size=(L, 2 * L - 1)) + 1j * rng.normal(size=(L, 2 * L - 1))
    flm[:2] = 0.0                       # rows are ell: no sub-spin power
    jflm = jnp.asarray(flm)

    exact = utils._inverse_latitudinal(jflm, utils._stable_thetas(L, nside), L=L, spin=2,
                                       nside=nside, reality=False)
    got = march.inverse_latitudinal(jflm, L=L, spin=2, nside=nside)
    assert got.shape == exact.shape

    ref, new = np.asarray(exact), np.asarray(got)
    scale = float(np.max(np.abs(ref)))
    np.testing.assert_allclose(new, ref, rtol=0.0, atol=1e-4 * scale)
    # Column 0 is written by neither half of the assembly and the contract asks for a zero there.
    assert not np.any(new[:, 0])


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
@pytest.mark.parametrize("nside", [16, 32, 48, 160, 256, 512])
def test_spin2_march_analysis_writes_every_lane_it_returns(nside):
    """The marched analysis output is finite and equals the exact route lane for lane.

    `pl.pallas_call`'s `out_shape` buffer arrives with whatever the caching allocator left behind.
    The analysis march stores one row per order at `ell - m0`, so the `nstart - m0` lanes below
    `max(m, spin)` belong to no `ell` at all and nothing writes them unless the kernel zeroes them
    itself.  A driver-side mask over the whole slab is not a substitute: the masked-out orders feed
    nothing downstream, so the GPU is free to drop the select, and that is how a first-execution-only
    NaN reached a pipeline at nside 1024 spin 2 with the wide m-window (9,440,250 of 9,440,256 alm
    entries, `.qwen/tmp/nan_where.log`).

    What this test pins down, demonstrated in both directions:

    * The ragged geometries compile.  A window is ragged when `L = 3*nside` is not a multiple of the
      128-row m-window, and a head-zeroing block of `(mb, NC)` on a ragged `mb` of 48, 80, 96 or 112
      is not a power of two, which the Triton lowering rejects outright.  `c095321` shipped exactly
      that masked block, and walking the shipped `_march_windows` decomposition over the lattice
      (`.qwen/tmp/lattice_s29.py`) shows it could not compile the spin-2 analysis march at nside
      **16, 32, 80, 112, 160, 416 or 928** — while every power of two from 64 to 4096 decomposes into
      128-row windows and was clean.  That is why the 146-test suite never saw it, and why nside 160
      is in this list alongside the tiny cases: it is a production-sized odd footprint, not a unit-test
      size.
    * Every lane the routine returns is finite and matches the exact route, including the sub-spin
      wedge, so a nonzero leak into those lanes fails here.

    What it does *not* do, and was written believing it did: the poisoning does not reproduce in
    process.  On `96fcbd1`, which has no in-kernel zeroing at all, all five cases pass — and priming
    the pool with 24 buffers of the slab's own byte count does not change that, every wedge lane still
    arrives exactly zero at nside 256 and 512 (3070/3070 and 6142/6142,
    `.qwen/tmp/poison_prime_s29.log`).  The fault that motivated this test needs the pipeline's cold
    pool at nside 1024, and it stays evidenced by that log and the trial counts, not by this test.
    The dirt-freeing below is kept because a nonzero leak is still caught, not because it is known to
    provoke one.

    The `ftm` comes from the pipeline's own ring step rather than a random array because its column
    layout is `L + m` over `2L` columns, which is what makes the window slices line up.
    """
    from gmaster import _spin_march_pallas as march

    lmax = 3 * nside - 1
    L = lmax + 1
    rng = np.random.default_rng(5)
    maps = jnp.asarray(rng.normal(size=12 * nside ** 2) + 1j * rng.normal(size=12 * nside ** 2))
    ftm = utils._forward_s2fft_ftm(maps, utils._spin_ring_analysis_tables(L, nside),
                                   L=L, nside=nside, reality=False)

    dirt = [jnp.full(n, np.nan)
            for n in (1 << 23, 1 << 21, 1 << 19, 1 << 17, 1 << 15, 1 << 13)]
    for buf in dirt:
        buf.block_until_ready()
    dirt = None
    gc.collect()

    got = np.asarray(march.forward_latitudinal(ftm, L=L, spin=2, nside=nside))
    assert np.all(np.isfinite(got)), "marched analysis read unwritten slab lanes"

    # Explicit copy: `np.asarray` hands back a read-once view of the JAX buffer, and the zeroing
    # below would raise `ValueError: assignment destination is read-only`.
    ref = np.array(utils._forward_latitudinal(ftm, L=L, spin=2, nside=nside,
                                              reality=False, L_lower=0), copy=True)
    # Rows `ell < spin` are where an uninitialized lane would land, so they stay in the comparison.
    # They are also the one place the two routes differ by convention: the march's recurrence starts
    # at `max(m, spin)` and stores zeros below it, while the table route keeps its sub-spin terms.
    # The pipeline applies the march's convention to both routes in `_finish_forward_s2fft`, so the
    # reference is zeroed here rather than the tolerance being loosened.  Measured residual on the
    # fixed kernel over rows `ell >= 2`: max abs 9.07e-08 / 5.92e-08 / 4.53e-08 / 2.43e-08 at nside
    # 16 / 32 / 48 / 256 against scales of 2.00e-01 / 1.15e-01 / 6.81e-02 / 1.39e-02
    # (`.qwen/tmp/resid_s29.log`), so the tolerance below holds by more than an order of magnitude.
    ref[:2] = 0.0
    np.testing.assert_allclose(got, ref, rtol=0.0, atol=1e-4 * float(np.max(np.abs(ref))))


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
@pytest.mark.parametrize("nside", [32, 48])
def test_fused_slab_route_is_bit_identical_to_the_split_route(nside):
    """Fusing the polarised slab transform may change the dispatch count and nothing else.

    The win is 2.5-7.9x at Nside 64-256 (`.qwen/tmp/slab_fuse2.log`, `.qwen/tmp/slab_fuse_synth.log`)
    and the two routes differ only in how many jit boundaries the same body is cut at, so equality here
    is exact rather than tolerance-based.  Asserting the slab is under `_SLAB_FUSE_MAX_BYTES` pins the
    gate: moving a geometry onto a boundary the fusion was measured and rejected on would otherwise
    happen silently.
    """
    lmax = 3 * nside - 1
    L = lmax + 1
    npix = 12 * nside ** 2
    rng = np.random.default_rng(7)
    maps = jnp.asarray(rng.normal(size=(2, npix)) * 1e-3)
    ell, order = utils._ell_order_arrays(lmax)

    a_slab, s_slab = utils._spin_slabs(L, 2, nside=nside)
    assert a_slab is not None and s_slab is not None
    assert utils._slab_bytes(a_slab) <= utils._SLAB_FUSE_MAX_BYTES

    tables = utils._spin_ring_analysis_tables(L, nside, maps.device)
    got = utils._map2alm_once_slab_fused(maps, tables, ell, order, spin=2, nside=nside,
                                         L=L, L_work=L, slab=a_slab)
    ref = utils._map2alm_once_slab_body(maps, tables, ell, order, spin=2, nside=nside,
                                        L=L, L_work=L, slab=a_slab)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))

    alm = jnp.asarray(_random_alms(rng, 2, utils.NmtAlmInfo(lmax), 2))
    s_tables = utils._spin_ring_synthesis_tables(L, nside, alm.device)
    got = utils._alm2map_core_slab_fused(alm, s_slab, s_tables, spin=2, nside=nside,
                                         L=L, L_work=L)
    ref = utils._alm2map_core_slab_body(alm, s_slab, s_tables, spin=2, nside=nside,
                                        L=L, L_work=L)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
@pytest.mark.parametrize("nside", [128, 256])
def test_traced_refinement_route_is_bit_identical_to_op_by_op(nside):
    """Tracing the scalar refinement loop may change the dispatch count and nothing else.

    The two cores are the same seven passes cut at a different number of jit
    boundaries, so equality is exact rather than tolerance-based: measured
    `rel = 0.000e+00` at Nside 256 (`.qwen/tmp/traceidentity_s32.py`) for a 9.4 %
    end-to-end win (`.qwen/tmp/tracepipe_s32.log`).  Asserting `L <= _PALLAS_TRACED_MAX_L`
    pins the gate from both sides -- Nside 512 was measured 17 % *slower* on the traced
    route and must not be moved onto it, and Nside 256 must not quietly fall back to
    op-by-op because the Legendre band stopped being hoistable.
    """
    lmax = 3 * nside - 1
    L = lmax + 1
    npix = 12 * nside ** 2
    assert L <= utils._PALLAS_TRACED_MAX_L
    rng = np.random.default_rng(11)
    maps = jnp.asarray(rng.normal(size=(1, npix)))
    ell, order = utils._ell_order_arrays(lmax)

    # The gate's own condition: the band must be concrete device buffers before a
    # program is allowed to read it.
    assert utils._trace_route_ready(nside, L)

    ref = np.asarray(utils._map2alm_core_pallas_eager(
        maps, ell, order, nside=nside, L=L, L_work=L, n_iter=3, spin=0))
    got = np.asarray(utils._map2alm_core_pallas_traced(
        maps, ell, order, nside=nside, L=L, L_work=L, n_iter=3, spin=0))
    np.testing.assert_array_equal(got, ref)


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
def test_band_build_declines_inside_a_trace_instead_of_raising():
    """A geometry first touched under `jax.grad` must transform, not die on `block_until_ready`.

    `_theta_matrix._band` drains its build caches with `slab.block_until_ready()` so
    that a cached executable cannot pin the block it produced; under a trace that
    raises `AttributeError: 'block_until_ready' is not available on traced array` and
    takes the whole call down (`.qwen/tmp/s24_256_0.log`, the reason the traced gate sat
    at Nside 128 for five sessions).  The builders now decline, the transform takes the
    fused kernel, and the gradient still comes back finite.
    """
    nside = 64
    lmax = 3 * nside - 1
    L = lmax + 1
    npix = 12 * nside ** 2
    rng = np.random.default_rng(13)
    ell, order = utils._ell_order_arrays(lmax)
    _theta_matrix.release()

    def scalar(maps_in):
        return jnp.sum(jnp.abs(utils._map2alm_core_pallas(
            maps_in[None, :], ell, order, nside=nside, L=L, L_work=L, n_iter=0,
            spin=0)))

    grad = np.asarray(jax.grad(scalar)(jnp.asarray(rng.normal(size=npix))))
    assert np.all(np.isfinite(grad))
    assert float(np.max(np.abs(grad))) > 0.0


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
def test_no_tracer_can_enter_the_table_caches():
    """Tracing the route dispatcher must neither raise nor leave a tracer in a cache.

    `_trace_route_ready` builds the Legendre band and its synthesis layout at top
    level, so it is also reachable *during* somebody else's trace (`jax.jit`,
    `jax.eval_shape`, `jax.grad`).  Both builders then have to decline: a cached
    tracer is the `UnexpectedTracerError` of Session 5j, and an uncached one means
    the concrete-buffer drain raises `AttributeError` inside the caller's trace --
    which is exactly how a first version of the hoist failed, on a path the pipeline
    itself never walks.
    """
    nside = 64
    L = 3 * nside
    _theta_matrix.release()

    out = jax.jit(lambda x: (utils._trace_route_ready(nside, L), x * 2.0)[1])(
        jnp.ones(4))
    np.testing.assert_allclose(np.asarray(out), 2.0)

    # Whatever the trace did, the same geometry must still build cleanly at top
    # level -- a decline that leaves the caches unusable is as bad as a crash.
    assert _theta_matrix.warm(nside, L)
    cached = [slab for groups in _theta_matrix._BAND_CACHE.values()
              for group in groups for slab in group]
    cached += [slab for groups, _ in _theta_matrix._SYNTH_CACHE.values()
               for group in groups for slab in group]
    assert cached, "the geometry should have been built eagerly by the dispatcher"
    for slab in cached:
        assert hasattr(slab, "block_until_ready")


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
@pytest.mark.parametrize("nside", [64, 128])
def test_shared_band_pair_is_bit_identical_to_two_transforms(nside):
    """Two spin-0 analyses over one band must be the two analyses, exactly.

    The pair runs both Richardson recursions in lockstep so each latitudinal
    contraction serves two maps for one read of the Legendre band.  Every operand
    of either transform is unchanged -- same ring FFT per map, same band, same
    accumulation order per channel -- so the assertion is equality, not tolerance,
    and it is the same equality measured on the isolated programs at Nside
    128-1024 (`.qwen/tmp/pairsettle_s33.log`, rel 0.00e+00) and end to end through
    a field and a coupling matrix (`.qwen/tmp/pairverify_s33.log`).  Asserting the
    route is engaged first keeps a silent decline from making the test vacuous.
    """
    lmax = 3 * nside - 1
    L = lmax + 1
    npix = 12 * nside ** 2
    assert utils._shared_band_route(nside, L)[0]
    minfo = nmt.NmtMapInfo(None, (npix,))
    ainfo = nmt.NmtAlmInfo(lmax)
    rng = np.random.default_rng(23)
    maps_a = jnp.asarray(rng.normal(size=(1, npix)))
    maps_b = jnp.asarray(rng.normal(size=(1, npix)))

    pair = utils.map2alm_pair(maps_a, maps_b, minfo, ainfo, n_iter=3)
    assert pair is not None
    for maps, got in ((maps_a, pair[0]), (maps_b, pair[1])):
        ref = np.asarray(nmt.map2alm(maps, 0, minfo, ainfo, n_iter=3))
        np.testing.assert_array_equal(np.asarray(got), ref)


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
def test_shared_band_pair_declines_when_no_route_serves(monkeypatch):
    """The pair follows the route that serves a single transform, and declines with it.

    With no band the pair rides the marched row (`_march_pair_route`) rather than
    giving up -- that is the whole point of pairing on the band-refused sizes -- so
    refusing the budget alone no longer refuses the pair.  Both routes have to be
    shut off for `map2alm_pair` to return None, which is the fallback contract: the
    field constructor treats None as "do what you did before" and makes two
    `map2alm` calls.
    """
    nside = 64
    L = 3 * nside
    npix = 12 * nside ** 2
    minfo = nmt.NmtMapInfo(None, (npix,))
    ainfo = nmt.NmtAlmInfo(L - 1)
    maps = jnp.asarray(np.zeros((1, npix)))
    assert utils.map2alm_pair(maps, maps, minfo, ainfo, n_iter=1) is not None
    monkeypatch.setattr(utils, "_MATRIX_BAND_BUDGET", 0)
    monkeypatch.setenv("GMASTER_SPIN0_MARCH", "0")
    assert utils._shared_band_route(nside, L) == (False, False)
    assert utils._march_pair_route(nside, L) is False
    assert utils.map2alm_pair(maps, maps, minfo, ainfo, n_iter=1) is None


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
@pytest.mark.parametrize("nside", [64, 128])
def test_marched_pair_shares_one_recurrence(monkeypatch, nside):
    """Where there is no band, two maps share the marched Legendre row.

    The march generates its row inside the kernel, so a second right-hand side adds
    an emit contraction to a row that is being computed anyway -- measured 0.752 of
    two calls at Nside 2048, where this is the live route, and 0.489 at Nside 512
    with the march forced (`.qwen/tmp/marchpair_2048.log`, `.qwen/tmp/marchpair_512.log`).
    Unlike the band pairing this cannot be bit-identical: widening the emit block from
    4 to 8 channels reassociates the theta reduction.  The bar below is that
    reassociation, measured at 3.9e-07 on this geometry at Nside 64 and 4.4e-06 at 256
    (`.qwen/tmp/marchpairwire_s34.log`), and the paired route's error against the fp64
    band is the shipped march's own error to the fifth digit.
    """
    monkeypatch.setenv("GMASTER_SPIN0_MARCH", "1")
    lmax = 3 * nside - 1
    L = lmax + 1
    npix = 12 * nside ** 2
    assert utils._shared_band_route(nside, L) == (False, False)
    assert utils._march_pair_route(nside, L) is True
    minfo = nmt.NmtMapInfo(None, (npix,))
    ainfo = nmt.NmtAlmInfo(lmax)
    rng = np.random.default_rng(29)
    maps_a = jnp.asarray(rng.normal(size=(1, npix)))
    maps_b = jnp.asarray(rng.normal(size=(1, npix)))

    pair = utils.map2alm_pair(maps_a, maps_b, minfo, ainfo, n_iter=3)
    assert pair is not None
    for maps, got in ((maps_a, pair[0]), (maps_b, pair[1])):
        ref = np.asarray(nmt.map2alm(maps, 0, minfo, ainfo, n_iter=3))
        np.testing.assert_allclose(
            np.asarray(got), ref, rtol=0.0, atol=1e-4 * float(np.max(np.abs(ref))))


def test_paired_fold_footprint_grows_with_the_geometry():
    """The pair gate's byte estimate is monotone, so it can only refuse the large sizes.

    The estimate is a sum of `L * ntheta`-scale terms, and a formula slip that made it fall
    with Nside would refuse the small geometries that the pairing actually wins on.
    """
    sizes = [utils._spin_march.fold_pair_bytes(n, 3 * n)
             for n in (256, 512, 1024, 2048, 3072, 4096)]
    assert all(a < b for a, b in zip(sizes, sizes[1:]))


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
@pytest.mark.parametrize("nside, fits", [(512, True), (1024, True), (2048, True),
                                         (3072, False), (4096, False)])
def test_paired_fold_gate_sides_at_the_measured_sizes(monkeypatch, nside, fits):
    """The pairing is offered where it has been measured to run and withheld where it has not.

    Nside 2048 is the end-to-end win (4558 -> 3961 ms, GPU peak 4.3 GiB,
    `.qwen/tmp/pairrun_s34.log`); Nside 4096 is a measured crash --
    `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 8.15GiB` inside this box's
    71.2 GiB pool, while the same geometry as two separate calls runs in 33.62 s at a
    16.3 GiB peak (`.qwen/tmp/pairrun_4096_s34.log`).  A refusal returns the caller to those
    two calls, so the gate costs nothing but the pairing.

    Both measurements were made with the azimuthal stage in float64, which is the shipped
    coupling, and the estimate is that footprint regardless of `set_ring_precision`, so these
    verdicts are the same in an fp32-ring session (where they are conservative).
    """
    monkeypatch.setenv("GMASTER_SPIN0_MARCH", "1")
    assert utils._spin_march.fold_pair_fits(nside, 3 * nside) is fits
    assert utils._march_pair_route(nside, 3 * nside) is fits


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
def test_latitudinal_adjoint_hands_back_the_operand_dtype():
    """The analysis VJP's cotangent has the operand's element type, not the kernel's.

    The latitudinal kernel accumulates in float64 whatever it is handed and its transpose
    runs the float64 synthesis kernel, so without a cast at the adjoint boundary a
    `set_ring_precision("fp32")` session hands the azimuthal stage a complex128 cotangent
    for a complex64 primal and reverse mode dies inside the chirp-Z multiply
    (`lax.mul requires arguments to have the same dtypes`).  That is what
    `test_fused_scalar_transform_gradients_match_generic_jax` hits under
    `--gm-ring-precision fp32`; this pins the boundary itself.
    """
    original = utils.nmt_params.ring_precision
    try:
        utils.set_ring_precision("fp32")
        nside = 16
        L = 3 * nside
        theta, weights, phase = utils._pallas_parameters(L, nside)
        ftm = jnp.ones((4 * nside - 1, L), dtype=jnp.complex64)

        def stage(x):
            return utils.scalar_forward_latitudinal(
                x, theta, weights, phase, L=L,
                block_size=utils._pallas_block_size(nside))

        out, vjp = jax.vjp(stage, ftm)
        assert out.dtype == jnp.complex128
        cotangent, = vjp(jnp.ones_like(out))
        assert cotangent.dtype == ftm.dtype
    finally:
        utils.set_ring_precision(original)


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
def test_paired_fold_declines_when_the_gate_refuses(monkeypatch):
    """A refused pair leaves the caller the two-call path rather than an allocation failure."""
    monkeypatch.setattr(utils._spin_march, "_PAIR_POOL_FACTOR", 10 ** 9)
    monkeypatch.setenv("GMASTER_SPIN0_MARCH", "1")
    nside = 64
    L = 3 * nside
    npix = 12 * nside ** 2
    assert utils._spin_march.fold_requested(nside, L) is True
    assert utils._march_pair_route(nside, L) is False
    minfo = nmt.NmtMapInfo(None, (npix,))
    ainfo = nmt.NmtAlmInfo(L - 1)
    maps = jnp.asarray(np.zeros((1, npix)))
    assert utils.map2alm_pair(maps, maps, minfo, ainfo, n_iter=1) is None


@pytest.mark.parametrize("nside", [512, 1024, 2048, 4096, 8192])
def test_synthesis_tile_is_per_spin_and_still_powers_of_two(nside):
    """The synthesis launch geometry agrees with the chunk the kernel is compiled against.

    `_inverse_fold_impl` derives `ntile` and `npad` from the tile width and `_call_synth` derives
    the kernel's `chunk` and its `out_shape` from the same helper, so a divergence between the two
    would show up as an out-of-bounds or a dropped theta row at a size no in-repo test can reach
    (the fold only serves Nside >= 2048, where a GPU run costs minutes).  The power-of-two assertion
    is the Triton lowering rule: every op's array must be a power of two, and the synthesis
    `out_shape` is `(mb, ntile, chunk, 4)`, so `ntile` has to stay one -- the bug class that shipped
    once as `Encountered an array of shape (96, 4)` from a ragged m-window.

    The width itself is measured, not guessed: spin 0's synthesis wants 1024 at Nside >= 2048
    (287.4 -> 243.7 ms at 2048, 1885.1 -> 1745.8 ms at 4096, bit-identical maps) while spin 2 wants
    512 (453.7 ms at 2048 against 508.1 at 256 and 524.6 at 1024).  See the `_ST0` note in
    `gmaster/_spin_march_pallas.py`.
    """
    from gmaster import _spin_march_pallas as smp

    ntheta = 4 * nside - 1
    north = (ntheta + 1) // 2
    st = smp._synth_tile(0, north)
    ntile = -(-north // st)
    assert st * ntile >= north > st * (ntile - 1)
    assert ntile & (ntile - 1) == 0, f"ntile={ntile} is not a power of two"
    assert st == (smp._ST0 if north >= 4 * smp._ST0 else smp._ST)
    # Spin 2's generic route takes the shipped width at every size, including the ones where the
    # spin-0 fold has moved.
    assert smp._synth_tile(2, ntheta) == smp._ST


@pytest.mark.parametrize("nside", [512, 1024, 2048, 4096, 8192])
def test_synth_order_window_fills_the_launch_ceiling(nside):
    """The synthesis order window is grown toward `_MARCH_GRID_CAP` and never past it.

    Every win and loss of the marched routes' order window turned out to be about programs per
    launch, not about the width (`_march_windows`: wins at <= 2048 programs, losses at 4096).  The
    folded spin-0 synthesis is the one route with room to fill the ceiling -- `_ST0` leaves it 4
    theta tiles at Nside 2048 and 8 at 4096 while the analysis has 16 and 32 -- and doing so is worth
    5% on the largest cell (spin-0 `alm2map` at 4096: 1740.0 ms / 1.14x -> 1650.3 ms / 1.20x,
    `.qwen/tmp/s29y.log`, `.qwen/tmp/swin_s29.log`).

    The second half is the trap this guard exists for: raising the *global* `GMASTER_MARCH_M_BLOCK`
    does not widen the analysis, it pushes `mb * ntile` over the ceiling, which silently drops the
    route to `_M_BLOCK` = 64 and costs 1.37x there (the 256 arm at Nside 2048 read 405.3 ms, the
    64-window value, not a 256-window one).  So the synthesis width must be independent of the
    analysis windows, and this asserts that at every size.
    """
    from gmaster import _spin_march_pallas as smp
    from gmaster import _spin_slice as ss

    L = 3 * nside - 1
    north = (4 * nside - 1 + 1) // 2
    ntile = -(-north // smp._synth_tile(0, north))
    win = smp._synth_windows(L, ntile, 0)
    if ntile < 4:
        # Below four theta tiles the wider window loses (Nside 512: 4.6 -> 4.9 ms), so the route
        # falls back to the shipped analysis rule verbatim rather than filling the ceiling.
        assert win == smp._march_windows(L, ntile, 0)
        return
    widths = {m1 - m0 for m0, m1, _ in win}
    assert len(widths) == 2 or len(widths) == 1      # full windows plus at most one truncated tail
    mb = max(widths)
    assert mb & (mb - 1) == 0, f"mb={mb} is not a power of two (Triton lowering)"
    assert mb >= ss._MARCH_M_BLOCK
    assert mb * ntile <= smp._MARCH_GRID_CAP
    # As close to the ceiling as a power of two gets, unless the cap on the width itself binds.
    assert mb == ss._MARCH_M_SYNTH0_MAX or 2 * mb * ntile > smp._MARCH_GRID_CAP
    # Coverage: every order 0..L-1 in exactly one window.
    assert [m0 for m0, _, _ in win] == list(range(0, L, mb))
    assert win[-1][1] == L

    # Spin 2 synthesis is the analysis rule verbatim, at every size.
    assert smp._synth_windows(L, ntile, 2) == smp._march_windows(L, ntile, 2)

    # The analysis windows do not move when the synthesis ceiling moves -- the failure mode that made
    # an earlier "256 is worse" measurement actually measure 64.
    before = smp._march_windows(L, ntile, 0)
    orig = ss._MARCH_M_SYNTH0_MAX
    try:
        ss._MARCH_M_SYNTH0_MAX = 4096
        assert smp._march_windows(L, ntile, 0) == before
    finally:
        ss._MARCH_M_SYNTH0_MAX = orig


def test_pool_headroom_survives_a_backend_without_allocator_stats(monkeypatch):
    """A backend that returns `None` from `memory_stats()` must mean "unbounded", not crash.

    `_pool_headroom` guards the Wigner-d layout build and documents "+inf when the device won't
    say", but it only caught a *raising* backend.  `jax.local_devices()[0].memory_stats()` returns
    `None` on the CPU backend instead of raising, so the very next line (`stats.get("pool_bytes")`)
    raised `AttributeError: 'NoneType' object has no attribute 'get'` out of `slabs_for`.
    """
    from gmaster import _spin_slice as ss

    class _Silent:
        def memory_stats(self):
            return None

    class _Raises:
        def memory_stats(self):
            raise RuntimeError("no statistics here")

    for device in (_Silent(), _Raises()):
        monkeypatch.setattr(jax, "local_devices", lambda: [device])
        assert ss._pool_headroom() == float("inf")

