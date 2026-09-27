"""Spherical harmonic transforms: agreement with NaMaster and equivalence of GMaster's internal routes.

Unless marked ``march_v2``, tests run on the exact fp64 routes (see ``conftest.py``) and are held
to NaMaster at ~1e-13.  Tests marked ``march_v2`` exercise the default float32 CUDA march, which
agrees with NaMaster to ~1e-6.  Tests that compare two routes performing the same arithmetic in
a different program structure assert bit-for-bit equality; the others state their tolerance.
"""

import gc

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from s2fft.utils import healpix_ffts, quadrature_jax

jax.config.update("jax_enable_x64", True)

import gmaster as nmt
from gmaster import utils
from gmaster._sht import theta_matrix as _theta_matrix
from gmaster._sht.cuda_gpu import on_cuda_gpu
from gmaster import _config
from gmaster._sht import healpix, rings


_HAS_NVIDIA_GPU = on_cuda_gpu()


def _random_alms(rng, nmaps, ainfo, spin):
    """Random healpy-ordered alms: real at m = 0 and zero below ell = spin."""
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
    """HEALPix alm2map and map2alm (with and without iterations) match NaMaster for spin 0 and 2."""
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
    """NmtMapInfo/NmtAlmInfo report NaMaster's lmax, sizes and offsets and compare by value."""
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
    """The transform and pseudo-inverse functions accept NaMaster's keyword names."""
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
    """Spin-2 transforms at the default band limit lmax = 3 Nside - 1 match NaMaster."""
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
    """Maps or alms with the wrong number of components for the spin are rejected."""
    minfo = nmt.NmtMapInfo(None, (12 * 4**2,))
    ainfo = nmt.NmtAlmInfo(7)
    with pytest.raises(ValueError, match="wrong shape"):
        nmt.map2alm(np.ones((2, minfo.npix)), 0, minfo, ainfo, n_iter=0)
    with pytest.raises(ValueError, match="wrong shape"):
        nmt.alm2map(np.ones((1, ainfo.nelem)), 2, minfo, ainfo)


@pytest.mark.parametrize("L,L_work", [(1, 1), (4, 4), (4, 7)])
def test_gathered_alm_unpack_matches_healpy_layout(L, L_work):
    """Unpacking healpy-ordered alms into the (ell, m) grid fills negative m by the reality
    condition, including padding to a larger working band.
    """
    ainfo = nmt.NmtAlmInfo(L - 1)
    rng = np.random.default_rng(L_work)
    alm = rng.normal(size=ainfo.nelem) + 1j * rng.normal(size=ainfo.nelem)
    expected = np.zeros((L_work, 2 * L_work - 1), dtype=np.complex128)
    expected[ainfo._ell, L_work - 1 + ainfo._m] = alm
    expected[ainfo._ell, L_work - 1 - ainfo._m] = np.where(
        ainfo._m > 0, (-1) ** ainfo._m * np.conj(alm), alm
    )
    np.testing.assert_array_equal(healpix._unpack_real(alm, L, L_work), expected)
    b_alm = rng.normal(size=ainfo.nelem) + 1j * rng.normal(size=ainfo.nelem)
    b_expected = np.zeros_like(expected)
    b_expected[ainfo._ell, L_work - 1 + ainfo._m] = b_alm
    b_expected[ainfo._ell, L_work - 1 - ainfo._m] = np.where(
        ainfo._m > 0, (-1) ** ainfo._m * np.conj(b_alm), b_alm
    )
    np.testing.assert_array_equal(
        healpix._unpack_spin(np.stack((alm, b_alm)), L, L_work),
        -(expected + 1j * b_expected),
    )


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
def test_fused_scalar_transforms_match_namaster_and_generic_jax():
    """The fused Pallas spin-0 transforms ('jax') match NaMaster and the generic s2fft route
    ('jax-generic') at Nside 64.
    """
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
    """The double-fp32 analysis ('jax-dfp32') matches the fp64 route and NaMaster to 3e-11.

    Only the analysis uses double-fp32; the synthesis (including the residual synthesis of
    the iterations) stays fp64.  The Pallas path is used from L = 128, and double-fp32 holds
    3e-11 up to L ~ 192 at Nside 64, so L = 160 is tested.
    """
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
        # The synthesis does not depend on this calculator choice.
        dfp32_map = nmt.alm2map(alms, 0, minfo, ainfo)
        np.testing.assert_allclose(dfp32_map, reference_map, atol=3e-11)
    finally:
        nmt.set_sht_calculator(original)


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
def test_fused_scalar_transform_gradients_match_generic_jax():
    """Gradients through the fused spin-0 transforms match those through the generic s2fft route."""
    nside = 16
    L = 3 * nside
    minfo = nmt.NmtMapInfo(None, (12 * nside**2,))
    ainfo = nmt.NmtAlmInfo(L - 1)
    key = jax.random.key(42)
    maps = jax.random.normal(key, (1, minfo.npix), dtype=jnp.float64)
    alms = jnp.asarray(_random_alms(np.random.default_rng(42), 1, ainfo, 0))

    def fused_analysis_loss(values):
        transformed = healpix._map2alm_core_pallas(
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
        transformed = healpix._alm2map_core_pallas(
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
    """The multi-GPU calculator gives the single-GPU transforms."""
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
    """The multi-GPU calculator gives the single-GPU fused spin-0 transforms at Nside 64."""
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
def test_matrix_theta_stage_matches_fused_kernel(monkeypatch):
    """The 'jax-matrix' analysis matches the fused kernel and NaMaster, and has ell >= m support.

    'jax-matrix' replaces the in-kernel Legendre recurrence with a cached m-banded matrix
    generated by the same recurrence, so it must agree with the fused kernel to fp64 rounding.
    """
    # The v2 CUDA march is the default at this size; this test covers the exact band /
    # Pallas route, so select that route.
    monkeypatch.setenv("GMASTER_MARCH_V2", "0")
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
        # The synthesis does not depend on the analysis calculator.
        matrix_map = nmt.alm2map(alms, 0, minfo, ainfo)
        np.testing.assert_allclose(matrix_map, reference_map, atol=1e-12)
        np.testing.assert_allclose(matrix_map, fused_map, atol=1e-14)

        L = lmax + 1
        theta = healpix._stable_thetas(L, nside)
        ftm = healpix._forward_healpix_fft(jnp.asarray(reference_map), L=L,
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
    """The positive-m ring synthesis equals the mirrored-window synthesis.

    `_inverse_ring_fft` mirrors the block into 2L coefficients and chirp-Z transforms all
    of it; `_inverse_ring_fft_herm` uses the Hermitian identity `2 Re P - Re F_0` and a
    chirp-Z of length `L + width - 1` instead of `2L - 1 + width`.  Polar rows are read from
    a concatenated slot buffer and the equatorial belt uses one plain inverse FFT.
    `L = 40` at Nside 32 is wider than the polar rings' `nphi`, so the cap chirp-Z must
    alias; `L = 71 > 4*nside` disables the belt shortcut and must still match.
    """
    positive = (
        jax.random.normal(jax.random.PRNGKey(11), (4 * nside - 1, L))
        + 1j * jax.random.normal(jax.random.PRNGKey(12), (4 * nside - 1, L))
    )
    reference = rings._inverse_ring_fft(positive, L=L, nside=nside)
    got = rings._inverse_ring_fft_herm(
        positive, rings._ring_synthesis_tables(L, nside), L=L, nside=nside
    )
    np.testing.assert_allclose(got, reference, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("nside, L", [(16, 47), (32, 95), (16, 64), (16, 71), (32, 40)])
def test_ring_analysis_matches_direct_dft_ring_by_ring(nside, L):
    """Every ring's m block equals an explicit `sum_p x_p exp(-2i pi p m / nphi)`.

    The azimuthal stage has two routes: the equatorial belt (`nphi = 4*nside`) is one plain
    FFT of a contiguous reshape, while the polar rings use a Bluestein chirp-Z at the
    map-wide width.  Both are checked against the definition on cap, boundary and belt
    rings.  `L = 64` is the boundary `L == 4*nside`; at `L = 71 > 4*nside` the belt shortcut
    is disabled and every ring goes through the chirp-Z.  `L = 40` at Nside 32 is wider than
    the polar rings' `nphi`, so their chirp-Z must alias.
    """
    nphi, start, _, _, _ = rings._ring_czt_constants_numpy(L, nside)
    map_flat = jnp.asarray(
        np.random.default_rng(4).normal(size=12 * nside ** 2)
    )
    got = np.asarray(
        rings._forward_ring_fft_positive(
            map_flat, rings._ring_analysis_tables(L, nside), L=L, nside=nside
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
    nphi, start, _, _, _ = rings._ring_czt_constants_numpy(L, nside)
    npix = 12 * nside ** 2
    ntheta = 4 * nside - 1
    rng = np.random.default_rng(7)
    signal = np.asarray(rng.normal(size=npix) + 1j * rng.normal(size=npix))
    centered = (jax.random.normal(jax.random.PRNGKey(3), (ntheta, 2 * L - 1))
                + 1j * jax.random.normal(jax.random.PRNGKey(4), (ntheta, 2 * L - 1)))

    forward = np.asarray(rings._forward_ring_fft_full(
        jnp.asarray(signal), rings._spin_ring_analysis_tables(L, nside),
        L=L, nside=nside))
    backward = np.asarray(rings._inverse_ring_fft_complex(
        centered, rings._spin_ring_synthesis_tables(L, nside), L=L, nside=nside))

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
    """The table-free spin-2 synthesis march matches the exact `flm_to_ftm` route.

    The march is only the default at sizes where no Wigner-d slice fits (`slice_declined`),
    far above test sizes, so it is called directly.  Nside 48 (L = 144) splits the orders
    into windows of 64/64/16, which checks that a ragged final window is assembled correctly.

    Rows `ell < |spin|` are zeroed because the march's recurrence starts at
    `ell = max(m, spin)`, while the exact route uses those terms; analysis outputs never
    contain them.  The march is float32: the measured relative error is 2.5e-6 to 1e-5 for
    Nside 16-128, so 1e-4 of scale is a safe tolerance.
    """
    from gmaster._sht import spin_march as march

    lmax = 3 * nside - 1
    L = lmax + 1
    rng = np.random.default_rng(11)
    flm = rng.normal(size=(L, 2 * L - 1)) + 1j * rng.normal(size=(L, 2 * L - 1))
    flm[:2] = 0.0                       # rows are ell: no sub-spin power
    jflm = jnp.asarray(flm)

    exact = healpix._inverse_latitudinal(jflm, healpix._stable_thetas(L, nside), L=L, spin=2,
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
    """The spin-2 analysis march writes every output lane and matches the exact route.

    The Pallas output buffer is uninitialised, and the march stores one row per order at
    `ell - m0`, so the `nstart - m0` lanes below `max(m, spin)` must be zeroed by the kernel
    itself; otherwise stale memory (e.g. NaN) can leak into the alms.  Masking the slab after
    the kernel is not a substitute: those orders feed nothing downstream, so the compiler may
    eliminate the mask.

    The test checks two things:

    * Ragged geometries compile.  A window is ragged when `L = 3*nside` is not a multiple of
      the 128-row m-window, and a zeroing block whose row count is not a power of two is
      rejected by the Triton lowering.  Nside 16, 32 and 160 give ragged windows whose row
      counts are not powers of two (160 is a realistic production size); the other sizes
      give power-of-two windows.
    * Every returned lane is finite and matches the exact route, including the sub-spin
      wedge, so a nonzero leak into those lanes fails.

    The pool is first filled with freed NaN buffers so that an unwritten lane is likely to
    read NaN, but this is not guaranteed to reproduce the failure.  The `ftm` comes from the
    real ring step because its column layout (`L + m` over `2L` columns) is what the window
    slices assume.
    """
    from gmaster._sht import spin_march as march

    lmax = 3 * nside - 1
    L = lmax + 1
    rng = np.random.default_rng(5)
    maps = jnp.asarray(rng.normal(size=12 * nside ** 2) + 1j * rng.normal(size=12 * nside ** 2))
    ftm = healpix._forward_s2fft_ftm(maps, rings._spin_ring_analysis_tables(L, nside),
                                   L=L, nside=nside, reality=False)

    dirt = [jnp.full(n, np.nan)
            for n in (1 << 23, 1 << 21, 1 << 19, 1 << 17, 1 << 15, 1 << 13)]
    for buf in dirt:
        buf.block_until_ready()
    dirt = None
    gc.collect()

    got = np.asarray(march.forward_latitudinal(ftm, L=L, spin=2, nside=nside))
    assert np.all(np.isfinite(got)), "marched analysis read unwritten slab lanes"

    # Explicit copy: `np.asarray` returns a read-only view of the JAX buffer, which the
    # zeroing below would otherwise fail to write.
    ref = np.array(healpix._forward_latitudinal(ftm, L=L, spin=2, nside=nside,
                                              reality=False, L_lower=0), copy=True)
    # Rows `ell < spin` are where an unwritten lane would appear, so they stay in the comparison.
    # They are also where the two routes differ by convention: the march stores zeros below
    # `max(m, spin)`, while the table route keeps sub-spin terms.  The pipeline applies the
    # march's convention to both routes (`_finish_forward_s2fft`), so the reference is zeroed
    # here.  Over rows `ell >= 2` the residual is ~5e-7 of scale at Nside 16 and ~2e-6 at
    # Nside 256, more than an order of magnitude inside the 1e-4 tolerance.
    ref[:2] = 0.0
    np.testing.assert_allclose(got, ref, rtol=0.0, atol=1e-4 * float(np.max(np.abs(ref))))


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
@pytest.mark.parametrize("nside", [32, 48])
def test_fused_slab_route_is_bit_identical_to_the_split_route(nside):
    """The fused spin-2 slab transform is bit-identical to the unfused one.

    Both routes run the same operations and differ only in how many jit boundaries they
    are split into, so equality is exact.  The slab must be below `_SLAB_FUSE_MAX_BYTES`
    for fusion to be used; asserting it guards against the tested sizes silently leaving
    the fused route.
    """
    lmax = 3 * nside - 1
    L = lmax + 1
    npix = 12 * nside ** 2
    rng = np.random.default_rng(7)
    maps = jnp.asarray(rng.normal(size=(2, npix)) * 1e-3)
    ell, order = healpix._ell_order_arrays(lmax)

    a_slab, s_slab = healpix._spin_slabs(L, 2, nside=nside)
    assert a_slab is not None and s_slab is not None
    assert healpix._slab_bytes(a_slab) <= healpix._SLAB_FUSE_MAX_BYTES

    tables = rings._spin_ring_analysis_tables(L, nside, maps.device)
    got = healpix._map2alm_once_slab_fused(maps, tables, ell, order, spin=2, nside=nside,
                                         L=L, L_work=L, slab=a_slab)
    ref = healpix._map2alm_once_slab_body(maps, tables, ell, order, spin=2, nside=nside,
                                        L=L, L_work=L, slab=a_slab)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))

    alm = jnp.asarray(_random_alms(rng, 2, utils.NmtAlmInfo(lmax), 2))
    s_tables = rings._spin_ring_synthesis_tables(L, nside, alm.device)
    got = healpix._alm2map_core_slab_fused(alm, s_slab, s_tables, spin=2, nside=nside,
                                         L=L, L_work=L)
    ref = healpix._alm2map_core_slab_body(alm, s_slab, s_tables, spin=2, nside=nside,
                                        L=L, L_work=L)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
@pytest.mark.parametrize("nside", [128, 256])
def test_traced_refinement_route_is_bit_identical_to_op_by_op(nside):
    """The traced spin-0 refinement loop is bit-identical to the op-by-op loop.

    Both run the same passes, split at a different number of jit boundaries, so equality is
    exact.  The traced route is used for `L <= _PALLAS_TRACED_MAX_L` (up to Nside 256; it is
    slower at Nside 512), and requires the Legendre band to be already built on the device.
    """
    lmax = 3 * nside - 1
    L = lmax + 1
    npix = 12 * nside ** 2
    assert L <= healpix._PALLAS_TRACED_MAX_L
    rng = np.random.default_rng(11)
    maps = jnp.asarray(rng.normal(size=(1, npix)))
    ell, order = healpix._ell_order_arrays(lmax)

    # The gate's own condition: the band must be concrete device buffers before a
    # program is allowed to read it.
    assert healpix._trace_route_ready(nside, L)

    ref = np.asarray(healpix._map2alm_core_pallas_eager(
        maps, ell, order, nside=nside, L=L, L_work=L, n_iter=3, spin=0))
    got = np.asarray(healpix._map2alm_core_pallas_traced(
        maps, ell, order, nside=nside, L=L, L_work=L, n_iter=3, spin=0))
    np.testing.assert_array_equal(got, ref)


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
def test_band_build_declines_inside_a_trace_instead_of_raising(monkeypatch):
    """A geometry first used under `jax.grad` must still transform, with a finite gradient.

    `_theta_matrix._band` calls `block_until_ready()` on the tables it builds, which is not
    available on traced arrays.  Under a trace the band builders must decline, so that the
    transform falls back to the fused kernel instead of raising.
    """
    # The v2 CUDA march is the default at this size; this test covers the exact band /
    # Pallas route, so select that route.
    monkeypatch.setenv("GMASTER_MARCH_V2", "0")
    nside = 64
    lmax = 3 * nside - 1
    L = lmax + 1
    npix = 12 * nside ** 2
    rng = np.random.default_rng(13)
    ell, order = healpix._ell_order_arrays(lmax)
    _theta_matrix.release()

    def scalar(maps_in):
        return jnp.sum(jnp.abs(healpix._map2alm_core_pallas(
            maps_in[None, :], ell, order, nside=nside, L=L, L_work=L, n_iter=0,
            spin=0)))

    grad = np.asarray(jax.grad(scalar)(jnp.asarray(rng.normal(size=npix))))
    assert np.all(np.isfinite(grad))
    assert float(np.max(np.abs(grad))) > 0.0


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
def test_no_tracer_can_enter_the_table_caches():
    """Calling the route dispatcher under a trace must not raise or cache a tracer.

    `_trace_route_ready` builds the Legendre band and its synthesis layout, so it can be
    reached during a user's `jax.jit`, `jax.eval_shape` or `jax.grad`.  Both builders must
    then decline: caching a tracer would cause an `UnexpectedTracerError` later, and
    calling `block_until_ready` on one raises `AttributeError` inside the caller's trace.
    """
    nside = 64
    L = 3 * nside
    _theta_matrix.release()

    out = jax.jit(lambda x: (healpix._trace_route_ready(nside, L), x * 2.0)[1])(
        jnp.ones(4))
    np.testing.assert_allclose(np.asarray(out), 2.0)

    # After the trace, the same geometry must still build cleanly at top level;
    # a decline that leaves the caches unusable is as bad as a crash.
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
def test_shared_band_pair_is_bit_identical_to_two_transforms(monkeypatch, nside):
    """Two spin-0 analyses sharing one Legendre band are bit-identical to two separate analyses.

    The pair runs both iterative refinements in lockstep so that each latitudinal
    contraction serves two maps for one read of the band.  Every operation on either map
    is unchanged (same ring FFT, same band, same accumulation order), so the assertion is
    equality.  The test first asserts that the paired route is used, so that a silent
    fallback cannot make it vacuous.
    """
    # This tests the *band* pair route.  The v2 CUDA march has its own pairing (tested below)
    # and is the default at these sizes, so it is disabled here.
    monkeypatch.setenv("GMASTER_MARCH_V2", "0")
    lmax = 3 * nside - 1
    L = lmax + 1
    npix = 12 * nside ** 2
    assert healpix._shared_band_route(nside, L)[0]
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
    """Paired analysis is refused, returning None, only when neither pairing route is available.

    Without a band the pair uses the marched Legendre row (`_march_pair_route`), so refusing
    the band budget alone does not refuse the pair; both routes must be disabled.  None
    tells the field constructor to fall back to two separate `map2alm` calls.
    """
    # The v2 CUDA march is the default at this size; this test covers the exact band /
    # Pallas route, so select that route.
    monkeypatch.setenv("GMASTER_MARCH_V2", "0")
    nside = 64
    L = 3 * nside
    npix = 12 * nside ** 2
    minfo = nmt.NmtMapInfo(None, (npix,))
    ainfo = nmt.NmtAlmInfo(L - 1)
    maps = jnp.asarray(np.zeros((1, npix)))
    assert utils.map2alm_pair(maps, maps, minfo, ainfo, n_iter=1) is not None
    monkeypatch.setattr(healpix, "_MATRIX_BAND_BUDGET", 0)
    monkeypatch.setenv("GMASTER_SPIN0_MARCH", "0")
    assert healpix._shared_band_route(nside, L) == (False, False)
    assert healpix._march_pair_route(nside, L) is False
    assert utils.map2alm_pair(maps, maps, minfo, ainfo, n_iter=1) is None


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
@pytest.mark.parametrize("nside", [64, 128])
def test_marched_pair_shares_one_recurrence(monkeypatch, nside):
    """Without a band, two maps share one marched Legendre row, to float32 accuracy.

    The march computes its Legendre row inside the kernel, so a second map only adds an
    extra contraction.  Unlike the band pairing this is not bit-identical: widening the
    output block from 4 to 8 channels changes the order of the theta reduction.  The
    measured difference is 4e-7 of scale at Nside 64 and 4e-6 at Nside 256, inside the
    1e-4 tolerance.
    """
    monkeypatch.setenv("GMASTER_SPIN0_MARCH", "1")
    lmax = 3 * nside - 1
    L = lmax + 1
    npix = 12 * nside ** 2
    assert healpix._shared_band_route(nside, L) == (False, False)
    assert healpix._march_pair_route(nside, L) is True
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
    """The pairing memory estimates increase with Nside, so the gate can only refuse large sizes.

    A formula error that made the estimate decrease with Nside would refuse the small sizes
    where pairing is most useful.
    """
    sizes = [healpix._spin_march.fold_pair_bytes(n, 3 * n)
             for n in (256, 512, 1024, 2048, 3072, 4096)]
    assert all(a < b for a, b in zip(sizes, sizes[1:]))
    from gmaster._sht import march_v2 as _march_v2

    v2_sizes = [_march_v2.pair_bytes(3 * n, n)
                for n in (256, 512, 1024, 2048, 4096)]
    assert all(a < b for a, b in zip(v2_sizes, v2_sizes[1:]))


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
@pytest.mark.parametrize("nside, fits", [(512, True), (1024, True), (2048, True),
                                         (3072, False), (4096, False)])
def test_paired_fold_gate_sides_at_the_measured_sizes(monkeypatch, nside, fits):
    """The paired spin-0 fold is offered where it fits and refused where it does not.

    Pairing a field with its mask on the Pallas march is allowed up to Nside 2048 and refused
    from Nside 3072, where the paired footprint (measured to exhaust a ~71 GiB pool at Nside
    4096) no longer fits; a refusal falls back to two separate calls.  The gate reads the pool
    limit, so it is meaningful with the default (preallocated) allocator.  The v2 CUDA march has
    its own, smaller estimate (`march_v2.pair_bytes`), so it is disabled here.
    """
    monkeypatch.setenv("GMASTER_MARCH_V2", "0")
    monkeypatch.setenv("GMASTER_SPIN0_MARCH", "1")
    assert healpix._spin_march.fold_pair_fits(nside, 3 * nside) is fits
    assert healpix._march_pair_route(nside, 3 * nside) is fits


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
def test_latitudinal_adjoint_hands_back_the_operand_dtype():
    """The analysis VJP returns a cotangent with the input's dtype, not the kernel's.

    The latitudinal kernel accumulates in float64 and its transpose runs the float64
    synthesis kernel.  Without a cast at the adjoint boundary, an fp32-ring session would
    pass a complex128 cotangent to the complex64 azimuthal stage and reverse mode would
    fail with a dtype mismatch.
    """
    original = utils.nmt_params.ring_precision
    try:
        utils.set_ring_precision("fp32")
        nside = 16
        L = 3 * nside
        theta, weights, phase = healpix._pallas_parameters(L, nside)
        ftm = jnp.ones((4 * nside - 1, L), dtype=jnp.complex64)

        def stage(x):
            return healpix.scalar_forward_latitudinal(
                x, theta, weights, phase, L=L,
                block_size=healpix._pallas_block_size(nside))

        out, vjp = jax.vjp(stage, ftm)
        assert out.dtype == jnp.complex128
        cotangent, = vjp(jnp.ones_like(out))
        assert cotangent.dtype == ftm.dtype
    finally:
        utils.set_ring_precision(original)


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
def test_paired_fold_declines_when_the_gate_refuses(monkeypatch):
    """When the memory gate refuses pairing, `map2alm_pair` returns None instead of failing to allocate."""
    # The v2 CUDA march is the default at this size; this test covers the exact band /
    # Pallas route, so select that route.
    monkeypatch.setenv("GMASTER_MARCH_V2", "0")
    monkeypatch.setattr(healpix._spin_march, "_PAIR_POOL_FACTOR", 10 ** 9)
    monkeypatch.setenv("GMASTER_SPIN0_MARCH", "1")
    nside = 64
    L = 3 * nside
    npix = 12 * nside ** 2
    assert healpix._spin_march.fold_requested(nside, L) is True
    assert healpix._march_pair_route(nside, L) is False
    minfo = nmt.NmtMapInfo(None, (npix,))
    ainfo = nmt.NmtAlmInfo(L - 1)
    maps = jnp.asarray(np.zeros((1, npix)))
    assert utils.map2alm_pair(maps, maps, minfo, ainfo, n_iter=1) is None


@pytest.mark.parametrize("nside", [512, 1024, 2048, 4096, 8192])
def test_synthesis_tile_is_per_spin_and_still_powers_of_two(nside):
    """The spin-0 synthesis theta tile gives a power-of-two tile count covering every ring.

    `_inverse_fold_impl` and `_call_synth` both derive their launch shapes from
    `_synth_tile`, so these properties must hold at every size, including those (Nside >=
    2048) that are too expensive to run in the test suite.  The Triton lowering requires
    power-of-two array shapes, and the synthesis `out_shape` is `(mb, ntile, chunk, 4)`, so
    `ntile` must be a power of two.

    Spin 0 uses the wider tile `_ST0` once there are at least four tiles of it (it was
    measured faster from Nside 2048); spin 2 always uses `_ST`.  See `_ST0` in
    `gmaster/_sht/spin_march.py`.
    """
    from gmaster._sht import spin_march as smp

    ntheta = 4 * nside - 1
    north = (ntheta + 1) // 2
    st = smp._synth_tile(0, north)
    ntile = -(-north // st)
    assert st * ntile >= north > st * (ntile - 1)
    assert ntile & (ntile - 1) == 0, f"ntile={ntile} is not a power of two"
    assert st == (smp._ST0 if north >= 4 * smp._ST0 else smp._ST)
    # Spin 2 uses the standard width at every size.
    assert smp._synth_tile(2, ntheta) == smp._ST


@pytest.mark.parametrize("nside", [512, 1024, 2048, 4096, 8192])
def test_synth_order_window_fills_the_launch_ceiling(nside):
    """The spin-0 synthesis order window grows toward `_MARCH_GRID_CAP` without exceeding it.

    Performance of the marched routes depends on the number of programs per launch rather
    than on the window width, so the spin-0 synthesis, which has fewer theta tiles than the
    analysis, widens its order window to fill the launch limit.

    The synthesis window must be chosen independently of the analysis windows: raising the
    global analysis window instead would push `mb * ntile` over the limit and silently
    fall back to the narrow default.  The test checks this at every size.
    """
    from gmaster._sht import spin_march as smp
    from gmaster._sht import spin_slice as ss

    L = 3 * nside - 1
    north = (4 * nside - 1 + 1) // 2
    ntile = -(-north // smp._synth_tile(0, north))
    win = smp._synth_windows(L, ntile, 0)
    if ntile < 4:
        # Below four theta tiles a wider window is slower, so the analysis rule is used as is.
        assert win == smp._march_windows(L, ntile, 0)
        return
    widths = {m1 - m0 for m0, m1, _ in win}
    assert len(widths) == 2 or len(widths) == 1      # full windows plus at most one truncated tail
    mb = max(widths)
    assert mb & (mb - 1) == 0, f"mb={mb} is not a power of two (Triton lowering)"
    assert mb >= ss._MARCH_M_BLOCK
    assert mb * ntile <= smp._MARCH_GRID_CAP
    # As close to the limit as a power of two allows, unless the maximum width is reached.
    assert mb == ss._MARCH_M_SYNTH0_MAX or 2 * mb * ntile > smp._MARCH_GRID_CAP
    # Coverage: every order 0..L-1 in exactly one window.
    assert [m0 for m0, _, _ in win] == list(range(0, L, mb))
    assert win[-1][1] == L

    # Spin 2 synthesis is the analysis rule verbatim, at every size.
    assert smp._synth_windows(L, ntile, 2) == smp._march_windows(L, ntile, 2)

    # The analysis windows must not change when the synthesis width limit changes.
    before = smp._march_windows(L, ntile, 0)
    orig = ss._MARCH_M_SYNTH0_MAX
    try:
        ss._MARCH_M_SYNTH0_MAX = 4096
        assert smp._march_windows(L, ntile, 0) == before
    finally:
        ss._MARCH_M_SYNTH0_MAX = orig


def test_pool_headroom_survives_a_backend_without_allocator_stats(monkeypatch):
    """`_pool_headroom` treats a backend without memory statistics as unbounded.

    The CPU backend returns `None` from `memory_stats()` rather than raising; both cases
    must give +inf instead of an exception.
    """
    from gmaster._sht import spin_slice as ss

    class _Silent:
        def memory_stats(self):
            return None

    class _Raises:
        def memory_stats(self):
            raise RuntimeError("no statistics here")

    for device in (_Silent(), _Raises()):
        monkeypatch.setattr(jax, "local_devices", lambda: [device])
        assert ss._pool_headroom() == float("inf")


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
@pytest.mark.parametrize("nside, traced", [(64, True), (128, True), (256, True),
                                           (512, True), (1024, False), (2048, False)])
def test_polar_refinement_trace_gate_sides_at_the_measured_sizes(nside, traced):
    """The traced spin-2 refinement loop is used only up to Nside 512.

    Up to Nside 512 (`L_work = 1536`) tracing the loop into one program was measured
    faster, with both fp64 and fp32 tables.  Above that `_spin_slabs` provides no slab pair,
    so the traced route does not apply.
    """
    L_work = 3 * nside
    maps = jnp.zeros((2, 12 * nside ** 2), dtype=jnp.float64)
    assert healpix._spin_slab_trace_ready(maps, L_work) is traced


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
def test_polar_refinement_stays_eager_under_a_trace():
    """Reverse mode keeps the per-pass boundaries it was measured with.

    `map2alm` converts its input with `jnp.asarray`, so the guard sees the array itself and a
    traced input means the caller is inside `grad`/`jit`.  One graph over `n_iter` refinement
    passes would keep every iteration's residual alive for the transpose, which has never been
    measured here, so a traced input must fall back to the eager loop rather than pick a route
    chosen for forward-only timings.
    """
    seen = []

    def probe(maps_in):
        seen.append(healpix._spin_slab_trace_ready(maps_in, 192))
        return 0.0

    jax.eval_shape(probe, jnp.zeros((2, 12 * 64 ** 2), dtype=jnp.float64))
    assert seen == [False]


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
@pytest.mark.parametrize("nside", [32, 48])
def test_traced_refinement_loop_matches_the_eager_loop(nside):
    """Tracing the `1+2*n_iter` spin-2 passes into one program does not change the result.

    Both arms call the same `_map2alm_once_slab_body`/`_alm2map_core_slab_body` functions,
    so the comparison is exact.
    """
    lmax = 3 * nside - 1
    L = lmax + 1
    npix = 12 * nside ** 2
    rng = np.random.default_rng(11)
    maps = jnp.asarray(rng.normal(size=(2, npix)) * 1e-3)
    ell, order = healpix._ell_order_arrays(lmax)
    a_slab, s_slab = healpix._spin_slabs(L, 2, nside=nside)
    assert a_slab is not None and s_slab is not None

    traced = healpix._map2alm_core_slab(maps, ell, order, spin=2, nside=nside, L=L,
                                      L_work=L, n_iter=2, analysis_slab=a_slab,
                                      synthesis_slab=s_slab)
    eager = healpix._map2alm_once_slab(maps, ell, order, spin=2, nside=nside, L=L,
                                     L_work=L, slab=a_slab)
    for _ in range(2):
        eager = healpix._map2alm_iteration_slab(
            eager, maps, ell, order, spin=2, nside=nside, L=L, L_work=L,
            analysis_slab=a_slab, synthesis_slab=s_slab)
    np.testing.assert_array_equal(np.asarray(traced), np.asarray(eager))


def test_chunked_rows_concatenates_static_slices():
    """Polar-cap chirp-Z batches are a concatenate of static row slices, not a rewrite."""
    rows = jnp.arange(10, dtype=jnp.float64)[:, None] * jnp.array([1.0, 2.0, 3.0])
    got = rings._chunked_rows(10, 3, lambda lo, hi: rows[lo:hi])
    np.testing.assert_array_equal(np.asarray(got), np.asarray(rows))


def test_polar_cap_chunking_matches_the_unchunked_ring_stage():
    """Processing polar-cap rings in chunks is bit-identical to processing them in one batch.

    At large Nside the spin-2 chirp-Z over all cap rows at once would need very large
    intermediates (8.6 GiB each at Nside 4096), so the rows are split into chunks of
    `_CAP_CHUNK_ROWS`.  Here Nside 16 (30 cap rows) is run with 4-row chunks and compared
    with the unchunked result, forward and inverse.
    """
    nside, L = 16, 47
    npix = 12 * nside ** 2
    ntheta = 4 * nside - 1
    rng = np.random.default_rng(7)
    signal = jnp.asarray(rng.normal(size=npix) + 1j * rng.normal(size=npix))
    centered = jnp.asarray(
        rng.normal(size=(ntheta, 2 * L - 1))
        + 1j * rng.normal(size=(ntheta, 2 * L - 1))
    )
    a_tables = rings._spin_ring_analysis_tables(L, nside)
    s_tables = rings._spin_ring_synthesis_tables(L, nside)
    fwd0 = np.asarray(rings._forward_ring_fft_full(signal, a_tables, L=L, nside=nside))
    inv0 = np.asarray(rings._inverse_ring_fft_complex(centered, s_tables, L=L, nside=nside))
    orig = rings._CAP_CHUNK_ROWS
    try:
        rings._CAP_CHUNK_ROWS = 4
        rings._forward_ring_fft_full.clear_cache()
        rings._inverse_ring_fft_complex.clear_cache()
        fwd1 = np.asarray(rings._forward_ring_fft_full(signal, a_tables, L=L, nside=nside))
        inv1 = np.asarray(rings._inverse_ring_fft_complex(centered, s_tables, L=L, nside=nside))
    finally:
        rings._CAP_CHUNK_ROWS = orig
        rings._forward_ring_fft_full.clear_cache()
        rings._inverse_ring_fft_complex.clear_cache()
    np.testing.assert_array_equal(fwd1, fwd0)
    np.testing.assert_array_equal(inv1, inv0)


def test_iteration_split_engages_at_nside_4096_and_not_at_nside_8():
    """The refinement iteration is split into two programs at Nside 4096 but not at Nside 8."""
    nside, L_work = 4096, 3 * 4096
    assert (4 * nside - 1) * 2 * L_work * 16 > healpix._ITERATION_SPLIT_BYTES
    nside_s, L_s = 8, 24
    assert (4 * nside_s - 1) * 2 * L_s * 16 <= healpix._ITERATION_SPLIT_BYTES


def test_split_refinement_iteration_matches_fused_program(monkeypatch):
    """Splitting the refinement iteration into two programs does not change the alms.

    Above `_ITERATION_SPLIT_BYTES` the iteration runs as two calls
    (`_alm2map_core_impl` then `_map2alm_once_impl`) instead of one jit.  Setting the
    threshold to zero forces that path at Nside 8, which is then compared with the fused
    path and with NaMaster.
    """
    reference = pytest.importorskip("pymaster")
    nside, lmax = 8, 10
    npix = 12 * nside ** 2
    minfo = nmt.NmtMapInfo(None, (npix,))
    ainfo = nmt.NmtAlmInfo(lmax)
    rng = np.random.default_rng(3)
    maps = jnp.asarray(rng.normal(size=(2, npix)))
    fused = np.asarray(nmt.map2alm(maps, 2, minfo, ainfo, n_iter=1))
    monkeypatch.setattr(healpix, "_ITERATION_SPLIT_BYTES", 0)
    split = np.asarray(nmt.map2alm(maps, 2, minfo, ainfo, n_iter=1))
    np.testing.assert_allclose(split, fused, atol=3e-13)
    ref_minfo = reference.NmtMapInfo(None, (npix,))
    ref_ainfo = reference.NmtAlmInfo(lmax)
    ref = reference.map2alm(np.asarray(maps), 2, ref_minfo, ref_ainfo, n_iter=1)
    np.testing.assert_allclose(split, ref, atol=3e-13)


def test_make_room_drops_ring_tables_only_when_the_pool_is_short(monkeypatch):
    """`make_room` drops the ring tables only when device memory is short.

    The live device is wrapped rather than replaced, because `Device.memory_stats` is
    read-only and a fake device object can break later tests in the same process.
    """
    nside, L = 8, 16
    rings.drop_ring_tables()
    rings._spin_ring_analysis_tables(L, nside)
    assert rings._spin_ring_analysis_tables.cache_info().currsize > 0
    inner = jax.devices()[0]

    class _Wrap:
        def __init__(self, stats):
            self._stats = stats

        def memory_stats(self):
            return self._stats

        def __getattr__(self, name):
            return getattr(inner, name)

    monkeypatch.setattr(jax, "devices", lambda: [_Wrap({"bytes_limit": 10 ** 12, "bytes_in_use": 10})])
    _config.make_room(10 ** 6)
    assert rings._spin_ring_analysis_tables.cache_info().currsize > 0

    monkeypatch.setattr(jax, "devices", lambda: [_Wrap({"bytes_limit": 100, "bytes_in_use": 90})])
    _config.make_room(50)
    assert rings._spin_ring_analysis_tables.cache_info().currsize == 0


def test_legendre_pool_bytes_is_unbounded_when_the_backend_is_silent(monkeypatch):
    """`_theta_matrix._pool_bytes` treats a backend without memory statistics as unbounded.

    As for `_pool_headroom`, a `None` from `memory_stats()` (the CPU backend) or an
    exception must give +inf rather than raising.
    """
    from gmaster._sht import theta_matrix as _theta_matrix

    class _Silent:
        def memory_stats(self):
            return None

    class _Raises:
        def memory_stats(self):
            raise RuntimeError("no statistics here")

    for device in (_Silent(), _Raises()):
        monkeypatch.setattr(jax, "local_devices", lambda: [device])
        assert _theta_matrix._pool_bytes() == float("inf")



@pytest.mark.march_v2
@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
@pytest.mark.parametrize("nside", [256, 512])
def test_v2_march_pair_matches_two_transforms(nside):
    """The v2 march's paired analysis and synthesis are the two transforms.

    The synthesis pair is exact -- the second map uses the accumulator lanes the spin-2 kernel
    would use for its second helicity, and nothing else in the kernel changes -- so it is asserted
    bit for bit.  The analysis pair is not: two maps per launch need the half-size theta tile, so
    the reduction tree over a tile differs and the result moves by the float32 rounding of the
    emit (measured 2.6e-07 relative at Nside 1024).  Its bar is that distance, not equality.
    """
    from gmaster._sht import march_v2 as _march_v2

    if not _march_v2.enabled(3 * nside):
        pytest.skip("v2 march unavailable")
    L = 3 * nside
    ntheta = 4 * nside - 1
    rng = np.random.default_rng(23)
    weights = quadrature_jax.quad_weights_transform(L, "healpix", nside)
    phase = -healpix_ffts.p2phi_rings_jax(jnp.arange(ntheta), nside)
    pa = jnp.asarray(rng.normal(size=(ntheta, L)) + 1j * rng.normal(size=(ntheta, L)))
    pb = jnp.asarray(rng.normal(size=(ntheta, L)) + 1j * rng.normal(size=(ntheta, L)))
    single = [_march_v2.forward_latitudinal_positive(p, weights, phase, L=L, nside=nside)
              for p in (pa, pb)]
    paired = _march_v2.forward_latitudinal_positive_pair(pa, pb, weights, phase, L=L, nside=nside)
    for got, ref in zip(paired, single):
        scale = float(np.max(np.abs(np.asarray(ref))))
        assert float(np.max(np.abs(np.asarray(got) - np.asarray(ref)))) < 1e-5 * scale

    alm_a = jnp.asarray(np.triu(rng.normal(size=(L, L))).T.astype(np.complex128))
    alm_b = jnp.asarray(np.triu(rng.normal(size=(L, L))).T.astype(np.complex128))
    syn_single = [_march_v2.inverse_latitudinal_positive(a, -phase, L=L, nside=nside)
                  for a in (alm_a, alm_b)]
    syn_pair = _march_v2.inverse_latitudinal_positive_pair(alm_a, alm_b, -phase, L=L, nside=nside)
    for got, ref in zip(syn_pair, syn_single):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


@pytest.mark.march_v2
@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
@pytest.mark.parametrize("nside, spin", [(32, 0), (32, 2), (64, 0), (64, 2), (128, 0), (128, 2)])
def test_default_route_agrees_with_namaster_at_small_geometries(nside, spin):
    """The default (v2 march) route agrees with NaMaster to 1e-5 of scale at every size tested.

    The exact fp64 band agrees with NaMaster to 1e-13; the march is float32 and agrees to ~1e-6.
    The exact routes are still there and are still pinned to 1e-13 by the tests above
    (`GMASTER_MARCH_V2=0`), so what this adds is a bar on the *default*: whatever the routing
    decides, the alms a user gets must be this close.
    """
    ref = pytest.importorskip("pymaster")
    lmax = 3 * nside - 1
    npix = 12 * nside ** 2
    rng = np.random.default_rng(19)
    maps = rng.normal(size=(1 if spin == 0 else 2, npix))
    got = np.asarray(nmt.map2alm(jnp.asarray(maps), spin,
                                 nmt.NmtMapInfo(None, (npix,)), nmt.NmtAlmInfo(lmax), n_iter=3))
    want = ref.map2alm(maps, spin, ref.NmtMapInfo(None, (npix,)), ref.NmtAlmInfo(lmax), n_iter=3)
    scale = float(np.max(np.abs(want)))
    assert float(np.max(np.abs(got - want))) < 1e-5 * scale


@pytest.mark.march_v2
@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
@pytest.mark.parametrize("nside", [16, 64, 256])
def test_ring_fold_residual_is_fft_of_ifft(nside):
    """`ring_fold_residual(F, Mf)` is the ring FFT of the ring IFFT of F, minus Mf.

    A ring with fewer pixels than 2L aliases, so the check runs through the polar caps as well as
    the belt; the fold sums in float64 and the bar is the complex64 rounding of the ring stage.
    """
    from gmaster._sht import march_v2 as _march_v2

    L = 3 * nside
    if not _march_v2.fold_available():
        pytest.skip("CUDA march library unavailable")
    rng = np.random.default_rng(nside)
    shape = (4 * nside - 1, L)
    F = jax.numpy.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape))
    Mf = jax.numpy.asarray((rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64))
    maps = jax.numpy.real(healpix._finish_inverse_pallas(F, L=L, nside=nside))
    explicit = np.asarray(rings._forward_ring_fft_positive(
        maps, rings._ring_analysis_tables(L, nside), L=L, nside=nside)) - np.asarray(Mf)
    folded = np.asarray(_march_v2.ring_fold_residual(F, Mf, nside=nside))
    assert np.max(np.abs(folded - explicit)) < 2e-6 * np.max(np.abs(explicit))


@pytest.mark.march_v2
@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
@pytest.mark.parametrize("nside", [32, 128])
def test_dc_latitudinal_matches_march(nside):
    """The sub-cubic divide-and-conquer engine has the folded march's contract and accuracy.

    The router only hands it L >= 6144, so it is exercised here directly on small geometries:
    both engines are float32-class, and their distance is that of the two against fp64.
    """
    from gmaster._sht import dc as _dc_lat
    from gmaster._sht import march_v2 as _march_v2
    from s2fft.utils import healpix_ffts, quadrature_jax

    L = 3 * nside
    if _dc_lat._build() is None or not _march_v2.fold_available():
        pytest.skip("CUDA libraries unavailable")
    rng = np.random.default_rng(nside)
    nring = 4 * nside - 1
    alm = np.tril(rng.normal(size=(L, L)) + 1j * rng.normal(size=(L, L)))
    alm = jax.numpy.asarray(alm)
    phase = healpix_ffts.p2phi_rings_jax(jax.numpy.arange(nring), nside)
    weights = quadrature_jax.quad_weights_transform(L, "healpix", nside)
    ring = jax.numpy.asarray(rng.normal(size=(nring, L)) + 1j * rng.normal(size=(nring, L)))
    pairs = (
        (_march_v2._inverse_fold_impl(alm, phase, _march_v2.geo_arrays(L, nside, 0),
                                      _march_v2.tables_for(L, nside, 0), L=L, nside=nside),
         _dc_lat.inverse_latitudinal_positive(alm, phase, L=L, nside=nside)),
        (_march_v2._forward_fold_impl(ring, weights, -phase, _march_v2.geo_arrays(L, nside, 0),
                                      _march_v2.tables_for(L, nside, 0), L=L, nside=nside),
         _dc_lat.forward_latitudinal_positive(ring, weights, -phase, L=L, nside=nside)),
    )
    for march, dc in pairs:
        march, dc = np.asarray(march), np.asarray(dc)
        assert np.max(np.abs(dc - march)) < 2e-5 * np.max(np.abs(march))


@pytest.mark.march_v2
@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
@pytest.mark.parametrize("nside", [64, 128])
def test_dc_spin2_latitudinal_matches_march(nside):
    """The divide-and-conquer engine's spin-2 route has the march's spin-2 contract."""
    from gmaster._sht import dc as _dc_lat
    from gmaster._sht import march_v2 as _march_v2

    L = 3 * nside
    if _dc_lat._build() is None or not _march_v2.fold_available():
        pytest.skip("CUDA libraries unavailable")
    rng = np.random.default_rng(nside)
    nt = 4 * nside - 1
    ftm = rng.normal(size=(nt, 2 * L)) + 1j * rng.normal(size=(nt, 2 * L))
    ftm[:, 0] = 0
    flm = rng.normal(size=(L, 2 * L - 1)) + 1j * rng.normal(size=(L, 2 * L - 1))
    ell = np.arange(L)[:, None]
    flm[(ell < np.abs(np.arange(-(L - 1), L))[None, :]) | (ell < 2)] = 0
    ftm, flm = jax.numpy.asarray(ftm), jax.numpy.asarray(flm)
    geo, tabs = _march_v2.geo_arrays(L, nside, 2), _march_v2.tables_for(L, nside, 2)
    pairs = (
        (_march_v2._forward_impl(ftm, geo, tabs, L=L, nside=nside),
         _dc_lat.forward_latitudinal_spin(ftm, L=L, spin=2, nside=nside)),
        (_march_v2._inverse_impl(flm, geo, tabs, L=L, nside=nside),
         _dc_lat.inverse_latitudinal_spin(flm, L=L, spin=2, nside=nside)),
    )
    for march, dc in pairs:
        march, dc = np.asarray(march), np.asarray(dc)
        assert np.max(np.abs(dc - march)) < 5e-5 * np.max(np.abs(march))


@pytest.mark.march_v2
@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="requires an NVIDIA GPU")
@pytest.mark.parametrize("spin", [0, 2])
def test_dc_node_space_refinement_matches_tree_space(monkeypatch, spin):
    """Refinement in the D&C engine's node space (V^T V = I) is the tree-space refinement."""
    from gmaster._sht import dc as _dc_lat
    from gmaster._sht import march_v2 as _march_v2

    if _dc_lat._build() is None or not _march_v2.fold_available():
        pytest.skip("CUDA libraries unavailable")
    monkeypatch.setattr(_dc_lat, "_MIN_L", 0)
    nside = 64
    npix = 12 * nside ** 2
    rng = np.random.default_rng(spin)
    mask = np.clip(rng.uniform(size=npix) + 0.3, 0, 1)
    maps = [rng.normal(size=npix)] if spin == 0 else [rng.normal(size=npix), rng.normal(size=npix)]
    out = []
    for node_space in (False, True):
        monkeypatch.setattr(healpix, "_NODE_SPACE", node_space)
        field = nmt.NmtField(mask, maps, n_iter=3, spin=spin)
        out.append(np.asarray(field.alm))
    assert np.max(np.abs(out[1] - out[0])) < 2e-5 * np.max(np.abs(out[0]))
