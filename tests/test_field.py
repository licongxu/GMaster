import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import gmaster as nmt


def test_standard_fields_and_coupled_spectra_match_namaster():
    reference = pytest.importorskip("pymaster")
    rng = np.random.default_rng(9)
    nside = 8
    lmax = 10
    npix = 12 * nside**2
    mask = rng.uniform(0.2, 1, npix)
    scalar = rng.normal(size=(1, npix))
    spin2 = rng.normal(size=(2, npix))

    got0 = nmt.NmtField(mask, scalar, lmax=lmax, lmax_mask=lmax)
    got2 = nmt.NmtField(mask, spin2, lmax=lmax, lmax_mask=lmax)
    ref0 = reference.NmtField(mask, scalar, lmax=lmax, lmax_mask=lmax)
    ref2 = reference.NmtField(mask, spin2, lmax=lmax, lmax_mask=lmax)

    np.testing.assert_allclose(got0.get_maps(), ref0.get_maps(), atol=1e-14)
    np.testing.assert_allclose(got2.get_maps(), ref2.get_maps(), atol=1e-14)
    np.testing.assert_allclose(got0.get_alms(), ref0.get_alms(), atol=4e-13)
    np.testing.assert_allclose(got2.get_alms(), ref2.get_alms(), atol=4e-13)
    for got_a, got_b, ref_a, ref_b in (
        (got0, got0, ref0, ref0),
        (got0, got2, ref0, ref2),
        (got2, got2, ref2, ref2),
    ):
        np.testing.assert_allclose(
            nmt.compute_coupled_cell(got_a, got_b),
            reference.compute_coupled_cell(ref_a, ref_b),
            atol=2e-13,
        )


def test_template_deprojection_matches_namaster():
    reference = pytest.importorskip("pymaster")
    rng = np.random.default_rng(10)
    nside = 4
    lmax = 7
    npix = 12 * nside**2
    mask = rng.uniform(0.3, 1, npix)
    template = rng.normal(size=(1, 1, npix))
    maps = 2.5 * template[0] + rng.normal(size=(1, npix))
    got = nmt.NmtField(mask, maps, templates=template, lmax=lmax, lmax_mask=lmax)
    ref = reference.NmtField(mask, maps, templates=template, lmax=lmax, lmax_mask=lmax)

    np.testing.assert_allclose(got.alphas, ref.alphas, atol=2e-14)
    np.testing.assert_allclose(got.get_maps(), ref.get_maps(), atol=2e-14)
    np.testing.assert_allclose(got.get_templates(), ref.get_templates(), atol=2e-14)
    np.testing.assert_allclose(got.get_alms(), ref.get_alms(), atol=3e-13)


def test_pure_fields_match_namaster():
    reference = pytest.importorskip("pymaster")
    healpy = pytest.importorskip("healpy")
    rng = np.random.default_rng(11)
    nside = 4
    lmax = 7
    npix = 12 * nside**2
    theta, _ = healpy.pix2ang(nside, np.arange(npix))
    mask = np.sin(theta) ** 4
    template = rng.normal(size=(1, 2, npix))
    maps = 0.3 * template[0] + rng.normal(size=(2, npix))

    for pure_e, pure_b in ((True, False), (False, True), (True, True)):
        options = dict(
            templates=template,
            purify_e=pure_e,
            purify_b=pure_b,
            lmax=lmax,
            lmax_mask=lmax,
        )
        got = nmt.NmtField(mask, maps, **options)
        ref = reference.NmtField(mask, maps, **options)
        np.testing.assert_allclose(got.alphas, ref.alphas, atol=2e-14)
        np.testing.assert_allclose(got.get_maps(), ref.get_maps(), atol=2e-13)
        np.testing.assert_allclose(got.get_alms(), ref.get_alms(), atol=2e-13)
        np.testing.assert_allclose(got.alm_temp, ref.alm_temp, atol=2e-13)

    masked_maps = maps * mask[None, :]
    got = nmt.NmtField(
        mask,
        masked_maps,
        purify_b=True,
        masked_on_input=True,
        lmax=lmax,
        lmax_mask=lmax,
    )
    ref = reference.NmtField(
        mask,
        masked_maps,
        purify_b=True,
        masked_on_input=True,
        lmax=lmax,
        lmax_mask=lmax,
    )
    np.testing.assert_allclose(got.get_alms(), ref.get_alms(), atol=2e-13)


def test_anisotropic_fields_match_namaster():
    reference = pytest.importorskip("pymaster")
    rng = np.random.default_rng(12)
    nside = 4
    lmax = 7
    npix = 12 * nside**2
    mask_11 = rng.uniform(0.5, 1, npix)
    mask_22 = rng.uniform(0.5, 1, npix)
    mask_12 = 0.1 * np.sqrt(mask_11 * mask_22)
    maps = rng.normal(size=(2, npix))
    options = dict(
        mask_22=mask_22, mask_12=mask_12, lmax=lmax, lmax_mask=lmax
    )
    got = nmt.NmtField(mask_11, maps, **options)
    ref = reference.NmtField(mask_11, maps, **options)

    np.testing.assert_allclose(got.get_mask(), ref.get_mask(), atol=0)
    np.testing.assert_allclose(
        got.get_anisotropic_mask(), ref.get_anisotropic_mask(), atol=0
    )
    np.testing.assert_allclose(got.get_maps(), ref.get_maps(), atol=1e-14)
    np.testing.assert_allclose(got.get_alms(), ref.get_alms(), atol=2e-13)
    np.testing.assert_allclose(
        got.get_anisotropic_mask_alms(),
        ref.get_anisotropic_mask_alms(),
        atol=2e-13,
    )
    np.testing.assert_allclose(
        nmt.compute_coupled_cell(got, got),
        reference.compute_coupled_cell(ref, ref),
        atol=2e-13,
    )


def test_mask_only_lite_and_invalid_options():
    npix = 12 * 4**2
    field = nmt.NmtField(np.ones(npix), None, spin=2, lmax=7, lmax_mask=7)
    with pytest.raises(ValueError, match="no alms"):
        field.get_alms()
    with pytest.raises(ValueError, match="supply field spin"):
        nmt.NmtField(np.ones(npix), None)
    with pytest.raises(ValueError, match="require lmax"):
        nmt.NmtField(
            np.ones(npix),
            np.ones((2, npix)),
            purify_b=True,
            lmax=6,
            lmax_mask=7,
        )
    with pytest.raises(ValueError, match="Both mask_22"):
        nmt.NmtField(np.ones(npix), None, spin=2, mask_22=np.ones(npix))
    with pytest.raises(ValueError, match="positive-definite"):
        nmt.NmtField(
            np.ones(npix),
            None,
            spin=2,
            mask_22=np.ones(npix),
            mask_12=2 * np.ones(npix),
        )
    with pytest.raises(ValueError, match="scalar fields"):
        nmt.NmtField(
            np.ones(npix),
            None,
            spin=0,
            mask_22=np.ones(npix),
            mask_12=np.zeros(npix),
        )
    with pytest.raises(NotImplementedError, match="Purification"):
        nmt.NmtField(
            np.ones(npix),
            np.ones((2, npix)),
            purify_b=True,
            mask_22=np.ones(npix),
            mask_12=np.zeros(npix),
            lmax=7,
            lmax_mask=7,
        )
