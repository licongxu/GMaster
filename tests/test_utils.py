import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import gmaster as nmt
from gmaster import utils


def test_mask_apodization_matches_namaster():
    reference = pytest.importorskip("pymaster")
    nside = 8
    mask = np.ones(12 * nside**2)
    mask[:50] = 0
    for apotype in ("C1", "C2", "Smooth"):
        np.testing.assert_allclose(
            nmt.mask_apodization(mask, 10, apotype),
            reference.mask_apodization(mask, 10, apotype),
            atol=1e-13,
        )


def test_synfast_spherical_is_reproducible_and_correlated():
    lmax = 5
    spectra = np.zeros((6, lmax + 1))
    spectra[0] = 1
    spectra[1] = 0.25
    spectra[3] = 2
    spectra[5] = 0.5
    first = nmt.synfast_spherical(2, spectra, [0, 2], seed=2, lmax=lmax)
    second = nmt.synfast_spherical(2, spectra, [0, 2], seed=2, lmax=lmax)
    different = nmt.synfast_spherical(2, spectra, [0, 2], seed=3, lmax=lmax)
    assert first.shape == (3, 48)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)

    covariance = jnp.asarray([[[1.0, 0.3], [0.3, 2.0]]])
    values, vectors = jnp.linalg.eigh(covariance)
    root = vectors * jnp.sqrt(values)[:, None, :]
    keys = jax.random.split(jax.random.key(4), 10_000)
    samples = jax.vmap(
        lambda key: utils._gaussian_alms(
            key, root, jnp.zeros(1, dtype=int), jnp.zeros(1, dtype=int)
        )[:, 0]
    )(keys)
    measured = jnp.einsum("si,sj->ij", samples.real, samples.real) / len(samples)
    np.testing.assert_allclose(measured, covariance[0], atol=0.06)

    bad = spectra.copy()
    bad[1] = 2
    with pytest.raises(ValueError, match="positive-semidefinite"):
        nmt.synfast_spherical(2, bad, [0, 2], seed=2, lmax=lmax)


def test_default_parameters_control_new_fields():
    original = nmt.get_default_params()
    try:
        nmt.set_n_iter_default(1)
        nmt.set_n_iter_default(2, mask=True)
        nmt.set_tol_pinv_default(1e-8)
        field = nmt.NmtField(np.ones(48), None, spin=0, lmax=3, lmax_mask=3)
        assert field.n_iter == 1
        assert field.n_iter_mask == 2
        assert nmt.get_default_params()["tol_pinv_default"] == 1e-8
        with pytest.raises(KeyError, match="jax"):
            nmt.set_sht_calculator("healpy")
    finally:
        nmt.set_n_iter_default(original["n_iter_default"])
        nmt.set_n_iter_default(original["n_iter_mask_default"], mask=True)
        nmt.set_tol_pinv_default(original["tol_pinv_default"])
