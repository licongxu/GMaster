import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import gmaster as nmt
from gmaster import utils
from gmaster._cuda_gpu import is_cuda_device_kind


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


def test_alm_index_arrays_are_cached_by_lmax():
    lmax = 4
    m = np.arange(lmax + 1)
    first = utils._alm_index_arrays(lmax, m)
    assert utils._alm_index_arrays(lmax, m) is first
    assert utils._alm_index_arrays(lmax + 2, np.arange(lmax + 3)) is not first

    np.testing.assert_array_equal(
        np.asarray(first[0]), np.concatenate([np.arange(i, lmax + 1) for i in m])
    )
    np.testing.assert_array_equal(np.asarray(first[1]), np.repeat(m, lmax + 1 - m))

    info = utils.NmtAlmInfo(lmax)
    assert info._ell is first[0]
    assert info._m is first[1]


def test_alm_index_cache_never_holds_a_tracer():
    from jax.core import Tracer

    lmax = 6

    @jax.jit
    def build_inside(x):
        info = utils.NmtAlmInfo(lmax)
        return x + info._m.sum() + info._ell.sum()

    assert jnp.isfinite(build_inside(jnp.asarray(0.0)))
    assert all(
        not isinstance(array, Tracer)
        for pair in utils._ALM_INDEX_CACHE.values()
        for array in pair
    )
    # The traced lmax is still usable outside the trace and caches cleanly there.
    outside = utils.NmtAlmInfo(lmax)
    assert utils._ALM_INDEX_CACHE[lmax] == (outside._ell, outside._m)


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
        for calculator in ("jax", "jax-single", "jax-mgpu", "jax-generic"):
            nmt.set_sht_calculator(calculator)
            assert nmt.get_default_params()["sht_calculator"] == calculator
        with pytest.raises(KeyError, match="jax"):
            nmt.set_sht_calculator("healpy")
    finally:
        nmt.set_n_iter_default(original["n_iter_default"])
        nmt.set_n_iter_default(original["n_iter_mask_default"], mask=True)
        nmt.set_tol_pinv_default(original["tol_pinv_default"])
        nmt.set_sht_calculator(original["sht_calculator"])


def test_cuda_gpu_gate_accepts_colab_tesla_t4():
    """Colab T4 is ``Tesla T4``; requiring ``NVIDIA`` silently skipped the march."""
    assert is_cuda_device_kind("Tesla T4")
    assert is_cuda_device_kind("Tesla L4")
    assert is_cuda_device_kind("NVIDIA RTX PRO 6000 Blackwell Workstation Edition")
    assert is_cuda_device_kind("cuda")
    assert "NVIDIA" not in "Tesla T4".upper()
    assert not is_cuda_device_kind("TPU v5")
    assert not is_cuda_device_kind("AMD Instinct MI250")
