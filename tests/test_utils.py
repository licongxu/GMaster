"""Utilities: apodization, simulations, alm indexing, global parameters and dispatch helpers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import gmaster as nmt
from gmaster import utils
from gmaster._sht.cuda_gpu import is_cuda_device_kind
from gmaster._sht import rings


def test_mask_apodization_matches_namaster():
    """Curved-sky C1, C2 and Smooth apodization match NaMaster."""
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
    """synfast_spherical is reproducible for a fixed seed, its Gaussian draws have the requested
    covariance (to the sampling noise of 10,000 draws), and a non-positive-semidefinite
    spectrum matrix is rejected.
    """
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
    """The (ell, m) index arrays are cached per lmax, follow the healpy layout, and are shared by NmtAlmInfo."""
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
    """Building NmtAlmInfo inside jax.jit must not store a tracer in the index cache."""
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
    # Outside the trace the same lmax builds concrete arrays and caches them.
    outside = utils.NmtAlmInfo(lmax)
    assert utils._ALM_INDEX_CACHE[lmax] == (outside._ell, outside._m)


def test_default_parameters_control_new_fields():
    """The global defaults (n_iter, n_iter_mask, tol_pinv, SHT calculator) are applied and validated."""
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


def test_latitudinal_method_chooses_march_or_dc():
    """set_latitudinal_method selects the latitudinal engine: 'march' disables divide-and-conquer,
    and 'dc'/'auto' still refuse it outside its supported band limits.
    """
    from gmaster._sht import dc as _dc_lat

    original = nmt.latitudinal_method()
    try:
        nmt.set_latitudinal_method("march")
        assert nmt.latitudinal_method() == "march"
        assert _dc_lat.enabled(6144) is False
        nmt.set_latitudinal_method("dc")
        assert _dc_lat.enabled(20000) is False
        nmt.set_latitudinal_method("auto")
        assert _dc_lat.enabled(10) is False
        with pytest.raises(KeyError, match="latitudinal"):
            nmt.set_latitudinal_method("healpy")
    finally:
        nmt.set_latitudinal_method(original)


def test_cuda_gpu_gate_accepts_colab_tesla_t4():
    """The CUDA device check accepts NVIDIA cards whose device kind lacks the word "NVIDIA".

    Colab's T4 reports itself as ``Tesla T4``; a check that required ``NVIDIA`` in the name
    would silently skip the CUDA routes on it.
    """
    assert is_cuda_device_kind("Tesla T4")
    assert is_cuda_device_kind("Tesla L4")
    assert is_cuda_device_kind("NVIDIA RTX PRO 6000 Blackwell Workstation Edition")
    assert is_cuda_device_kind("cuda")
    assert "NVIDIA" not in "Tesla T4".upper()
    assert not is_cuda_device_kind("TPU v5")
    assert not is_cuda_device_kind("AMD Instinct MI250")


@pytest.mark.parametrize("two_nphi", [8, 32760, 65528, 65536, 131064])
def test_complex64_chirp_matches_exact_integer_reduction(two_nphi):
    """`_chirp_c64` reduces q^2 mod 2 nphi exactly at every HEALPix ring size up to Nside 16384.

    Squaring the index in int32 overflows once 2 nphi reaches 65528 (Nside 8192 rings), which
    would put O(1) errors into the polar-cap ring transforms.  The 1e-6 tolerance is the
    complex64 rounding of the chirp itself.
    """
    import jax.numpy as jnp

    index = jnp.arange(-3 * two_nphi, 3 * two_nphi, 7, dtype=jnp.int64)
    m = jnp.asarray([[two_nphi]], dtype=jnp.int64)
    got = np.asarray(rings._chirp_c64(index, m, sign=1.0, wide=two_nphi > 65536))
    q = np.asarray(index, dtype=object)
    reduced = np.array([(int(v) * int(v)) % two_nphi for v in q], dtype=np.float64)
    exact = np.exp(1j * reduced * (2 * np.pi / two_nphi))[None, :]
    assert np.max(np.abs(got - exact)) < 1e-6
