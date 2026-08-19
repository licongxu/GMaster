import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import gmaster as nmt


def test_curved_binning_matches_namaster():
    reference = pytest.importorskip("pymaster")
    rng = np.random.default_rng(1)
    cls = rng.normal(size=(4, 65))
    got_bin = nmt.NmtBin.from_lmax_linear(64, 7, is_Dell=True)
    ref_bin = reference.NmtBin.from_lmax_linear(64, 7, is_Dell=True)

    np.testing.assert_allclose(
        got_bin.get_effective_ells(), ref_bin.get_effective_ells()
    )
    np.testing.assert_allclose(got_bin.bin_cell(cls), ref_bin.bin_cell(cls), rtol=1e-13)
    np.testing.assert_allclose(
        got_bin.unbin_cell(got_bin.bin_cell(cls)),
        ref_bin.unbin_cell(ref_bin.bin_cell(cls)),
        rtol=1e-13,
    )


def test_curved_custom_bands_and_jit():
    b = nmt.NmtBin.from_edges([0, 4, 9], [4, 9, 16])
    cls = jnp.arange(16.0)
    np.testing.assert_allclose(jax.jit(b.bin_cell)(cls), [1.5, 6.0, 12.0])
    np.testing.assert_array_equal(b.get_nell_list(), [4, 5, 7])
    np.testing.assert_array_equal(
        b.unbin_cell(jnp.array([1.0, 2.0, 3.0]))[:9], [1] * 4 + [2] * 5
    )


def test_curved_rejects_zero_weight_band_and_wrong_shapes():
    with pytest.raises(RuntimeError):
        nmt.NmtBin(
            ells=np.arange(4),
            bpws=np.array([0, 0, 1, 1]),
            weights=np.array([1, 1, 0, 0]),
        )
    b = nmt.NmtBin.from_lmax_linear(16, 3)
    with pytest.raises(ValueError, match="wrong size"):
        b.bin_cell(np.ones((2, 3, 4)))
    with pytest.raises(ValueError, match="wrong size"):
        b.unbin_cell(np.ones((2, 3, 4)))


def test_flat_binning_matches_namaster():
    reference = pytest.importorskip("pymaster")
    l0 = np.arange(2, 42, 5.0)
    lf = l0 + 5
    ells = np.arange(50.0)
    cls = np.stack([ells, ells**2])
    got_bin = nmt.NmtBinFlat(l0, lf)
    ref_bin = reference.NmtBinFlat(l0, lf)

    np.testing.assert_allclose(
        got_bin.get_effective_ells(), ref_bin.get_effective_ells()
    )
    np.testing.assert_allclose(
        got_bin.bin_cell(ells, cls), ref_bin.bin_cell(ells, cls), rtol=1e-13
    )
    np.testing.assert_allclose(
        got_bin.unbin_cell(got_bin.bin_cell(ells, cls), ells),
        ref_bin.unbin_cell(ref_bin.bin_cell(ells, cls), ells),
        rtol=1e-13,
    )
