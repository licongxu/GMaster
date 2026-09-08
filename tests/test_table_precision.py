import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import gmaster as nmt
from gmaster import _spin_slice, _theta_matrix, utils

_HAS_NVIDIA_GPU = any(
    device.platform == "gpu" and "NVIDIA" in device.device_kind.upper()
    for device in jax.devices()
)


@pytest.fixture(autouse=True)
def _restore_precision():
    yield
    utils.set_table_precision("fp64")
    utils.set_ring_precision("follow")
    _theta_matrix.release()
    _spin_slice.clear_cache()


def _clear():
    _theta_matrix.release()
    _spin_slice.clear_cache()


def test_float64_is_the_default():
    assert utils.nmt_params.table_dtype == "fp64"
    assert utils.table_dtype() == jnp.float64


def test_unknown_precision_is_rejected():
    with pytest.raises(KeyError):
        utils.set_table_precision("fp16")
    with pytest.raises(KeyError):
        utils.set_ring_precision("fp16")


def test_ring_precision_follows_the_tables_only_while_it_follows():
    """`follow` is the historical coupling; `fp64`/`fp32` pin it independently.

    The coupling is not cosmetic: the analysis chirp-Z casts the map pixels to the
    chirp's own real dtype, so under `follow` an fp32 session analyzes the map in
    float32 and the ring sums come out ~1e-7.  Held to `fp64` the same session
    reproduces the float64 ring sums exactly, which is what makes the float32 table
    route pass the transform-parity suite.
    """
    assert utils.nmt_params.ring_precision == "follow"
    assert utils.ring_dtype() == jnp.complex128

    utils.set_table_precision("fp32")
    assert utils.ring_dtype() == jnp.complex64

    utils.set_ring_precision("fp64")
    assert utils.ring_dtype() == jnp.complex128
    assert utils.table_dtype() == jnp.float32

    utils.set_ring_precision("fp32")
    assert utils.ring_dtype() == jnp.complex64


def test_ring_switch_rebuilds_the_ring_tables():
    nside, L = 32, 96
    fp64 = utils._ring_analysis_tables(L, nside)
    assert fp64[0].dtype == jnp.complex128

    utils.set_ring_precision("fp32")
    fp32 = utils._ring_analysis_tables(L, nside)
    assert fp32[0].dtype == jnp.complex64
    # The chirp is a phase ramp: the two precisions are the same numbers, rounded.
    np.testing.assert_allclose(
        np.asarray(fp32[0]), np.asarray(fp64[0]).astype(np.complex64),
        rtol=0.0, atol=0.0)


def test_band_storage_follows_the_selection():
    nside, L = 32, 96
    _clear()
    band = _theta_matrix._band(_theta_matrix.band_geometry(nside, L))
    assert band[0][0].dtype == jnp.float64

    utils.set_table_precision("fp32")
    band32 = _theta_matrix._band(_theta_matrix.band_geometry(nside, L))
    assert band32[0][0].dtype == jnp.float32
    # The recurrence stays float64: the single-precision band is the double
    # precision one rounded.  It is rounded as it is emitted rather than after
    # the fact (so no float64 twin of a block is ever resident), which leaves the
    # two differing only in the exp2-underflow tail, around 1e-38.
    for even, odd in zip(band, band32):
        for full, half in zip(even, odd):
            np.testing.assert_allclose(
                np.asarray(half), np.asarray(full).astype(np.float32),
                rtol=0.0, atol=1e-30)


def test_precision_switch_rebuilds_rather_than_reusing():
    nside, L = 32, 96
    _clear()
    fp64_key = _theta_matrix.band_geometry(nside, L)
    assert _theta_matrix._band(fp64_key) is not None
    assert len(_theta_matrix._BAND_CACHE) == 1
    utils.set_table_precision("fp32")
    # The switch clears the caches, and the new geometry key carries the dtype, so
    # a caller can never be handed the other precision's bytes.
    assert len(_theta_matrix._BAND_CACHE) == 0
    fp32_key = _theta_matrix.band_geometry(nside, L)
    assert fp32_key != fp64_key
    assert _theta_matrix._band(fp32_key) is not None
    assert len(_theta_matrix._BAND_CACHE) == 1
    assert _theta_matrix._band(fp64_key) is not None
    assert len(_theta_matrix._BAND_CACHE) == 2


@pytest.mark.parametrize("prec", ["fp64", "fp32"])
def test_contracts_reduce_in_the_storage_width_without_truncating(prec):
    """The width cast must be applied part-by-part, never to a complex operand.

    Both band contractions reduce in the table's storage precision and widen the
    partials afterwards.  Casting the *operand* instead is a one-character mistake
    (`convert_element_type(rhs, slab.dtype)` on a complex128 array truncates it),
    and it silently zeroes the imaginary half of the synthesis sum: the Nside 512
    decoupled cell came out wrong by 7.5 relative, in float64 as much as float32.
    """
    utils.set_table_precision(prec)
    nside, L = 32, 96
    _clear()
    geometry = _theta_matrix.band_geometry(nside, L)

    rng = np.random.default_rng(3)
    analysis = _theta_matrix._band(geometry)[0][0]
    # Analysis slabs are (ell_rows, m_local, j); the synthesis route may keep that
    # layout (strided reduce) or re-lay it out as (m_local, j, ell_rows).
    bands, layout = _theta_matrix._synth_band(geometry)
    assert (bands[0][0].shape == analysis.shape) == (
        layout == _theta_matrix.THETA_CONTIG)

    # The synthesis RHS is (m_local, ell_rows) whichever layout the slab has.
    rhs = jnp.asarray(rng.normal(size=(analysis.shape[1], analysis.shape[0]))
                      + 1j * rng.normal(size=(analysis.shape[1], analysis.shape[0])))
    spellings = {
        "leading": (_theta_matrix._contract_ell_leading, analysis, "emj,me->mj"),
        "trailing": (_theta_matrix._contract_ell,
                     jnp.transpose(analysis, (1, 2, 0)), "mje,me->mj"),
    }
    for name, (contract, slab, pattern) in spellings.items():
        got = np.asarray(contract(slab, rhs))
        want = np.einsum(pattern, np.asarray(slab), np.asarray(rhs))
        assert got.dtype == np.complex128, name
        # A truncated complex operand would leave this exactly zero.
        assert np.max(np.abs(want.imag)) > 0, name
        # A float32 table has a subnormal tail down to ~1e-45, where accumulating
        # in float32 makes the *relative* error of an individual entry meaningless.
        # What matters is the error against the scale the entry is summed into.
        scale = float(np.max(np.abs(want)))
        np.testing.assert_allclose(got, want, rtol=1e-12 if prec == "fp64" else 1e-5,
                                   atol=1e-5 * scale, err_msg=name)

    chan = jnp.asarray(rng.normal(size=analysis.shape[1:] + (2,)))
    acc = np.asarray(_theta_matrix._contract_theta(analysis, chan))
    want = np.einsum("emj,mjc->emc", np.asarray(analysis), np.asarray(chan))
    assert acc.dtype == np.float64
    np.testing.assert_allclose(acc, want, rtol=1e-12 if prec == "fp64" else 1e-5,
                               atol=1e-5 * float(np.max(np.abs(want))))


def test_byte_estimates_follow_the_storage_precision():
    fp64 = _theta_matrix.band_bytes(128, 384, dtype=jnp.float64)
    fp32 = _theta_matrix.band_bytes(128, 384, dtype=jnp.float32)
    assert fp32 * 2 == fp64
    tri64 = _spin_slice.triangle_bytes(128, 384, jnp.float64)
    tri32 = _spin_slice.triangle_bytes(128, 384, jnp.float32)
    assert tri32 * 2 == tri64


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="SHT dispatch requires an NVIDIA GPU")
@pytest.mark.parametrize("spin", [0, 2])
def test_float32_tables_keep_the_coupling_matrix(spin):
    """fp32 tables cost representation error and nothing else.

    Measured on the full pipeline: 5.2e-8 relative on the decoupled Cls against
    float64 tables, at every size tried.  The bound here is that, plus headroom;
    float64 stays the default precisely because it is ~1e-12 instead.
    """
    reference = pytest.importorskip("pymaster")
    nside = 32
    npix = 12 * nside**2
    lmax = 3 * nside - 1
    rng = np.random.default_rng(11)
    theta = hp.pix2ang(nside, np.arange(npix))[0]
    mask = np.clip((np.cos(theta) + 0.35) / 0.7, 0, 1) ** 2
    maps = [rng.normal(size=npix) for _ in range(3)]
    field_maps = [mask * maps[0]] if spin == 0 else [maps[1], maps[2]]

    ref = reference.NmtField(mask, field_maps, n_iter=3)
    rw = reference.NmtWorkspace()
    rw.compute_coupling_matrix(ref, ref, reference.NmtBin.from_lmax_linear(lmax, 12))
    ref_out = np.asarray(
        rw.decouple_cell(reference.compute_coupled_cell(ref, ref))
    )

    bins = nmt.NmtBin.from_lmax_linear(lmax, 12)
    jm, jmaps = jnp.asarray(mask), [jnp.asarray(m) for m in field_maps]

    def pipeline():
        f = nmt.NmtField(jm, jmaps, n_iter=3)
        w = nmt.NmtWorkspace()
        w.compute_coupling_matrix(f, f, bins)
        out = w.decouple_cell(nmt.compute_coupled_cell(f, f))
        jax.block_until_ready(out)
        return np.asarray(out)

    _clear()
    fp64 = pipeline()
    np.testing.assert_allclose(fp64, ref_out, rtol=1e-9, atol=1e-9)

    utils.set_table_precision("fp32")
    _clear()
    fp32 = pipeline()
    scale = max(float(np.max(np.abs(fp64))), 1e-300)
    assert float(np.max(np.abs(fp32 - fp64))) / scale < 1e-6
