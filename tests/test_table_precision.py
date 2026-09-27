"""Precision controls: table precision, ring (azimuthal) precision and coupling precision.

These tests cover the defaults, the effect of switching, and cache invalidation.  The
session fixture in ``conftest.py`` always runs this module at fp64.
"""

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import gmaster as nmt
from gmaster import utils, workspaces
from gmaster._sht import spin_slice as _spin_slice
from gmaster._sht import theta_matrix as _theta_matrix
from gmaster._sht.cuda_gpu import on_cuda_gpu
from gmaster._sht import rings

_HAS_NVIDIA_GPU = on_cuda_gpu()


@pytest.fixture(autouse=True)
def _restore_precision():
    """Restore the default precisions and drop the table caches after each test."""
    yield
    utils.set_table_precision("fp64")
    utils.set_ring_precision("follow")
    nmt.set_coupling_precision("auto")     # the default: fp32 where the v2 march is used
    _theta_matrix.release()
    _spin_slice.clear_cache()


def _clear():
    """Drop the Legendre-band and Wigner-d table caches."""
    _theta_matrix.release()
    _spin_slice.clear_cache()


def test_float64_is_the_default():
    """Tables are stored in float64 unless the user asks otherwise."""
    assert utils.nmt_params.table_dtype == "fp64"
    assert utils.table_dtype() == jnp.float64


def test_unknown_precision_is_rejected():
    """Every precision setter rejects an unsupported value."""
    with pytest.raises(KeyError):
        utils.set_table_precision("fp16")
    with pytest.raises(KeyError):
        utils.set_ring_precision("fp16")
    with pytest.raises(KeyError):
        nmt.set_coupling_precision("fp16")


def test_ring_precision_follows_the_tables_only_while_it_follows():
    """Under `follow` the ring precision tracks the table precision; `fp64`/`fp32` pin it.

    This matters for accuracy: the analysis chirp-Z casts the map pixels to the chirp's
    real dtype, so under `follow` an fp32 table session analyses the map in float32 and
    the ring sums are only good to ~1e-7.  With rings pinned to `fp64` the same session
    reproduces the float64 ring sums exactly.
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
    """Changing the ring precision rebuilds the chirp tables in the new dtype, with the same values."""
    nside, L = 32, 96
    fp64 = rings._ring_analysis_tables(L, nside)
    assert fp64[0].dtype == jnp.complex128

    utils.set_ring_precision("fp32")
    fp32 = rings._ring_analysis_tables(L, nside)
    assert fp32[0].dtype == jnp.complex64
    # The chirp is a phase ramp.  The complex64 one is evaluated in float32 (exactly reduced
    # angle, float32 sincos), so it agrees with the rounded float64 ramp to a few float32 ulps.
    np.testing.assert_allclose(
        np.asarray(fp32[0]), np.asarray(fp64[0]).astype(np.complex64),
        rtol=0.0, atol=4 * np.finfo(np.float32).eps)


def test_band_storage_follows_the_selection():
    """The Legendre band is stored in the selected table dtype."""
    nside, L = 32, 96
    _clear()
    band = _theta_matrix._band(_theta_matrix.band_geometry(nside, L))
    assert band[0][0].dtype == jnp.float64

    utils.set_table_precision("fp32")
    band32 = _theta_matrix._band(_theta_matrix.band_geometry(nside, L))
    assert band32[0][0].dtype == jnp.float32
    # The recurrence runs in float64 and each value is rounded to float32 as it is
    # stored (so no float64 copy of a block is kept), so the two bands differ only
    # in the exp2-underflow tail, around 1e-38.
    for even, odd in zip(band, band32):
        for full, half in zip(even, odd):
            np.testing.assert_allclose(
                np.asarray(half), np.asarray(full).astype(np.float32),
                rtol=0.0, atol=1e-30)


def test_precision_switch_rebuilds_rather_than_reusing():
    """Switching precision clears the band cache and keys new entries by dtype, so tables are never mixed."""
    nside, L = 32, 96
    _clear()
    fp64_key = _theta_matrix.band_geometry(nside, L)
    assert _theta_matrix._band(fp64_key) is not None
    assert len(_theta_matrix._BAND_CACHE) == 1
    utils.set_table_precision("fp32")
    # The switch clears the cache, and the geometry key includes the dtype, so a
    # caller can never receive tables of the other precision.
    assert len(_theta_matrix._BAND_CACHE) == 0
    fp32_key = _theta_matrix.band_geometry(nside, L)
    assert fp32_key != fp64_key
    assert _theta_matrix._band(fp32_key) is not None
    assert len(_theta_matrix._BAND_CACHE) == 1
    assert _theta_matrix._band(fp64_key) is not None
    assert len(_theta_matrix._BAND_CACHE) == 2


@pytest.mark.parametrize("prec", ["fp64", "fp32"])
def test_contracts_reduce_in_the_storage_width_without_truncating(prec):
    """Band contractions reduce in the storage precision and return complex128 without truncation.

    Both contractions reduce in the table's storage precision and widen the partial sums
    afterwards.  The cast must be applied to the real and imaginary parts separately:
    casting a complex128 operand to a real table dtype discards its imaginary part, which
    would silently zero half of the synthesis sum in float64 as well as float32.
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
    """Memory estimates for the band and the Wigner-d triangle halve at float32."""
    fp64 = _theta_matrix.band_bytes(128, 384, dtype=jnp.float64)
    fp32 = _theta_matrix.band_bytes(128, 384, dtype=jnp.float32)
    assert fp32 * 2 == fp64
    tri64 = _spin_slice.triangle_bytes(128, 384, jnp.float64)
    tri32 = _spin_slice.triangle_bytes(128, 384, jnp.float32)
    assert tri32 * 2 == tri64


@pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="SHT dispatch requires an NVIDIA GPU")
@pytest.mark.parametrize("spin", [0, 2])
def test_float32_tables_keep_the_coupling_matrix(spin):
    """float32 tables change the decoupled Cls only by their representation error.

    The full pipeline with float32 tables agrees with the float64 one to ~5e-8 relative on
    the decoupled Cls; the 1e-6 bound adds headroom.  float64 remains the default because
    it agrees with pymaster to ~1e-12.
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


def test_float32_coupling_quadrature_keeps_the_cells():
    """`set_coupling_precision("fp32")` really switches the polarised coupling quadrature.

    float32 operands in `_general_coupling_matrix_quadrature` move the coupling matrix by
    ~2e-6 relative to float64 (well inside the 1e-4 bound), which is below the error that
    float32 tables already contribute to the decoupled Cls.  The difference must also be
    nonzero, which proves the switch took effect.
    """
    lmax = 95
    rng = np.random.default_rng(7)
    ell = np.arange(lmax + 1)
    pcl = np.exp(-ell / 60.0) * (1.0 + 0.1 * rng.normal(size=ell.size))

    def matrix():
        return np.asarray(nmt.get_general_coupling_matrix(pcl, 2, 2, 2, 2, parity="both"))

    nmt.set_coupling_precision("fp64")
    jax.clear_caches()
    fp64 = matrix()

    nmt.set_coupling_precision("fp32")
    # The flag is read at trace time, so without clearing the caches a program compiled
    # earlier in this process would keep its float64 operands and the test would be vacuous.
    jax.clear_caches()
    fp32 = matrix()

    nmt.set_coupling_precision("fp64")
    jax.clear_caches()

    scale = float(np.max(np.abs(fp64)))
    rel = float(np.max(np.abs(fp32 - fp64))) / scale
    # Exactly zero here means the switch was ignored, not that it was accurate.
    assert rel > 0.0
    assert rel < 1e-4


def test_float32_scalar_coupling_keeps_the_matrix():
    """`set_coupling_precision("fp32")` also switches the scalar (TT) coupling recurrence.

    `_coupling_matrix_tt` keeps its log-cumsum table and offset accumulator in float64 and
    rounds only the per-term products, so the float32 matrix is within ~2e-7 relative of the
    float64 one (bound 1e-5).  Above ``_TT_QUADRATURE_LMAX`` the dispatcher uses the fp64
    quadrature instead, so this test stays below that threshold.
    """
    lmax = 31
    rng = np.random.default_rng(17)
    ell = np.arange(2 * lmax + 1)
    pcl = np.exp(-ell / 90.0) * (1.0 + 0.1 * rng.normal(size=ell.size))

    def matrix():
        return np.asarray(workspaces._coupling_matrix_tt(jnp.asarray(pcl), lmax=lmax))

    nmt.set_coupling_precision("fp64")
    jax.clear_caches()
    fp64 = matrix()

    nmt.set_coupling_precision("fp32")
    jax.clear_caches()
    fp32 = matrix()

    nmt.set_coupling_precision("fp64")
    jax.clear_caches()

    scale = float(np.max(np.abs(fp64)))
    rel = float(np.max(np.abs(fp32 - fp64))) / scale
    assert rel > 0.0
    assert rel < 1e-5


