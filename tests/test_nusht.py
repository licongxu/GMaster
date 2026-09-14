"""``gmaster.nusht``: the GPU general (non-uniform) spherical harmonic transform.

Every claim here is checked against ``ducc0.sht.experimental``, which is the reference the module
is written to be a drop-in replacement for.  The accuracy floor is the march's float32 arithmetic
(~4e-7 at L = 64, ~6e-6 at lmax 4095), not the NUFFT, so the tolerances below are set from that.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import ducc0

from gmaster import nusht


_HAS_NVIDIA_GPU = any(
    device.platform == "gpu" and "NVIDIA" in device.device_kind.upper()
    for device in jax.devices()
)
pytestmark = pytest.mark.skipif(not _HAS_NVIDIA_GPU, reason="needs an NVIDIA GPU")

L = 64
LMAX = L - 1
TOL = 5e-6


def _alm(spin, rng):
    ncomp = 1 if spin == 0 else 2
    nelem = L * (L + 1) // 2
    mstart = nusht.default_mstart(LMAX)
    alm = rng.normal(size=(ncomp, nelem)) + 1j * rng.normal(size=(ncomp, nelem))
    for m in range(L):
        alm[:, int(mstart[m]) + np.arange(m, L)] *= np.arange(m, L) >= max(m, abs(spin))
    alm[:, :L] = alm[:, :L].real
    return np.ascontiguousarray(alm), mstart


def _points(rng, n=2000):
    loc = np.empty((n, 2))
    loc[:, 0] = np.arccos(rng.uniform(-1, 1, n))
    loc[:, 1] = rng.uniform(0, 2 * np.pi, n)
    return loc


def test_grid_for_covers_the_band_limit():
    for lmax in (63, 1023, 4095):
        ntheta, nphi = nusht.grid_for(lmax)
        assert ntheta >= lmax + 1 and nphi >= 2 * lmax + 1
        assert ntheta % 2 == 0 and nphi % 2 == 0
    assert nusht.grid_for(4095) == (4096, 8192)      # powers of two at lmax = 2^k - 1
    with pytest.raises(ValueError):
        nusht.grid_for(63, ntheta=10)


@pytest.mark.parametrize("spin", [0, 2])
def test_ring_spectrum_matches_ducc(spin):
    """Stage 1: the march on the Fejer-1 rings reproduces ducc0's isolatitude synthesis."""
    rng = np.random.default_rng(5)
    alm, mstart = _alm(spin, rng)
    ntheta, nphi = nusht.grid_for(LMAX)
    theta = nusht._thetas(ntheta)
    ncomp = alm.shape[0]
    mp = ducc0.sht.experimental.synthesis(
        alm=alm, spin=spin, lmax=LMAX, mmax=LMAX, theta=theta,
        nphi=np.full(ntheta, nphi, np.uint64), phi0=np.zeros(ntheta),
        ringstart=(np.arange(ntheta) * nphi).astype(np.uint64), mstart=mstart,
        nthreads=4).reshape(ncomp, ntheta, nphi)
    field = mp[0] if spin == 0 else mp[0] + 1j * mp[1]
    want = np.fft.fft(field, axis=1) / nphi

    garr, npad = nusht._geometry(ntheta, L, spin)
    tabs = nusht._tables(ntheta, L, spin)
    idx, ok, _b, _bk = nusht._alm_index(LMAX, LMAX, tuple(int(v) for v in mstart), 1, abs(spin))
    a = [nusht._unpack(jnp.asarray(alm[k]), idx, ok, L=L) for k in range(ncomp)]
    got = np.zeros((nphi, ntheta), complex)
    if spin == 0:
        pos = np.asarray(nusht._march_syn_s0(a[0], garr, tabs, L=L, npad=npad, ntheta=ntheta))
        got[:L] = pos
        got[nphi - L + 1:] = np.conj(pos[1:L][::-1])
    else:
        rs = (1.0 - 2.0 * (jnp.arange(L) % 2))[None, :]
        pos, neg = nusht._march_syn_spin(
            -(a[0] + 1j * a[1]), -rs * jnp.conj(a[0] - 1j * a[1]), garr, tabs,
            L=L, npad=npad, ntheta=ntheta, spin=spin)
        got[:L] = np.asarray(pos)
        got[nphi - L + 1:] = np.asarray(neg)[1:L][::-1]
    assert np.max(np.abs(got.T - want)) / np.max(np.abs(want)) < TOL


@pytest.mark.parametrize("spin", [0, 2])
def test_march_is_transpose(spin):
    """``_ffi_analysis`` is the transpose of ``_ffi_synthesis`` -- for the direct channel as it
    stands, and for the mirror channel once ``(-1)^(ell + s)`` is applied (which is what
    ``_march_ana_spin`` does)."""
    ntheta, _ = nusht.grid_for(LMAX)
    garr, npad = nusht._geometry(ntheta, L, spin)
    tabs = nusht._tables(ntheta, L, spin)
    rng = np.random.default_rng(3)

    def rc(*shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    tri = np.tril(np.ones((L, L), bool))
    zero_a = jnp.zeros((L, L), complex)
    zero_g = jnp.zeros((L, ntheta), complex)
    a = jnp.asarray(np.where(tri, rc(L, L), 0))
    g = jnp.asarray(rc(L, ntheta))
    if spin == 0:
        f = nusht._march_syn_s0(a, garr, tabs, L=L, npad=npad, ntheta=ntheta)
        b = nusht._march_ana_s0(g, garr, tabs, L=L, npad=npad, ntheta=ntheta)
        pairs = [(np.sum(np.asarray(f) * np.asarray(g)), np.sum(np.asarray(a) * np.asarray(b)))]
    else:
        kw = dict(L=L, npad=npad, ntheta=ntheta, spin=spin)
        fp, _ = nusht._march_syn_spin(a, zero_a, garr, tabs, **kw)
        bp, _ = nusht._march_ana_spin(g, zero_g, garr, tabs, **kw)
        _, fm = nusht._march_syn_spin(zero_a, a, garr, tabs, **kw)
        _, bm = nusht._march_ana_spin(zero_g, g, garr, tabs, **kw)
        pairs = [(np.sum(np.asarray(fp) * np.asarray(g)), np.sum(np.asarray(a) * np.asarray(bp))),
                 (np.sum(np.asarray(fm) * np.asarray(g)), np.sum(np.asarray(a) * np.asarray(bm)))]
    for lhs, rhs in pairs:
        assert abs(lhs - rhs) / abs(lhs) < TOL


@pytest.mark.parametrize("spin", [0, 2])
def test_synthesis_general_matches_ducc(spin):
    rng = np.random.default_rng(7)
    alm, mstart = _alm(spin, rng)
    loc = _points(rng)
    want = ducc0.sht.experimental.synthesis_general(
        alm=alm, spin=spin, lmax=LMAX, mmax=LMAX, mstart=mstart, loc=loc, epsilon=1e-12,
        nthreads=4)
    got = np.asarray(nusht.synthesis_general(alm, spin=spin, lmax=LMAX, loc=loc, epsilon=1e-9))
    assert got.shape == want.shape
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < TOL


@pytest.mark.parametrize("spin", [0, 2])
def test_adjoint_synthesis_general_matches_ducc(spin):
    rng = np.random.default_rng(9)
    _, mstart = _alm(spin, rng)
    loc = _points(rng)
    values = np.ascontiguousarray(rng.normal(size=(1 if spin == 0 else 2, loc.shape[0])))
    want = ducc0.sht.experimental.adjoint_synthesis_general(
        map=values, spin=spin, lmax=LMAX, mmax=LMAX, mstart=mstart, loc=loc, epsilon=1e-12,
        nthreads=4)
    got = np.asarray(nusht.adjoint_synthesis_general(
        values, spin=spin, lmax=LMAX, loc=loc, epsilon=1e-9))
    assert got.shape == want.shape
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < TOL


@pytest.mark.parametrize("upsampfac", [1.25, 2.0])
def test_upsampfac_does_not_change_the_answer(upsampfac):
    rng = np.random.default_rng(11)
    alm, mstart = _alm(0, rng)
    loc = _points(rng)
    want = ducc0.sht.experimental.synthesis_general(
        alm=alm, spin=0, lmax=LMAX, mmax=LMAX, mstart=mstart, loc=loc, epsilon=1e-12, nthreads=4)
    got = np.asarray(nusht.synthesis_general(
        alm, spin=0, lmax=LMAX, loc=loc, epsilon=1e-6, upsampfac=upsampfac))
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < TOL


def test_custom_mstart_packing():
    """A non-default (but still valid) mstart must give the same field."""
    rng = np.random.default_rng(13)
    alm, mstart = _alm(0, rng)
    loc = _points(rng, 500)
    stride = L                                   # dense (mmax+1, lmax+1) rectangle
    mstart2 = (np.arange(L) * stride).astype(np.uint64)
    alm2 = np.zeros((1, L * L), complex)
    for m in range(L):
        alm2[0, int(mstart2[m]) + np.arange(m, L)] = alm[0, int(mstart[m]) + np.arange(m, L)]
    want = np.asarray(nusht.synthesis_general(alm, spin=0, lmax=LMAX, loc=loc, epsilon=1e-9))
    got = np.asarray(nusht.synthesis_general(alm2, spin=0, lmax=LMAX, loc=loc, epsilon=1e-9,
                                             mstart=mstart2))
    assert np.max(np.abs(got - want)) / np.max(np.abs(want)) < 1e-14
