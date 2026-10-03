"""``gmaster.lensing``: quadratic lensing estimators and their normalisation.

* The odd-spin (1, 3) transforms the estimators use match healpy.
* ``LensingQE.qe_all`` matches falafel's ``qe_all`` (HEALPix pixelization), estimator by estimator.
* ``normalization`` matches an exact sum over Wigner 3j symbols (ducc0's Schulten-Gordon
  recursion) of the Okamoto-Hu response, and tempura where it is installed.
* On first-order lensed simulations, normalised reconstructions recover the input potential.

The transform comparisons are limited by the float32 march (~1e-5 relative to the map maximum).
"""

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import healpy as hp

from gmaster._sht.cuda_gpu import on_cuda_gpu
from gmaster.lensing import LensingQE, _est_terms, normalization

pytestmark = pytest.mark.march_v2
needs_gpu = pytest.mark.skipif(not on_cuda_gpu(), reason="the odd-spin march needs an NVIDIA GPU")

ESTS = ["TT", "TE", "EE", "EB", "TB", "mv", "mvpol"]


def _spectra(lmax, noise_arcmin=6.0):
    """Smooth, CMB-like spectra (no CAMB dependency) and diagonal-filter totals."""
    ell = np.arange(lmax + 1.0)
    shape = 1e3 / (ell + 10.0) ** 2 * np.exp(-((ell / 900.0) ** 2))
    ucls = {"TT": 2.0e3 * shape, "EE": 40.0 * shape * (ell / (ell + 30.0)) ** 2,
            "BB": 0.5 * shape * (ell / (ell + 30.0)) ** 2, "TE": 150.0 * shape * np.cos(ell / 60.0)}
    nl = (noise_arcmin * np.pi / 10800) ** 2 * np.ones(lmax + 1)
    tcls = {"TT": ucls["TT"] + nl, "EE": ucls["EE"] + 2 * nl, "BB": ucls["BB"] + 2 * nl,
            "TE": ucls["TE"]}
    return ucls, tcls


def _filtered_alms(lmax, tcls, seed=1):
    rng = np.random.default_rng(seed)
    n = hp.Alm.getsize(lmax)
    _, m = hp.Alm.getlm(lmax)
    out = []
    for key in ("TT", "EE", "BB"):
        a = (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2)
        a[m == 0] = a[m == 0].real * np.sqrt(2)
        f = np.where(np.arange(lmax + 1) >= 2, 1 / np.sqrt(tcls[key]), 0.0)
        out.append(hp.almxfl(a, f))
    return out


# ------------------------------------------------------------------ odd-spin transforms
@needs_gpu
@pytest.mark.parametrize("spin", [1, 2, 3])
def test_spin_transforms_match_healpy(spin):
    import gmaster as nmt

    nside, lmax = 64, 128
    minfo, ainfo = nmt.NmtMapInfo(None, [12 * nside ** 2]), nmt.NmtAlmInfo(lmax)
    rng = np.random.default_rng(spin)
    n = hp.Alm.getsize(lmax)
    ell, m = hp.Alm.getlm(lmax)
    alm = rng.normal(size=(2, n)) + 1j * rng.normal(size=(2, n))
    alm[:, m == 0] = alm[:, m == 0].real
    alm[:, ell < spin] = 0
    ref = np.array(hp.alm2map_spin(list(alm), nside, spin, lmax))
    got = np.asarray(nmt.alm2map(alm, spin, minfo, ainfo))
    assert np.abs(got - ref).max() < 2e-5 * np.abs(ref).max()
    aref = np.array(hp.map2alm_spin(list(ref), spin, lmax))
    agot = np.asarray(nmt.map2alm(ref, spin, minfo, ainfo, n_iter=0))
    assert np.abs(agot - aref).max() < 2e-5 * np.abs(aref).max()


# ----------------------------------------------------------------- estimator vs falafel
@needs_gpu
def test_qe_all_matches_falafel():
    fqe = pytest.importorskip("falafel.qe")
    nside, mlmax = 128, 256
    ucls, tcls = _spectra(mlmax)
    fT, fE, fB = _filtered_alms(mlmax, tcls)
    ref = fqe.qe_all(fqe.pixelization(nside=nside), ucls, mlmax,
                     fTalm=fT, fEalm=fE, fBalm=fB, estimators=ESTS)
    got = LensingQE(nside, mlmax).qe_all(ucls, fTalm=fT, fEalm=fE, fBalm=fB, estimators=ESTS)
    for e in ESTS:
        for c in (0, 1):
            r, g = np.asarray(ref[e][c]), np.asarray(got[e][c])
            assert np.abs(g - r).max() < 3e-5 * np.abs(r).max(), (e, c)


@needs_gpu
def test_qe_mv_is_sum_of_estimators():
    """falafel's mv (and mvpol) is the unnormalised sum of the separate estimators."""
    nside, mlmax = 64, 128
    ucls, tcls = _spectra(mlmax)
    fT, fE, fB = _filtered_alms(mlmax, tcls)
    got = LensingQE(nside, mlmax).qe_all(ucls, fTalm=fT, fEalm=fE, fBalm=fB, estimators=ESTS)
    total = sum(np.asarray(got[e]) for e in ("TT", "TE", "EE", "EB", "TB"))
    pol = sum(np.asarray(got[e]) for e in ("EE", "EB"))
    scale = np.abs(np.asarray(got["mv"])).max()
    assert np.abs(np.asarray(got["mv"]) - total).max() < 1e-5 * scale
    assert np.abs(np.asarray(got["mvpol"]) - pol).max() < 1e-5 * np.abs(pol).max()


# ------------------------------------------------------------------------ normalisation
def _exact_response(ests, L, lmin, lmax, u, filt):
    """``R_L`` (gradient, curl) summed directly over the band ``|l1 - l2| <= L`` with exact 3j."""
    import ducc0

    spec = {e: _est_terms(e, u, u) for e in ests}
    out = {e: np.zeros(2) for e in ests}
    sqL = np.sqrt(L * (L + 1.0))
    for l1 in range(lmin, lmax + 1):
        l2 = np.arange(l1 - L, l1 + L + 1)
        ok = (l2 >= lmin) & (l2 <= lmax)
        l2c = np.clip(l2, 0, lmax)
        sign_sum = np.where((l1 + l2 + L) % 2 == 0, 1.0, -1.0)
        cache = {}

        def tj(m, m3):                      # (l2 L l1; -m-m3, m, m3) on the band
            if (m, m3) not in cache:
                arr = np.zeros(2 * L + 1)
                if l1 >= abs(m3):
                    lo, vals = ducc0.misc.wigner3j_int(L, l1, m, m3)
                    idx = np.arange(lo, lo + len(vals)) - (l1 - L)
                    keep = (idx >= 0) & (idx < 2 * L + 1)
                    arr[idx[keep]] = vals[keep]
                cache[(m, m3)] = arr
            return cache[(m, m3)]

        def F(s, outer, curl):
            d = l1 if outer == 2 else l2c
            am = np.sqrt(np.maximum((d + s) * (d - s + 1.0), 0.0))
            ap = np.sqrt(np.maximum((d - s) * (d + s + 1.0), 0.0))
            if outer == 2:
                jm, jp = tj(-1, 1 - s), tj(1, -1 - s)
            else:
                jm, jp = sign_sum * tj(-1, s), sign_sum * tj(1, s)
            return -0.5 * sqL * (am * jm + (-1.0 if curl else 1.0) * ap * jp)

        def coef(t):
            v = t[4] * np.ones(2 * L + 1)
            if t[2] is not None:
                v = v * t[2][l1]
            if t[3] is not None:
                v = v * t[3][l2c]
            return v

        n12 = (2.0 * l1 + 1.0) * (2.0 * l2c + 1.0) / (4.0 * np.pi)
        for e, ((X, Y), Wt, ft, parity) in spec.items():
            fil = filt[X][l1] * np.where(ok, filt[Y][l2c], 0.0) * n12
            for c, curl in enumerate((False, True)):
                sigma = -parity if curl else parity
                sector = np.where(sign_sum == sigma, fil, 0.0)
                for wt in Wt:
                    fw = F(wt[0], wt[1], curl) * coef(wt) * sector
                    for gt in ft:
                        cross = sigma if wt[1] != gt[1] else 1.0
                        out[e][c] += cross * np.sum(fw * F(gt[0], gt[1], curl) * coef(gt))
    return out


def test_normalization_matches_exact_3j_sum():
    pytest.importorskip("ducc0")
    lmin, lmax = 5, 60
    ucls, tcls = _spectra(lmax)
    ell = np.arange(lmax + 1)
    u = {k: np.where(ell >= lmin, v, 0.0) for k, v in ucls.items()}
    filt = {X: np.where(ell >= lmin, 1 / tcls[k], 0.0) for X, k in (("T", "TT"), ("E", "EE"), ("B", "BB"))}
    base = ["TT", "TE", "EE", "EB", "TB"]
    got = normalization(base, ucls, tcls, lmin, lmax)
    for L in (2, 3, 7, 20, 41, 60):
        exact = _exact_response(base, L, lmin, lmax, u, filt)
        for e in base:
            for c in (0, 1):
                assert got[e][c][L] * exact[e][c] == pytest.approx(1.0, abs=1e-9), (e, c, L)


def test_normalization_mv_combines_inverse_noise():
    lmin, lmax = 2, 80
    ucls, tcls = _spectra(lmax)
    A = normalization(["TT", "TE", "EE", "EB", "TB", "mv", "mvpol"], ucls, tcls, lmin, lmax)
    for c, lo in ((0, 1), (1, 2)):
        inv = sum(1 / A[e][c][lo:] for e in ("TT", "TE", "EE", "EB", "TB"))
        np.testing.assert_allclose(A["mv"][c][lo:], 1 / inv, rtol=1e-12)
        np.testing.assert_allclose(A["mvpol"][c][lo:], 1 / (1 / A["EE"][c][lo:] + 1 / A["EB"][c][lo:]),
                                   rtol=1e-12)


def test_normalization_matches_tempura():
    pytempura = pytest.importorskip("pytempura")
    lmin, lmax, Lmax = 30, 600, 500
    ucls, tcls = _spectra(lmax)
    ests = ["TT", "TE", "EE", "EB", "TB", "MV", "MVPOL"]
    got = normalization(ests, ucls, tcls, lmin, lmax, Lmax=Lmax)
    ref = pytempura.get_norms(ests, ucls, ucls, tcls, lmin, lmax, k_ellmax=Lmax)
    for e in ests:
        for c in (0, 1):
            # tempura's own low-L cancellation error is ~1e-4; it is exact at L < Lmax above that.
            np.testing.assert_allclose(got[e][c][2:Lmax], ref[e][c][2:Lmax], rtol=3e-4, err_msg=e)
            np.testing.assert_allclose(got[e][c][50:Lmax], ref[e][c][50:Lmax], rtol=1e-6, err_msg=e)


# ---------------------------------------------------------------- end-to-end response
@needs_gpu
def test_reconstruction_recovers_input_lensing():
    """First-order lensed sims (lensed with healpy, independently of GMaster) -> QE -> A_L.

    The estimator's first-order term ``Q[dX, X] + Q[X, dX]`` (``dX`` the lensing of the sky
    ``X``) is formed with the separate gradient-leg inputs, so neither the Gaussian noise nor the
    second-order term enters; its cross-spectrum with the input potential, over the input's
    auto-spectrum, must be 1 for the gradient and 0 for the curl (averaged over CMB skies).
    """
    nside, lmax, lmin = 256, 512, 2
    ell = np.arange(lmax + 1.0)
    ucls, _ = _spectra(lmax)
    ucls["BB"] = np.zeros(lmax + 1)
    nl = (3 * np.pi / 10800) ** 2 * np.ones(lmax + 1)
    tcls = {"TT": ucls["TT"] + nl, "EE": ucls["EE"] + 2 * nl, "BB": 2 * nl}
    clpp = np.where(ell > 1, 1.5e-7 / ((ell + 1.0) ** 2 * (ell + 30.0) ** 2), 0.0)   # L^4 C ~ 1e-7
    A = normalization(ESTS, ucls, tcls, lmin, lmax)
    qe = LensingQE(nside, lmax)
    sq = lambda f: np.sqrt(np.maximum(f, 0))
    np.random.seed(7)
    # The lensing products reach 2 lmax, so the sky is lensed at an nside resolving them.
    nside_sim = 2 * nside

    def spin1(alm, f):
        q, u = hp.alm2map_spin([hp.almxfl(alm, f), 0 * alm], nside_sim, 1, lmax)
        return q + 1j * u

    f3, f1 = sq((ell - 2) * (ell + 3)), sq((ell + 2) * (ell - 1))
    fl = {k: np.where(ell >= lmin, 1 / tcls[k], 0.0) for k in ("TT", "EE", "BB")}

    def filt(t, e, b):
        return hp.almxfl(t, fl["TT"]), hp.almxfl(e, fl["EE"]), hp.almxfl(b, fl["BB"])

    def linear(sky, lensed):
        """Q[d sky (gradient leg), sky] + Q[sky (gradient leg), d sky]: the first-order term."""
        (t0, e0, b0), (t1, e1, b1) = filt(*sky), filt(*lensed)
        a = qe.qe_all(ucls, fTalm=t0, fEalm=e0, fBalm=b0, xfTalm=t1, xfEalm=e1, xfBalm=b1,
                      estimators=ESTS)
        b = qe.qe_all(ucls, fTalm=t1, fEalm=e1, fBalm=b1, xfTalm=t0, xfEalm=e0, xfBalm=b0,
                      estimators=ESTS)
        return {k: np.asarray(a[k]) + np.asarray(b[k]) for k in ESTS}

    # Independent (CMB, phi) realisations; the response is checked against its standard error.
    nsim = 10
    band = slice(60, 400)
    ratios = {e: np.zeros((nsim, 2)) for e in ESTS}
    for i in range(nsim):
        talm, ealm, balm = hp.synalm([ucls["TT"], ucls["EE"], ucls["BB"], ucls["TE"]], lmax=lmax,
                                     new=True)
        palm = hp.synalm(clpp, lmax=lmax)
        auto = hp.alm2cl(palm)[band].sum()
        dphi = spin1(palm, -sq(ell * (ell + 1)))
        T1 = np.real(dphi * np.conj(spin1(talm, -sq(ell * (ell + 1)))))
        q3, u3 = hp.alm2map_spin([hp.almxfl(ealm, f3), hp.almxfl(balm, f3)], nside_sim, 3, lmax)
        q1, u1 = hp.alm2map_spin([hp.almxfl(ealm, f1), hp.almxfl(balm, f1)], nside_sim, 1, lmax)
        P1 = 0.5 * (dphi * -(q1 + 1j * u1) + np.conj(dphi) * (q3 + 1j * u3))
        t1 = hp.map2alm(T1, lmax=lmax, iter=3)
        e1, b1 = hp.map2alm_spin([P1.real, P1.imag], 2, lmax)
        lin = linear((talm, ealm, balm), (t1, e1, b1))
        for e in ESTS:
            for c in (0, 1):
                d = hp.almxfl(lin[e][c], A[e][c])
                ratios[e][i, c] = hp.alm2cl(d, palm)[band].sum() / auto
    for e in ESTS:
        mean = ratios[e].mean(axis=0)
        sem = ratios[e].std(axis=0, ddof=1) / np.sqrt(nsim)
        print(f"{e}: gradient response {mean[0]:.4f} +- {sem[0]:.4f}, curl {mean[1]:+.4f} +- {sem[1]:.4f}")
        assert abs(mean[0] - 1) < 4 * sem[0] + 0.005, e
        assert abs(mean[1]) < 4 * sem[1] + 0.005, e
