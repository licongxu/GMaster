"""Curved-sky CMB lensing quadratic estimators on GMaster's GPU transforms.

The estimators are those of falafel (``falafel.qe.qe_all``, simonsobs/falafel), the
reconstruction code of the ACT DR6 and SO lensing pipelines, term for term and in its
conventions, on a HEALPix grid.  Every estimator is a product of real-space maps,

    Q[a, b] = sqrt(L (L+1)) A_1[ S_1(g_l a_lm) S_0(b_lm) ]           (temperature)

where ``S_s`` is a spin-s synthesis and ``A_1`` the spin-1 analysis, so its cost is that of the
transforms.  The polarised legs use spin-1, spin-2 and spin-3 transforms.  All transforms run
on the GPU through :func:`gmaster.alm2map` / :func:`gmaster.map2alm`.

Inputs are inverse-variance (C^-1) filtered alms; outputs are the unnormalised gradient and curl
alms ``(2, nalm)``, as from falafel.  Multiply by the normalisation ``A_L`` (from
:func:`normalization` or any other source) to obtain ``phi_LM`` / ``omega_LM``.

Example
-------
>>> qe = LensingQE(nside=2048, mlmax=4000)
>>> rec = qe.qe_all(ucls, fTalm=t, fEalm=e, fBalm=b, estimators=["TT", "mv"])
>>> phi_tt = qe.almxfl(rec["TT"][0], norm_tt)
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .utils import NmtAlmInfo, NmtMapInfo, alm2map, map2alm

__all__ = ["LensingQE", "ESTIMATORS", "normalization", "gauss_legendre"]

ESTIMATORS = ("TT", "TE", "EE", "EB", "TB", "mv", "mvpol")


class LensingQE:
    """Quadratic lensing estimators on a HEALPix grid.

    Parameters
    ----------
    nside : int
        HEALPix resolution of the real-space products.
    mlmax : int
        Band limit of the input and output alms (healpy-packed, ``mmax = mlmax``).
    n_iter : int
        Jacobi iterations of the final spin-1 analysis.  0 (the default) is falafel's
        HEALPix behaviour (``healpy.map2alm_spin``).
    """

    # Array module of the estimator graph; the transforms are `_synth` and
    # `deflection_map_to_phi_curl_alms`, so a subclass can run the same graph elsewhere.
    xp = jnp

    def __init__(self, nside, mlmax, n_iter=0):
        self.nside = int(nside)
        if self.nside < 1 or self.nside & (self.nside - 1):
            raise ValueError(f"nside must be a power of two, got {nside}")
        self.n_iter = int(n_iter)
        self.minfo = NmtMapInfo(None, [12 * self.nside ** 2])
        self.ainfo = NmtAlmInfo(int(mlmax))
        self._set_band_limit(mlmax, self.ainfo._ell)

    def _set_band_limit(self, mlmax, ell):
        self.mlmax = int(mlmax)
        self.ell = self.xp.asarray(ell)
        ells = np.arange(self.mlmax + 1, dtype=np.float64)
        self._fl_grad = _below(np.sqrt(ells * (ells + 1.0)), 2)
        self._fl_m1 = _below(np.sqrt(np.maximum((ells - 1.0) * (ells + 2.0), 0.0)), 2)
        self._fl_p3 = _below(np.sqrt(np.maximum((ells - 2.0) * (ells + 3.0), 0.0)), 2)
        self._fl_kappa = np.sqrt(ells * (ells + 1.0))

    # ---------------------------------------------------------------- harmonic helpers
    def almxfl(self, alm, fl):
        """``alm * fl[ell]``; ``fl`` indexed from ell = 0, at least ``mlmax + 1`` long."""
        fl = np.asarray(fl, dtype=np.float64)
        if fl.shape[-1] <= self.mlmax:
            fl = np.concatenate([fl, np.zeros(self.mlmax + 1 - fl.shape[-1])])
        return self.xp.asarray(alm) * self.xp.asarray(fl[: self.mlmax + 1])[self.ell]

    def _synth(self, alms, spin):
        return alm2map(jnp.stack(alms), spin, self.minfo, self.ainfo)

    def _complex_synth(self, e, b, spin):
        """``Q + iU`` of the spin-``spin`` synthesis of gradient ``e`` and curl ``b``."""
        q, u = self._synth([e, b], spin)
        return q + 1j * u

    # ------------------------------------------------------------- estimator pieces
    def deflection_map_to_phi_curl_alms(self, dmap):
        """Spin-1 analysis of a complex deflection-like map, times ``sqrt(L(L+1))``.

        Returns ``(2, nalm)``: the gradient (lensing) and curl components.
        """
        maps = jnp.stack([jnp.real(dmap), jnp.imag(dmap)])
        res = map2alm(maps, 1, self.minfo, self.ainfo, n_iter=self.n_iter)
        return self.almxfl(res, self._fl_kappa)

    def temperature_deflection(self, xalm, yalm):
        """falafel ``qe_spin_temperature_deflection``: ``-(grad X)(Y)`` as a complex spin-1 map."""
        e = -self.almxfl(xalm, self._fl_grad)
        grad = self._complex_synth(e, self.xp.zeros_like(e), 1)
        ymap = self._synth([yalm], 0)[0]
        return -grad * ymap

    def pol_deflection(self, x_e, x_b, y_e, y_b):
        """falafel ``qe_spin_pol_deflection`` as a complex spin-1 map."""
        # +2 leg: spin-3 synthesis of sqrt((l-2)(l+3)) (E, B), sign -1 (falafel's `gradient_spin`).
        grad_p2 = self._complex_synth(self.almxfl(x_e, self._fl_p3),
                                      self.almxfl(x_b, self._fl_p3), 3)
        # -2 leg: the (Q - iU) helicity of a spin-1 synthesis of -sqrt((l-1)(l+2)) (E, B).
        g1 = self._complex_synth(-self.almxfl(x_e, self._fl_m1),
                                 -self.almxfl(x_b, self._fl_m1), 1)
        grad_m2 = self.xp.conj(g1)
        y = self._complex_synth(y_e, y_b, 2)
        return (-grad_m2 * y - grad_p2 * self.xp.conj(y)) / 2

    # ------------------------------------------------------------------ all estimators
    def qe_all(self, response_cls, fTalm=None, fEalm=None, fBalm=None,
               estimators=("TT", "TE", "EE", "EB", "TB", "mv", "mvpol"),
               xfTalm=None, xfEalm=None, xfBalm=None):
        """Unnormalised lensing estimators, as ``falafel.qe.qe_all``.

        Parameters
        ----------
        response_cls : dict
            Lensed (or gradient) spectra ``"TT", "TE", "EE", "BB"`` from ell = 0, used as the
            Wiener weights of the gradient leg.
        fTalm, fEalm, fBalm : array_like, shape (nalm,)
            C^-1 filtered alms (the inverse-variance leg).
        estimators : sequence of str
            Any of ``"TT", "TE", "EE", "EB", "TB", "mv", "mvpol"`` (also ``"MV"``, ``"MVPOL"``).
        xfTalm, xfEalm, xfBalm : array_like, optional
            Alms for the gradient leg (default: the same as the filtered alms).

        Returns
        -------
        dict of jax.Array, each shape (2, nalm)
            Gradient and curl alms per estimator.
        """
        ests = list(estimators)
        xp = self.xp
        as_alm = lambda a: None if a is None else xp.asarray(a, dtype=xp.complex128)
        fTalm, fEalm, fBalm = as_alm(fTalm), as_alm(fEalm), as_alm(fBalm)
        xfTalm = fTalm if xfTalm is None else as_alm(xfTalm)
        xfEalm = fEalm if xfEalm is None else as_alm(xfEalm)
        xfBalm = fBalm if xfBalm is None else as_alm(xfBalm)
        present = [a for a in (fTalm, fEalm, fBalm) if a is not None]
        if not present:
            raise ValueError("Supply at least one filtered alm")
        zero = xp.zeros_like(present[0])
        alm_cache = {}

        def mixing(specs, alms):
            out = zero
            for spec, alm in zip(specs, alms):
                if alm is not None:
                    out = out + self.almxfl(alm, response_cls[spec])
            return out

        def xalm(name):
            if name not in alm_cache:
                alm_cache[name] = {
                    "t": lambda: mixing(["TT", "TE"], [xfTalm, xfEalm]),
                    "t_e0": lambda: mixing(["TE"], [xfEalm]),
                    "t0": lambda: mixing(["TT"], [xfTalm]),
                    "e": lambda: mixing(["EE", "TE"], [xfEalm, xfTalm]),
                    "e_t0": lambda: mixing(["TE"], [xfTalm]),
                    "e0": lambda: mixing(["EE"], [xfEalm]),
                }[name]()
            return alm_cache[name]

        orz = lambda a: zero if a is None else a
        want = {e.upper() for e in ests}
        unknown = want - {"TT", "TE", "EE", "EB", "TB", "MV", "MVPOL"}
        if unknown:
            raise ValueError(f"Unknown estimators {sorted(unknown)}")
        # mv and mvpol are falafel's sums of the separate unnormalised estimators (the transforms
        # are linear), so only the five base estimators are formed and the combinations added.
        base = set(want - {"MV", "MVPOL"})
        if "MV" in want:
            base |= {"TT", "TE", "EE", "EB", "TB"}
        if "MVPOL" in want:
            base |= {"EE", "EB"}

        # Each distinct leg map is synthesised once (falafel re-synthesises the shared ones per
        # estimator: 24 syntheses and 7 analyses for the full set, here at most 9 and 5) and
        # dropped as soon as the last estimator using it is formed.
        maps = {}

        def leg(name):
            if name not in maps:
                if name == "T":
                    maps[name] = self._synth([orz(fTalm)], 0)[0]
                elif name in ("Pe", "Pb"):
                    e, b = (orz(fEalm), zero) if name == "Pe" else (zero, orz(fBalm))
                    maps[name] = self._complex_synth(e, b, 2)
                elif name.startswith("g:"):          # spin-1 gradient of a temperature-like leg
                    x = -self.almxfl(xalm(name[2:]), self._fl_grad)
                    maps[name] = self._complex_synth(x, zero, 1)
                elif name.startswith("p3:"):         # +2 leg of a polarised gradient
                    x = self.almxfl(xalm(name[3:]), self._fl_p3)
                    maps[name] = self._complex_synth(x, zero, 3)
                elif name.startswith("m1:"):         # -2 leg of a polarised gradient
                    x = -self.almxfl(xalm(name[3:]), self._fl_m1)
                    maps[name] = xp.conj(self._complex_synth(x, zero, 1))
            return maps[name]

        def drop(*names):
            for name in names:
                maps.pop(name, None)

        def pol(x, y):
            """falafel's polarised deflection for gradient leg ``x`` and Y map ``y``."""
            return (-leg("m1:" + x) * leg(y) - leg("p3:" + x) * xp.conj(leg(y))) / 2

        project = self.deflection_map_to_phi_curl_alms
        out = {}
        if "TT" in base:
            out["TT"] = project(-leg("g:t0") * leg("T"))
            drop("g:t0")
        if "TE" in base:
            dmap = -leg("g:t_e0") * leg("T")
            drop("g:t_e0", "T")
            dmap = dmap + pol("e_t0", "Pe")
            out["TE"] = project(dmap)
            del dmap
        drop("T")
        if "TB" in base:
            out["TB"] = project(pol("e_t0", "Pb"))
        drop("p3:e_t0", "m1:e_t0")
        if "EE" in base:
            out["EE"] = project(pol("e0", "Pe"))
        drop("Pe")
        if "EB" in base:
            out["EB"] = project(pol("e0", "Pb"))
        drop("p3:e0", "m1:e0", "Pb")

        results = {}
        for e in ests:
            E = e.upper()
            if E == "MV":
                results[e] = out["TT"] + out["TE"] + out["EE"] + out["EB"] + out["TB"]
            elif E == "MVPOL":
                results[e] = out["EE"] + out["EB"]
            else:
                results[e] = out[E]
        return results


# ------------------------------------------------------------------------------ normalisation
#
# falafel's unnormalised estimator for the pair XY is, in harmonic space (Okamoto & Hu 2003),
#
#     g_LM = (-1)^M sum (l1 l2 L; m1 m2 -M) W_{l1 l2 L} X_{l1 m1} Y_{l2 m2},
#
# with W the gradient-leg term(s) of the lensing response f (Okamoto & Hu, Table I) built from the
# weight spectra and the C^-1 filters.  Its response to phi is
#
#     R_L = 1 / (2L+1) sum_{l1 l2} W_{l1 l2 L} f_{l1 l2 L},       A_L = 1 / R_L,
#
# which is tempura's normalisation.  Every F factor is expanded with the ladder identity
#
#     [L(L+1) + b(b+1) - a(a+1)] (a L b; s 0 -s)
#         = -sqrt(L(L+1)) [ sqrt((b+s)(b-s+1)) (a L b; s -1 1-s) + sqrt((b-s)(b+s+1)) (a L b; s 1 -1-s) ]
#
# (the curl response takes the difference), and each product of two 3j symbols summed over
# (l1, l2) is a Gauss-Legendre integral of three Wigner-d functions,
#
#     sum a_l1 b_l2 (l1 l2 L; m)(l1 l2 L; m') = 1/2 int dmu [sum a d^l1_{m1 m1'}] [sum b d^l2_{m2 m2'}] d^L_{m3 m3'}.
#
# Parity: (-1)^(l1+l2+L) (3j m) = (3j -m), so the even / odd projections are half-sums over m' -> -m'.
# A term whose undifferentiated field is l2 carries (-1)^(l1+l2+L) from bringing its azimuthal 3j
# to the (l1 l2 L; m1 m2 -M) basis, so W-f cross terms between the two orderings take the parity
# sigma of the sector: invisible for the even gradient sums, a sign for the curl (and for Okamoto
# & Hu's EB minus sign).

# For each estimator: the l1 / l2 fields, the W terms and f terms as (spin, outer index, spectrum
# on l1, spectrum on l2, sign), and the gradient parity (+1 even, -1 odd).
#   An F term (s, o, ...) is s_F_{l_o L l_d}, with l_d the other index (the differentiated field).
def _est_terms(est, w, u):
    """W and f terms of one estimator (``w``: weight spectra, ``u``: response spectra)."""
    one = None
    if est == "TT":
        W = [(0, 2, w["TT"], one, 1.0)]
        f = [(0, 2, u["TT"], one, 1.0), (0, 1, one, u["TT"], 1.0)]
        return ("T", "T"), W, f, +1
    if est == "EE":
        W = [(2, 2, w["EE"], one, 1.0)]
        f = [(2, 2, u["EE"], one, 1.0), (2, 1, one, u["EE"], 1.0)]
        return ("E", "E"), W, f, +1
    if est == "TE":
        W = [(2, 2, w["TE"], one, 1.0), (0, 1, one, w["TE"], 1.0)]
        f = [(2, 2, u["TE"], one, 1.0), (0, 1, one, u["TE"], 1.0)]
        return ("T", "E"), W, f, +1
    if est == "EB":
        # No lensing of B into E: the unlensed B is taken to vanish, as in tempura.
        W = [(2, 2, w["EE"], one, 1.0)]
        f = [(2, 2, u["EE"], one, 1.0)]
        return ("E", "B"), W, f, -1
    if est == "TB":
        W = [(2, 2, w["TE"], one, 1.0)]
        f = [(2, 2, u["TE"], one, 1.0)]
        return ("T", "B"), W, f, -1
    raise ValueError(f"No normalisation for estimator {est!r}")


def _ladder(term, curl, ells):
    """The two 3j pieces of one F term: (coef on l1, coef on l2, sign, canonical m triple)."""
    s, outer, c1, c2, sign = term
    diff = 3 - outer
    lb = ells
    a_minus = np.sqrt(np.maximum((lb + s) * (lb - s + 1.0), 0.0))
    a_plus = np.sqrt(np.maximum((lb - s) * (lb + s + 1.0), 0.0))
    ones = np.ones_like(ells)
    c1 = ones if c1 is None else c1
    c2 = ones if c2 is None else c2
    out = []
    for amp, (ma, mL, mb), rho in ((a_minus, (s, -1, 1 - s), 1.0),
                                   (a_plus, (s, 1, -1 - s), -1.0 if curl else 1.0)):
        k1, k2 = (c1, c2 * amp) if diff == 2 else (c1 * amp, c2)
        if outer == 1:      # (l1 L l2; ma mL mb) -> (l1 l2 L; ma mb mL), odd permutation
            m = (-ma, -mb, -mL)
        else:               # (l2 L l1; ma mL mb) -> (l1 l2 L; mb ma mL), cyclic
            m = (mb, ma, mL)
        out.append((k1, k2, sign * rho, m))
    return out


def gauss_legendre(n, newton=3):
    """Gauss-Legendre nodes and weights, Newton-polished in float64 on the device.

    numpy's ``leggauss`` (and scipy's ``roots_legendre``) lose digits at large ``n``: at
    ``n ~ 2000`` their rules integrate ``P_l^2`` with ~1e-11 (1e-10) error, which the low-L lensing
    response amplifies ~1e5-fold.  A few Newton steps on ``P_n`` bring that to ~1e-13.
    """
    x0 = jnp.asarray(np.polynomial.legendre.leggauss(int(n))[0])

    def legendre(x):
        def step(carry, k):
            p0, p1 = carry
            return (p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k), None
        (p0, p1), _ = jax.lax.scan(step, (jnp.ones_like(x), x), jnp.arange(2, n + 1, dtype=x.dtype))
        return p1, n * (x * p1 - p0) / (x * x - 1)

    @jax.jit
    def polish(x):
        for _ in range(newton):
            p, dp = legendre(x)
            x = x - p / dp
        _, dp = legendre(x)
        return x, 2 / ((1 - x * x) * dp * dp)

    return polish(x0)


def _d_recurrence(mn, x):
    """Seed and one-step map of the normalised ``d^l_{m n}(x)`` recurrence for each row of ``mn``.

    Runs on ``x`` itself: going through ``beta = arccos x`` costs ~1e-11 near the poles.
    """
    m = mn[:, 0].astype(x.dtype)[:, None]
    n = mn[:, 1].astype(x.dtype)[:, None]
    l0 = jnp.maximum(jnp.abs(m), jnp.abs(n))
    mu, nu = jnp.abs(m - n), jnp.abs(m + n)
    gl = jax.scipy.special.gammaln
    seed = ((-1.0) ** (((n - m + mu) // 2) % 2) * jnp.sqrt((1 - x) / 2) ** mu
            * jnp.sqrt((1 + x) / 2) ** nu
            * jnp.exp(0.5 * (gl(mu + nu + 1.0) - gl(mu + 1.0) - gl(nu + 1.0))))

    def advance(prev, cur, l):
        """``d^{l+1}`` from ``d^{l-1}``, ``d^l``."""
        lp = l + 1.0
        den = jnp.sqrt(jnp.maximum((lp * lp - m * m) * (lp * lp - n * n), 1e-300))
        a = (2 * l + 1) * lp / den
        mnt = jnp.where(l > 0, m * n / jnp.maximum(l * lp, 1.0), 0.0)
        b = jnp.sqrt(jnp.maximum((l * l - m * m) * (l * l - n * n), 0.0)) / jnp.maximum(l, 1.0) * lp / den
        rec = jnp.where(l > 0, a * (x - mnt) * cur - b * prev, x * cur)
        return jnp.where(lp < l0, 0.0, jnp.where(lp == l0, seed, rec))

    first = jnp.where(l0 == 0, seed, 0.0)
    return first, advance


@jax.jit
def _correlation_functions(coef, group, mn, x):
    """``zeta[p](x) = sum_l coef[p, l] d^l_{m n}(x)`` with ``(m, n) = mn[group[p]]``.

    One pass over ``l`` advances every ``(m, n)`` group and accumulates all correlation
    functions; no ``d`` table is stored.
    """
    first, advance = _d_recurrence(mn, x)

    def step(carry, inputs):
        prev, cur, acc = carry
        l, c = inputs
        acc = acc + c[:, None] * cur[group]
        return (cur, advance(prev, cur, l), acc), None

    acc0 = jnp.zeros((coef.shape[0], x.shape[0]), x.dtype)
    ells = jnp.arange(coef.shape[1], dtype=x.dtype)
    (_, _, acc), _ = jax.lax.scan(step, (jnp.zeros_like(first), first, acc0), (ells, coef.T))
    return acc


@partial(jax.jit, static_argnames=("Lmax",))
def _project(f, group, mn, x, *, Lmax):
    """``out[L, q] = sum_x f[q](x) d^L_{m n}(x)`` for ``L <= Lmax``, ``(m, n) = mn[group[q]]``."""
    first, advance = _d_recurrence(mn, x)

    def step(carry, l):
        prev, cur = carry
        return (cur, advance(prev, cur, l)), jnp.sum(f * cur[group], axis=1)

    _, rows = jax.lax.scan(step, (jnp.zeros_like(first), first),
                           jnp.arange(Lmax + 1, dtype=x.dtype))
    return rows


def normalization(estimators, ucls, ocls, lmin, lmax, Lmax=None, wcls=None):
    """Full-sky normalisation ``A_L`` of falafel's lensing estimators (tempura's convention).

    ``phi_LM = A_L g_LM`` for the gradient and likewise for the curl, where ``g_LM`` is the output
    of :meth:`LensingQE.qe_all` with C^-1 filters ``1 / ocls`` on ``lmin <= l <= lmax``.  Runs on
    the default JAX device in float64: Gauss-Legendre quadrature of Wigner-d correlation functions
    (the separable form of the Okamoto-Hu response), the same device machinery as GMaster's
    MASTER coupling matrices.

    Parameters
    ----------
    estimators : sequence of str
        Any of ``"TT", "TE", "EE", "EB", "TB", "mv", "mvpol"``.
    ucls : dict
        Response spectra ``"TT", "EE", "BB", "TE"`` from ell = 0 (lensed, or the gradient
        spectra for an unbiased response).
    ocls : dict
        Total (signal + noise) spectra ``"TT", "EE", "BB"`` of the diagonal filters.
    lmin, lmax : int
        Multipole range of the CMB fields.
    Lmax : int, optional
        Largest lensing multipole (default ``lmax``).
    wcls : dict, optional
        Weight spectra of the estimator (default ``ucls``), tempura's ``response_cls_weights``.

    Returns
    -------
    dict of ndarray, each shape (2, Lmax + 1)
        Gradient and curl normalisations per estimator (zero where undefined).
    """
    Lmax = int(lmax if Lmax is None else Lmax)
    lmin, lmax = int(lmin), int(lmax)
    wcls = ucls if wcls is None else wcls
    ells = np.arange(lmax + 1, dtype=np.float64)

    def cut(c):
        c = np.zeros(lmax + 1) + np.asarray(c, np.float64)[: lmax + 1]
        c[:lmin] = 0.0
        return c

    u = {k: cut(v) for k, v in ucls.items()}
    w = {k: cut(v) for k, v in wcls.items()}
    filt = {}
    for X, key in (("T", "TT"), ("E", "EE"), ("B", "BB")):
        if key in ocls:
            c = cut(ocls[key])
            filt[X] = np.where((ells > 1) & (c > 0), 1.0 / np.where(c > 0, c, 1.0), 0.0)
    upper = {e.upper() for e in estimators}
    wanted = sorted((upper - {"MV", "MVPOL"})
                    | ({"TT", "TE", "EE", "EB", "TB"} if "MV" in upper else set())
                    | ({"EE", "EB"} if "MVPOL" in upper else set()))

    # Every response is a sum of pieces 1/2 int zeta_a zeta_b d^L (see the notes above `_est_terms`).
    coefs, groups_l, pieces = [], {}, []
    two_l = 2.0 * ells + 1.0

    def zeta_index(vec, mn):
        g = groups_l.setdefault(mn, len(groups_l))
        coefs.append((vec, g))
        return len(coefs) - 1

    for est in wanted:
        (X, Y), Wt, ft, parity = _est_terms(est, w, u)
        for curl in (False, True):
            sigma = -parity if curl else parity
            for wterm in Wt:
                for gterm in ft:
                    cross = sigma if wterm[1] != gterm[1] else 1.0
                    for k1w, k2w, sw, mw in _ladder(wterm, curl, ells):
                        for k1g, k2g, sg, mg in _ladder(gterm, curl, ells):
                            a = two_l * k1w * k1g * filt[X]
                            b = two_l * k2w * k2g * filt[Y]
                            for mm, par in ((mg, 1.0), (tuple(-v for v in mg), sigma)):
                                ia = zeta_index(a, (mw[0], mm[0]))
                                ib = zeta_index(b, (mw[1], mm[1]))
                                pieces.append((est, int(curl), ia, ib, (mw[2], mm[2]),
                                               0.25 * cross * sw * sg * par))
    xq, wq = gauss_legendre((2 * lmax + Lmax) // 2 + 2)
    mn_l = jnp.asarray(sorted(groups_l, key=groups_l.get), dtype=jnp.int32)
    zeta = _correlation_functions(jnp.asarray(np.stack([c for c, _ in coefs])),
                                  jnp.asarray([g for _, g in coefs], dtype=jnp.int32), mn_l, xq)
    groups_L = {}
    for p in pieces:
        groups_L.setdefault(p[4], len(groups_L))
    mn_L = jnp.asarray(sorted(groups_L, key=groups_L.get), dtype=jnp.int32)
    ia = jnp.asarray([p[2] for p in pieces])
    ib = jnp.asarray([p[3] for p in pieces])
    f = wq[None, :] * zeta[ia] * zeta[ib]
    proj = _project(f, jnp.asarray([groups_L[p[4]] for p in pieces], dtype=jnp.int32), mn_L, xq,
                    Lmax=Lmax)                                               # (Lmax + 1, npieces)
    proj = np.asarray(proj)
    L = np.arange(Lmax + 1, dtype=np.float64)
    resp = {e: np.zeros((2, Lmax + 1)) for e in wanted}
    for k, (est, c, _, _, _, factor) in enumerate(pieces):
        resp[est][c] += factor * proj[:, k]
    base = {}
    for est in wanted:
        out = np.zeros((2, Lmax + 1))
        for c, lo in ((0, 1), (1, 2)):
            r = resp[est][c] * 0.25 * L * (L + 1.0) / (4.0 * np.pi)
            ok = (L >= lo) & (r != 0)
            out[c, ok] = 1.0 / r[ok]
        base[est] = out

    res = {}
    for e in estimators:
        E = e.upper()
        if E in ("MV", "MVPOL"):
            parts = ["TT", "TE", "EE", "EB", "TB"] if E == "MV" else ["EE", "EB"]
            comb = np.zeros((2, Lmax + 1))
            for c, lo in ((0, 1), (1, 2)):
                inv = sum(np.where(base[p][c, lo:] != 0,
                                   1.0 / np.where(base[p][c, lo:] != 0, base[p][c, lo:], 1.0), 0.0)
                          for p in parts)
                comb[c, lo:] = np.where(inv != 0, 1.0 / np.where(inv != 0, inv, 1.0), 0.0)
            res[e] = comb
        else:
            res[e] = base[E]
    return res


def _below(fl, lmin):
    fl = np.array(fl, dtype=np.float64)
    fl[:lmin] = 0.0
    return fl
