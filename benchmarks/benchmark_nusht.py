"""GMaster's GPU general SHT (``gmaster.nusht``) against ducc0 on the same box.

One RTX PRO 6000 Blackwell versus ``ducc0.sht.experimental.{synthesis,adjoint_synthesis}_general``
at 96 and 32 threads, same alm, same point set, same process.  Warm, min-of-``--reps``.

What is excluded from every timing, on both sides:
  * plan construction -- the cufinufft plan and its ``setpts`` (the point sort) for GMaster, and
    ducc0's first call for its internal tables.  Both papers time this way.
  * the march's window tables (``nusht._tables``), which are resident device arrays built once
    per geometry, exactly as ``_march_v2.tables_for`` does for the HEALPix path.
  * host-device copies.  The alm / map inputs are already jax device arrays, the outputs are left
    on the device, and the timing ends at a device synchronise.  ducc0's inputs and outputs are
    host arrays and it has no copy to pay, so this favours neither side beyond the PCIe traffic a
    real GPU pipeline would pay once per field.

Accuracy is matched rather than assumed equal: ``--match`` runs both codes against a 1e-12
reference first and picks the ducc0 ``epsilon`` closest to GMaster's measured ``eps_eff``.
"""

import argparse
import time

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import ducc0  # noqa: E402

from gmaster import nusht  # noqa: E402

DUCC_EPS = np.array([1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 1e-8, 1e-10, 1e-12])


def make_alm(lmax, spin, rng):
    L = lmax + 1
    ncomp = 1 if spin == 0 else 2
    nelem = L * (L + 1) // 2
    ell = np.concatenate([np.arange(m, L) for m in range(L)])
    emm = np.concatenate([np.full(L - m, m) for m in range(L)])
    alm = rng.normal(size=(ncomp, nelem)) + 1j * rng.normal(size=(ncomp, nelem))
    alm *= (ell >= np.maximum(emm, abs(spin)))[None, :] / (1.0 + ell)[None, :]
    alm[:, emm == 0] = alm[:, emm == 0].real
    return np.ascontiguousarray(alm)


def make_points(kind, lmax, npoint, rng):
    """``uniform``: i.i.d. on the sphere.  ``deflected``: a Fejer-like grid displaced by a smooth
    2 arcmin deflection field, i.e. the lensing-remapping geometry, where the points are nearly
    equispaced instead of Poisson-clustered (the NUFFT spreading cost depends on that)."""
    if kind == "uniform":
        loc = np.empty((npoint, 2))
        loc[:, 0] = np.arccos(rng.uniform(-1, 1, npoint))
        loc[:, 1] = rng.uniform(0, 2 * np.pi, npoint)
        return loc
    nt = max(int(np.sqrt(npoint / 2)), 2)
    npx = max(npoint // nt, 2)
    th = (2 * np.arange(nt) + 1) * np.pi / (2 * nt)
    ph = 2 * np.pi * np.arange(npx) / npx
    T, P = np.meshgrid(th, ph, indexing="ij")
    amp = 2.0 / 60.0 * np.pi / 180.0                      # 2 arcmin, a CMB lensing deflection
    T = T + amp * np.sin(3 * T) * np.cos(5 * P)
    P = P + amp * np.cos(7 * T) * np.sin(2 * P) / np.maximum(np.sin(T), 1e-3)
    loc = np.empty((T.size, 2))
    loc[:, 0] = np.clip(T.ravel(), 1e-9, np.pi - 1e-9)
    loc[:, 1] = np.mod(P.ravel(), 2 * np.pi)
    return loc


def gpu_sync():
    nusht._cupy().cuda.get_current_stream().synchronize()


def gpu_time(fn, reps):
    out = fn()
    jax.block_until_ready(out)
    gpu_sync()
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        gpu_sync()
        best = min(best, time.perf_counter() - t0)
    return best, out


def cpu_time(fn, reps):
    out = fn()
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def relerr(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b)))


def _match_eps(alm, vals_h, ref2, ref1, eps2, eps1, lmax, spin, mstart, loc, nthreads):
    """The ducc0 ``epsilon`` whose measured ``eps_eff`` lands nearest GMaster's."""
    out = []
    for target, direction in ((eps2, 2), (eps1, 1)):
        best, bestd = DUCC_EPS[0], np.inf
        for de in DUCC_EPS[DUCC_EPS >= 1e-7]:
            if direction == 2:
                o = ducc0.sht.experimental.synthesis_general(
                    alm=alm, spin=spin, lmax=lmax, mmax=lmax, mstart=mstart, loc=loc,
                    epsilon=float(de), nthreads=nthreads)
                e = relerr(o, ref2)
            else:
                o = ducc0.sht.experimental.adjoint_synthesis_general(
                    map=vals_h, spin=spin, lmax=lmax, mmax=lmax, mstart=mstart, loc=loc,
                    epsilon=float(de), nthreads=nthreads)
                e = relerr(o, ref1)
            d = abs(np.log(max(e, 1e-16)) - np.log(target))
            if d < bestd:
                best, bestd = float(de), d
        out.append(best)
    return out[0], out[1]


def run_cell(lmax, spin, kind, args, rng):
    L = lmax + 1
    mstart = nusht.default_mstart(lmax)
    alm = make_alm(lmax, spin, rng)
    npoint = args.npoint or lmax ** 2
    loc = make_points(kind, lmax, npoint, rng)
    npoint = loc.shape[0]
    ntheta, nphi = nusht.grid_for(lmax)

    alm_d = jnp.asarray(alm)
    nusht._tables(ntheta, L, spin)                       # resident window tables, excluded below
    plan2 = nusht._NufftPlan(2, ntheta, nphi, loc, args.epsilon, args.upsampfac, np.complex128)
    plan1 = nusht._NufftPlan(1, ntheta, nphi, loc, args.epsilon, args.upsampfac, np.complex128)

    syn = lambda: nusht.synthesis_general(                # noqa: E731
        alm_d, spin=spin, lmax=lmax, loc=loc, epsilon=args.epsilon,
        upsampfac=args.upsampfac, plan=plan2)
    t_syn, vals = gpu_time(syn, args.reps)
    vals_h = np.ascontiguousarray(np.asarray(vals))
    vals_d = jnp.asarray(vals_h)
    adj = lambda: nusht.adjoint_synthesis_general(        # noqa: E731
        vals_d, spin=spin, lmax=lmax, loc=loc, epsilon=args.epsilon,
        upsampfac=args.upsampfac, plan=plan1)
    t_adj, aout = gpu_time(adj, args.reps)

    # stage split
    idx, ok = nusht._alm_index(lmax, lmax, tuple(int(v) for v in mstart), 1, abs(spin),
                               alm.shape[1])[:2]
    garr, npad = nusht._geometry(ntheta, L, spin)
    tabs = nusht._tables(ntheta, L, spin)
    t_ftm, C = gpu_time(lambda: nusht._ftm_to_torus(
        alm_d, idx, ok, garr, tabs, npad=npad, L=L, spin=spin, ntheta=ntheta, nphi=nphi,
        cdtype=jnp.complex128, wdtype=jnp.complex64), args.reps)
    t_nu2, _ = gpu_time(lambda: plan2(C), args.reps)
    vcx = vals_d[0] if spin == 0 else vals_d[0] + 1j * vals_d[1]
    t_nu1, C1 = gpu_time(lambda: plan1(vcx), args.reps)
    back, bok = nusht._alm_index(lmax, lmax, tuple(int(v) for v in mstart), 1, abs(spin),
                                 alm.shape[1])[2:]
    t_rest, _ = gpu_time(lambda: nusht._torus_to_alm(
        C1, back, bok, garr, tabs, npad=npad, L=L, spin=spin, ntheta=ntheta, nphi=nphi,
        wdtype=jnp.complex64), args.reps)
    t_march, _ = gpu_time(lambda: _march_only(alm_d, idx, ok, garr, tabs, npad, L, spin, ntheta),
                          args.reps)

    # accuracy: eps_eff against ducc0 at 1e-12, then the matched ducc0 epsilon
    ref2 = ducc0.sht.experimental.synthesis_general(
        alm=alm, spin=spin, lmax=lmax, mmax=lmax, mstart=mstart, loc=loc, epsilon=1e-12,
        nthreads=args.nthreads[0])
    eps2 = relerr(vals_h, ref2)
    ref1 = ducc0.sht.experimental.adjoint_synthesis_general(
        map=vals_h, spin=spin, lmax=lmax, mmax=lmax, mstart=mstart, loc=loc, epsilon=1e-12,
        nthreads=args.nthreads[0])
    eps1 = relerr(aout, ref1)
    # Match on *measured* accuracy, not on the requested tolerance: ducc0's epsilon is a bound
    # and its realised eps_eff runs one to two decades below it, so matching epsilon-to-eps_eff
    # would hand ducc0 a harder problem than GMaster solves.
    de2, de1 = _match_eps(alm, vals_h, ref2, ref1, eps2, eps1, lmax, spin, mstart, loc,
                          args.nthreads[0])

    rows = []
    for nt in args.nthreads:
        t, o = cpu_time(lambda: ducc0.sht.experimental.synthesis_general(  # noqa: B023
            alm=alm, spin=spin, lmax=lmax, mmax=lmax, mstart=mstart, loc=loc, epsilon=de2,
            nthreads=nt), args.reps)
        rows.append(("ducc0-t%d" % nt, 2, t, relerr(o, ref2)))
        t, o = cpu_time(lambda: ducc0.sht.experimental.adjoint_synthesis_general(  # noqa: B023
            map=vals_h, spin=spin, lmax=lmax, mmax=lmax, mstart=mstart, loc=loc, epsilon=de1,
            nthreads=nt), args.reps)
        rows.append(("ducc0-t%d" % nt, 1, t, relerr(o, ref1)))

    print(f"\n== lmax={lmax} spin={spin} points={kind} N={npoint} ntheta={ntheta} nphi={nphi} "
          f"upsampfac={args.upsampfac} gmaster-eps={args.epsilon:g} reps={args.reps}")
    print(f"   stage split (type 2): march {t_march*1e3:7.1f} | march+double+FFT "
          f"{t_ftm*1e3:7.1f} | NUFFT {t_nu2*1e3:7.1f} | total {t_syn*1e3:7.1f} ms")
    print(f"   stage split (type 1): NUFFT {t_nu1*1e3:7.1f} | ifft+undouble+march+pack "
          f"{t_rest*1e3:7.1f} | total {t_adj*1e3:7.1f} ms")
    print(f"   {'code':10s} {'type':>4s} {'time [ms]':>11s} {'eps_eff':>10s} "
          f"{'ducc eps':>9s} {'speedup':>8s}")
    print(f"   {'gmaster':10s} {2:4d} {t_syn*1e3:11.1f} {eps2:10.2e} {'-':>9s} {'-':>8s}")
    print(f"   {'gmaster':10s} {1:4d} {t_adj*1e3:11.1f} {eps1:10.2e} {'-':>9s} {'-':>8s}")
    for name, typ, t, e in rows:
        ours = t_syn if typ == 2 else t_adj
        de = de2 if typ == 2 else de1
        print(f"   {name:10s} {typ:4d} {t*1e3:11.1f} {e:10.2e} {de:9.0e} {t/ours:8.2f}x")


def _march_only(alm_d, idx, ok, garr, tabs, npad, L, spin, ntheta):
    if spin == 0:
        a = nusht._unpack(alm_d[0], idx, ok, L=L)
        return nusht._march_syn_s0(a, garr, tabs, L=L, npad=npad, ntheta=ntheta)
    a1 = nusht._unpack(alm_d[0], idx, ok, L=L)
    a2 = nusht._unpack(alm_d[1], idx, ok, L=L)
    rs = (1.0 - 2.0 * (jnp.arange(L) % 2))[None, :]
    return nusht._march_syn_spin(-(a1 + 1j * a2), -rs * jnp.conj(a1 - 1j * a2), garr, tabs,
                                 L=L, npad=npad, ntheta=ntheta, spin=spin)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lmax", type=int, nargs="+", default=[1023, 2047, 4095])
    p.add_argument("--spin", type=int, nargs="+", default=[0, 2])
    p.add_argument("--points", nargs="+", default=["uniform", "deflected"])
    p.add_argument("--upsampfac", type=float, default=1.25)
    p.add_argument("--epsilon", type=float, default=1e-6,
                   help="cufinufft tolerance; 1e-6 is the knee -- see synthesis_general")
    p.add_argument("--npoint", type=int, default=0, help="default lmax**2")
    p.add_argument("--nthreads", type=int, nargs="+", default=[96, 32])
    p.add_argument("--reps", type=int, default=5)
    args = p.parse_args()
    rng = np.random.default_rng(20260913)
    for lmax in args.lmax:
        for spin in args.spin:
            for kind in args.points:
                run_cell(lmax, spin, kind, args, rng)
                nusht.drop_tables()


if __name__ == "__main__":
    main()
