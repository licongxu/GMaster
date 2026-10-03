"""Lensing quadratic estimator: GMaster (GPU, HEALPix) vs falafel (CPU) on the same inputs.

Opponents, as run by the SO / ACT DR6 pipeline (so-lenspipe):
  falafel-car   falafel.qe.qe_all with pixell's CAR fejer1 full-sky geometry (ducc0 transforms,
                all host threads, float32 maps) -- the production reconstruction path;
  falafel-hpx   falafel.qe.qe_all on HEALPix (healpy transforms), the same nside as GMaster;
  shared-car    GMaster's estimator graph (each distinct transform once) run on the CPU with
                numpy and pixell/ducc0 CAR fejer1 transforms: what falafel would cost with the
                same transform savings, i.e. the GPU's own contribution.  Its transform and
                numpy-glue times are reported separately.
GMaster runs gmaster.lensing.LensingQE.qe_all on HEALPix at the power-of-two nside >= mlmax / 2.

Every code gets identical C^-1-filtered alms (float64), the same response spectra and the same
estimator list.  Times are wall-clock medians of `--reps` warm calls; GMaster's first (compiling)
call is reported separately.  The normalisation is timed separately (GMaster on the GPU, tempura on
the CPU).  Results are appended as JSON lines to `--out`.

Usage:
  python benchmarks/benchmark_lensing_qe.py --mlmax 2000 --codes gmaster,falafel-car,falafel-hpx
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np


def theory(lmax):
    import camb

    pars = camb.set_params(H0=67.5, ombh2=0.022, omch2=0.122, As=2e-9, ns=0.965,
                           lmax=lmax + 500, lens_potential_accuracy=1)
    cl = camb.get_results(pars).get_lensed_scalar_cls(CMB_unit="muK", raw_cl=True)[: lmax + 1]
    ell = np.arange(lmax + 1.0)
    ucls = {"TT": cl[:, 0], "EE": cl[:, 1], "BB": cl[:, 2], "TE": cl[:, 3]}
    beam = np.exp(-ell * (ell + 1) * (1.4 * np.pi / 10800) ** 2 / (8 * np.log(2)))
    nT = (6 * np.pi / 10800) ** 2 / beam ** 2
    tcls = {"TT": ucls["TT"] + nT, "EE": ucls["EE"] + 2 * nT, "BB": ucls["BB"] + 2 * nT,
            "TE": ucls["TE"]}
    return ucls, tcls


def filtered_alms(mlmax, tcls, lmin, lmax, seed=1):
    import healpy as hp

    rng = np.random.default_rng(seed)
    n = hp.Alm.getsize(mlmax)
    ell, m = hp.Alm.getlm(mlmax)
    out = []
    for key in ("TT", "EE", "BB"):
        a = (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2)
        a[m == 0] = a[m == 0].real * np.sqrt(2)
        c = np.zeros(mlmax + 1)
        c[: len(tcls[key])] = tcls[key][: mlmax + 1]
        f = np.where((np.arange(mlmax + 1) >= lmin) & (np.arange(mlmax + 1) <= lmax) & (c > 0),
                     1 / np.sqrt(np.where(c > 0, c, 1)), 0)
        out.append(hp.almxfl(a, f))
    return out


def cpu_shared_qe(mlmax, res_arcmin):
    """GMaster's estimator graph on the CPU: numpy arrays and pixell (ducc0) CAR transforms."""
    import healpy as hp
    from pixell import curvedsky as cs, enmap, utils

    from gmaster.lensing import LensingQE

    shape, wcs = enmap.fullsky_geometry(res=res_arcmin * utils.arcmin, variant="fejer1")

    class CpuQE(LensingQE):
        xp = np

        def __init__(self):
            self.clock = 0.0
            self._set_band_limit(mlmax, hp.Alm.getlm(mlmax)[0])

        def _synth(self, alms, spin):
            t0 = time.perf_counter()
            omap = enmap.empty((1 if spin == 0 else 2,) + shape, wcs, dtype=np.float32)
            alm = np.asarray(alms[0] if spin == 0 else np.stack(alms), dtype=np.complex64)
            out = cs.alm2map(alm, omap if spin else omap[0], spin=spin)
            self.clock += time.perf_counter() - t0
            return [out] if spin == 0 else out

        def deflection_map_to_phi_curl_alms(self, dmap):
            t0 = time.perf_counter()
            maps = enmap.enmap(np.stack([dmap.real, dmap.imag]).astype(np.float32), wcs)
            res = cs.map2alm(maps, spin=1, lmax=mlmax)
            self.clock += time.perf_counter() - t0
            return self.almxfl(res, self._fl_kappa)

    return CpuQE(), f"CAR fejer1 {res_arcmin:.2f}' {shape}"


def timed(fn, reps, block=lambda r: r):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        block(fn())
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), times


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mlmax", type=int, default=2000)
    p.add_argument("--lmin", type=int, default=100)
    p.add_argument("--lmax", type=int, default=None, help="CMB lmax (default 0.75 mlmax)")
    p.add_argument("--ests", default="TT,TE,EE,EB,TB,mv,mvpol")
    p.add_argument("--codes", default="gmaster,falafel-car,falafel-hpx")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--check", action="store_true", help="compare GMaster against falafel-hpx outputs")
    p.add_argument("--out", default="benchmarks/lensing_qe_results.jsonl")
    p.add_argument("--falafel", default=".qwen/tmp/lens_refs/falafel")
    args = p.parse_args()
    sys.path.insert(0, args.falafel)

    mlmax = args.mlmax
    lmax = args.lmax or int(0.75 * mlmax)
    nside = 1 << int(np.ceil(np.log2(mlmax / 2)))
    ests = args.ests.split(",")
    ucls, tcls = theory(mlmax)
    fT, fE, fB = filtered_alms(mlmax, tcls, args.lmin, lmax)
    record = dict(mlmax=mlmax, lmin=args.lmin, lmax=lmax, nside=nside, ests=ests,
                  host=platform.node(), cpu_count=os.cpu_count(),
                  omp=os.environ.get("OMP_NUM_THREADS", "unset"), results={})
    print(f"mlmax {mlmax} (nside {nside}), CMB l {args.lmin}-{lmax}, ests {ests}, "
          f"omp={record['omp']}", flush=True)
    outputs = {}

    for code in args.codes.split(","):
        if code == "gmaster":
            os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
            import jax

            jax.config.update("jax_enable_x64", True)
            from gmaster.lensing import LensingQE

            qe = LensingQE(nside, mlmax)
            run = lambda: qe.qe_all(ucls, fTalm=fT, fEalm=fE, fBalm=fB, estimators=ests)
            t0 = time.perf_counter()
            res = jax.block_until_ready(run())
            first = time.perf_counter() - t0
            med, times = timed(run, args.reps, jax.block_until_ready)
            outputs[code] = {k: np.asarray(v) for k, v in res.items()}
            dev = str(jax.devices()[0])
            record["results"][code] = dict(median_s=med, times=times, first_call_s=first,
                                           device=dev)
            print(f"  gmaster ({dev}): {med:.3f} s warm (first call {first:.1f} s)", flush=True)
        elif code in ("falafel-car", "falafel-hpx"):
            from falafel import qe as fqe

            if code == "falafel-car":
                from pixell import enmap, utils

                res_arcmin = 2.0 * 4000 / mlmax
                shape, wcs = enmap.fullsky_geometry(res=res_arcmin * utils.arcmin,
                                                    variant="fejer1")
                px = fqe.pixelization(shape=shape, wcs=wcs)
                desc = f"CAR fejer1 {res_arcmin:.2f}' {shape}"
            else:
                px = fqe.pixelization(nside=nside)
                desc = f"HEALPix nside {nside}"
            run = lambda: fqe.qe_all(px, ucls, mlmax, fTalm=fT, fEalm=fE, fBalm=fB,
                                     estimators=ests)
            res = run()  # warm-up (plans, caches)
            med, times = timed(run, args.reps)
            outputs[code] = {k: np.asarray(v) for k, v in res.items()}
            record["results"][code] = dict(median_s=med, times=times, geometry=desc)
            print(f"  {code} ({desc}): {med:.3f} s", flush=True)
        elif code == "shared-car":
            qe, desc = cpu_shared_qe(mlmax, 2.0 * 4000 / mlmax)
            run = lambda: qe.qe_all(ucls, fTalm=fT, fEalm=fE, fBalm=fB, estimators=ests)
            res = run()
            qe.clock = 0.0
            med, times = timed(run, args.reps)
            sht = qe.clock / args.reps
            outputs[code] = {k: np.asarray(v) for k, v in res.items()}
            record["results"][code] = dict(median_s=med, times=times, geometry=desc,
                                           transforms_s=sht)
            print(f"  shared-car ({desc}): {med:.3f} s (transforms {sht:.3f} s, numpy glue "
                  f"{med - sht:.3f} s)", flush=True)
        elif code == "norm":
            os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
            import jax

            jax.config.update("jax_enable_x64", True)
            from gmaster.lensing import normalization

            norm_ests = [e.upper() for e in ests]
            run = lambda: normalization(norm_ests, ucls, tcls, args.lmin, lmax, Lmax=lmax)
            t0 = time.perf_counter()
            run()
            first = time.perf_counter() - t0
            med, times = timed(run, args.reps)
            record["results"]["norm-gmaster"] = dict(median_s=med, times=times,
                                                     first_call_s=first,
                                                     device=str(jax.devices()[0]))
            print(f"  normalisation gmaster: {med:.3f} s warm (first {first:.1f} s)", flush=True)
            try:
                import pytempura

                med, times = timed(lambda: pytempura.get_norms(norm_ests, ucls, ucls, tcls,
                                                               args.lmin, lmax, k_ellmax=lmax),
                                   args.reps)
                record["results"]["norm-tempura"] = dict(median_s=med, times=times)
                print(f"  normalisation tempura: {med:.3f} s", flush=True)
            except ImportError:
                pass
        else:
            raise ValueError(code)

    if args.check and "gmaster" in outputs:
        for ref in ("falafel-hpx", "falafel-car", "shared-car"):
            if ref not in outputs:
                continue
            errs = {}
            for e in ests:
                g, r = outputs["gmaster"][e], outputs[ref][e]
                errs[e] = float(max(np.abs(g[c] - r[c]).max() / np.abs(r[c]).max() for c in (0, 1)))
            record[f"max_rel_diff_vs_{ref}"] = errs
            print(f"  max |gmaster - {ref}| / max|{ref}|: "
                  + ", ".join(f"{k} {v:.1e}" for k, v in errs.items()), flush=True)

    if args.check and "falafel-hpx" in outputs and "falafel-car" in outputs:
        errs = {e: float(max(np.abs(outputs["falafel-hpx"][e][c] - outputs["falafel-car"][e][c]).max()
                             / np.abs(outputs["falafel-car"][e][c]).max() for c in (0, 1)))
                for e in ests}
        record["max_rel_diff_falafel_hpx_vs_car"] = errs
        print("  pixelisation alone, max |falafel-hpx - falafel-car| / max|falafel-car|: "
              + ", ".join(f"{k} {v:.1e}" for k, v in errs.items()), flush=True)

    with open(args.out, "a") as fh:
        fh.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
