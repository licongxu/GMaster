"""Recurrence stress test on HEALPix: white (flat-spectrum) alm, per-ell errors.

The ACT pseudo-Cl test is dominated by low ell (red spectrum), which hides
high-ell recurrence error.  Here every (l, m) has unit variance.  Against ducc0
float64 on the same HEALPix grid:

  syn   : alm2map of the white alm (rms and max error relative to the map rms)
  ana   : one analysis (n_iter = 0) of the float64-synthesised map, compared with
          ducc0's own n_iter = 0 analysis, so the HEALPix quadrature error cancels
          and only the transform (recurrence + float32 data) error remains.
          Per ell: sqrt(sum_m |da|^2 / sum_m |a|^2).

  python bench_white.py <gmaster|shtns32|shtns64> <nside>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_white.jsonl")


def reference(nside):
    import ducc0
    lmax = 3 * nside - 1
    nalm = (lmax + 1) * (lmax + 2) // 2
    rng = np.random.default_rng(7)
    alm = (rng.normal(size=nalm) + 1j * rng.normal(size=nalm)) / np.sqrt(2)
    alm[: lmax + 1] = alm[: lmax + 1].real * np.sqrt(2)
    g = ducc0.healpix.Healpix_Base(nside, "RING").sht_info()
    m = ducc0.sht.experimental.synthesis(alm=alm[None], spin=0, lmax=lmax, nthreads=8, **g)[0]
    a0 = ducc0.sht.experimental.adjoint_synthesis(map=m[None], spin=0, lmax=lmax, nthreads=8,
                                                  **g)[0] * (4 * np.pi / m.size)
    return alm, m, a0


def per_ell(d, ref, lmax):
    import healpy as hp
    num = hp.alm2cl(d, lmax=lmax)
    den = hp.alm2cl(ref, lmax=lmax)
    return np.sqrt(num / den)


def main(mode, nside):
    lmax = 3 * nside - 1
    alm, m, a0 = reference(nside)
    if mode == "gmaster":
        os.environ.setdefault("JAX_ENABLE_X64", "1")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ["GMASTER_DC"] = "0"
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        import gmaster as gm
        from gmaster import utils
        gm.set_latitudinal_method("march")
        mi, ai = utils.NmtMapInfo(None, [m.size]), utils.NmtAlmInfo(lmax)
        syn = np.asarray(utils.alm2map(jnp.asarray(alm[None]), 0, mi, ai)[0], np.float64)
        ana = np.asarray(utils.map2alm(jnp.asarray(m[None]), 0, mi, ai, n_iter=0)[0], np.complex128)
    else:
        os.environ["SHTNS_GPU_REC_PREC"] = {"shtns32": "1", "shtns64": "2"}[mode]
        sys.path.insert(0, HERE)
        import cupy as cp
        from shtns_healpix import HealpixSHT
        T = HealpixSHT(nside)
        syn = cp.asnumpy(T.alm2map(cp.asarray(alm.astype(np.complex64)))).astype(np.float64)
        ana = cp.asnumpy(T.map2alm(cp.asarray(m.astype(np.float32)), n_iter=0)).astype(np.complex128)
    rms = np.sqrt(np.mean(m ** 2))
    e_l = per_ell(ana - a0, a0, lmax)
    rec = {"mode": mode, "nside": nside,
           "syn_rms_rel": float(np.sqrt(np.mean((syn - m) ** 2)) / rms),
           "syn_max_rel": float(np.max(np.abs(syn - m)) / rms),
           "ana_rms_rel": float(np.linalg.norm(ana - a0) / np.linalg.norm(a0)),
           "ana_ell_median": float(np.median(e_l[2:])), "ana_ell_max": float(e_l[2:].max()),
           "ana_ell_argmax": int(np.argmax(e_l[2:]) + 2),
           "ana_ell_profile": [float(x) for x in e_l[:: max(1, lmax // 24)]]}
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("RESULT", json.dumps({k: v for k, v in rec.items() if k != "ana_ell_profile"}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
