"""NaMaster transform times: `alm2map` and one-pass `map2alm` (n_iter 0), five warm runs.

Same operator and inputs as `run_worker_sht` in `benchmarks/run_paper_v2_benchmarks.py` (which
produced the Nside <= 4096 points), with ducc0 pinned to `cores` threads as in
`bench_act_warm.py`.  One untimed call of each first.

  python bench_nm_sht.py <nside> <spin> <cores>
"""
import json
import os
import sys
import time

import numpy as np


def main():
    nside, spin, cores = (int(x) for x in sys.argv[1:4])
    reps = int(os.environ.get("REPS", "5"))
    import pymaster as nmt
    import pymaster.utils as u

    orig = u._ducc_kwargs

    def wrapped(*args, **kwargs):
        out = orig(*args, **kwargs)
        out["nthreads"] = cores
        return out

    u._ducc_kwargs = wrapped

    lmax, npix = 3 * nside - 1, 12 * nside**2
    nmaps = 1 if spin == 0 else 2
    rng = np.random.default_rng(42 + nside + spin)
    maps = rng.normal(size=(nmaps, npix))
    minfo, ainfo = nmt.NmtMapInfo(None, (npix,)), nmt.NmtAlmInfo(lmax)
    alms = rng.normal(size=(nmaps, ainfo.nelem)) + 1j * rng.normal(size=(nmaps, ainfo.nelem))
    alms[:, : lmax + 1] = alms[:, : lmax + 1].real

    nmt.map2alm(maps, spin, minfo, ainfo, n_iter=0)
    nmt.alm2map(alms, spin, minfo, ainfo)
    ana, syn = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        nmt.map2alm(maps, spin, minfo, ainfo, n_iter=0)
        ana.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter()
        nmt.alm2map(alms, spin, minfo, ainfo)
        syn.append((time.perf_counter() - t0) * 1e3)
    print("RESULT " + json.dumps({
        "engine": "namaster", "cores": cores, "nside": nside, "spin": spin,
        "ana_times_ms": ana, "syn_times_ms": syn,
        "ana_ms": float(np.median(ana)), "syn_ms": float(np.median(syn)),
    }), flush=True)


if __name__ == "__main__":
    main()
