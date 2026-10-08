"""Pseudo-Cl and transform accuracy/time: SHTns-on-HEALPix vs GMaster vs NaMaster.

Every mode analyses the same masked ACT DR6 temperature map (PA4 f150 night,
cached HEALPix maps) on the HEALPix grid with ``n_iter = 3``, at ``lmax = 3 nside - 1``.

  namaster : pymaster (float64, ducc0 on the CPU).  The reference.
  gmaster  : gmaster.utils.map2alm, latitudinal method "march" (float32 (v, D) march).
  shtns32  : SHTns GPU Legendre kernels with SHTns's own float32 recurrence
             (SHTNS_GPU_REC_PREC=1), HEALPix ring FFTs from shtns_healpix.py.
  shtns64  : the same with SHTns's float64 recurrence (SHTNS_GPU_REC_PREC=2); data,
             FFTs and sums stay float32.  Control for the recurrence.

  python bench_pcl.py <mode> <nside>      # appends one JSON line to results.jsonl
  python bench_pcl.py compare             # accuracy table vs namaster

One mode per process (SHTns reads SHTNS_GPU_REC_PREC at plan time; JAX and cupy
should not share a card).  Run on an otherwise idle GPU.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = "/scratch/scratch-lxu/agent_dev/auto_research_agent/GMaster/.qwen/tmp/accuracy/act_cache"
OUTDIR = os.environ.get("SHTNS_HP_OUT",
                        "/scratch/scratch-lxu/agent_dev/auto_research_agent/GMaster/.qwen/tmp/shtns_hp/out")
RESULTS = os.path.join(HERE, "results.jsonl")
REPS = 5


def load(nside):
    import healpy as hp
    src = min(n for n in (256, 512, 1024, 2048, 4096, 8192) if n >= nside)
    t = np.load(f"{CACHE}/act_T_nside{src}.npy").astype(np.float64)
    w = np.load(f"{CACHE}/act_w_nside{src}.npy").astype(np.float64)
    if src != nside:
        t, w = hp.ud_grade(t, nside), hp.ud_grade(w, nside)
    mask = (w > 0).astype(np.float64)
    return mask, mask * t


def alm2cl(alm, lmax):
    import healpy as hp
    return hp.alm2cl(np.asarray(alm, np.complex128), lmax=lmax)


def timed(fn, sync, reps=REPS):
    fn(); sync()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(); sync()
        out.append(time.perf_counter() - t0)
    return out


def ducc_synthesis(alm, nside, lmax):
    import ducc0
    g = ducc0.healpix.Healpix_Base(nside, "RING").sht_info()
    return ducc0.sht.experimental.synthesis(alm=np.asarray(alm, np.complex128)[None], spin=0,
                                            lmax=lmax, nthreads=8, **g)[0]


def rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def run_namaster(nside):
    import pymaster as nmt
    mask, mt = load(nside)
    lmax = 3 * nside - 1
    t0 = time.perf_counter()
    f = nmt.NmtField(mask, [mt], n_iter=3, lmax=lmax, lmax_mask=lmax, masked_on_input=True)
    t_field = time.perf_counter() - t0
    alm = f.get_alms()[0]
    cl = nmt.compute_coupled_cell(f, f)[0]
    np.save(f"{OUTDIR}/namaster_alm_{nside}.npy", alm)
    np.save(f"{OUTDIR}/namaster_cl_{nside}.npy", cl)
    np.save(f"{OUTDIR}/fsky_{nside}.npy", np.array(mask.mean()))
    return {"field_seconds_incl_mask": t_field, "threads": os.environ.get("OMP_NUM_THREADS")}


def run_gmaster(nside):
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ["GMASTER_DC"] = "0"
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import gmaster as gm
    from gmaster import utils
    gm.set_latitudinal_method("march")
    _, mt = load(nside)
    lmax = 3 * nside - 1
    minfo, ainfo = utils.NmtMapInfo(None, [mt.size]), utils.NmtAlmInfo(lmax)
    x = jnp.asarray(mt[None])
    sync = lambda: None
    res = {}
    out = {}
    def ana(n):
        out["a"] = jax.block_until_ready(utils.map2alm(x, 0, minfo, ainfo, n_iter=n))
    res["map2alm_iter3_s"] = timed(lambda: ana(3), sync)
    alm = np.asarray(out["a"][0])
    res["map2alm_iter0_s"] = timed(lambda: ana(0), sync)
    alm0 = np.asarray(out["a"][0])
    ref = np.load(f"{OUTDIR}/namaster_alm_{nside}.npy")
    r = jnp.asarray(ref[None])
    def syn():
        out["m"] = jax.block_until_ready(utils.alm2map(r, 0, minfo, ainfo))
    res["alm2map_s"] = timed(syn, sync)
    res["syn_err_vs_ducc"] = rel(np.asarray(out["m"][0], np.float64), ducc_synthesis(ref, nside, lmax))
    return finish("gmaster", nside, alm, alm0, res)


def run_shtns(nside, rec):
    os.environ["SHTNS_GPU_REC_PREC"] = {"shtns32": "1", "shtns64": "2"}[rec]
    sys.path.insert(0, HERE)
    import cupy as cp
    from shtns_healpix import HealpixSHT
    _, mt = load(nside)
    lmax = 3 * nside - 1
    t0 = time.perf_counter()
    T = HealpixSHT(nside)
    cp.cuda.Device().synchronize()
    res = {"plan_s": time.perf_counter() - t0, "rec_bytes": int(T.sh.sizeof_real_g) if hasattr(T.sh, "sizeof_real_g") else None}
    x = cp.asarray(mt.astype(np.float32))
    sync = cp.cuda.Device().synchronize
    out = {}
    res["map2alm_iter3_s"] = timed(lambda: out.__setitem__("a", T.map2alm(x, n_iter=3)), sync)
    alm = cp.asnumpy(out["a"])
    res["map2alm_iter0_s"] = timed(lambda: out.__setitem__("a", T.map2alm(x, n_iter=0)), sync)
    alm0 = cp.asnumpy(out["a"])
    ref = np.load(f"{OUTDIR}/namaster_alm_{nside}.npy")
    r = cp.asarray(ref.astype(np.complex64))
    res["alm2map_s"] = timed(lambda: out.__setitem__("m", T.alm2map(r)), sync)
    res["syn_err_vs_ducc"] = rel(cp.asnumpy(out["m"]).astype(np.float64), ducc_synthesis(ref, nside, lmax))
    # split of one analysis: ring FFT stage vs SHTns Legendre kernel
    F = T.map2fourier(x)
    res["ana_fft_stage_s"] = timed(lambda: T.map2fourier(x), sync)
    a = cp.empty(T.nlm, np.complex64)
    res["ana_legendre_s"] = timed(lambda: T.lib.cu_fourier_to_SH_float(T.cfg, F.data.ptr, a.data.ptr, lmax), sync)
    return finish(rec, nside, alm, alm0, res)


def finish(mode, nside, alm, alm0, res):
    lmax = 3 * nside - 1
    np.save(f"{OUTDIR}/{mode}_cl_{nside}.npy", alm2cl(alm, lmax))
    ref = np.load(f"{OUTDIR}/namaster_alm_{nside}.npy")
    res["alm_maxrel_vs_namaster"] = rel(alm, ref)
    res["alm_rms_rel_vs_namaster"] = float(np.linalg.norm(alm - ref) / np.linalg.norm(ref))
    res["alm_iter0_rms_rel_vs_namaster"] = float(np.linalg.norm(alm0 - ref) / np.linalg.norm(ref))
    return res


def compare(nside):
    lmax = 3 * nside - 1
    ell = np.arange(lmax + 1)
    ref = np.load(f"{OUTDIR}/namaster_cl_{nside}.npy")
    fsky = float(np.load(f"{OUTDIR}/fsky_{nside}.npy"))
    sig = ref * np.sqrt(2.0 / ((2 * ell + 1) * fsky))
    good = ell >= 2
    nb = 30
    edges = np.arange(2, lmax + 1, nb)
    out = {}
    for mode in ("gmaster", "shtns32", "shtns64"):
        p = f"{OUTDIR}/{mode}_cl_{nside}.npy"
        if not os.path.exists(p):
            continue
        cl = np.load(p)
        d = np.abs(cl - ref)[good] / ref[good]
        dcv = np.abs(cl - ref)[good] / sig[good]
        bp = lambda c: np.array([c[a:a + nb].mean() for a in edges[:-1]])
        bsig = np.array([np.sqrt(np.sum(sig[a:a + nb] ** 2)) / nb for a in edges[:-1]])
        dbp = np.abs(bp(cl) - bp(ref)) / bsig
        out[mode] = {"cl_rel_median": float(np.median(d)), "cl_rel_max": float(d.max()),
                     "cl_rel_p99": float(np.quantile(d, 0.99)),
                     "max_dcl_over_cv_per_ell": float(dcv.max()),
                     "max_dbandpower_over_cv_dl30": float(dbp.max())}
    return out


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    mode = sys.argv[1]
    if mode == "compare":
        rows = [json.loads(s) for s in open(RESULTS)] if os.path.exists(RESULTS) else []
        for nside in sorted({r["nside"] for r in rows}):
            print(nside, json.dumps(compare(nside), indent=1))
        sys.exit()
    nside = int(sys.argv[2])
    fn = {"namaster": run_namaster, "gmaster": run_gmaster}.get(mode)
    res = fn(nside) if fn else run_shtns(nside, mode)
    rec = {"mode": mode, "nside": nside, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), **res}
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("RESULT", json.dumps(rec), flush=True)
