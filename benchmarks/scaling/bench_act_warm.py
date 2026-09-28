"""End-to-end MASTER on the ACT DR6 map and footprint: one cold call, then warm timed calls.

    python bench_act_warm.py NSIDES SPIN [ENGINE [CORES]]
        NSIDES   comma-separated, e.g. 64,128,...,8192
        SPIN     0 (Stokes I) or 2 (Q, U)
        ENGINE   gmaster | namaster (default namaster)
        CORES    threads given to NaMaster's ducc0 (default 32)

Reads `act_{T,Q,U,w}_nside{N}.npy` from $ACT_CACHE (made by `examples/act_dr6/prepare.py`);
lower resolutions are `ud_grade`d from the nearest cached one.  The mask is the nonzero
footprint.  Timed: NmtField (n_iter 3) + NmtWorkspace + coupled cell + decouple, linear bins of 30.
Prints one `RESULT {json}` line per Nside with every warm time and the peak memory, and saves the
spectra to $ACT_OUT (default `act_cl`).  $ACT_WARM_REPS sets the number of warm runs (default 5).
"""
import json
import os
import resource
import sys
import time
import traceback

import healpy as hp
import numpy as np

CACHED = (256, 512, 1024, 2048, 4096, 8192)
ROOT = os.environ.get("ACT_CACHE", "act_cache")   # act_{T,Q,U,w}_nside{N}.npy, see examples/act_dr6


def _backend(engine, cores):
    if engine == "namaster":
        import pymaster as nmt
        import pymaster.utils as u
        orig = u._ducc_kwargs
        def wrapped(*args, **kwargs):
            out = orig(*args, **kwargs)
            out["nthreads"] = int(cores)
            return out
        u._ducc_kwargs = wrapped
        import gmaster._nmt_bin64 as bin64
        bin64.patch_pymaster()
        return nmt, None
    import jax
    import gmaster as nmt
    nmt.set_latitudinal_method("auto")
    return nmt, jax.block_until_ready


def load(nside):
    src = min(n for n in CACHED if n >= nside)
    t = np.load(f"{ROOT}/act_T_nside{src}.npy")
    w = np.load(f"{ROOT}/act_w_nside{src}.npy")
    if src != nside:
        t = hp.ud_grade(t.astype(np.float64), nside).astype(np.float32)
        w = hp.ud_grade(w.astype(np.float64), nside).astype(np.float32)
    return t, w


def load_pol(nside):
    src = min(n for n in CACHED if n >= nside)
    q = np.load(f"{ROOT}/act_Q_nside{src}.npy")
    u = np.load(f"{ROOT}/act_U_nside{src}.npy")
    w = np.load(f"{ROOT}/act_w_nside{src}.npy")
    if src != nside:
        q = hp.ud_grade(q.astype(np.float64), nside).astype(np.float32)
        u = hp.ud_grade(u.astype(np.float64), nside).astype(np.float32)
        w = hp.ud_grade(w.astype(np.float64), nside).astype(np.float32)
    return q, u, w


def _memory(engine):
    """Peak memory of this process so far: host RSS, and for GMaster the device pool's peak.

    Run one nside per process for the peaks to be that cell's (the untimed first call included).
    """
    out = {"host_peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}
    if engine != "namaster":
        import jax
        stats = jax.devices()[0].memory_stats() or {}
        out["gpu_peak_gib"] = stats.get("peak_bytes_in_use", 0) / 2**30
    return out


def one(nside, spin, engine, cores):
    nmt, block = _backend(engine, cores)
    if spin == 0:
        maps, w = [load(nside)[0]], load(nside)[1]
    else:
        q, u, w = load_pol(nside)
        maps = [q, u]
    mask = (w > 0).astype(np.float64)
    bins = nmt.NmtBin.from_lmax_linear(3 * nside - 1, 30)

    def once():
        print("STAGE field", flush=True)
        field = nmt.NmtField(mask, maps, n_iter=3, spin=spin) if spin else nmt.NmtField(mask, maps, n_iter=3)
        if block is not None:
            block(field.get_alms())
        print("STAGE coupling", flush=True)
        work = nmt.NmtWorkspace()
        work.compute_coupling_matrix(field, field, bins)
        cl = work.decouple_cell(nmt.compute_coupled_cell(field, field))
        if block is not None:
            block(cl)
        return np.asarray(cl[0] if spin == 0 else cl)

    reps = int(os.environ.get("ACT_WARM_REPS", "5"))
    cl = once()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        cl = once()
        samples.append(time.perf_counter() - t0)
    tag = "nm32" if engine == "namaster" and cores == 32 else f"{engine}{cores}"
    # A one-shot recheck must not replace the five-run spectra.
    stem = tag if reps == 5 else f"re1_{tag}"
    os.makedirs(os.environ.get("ACT_OUT", "act_cl"), exist_ok=True)
    out = os.path.join(os.environ.get("ACT_OUT", "act_cl"), f"act_cl_{stem}_n{nside}_s{spin}.npy")
    np.save(out, cl)
    print("RESULT " + json.dumps({
        "engine": engine, "cores": cores, "nside": nside, "spin": spin,
        "seconds": samples, "cl_path": out, "nell": int(np.shape(cl)[-1]),
        **_memory(engine),
    }), flush=True)


def main():
    spin = int(sys.argv[2])
    engine = sys.argv[3] if len(sys.argv) > 3 else "namaster"
    cores = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    for nside in (int(x) for x in sys.argv[1].split(",")):
        try:
            one(nside, spin, engine, cores)
        except Exception as exc:
            print(f"FAIL nside={nside} spin={spin} {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
