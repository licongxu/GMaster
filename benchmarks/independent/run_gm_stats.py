"""GMaster v2 wall-clock mean/std after warmup. Does not modify gmaster source."""
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("GMASTER_MARCH_V2", "1")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import gmaster as nmt


def _block(result):
    jax.tree.map(
        lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else None,
        result,
        is_leaf=lambda x: hasattr(x, "block_until_ready"),
    )
    pending = [getattr(result, "__dict__", None)]
    while pending:
        for value in (pending.pop() or {}).values():
            if hasattr(value, "block_until_ready"):
                value.block_until_ready()
            elif isinstance(value, dict):
                pending.append(value)
            elif isinstance(value, (list, tuple)):
                pending.append(dict(enumerate(value)))


def _stats(samples_s):
    a = np.asarray(samples_s, np.float64) * 1e3
    return {
        "n": int(a.size),
        "samples_ms": [float(x) for x in a],
        "median_ms": float(np.median(a)),
        "mean_ms": float(np.mean(a)),
        "std_ms": float(np.std(a, ddof=1) if a.size > 1 else 0.0),
        "min_ms": float(np.min(a)),
        "max_ms": float(np.max(a)),
        "cv": float((np.std(a, ddof=1) / np.mean(a)) if a.size > 1 and np.mean(a) else 0.0),
    }


def _nrep(nside, kind):
    if kind == "sht":
        return 7 if nside <= 2048 else 5
    if nside <= 1024:
        return 7
    if nside == 2048:
        return 5
    return 3


def sht_cell(nside, spin):
    lmax = 3 * nside - 1
    npix = 12 * nside**2
    nmaps = 1 if spin == 0 else 2
    rng = np.random.default_rng(42 + nside + spin)
    maps = jax.device_put(rng.normal(size=(nmaps, npix)))
    alms = rng.normal(size=(nmaps, nmt.NmtAlmInfo(lmax).nelem)) + 1j * rng.normal(
        size=(nmaps, nmt.NmtAlmInfo(lmax).nelem)
    )
    alms[:, : lmax + 1] = alms[:, : lmax + 1].real
    alms = jax.device_put(alms)
    map_info = nmt.NmtMapInfo(None, (npix,))
    alm_info = nmt.NmtAlmInfo(lmax)
    nmt.set_ring_precision("fp32")
    nmt.set_table_precision("fp32")
    nmt.set_coupling_precision("fp32")
    jax.clear_caches()
    gc.collect()
    nmt.map2alm(maps, spin, map_info, alm_info, n_iter=0).block_until_ready()
    nmt.alm2map(alms, spin, map_info, alm_info).block_until_ready()
    ana, syn = [], []
    for _ in range(_nrep(nside, "sht")):
        t0 = time.perf_counter()
        nmt.map2alm(maps, spin, map_info, alm_info, n_iter=0).block_until_ready()
        ana.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        nmt.alm2map(alms, spin, map_info, alm_info).block_until_ready()
        syn.append(time.perf_counter() - t0)
    return {"nside": nside, "spin": spin, "ana": _stats(ana), "syn": _stats(syn)}


def pipe_cell(nside, spin):
    lmax = 3 * nside - 1
    npix = 12 * nside**2
    rng = np.random.default_rng(nside + spin)
    theta = np.arccos(1 - 2 * (np.arange(npix) + 0.5) / npix)
    mask = np.clip((np.cos(theta) + 0.35) / 0.7, 0, 1) ** 2
    mt, mq, mu = rng.normal(size=npix), rng.normal(size=npix), rng.normal(size=npix)
    m, mt, mq, mu = map(jnp.asarray, (mask, mt, mq, mu))
    nmt.set_ring_precision("fp32")
    nmt.set_table_precision("fp32")
    nmt.set_coupling_precision("fp32")
    jax.clear_caches()
    gc.collect()
    bins = nmt.NmtBin.from_lmax_linear(lmax, 30)

    def full():
        ff = nmt.NmtField(m, [mt], n_iter=3) if spin == 0 else nmt.NmtField(m, [mq, mu], n_iter=3, spin=2)
        ww = nmt.NmtWorkspace()
        ww.compute_coupling_matrix(ff, ff, bins)
        cc = nmt.compute_coupled_cell(ff, ff)
        return ww.decouple_cell(cc), cc

    _block(full())
    samples = []
    for _ in range(_nrep(nside, "pipe")):
        t0 = time.perf_counter()
        _block(full())
        samples.append(time.perf_counter() - t0)
    return {"nside": nside, "spin": spin, "total": _stats(samples)}


def main():
    out = Path(__file__).resolve().parent / "results_gm_stats.json"
    data = {"sht": [], "pipeline": []}
    nsides = [64, 128, 256, 512, 1024, 2048, 4096]
    for nside in nsides:
        for spin in (0, 2):
            print(f"SHT nside={nside} spin={spin}", flush=True)
            row = sht_cell(nside, spin)
            data["sht"].append(row)
            a, s = row["ana"], row["syn"]
            print(
                f"  ana mean={a['mean_ms']:.3f} std={a['std_ms']:.3f} med={a['median_ms']:.3f} cv={a['cv']:.3f} "
                f"syn mean={s['mean_ms']:.3f} std={s['std_ms']:.3f}",
                flush=True,
            )
            out.write_text(json.dumps(data, indent=2))
    for nside in nsides:
        for spin in (0, 2):
            if nside == 4096 and spin == 2:
                print("PIPE skip 4096 spin 2 (OOM)", flush=True)
                continue
            print(f"PIPE nside={nside} spin={spin}", flush=True)
            row = pipe_cell(nside, spin)
            data["pipeline"].append(row)
            t = row["total"]
            print(
                f"  total mean={t['mean_ms']:.2f} std={t['std_ms']:.2f} med={t['median_ms']:.2f} cv={t['cv']:.3f}",
                flush=True,
            )
            out.write_text(json.dumps(data, indent=2))
    print("GM_STATS_DONE", out)


if __name__ == "__main__":
    main()
