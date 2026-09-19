"""NaMaster (pymaster) CPU wallclock benchmark for spin-0 TT MASTER.

Matches the GMaster trust-demo timing scope: field + coupling matrix +
coupled cell + decouple; I/O excluded.  Synthetic full-sky Gaussian map
with a ones mask (FLAMINGO-style f_sky=1).

Warm time = last of 2 timed runs after one cold discard.
"""

from __future__ import annotations

import argparse
import gc
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pymaster as nmt


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _pymaster_version() -> str:
    return getattr(nmt, "__version__", getattr(nmt, "version", "unknown"))


def _run_pipeline(mask: np.ndarray, map_t: np.ndarray, lmax: int, nlb: int, n_iter: int):
    field = nmt.NmtField(mask, [map_t], n_iter=n_iter)
    bins = nmt.NmtBin.from_lmax_linear(lmax, nlb)
    workspace = nmt.NmtWorkspace()
    workspace.compute_coupling_matrix(field, field, bins)
    cl_coupled = nmt.compute_coupled_cell(field, field)
    return workspace.decouple_cell(cl_coupled)


def benchmark_nside(
    nside: int,
    *,
    nlb: int,
    n_iter: int,
    seed: int,
    warm_repeats: int = 2,
) -> dict:
    lmax = 3 * nside - 1
    npix = 12 * nside**2
    rng = np.random.default_rng(seed)
    mask = np.ones(npix, dtype=np.float64)
    map_t = rng.normal(size=npix).astype(np.float64)

    # Cold discard (not reported as warm).
    _run_pipeline(mask, map_t, lmax, nlb, n_iter)
    gc.collect()

    cold_times: list[float] = []
    warm_times: list[float] = []
    for i in range(warm_repeats):
        t0 = time.perf_counter()
        _run_pipeline(mask, map_t, lmax, nlb, n_iter)
        elapsed = time.perf_counter() - t0
        if i == 0:
            cold_times.append(elapsed)
        else:
            warm_times.append(elapsed)

    return {
        "nside": nside,
        "lmax": lmax,
        "npix": npix,
        "cold_s": cold_times[0] if cold_times else float("nan"),
        "warm_s": warm_times[-1] if warm_times else float("nan"),
        "runs_s": cold_times + warm_times,
    }


def _markdown_table(rows: list[dict], meta: dict) -> str:
    lines = [
        "## NaMaster CPU wallclock (spin-0 TT MASTER)",
        "",
        f"- **CPU:** {meta['cpu_model']}",
        f"- **Threads:** {meta['threads']} (`OMP_NUM_THREADS={meta['omp']}`)",
        f"- **pymaster:** {meta['pymaster']}",
        f"- **nlb:** {meta['nlb']}, **n_iter:** {meta['n_iter']}, **mask:** ones (f_sky=1)",
        f"- **lmax:** 3×nside−1",
        f"- **Timed:** field + coupling matrix + coupled cell + decouple (I/O excluded)",
        f"- **Warm:** last of {meta['warm_repeats']} runs after 1 cold discard",
        f"- **GPU:** disabled (`CUDA_VISIBLE_DEVICES={meta['cuda']!r}`)",
        "",
        "| nside | lmax | cold (s) | warm (s) |",
        "|------:|-----:|---------:|---------:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['nside']} | {row['lmax']} | {row['cold_s']:.3f} | {row['warm_s']:.3f} |"
        )
    if meta.get("stopped"):
        lines.extend(["", f"**Stopped at nside {meta['stopped']}:** {meta['stop_reason']}"])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nsides",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024, 2048, 4096],
    )
    parser.add_argument("--nlb", type=int, default=50)
    parser.add_argument("--n-iter", type=int, default=3)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--warm-repeats", type=int, default=2)
    parser.add_argument(
        "--max-warm-minutes",
        type=float,
        default=60.0,
        help="Abort remaining nsides if warm run exceeds this many minutes.",
    )
    parser.add_argument("--out", type=str, default=".qwen/tmp/namaster_cpu_timings.md")
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    threads = len(os.sched_getaffinity(0))
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(threads)

    meta = {
        "cpu_model": _cpu_model(),
        "threads": threads,
        "omp": os.environ.get("OMP_NUM_THREADS", "unset"),
        "pymaster": _pymaster_version(),
        "nlb": args.nlb,
        "n_iter": args.n_iter,
        "warm_repeats": args.warm_repeats,
        "cuda": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }

    print(
        f"NaMaster CPU benchmark: threads={meta['threads']} omp={meta['omp']} "
        f"pymaster={meta['pymaster']}",
        flush=True,
    )
    print(f"CPU: {meta['cpu_model']}", flush=True)

    rows: list[dict] = []
    for nside in sorted(args.nsides):
        print(f"\n--- nside={nside} ---", flush=True)
        try:
            row = benchmark_nside(
                nside,
                nlb=args.nlb,
                n_iter=args.n_iter,
                seed=args.seed,
                warm_repeats=args.warm_repeats,
            )
        except MemoryError as exc:
            meta["stopped"] = nside
            meta["stop_reason"] = f"MemoryError: {exc}"
            print(f"OOM at nside={nside}: {exc}", flush=True)
            break

        runs_str = ", ".join(f"{t:.3f}" for t in row["runs_s"])
        print(
            f"nside={nside} lmax={row['lmax']} runs=[{runs_str}] "
            f"cold={row['cold_s']:.3f}s warm={row['warm_s']:.3f}s",
            flush=True,
        )
        rows.append(row)

        warm_min = row["warm_s"] / 60.0
        if warm_min > args.max_warm_minutes:
            meta["stopped"] = nside
            meta["stop_reason"] = (
                f"warm run {row['warm_s']:.1f}s exceeds --max-warm-minutes={args.max_warm_minutes}"
            )
            print(meta["stop_reason"], flush=True)
            break

    table = _markdown_table(rows, meta)
    print("\n" + table, flush=True)

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(table + "\n")
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
