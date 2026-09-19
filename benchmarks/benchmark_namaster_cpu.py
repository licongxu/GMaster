"""NaMaster (pymaster) CPU wallclock benchmark for spin-0 TT MASTER.

Research Ops checklist:
  - nside sweep with fixed RNG seed, nlb=50 linear bins, n_iter=3, ones mask
  - warm = last of >=2 timed runs after one untimed discard
  - timer covers field + coupling matrix + coupled cell + decouple only
  - logs nproc, OMP_NUM_THREADS, CPU model, pymaster + ducc0 versions
"""

from __future__ import annotations

import argparse
import gc
import os
import platform
import sys
import time

import ducc0
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


def _ducc0_version() -> str:
    return getattr(ducc0, "__version__", "unknown")


def _run_pipeline(mask: np.ndarray, map_t: np.ndarray, lmax: int, nlb: int, n_iter: int):
    """Timed scope: field + coupling + coupled cell + decouple (no I/O)."""
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
    timed_repeats: int = 2,
) -> dict:
    lmax = 3 * nside - 1
    npix = 12 * nside**2
    rng = np.random.default_rng(seed)
    mask = np.ones(npix, dtype=np.float64)
    map_t = rng.normal(size=npix).astype(np.float64)

    # Untimed discard.
    _run_pipeline(mask, map_t, lmax, nlb, n_iter)
    gc.collect()

    timed: list[float] = []
    for _ in range(timed_repeats):
        t0 = time.perf_counter()
        _run_pipeline(mask, map_t, lmax, nlb, n_iter)
        timed.append(time.perf_counter() - t0)

    return {
        "nside": nside,
        "lmax": lmax,
        "warm_s": timed[-1],
        "timed_s": timed,
    }


def _markdown_table(rows: list[dict], meta: dict) -> str:
    lines = [
        "## NaMaster Cloud-CPU warm wallclock (spin-0 TT MASTER)",
        "",
        f"- **CPU:** {meta['cpu_model']}",
        f"- **nproc:** {meta['nproc']}",
        f"- **OMP_NUM_THREADS:** {meta['omp']}",
        f"- **pymaster:** {meta['pymaster']}",
        f"- **ducc0:** {meta['ducc0']}",
        f"- **map seed:** {meta['seed']} (same at every nside)",
        f"- **nlb:** {meta['nlb']} linear, **n_iter:** {meta['n_iter']}, **mask:** full-sky ones",
        f"- **lmax:** 3×nside−1",
        f"- **Timed:** field + coupling matrix + coupled cell + decouple (no I/O)",
        f"- **Warm:** last of {meta['timed_repeats']} timed runs after 1 untimed discard",
        f"- **GPU:** disabled (`CUDA_VISIBLE_DEVICES={meta['cuda']!r}`)",
        "",
        "| nside | lmax | warm (s) |",
        "|------:|-----:|---------:|",
    ]
    for row in rows:
        lines.append(f"| {row['nside']} | {row['lmax']} | {row['warm_s']:.3f} |")
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
    parser.add_argument(
        "--timed-repeats",
        type=int,
        default=2,
        help="Number of timed runs; warm = last (must be >= 2).",
    )
    parser.add_argument(
        "--max-warm-minutes",
        type=float,
        default=60.0,
        help="Stop sweep if warm run exceeds this many minutes.",
    )
    parser.add_argument("--out", type=str, default=".qwen/tmp/namaster_cpu_timings.md")
    args = parser.parse_args()
    if args.timed_repeats < 2:
        parser.error("--timed-repeats must be >= 2")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    nproc = os.cpu_count() or 0
    threads = len(os.sched_getaffinity(0))
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(threads)

    meta = {
        "cpu_model": _cpu_model(),
        "nproc": nproc,
        "threads": threads,
        "omp": os.environ.get("OMP_NUM_THREADS", "unset"),
        "pymaster": _pymaster_version(),
        "ducc0": _ducc0_version(),
        "seed": args.seed,
        "nlb": args.nlb,
        "n_iter": args.n_iter,
        "timed_repeats": args.timed_repeats,
        "cuda": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }

    print(
        f"NaMaster CPU: nproc={meta['nproc']} threads={meta['threads']} "
        f"OMP_NUM_THREADS={meta['omp']}",
        flush=True,
    )
    print(
        f"CPU: {meta['cpu_model']} | pymaster={meta['pymaster']} ducc0={meta['ducc0']} "
        f"seed={meta['seed']} n_iter={meta['n_iter']}",
        flush=True,
    )

    rows: list[dict] = []
    for nside in sorted(args.nsides):
        print(f"\n--- nside={nside} ---", flush=True)
        try:
            row = benchmark_nside(
                nside,
                nlb=args.nlb,
                n_iter=args.n_iter,
                seed=args.seed,
                timed_repeats=args.timed_repeats,
            )
        except MemoryError as exc:
            meta["stopped"] = nside
            meta["stop_reason"] = f"MemoryError: {exc}"
            print(f"OOM at nside={nside}: {exc}", flush=True)
            break

        runs_str = ", ".join(f"{t:.3f}" for t in row["timed_s"])
        print(
            f"nside={nside} lmax={row['lmax']} timed=[{runs_str}] warm={row['warm_s']:.3f}s",
            flush=True,
        )
        rows.append(row)

        out_path = args.out
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as fh:
            fh.write(_markdown_table(rows, meta) + "\n")

        if row["warm_s"] / 60.0 > args.max_warm_minutes:
            meta["stopped"] = nside
            meta["stop_reason"] = (
                f"warm {row['warm_s']:.1f}s exceeds --max-warm-minutes={args.max_warm_minutes}"
            )
            print(meta["stop_reason"], flush=True)
            break

    table = _markdown_table(rows, meta)
    print("\n" + table, flush=True)
    print(f"\nWrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
