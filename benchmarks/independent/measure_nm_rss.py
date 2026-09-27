"""Measure NaMaster end-to-end MASTER peak RSS. CPU only, one pipeline per cell."""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results_nm_rss.json"
NSIDES = (64, 128, 256, 512, 1024, 2048, 4096)
SPINS = (0, 2)
NTHREADS = 96


def _worker(nside: int, spin: int) -> dict:
    import numpy as np
    import pymaster as nmt

    sys.path.insert(0, str(ROOT))
    from benchmarks.run_paper_v2_benchmarks import _patch_pymaster_i64

    _patch_pymaster_i64()
    lmax = 3 * nside - 1
    npix = 12 * nside**2
    rng = np.random.default_rng(nside + spin)
    theta = np.arccos(1 - 2 * (np.arange(npix) + 0.5) / npix)
    mask = np.clip((np.cos(theta) + 0.35) / 0.7, 0, 1) ** 2
    map_t = rng.normal(size=npix)
    map_q = rng.normal(size=npix)
    map_u = rng.normal(size=npix)
    bins = nmt.NmtBin.from_lmax_linear(lmax, 30)
    if spin == 0:
        field = nmt.NmtField(mask, [map_t], n_iter=3)
    else:
        field = nmt.NmtField(mask, [map_q, map_u], n_iter=3, spin=2)
    workspace = nmt.NmtWorkspace()
    workspace.compute_coupling_matrix(field, field, bins)
    coupled = nmt.compute_coupled_cell(field, field)
    workspace.decouple_cell(coupled)
    return {
        "nside": nside,
        "spin": spin,
        "omp_threads": int(os.environ.get("OMP_NUM_THREADS") or 0),
        "peak_rss_gb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6),
    }


def main() -> None:
    if "--cell" in sys.argv:
        i = sys.argv.index("--cell")
        print("NM_RSS_JSON:" + json.dumps(_worker(int(sys.argv[i + 1]), int(sys.argv[i + 2]))))
        return

    rows = []
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = str(NTHREADS)
    env["OPENBLAS_NUM_THREADS"] = "1"
    py = sys.executable
    script = str(Path(__file__).resolve())
    for nside in NSIDES:
        for spin in SPINS:
            print(f"NaMaster RSS nside={nside} spin={spin} omp={NTHREADS}", flush=True)
            proc = subprocess.run(
                [py, script, "--cell", str(nside), str(spin)],
                cwd=str(ROOT),
                env=env,
                capture_output=True,
                text=True,
            )
            line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("NM_RSS_JSON:")), "")
            if proc.returncode != 0 or not line:
                print(proc.stdout)
                print(proc.stderr)
                raise SystemExit(f"failed nside={nside} spin={spin} code={proc.returncode}")
            row = json.loads(line.split(":", 1)[1])
            rows.append(row)
            print(f"  peak_rss_gb={row['peak_rss_gb']:.3f}", flush=True)
            OUT.write_text(json.dumps({"pipeline": rows}, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
