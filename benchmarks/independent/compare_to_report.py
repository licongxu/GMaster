"""Diff an independent v2 JSON run against docs/v2_march_benchmark_report.md gold numbers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Isolated SHT table (report §2). Times in ms. rel is GM vs DUCC |da_lm|/max|a_lm|.
SHT = {
    (64, 0): dict(ducc_ana=0.52, fp32_ana=0.18, fp64_ana=0.24, ducc_syn=0.38, fp32_syn=0.15, fp64_syn=0.22, rel=2.1e-6),
    (64, 2): dict(ducc_ana=0.84, fp32_ana=0.25, fp64_ana=0.31, ducc_syn=0.65, fp32_syn=0.22, fp64_syn=0.29, rel=2.4e-6),
    (128, 0): dict(ducc_ana=1.15, fp32_ana=0.28, fp64_ana=0.38, ducc_syn=0.82, fp32_syn=0.25, fp64_syn=0.36, rel=3.8e-6),
    (128, 2): dict(ducc_ana=1.95, fp32_ana=0.42, fp64_ana=0.53, ducc_syn=1.45, fp32_syn=0.39, fp64_syn=0.51, rel=4.2e-6),
    (256, 0): dict(ducc_ana=2.87, fp32_ana=0.46, fp64_ana=0.62, ducc_syn=1.93, fp32_syn=0.42, fp64_syn=0.66, rel=6.5e-6),
    (256, 2): dict(ducc_ana=4.88, fp32_ana=0.77, fp64_ana=0.93, ducc_syn=3.37, fp32_syn=0.83, fp64_syn=0.95, rel=7.1e-6),
    (512, 0): dict(ducc_ana=13.55, fp32_ana=1.86, fp64_ana=2.19, ducc_syn=11.97, fp32_syn=1.81, fp64_syn=2.47, rel=1.4e-5),
    (512, 2): dict(ducc_ana=25.13, fp32_ana=2.23, fp64_ana=3.88, ducc_syn=21.93, fp32_syn=2.52, fp64_syn=4.04, rel=1.5e-5),
    (1024, 0): dict(ducc_ana=55.10, fp32_ana=7.17, fp64_ana=10.09, ducc_syn=50.27, fp32_syn=7.21, fp64_syn=10.13, rel=2.5e-5),
    (1024, 2): dict(ducc_ana=104.73, fp32_ana=13.26, fp64_ana=18.38, ducc_syn=95.91, fp32_syn=12.92, fp64_syn=18.14, rel=2.6e-5),
    (2048, 0): dict(ducc_ana=285.48, fp32_ana=41.81, fp64_ana=53.52, ducc_syn=269.26, fp32_syn=36.59, fp64_syn=49.18, rel=5.7e-5),
    (2048, 2): dict(ducc_ana=571.50, fp32_ana=81.75, fp64_ana=101.64, ducc_syn=538.71, fp32_syn=81.41, fp64_syn=99.91, rel=4.9e-5),
    (4096, 0): dict(ducc_ana=1832.55, fp32_ana=265.50, fp64_ana=308.56, ducc_syn=1833.50, fp32_syn=301.49, fp64_syn=347.36, rel=1.1e-4),
    (4096, 2): dict(ducc_ana=3794.93, fp32_ana=674.14, fp64_ana=764.60, ducc_syn=3673.44, fp32_syn=657.13, fp64_syn=735.39, rel=1.1e-4),
}

# Pipeline summary (report §3). Bandpower / coupled rel. dev. as printed.
PIPE = {
    (64, 0): dict(nm=8.1, fp32=5.6, fp64=5.5, rel=1.46e-7, crel=1.76e-7),
    (64, 2): dict(nm=12.7, fp32=2.2, fp64=2.5, rel=2.69e-7, crel=2.31e-7),
    (128, 0): dict(nm=20.3, fp32=1.7, fp64=1.8, rel=1.54e-7, crel=1.78e-7),
    (128, 2): dict(nm=39.5, fp32=3.5, fp64=4.3, rel=3.50e-7, crel=2.84e-7),
    (256, 0): dict(nm=58.7, fp32=5.3, fp64=5.5, rel=5.32e-7, crel=7.81e-7),
    (256, 2): dict(nm=158.5, fp32=10.6, fp64=19.3, rel=6.91e-7, crel=6.87e-7),
    (512, 0): dict(nm=310.2, fp32=22.5, fp64=24.3, rel=6.30e-7, crel=2.82e-6),
    (512, 2): dict(nm=642.1, fp32=38.6, fp64=99.7, rel=3.13e-6, crel=3.45e-6),
    (1024, 0): dict(nm=1658.2, fp32=126.8, fp64=149.2, rel=2.38e-6, crel=1.41e-5),
    (1024, 2): dict(nm=3132.8, fp32=161.2, fp64=529.4, rel=5.63e-6, crel=2.87e-5),
    (2048, 0): dict(nm=10239.9, fp32=1614.0, fp64=1750.1, rel=1.74e-5, crel=6.42e-5),
    (2048, 2): dict(nm=17673.2, fp32=992.1, fp64=3659.2, rel=1.37e-5, crel=2.62e-5),
    (4096, 0): dict(nm=71583.0, fp32=7458.0, fp64=9820.0, rel=6.20e-5, crel=1.09e-4),
    (4096, 2): dict(nm=108240.0, fp32=8805.0, fp64=14210.0, rel=6.50e-5, crel=1.10e-4),
}


def _time_ok(got, gold, *, rel=0.25, abs_ms=2.0):
    if gold != gold or got != got:
        return False, "nan"
    if gold == 0:
        return abs(got) <= abs_ms, f"got={got:.3g} gold=0"
    frac = abs(got - gold) / abs(gold)
    return frac <= rel or abs(got - gold) <= abs_ms, f"got={got:.3g} gold={gold:.3g} frac={frac:.2f}"


def _acc_ok(got, gold, *, lo=0.4, hi=2.5):
    if gold != gold or got != got:
        return False, "nan"
    ratio = got / gold if gold else float("inf")
    return lo <= ratio <= hi, f"got={got:.3e} gold={gold:.3e} ratio={ratio:.2f}"


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/independent/results_v2_independent.json")
    data = json.loads(path.read_text())
    sht = {(r["nside"], r["spin"]): r for r in data.get("sht", [])}
    pipe = {(r["nside"], r["spin"]): r for r in data.get("pipeline", [])}

    fail = 0
    miss = 0
    print(f"INDEPENDENT vs REPORT  file={path}")
    print("=== SHT ===")
    for key, gold in SHT.items():
        row = sht.get(key)
        if row is None:
            print(f"MISSING SHT {key}")
            miss += 1
            continue
        checks = [
            ("ducc_ana", row["ref"]["ana_ms"], gold["ducc_ana"]),
            ("fp32_ana", row["fp32"]["ana_ms"], gold["fp32_ana"]),
            ("fp64_ana", row["fp64"]["ana_ms"], gold["fp64_ana"]),
            ("ducc_syn", row["ref"]["syn_ms"], gold["ducc_syn"]),
            ("fp32_syn", row["fp32"]["syn_ms"], gold["fp32_syn"]),
            ("fp64_syn", row["fp64"]["syn_ms"], gold["fp64_syn"]),
        ]
        for name, got, g in checks:
            ok, msg = _time_ok(got, g)
            if not ok:
                print(f"FAIL SHT {key} {name} {msg}")
                fail += 1
        ok, msg = _acc_ok(row["fp32"]["rel_dalm"], gold["rel"])
        if not ok:
            print(f"FAIL SHT {key} rel_dalm {msg}")
            fail += 1

    print("=== PIPELINE ===")
    for key, gold in PIPE.items():
        row = pipe.get(key)
        if row is None:
            print(f"MISSING PIPE {key}")
            miss += 1
            continue
        ref_t = row.get("ref", {}).get("times", {}).get("total", float("nan"))
        for name, got, g in (
            ("nm", ref_t, gold["nm"]),
            ("fp32", row["fp32"]["times"]["total"], gold["fp32"]),
            ("fp64", row["fp64"]["times"]["total"], gold["fp64"]),
        ):
            ok, msg = _time_ok(got, g, rel=0.30, abs_ms=5.0)
            if not ok:
                print(f"FAIL PIPE {key} {name} {msg}")
                fail += 1
        acc = row["fp32"].get("accuracy", {})
        ok, msg = _acc_ok(acc.get("rel_dCl", float("nan")), gold["rel"])
        if not ok:
            print(f"FAIL PIPE {key} rel_dCl {msg}")
            fail += 1
        ok, msg = _acc_ok(acc.get("rel_coupled_diff", float("nan")), gold["crel"])
        if not ok:
            print(f"FAIL PIPE {key} rel_coupled {msg}")
            fail += 1

    print(f"SUMMARY fail={fail} missing={miss}")
    if fail or miss:
        sys.exit(1)


if __name__ == "__main__":
    main()
