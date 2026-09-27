"""Relative difference of saved ACT spectra against the 32-core NaMaster files."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path("benchmarks/independent")


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        n = min(a.shape[-1], b.shape[-1])
        a, b = a[..., :n], b[..., :n]
    denom = np.max(np.abs(b))
    if denom == 0:
        return float("nan")
    return float(np.max(np.abs(a - b)) / denom)


def main():
    rows = []
    for ref in sorted(HERE.glob("act_cl_nm32_n*_s*.npy")):
        parts = ref.stem.split("_")
        nside, spin = parts[-2][1:], parts[-1][1:]
        for other in sorted(HERE.glob(f"act_cl_*_n{nside}_s{spin}.npy")):
            if other == ref:
                continue
            rows.append({
                "nside": int(nside), "spin": int(spin),
                "other": other.name, "rel": rel(np.load(other), np.load(ref)),
            })
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    sys.exit(main())
