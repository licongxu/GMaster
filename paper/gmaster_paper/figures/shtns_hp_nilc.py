"""SHTns a_lm of the masked NILC T and E maps on the HEALPix grid (fig_act_spectra).

Same field as NmtField(w, [map], n_iter=3) in act_nilc_cl.py: the map times the clipped
NILC footprint w, analysed on the HEALPix grid with n_iter = 3.  The spherical-harmonic
step is SHTns's GPU Legendre kernels with SHTns's own float32 recurrence
(SHTNS_GPU_REC_PREC=1, everything float32), through the HEALPix port in
benchmarks/shtns_healpix/shtns_healpix.py.  plot_act_spectra.py decouples these a_lm with
the same GMaster binned coupling matrices as the GMaster curve.

  CUDA_VISIBLE_DEVICES=0 python shtns_hp_nilc.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE = ROOT / ".qwen" / "tmp" / "act_nilc"
OUT = HERE / "cl_roundtrip" / "act_nilc_shtns_hp_fp32.npz"
NSIDE = 2048


def main():
    os.environ["SHTNS_GPU_REC_PREC"] = "1"
    sys.path.insert(0, str(ROOT / "benchmarks" / "shtns_healpix"))
    import cupy as cp
    from shtns_healpix import HealpixSHT

    w = np.clip(np.load(CACHE / f"mask_n{NSIDE}.npy"), 0, None)
    sht = HealpixSHT(NSIDE, precision="fp32")
    out = {}
    for name in ("T", "E"):
        m = cp.asarray((w * np.load(CACHE / f"{name}_n{NSIDE}.npy")).astype(np.float32))
        t0 = time.perf_counter()
        a = sht.map2alm(m, n_iter=3)
        cp.cuda.Device().synchronize()
        print(f"{name}: map2alm n_iter=3 {time.perf_counter() - t0:.3f}s", flush=True)
        out["alm" + name] = cp.asnumpy(a).astype(np.complex128)
    np.savez(OUT, nside=NSIDE, precision="fp32 recurrence (SHTNS_GPU_REC_PREC=1)", **out)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
