"""s2fft HEALPix SHT timing (GPU, JAX) at L = 3 nside, spin 0 and 2.

Methods: "jax" (on-the-fly Price-McEwen recursion) and "jax_cuda" (same with s2fft's custom
CUDA HEALPix FFT primitive, when its extension is built).  Precision follows JAX_ENABLE_X64
(run once with =1 and once with =0).  Warm medians of jitted calls, device-resident inputs.
Spin-0 accuracy: forward vs ducc0 adjoint synthesis * 4 pi / npix (s2fft's HEALPix quadrature
is the constant pixel weight), max |d flm| / max |ref| over m >= 0.

Usage: JAX_ENABLE_X64=1 python bench_s2fft.py <method> <nside> [<nside> ...]
"""
import os
import sys
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", os.environ.get("JAX_ENABLE_X64", "1") == "1")
import jax.numpy as jnp
import ducc0
import s2fft


def med(f, reps=5):
    jax.block_until_ready(f())
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(f())
        ts.append(time.perf_counter() - t)
    return 1e3 * float(np.median(ts))


def main():
    method = sys.argv[1]
    x64 = jax.config.jax_enable_x64
    rdt, cdt = (np.float64, np.complex128) if x64 else (np.float32, np.complex64)
    for nside in [int(a) for a in sys.argv[2:]]:
        L = 3 * nside
        npix = 12 * nside * nside
        rng = np.random.default_rng(0)
        out = []
        for spin in (0, 2):
            if spin == 0:
                fmap = rng.normal(size=npix)
                fd = jnp.asarray(fmap.astype(rdt))
            else:
                fmap = rng.normal(size=npix) + 1j * rng.normal(size=npix)
                fd = jnp.asarray(fmap.astype(cdt))
            reality = spin == 0
            fwd = jax.jit(lambda f: s2fft.forward(f, L, spin=spin, nside=nside, sampling="healpix",
                                                  method=method, reality=reality))
            try:
                flm = fwd(fd)
                tf = med(lambda: fwd(fd))
                inv = jax.jit(lambda a: s2fft.inverse(a, L, spin=spin, nside=nside, sampling="healpix",
                                                      method=method, reality=reality))
                ti = med(lambda: inv(flm))
            except Exception as exc:  # noqa: BLE001 - report and move on (OOM at large L)
                out.append(f"spin{spin}: FAILED {type(exc).__name__}: {str(exc)[:80]}")
                continue
            err = ""
            if spin == 0:
                ref = _ducc_healpix_adjoint(fmap, L - 1, nside) * (4 * np.pi / npix)
                got = np.asarray(flm)
                gm = np.zeros_like(ref)
                idx = 0
                for m in range(L):
                    gm[idx:idx + L - m] = got[m:, L - 1 + m]
                    idx += L - m
                err = f" err {np.max(np.abs(gm - ref)) / np.max(np.abs(ref)):.1e}"
            out.append(f"spin{spin}: fwd {tf:.2f} inv {ti:.2f} ms{err}")
        print(f"s2fft method={method} x64={x64} nside={nside} L={L} | " + "  ".join(out), flush=True)


def _ducc_healpix_adjoint(fmap, lmax, nside):
    base = ducc0.healpix.Healpix_Base(nside, "RING")
    geom = base.sht_info()
    return ducc0.sht.experimental.adjoint_synthesis(
        map=fmap[None].astype(np.float64), spin=0, lmax=lmax, nthreads=96, **geom)[0]


if __name__ == "__main__":
    main()
