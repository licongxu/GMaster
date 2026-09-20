# Kaggle T4 nside ladder (19 Sep 2026)

Live run of `examples/gmaster_kaggle_nside_sweep.ipynb` on Kaggle **GPU T4 x2**,
one visible device (`CUDA_VISIBLE_DEVICES=0`), vs NaMaster on the session CPUs.

- Kernel: https://www.kaggle.com/code/licongxu/notebook653f325310
- JAX 0.7.2, `device=Tesla T4`, 4 CPU cores, 33.7 GB host RAM, `nvcc` present
- `cuda_gpu=True`, `kind_ok=True`, **v2 CUDA march library: enabled**
- Spin 0, `n_iter=3`, `lmax = 3 nside - 1`, `NmtBin.from_lmax_linear(lmax, 50)`
- Galactic cut 20° + C1 (2°). Cheap `C_ell ~ 1/(ell(ell+1))` + 10 µK-arcmin noise
  (not CAMB). TT coupling route: GEMM at every size below.

Warmed GMaster is the timed pass after the first (compile) call.

| nside | lmax | march | GM cold s | GM warm s | field s | coup s | NM s | NM/GM | rms(GM/NM−1) | device peak GB |
|------:|-----:|:-----:|----------:|----------:|--------:|-------:|-----:|------:|-------------:|---------------:|
|    64 |  191 |  yes  |     11.28 |      0.04 |    0.01 |   0.01 | 0.08 |  2.17 |     2.70e-08 |          0.045 |
|   128 |  383 |  yes  |     11.87 |      0.04 |    0.01 |   0.02 | 0.25 |  6.75 |     4.42e-08 |          0.088 |
|   256 |  767 |  yes  |     17.17 |      0.07 |    0.03 |   0.03 | 1.77 | 26.29 |     6.58e-08 |          0.219 |
|   512 | 1535 |  yes  |     15.62 |      0.36 |    0.18 |   0.10 |12.93 | 37.08 |     3.09e-07 |          0.923 |
|  1024 | 3071 |  yes  |     40.49 |      2.02 |    0.95 |   1.04 |98.70 | 48.90 |     1.15e-06 |           3.67 |
|  2048 | 6143 |  yes  |         — |         — |       — |      — |    — |     — |            — | OOM            |
|  4096 |12287 |  yes  |         — |         — |       — |      — |    — |     — |            — | OOM            |

**2048** died in `map2alm_core_pallas_pair` with

```
jaxlib._jax.XlaRuntimeError: INTERNAL: Failed to allocate 402653184 bytes for new constant
```

(~384 MB). T4 is 16 GB; this is fragmentation after the nside=1024 kernels, not
a 15 GB working set.

**4096** printed `=== nside=4096  lmax=12287  march=True  TT=GEMM ===` after that
OOM and never returned a GMaster or NaMaster time. Same 16 GB T4, four times the
pixels and twice the L of the allocation that already failed. Reported as OOM;
the ladder does not claim a T4 number at 2048 or 4096.

A separate single-nside CAMB demo on the same T4 at NSIDE=1024 was GMaster
1.98 s warmed vs NaMaster 87.65 s (44.3×), march on, 3.33 GB, but
`rms(GM/NM-1)=6.17e-4` on that CAMB+noise map. The 1/ell(ell+1) ladder at 1024
is 1.15e-6.
