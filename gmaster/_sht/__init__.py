"""Spherical-harmonic transform engines (private).

The public transforms `gmaster.map2alm` / `gmaster.alm2map` hand HEALPix maps to
`healpix`, which splits every transform into two separable stages:

    maps  --[ring FFT, `rings`]-->  ftm(theta, m)  --[latitudinal engine]-->  alm(ell, m)

and adds NaMaster's Jacobi refinement (`n_iter`) around them.  The latitudinal stage is
where the cost is, and the engine that runs it is chosen per call from the band limit
``L = lmax + 1``, the spin, the device and the memory available:

    march_v2       CUDA float32 difference-form Wigner-d march (default on NVIDIA GPUs;
                   an OpenMP port runs when JAX is on the CPU)
    dc             divide-and-conquer engine, O(L^2 log L), for L in
                   [GMASTER_DC_MIN_L, GMASTER_DC_MAX_L]; plan built once on the host
    theta_matrix   spin 0 with a precomputed Legendre band (small L)
    spin_slice     spin 2 with precomputed Wigner-d slabs (small L)
    sht_pallas     portable Pallas kernels with on-the-fly recurrences
    spin_march     Pallas difference-form march (portable fallback of march_v2)

Supporting modules: `band_pallas` (band builder), `spin_contract` (slab contraction),
`dfp32` (double-float32 analysis), `cuda_gpu` (device detection).

`gmaster.set_latitudinal_method("march" | "dc" | "auto")` pins the engine choice.
"""
