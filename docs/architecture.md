# GMaster architecture

This guide is for people who want to understand, maintain or extend GMaster. Users of the
estimator only need the [README](../README.md) and the [tutorials](../tutorials/).

## Package layout

```
gmaster/
  __init__.py          public API: `import gmaster as nmt` mirrors `import pymaster as nmt`
  bins.py              NmtBin, NmtBinFlat                         (bandpower binning)
  field.py             NmtField                                   (curved-sky fields)
  field_flat.py        NmtFieldFlat                               (flat-sky fields)
  field_catalog.py     NmtFieldCatalog, ...Clustering, ...Momentum (point-set fields)
  workspaces.py        NmtWorkspace, compute_coupled_cell, ...    (curved-sky MASTER)
  workspaces_flat.py   NmtWorkspaceFlat, ...                      (flat-sky MASTER)
  covariance.py        NmtCovarianceWorkspace, gaussian_covariance, ...
  utils.py             NmtMapInfo, NmtAlmInfo, map2alm, alm2map, mask_apodization,
                       synfast_spherical, ... (the counterpart of pymaster.utils)
  nusht.py             general (non-uniform) SHT, an add-on beyond NaMaster
  _config.py           run-time settings and the device-memory policy
  _coupling_tt_cuda.py CUDA kernel for the spin-0 coupling matrix
  _nmt_bin64.py        64-bit patch for NaMaster's MCM binning (used by the benchmarks)
  _sht/                spherical-harmonic transform engines (private)
    healpix.py           routing and Jacobi refinement of the HEALPix transforms
    rings.py             azimuthal stage: ring FFTs and chirp-Z transforms
    march_v2.py          CUDA float32 difference-form Wigner-d march (default on NVIDIA GPUs)
    dc.py                divide-and-conquer latitudinal engine, O(L^2 log L)
    theta_matrix.py      spin 0 with a precomputed Legendre band (small L)
    spin_slice.py        spin 2 with precomputed Wigner-d slabs (small L)
    sht_pallas.py, band_pallas.py, spin_march.py, spin_contract.py, dfp32.py
                         portable Pallas kernels and helpers
    cuda_gpu.py          device detection
  _native/             CUDA (.cu) and C++ (.cc/.cpp) sources, compiled on first use
```

Everything without a leading underscore is public and follows NaMaster 3.0's documented
behaviour; the private modules can change between releases.

## Data flow of one power spectrum

```
maps, mask ──► NmtField ──► a_lm of the masked map (map2alm, n_iter Jacobi iterations)
                  │
mask ─────────────┴──► a_lm of the mask ──► pseudo-C_ell of the mask
                                              │
bins ─────────────────► NmtWorkspace ◄────────┘  mode-coupling matrix M_ll' (depends on masks only),
                             │                   binned and inverted
fields ──► compute_coupled_cell ──► pseudo-C_ell ──► workspace.decouple_cell ──► bandpowers
```

The two expensive steps are the spherical-harmonic transforms (`map2alm` inside `NmtField`) and
the coupling matrix (`NmtWorkspace.compute_coupling_matrix`).

### Transforms

A HEALPix transform is split into two separable stages (`_sht/healpix.py`):

1. **Azimuthal** (`_sht/rings.py`): an FFT along each iso-latitude ring gives `ftm(theta, m)`.
   Equatorial rings have `4 nside >= L` pixels and take one batched FFT; polar rings are
   shorter than the band limit and use a chirp-Z (Bluestein) transform.
2. **Latitudinal**: for each order `m`, a sum over rings of `ftm` times the Wigner-d function
   `d^l_{m,-s}(theta)` (Legendre functions for spin 0). This is `O(L^3)` work and the stage every
   engine in `_sht/` implements differently.

Which latitudinal engine runs is decided per call from the band limit `L = lmax + 1`, the spin,
the device and the memory available:

| engine | used for | notes |
|---|---|---|
| `march_v2` | default on NVIDIA GPUs, both spins | float32 difference-form recurrence with a float64 seed; rows are generated in registers, nothing is tabulated. An OpenMP port runs when JAX is on the CPU. |
| `dc` | `GMASTER_DC_MIN_L <= L <= GMASTER_DC_MAX_L` (Nside 2048-4096 by default) | Christoffel-Darboux kernel, Cuppen / Gu-Eisenstat divide-and-conquer tree and 1-D fast multipole methods; `O(L^2 log L)`. Needs a plan built once on the host (minutes at Nside 2048-4096, cached in `~/.cache/gmaster`). |
| `theta_matrix`, `spin_slice` | small band limits, or `GMASTER_MARCH_V2=0` | exact float64 contraction with precomputed Legendre / Wigner-d tables held on the device while they fit. |
| `sht_pallas`, `spin_march` | portable fallbacks | Pallas kernels with on-the-fly recurrences. |

`gmaster.set_latitudinal_method("march" | "dc" | "auto")` pins the choice.

`map2alm` adds NaMaster's Jacobi refinement: each iteration synthesises the current `a_lm`,
subtracts the input map and analyses the residual. On the march routes the residual is formed
in ring-Fourier space (the fold of the synthesised spectrum against the map's own spectrum), so
no ring FFT runs inside the loop. At Nside 8192 the polarised transform runs as separate programs
with complex64 spectra so that one 96 GB card can hold it.

### Coupling matrices

`workspaces.py` computes the mode-coupling matrix of any spin pair:

* spin 0 x spin 0: Gauss-Legendre quadrature of products of Legendre polynomials, using
  `P_l(-x) = (-1)^l P_l(x)` to work on half of the nodes (three quarter-size matrix products); the
  Wigner-3j recurrence below `GMASTER_TT_QUADRATURE_LMAX` (48), with a CUDA kernel
  (`_coupling_tt_cuda.py`) where available; NaMaster's Toeplitz approximation is supported
  (`l_toeplitz`, `l_exact`, `dl_band`);
* any other spin pair (including pure-E/B): exact Gauss-Legendre quadrature of products of
  Wigner-d functions, i.e. two matrix products per block. Above `GMASTER_WD_STREAM_GIB` of tables
  (Nside 8192 spin 2) the quadrature is summed over blocks of nodes instead of holding the tables.

Large polarised coupling matrices are kept as their distinct blocks (`_BlockMCM`, `_RowPieces`),
and bandpower windows are built on demand.

## Precision

GMaster's defaults on an NVIDIA GPU use float32 where the result stays below NaMaster's own
HEALPix quadrature error: the latitudinal march, the ring FFTs, and the operands (not the
accumulation) of the polarised coupling quadrature. The end-to-end difference from NaMaster is
~1e-6 to 1e-4 of the spectrum depending on resolution (see `benchmarks/README.md`).

| knob | effect |
|---|---|
| `JAX_ENABLE_X64=1` | required; GMaster warns if it is off |
| `GMASTER_MARCH_V2=0` | exact float64 table routes instead of the float32 march |
| `set_ring_precision("fp64")` | float64 ring FFTs |
| `set_coupling_precision("fp64")` | float64 coupling-matrix operands |
| `set_table_precision("fp32")` | float32 storage of precomputed tables (more geometries fit) |

## Device memory

Geometry-only tables (ring chirp factors, march window tables, Wigner-d quadrature tables,
Legendre bands) are cached on the device between calls. `_config.make_room(nbytes)` evicts them
when a large allocation is coming: the cheapest caches first (registered in `_ROOM_HOOKS`), then
the ring tables, and only then the march tables (`_ROOM_HOOKS_LAST`). For Nside >= 4096 run with
`XLA_PYTHON_CLIENT_PREALLOCATE=false` so that the pool grows on demand; GMaster selects CUDA's
asynchronous allocator (`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`) unless you chose one.

## Environment variables

User-facing:

| variable | default | meaning |
|---|---|---|
| `GMASTER_MARCH_V2` | `1` | use the CUDA march (0: exact float64 routes) |
| `GMASTER_DC` | `1` | allow the divide-and-conquer engine in its band-limit window |
| `GMASTER_DC_MIN_L`, `GMASTER_DC_MAX_L` | `6144`, `12288` | that window |
| `GMASTER_DC_THREADS` | `8` | host threads for building a divide-and-conquer plan |
| `GMASTER_CUDA_CACHE` | `~/.cache/gmaster` | compiled kernels and divide-and-conquer plans |
| `GMASTER_NVCC` | on `PATH` | the CUDA compiler to use |
| `GMASTER_COUPLING_PRECISION` | `auto` | as `set_coupling_precision` |
| `GMASTER_CUFINUFFT_PATH` | unset | extra import path for `cufinufft` (`nusht`) |

The remaining `GMASTER_*` variables are tuning knobs of individual kernels (tile sizes, cache
budgets, route thresholds). Each is documented where it is read; `grep -rn GMASTER_ gmaster/`
lists them.

## Native code

`_native/cuda/march_v2.cu`, `_native/cuda/dc_lat.cu` and `_native/cuda/coupling_tt.cu` are XLA FFI
custom calls compiled with `nvcc` on first use; `_native/cpu/march_v2_cpu.cc` (the OpenMP march)
and `_native/cpu/dc_plan.cpp` (the divide-and-conquer plan builder) with `g++`. Shared libraries
are cached in `$GMASTER_CUDA_CACHE` under a hash of the source, the compiler flags and the JAX
version, so editing a source file triggers a rebuild on the next call.

## Testing

```bash
pip install -e ".[test]"          # pytest, pymaster (NaMaster) and ducc0 are the references
JAX_ENABLE_X64=1 pytest tests
```

Most tests compare against NaMaster or ducc0 on the same inputs. Tests that exercise the CUDA
march are marked `march_v2`; the others run the exact float64 routes (`tests/conftest.py`).
Leave `OMP_NUM_THREADS` unset when running the suite: some catalogue tests compare against
NaMaster's ducc0 NUFFT, whose result depends slightly on its thread count.

## Formal verification

`formal/lean/` holds Lean 4 (Mathlib) proofs of the identities the difference-form march relies on
(the Jacobi-polynomial form of the Wigner-d rows, the recurrence and its difference form, the
exponent bookkeeping of the emit step, the parity fold). `lake build` checks them; see
`formal/lean/README.md`. The derivations are written out in `docs/notes/`.
