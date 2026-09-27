# ACT DR6 examples

Two analyses of the Atacama Cosmology Telescope DR6 maps (Naess et al. 2025,
[arXiv:2503.14451](https://arxiv.org/abs/2503.14451)), PA4 f150 night-time, source-free HEALPix
maps at native Nside 8192.

## 1. Get the data and cache it

Download the DR6.02 standard maps from NASA's LAMBDA archive (the files named below), then cache
them as numpy arrays so that the scripts time the estimator and not the FITS reading:

```bash
python examples/act_dr6/prepare.py --data-dir /path/to/act_dr6.02_maps_standard --out act_cache
# for the split cross-spectrum, also the splits, the inverse-variance weight (needs pixell) and the beam:
python examples/act_dr6/prepare.py --data-dir /path/to/act_dr6.02_maps_standard --out act_cache \
    --nsides 4096 --splits 0,1 --beam /path/to/coadd_pa4_f150_night_beam_tform_instant.txt
```

Files read from `--data-dir`:

* `act_dr6.02_std_AA_night_pa4_f150_4way_coadd_map_srcfree_healpix.fits`
* `act_dr6.02_std_AA_night_pa4_f150_4way_set{0,1}_map_srcfree_healpix.fits` (with `--splits`)
* `act_dr6.02_std_AA_night_pa4_f150_4way_coadd_ivar.fits` (with `--splits`)

## 2. GMaster against NaMaster on the same map

```bash
python examples/act_dr6/gmaster_vs_namaster.py --cache act_cache --nside 4096 --out act_dr6_overlay
```

Runs the full MASTER estimator (TT, footprint mask, `n_iter=3`, bins of 50) in both codes on the
same arrays, times the estimator calls only, and writes both spectra, their ratio and a figure.
`--skip-namaster` runs GMaster only; `--exact` uses GMaster's all-float64 route.

## 3. A noise-free TT spectrum from two splits

```bash
python examples/act_dr6/split_cross_spectrum.py --cache act_cache --nside 4096 --out act_dr6_cross
```

The cross-spectrum of two data splits has no noise bias, so the acoustic peaks appear directly.
The inverse-variance map weights the footprint and the nominal beam is deconvolved through the
mode-coupling matrix; the auto-spectrum of one split is shown for the noise level.

The same cached maps feed the end-to-end benchmark in `benchmarks/scaling/bench_act_warm.py`.
