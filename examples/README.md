# Examples

Command-line scripts. For a guided introduction start with the [tutorials](../tutorials/).

## `power_spectrum_from_fits.py`: spectra of your own maps

```bash
# temperature: 1-degree C1 apodization, bins of 20 multipoles, 5 arcmin Gaussian beam deconvolved
python examples/power_spectrum_from_fits.py cmb.fits mask.fits --apodize 1.0 --nlb 20 --fwhm 5 --plot tt.png

# temperature and polarization (fields 0, 1, 2 of the FITS file), cross-spectrum of two data splits
python examples/power_spectrum_from_fits.py split1.fits mask.fits --pol --cross split2.fits --apodize 2

# B-mode purification on a small patch, band limit 1500
python examples/power_spectrum_from_fits.py sky.fits patch.fits --pol --purify-b --apodize 5 --lmax 1500
```

The output `.npz` holds `ell_eff`, one array per spectrum (`TT`, `TE`, `TB`, `EE`, `EB`, `BE`,
`BB`) and the bandpower window functions, so a theory spectrum `cl_th` (from `ell = 0` to `lmax`)
is compared with the data as `np.einsum("ibjl,jl->ib", windows, cl_th_list)`.
`python examples/power_spectrum_from_fits.py --help` lists all options.

## `act_dr6/`: real data

Scripts that run on the public [ACT DR6](https://arxiv.org/abs/2503.14451) maps: the
comparison of GMaster and NaMaster on the same map and mask, and a noise-bias-free temperature
power spectrum from the cross-correlation of two data splits. See
[act_dr6/README.md](act_dr6/README.md).
