# Tutorials

Executed Jupyter notebooks. Each runs in a few minutes on a GPU (Nside 256-512) and is
self-contained; they are meant to be read in order.

| notebook | what it covers |
|---|---|
| [01_quickstart_temperature](01_quickstart_temperature.ipynb) | a masked CMB temperature map: mask apodization, `NmtField`, `NmtBin`, `NmtWorkspace`, bandpower windows, comparison with theory |
| [02_polarization_EB](02_polarization_EB.ipynb) | spin-2 fields; TT, TE, EE, BB; E-to-B leakage on a small patch and B-mode purification |
| [03_workspaces_simulations_covariance](03_workspaces_simulations_covariance.ipynb) | reusing a workspace for 200 simulations, saving it to disk, the Gaussian covariance against the simulations, a $\chi^2$ test |
| [04_beams_noise_deprojection](04_beams_noise_deprojection.ipynb) | beam deconvolution, noise-free cross-spectra of data splits, template deprojection and its bias |
| [05_namaster_crosscheck_performance](05_namaster_crosscheck_performance.ipynb) | the same estimator in GMaster and NaMaster, timings, settings for large maps |
| [colab_gmaster_vs_namaster](colab_gmaster_vs_namaster.ipynb) | the side-by-side comparison on a free Colab or Kaggle GPU, installing everything in the notebook |
| [kaggle_nside_sweep](kaggle_nside_sweep.ipynb) | the comparison over Nside 64-4096 on a Kaggle T4 (results: `benchmarks/kaggle_t4_nside_ladder.md`) |

Requirements: `pip install "gmaster[examples]"` (matplotlib, Jupyter, CAMB); tutorial 5 also
needs NaMaster (`pip install pymaster`). The notebooks set `JAX_ENABLE_X64=1` themselves. On a
machine without a GPU they run on the CPU, slowly.
