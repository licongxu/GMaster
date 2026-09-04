"""GPU-accelerated pseudo-C_ell estimation with a NaMaster-compatible API."""

import logging

import jax

from . import utils
from .bins import NmtBin, NmtBinFlat
from .field import NmtField
from .field_flat import NmtFieldFlat
from .field_catalog import (
    NmtFieldCatalog,
    NmtFieldCatalogClustering,
    NmtFieldCatalogMomentum,
)
from .covariance import (
    NmtCovarianceWorkspace,
    NmtCovarianceWorkspaceFlat,
    gaussian_covariance,
    gaussian_covariance_flat,
    get_iNKA_cell,
)
from .utils import (
    NmtAlmInfo,
    NmtMapInfo,
    alm2map,
    get_default_params,
    mask_apodization,
    mask_apodization_flat,
    map2alm,
    moore_penrose_pinvh,
    nmt_params,
    set_n_iter_default,
    set_sht_calculator,
    set_table_precision,
    set_tol_pinv_default,
    table_dtype,
    synfast_spherical,
    synfast_flat,
)
from .workspaces import (
    NmtWorkspace,
    compute_coupled_cell,
    compute_coupled_cell_flat,
    compute_full_master,
    deprojection_bias,
    get_general_coupling_matrix,
    get_master_coefficients,
    uncorr_noise_deprojection_bias,
)
from .workspaces_flat import (
    NmtWorkspaceFlat,
    compute_full_master_flat,
    deprojection_bias_flat,
)

__all__ = [
    "NmtAlmInfo",
    "NmtBin",
    "NmtBinFlat",
    "NmtField",
    "NmtFieldFlat",
    "NmtFieldCatalog",
    "NmtFieldCatalogClustering",
    "NmtFieldCatalogMomentum",
    "NmtCovarianceWorkspace",
    "NmtCovarianceWorkspaceFlat",
    "NmtMapInfo",
    "NmtWorkspace",
    "NmtWorkspaceFlat",
    "alm2map",
    "compute_coupled_cell",
    "compute_coupled_cell_flat",
    "compute_full_master",
    "compute_full_master_flat",
    "deprojection_bias",
    "deprojection_bias_flat",
    "get_general_coupling_matrix",
    "get_master_coefficients",
    "gaussian_covariance",
    "gaussian_covariance_flat",
    "get_iNKA_cell",
    "map2alm",
    "moore_penrose_pinvh",
    "nmt_params",
    "get_default_params",
    "mask_apodization",
    "mask_apodization_flat",
    "set_n_iter_default",
    "set_sht_calculator",
    "set_table_precision",
    "set_tol_pinv_default",
    "table_dtype",
    "synfast_spherical",
    "synfast_flat",
    "uncorr_noise_deprojection_bias",
    "utils",
]
__version__ = "0.1.0"

if not jax.config.read("jax_enable_x64"):
    logging.getLogger("gmaster").warning(
        "JAX 64-bit precision is disabled; set JAX_ENABLE_X64=1 for "
        "NaMaster-level numerical accuracy."
    )
