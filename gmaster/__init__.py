"""GMaster: GPU pseudo-C_ell power-spectrum estimation with a NaMaster-compatible API.

GMaster is a JAX/CUDA implementation of the NaMaster (``pymaster``) pseudo-C_ell / MASTER
estimator (Alonso, Sanchez & Slosar 2019).  Its public API mirrors ``pymaster``, so an
existing pipeline usually runs on a GPU by changing a single import::

    import gmaster as nmt          # instead of: import pymaster as nmt

    f = nmt.NmtField(mask, [map_t])
    b = nmt.NmtBin.from_nside_linear(nside, 4)
    w = nmt.NmtWorkspace.from_fields(f, f, b)
    cl = w.decouple_cell(nmt.compute_coupled_cell(f, f))

Arrays are returned as JAX device arrays; ``numpy.asarray`` converts them.  Set the
environment variable ``JAX_ENABLE_X64=1`` before importing JAX: without 64-bit support
the results do not reach NaMaster-level accuracy (a warning is logged at import).

Main entry points
    Fields         `NmtField`, `NmtFieldFlat`, `NmtFieldCatalog`,
                   `NmtFieldCatalogClustering`, `NmtFieldCatalogMomentum`
    Binning        `NmtBin`, `NmtBinFlat`
    Spectra        `NmtWorkspace`, `NmtWorkspaceFlat`, `compute_coupled_cell`,
                   `compute_full_master`, `deprojection_bias`
    Covariances    `NmtCovarianceWorkspace`, `gaussian_covariance`
    Utilities      `mask_apodization`, `synfast_spherical`, `map2alm`, `alm2map`
    Settings       `set_n_iter_default`, `set_tol_pinv_default`, and the GPU-specific
                   `set_latitudinal_method`, `set_ring_precision`, `set_table_precision`,
                   `set_coupling_precision`
"""

import logging
import os

# Use CUDA's asynchronous (virtual-address-backed) allocator.  JAX's default BFC pool
# fragments over long pipelines at large Nside, and multi-GiB buffers such as the Nside 4096
# spin-2 coupling matrix can then fail to allocate although enough memory is free.  Only set
# when the user has not chosen an allocator; JAX reads it when the backend initialises, so
# importing gmaster before the first device use is sufficient.
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")

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
    latitudinal_method,
    set_latitudinal_method,
    set_n_iter_default,
    set_sht_calculator,
    set_ring_precision,
    set_table_precision,
    set_tol_pinv_default,
    ring_dtype,
    table_dtype,
    synfast_spherical,
    synfast_flat,
)
from .workspaces import (
    NmtWorkspace,
    compute_coupled_cell,
    compute_coupled_cell_flat,
    compute_full_master,
    coupling_precision,
    deprojection_bias,
    get_general_coupling_matrix,
    get_master_coefficients,
    set_coupling_precision,
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
    "coupling_precision",
    "deprojection_bias",
    "deprojection_bias_flat",
    "get_general_coupling_matrix",
    "get_master_coefficients",
    "gaussian_covariance",
    "gaussian_covariance_flat",
    "get_iNKA_cell",
    "latitudinal_method",
    "map2alm",
    "moore_penrose_pinvh",
    "nmt_params",
    "get_default_params",
    "mask_apodization",
    "mask_apodization_flat",
    "set_latitudinal_method",
    "set_coupling_precision",
    "set_n_iter_default",
    "set_sht_calculator",
    "set_ring_precision",
    "set_table_precision",
    "set_tol_pinv_default",
    "ring_dtype",
    "table_dtype",
    "synfast_spherical",
    "synfast_flat",
    "uncorr_noise_deprojection_bias",
    "utils",
]
__version__ = "1.0.0"

if not jax.config.read("jax_enable_x64"):
    logging.getLogger("gmaster").warning(
        "JAX 64-bit precision is disabled; set JAX_ENABLE_X64=1 for "
        "NaMaster-level numerical accuracy."
    )
