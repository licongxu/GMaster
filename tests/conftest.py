"""Session-wide precision options and the comparison tolerance that goes with them.

Most assertions in this suite compare a GMaster array against a value that
pymaster/ducc0 computed in float64, with tolerances such as ``atol=2e-13``.  Those
tolerances assume the default fp64 Legendre/Wigner tables, which agree with the
recurrence that generated them to ~1e-16.

``set_table_precision("fp32")`` stores those tables in float32, whose own
representation error is ~1e-7 -- six orders of magnitude above the fp64
tolerances, and unrelated to the code under test.  ``--gm-precision=fp32``
therefore does two things:

* pins the whole session to that precision, re-applying it before every test so
  that a module fixture which restores the default cannot switch back to fp64
  partway through the run;
* puts a floor under the ``atol`` of ``numpy.testing.assert_allclose`` while fp32
  is active.  The floor is 2e-6 *of the compared quantity* (``max|desired|``),
  not a fixed absolute value, because the suite compares O(1) ring sums and
  O(1e-12) decoupled ``Cl``s in the same run.

``--gm-ring-precision`` sets the precision of the azimuthal (ring FFT) transforms
separately.  The analysis chirp-Z casts the map pixels to the chirp's real dtype,
so float32 rings analyse the map in float32 and cannot pass the direct-DFT ring
checks in ``test_sht.py`` at fp64 tolerances; with rings held at ``fp64`` the fp32
table route passes the whole suite.

At the default fp64 none of this is installed: ``numpy.testing.assert_allclose``
is the real function and every tolerance is exactly what its test file says.
"""

import numpy as np
import pytest

# Relative bar the fp32 table route is held to, scaled by max|desired| at each call site.
# 2e-6 is the loosest atol already written into this suite and about the size of the
# float32 table's own representation error.
FP32_ATOL = 2e-6

_REAL_ASSERT_ALLCLOSE = np.testing.assert_allclose


def pytest_addoption(parser):
    parser.addoption(
        "--gm-precision",
        default="fp64",
        choices=("fp64", "fp32"),
        help="GMaster table precision for the whole session (default: the shipped fp64).",
    )
    parser.addoption(
        "--gm-ring-precision",
        default="follow",
        choices=("follow", "fp64", "fp32"),
        help="GMaster azimuthal transform precision; 'follow' tracks the table "
        "precision, which is the shipped behaviour.",
    )


def _floored_assert_allclose(actual, desired, *args, **kwargs):
    from gmaster import nmt_params

    if nmt_params.table_dtype == "fp32" or nmt_params.ring_precision == "fp32":
        # A fixed absolute tolerance is meaningless across this suite's scales: transform
        # checks compare O(1) ring sums while a decoupled Cl is O(1e-12), and 2e-6
        # absolute would be vacuous for the latter.  The floor is 2e-6 *of the compared
        # quantity*, so it means the same everywhere: agreement to two parts per million.
        #
        # fp32 rings also trigger the floor.  The generic s2fft path ignores the ring
        # precision, so an fp32-ring check compares a float32-ring GMaster result against
        # a float64 reference and can only agree to the ring's own rounding: about 3.5e-7
        # of scale on the analysis gradient and 1.7e-7 on the synthesis (versus 2.5e-13
        # with float64 rings), well inside the 2e-6 floor.
        scale = float(np.max(np.abs(np.asarray(desired))))
        kwargs["atol"] = max(float(kwargs.get("atol", 0.0) or 0.0), FP32_ATOL * scale)
    return _REAL_ASSERT_ALLCLOSE(actual, desired, *args, **kwargs)


def pytest_configure(config):
    precision = config.getoption("--gm-precision")
    ring = config.getoption("--gm-ring-precision")
    if precision == "fp64" and ring == "follow":
        return
    import jax

    jax.config.update("jax_enable_x64", True)
    import gmaster as nmt

    nmt.set_table_precision(precision)
    nmt.set_ring_precision(ring)
    np.testing.assert_allclose = _floored_assert_allclose
    print(f"SESSION table precision = {nmt.table_dtype().__name__} "
          f"ring precision = {nmt.ring_dtype().__name__} "
          f"atol floor = {FP32_ATOL:g}", flush=True)


@pytest.fixture(autouse=True)
def _exact_sht_unless_v2(request, monkeypatch):
    """Run unmarked tests on the exact fp64 transform routes.

    The default transform is the float32 v2 CUDA march (~1e-6 relative agreement with NaMaster
    on isolated transforms; decoupled bandpowers on real maps follow benchmarks/README.md),
    while most tests hold GMaster to NaMaster at ~1e-13, which only the exact fp64 routes
    reach.  Tests of the default route carry `@pytest.mark.march_v2`; every other test
    runs with `GMASTER_MARCH_V2=0`.
    """
    if request.node.get_closest_marker("march_v2") is None:
        monkeypatch.setenv("GMASTER_MARCH_V2", "0")


@pytest.fixture(autouse=True)
def _session_table_precision(request):
    """Re-apply the session's ``--gm-precision``/``--gm-ring-precision`` around every test."""
    wanted = request.config.getoption("--gm-precision")
    wanted_ring = request.config.getoption("--gm-ring-precision")
    if wanted == "fp64" and wanted_ring == "follow":
        yield
        return
    from gmaster import nmt_params, set_ring_precision, set_table_precision

    # test_table_precision.py tests the default precision and what happens when it is
    # changed (the default is fp64, fp32 halves the table size, switching rebuilds the
    # caches).  Those tests must start from fp64, so that module always runs at fp64
    # even when the session is pinned to fp32.
    if request.node.path.name == "test_table_precision.py":
        if nmt_params.table_dtype != "fp64":
            set_table_precision("fp64")
        yield
        set_table_precision(wanted)
        return
    if nmt_params.table_dtype != wanted:
        set_table_precision(wanted)
    if nmt_params.ring_precision != wanted_ring:
        set_ring_precision(wanted_ring)
    yield
    if nmt_params.table_dtype != wanted:
        set_table_precision(wanted)
    if nmt_params.ring_precision != wanted_ring:
        set_ring_precision(wanted_ring)
