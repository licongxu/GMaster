"""Session-wide table precision and the comparison bar that belongs to it.

Every assertion in this suite compares a device array against a value that
pymaster/ducc0 computed in float64, against bars written as ``atol=2e-13`` or similar.
Those bars are right at the shipped fp64 table precision, where a table transform and
the recurrence that generated it agree to ~1e-16.

``set_table_precision("fp32")`` stores the precomputed Legendre/Wigner tables in
float32.  Their own representation error is ~1e-7, which is six orders of magnitude
above the fp64 bars and has nothing to do with the code under test, so the fp32 arm
needs a stated bar instead of the fp64 one.  ``--gm-precision=fp32`` therefore does two
things:

* pins the whole session to that precision, re-asserting it before every test so that a
  module's own restore-the-default fixture cannot hand fp64 back halfway through the run
  (a `-c`/`-p` bootstrap that imports before the test modules enable x64 gets this
  wrong in both directions);
* puts a floor under the ``atol`` of ``numpy.testing.assert_allclose`` for as long as
  fp32 is live.  The floor is the 2e-6 absolute bar this suite already uses for its
  loosest comparisons, applied only while fp32 is the live precision.

At the default fp64 nothing here is installed: ``numpy.testing.assert_allclose`` is the
real function and every bar is exactly what its test file says.
"""

import numpy as np
import pytest

# Absolute bar the fp32 table route is held to, measured against the float64 reference.
# It is the loosest atol already present in this suite, not a new, looser invention.
FP32_ATOL = 2e-6

_REAL_ASSERT_ALLCLOSE = np.testing.assert_allclose


def pytest_addoption(parser):
    parser.addoption(
        "--gm-precision",
        default="fp64",
        choices=("fp64", "fp32"),
        help="GMaster table precision for the whole session (default: the shipped fp64).",
    )


def _floored_assert_allclose(actual, desired, *args, **kwargs):
    from gmaster import nmt_params

    if nmt_params.table_dtype == "fp32":
        kwargs["atol"] = max(float(kwargs.get("atol", 0.0) or 0.0), FP32_ATOL)
    return _REAL_ASSERT_ALLCLOSE(actual, desired, *args, **kwargs)


def pytest_configure(config):
    if config.getoption("--gm-precision") == "fp64":
        return
    import jax

    jax.config.update("jax_enable_x64", True)
    import gmaster as nmt

    nmt.set_table_precision(config.getoption("--gm-precision"))
    np.testing.assert_allclose = _floored_assert_allclose
    print(f"SESSION table precision = {nmt.table_dtype().__name__} "
          f"atol floor = {FP32_ATOL:g}", flush=True)


@pytest.fixture(autouse=True)
def _session_table_precision(request):
    wanted = request.config.getoption("--gm-precision")
    if wanted == "fp64":
        yield
        return
    from gmaster import nmt_params, set_table_precision

    # Every test in this module is about the shipped default and the effect of leaving
    # it — that the default is fp64, that selecting fp32 halves the band, that switching
    # rebuilds rather than reuses.  A session pinned to fp32 has already left it, so the
    # module runs at fp64 regardless; its subject is the policy, not the arithmetic.
    if request.node.path.name == "test_table_precision.py":
        if nmt_params.table_dtype != "fp64":
            set_table_precision("fp64")
        yield
        set_table_precision(wanted)
        return
    if nmt_params.table_dtype != wanted:
        set_table_precision(wanted)
    yield
    if nmt_params.table_dtype != wanted:
        set_table_precision(wanted)
