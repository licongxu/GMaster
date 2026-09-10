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
  fp32 is live.  The floor is 2e-6 *of the compared quantity* — ``max|desired|``, not a
  fixed absolute — because this suite asserts on O(1) ring sums and O(1e-12) decoupled
  ``Cl``s in the same run, and one absolute number can only be meaningless for one of
  them.

``--gm-ring-precision`` pins the azimuthal transforms separately.  It matters because
``set_table_precision`` used to move them too: the analysis chirp-Z casts the map pixels
to the chirp's own real dtype, so a float32 table session analyzed the map in float32 and
failed every direct-DFT ring parity in ``test_sht.py``.  Held to ``fp64`` the fp32 table
route passes the whole suite.

At the default fp64 nothing here is installed: ``numpy.testing.assert_allclose`` is the
real function and every bar is exactly what its test file says.
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
        # A fixed absolute bar is meaningless across this suite's scales: transform
        # parities compare O(1) ring sums while a decoupled Cl is O(1e-12), and 2e-6
        # absolute would be vacuous for the latter.  The floor is 2e-6 *of the compared
        # quantity*, so it says the same thing everywhere: agree to two parts per million.
        #
        # The ring precision joins the condition because a float32 azimuthal stage is the
        # same kind of licence.  The generic s2fft path does not read it at all -- its alms
        # and its gradients are bit-identical between `set_ring_precision("fp64")` and
        # `"fp32"` (`.qwen/tmp/ad_probe2_s35.log`) -- so an fp32-ring parity test compares a
        # float32-ringed GMaster against a float64 reference and can only agree to the
        # ring's own rounding: measured 3.5e-07 of scale on the analysis gradient and
        # 1.7e-07 on the synthesis, against 2.5e-13 with float64 rings.  The fused
        # float32-ring gradient is as close to the exact one as its own forward transform
        # is, which is what the floor says.
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
def _session_table_precision(request):
    wanted = request.config.getoption("--gm-precision")
    wanted_ring = request.config.getoption("--gm-ring-precision")
    if wanted == "fp64" and wanted_ring == "follow":
        yield
        return
    from gmaster import nmt_params, set_ring_precision, set_table_precision

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
    if nmt_params.ring_precision != wanted_ring:
        set_ring_precision(wanted_ring)
    yield
    if nmt_params.table_dtype != wanted:
        set_table_precision(wanted)
    if nmt_params.ring_precision != wanted_ring:
        set_ring_precision(wanted_ring)
