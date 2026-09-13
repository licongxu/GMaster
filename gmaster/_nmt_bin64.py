"""64-bit binning of NaMaster's unbinned mode-coupling matrix.

NaMaster's C ``nmt_bin_mcm`` / ``nmt_bin_mcm_oneside`` index the flattened
MCM with ``int``.  At spin-2, ``ncls=4``, ``nls=lmax+1=12288`` the linear
index is ``ncls^2 nls^2 = 2.416e9``, past ``INT_MAX`` (2^31-1).  SWIG also
passes the flattened length as ``int DIM1``, so the assertion
``nmcm_in == ncl*ncl*nls*nls`` both-overflows and passes.  The process then
segfaults inside the C extension (HANDOFF addendum 25).

This module reimplements those two kernels with 64-bit NumPy indexing so
pymaster can finish Nside=4096 spin 2 without modifying the NaMaster tree.
The 3j coefficient build (``get_xis``) is unchanged.
"""

import numpy as np

_INT32_MAX = 2 ** 31 - 1
_PATCHED = False
_ORIG = None
_LOGGED = False


def mcm_needs_i64(nls, ncls):
    """True when the flattened MCM length does not fit in a signed 32-bit int."""
    return ncls * ncls * nls * nls > _INT32_MAX


def _band_weights(bins):
    nb = bins.get_n_bands()
    ells = [np.asarray(bins.get_ell_list(b), dtype=np.int64) for b in range(nb)]
    wgt = [np.asarray(bins.get_weight_list(b), dtype=np.float64) for b in range(nb)]
    fell = [np.asarray(bins.get_fell_list(b), dtype=np.float64) for b in range(nb)]
    return ells, wgt, fell


def bin_mcm_i64(mcm_in, bins, ncls, beam1, beam2, oneside, norm_type=0, w2=0.0):
    """Match ``nmt_bin_mcm`` / ``nmt_bin_mcm_oneside`` with 64-bit indices."""
    nls = int(bins.lmax) + 1
    mcm4 = np.asarray(mcm_in, dtype=np.float64).reshape(nls, ncls, nls, ncls)
    beam1 = np.asarray(beam1, dtype=np.float64)
    beam2 = np.asarray(beam2, dtype=np.float64)
    ells, wgt, fell = _band_weights(bins)
    nb = len(ells)

    if int(norm_type) != 0 and not oneside:
        out = np.zeros((nb * ncls, nb * ncls), dtype=np.float64)
        diag = nb * ncls
        out[np.arange(diag), np.arange(diag)] = float(w2)
        return out

    left = np.zeros((nb, nls), dtype=np.float64)
    for ib, (e, w, f) in enumerate(zip(ells, wgt, fell)):
        left[ib, e] = w * f
    # (nb, ncls, nls, ncls): contract the binned ell against C layout (l, icl, l', icl')
    slab = np.tensordot(left, mcm4, axes=(1, 0))
    slab = slab * (beam1 * beam2)[None, None, :, None]
    if oneside:
        return slab.reshape(nb * ncls, nls * ncls)

    right = np.zeros((nls, nb), dtype=np.float64)
    for ib, (e, f) in enumerate(zip(ells, fell)):
        right[e, ib] = 1.0 / f
    # (nb, ncls, ncls, nb) then swap the last two axes to (ib, icl, ib', icl')
    out = np.tensordot(slab, right, axes=(2, 0))
    return np.moveaxis(out, 3, 2).reshape(nb * ncls, nb * ncls)


def _patched_bin_mcm(self, mcm_in, norm_type, w2, beam1, beam2, oneside):
    nls = self.lmax + 1
    ncls = len(mcm_in) // nls
    if mcm_needs_i64(nls, ncls):
        global _LOGGED
        if not _LOGGED:
            print(
                f"nmt_bin64: numpy i64 binning nls={nls} ncls={ncls} "
                f"flat={ncls * ncls * nls * nls}",
                flush=True,
            )
            _LOGGED = True
        return bin_mcm_i64(
            mcm_in, self, ncls, beam1, beam2, oneside,
            norm_type=norm_type, w2=w2,
        )
    return _ORIG(self, mcm_in, norm_type, w2, beam1, beam2, oneside)


def patch_pymaster():
    """Install the 64-bit binning fallback on ``pymaster.NmtBin._bin_mcm``."""
    global _PATCHED, _ORIG
    if _PATCHED:
        return
    import pymaster as reference
    _ORIG = reference.NmtBin._bin_mcm
    reference.NmtBin._bin_mcm = _patched_bin_mcm
    _PATCHED = True
