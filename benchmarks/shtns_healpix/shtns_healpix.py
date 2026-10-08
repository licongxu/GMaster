"""SHTns on the HEALPix grid, on the GPU, with SHTns's own Legendre recurrence.

SHTns only knows grids whose rings all have the same number of longitudes.  This
module keeps SHTns for the latitudinal (Legendre) step and does the HEALPix
longitude step itself:

* the SHTns GPU plan is built on ``4 nside`` latitudes (the HEALPix rings, with the
  equator ring listed twice at half weight so the grid is even and symmetric), and
  its colatitudes and weights are replaced by the HEALPix ones through
  ``cushtns_set_latitudes`` (a local patch of ``sht_gpu.cu``);
* ``cu_SH_to_fourier_float`` / ``cu_fourier_to_SH_float`` (same patch) run SHTns's
  Legendre kernels only, on ``F_m(theta)`` in the phi-contiguous layout
  ``[m][theta]`` complex64;
* ring FFTs: one batched FFT for the ``2 nside + 1`` equatorial-belt rings (all
  ``4 nside`` long) and one batched Bluestein (chirp-z) FFT for the ``4 nside - 4``
  polar-cap rings of different lengths.  Modes ``m >= n_ring`` alias onto the ring,
  and each ring's first-pixel phase is applied, as in HEALPix/ducc0.

The recurrence precision is SHTns's choice unless ``SHTNS_GPU_REC_PREC`` is set
before the plan is built: ``1`` forces SHTns's float32 recurrence (its standard
three-term form; SHTns disables its Ishioka variant in that mode), ``2`` forces
float64.  Data, FFTs and sums are float32 in both cases.

Requires the patched build in /scratch/scratch-lxu/shtns_build/shtns-git and
``SHTNS_GPU_NO_FFT=1`` (set here) so the plan does not need a cuFFT layout that
cuFFT does not support.
"""

import ctypes
import os
import sys

import numpy as np

SHTNS_DIR = "/scratch/scratch-lxu/shtns_build/shtns-git"


def _import_shtns():
    os.environ["SHTNS_GPU_NO_FFT"] = "1"
    if SHTNS_DIR not in sys.path:
        sys.path.insert(0, SHTNS_DIR)
    import shtns
    return shtns


def healpix_rings(nside):
    """Per-ring (cos theta, npix, first pixel, phi0) in RING order, north to south."""
    nside = int(nside)
    nring = 4 * nside - 1
    i = np.arange(1, nring + 1)
    npix = np.where(i < nside, 4 * i, np.where(i > 3 * nside, 4 * (4 * nside - i), 4 * nside))
    start = np.concatenate([[0], np.cumsum(npix)[:-1]])
    z = np.empty(nring)
    north = i < nside
    south = i > 3 * nside
    belt = ~(north | south)
    z[north] = 1 - i[north] ** 2 / (3.0 * nside**2)
    z[south] = -(1 - (4 * nside - i[south]) ** 2 / (3.0 * nside**2))
    z[belt] = 4.0 / 3 - 2 * i[belt] / (3.0 * nside)
    phi0 = np.empty(nring)
    cap = north | south
    phi0[cap] = np.pi / npix[cap]                          # pi / (4 i)
    shifted = ((i - nside) % 2) == 0
    phi0[belt] = np.where(shifted[belt], np.pi / (4 * nside), 0.0)
    return z, npix, start, phi0


class HealpixSHT:
    """Spin-0 ``alm2map`` / ``map2alm`` on HEALPix with SHTns's GPU Legendre step.

    ``alm`` is complex64 in healpy order (m-major, ``l = m..lmax``), which is also
    SHTns's order.  Maps are float32 in HEALPix RING order.  All arrays are cupy.
    """

    def __init__(self, nside, lmax=None):
        import cupy as cp
        shtns = _import_shtns()
        self.cp = cp
        self.nside = nside = int(nside)
        self.lmax = lmax = 3 * nside - 1 if lmax is None else int(lmax)
        self.mmax = lmax
        self.npix = 12 * nside * nside
        self.nlat = 4 * nside                         # equator listed twice
        self.nphi_plan = 2 * lmax + 2
        sh = shtns.sht(lmax, lmax, 1, shtns.sht_orthonormal)
        flags = (shtns.sht_gauss | shtns.SHT_PHI_CONTIGUOUS | shtns.SHT_ALLOW_GPU
                 | shtns.SHT_FP32)
        sh.set_grid(self.nlat, self.nphi_plan, flags=flags)
        self.sh = sh
        self.cfg = int(sh.this)
        self.nlm = sh.nlm
        mod = sys.modules.get("_shtns_cuda") or sys.modules["_shtns"]
        lib = ctypes.CDLL(mod.__file__)
        for name in ("cu_SH_to_fourier_float", "cu_fourier_to_SH_float"):
            getattr(lib, name).argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int]
        lib.cushtns_set_latitudes.argtypes = [ctypes.c_void_p] * 3
        lib.cushtns_set_latitudes.restype = ctypes.c_int
        self.lib = lib

        z, npix, start, phi0 = healpix_rings(nside)
        eq = 2 * nside - 1                            # equator ring index
        lat_ring = np.concatenate([np.arange(2 * nside), np.arange(eq, 4 * nside - 1)])
        ct = np.ascontiguousarray(z[lat_ring])
        ct[eq] = ct[eq + 1] = 0.0
        w = 2.0 * npix[lat_ring] / self.npix
        w[eq] *= 0.5
        w[eq + 1] *= 0.5
        ct = np.ascontiguousarray(ct, np.float64)
        w = np.ascontiguousarray(w, np.float64)
        err = lib.cushtns_set_latitudes(self.cfg, ct.ctypes.data, w.ctypes.data)
        if err != 0:
            raise RuntimeError("cushtns_set_latitudes refused the HEALPix rings")
        self._setup_rings(z, npix, start, phi0, lat_ring)
        self._F = cp.zeros((self.mmax + 1) * self.nlat, np.complex64)
        self._alm = cp.zeros(self.nlm, np.complex64)

    # ------------------------------------------------------------------ rings
    def _setup_rings(self, z, npix, start, phi0, lat_ring):
        cp = self.cp
        nside, mmax = self.nside, self.mmax
        nring = 4 * nside - 1
        belt = np.arange(nside - 1, 3 * nside)        # rings with 4 nside pixels
        cap = np.concatenate([np.arange(nside - 1), np.arange(3 * nside, nring)])
        self.belt_start = int(start[belt[0]])
        self.belt_len = int(4 * nside * len(belt))
        self.nbelt = len(belt)
        self.ncap = len(cap)
        self.lat_ring = cp.asarray(lat_ring.astype(np.int32))
        self.ring_phi0 = cp.asarray(phi0)
        self.ring_npix = cp.asarray(npix.astype(np.int32))
        # row of each ring in the stacked spectrum table D[ring, k]: belt rows first
        n_cap_max = int(npix[cap].max()) if len(cap) else 0
        self.kmax = max(4 * nside, n_cap_max)
        # Bluestein tables for the polar caps
        M = 1
        while M < 2 * n_cap_max - 1:
            M *= 2
        self.M = M
        if len(cap):
            ncap = len(cap)
            n_c = npix[cap].astype(np.int64)
            t = np.arange(M, dtype=np.int64)
            tt = np.minimum(t, M - t)                         # |t| for wrapped kernel
            # chirp c_t = exp(i pi t^2 / n) with t^2 reduced mod 2n in integers
            ph_b = np.empty((ncap, M))
            ph_a = np.empty((ncap, M))
            valid_b = np.empty((ncap, M), bool)
            for r in range(ncap):
                n = n_c[r]
                ph_b[r] = np.pi * ((tt * tt) % (2 * n)) / n
                valid_b[r] = tt < n
                ph_a[r] = np.pi * ((t * t) % (2 * n)) / n
            b = np.where(valid_b, np.exp(1j * ph_b), 0)
            a = np.where(t[None, :] < n_c[:, None], np.exp(-1j * ph_a), 0)
            self.cap_chirp = cp.asarray(a.astype(np.complex64))        # conj(c_j), j < n
            self.cap_kernel_f = cp.fft.fft(cp.asarray(b.astype(np.complex64)), axis=1)
            del a, b, ph_a, ph_b, valid_b
            # gather / scatter indices of cap pixels into the (ncap, M) padded buffer
            rows = np.repeat(np.arange(ncap), n_c)
            cols = np.concatenate([np.arange(n) for n in n_c])
            pix = np.concatenate([start[r] + np.arange(npix[r]) for r in cap])
            self.cap_pix = cp.asarray(pix.astype(np.int64))
            self.cap_flat = cp.asarray((rows * M + cols).astype(np.int64))
        # stacked spectrum table: rows = rings in order [belt..., cap...]
        order = np.concatenate([belt, cap])
        row_of_ring = np.empty(nring, np.int32)
        row_of_ring[order] = np.arange(nring)
        self.row_of_ring = cp.asarray(row_of_ring)
        self.cap_ring_index = cp.asarray(cap.astype(np.int32))

        self._expand = cp.RawKernel(r"""
        #include <cuComplex.h>
        extern "C" __global__ void expand(const float2* D, const int ld, const int* row_of_ring,
            const int* lat_ring, const int* npix, const double* phi0, float2* F,
            const int nlat, const int mmax, const float scale_num) {
            // F[m*nlat + it] = (nphi_plan / n) * D[row, m mod n] * exp(-i m phi0)
            int it = blockIdx.x * blockDim.x + threadIdx.x;
            int m = blockIdx.y;
            if (it >= nlat) return;
            int r = lat_ring[it];
            int n = npix[r];
            float2 d = D[(long)row_of_ring[r] * ld + (m % n)];
            double s, c;
            sincos(-(double)m * phi0[r], &s, &c);
            float sc = scale_num / n;
            float2 out;
            out.x = sc * (d.x * (float)c - d.y * (float)s);
            out.y = sc * (d.x * (float)s + d.y * (float)c);
            F[(long)m * nlat + it] = out;
        }
        """, "expand")
        self._fold = cp.RawKernel(r"""
        extern "C" __global__ void fold(const float2* F, const int nlat, const int mmax,
            const int* row_of_ring, const int* ring_to_lat, const int* npix, const double* phi0,
            float2* H, const int ld, const int nring) {
            // H[row, k] = sum_{m = k mod n, m <= mmax} (m ? 2 : 1) F[m, lat] exp(+i m phi0)
            int k = blockIdx.x * blockDim.x + threadIdx.x;
            int r = blockIdx.y;
            int n = npix[r];
            if (k >= n) return;
            int it = ring_to_lat[r];
            double ph = phi0[r];
            float hx = 0.f, hy = 0.f;
            for (int m = k; m <= mmax; m += n) {
                float2 f = F[(long)m * nlat + it];
                double s, c;
                sincos((double)m * ph, &s, &c);
                float w = m ? 2.f : 1.f;
                hx += w * (f.x * (float)c - f.y * (float)s);
                hy += w * (f.x * (float)s + f.y * (float)c);
            }
            H[(long)row_of_ring[r] * ld + k] = make_float2(hx, hy);
        }
        """, "fold")
        ring_to_lat = np.arange(nring, dtype=np.int32)           # north copy of each ring
        ring_to_lat[2 * nside:] = np.arange(2 * nside, nring) + 1
        ring_to_lat[2 * nside - 1] = 2 * nside - 1
        self.ring_to_lat = cp.asarray(ring_to_lat)
        self.nring = nring

    def _cap_dft(self, x_pad):
        """Forward DFT (exp(-2 pi i jk/n)) of every cap row; x_pad is (ncap, M) complex64."""
        cp = self.cp
        y = cp.fft.ifft(cp.fft.fft(x_pad * self.cap_chirp, axis=1) * self.cap_kernel_f, axis=1)
        return y * self.cap_chirp              # conj(c_k) for k < n; rows beyond n unused

    # --------------------------------------------------------------- transforms
    def map2fourier(self, m):
        """Ring Fourier sums for every SHTns latitude, scaled for SHTns analysis."""
        cp = self.cp
        nside, ld = self.nside, self.kmax
        D = cp.empty((self.nring, ld), np.complex64)
        belt = m[self.belt_start:self.belt_start + self.belt_len].reshape(self.nbelt, 4 * nside)
        D[:self.nbelt, :4 * nside] = cp.fft.fft(belt.astype(np.complex64), axis=1)
        if self.ncap:
            buf = cp.zeros(self.ncap * self.M, np.complex64)
            buf[self.cap_flat] = m[self.cap_pix]
            Y = self._cap_dft(buf.reshape(self.ncap, self.M))
            D[self.nbelt:, :] = Y[:, :ld]
        thr = 128
        self._expand(((self.nlat + thr - 1) // thr, self.mmax + 1), (thr,),
                     (D, np.int32(ld), self.row_of_ring, self.lat_ring, self.ring_npix,
                      self.ring_phi0, self._F, np.int32(self.nlat), np.int32(self.mmax),
                      np.float32(self.nphi_plan)))
        return self._F

    def fourier2map(self, F):
        cp = self.cp
        nside, ld = self.nside, self.kmax
        H = cp.zeros((self.nring, ld), np.complex64)
        thr = 128
        self._fold(((ld + thr - 1) // thr, self.nring), (thr,),
                   (F, np.int32(self.nlat), np.int32(self.mmax), self.row_of_ring,
                    self.ring_to_lat, self.ring_npix, self.ring_phi0, H, np.int32(ld),
                    np.int32(self.nring)))
        out = cp.empty(self.npix, np.float32)
        # f_j = Re sum_k H_k exp(+2 pi i jk/n) = Re DFT_forward(conj H)_j
        hb = H[:self.nbelt, :4 * nside]
        out[self.belt_start:self.belt_start + self.belt_len] = (
            cp.fft.ifft(hb, axis=1).real * (4 * nside)).ravel()
        if self.ncap:
            buf = cp.zeros((self.ncap, self.M), np.complex64)
            buf[:, :ld] = cp.conj(H[self.nbelt:, :])
            # zero entries k >= n of each cap row
            k = cp.arange(self.M, dtype=np.int32)[None, :]
            buf *= (k < self.ring_npix[self.cap_ring_index][:, None])
            Y = self._cap_dft(buf)
            out[self.cap_pix] = Y.ravel()[self.cap_flat].real
        return out

    def alm2map(self, alm):
        alm = self.cp.ascontiguousarray(alm, np.complex64)
        self.lib.cu_SH_to_fourier_float(self.cfg, alm.data.ptr, self._F.data.ptr, self.lmax)
        return self.fourier2map(self._F)

    def map2alm(self, m, n_iter=0):
        cp = self.cp
        m = cp.ascontiguousarray(m, np.float32)
        alm = self._analyse(m)
        for _ in range(n_iter):
            alm = alm + self._analyse(m - self.alm2map(alm))
        return alm

    def _analyse(self, m):
        F = self.map2fourier(m)
        out = self.cp.empty(self.nlm, np.complex64)
        self.lib.cu_fourier_to_SH_float(self.cfg, F.data.ptr, out.data.ptr, self.lmax)
        return out
