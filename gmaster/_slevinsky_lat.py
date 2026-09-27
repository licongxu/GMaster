"""Scalar latitudinal SHT via Slevinsky's backward-stable order reduction.

Normalized associated Legendre functions of order ``m`` are reduced to order
``0`` or ``1`` by the Givens product in Slevinsky, *Fast and backward stable
transforms between spherical harmonic expansions and bivariate Fourier series*
(Appl. Comput. Harmon. Anal. 47, 2019, eq. 11). Sines and cosines are streamed
from that closed form; nothing is stored over ``(ell, m, theta)``.

The public spin-0 contract (checked against ``_march_v2`` and a direct
``lpmv`` sum) is

    synthesis ring = exp(i m phase) * alm[ell] * sqrt((2ell+1)/4pi) * d_ell^m
    analysis alm   = sqrt((2ell+1)/4pi) * sum_i pos[i] w[i] exp(i m phase) d_ell^m

with ``d_ell^m = sqrt((ell-m)!/(ell+m)!) P_ell^m`` (Condon–Shortley). That is
``\\tilde P / sqrt(2 pi)`` times the coefficient, which is why every conversion
below divides by ``sqrt(2 pi)``.
"""
from __future__ import annotations

import math

import numpy as np

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _sc(l, m):
    den = (l + 2 * m + 3) * (l + 2 * m + 4)
    s = math.sqrt((l + 1) * (l + 2) / den)
    c = math.sqrt((2 * m + 2) * (2 * l + 2 * m + 5) / den)
    return s, c


def _hi2lo(A, m1, m2, n):
    """In-place copy of FastTransforms ``kernel_sph_hi2lo`` on one real vector."""
    A = np.array(A, dtype=np.float64, copy=True)
    for j in range(m2 - 2, m1 - 1, -2):
        for k in range(n - 3 - j, -1, -1):
            S, C = _sc(k, j)
            x = A[k]
            y = A[k + 2]
            A[k] = C * x + S * y
            A[k + 2] = C * y - S * x
    return A


def _lo2hi(A, m1, m2, n):
    """Transpose of ``_hi2lo`` (FastTransforms ``kernel_sph_lo2hi``)."""
    A = np.array(A, dtype=np.float64, copy=True)
    for j in range(m1, m2, 2):
        for k in range(0, n - 2 - j):
            S, C = _sc(k, j)
            x = A[k]
            y = A[k + 2]
            A[k] = C * x - S * y
            A[k + 2] = S * x + C * y
    return A


def _basis(x, L):
    """Rows ``(k, theta)`` of ``\\tilde P_k^0`` and of ``\\tilde P_{k+1}^1``."""
    ntheta = x.size
    P = np.empty((L, ntheta), dtype=np.float64)
    P[0] = 1.0
    if L > 1:
        P[1] = x
    for n in range(2, L):
        P[n] = ((2 * n - 1) * x * P[n - 1] - (n - 1) * P[n - 2]) / n
    scale0 = np.sqrt(np.arange(L, dtype=np.float64) + 0.5)
    T0 = scale0[:, None] * P

    A1 = np.zeros((L, ntheta), dtype=np.float64)
    if L > 1:
        # P_1^1 = -sqrt(1-x^2); (l-1) P_l^1 = x(2l-1) P_{l-1}^1 - l P_{l-2}^1
        A1[1] = -np.sqrt(np.maximum(1.0 - x * x, 0.0))
        for l in range(2, L):
            A1[l] = (x * (2 * l - 1) * A1[l - 1] - l * A1[l - 2]) / (l - 1)
    # index k holds \\tilde P_{k+1}^1, k = 0..L-2
    T1 = np.zeros_like(T0)
    if L > 1:
        ell = np.arange(1, L, dtype=np.float64)
        fac = np.sqrt((ell + 0.5) / (ell * (ell + 1.0)))
        T1[: L - 1] = fac[:, None] * A1[1:]
    return T0, T1


def _theta(L, nside):
    from gmaster.utils import _stable_thetas

    return np.asarray(_stable_thetas(L, nside), dtype=np.float64)


def forward_positive(positive, weights, phase, L, nside):
    """``(nring, L)`` complex ring block -> ``(L, L)`` ``(ell, m)`` coefficients."""
    theta = _theta(L, nside)
    T0, T1 = _basis(np.cos(theta), L)
    pos = np.asarray(positive, dtype=np.complex128)
    w = np.asarray(weights, dtype=np.float64)
    ph = np.asarray(phase, dtype=np.float64)
    out = np.zeros((L, L), dtype=np.complex128)
    for m in range(L):
        g = pos[:, m] * w * np.exp(1j * m * ph)
        m1 = m & 1
        beta = (T0 if m1 == 0 else T1) @ g
        ar = _lo2hi(beta.real, m1, m, L)
        ai = _lo2hi(beta.imag, m1, m, L)
        out[m:, m] = (ar[0 : L - m] + 1j * ai[0 : L - m]) * _INV_SQRT_2PI
    return out


def inverse_positive(alm, phase, L, nside):
    """``(L, L)`` coefficients -> ``(nring, L)`` ring block, phase included."""
    theta = _theta(L, nside)
    T0, T1 = _basis(np.cos(theta), L)
    alm = np.asarray(alm, dtype=np.complex128)
    ph = np.asarray(phase, dtype=np.float64)
    nring = theta.size
    out = np.zeros((nring, L), dtype=np.complex128)
    for m in range(L):
        Ar = np.zeros(L)
        Ai = np.zeros(L)
        Ar[: L - m] = alm[m:, m].real * _INV_SQRT_2PI
        Ai[: L - m] = alm[m:, m].imag * _INV_SQRT_2PI
        m1 = m & 1
        Br = _hi2lo(Ar, m1, m, L)
        Bi = _hi2lo(Ai, m1, m, L)
        basis = T0 if m1 == 0 else T1
        val = basis.T @ (Br + 1j * Bi)
        out[:, m] = val * np.exp(1j * m * ph)
    return out
