"""Numba-parallel fused CPU kernel for the spectral pre-FFT stage.

Fuses: apodization subtraction, per-A-line mean removal, banded sinc
k-linearisation, mean re-add and complex window multiply into one pass,
parallelised over A-lines. Numerically equivalent to SpectralProcessor.linearise
followed by `* window` (same operation order per sample).
"""
import numba as nb
import numpy as np


@nb.njit(parallel=True, fastmath=False, cache=True)
def fused_pre_fft(raw, apod_size, idx, w, window, out):
    # raw (B, interf, N) int16/uint16/float (numba specialises per dtype);
    # idx/w (T, N); window (N,) complex; out (B, nX, N) complex
    B, interf, N = raw.shape
    nX = interf - apod_size
    T = idx.shape[0]
    apod = np.empty((B, N), dtype=w.dtype)
    for b in nb.prange(B):
        for j in range(N):
            s = 0.0
            for a in range(apod_size):
                s += raw[b, a, j]
            apod[b, j] = s / apod_size
    for p in nb.prange(B * nX):
        b = p // nX
        x = p - b * nX
        d = np.empty(N, dtype=w.dtype)
        m = 0.0
        for j in range(N):
            d[j] = raw[b, apod_size + x, j] - apod[b, j]
            m += d[j]
        m = m / N
        for j in range(N):
            d[j] -= m
        for i in range(N):
            acc = 0.0
            for t in range(T):
                acc += w[t, i] * d[idx[t, i]]
            out[b, x, i] = (acc + m) * window[i]
