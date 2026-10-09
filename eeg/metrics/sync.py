"""
Phase synchrony (Q1_pipeline.md Sec 5.1): band-limited Hilbert phases,
Kuramoto order parameter across channels and metastability.

Compute on CSD (surface-Laplacian) data to limit volume conduction.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert


def bandpass(X: np.ndarray, sfreq: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    sos = butter(order, [lo, hi], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, np.atleast_2d(X), axis=1)


def analytic(X: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    """Complex analytic signal of the band-passed data, (n_channels, n_times)."""
    return hilbert(bandpass(X, sfreq, *band), axis=1)


def kuramoto_order(phases: np.ndarray) -> np.ndarray:
    """r(t) = |mean_j exp(i phi_j(t))| (BQI eq. 98); phases: (n_channels, n_times)."""
    return np.abs(np.exp(1j * phases).mean(axis=0))


def synchrony_metrics(X: np.ndarray, sfreq: float, band: tuple[float, float], trim_s: float = 0.5) -> dict:
    """Mean Kuramoto order parameter and metastability (SD of r(t)), with
    `trim_s` seconds removed at both edges to avoid filter/Hilbert edge effects."""
    z = analytic(X, sfreq, band)
    k = int(trim_s * sfreq)
    if k > 0 and z.shape[1] > 2 * k + 10:
        z = z[:, k:-k]
    r = kuramoto_order(np.angle(z))
    return {"kuramoto_r": float(r.mean()), "metastability": float(r.std())}
