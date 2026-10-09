"""
Phase-randomised surrogates for the spectral control of Q1_pipeline.md Sec 5.2.

`phase_randomize(X, multivariate=True)` adds the *same* random phase to every
channel at each frequency, so every channel's power spectrum and every
cross-spectrum are preserved while any nonlinear / higher-order structure is
destroyed (Prichard & Theiler 1994). `multivariate=False` draws independent
phases per channel, which also destroys inter-channel phase relations.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def phase_randomize(X: np.ndarray, multivariate: bool = True, rng=None) -> np.ndarray:
    """X: (n_channels, n_times) real array. Returns a surrogate with identical
    amplitude spectra."""
    rng = np.random.default_rng(rng)
    X = np.asarray(X, dtype=float)
    n_ch, n = X.shape
    F = np.fft.rfft(X, axis=1)
    n_freq = F.shape[1]
    shape = (1, n_freq) if multivariate else (n_ch, n_freq)
    phases = rng.uniform(0, 2 * np.pi, shape)
    phases[:, 0] = 0.0                       # keep the mean real
    if n % 2 == 0:
        phases[:, -1] = 0.0                  # Nyquist bin must stay real
    return np.fft.irfft(F * np.exp(1j * phases), n=n, axis=1)


def spectrally_controlled(metric: Callable[[np.ndarray], float], X: np.ndarray, n_surrogates: int = 20,
                          multivariate: bool = True, rng=None) -> dict:
    """Spectrum-independent version of a metric (Sec 5.2):

        X_spec-indep = (X_real - mean(X_surr)) / sd(X_surr)

    Returns the raw value, surrogate mean/sd and the z-scored value.
    """
    rng = np.random.default_rng(rng)
    real = float(metric(X))
    surr = np.array([metric(phase_randomize(X, multivariate=multivariate, rng=rng))
                     for _ in range(n_surrogates)], dtype=float)
    sd = float(surr.std(ddof=1)) if n_surrogates > 1 else float("nan")
    return {"raw": real, "surr_mean": float(surr.mean()), "surr_sd": sd,
            "z": (real - surr.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else float("nan"),
            "diff": real - float(surr.mean())}
