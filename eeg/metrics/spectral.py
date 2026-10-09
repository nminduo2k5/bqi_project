"""
Spectral metrics (Q1_pipeline.md Sec 5.1): aperiodic exponent/offset and
relative band power.

The aperiodic fit is a robust log-log linear fit (no knee) of the Welch PSD
over [fmin, fmax], iteratively excluding frequencies whose residual is above
`peak_threshold` standard deviations (oscillatory peaks such as alpha). When
`specparam` (or `fooof`) is installed, `aperiodic_fit(..., backend="specparam")`
uses it instead; both report  log10 P(f) = offset - exponent * log10 f.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0),
         "gamma_low": (30.0, 40.0)}


def psd(X: np.ndarray, sfreq: float, nperseg_s: float = 2.0):
    """Welch PSD per channel. Returns (freqs, P) with P: (n_channels, n_freqs)."""
    X = np.atleast_2d(X)
    nperseg = min(int(nperseg_s * sfreq), X.shape[1])
    f, P = welch(X, fs=sfreq, nperseg=nperseg, axis=1)
    return f, P


def _robust_fit(logf: np.ndarray, logp: np.ndarray, peak_threshold: float = 1.0, n_iter: int = 5):
    mask = np.ones_like(logf, dtype=bool)
    slope, intercept = np.polyfit(logf, logp, 1)
    for _ in range(n_iter):
        resid = logp - (intercept + slope * logf)
        sd = resid[mask].std()
        new = resid < peak_threshold * sd if sd > 0 else mask
        if new.sum() < 3 or np.array_equal(new, mask):
            break
        mask = new
        slope, intercept = np.polyfit(logf[mask], logp[mask], 1)
    return -slope, intercept


def aperiodic_fit(freqs: np.ndarray, P: np.ndarray, fmin: float = 1.0, fmax: float = 40.0,
                  backend: str = "robust") -> dict:
    """Aperiodic exponent and offset per channel plus their channel means."""
    P = np.atleast_2d(P)
    sel = (freqs >= fmin) & (freqs <= fmax) & (freqs > 0)
    f = freqs[sel]
    if backend == "specparam":
        try:
            from specparam import SpectralGroupModel as Group
        except ImportError:
            from fooof import FOOOFGroup as Group  # type: ignore
        g = Group(aperiodic_mode="fixed", verbose=False)
        g.fit(f, P[:, sel])
        ap = g.get_params("aperiodic_params")
        offset, exponent = ap[:, 0], ap[:, 1]
    else:
        res = [_robust_fit(np.log10(f), np.log10(p)) for p in P[:, sel]]
        exponent = np.array([r[0] for r in res])
        offset = np.array([r[1] for r in res])
    return {"exponent": exponent, "offset": offset,
            "exponent_mean": float(np.mean(exponent)), "offset_mean": float(np.mean(offset))}


def relative_band_power(freqs: np.ndarray, P: np.ndarray, bands: dict = BANDS,
                        fmin: float = 1.0, fmax: float = 40.0) -> dict:
    """Band power / total power in [fmin, fmax], averaged over channels."""
    P = np.atleast_2d(P)
    tot_sel = (freqs >= fmin) & (freqs <= fmax)
    total = np.trapezoid(P[:, tot_sel], freqs[tot_sel], axis=1)
    out = {}
    for name, (lo, hi) in bands.items():
        sel = (freqs >= lo) & (freqs < hi)
        out[name] = float(np.mean(np.trapezoid(P[:, sel], freqs[sel], axis=1) / total))
    return out
