"""
Coalition entropies (Schartner et al. 2015), Q1_pipeline.md Sec 5.1.

ACE: binarise each channel's amplitude envelope at its mean; the coalition at
time t is the set of "active" channels. ACE = entropy of the coalition
distribution, normalised by the entropy obtained after shuffling each
channel independently in time.

SCE: for each channel i, the synchrony coalition at t is the binary vector of
channels j != i whose instantaneous phase difference is below `threshold`
radians. SCE = mean over channels of the coalition entropy, normalised the
same way.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.signal import hilbert

from .lzc import binarize_envelope


def _entropy_of_rows(B: np.ndarray) -> float:
    """Shannon entropy (bits) of the empirical distribution of binary columns of B
    (n_bits, n_times)."""
    packed = np.packbits(B.astype(np.uint8), axis=0)
    keys = [bytes(col) for col in packed.T]
    counts = np.array(list(Counter(keys).values()), dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _shuffle_rows(B: np.ndarray, rng) -> np.ndarray:
    return np.array([rng.permutation(row) for row in B])


def ace(X: np.ndarray, rng=None) -> float:
    rng = np.random.default_rng(rng)
    B = binarize_envelope(X)
    h = _entropy_of_rows(B)
    h0 = _entropy_of_rows(_shuffle_rows(B, rng))
    return h / h0 if h0 > 0 else 0.0


def sce(X: np.ndarray, threshold: float = 0.8, rng=None) -> float:
    """X should already be band-limited (e.g. 1-40 Hz); phases from Hilbert."""
    rng = np.random.default_rng(rng)
    X = np.asarray(X, dtype=float)
    ph = np.angle(hilbert(X - X.mean(axis=1, keepdims=True), axis=1))
    n = X.shape[0]
    vals = []
    for i in range(n):
        d = np.angle(np.exp(1j * (ph[i][None, :] - np.delete(ph, i, axis=0))))
        B = (np.abs(d) < threshold).astype(np.uint8)
        h = _entropy_of_rows(B)
        h0 = _entropy_of_rows(_shuffle_rows(B, rng))
        vals.append(h / h0 if h0 > 0 else 0.0)
    return float(np.mean(vals))
