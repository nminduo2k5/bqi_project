"""
Autoregressive integrated information Phi_AR (Barrett & Seth 2011),
Q1_pipeline.md Sec 5.1 (exploratory integration index, *not* IIT Phi).

For a stationary Gaussian process with lag tau:

    I(X_past; X_now) = 1/2 log det Sigma(X) / det Sigma(X_now | X_past)
    phi[P]          = I(X) - sum_k I(M^k)          for a bipartition P = {M^1, M^2}
    K[P]            = min_k 1/2 log((2 pi e)^{|M^k|} det Sigma(M^k))   (normaliser)

The minimum information bipartition (MIB) minimises phi[P] / K[P];
Phi_AR = phi[MIB]. Exhaustive over bipartitions, so keep n <= ~10 channels.

phi_WMS can be negative when the parts share redundant information (Mediano
et al. 2019); gate G0 caught this on a coupled VAR. The primary index is
therefore Phi_R = phi_WMS + Red (Mediano et al. 2021), with the minimum-
mutual-information redundancy Red = min_{i,j} I(M^i_past; M^j_now) over the
two parts, which is >= 0 for these Gaussian processes. Both are returned.
"""
from __future__ import annotations

import itertools

import numpy as np


def _cond_cov(S_now: np.ndarray, S_cross: np.ndarray, S_past: np.ndarray) -> np.ndarray:
    return S_now - S_cross @ np.linalg.solve(S_past, S_cross.T)


def _mutual_info(S0: np.ndarray, S_lag: np.ndarray, idx) -> float:
    """I(X_past; X_now) restricted to channels idx, from the lag-0 covariance S0
    and the lag-tau cross-covariance S_lag = Cov(X_now, X_past)."""
    ix = np.ix_(idx, idx)
    s0 = S0[ix]
    cond = _cond_cov(s0, S_lag[ix], s0)
    _, ld0 = np.linalg.slogdet(s0)
    _, ldc = np.linalg.slogdet(cond)
    return 0.5 * (ld0 - ldc)


def _cross_mi(S0: np.ndarray, S_lag: np.ndarray, src, dst) -> float:
    """I(X^src_past; X^dst_now) for a stationary Gaussian process."""
    s_dst = S0[np.ix_(dst, dst)]
    s_src = S0[np.ix_(src, src)]
    c = S_lag[np.ix_(dst, src)]  # Cov(now_dst, past_src)
    cond = s_dst - c @ np.linalg.solve(s_src, c.T)
    return 0.5 * (np.linalg.slogdet(s_dst)[1] - np.linalg.slogdet(cond)[1])


def _entropy(S0: np.ndarray, idx) -> float:
    s0 = S0[np.ix_(idx, idx)]
    _, ld = np.linalg.slogdet(s0)
    return 0.5 * (len(idx) * np.log(2 * np.pi * np.e) + ld)


def phi_ar(X: np.ndarray, lag: int = 1) -> dict:
    """X: (n_channels, n_times). Returns Phi_AR at the MIB, the MIB and the
    whole-system mutual information."""
    X = np.asarray(X, dtype=float)
    X = X - X.mean(axis=1, keepdims=True)
    n, T = X.shape
    now, past = X[:, lag:], X[:, :-lag]
    m = T - lag
    S0 = (X @ X.T) / T
    S0 = S0 + 1e-10 * np.trace(S0) / n * np.eye(n)
    S_lag = now @ past.T / m
    full = list(range(n))
    I_whole = _mutual_info(S0, S_lag, full)
    best = None
    for r in range(1, n // 2 + 1):
        for part in itertools.combinations(full, r):
            if 2 * r == n and 0 not in part:
                continue  # each balanced bipartition once
            other = [i for i in full if i not in part]
            a, b = list(part), other
            phi_wms = I_whole - _mutual_info(S0, S_lag, a) - _mutual_info(S0, S_lag, b)
            red = min(_cross_mi(S0, S_lag, x, y) for x in (a, b) for y in (a, b))
            phi_r = phi_wms + red
            K = min(_entropy(S0, a), _entropy(S0, b))
            norm = phi_r / abs(K) if K != 0 else np.inf
            if best is None or norm < best[0]:
                best = (norm, phi_r, phi_wms, tuple(part))
    return {"phi_r": float(best[1]), "phi_r_norm": float(best[0]), "phi_wms": float(best[2]),
            "mib": best[3], "I_whole": float(I_whole), "phi_ar": float(best[1])}
