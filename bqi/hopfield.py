"""
Classic and modern (dense) Hopfield networks as SNIS proxies, paper Sec 9.2.

    Classic Hopfield capacity:  N_max ~ 0.14 n     (linear)
    Modern Hopfield capacity:   N_max ~ e^{n/2}    (exponential, softmax update)

Reproduces the "retrieval error rate vs number of stored patterns" figure,
comparing classic Hopfield, modern (softmax) Hopfield, and Transformer
dot-product attention (shown in the paper to be equivalent to modern Hopfield
retrieval, Ramsauer et al. 2021).
"""
from __future__ import annotations
import numpy as np


def store_patterns_classic(patterns: np.ndarray) -> np.ndarray:
    """Hebbian outer-product storage rule for classic Hopfield networks.
    patterns: (P, n) matrix of {-1,+1} patterns. Returns weight matrix (n,n)."""
    P, n = patterns.shape
    W = (patterns.T @ patterns) / n
    np.fill_diagonal(W, 0.0)
    return W


def retrieve_classic(W: np.ndarray, probe: np.ndarray, n_iter: int = 10) -> np.ndarray:
    """Asynchronous-style (batched, sign-update) retrieval dynamics."""
    x = probe.copy()
    for _ in range(n_iter):
        x = np.sign(W @ x)
        x[x == 0] = 1
    return x


def retrieve_modern(patterns: np.ndarray, probe: np.ndarray, beta: float = 8.0,
                     n_iter: int = 3) -> np.ndarray:
    """Modern (dense associative memory) Hopfield update rule, eq. from
    Ramsauer et al. (2021): x_new = X^T softmax(beta * X x), which the paper
    shows is algebraically identical to Transformer dot-product attention
    with a single query."""
    x = probe.astype(float).copy()
    X = patterns.astype(float)
    for _ in range(n_iter):
        scores = beta * (X @ x)
        scores = scores - scores.max()
        w = np.exp(scores)
        w = w / w.sum()
        x = X.T @ w
    return np.sign(x)


def transformer_attention_retrieve(patterns: np.ndarray, probe: np.ndarray, d_k: float | None = None) -> np.ndarray:
    """Single-query scaled dot-product attention retrieval -- shown by
    Ramsauer et al. (2021) / paper Sec 9.2 to be equivalent to a single
    update step of the modern Hopfield network with beta = 1/sqrt(d_k)."""
    n = patterns.shape[1]
    d_k = d_k or n
    scores = (patterns.astype(float) @ probe.astype(float)) / np.sqrt(d_k)
    scores = scores - scores.max()
    w = np.exp(scores)
    w = w / w.sum()
    out = patterns.astype(float).T @ w
    return np.sign(out)


def random_patterns(P: int, n: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=(P, n))


def corrupt(pattern: np.ndarray, flip_fraction: float, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = pattern.copy()
    n_flip = int(round(flip_fraction * len(x)))
    idx = rng.choice(len(x), size=n_flip, replace=False)
    x[idx] *= -1
    return x


def retrieval_error_experiment(n: int = 100, pattern_counts=None, flip_fraction: float = 0.1,
                                trials_per_count: int = 20, beta: float = 8.0, seed: int = 0):
    """Reproduce Fig 'retrieval error rate vs number of stored patterns' for
    classic Hopfield (linear capacity ~0.14n), modern Hopfield (exponential
    capacity ~e^{n/2}), and Transformer attention (equivalent to modern
    Hopfield). Returns a dict of arrays ready for plotting / a DataFrame.
    """
    import pandas as pd
    rng = np.random.default_rng(seed)
    if pattern_counts is None:
        max_p = int(np.exp(n / 2)) if n <= 20 else 2000  # e^{n/2} explodes for n>~20; cap for large n
        pattern_counts = sorted(set(
            list(range(1, min(20, max_p) + 1))
            + [int(x) for x in np.geomspace(2, max(max_p, 3), num=25)]
        ))
        pattern_counts = [p for p in pattern_counts if p >= 1]

    rows = []
    for P in pattern_counts:
        classic_errors, modern_errors, transformer_errors = [], [], []
        for trial in range(trials_per_count):
            patterns = random_patterns(P, n, seed=int(rng.integers(0, 10 ** 6)))
            target_idx = rng.integers(0, P)
            target = patterns[target_idx]
            probe = corrupt(target, flip_fraction, seed=int(rng.integers(0, 10 ** 6)))

            W = store_patterns_classic(patterns)
            recon_classic = retrieve_classic(W, probe)
            classic_errors.append(np.mean(recon_classic != target))

            recon_modern = retrieve_modern(patterns, probe, beta=beta)
            modern_errors.append(np.mean(recon_modern != target))

            recon_tf = transformer_attention_retrieve(patterns, probe, d_k=n)
            transformer_errors.append(np.mean(recon_tf != target))

        rows.append({
            "n_patterns": P,
            "classic_error_rate": float(np.mean(classic_errors)),
            "modern_error_rate": float(np.mean(modern_errors)),
            "transformer_error_rate": float(np.mean(transformer_errors)),
        })
    return pd.DataFrame(rows)


def theoretical_capacity_classic(n: int) -> float:
    """N_max ~ 0.14 n (Amit, Gutfreund & Sompolinsky, 1985; cited via paper Sec 9.2)."""
    return 0.14 * n


def theoretical_capacity_modern(n: int) -> float:
    """N_max ~ e^{n/2} (Ramsauer et al., 2021; paper Table params)."""
    return float(np.exp(n / 2))
