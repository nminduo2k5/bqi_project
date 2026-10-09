"""
Functional networks (Q1_pipeline.md Sec 5.1): weighted phase-lag index (wPLI,
Vinck et al. 2011) and graph metrics.

The algebraic connectivity (BQI prediction H5) is computed on the *full
weighted* wPLI matrix with the normalised Laplacian of BQI eq. (96): on
sparse proportional thresholds (e.g. density 0.2 with 19 channels) the graph
is typically disconnected and lambda_2 = 0 regardless of the data (gate G0
caught this). Efficiency, clustering and small-worldness use thresholded
binary graphs, normalised against degree-preserving null graphs.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from .sync import analytic


def wpli(Z: np.ndarray) -> np.ndarray:
    """wPLI from analytic signals Z: (n_channels, n_times) or a list of such
    epochs (expectation over time samples and epochs):

        wPLI_ij = |E[Im(S_ij)]| / E[|Im(S_ij)|],  S_ij = z_i conj(z_j)
    """
    Zs = [Z] if np.asarray(Z).ndim == 2 else list(Z)
    n = Zs[0].shape[0]
    num = np.zeros((n, n))
    den = np.zeros((n, n))
    for z in Zs:
        im = np.imag(z[:, None, :] * np.conj(z[None, :, :]))
        num += im.sum(axis=2)
        den += np.abs(im).sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        W = np.where(den > 0, np.abs(num) / den, 0.0)
    np.fill_diagonal(W, 0.0)
    return W


def wpli_matrix(X: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    return wpli(analytic(X, sfreq, band))


def proportional_threshold(W: np.ndarray, density: float) -> np.ndarray:
    """Keep the strongest `density` fraction of off-diagonal edges (weights kept)."""
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    iu = np.triu_indices(n, 1)
    vals = W[iu]
    k = max(1, int(round(density * len(vals))))
    cut = np.sort(vals)[-k]
    A = np.where(W >= cut, W, 0.0)
    np.fill_diagonal(A, 0.0)
    return np.maximum(A, A.T)


def fiedler(A: np.ndarray, normalized: bool = False) -> float:
    """Second-smallest Laplacian eigenvalue of a weighted adjacency matrix
    (0 if the graph is disconnected)."""
    A = np.asarray(A, dtype=float)
    d = A.sum(axis=1)
    L = np.diag(d) - A
    if normalized:
        with np.errstate(divide="ignore"):
            dinv = np.where(d > 0, 1 / np.sqrt(d), 0.0)
        L = dinv[:, None] * L * dinv[None, :]
    return float(max(np.sort(np.linalg.eigvalsh(L))[1], 0.0))


def _null_graphs(G: nx.Graph, n_null: int, rng) -> list:
    out = []
    m = G.number_of_edges()
    for _ in range(n_null):
        H = G.copy()
        try:
            nx.double_edge_swap(H, nswap=5 * m, max_tries=50 * m, seed=int(rng.integers(0, 2 ** 31)))
        except (nx.NetworkXError, nx.NetworkXAlgorithmError):
            pass
        out.append(H)
    return out


def graph_metrics(W: np.ndarray, density: float = 0.2, n_null: int = 10, rng=None) -> dict:
    """Fiedler values on the full weighted matrix W (normalised = primary, H5);
    efficiency/clustering/path length on the binary graph thresholded at
    `density`, normalised by degree-preserving null graphs
    (sigma = (C/C_null) / (L/L_null))."""
    rng = np.random.default_rng(rng)
    A = proportional_threshold(W, density)
    G = nx.from_numpy_array((A > 0).astype(int))
    out = {"fiedler_norm": fiedler(W, normalized=True), "fiedler": fiedler(W),
           "mean_weight": float(W[np.triu_indices(len(W), 1)].mean()),
           "n_components": int(nx.number_connected_components(G)),
           "E_glob": float(nx.global_efficiency(G)), "clustering": float(nx.average_clustering(G))}
    connected = nx.is_connected(G)
    out["path_length"] = float(nx.average_shortest_path_length(G)) if connected else float("nan")
    if n_null:
        nulls = _null_graphs(G, n_null, rng)
        C0 = np.mean([nx.average_clustering(H) for H in nulls])
        L0 = np.mean([nx.average_shortest_path_length(H) for H in nulls if nx.is_connected(H)] or [np.nan])
        E0 = np.mean([nx.global_efficiency(H) for H in nulls])
        out["E_glob_norm"] = out["E_glob"] / E0 if E0 > 0 else float("nan")
        out["sigma"] = ((out["clustering"] / C0) / (out["path_length"] / L0)
                        if connected and C0 > 0 and np.isfinite(L0) else float("nan"))
    return out
