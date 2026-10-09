"""
Graph-theoretic brain network model + Kuramoto synchronisation, paper Sec 4.4
(graph-theoretic model, Fiedler value, small-world index) and Sec 9.4
(spectral graph theory, Table kuramoto).
"""
from __future__ import annotations
import numpy as np
import networkx as nx


def build_small_world_brain_graph(n_nodes: int = 90, k: int = 6, p_rewire: float = 0.1,
                                   seed: int = 0) -> nx.Graph:
    """Watts-Strogatz small-world graph as a schematic brain functional
    connectivity network (paper Fig 'brain functional connectivity graph')."""
    return nx.watts_strogatz_graph(n_nodes, k, p_rewire, seed=seed)


def fiedler_value(G: nx.Graph) -> float:
    """Second-smallest eigenvalue of the graph Laplacian (algebraic
    connectivity), used in Table params (Fiedler value at wakefulness vs
    deep meditation)."""
    L = nx.laplacian_matrix(G).toarray().astype(float)
    eigvals = np.sort(np.linalg.eigvalsh(L))
    return float(eigvals[1])


def small_world_index(G: nx.Graph, n_random: int = 5, seed: int = 0) -> float:
    """sigma_SW = (C/C_rand) / (L/L_rand), the classic small-world index
    (Humphries & Gurney 2008), matching the paper's sigma_SW ~ 3.1 claim."""
    C = nx.average_clustering(G)
    try:
        Lp = nx.average_shortest_path_length(G)
    except nx.NetworkXError:
        Lp = np.inf
    rng = np.random.default_rng(seed)
    C_rand_list, Lp_rand_list = [], []
    n, m = G.number_of_nodes(), G.number_of_edges()
    for i in range(n_random):
        Gr = nx.gnm_random_graph(n, m, seed=int(rng.integers(0, 10 ** 6)))
        C_rand_list.append(nx.average_clustering(Gr))
        try:
            Lp_rand_list.append(nx.average_shortest_path_length(Gr))
        except nx.NetworkXError:
            pass
    C_rand = np.mean(C_rand_list) if C_rand_list else 1e-6
    Lp_rand = np.mean(Lp_rand_list) if Lp_rand_list else Lp
    sigma = (C / max(C_rand, 1e-9)) / (Lp / max(Lp_rand, 1e-9))
    return float(sigma)


def global_efficiency(G: nx.Graph) -> float:
    return float(nx.global_efficiency(G))


def meditation_graph_transform(G: nx.Graph, boost_fraction: float = 0.08, seed: int = 0) -> nx.Graph:
    """Approximate 'deep meditation increases E_glob by ~8% and sigma_SW by
    ~12%' (paper Fig connectivity graph caption) by adding a small fraction
    of new long-range edges (rewiring toward more efficient global topology,
    consistent with reported meditation-related connectivity changes)."""
    rng = np.random.default_rng(seed)
    G2 = G.copy()
    n_new = int(round(boost_fraction * G.number_of_edges()))
    nodes = list(G2.nodes())
    added = 0
    attempts = 0
    while added < n_new and attempts < n_new * 20:
        u, v = rng.choice(nodes, size=2, replace=False)
        attempts += 1
        if not G2.has_edge(u, v):
            G2.add_edge(u, v)
            added += 1
    return G2


# --- Kuramoto oscillator model (order parameter r, Table kuramoto) --------

def simulate_kuramoto(n_oscillators: int, coupling_K: float, T: float = 20.0, dt: float = 0.01,
                       natural_freq_std: float = 1.0, seed: int = 0):
    """Standard Kuramoto model:
        dtheta_i/dt = omega_i + (K/N) sum_j sin(theta_j - theta_i)
    Returns the time series of the order parameter r(t) = |mean_j e^{i theta_j}|.
    """
    rng = np.random.default_rng(seed)
    omega = rng.normal(0, natural_freq_std, n_oscillators)
    theta = rng.uniform(0, 2 * np.pi, n_oscillators)
    n_steps = int(T / dt)
    r_trace = np.zeros(n_steps)
    for t in range(n_steps):
        z = np.mean(np.exp(1j * theta))
        r_trace[t] = np.abs(z)
        psi = np.angle(z)
        dtheta = omega + coupling_K * r_trace[t] * np.sin(psi - theta)
        theta = theta + dt * dtheta
    return r_trace


def simulate_kuramoto_graph(W: np.ndarray, coupling_K: float, T: float = 20.0, dt: float = 0.01,
                            natural_freq_std: float = 1.0, natural_freq_mean: float = 0.0,
                            seed: int = 0, return_phases: bool = False):
    """Kuramoto model on a weighted graph, paper eq. (97):

        dphi_i/dt = omega_i + (K/n) sum_j W_ij sin(phi_j - phi_i)

    `simulate_kuramoto` above is the all-to-all (mean-field) special case
    W = 1. Returns r(t) (eq. 98) and, optionally, the phase matrix (steps x n).
    """
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    rng = np.random.default_rng(seed)
    omega = rng.normal(natural_freq_mean, natural_freq_std, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    n_steps = int(T / dt)
    r_trace = np.zeros(n_steps)
    phases = np.zeros((n_steps, n)) if return_phases else None
    for t in range(n_steps):
        z = np.exp(1j * theta)
        r_trace[t] = np.abs(z.mean())
        if return_phases:
            phases[t] = theta
        # sum_j W_ij sin(theta_j - theta_i) = Im(conj(z_i) * (W z)_i)
        coupling = np.imag(np.conj(z) * (W @ z))
        theta = theta + dt * (omega + coupling_K / n * coupling)
    return (r_trace, phases) if return_phases else r_trace


def graph_adjacency(G: nx.Graph) -> np.ndarray:
    return nx.to_numpy_array(G, weight="weight", nodelist=sorted(G.nodes()))


# Coupling strengths per consciousness state x frequency band, calibrated so
# the resulting stationary Kuramoto order parameter r approximately matches
# Table kuramoto's r_bar(theta/alpha/gamma) pattern (higher order under
# meditation, lowest under anaesthesia).
STATE_BAND_COUPLING = {
    "Wakefulness (rest)":   {"theta": 1.2, "alpha": 2.0, "gamma": 0.9},
    "Task-engaged":         {"theta": 1.0, "alpha": 1.7, "gamma": 1.4},
    "Light meditation":     {"theta": 1.8, "alpha": 2.1, "gamma": 1.3},
    "Deep meditation":      {"theta": 2.6, "alpha": 2.3, "gamma": 1.9},
    "NREM sleep":           {"theta": 0.7, "alpha": 0.9, "gamma": 0.5},
    "Propofol anaesthesia": {"theta": 0.5, "alpha": 0.6, "gamma": 0.35},
}


def generate_kuramoto_dataset(n_oscillators: int = 60, n_trials: int = 10, seed: int = 0):
    """Generate a per-trial synthetic dataset of stationary Kuramoto order
    parameters across states x frequency bands (a new dataset reproducing
    the qualitative pattern of Table kuramoto)."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for state, bands in STATE_BAND_COUPLING.items():
        for band, K in bands.items():
            for trial in range(n_trials):
                r_trace = simulate_kuramoto(n_oscillators, K, T=8.0, dt=0.02,
                                             seed=int(rng.integers(0, 10 ** 6)))
                r_stationary = float(np.mean(r_trace[-50:]))
                rows.append({"state": state, "band": band, "trial": trial, "r": r_stationary})
    return pd.DataFrame(rows)
