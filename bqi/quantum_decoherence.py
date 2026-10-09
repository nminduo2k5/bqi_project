"""
Consciousness field theory: quantum formalisation, paper Sec 6.

  * Brain as an open quantum system: von Neumann entropy dynamics
        S(rho) = -Tr[rho ln rho]
    under a simple Lindblad-type decoherence channel.
  * Orch-OR collapse timescale: tau = hbar / E_G  (Penrose-Hameroff).
"""
from __future__ import annotations
import numpy as np

HBAR = 1.054571817e-34  # J*s
G_NEWTON = 6.674e-11    # m^3 kg^-1 s^-2


def von_neumann_entropy(rho: np.ndarray) -> float:
    """S(rho) = -Tr[rho ln rho], computed via eigenvalues (numerically stable)."""
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-14]
    return float(-np.sum(eigvals * np.log(eigvals)))


def random_density_matrix(dim: int, purity: float = 1.0, seed: int | None = None) -> np.ndarray:
    """Generate a random density matrix with a given approximate purity via
    a convex mixture of a random pure state and the maximally mixed state."""
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    psi /= np.linalg.norm(psi)
    rho_pure = np.outer(psi, psi.conj())
    rho_mixed = np.eye(dim) / dim
    rho = purity * rho_pure + (1 - purity) * rho_mixed
    return rho


def lindblad_dephasing_step(rho: np.ndarray, gamma: float, dt: float) -> np.ndarray:
    """One Euler step of pure-dephasing Lindblad evolution in the energy
    eigenbasis: off-diagonal coherences decay exponentially at rate gamma,
    populations (diagonal) are conserved. This is the simplest model of
    environmentally induced decoherence for an open quantum system."""
    dim = rho.shape[0]
    decay = np.exp(-gamma * dt)
    mask = np.ones((dim, dim)) * decay
    np.fill_diagonal(mask, 1.0)
    return rho * mask


def simulate_entropy_dynamics(dim: int, gamma: float, T: float = 5.0, dt: float = 0.01,
                               purity0: float = 0.98, seed: int = 0):
    """Simulate S(rho(t)) for a decohering system, reproducing the shape of
    Fig 'von Neumann entropy dynamics for three states' (paper Sec 6):
    entropy rises monotonically from a low-entropy (coherent) initial state
    toward the maximally-mixed-state entropy ln(dim), at a rate set by the
    decoherence rate `gamma` (larger gamma -> faster collapse to classicality).
    """
    rho = random_density_matrix(dim, purity=purity0, seed=seed)
    n_steps = int(T / dt)
    S_trace = np.zeros(n_steps)
    t_trace = np.linspace(0, T, n_steps)
    for i in range(n_steps):
        S_trace[i] = von_neumann_entropy(rho)
        rho = lindblad_dephasing_step(rho, gamma, dt)
    return {"t": t_trace, "S": S_trace, "S_max": np.log(dim)}


def orch_or_collapse_time(mass_kg: float, delta_x_m: float) -> float:
    """Orch-OR collapse timescale (Penrose-Hameroff): tau = hbar / E_G,
    with the gravitational self-energy of the mass superposition estimated
    as E_G ~ G m^2 / delta_x (a standard simplified Diosi-Penrose estimate).
    """
    E_G = G_NEWTON * (mass_kg ** 2) / delta_x_m
    return HBAR / E_G


def generate_decoherence_dataset(dims=(2, 4, 8), gammas=(0.5, 2.0, 8.0), n_trials=5, seed=0):
    """New dataset: entropy-time trajectories across a grid of Hilbert-space
    dimensions and decoherence rates, for downstream regression/validation
    that S(t) rises monotonically toward ln(dim)."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for dim in dims:
        for gamma in gammas:
            for trial in range(n_trials):
                res = simulate_entropy_dynamics(dim, gamma, T=3.0, dt=0.02,
                                                  seed=int(rng.integers(0, 10 ** 6)))
                rows.append({
                    "dim": dim, "gamma": gamma, "trial": trial,
                    "S_final": res["S"][-1], "S_max": res["S_max"],
                    "S_initial": res["S"][0],
                })
    return pd.DataFrame(rows)
