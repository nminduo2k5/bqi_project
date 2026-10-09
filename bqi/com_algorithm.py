"""
Algorithm 2 — COM: Consciousness Optimisation Model (paper Sec. 8.2, alg:com).

Phase 1 — DMN Suppression (closed-loop BCI stimulation)
Phase 2 — Free-Energy Minimisation (drives Phi upward)
Phase 3 — SNIS Query Expansion (drives QoC toward target)

A literal, runnable translation of the pseudocode, reusing the BQI attention
mechanism for Phase 3 and a simple IIT-approx proxy for Phi(t) in Phase 2 so
the whole loop runs end-to-end without the NP-hard exact Phi computation.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from .attention import bqi_attention


@dataclass
class COMConfig:
    T: int = 90               # total time steps (must be divisible by 3 for clean phases)
    d_mu: int = 8              # dimensionality of belief mu
    d_snis: int = 8
    n_snis: int = 50
    d_k: int = 8
    QoC_target: float = 1.0
    K_gain: float = 0.4        # closed-loop BCI gain
    eta_dmn: float = 0.05
    eta_F: float = 0.05
    eta_S: float = 0.15
    alpha: float = 2.34        # Phi -> QoC gain (Table ode_params)
    beta: float = 1.87         # DMN -> QoC gain
    gamma0: float = 0.02
    seed: int = 0


def iit_approx(mu: np.ndarray, A_dmn: float) -> float:
    """'IIT-approx(mu, A_DMN)' proxy referenced in the COM pseudocode:
    integration increases with the dispersion (variance) of belief mu and
    decreases with DMN activity (which competes for the same resource pool)."""
    dispersion = float(np.var(mu))
    return dispersion / (1e-3 + A_dmn)


def run_com_algorithm(cfg: COMConfig, B0: np.ndarray | None = None):
    """Executes Algorithm 2 end-to-end and returns the full time series plus
    final (B*, QoC(T), Delta_S)."""
    rng = np.random.default_rng(cfg.seed)
    T = cfg.T
    d_mu = cfg.d_mu

    mu = rng.normal(0, 0.5, d_mu) if B0 is None else B0.copy()
    M = rng.normal(0, 1, (cfg.n_snis, cfg.d_snis))
    W_Q = rng.normal(0, 1 / np.sqrt(d_mu), (d_mu, cfg.d_k))

    A_dmn = 1.0     # measure(B0) -> initial DMN activity (normalised, starts high)
    Phi = 0.05       # measure(B0) -> initial Phi
    QoC = 0.0        # measure(B0) -> initial QoC

    trace = {"t": [], "A_dmn": [], "Phi": [], "QoC": [], "mu_norm": []}

    def record(t):
        trace["t"].append(t)
        trace["A_dmn"].append(A_dmn)
        trace["Phi"].append(Phi)
        trace["QoC"].append(QoC)
        trace["mu_norm"].append(float(np.linalg.norm(mu)))

    B_init_norm = np.linalg.norm(mu) + 1e-9

    third = max(1, T // 3)

    # ---- Phase 1: DMN Suppression -----------------------------------------
    for t in range(1, third + 1):
        u_stim = cfg.K_gain * (cfg.QoC_target - QoC)         # closed-loop BCI gain
        grad_att_F = A_dmn * 0.3 + rng.normal(0, 0.01)         # proxy for grad_att F
        A_dmn = A_dmn * (1 - cfg.eta_dmn * grad_att_F * (1 + u_stim))
        A_dmn = max(A_dmn, 1e-4)
        record(t)

    # ---- Phase 2: Free-Energy Minimisation --------------------------------
    k_int, k_d, A_max = 0.081, 0.034, 1.0
    dt = 1.0
    for t in range(third + 1, 2 * third + 1):
        # variational free energy proxy: complexity - accuracy, decreasing as mu -> 0
        grad_F = mu - 0.1 * np.sign(mu)
        mu = mu - cfg.eta_F * grad_F
        # Phi(t) <- IIT-approx(mu, A_DMN): integrated via the same qualitative
        # law as the DMN/Phi/QoC ODE system (eq. 26), so Phi rises as A_DMN falls
        dPhi = k_int * (1 - A_dmn / A_max) - k_d * Phi
        Phi = max(Phi + dt * dPhi, 0.0)
        record(t)

    # ---- Phase 3: SNIS Query Expansion -------------------------------------
    for t in range(2 * third + 1, T + 1):
        Q = (W_Q.T @ mu).reshape(1, -1)
        A, _ = bqi_attention(Q, M[:, :cfg.d_k] if M.shape[1] >= cfg.d_k else M, M)
        mu = mu + cfg.eta_S * A.flatten()[: len(mu)]
        QoC = cfg.alpha * Phi - cfg.beta * A_dmn + cfg.gamma0
        record(t)

    B_star = mu.copy()
    delta_S = float(np.linalg.norm(B_star) / B_init_norm)

    return {
        "B_star": B_star,
        "QoC_final": QoC,
        "delta_S": delta_S,
        "trace": trace,
    }
