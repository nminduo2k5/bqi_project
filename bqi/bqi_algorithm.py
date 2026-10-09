"""
Algorithm 1 — BQI-Algorithm: Three-Phase Belief Update (paper Sec. 4.5).

Phase 1 — Hierarchical Prediction-Error Computation
Phase 2 — Variational Free-Energy Minimisation
Phase 3 — SNIS Query and Decoding

This is a direct, literal translation of the pseudocode in the paper into
runnable Python, using linear-Gaussian generative/observation models so the
free-energy gradient and Hessian are available in closed form (needed to
verify the convergence guarantee, eq. 15-16).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from .attention import bqi_attention, softmax


@dataclass
class BQIConfig:
    n_levels: int = 3
    dims: tuple = (8, 8, 6, 4)   # level 0..L dimensions, level 0 = observation dim
    n_snis: int = 64             # |SNIS proxy| rows
    d_snis: int = 4              # SNIS proxy embedding dim (= dims[-1], top belief dim)
    d_k: int = 4
    eta: float = 0.05
    alpha: float = 0.3           # SNIS-informed belief update gain
    T: int = 200
    tol: float = 1e-5
    seed: int = 0


class BQIModel:
    """Runnable version of Algorithm 1 (alg:bqi)."""

    def __init__(self, cfg: BQIConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        L = cfg.n_levels
        dims = cfg.dims
        assert len(dims) == L + 1

        # Top-down generative weights f^(l): level l+1 -> level l, l=1..L-1
        self.f = [rng.normal(0, 0.4, (dims[l], dims[l + 1])) for l in range(1, L)]
        # Observation weights g^(l): predicts o^(l) (taken = mu^(l) itself, i.e. g=I)
        self.g = [np.eye(dims[l]) for l in range(1, L + 1)]
        # Precision matrices Pi^(l)
        self.Pi = [np.diag(rng.uniform(0.5, 1.5, dims[l])) for l in range(1, L + 1)]
        # Beliefs mu^(l), l=1..L
        self.mu = [rng.normal(0, 0.1, dims[l]) for l in range(1, L + 1)]
        # Synaptic parameters theta^(l) (here: the f matrices themselves are theta)
        self.theta = [w.copy() for w in self.f]

        # SNIS proxy M in R^{n_s x d}
        self.M = rng.normal(0, 1, (cfg.n_snis, cfg.d_snis))
        # Attention projections
        self.W_Q = rng.normal(0, 1 / np.sqrt(dims[L]), (dims[L], cfg.d_k))

        self.rng = rng

    # ---- Phase 1 -------------------------------------------------------
    def phase1_prediction_errors(self, o: np.ndarray):
        """Hierarchical prediction-error computation, top-down predictions g^(l)(mu^(l))
        compared against observations o^(l) (level 0 clamped to `o`, higher
        levels compare against their own current belief as a self-consistency
        residual — matching the paper's o^(l) - o_hat^(l) with o^(l)=mu^(l))."""
        L = self.cfg.n_levels
        mu_full = [o] + self.mu
        eps = []
        for l in range(1, L + 1):
            o_hat = self.g[l - 1] @ mu_full[l]
            o_l = mu_full[l]
            eps.append(self.Pi[l - 1] @ (o_l - o_hat))
        return eps

    # ---- Free energy for phase 2 ---------------------------------------
    def free_energy(self, o: np.ndarray) -> float:
        L = self.cfg.n_levels
        mu_full = [o] + self.mu + [np.zeros(1)]
        F = 0.0
        for l in range(1, L + 1):
            if l < L:
                pred = self.f[l - 1] @ mu_full[l + 1]
            else:
                pred = np.zeros(self.cfg.dims[l])
            resid = mu_full[l] - pred
            F += 0.5 * resid @ self.Pi[l - 1] @ resid
        return float(F)

    def free_energy_grad_mu(self, o: np.ndarray):
        """Numerical gradient of F wrt each mu^(l) (finite differences), used
        for the generic Phase-2 update so the model stays faithful to the
        pseudocode without hand-deriving every partial derivative."""
        L = self.cfg.n_levels
        grads = []
        h = 1e-5
        base_mu = [m.copy() for m in self.mu]
        for l in range(L):
            g = np.zeros_like(self.mu[l])
            for i in range(len(self.mu[l])):
                self.mu[l][i] += h
                Fp = self.free_energy(o)
                self.mu[l][i] -= 2 * h
                Fm = self.free_energy(o)
                self.mu[l][i] += h
                g[i] = (Fp - Fm) / (2 * h)
            grads.append(g)
        self.mu = base_mu
        return grads

    # ---- Phase 2 ---------------------------------------------------------
    def phase2_minimise_free_energy(self, o: np.ndarray):
        cfg = self.cfg
        k = 0
        F_trace = [self.free_energy(o)]
        while k < cfg.T:
            grads = self.free_energy_grad_mu(o)
            grad_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
            if grad_norm <= cfg.tol:
                break
            for l in range(cfg.n_levels):
                self.mu[l] = self.mu[l] - cfg.eta * grads[l]
            F_trace.append(self.free_energy(o))
            k += 1
        return np.array(F_trace)

    # ---- Phase 3 -----------------------------------------------------------
    def phase3_snis_query_decode(self):
        cfg = self.cfg
        mu_top = self.mu[-1]               # top-level belief acts as phi_E(mu, c)
        Q = (self.W_Q.T @ mu_top).reshape(1, -1)          # (1, d_k)
        K = self.M                                        # (n_s, d_snis) treated as K
        A, weights = bqi_attention(Q, K[:, :cfg.d_k] if K.shape[1] >= cfg.d_k else K,
                                    self.M)
        mu_star_top = mu_top + cfg.alpha * A.flatten()[: len(mu_top)]
        perception = mu_star_top  # Phi_D taken as identity readout for demonstration
        return mu_star_top, perception, weights

    # ---- Full algorithm ---------------------------------------------------
    def run(self, o: np.ndarray):
        """Executes Algorithm 1 end-to-end and returns a result dict."""
        eps = self.phase1_prediction_errors(o)
        F_trace = self.phase2_minimise_free_energy(o)
        mu_star, perception, attn_weights = self.phase3_snis_query_decode()
        return {
            "prediction_errors": eps,
            "free_energy_trace": F_trace,
            "mu_star": mu_star,
            "perception": perception,
            "attention_weights": attn_weights,
            "F_final": F_trace[-1],
            "n_iterations": len(F_trace) - 1,
        }
