"""
Bayesian Predictive Coding (paper Sec. 3.1).

Implements:
  * Single-level variational free energy, eq. (2):
        F(mu) = KL[q(s;mu) || p(s)] - E_q[ln p(o|s)]
  * Free-energy gradient under a Laplace approximation (Proposition, eq. 3).
  * Hierarchical BPC recurrence, eq. (5)-(7):
        o_hat^(l) = g^(l)(mu^(l))
        eps^(l)   = Pi^(l) (mu^(l) - f^(l)(mu^(l+1)))
        mu_dot^(l) = f_mu^(l)^T eps^(l) - eps^(l-1)
  * BQI Convergence Theorem, eq. (15)-(16): exponential convergence of the
    gradient flow when F is strongly convex.

All models here use *linear-Gaussian* f and g so that F is quadratic, which
lets us verify the exponential-convergence guarantee in closed form and with
gradient descent.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


def free_energy(mu: np.ndarray, o: np.ndarray, prior_mean: np.ndarray,
                 Pi_s: np.ndarray, Pi_o: np.ndarray, g: np.ndarray) -> float:
    """Single-level variational free energy F(mu), eq. (2), Laplace/Gaussian case.

    q(s;mu) = N(mu, Pi_s^-1); p(s) = N(prior_mean, Pi_s^-1) (same precision,
    complexity term becomes a quadratic Mahalanobis distance); observation
    model o_hat = g @ mu (linear g), noise precision Pi_o.

    Complexity = 0.5 (mu-prior)^T Pi_s (mu-prior)
    Accuracy   = -0.5 (o - g mu)^T Pi_o (o - g mu)   [dropped from -E_q ln p(o|s)]
    """
    complexity = 0.5 * (mu - prior_mean) @ Pi_s @ (mu - prior_mean)
    resid = o - g @ mu
    accuracy_term = 0.5 * resid @ Pi_o @ resid  # this is -E_q[ln p(o|s)] up to a constant
    return float(complexity + accuracy_term)


def free_energy_gradient(mu: np.ndarray, o: np.ndarray, prior_mean: np.ndarray,
                          Pi_s: np.ndarray, Pi_o: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Gradient of F wrt mu (Proposition, eq. 3), linear-Gaussian special case:

        grad F = Pi_s (mu - prior_mean) - g^T Pi_o (o - g mu)
    """
    return Pi_s @ (mu - prior_mean) - g.T @ Pi_o @ (o - g @ mu)


def gradient_descent_free_energy(mu0, o, prior_mean, Pi_s, Pi_o, g,
                                  eta=0.05, n_iter=200):
    """Run gradient flow mu_dot = -grad F(mu) via discrete steps and track F(t).

    Returns (mu_final, F_trace) where F_trace lets us check the exponential
    decay bound of the BQI Convergence Theorem, eq. (15).
    """
    mu = mu0.copy()
    F_trace = [free_energy(mu, o, prior_mean, Pi_s, Pi_o, g)]
    for _ in range(n_iter):
        grad = free_energy_gradient(mu, o, prior_mean, Pi_s, Pi_o, g)
        mu = mu - eta * grad
        F_trace.append(free_energy(mu, o, prior_mean, Pi_s, Pi_o, g))
    return mu, np.array(F_trace)


def theoretical_convergence_bound(F0, Fstar, lambda_min, t):
    """Eq. (15): F(mu(t)) - F* <= (F(mu0) - F*) exp(-2 lambda_min t)."""
    return (F0 - Fstar) * np.exp(-2.0 * lambda_min * t)


@dataclass
class HierarchicalBPC:
    """Hierarchical Bayesian Predictive Coding, eq. (5)-(7), Sec 3.1.2.

    Level 0 is sensory input (mu^(0) = o, clamped). Levels 1..L hold beliefs.
    f^(l), g^(l) are linear maps (weight matrices) mapping level l+1 -> l.
    Pi^(l) are precision (inverse-covariance) matrices at each level.
    """
    n_levels: int
    dims: list  # dims[l] = dimensionality of level l, l=0..L
    seed: int = 0

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        L = self.n_levels
        assert len(self.dims) == L + 1, "dims must have length n_levels+1 (level 0..L)"
        # generative (top-down) weights f^(l): level l+1 -> level l, for l=1..L-1
        # (level L has a flat prior, so there is no f^(L))
        self.f = [rng.normal(0, 0.5, size=(self.dims[l], self.dims[l + 1]))
                  for l in range(1, L)]
        # observation weights g^(l) = identity-like (predicts o^(l) from mu^(l))
        self.g = [np.eye(self.dims[l]) for l in range(1, L + 1)]
        # precisions per level (diagonal, positive)
        self.Pi = [np.diag(rng.uniform(0.5, 2.0, size=self.dims[l])) for l in range(1, L + 1)]
        self.mu = [rng.normal(0, 0.1, size=self.dims[l]) for l in range(1, L + 1)]

    def step(self, o: np.ndarray, eta: float = 0.05):
        """One Euler step of the belief-update recurrence eq. (7) for all levels.

        eps^(l)    = Pi^(l) (mu^(l) - f^(l)(mu^(l+1)))     [top level predicts from a flat prior, mu^(L+1)=0]
        mu_dot^(l) = f^(l-1)_mu^T eps^(l-1)  -  eps^(l)     [top-down prediction credit minus own error]
        with mu^(0) = o clamped (boundary condition).
        """
        L = self.n_levels
        mu_full = [o] + self.mu + [np.zeros(1)]  # sentinel for level L+1 (flat prior)

        # prediction errors eps^(l) for l = 1..L
        eps = []
        for l in range(1, L + 1):
            if l < L:
                pred = self.f[l - 1] @ mu_full[l + 1]
            else:
                pred = np.zeros(self.dims[l])  # flat prior at the top, eps^(L+1) = 0
            eps.append(self.Pi[l - 1] @ (mu_full[l] - pred))

        # belief update: mu_dot^(l) = f^(l-1)_mu^T eps^(l-1) - eps^(l), l=1..L
        new_mu = []
        for l in range(1, L + 1):
            if l == 1:
                credit = np.zeros(self.dims[l])  # eps^(0) undefined (level 0 is clamped data)
            else:
                credit = self.f[l - 2].T @ eps[l - 2]  # f^(l-1)_mu^T eps^(l-1)
            mu_dot = credit - eps[l - 1]
            new_mu.append(self.mu[l - 1] + eta * mu_dot)
        self.mu = new_mu
        return eps

    def run(self, o: np.ndarray, n_steps: int = 100, eta: float = 0.05):
        """Run n_steps of belief updating; return trace of total prediction-error energy."""
        energies = []
        for _ in range(n_steps):
            eps = self.step(o, eta=eta)
            energy = float(sum(np.sum(e ** 2) for e in eps))
            energies.append(energy)
        return np.array(energies)


def active_inference_expected_free_energy(q_o, q_s, p_joint_log):
    """Expected free energy G(pi), eq. (8) (Monte-Carlo estimate).

    q_o, q_s: samples (n_samples, dim) from q(o|pi), q(s|pi)
    p_joint_log: callable(o, s) -> log p(o,s)
    """
    n = q_o.shape[0]
    # ln q(s|pi): approximate via Gaussian KDE-free plug-in using sample covariance
    mean_s = q_s.mean(axis=0)
    cov_s = np.cov(q_s.T) + 1e-6 * np.eye(q_s.shape[1])
    inv_cov = np.linalg.inv(cov_s)
    G = 0.0
    for i in range(n):
        diff = q_s[i] - mean_s
        ln_q = -0.5 * diff @ inv_cov @ diff
        ln_p = p_joint_log(q_o[i], q_s[i])
        G += ln_q - ln_p
    return G / n
