"""
Joint ODE model: DMN activity, Phi, and Quality of Consciousness (QoC).
Paper Sec. 7.2, eq. (25)-(27), Table ode_params.

    dA_DMN/dt = -k_inh * A_DMN(t) + k_spont * xi(t)
    dPhi/dt   = k_int * (1 - A_DMN(t)/A_max) - k_d * Phi(t)
    dQoC/dt   = alpha * dPhi/dt - beta * dA_DMN/dt - gamma * QoC(t)

Steady states. With E[xi] = 0 the deterministic steady state is A* = 0,
Phi* = k_int / k_d and, because eq. (27) is driven only by the *derivatives*
of Phi and A_DMN, gamma * QoC* = 0 => QoC* = 0: QoC is a transient, never a
level. The paper's eq. (55)-(57) are kept as `steady_states_paper` for
comparison. `StateSpaceDMN` is the corrected model of Q1_pipeline.md Sec 7.2:
A_DMN is an Ornstein-Uhlenbeck process around a baseline A0 > 0 and QoC is an
algebraic read-out with a non-zero steady state.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

# Published parameter estimates, Table ode_params (Sec 7.2), used as defaults.
ODE_PARAMS_REFERENCE = {
    "k_inh": 0.043,     # DMN suppression rate, 1/min
    "k_spont": 0.027,   # spontaneous DMN activation
    "k_int": 0.081,     # Phi integration rate, 1/min
    "k_d": 0.034,       # Phi decay rate, 1/min
    "alpha": 2.34,      # Phi -> QoC gain
    "beta": 1.87,       # DMN -> QoC (negative) gain
    "gamma": 0.19,      # QoC decay rate
    "A_max": 1.0,
    "sigma_xi": 0.05,   # noise std for xi(t)
}


def steady_states(params: dict) -> dict:
    """Correct steady state of eq. (25)-(27) under zero-mean noise xi.

    Deterministic part: A* = 0, Phi* = (k_int/k_d)(1 - A*/A_max), QoC* = 0.
    Fluctuations: with xi white noise of intensity sigma_xi, A_DMN is an OU
    process with stationary variance k_spont^2 sigma_xi^2 / (2 k_inh).
    """
    k_inh, k_spont = params["k_inh"], params["k_spont"]
    k_int, k_d, A_max = params["k_int"], params["k_d"], params["A_max"]
    sigma_xi = params["sigma_xi"]
    A_star = 0.0
    return {
        "A_DMN_star": A_star,
        "Phi_star": (k_int / k_d) * (1 - A_star / A_max),
        "QoC_star": 0.0,
        "A_DMN_stationary_sd": float(k_spont * sigma_xi / np.sqrt(2 * k_inh)),
    }


def steady_states_paper(params: dict) -> dict:
    """Closed forms printed in the paper, eq. (55)-(57). They do not satisfy
    dA/dt = dPhi/dt = dQoC/dt = 0 for zero-mean xi (see `steady_states`)."""
    k_inh, k_spont = params["k_inh"], params["k_spont"]
    k_int, k_d = params["k_int"], params["k_d"]
    alpha, beta, gamma = params["alpha"], params["beta"], params["gamma"]
    A_max, sigma_xi = params["A_max"], params["sigma_xi"]

    A_dmn_star = (k_spont / k_inh) * np.sqrt(2 * np.pi * sigma_xi ** 2)
    Phi_star = (k_int / k_d) * (1 - A_dmn_star / A_max)
    QoC_star = (1 / gamma) * (
        alpha * k_int * (A_max - A_dmn_star) / (A_max * k_d)
        + beta * (k_spont / k_inh) * np.sqrt(2 * np.pi * sigma_xi ** 2)
    )
    return {"A_DMN_star": A_dmn_star, "Phi_star": Phi_star, "QoC_star": QoC_star}


def _rhs(t, y, params, xi_func, inh_multiplier):
    """Right-hand side of the 3-state ODE system. `inh_multiplier(t)` allows
    modelling an elevated k_inh for a "meditator" trajectory vs a control."""
    A_dmn, Phi, QoC = y
    k_inh = params["k_inh"] * inh_multiplier(t)
    k_spont, k_int, k_d = params["k_spont"], params["k_int"], params["k_d"]
    alpha, beta, gamma, A_max = params["alpha"], params["beta"], params["gamma"], params["A_max"]

    xi = xi_func(t)
    dA_dmn = -k_inh * A_dmn + k_spont * xi
    dPhi = k_int * (1 - A_dmn / A_max) - k_d * Phi
    dQoC = alpha * dPhi - beta * dA_dmn - gamma * QoC
    return [dA_dmn, dPhi, dQoC]


def simulate(params: dict, y0=(1.0, 0.0, 0.0), t_span=(0, 40), n_points=400,
             meditator: bool = False, seed: int = 0):
    """Integrate the ODE system with scipy.solve_ivp.

    meditator=True multiplies k_inh by ~5-10x over time (closed-loop DMN
    suppression from sustained practice), reproducing the qualitative shape
    of Fig. ode_sim (meditator trajectory reaching a low-DMN/high-Phi/high-QoC
    attractor faster than control).
    """
    rng = np.random.default_rng(seed)
    noise_trace = rng.normal(0, 1, 2000)
    t_grid = np.linspace(*t_span, 2000)

    def xi_func(t):
        idx = min(int(t / t_span[1] * (len(noise_trace) - 1)), len(noise_trace) - 1)
        return noise_trace[max(idx, 0)]

    if meditator:
        def inh_multiplier(t):
            return 1.0 + 9.0 * (1 - np.exp(-t / 8.0))  # ramps k_inh up ~10x
    else:
        def inh_multiplier(t):
            return 1.0

    t_eval = np.linspace(*t_span, n_points)
    sol = solve_ivp(_rhs, t_span, y0, t_eval=t_eval, args=(params, xi_func, inh_multiplier),
                     method="RK45", max_step=0.1)
    return {"t": sol.t, "A_DMN": sol.y[0], "Phi": sol.y[1], "QoC": sol.y[2]}


def generate_ode_param_dataset(n_subjects: int = 200, seed: int = 0):
    """Synthetic *individual-subject* parameter dataset: sample each ODE
    parameter from a Normal distribution centred on the published estimate
    with the published 95% CI implying its standard error (Table ode_params),
    producing a new per-subject dataset for downstream statistical testing
    (e.g. recovering the population mean via the sample mean)."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    ci_halfwidth = {  # approx (upper-lower)/2 from Table ode_params
        "k_inh": 0.005, "k_spont": 0.003, "k_int": 0.009, "k_d": 0.004,
        "alpha": 0.35, "beta": 0.41, "gamma": 0.06,
    }
    rows = []
    for s in range(n_subjects):
        row = {"subject": s}
        for p, ref in ODE_PARAMS_REFERENCE.items():
            if p in ci_halfwidth:
                se = ci_halfwidth[p] / 1.96
                row[p] = float(rng.normal(ref, se))
            else:
                row[p] = ref
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Corrected stochastic model (Q1_pipeline.md Sec 7.2)
# =============================================================================
@dataclass
class StateSpaceDMN:
    """Linear-Gaussian state-space model with latent x = [A_DMN, Phi]:

        dA   = -k_inh (A - A0) dt + sigma_A dW_A
        dPhi = [k_int (1 - A / A_max) - k_d Phi] dt + sigma_Phi dW_Phi
        QoC  = alpha Phi - beta A + gamma0                (algebraic, QoC* != 0)

    Observations (Sec 7.2): an EEG feature vector every epoch,
        y_t = x_t + eta,  eta ~ N(0, diag(r_A^2, r_Phi^2)),
    and a self-report at probe times, rating = QoC + eps, eps ~ N(0, r_Q^2).
    Rates are per minute.
    """
    k_inh: float = 0.20
    A0: float = 0.60
    sigma_A: float = 0.10
    k_int: float = 0.081
    k_d: float = 0.034
    sigma_Phi: float = 0.05
    alpha: float = 2.34
    beta: float = 1.87
    gamma0: float = 0.0
    r_A: float = 0.10
    r_Phi: float = 0.10
    r_Q: float = 0.30
    A_max: float = 1.0

    FREE = ("k_inh", "A0", "sigma_A", "k_int", "k_d", "sigma_Phi",
            "alpha", "beta", "gamma0", "r_A", "r_Phi", "r_Q")
    POSITIVE = ("k_inh", "sigma_A", "k_int", "k_d", "sigma_Phi", "r_A", "r_Phi", "r_Q")

    # ---- continuous-time system  dx = (M x + u) dt + G dW -------------------
    def drift(self):
        M = np.array([[-self.k_inh, 0.0],
                      [-self.k_int / self.A_max, -self.k_d]])
        u = np.array([self.k_inh * self.A0, self.k_int])
        return M, u

    def stationary_mean(self) -> np.ndarray:
        M, u = self.drift()
        return -np.linalg.solve(M, u)

    def steady_states(self) -> dict:
        A, Phi = self.stationary_mean()
        return {"A_DMN_star": float(A), "Phi_star": float(Phi),
                "QoC_star": float(self.alpha * Phi - self.beta * A + self.gamma0)}

    def discretize(self, dt: float):
        """Exact discretisation: x_{k+1} = F x_k + c + w, w ~ N(0, Q)
        (Van Loan's method for Q)."""
        M, u = self.drift()
        n = 2
        F = expm(M * dt)
        c = np.linalg.solve(M, (F - np.eye(n)) @ u)
        GG = np.diag([self.sigma_A ** 2, self.sigma_Phi ** 2])
        V = np.zeros((2 * n, 2 * n))
        V[:n, :n] = -M
        V[:n, n:] = GG
        V[n:, n:] = M.T
        E = expm(V * dt)
        Q = E[n:, n:].T @ E[:n, n:]
        return F, c, 0.5 * (Q + Q.T)

    def stationary_cov(self) -> np.ndarray:
        from scipy.linalg import solve_continuous_lyapunov
        M, _ = self.drift()
        return solve_continuous_lyapunov(M, -np.diag([self.sigma_A ** 2, self.sigma_Phi ** 2]))

    def readout(self) -> np.ndarray:
        return np.array([-self.beta, self.alpha])

    def probe_indices(self, n_epochs: int, probe_every: int | None) -> np.ndarray:
        """Epochs at which a rating is collected for a regular schedule."""
        if not probe_every:
            return np.zeros(0, dtype=int)
        return np.arange(probe_every - 1, n_epochs, probe_every)

    def simulate(self, n_epochs: int, dt: float, probe_every: int | None = 12,
                 x0: np.ndarray | None = None, seed: int = 0, probe_idx=None) -> dict:
        """Simulate one session: latent path, EEG observations every epoch and
        ratings at probe epochs (NaN elsewhere).

        The initial state is drawn from the stationary distribution
        N(stationary_mean, stationary_cov) unless `x0` is given, which is the
        assumption of the Kalman likelihood and of the exact Fisher information.
        `probe_idx` (explicit epoch indices) overrides the regular `probe_every`
        schedule.
        """
        rng = np.random.default_rng(seed)
        F, c, Q = self.discretize(dt)
        L = np.linalg.cholesky(Q + 1e-15 * np.eye(2))
        x = np.empty((n_epochs, 2))
        if x0 is None:
            P0 = np.linalg.cholesky(self.stationary_cov() + 1e-15 * np.eye(2))
            x[0] = self.stationary_mean() + P0 @ rng.normal(size=2)
        else:
            x[0] = x0
        for k in range(1, n_epochs):
            x[k] = F @ x[k - 1] + c + L @ rng.normal(size=2)
        y = x + rng.normal(size=x.shape) * np.array([self.r_A, self.r_Phi])
        qoc = x @ self.readout() + self.gamma0
        rating = np.full(n_epochs, np.nan)
        idx = (np.asarray(probe_idx, dtype=int) if probe_idx is not None
               else self.probe_indices(n_epochs, probe_every))
        if len(idx):
            rating[idx] = qoc[idx] + rng.normal(0, self.r_Q, len(idx))
        return {"t": np.arange(n_epochs) * dt, "x": x, "y": y, "QoC": qoc, "rating": rating}

    def with_params(self, **kw) -> "StateSpaceDMN":
        d = {k: getattr(self, k) for k in self.FREE + ("A_max",)}
        d.update(kw)
        return StateSpaceDMN(**d)
