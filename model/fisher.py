"""
Exact expected Fisher information for StateSpaceDMN (Q1_pipeline_model.md Sec 5.2).

All observations of one session form a Gaussian vector Y ~ N(mu(theta), Sigma(theta)):
the latent path is stationary with Cov(x_t, x_s) = F^(t-s) P (t >= s), and each
observation is a linear read-out of one x_t plus independent noise. Hence

    I_ij = d_i mu^T Sigma^-1 d_j mu + 1/2 tr(Sigma^-1 d_i Sigma Sigma^-1 d_j Sigma)

exactly (no simulated data, no Monte Carlo noise). Sessions with the same
design are independent and identically distributed, so N subjects give N * I.
Derivatives are central finite differences of (mu, Sigma) in the transformed
parameters used by the MLE (log for positive parameters).

Rothenberg (1971): in a regular model, theta is locally identifiable iff the
information matrix is non-singular; the null space of I gives the parameter
combinations that the data cannot distinguish (structural non-identifiability).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bqi.dmn_ode import StateSpaceDMN

# Observation configurations (Q1_pipeline_model.md Sec 5.1)
CONFIGS = {
    "a": {"y_A": False, "y_Phi": False, "rating": True},
    "b": {"y_A": True, "y_Phi": False, "rating": True},
    "c": {"y_A": False, "y_Phi": True, "rating": True},
    "d": {"y_A": True, "y_Phi": True, "rating": True},
}


@dataclass(frozen=True)
class Design:
    """One session's sampling design (identical for every subject)."""
    n_epochs: int
    dt: float                      # minutes per epoch
    probe_idx: tuple               # epochs with a rating
    config: str = "d"

    @classmethod
    def regular(cls, session_min: float, probe_every_min: float, epoch_s: float | None = None,
                config: str = "d") -> "Design":
        from model import config as CFG
        dt = (epoch_s if epoch_s is not None else CFG.EPOCH_S) / 60.0
        n = int(round(session_min / dt))
        k = max(1, int(round(probe_every_min / dt)))
        return cls(n, dt, tuple(range(k - 1, n, k)), config)


# =============================================================================
# Parameter transform (matches eeg.model_fit._pack/_unpack)
# =============================================================================
def to_theta(model: StateSpaceDMN, names) -> np.ndarray:
    return np.array([np.log(getattr(model, k)) if k in StateSpaceDMN.POSITIVE else getattr(model, k)
                     for k in names], dtype=float)


def from_theta(theta: np.ndarray, names, base: StateSpaceDMN) -> StateSpaceDMN:
    kw = {k: float(np.exp(v)) if k in StateSpaceDMN.POSITIVE else float(v) for k, v in zip(names, theta)}
    return base.with_params(**kw)


# =============================================================================
# Exact moments of one session's observation vector
# =============================================================================
def session_moments(model: StateSpaceDMN, design: Design) -> tuple[np.ndarray, np.ndarray]:
    """Mean vector and covariance matrix of all observations of one session.

    Observation order: for each epoch t, y_A (if observed), y_Phi (if observed),
    then the rating if t is a probe epoch.
    """
    cfg = CONFIGS[design.config]
    T = design.n_epochs
    F, _, _ = model.discretize(design.dt)
    mu = model.stationary_mean()
    P = model.stationary_cov()
    h = model.readout()

    Fk = np.empty((T, 2, 2))
    Fk[0] = np.eye(2)
    for k in range(1, T):
        Fk[k] = F @ Fk[k - 1]
    C = Fk @ P                                    # C[k] = Cov(x_{s+k}, x_s)
    lag = np.subtract.outer(np.arange(T), np.arange(T))
    blocks = C[np.abs(lag)]
    blocks = np.where((lag >= 0)[..., None, None], blocks, np.swapaxes(blocks, -1, -2))
    Sx = blocks.transpose(0, 2, 1, 3).reshape(2 * T, 2 * T)

    probes = set(design.probe_idx)
    rows, offs, noise = [], [], []
    for t in range(T):
        if cfg["y_A"]:
            r = np.zeros(2 * T); r[2 * t] = 1.0
            rows.append(r); offs.append(0.0); noise.append(model.r_A ** 2)
        if cfg["y_Phi"]:
            r = np.zeros(2 * T); r[2 * t + 1] = 1.0
            rows.append(r); offs.append(0.0); noise.append(model.r_Phi ** 2)
        if cfg["rating"] and t in probes:
            r = np.zeros(2 * T); r[2 * t: 2 * t + 2] = h
            rows.append(r); offs.append(model.gamma0); noise.append(model.r_Q ** 2)
    S = np.array(rows)
    mean = S @ np.tile(mu, T) + np.array(offs)
    cov = S @ Sx @ S.T + np.diag(noise)
    return mean, 0.5 * (cov + cov.T)


def gaussian_loglik(model: StateSpaceDMN, design: Design, sessions: list[np.ndarray]) -> float:
    """Log-likelihood of stacked observation vectors (reference for the Kalman filter)."""
    m, S = session_moments(model, design)
    L = np.linalg.cholesky(S)
    ld = 2 * np.log(np.diag(L)).sum()
    out = 0.0
    for y in sessions:
        a = np.linalg.solve(L, y - m)
        out -= 0.5 * (len(y) * np.log(2 * np.pi) + ld + a @ a)
    return float(out)


def stack_observations(sim: dict, design: Design) -> np.ndarray:
    """Observation vector of one simulated session in `session_moments` order."""
    cfg = CONFIGS[design.config]
    probes = set(design.probe_idx)
    out = []
    for t in range(design.n_epochs):
        if cfg["y_A"]:
            out.append(sim["y"][t, 0])
        if cfg["y_Phi"]:
            out.append(sim["y"][t, 1])
        if cfg["rating"] and t in probes:
            out.append(sim["rating"][t])
    return np.array(out)


# =============================================================================
# Fisher information
# =============================================================================
def fisher_exact(model: StateSpaceDMN, design: Design, names=None, n_subjects: int = 1,
                 rel_step: float = 1e-5) -> np.ndarray:
    """Exact expected Fisher information of `n_subjects` independent sessions,
    w.r.t. the transformed parameters (log for positive ones)."""
    names = list(names or StateSpaceDMN.FREE)
    theta0 = to_theta(model, names)
    m0, S0 = session_moments(model, design)
    L = np.linalg.cholesky(S0)

    def Sinv_times(M):
        return np.linalg.solve(L.T, np.linalg.solve(L, M))

    dmu, A = [], []
    for i in range(len(names)):
        step = rel_step * max(1.0, abs(theta0[i]))
        tp, tm = theta0.copy(), theta0.copy()
        tp[i] += step
        tm[i] -= step
        mp, Sp = session_moments(from_theta(tp, names, model), design)
        mm, Sm = session_moments(from_theta(tm, names, model), design)
        dmu.append((mp - mm) / (2 * step))
        A.append(Sinv_times((Sp - Sm) / (2 * step)))   # Sigma^-1 dSigma_i
    dmu = np.array(dmu)
    W = Sinv_times(dmu.T)                               # Sigma^-1 dmu
    I = dmu @ W
    for i in range(len(names)):
        for j in range(i, len(names)):
            v = 0.5 * np.sum(A[i] * A[j].T)             # 1/2 tr(A_i A_j)
            I[i, j] += v
            if j != i:
                I[j, i] += v
    return n_subjects * 0.5 * (I + I.T)


def crb(I: np.ndarray, names, model: StateSpaceDMN) -> dict:
    """Cramer-Rao lower bound for every parameter.

    `se` is in the transformed scale; `rel_se` is the relative standard error
    (= se for log-parameters, se/|value| for linear ones, NaN if the value is 0).
    Non-invertible information returns inf.
    """
    names = list(names)
    try:
        cov = np.linalg.inv(I)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
        if not np.all(np.isfinite(se)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        se = np.full(len(names), np.inf)
    out = {}
    for k, s in zip(names, se):
        v = getattr(model, k)
        if k in StateSpaceDMN.POSITIVE:
            rel = s
        else:
            rel = s / abs(v) if abs(v) > 1e-12 else float("nan")
        out[k] = {"se": float(s), "rel_se": float(rel), "log_scale": k in StateSpaceDMN.POSITIVE}
    return out


def identifiability_spectrum(I: np.ndarray, names, tol: float = 1e-8) -> dict:
    """Rothenberg analysis of an information matrix.

    Parameters with zero information (do not enter the likelihood at all) are
    reported separately. The remaining matrix is scaled to unit diagonal; its
    eigenvalues below `tol` (relative to the largest) define non-identifiable
    directions, reported as parameter loadings.
    """
    names = list(names)
    d = np.diag(I).copy()
    absent = [k for k, v in zip(names, d) if v <= 1e-12 * max(d.max(), 1e-300)]
    keep = [i for i, k in enumerate(names) if k not in absent]
    sub = I[np.ix_(keep, keep)]
    s = np.sqrt(np.diag(sub))
    C = sub / np.outer(s, s)
    w, V = np.linalg.eigh(C)
    rel = w / w.max()
    null = []
    for j in np.flatnonzero(rel < tol):
        v = V[:, j]
        load = {names[keep[i]]: round(float(v[i]), 3) for i in np.argsort(-np.abs(v)) if abs(v[i]) > 0.05}
        null.append({"rel_eigenvalue": float(rel[j]), "direction": load})
    return {"absent": absent, "n_params": len(names), "rank": int(np.sum(rel >= tol)),
            "n_identifiable_candidates": len(keep), "min_rel_eigenvalue": float(rel.min()),
            "condition_number": float(1 / max(rel.min(), 1e-300)), "null_directions": null,
            "identified": [names[keep[i]] for i in range(len(keep))] if not null else None}
