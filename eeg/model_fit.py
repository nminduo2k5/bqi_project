"""
Maximum-likelihood fit of the corrected DMN-Phi-QoC state-space model
(`bqi.dmn_ode.StateSpaceDMN`, Q1_pipeline.md Sec 7.2) with a Kalman filter,
plus profile likelihoods for identifiability (gate G3).

A session is a dict with `y` (T x 2 EEG-derived observations of [A, Phi],
NaN allowed) and `rating` (length T, NaN except at probe epochs). Sessions
share parameters; they are independent given the parameters.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from bqi.dmn_ode import StateSpaceDMN

LOG2PI = np.log(2 * np.pi)


def _pack(model: StateSpaceDMN, names) -> np.ndarray:
    return np.array([np.log(getattr(model, k)) if k in StateSpaceDMN.POSITIVE else getattr(model, k)
                     for k in names])


def _unpack(theta: np.ndarray, names, base: StateSpaceDMN) -> StateSpaceDMN:
    kw = {k: float(np.exp(v)) if k in StateSpaceDMN.POSITIVE else float(v) for k, v in zip(names, theta)}
    return base.with_params(**kw)


def kalman_loglik(model: StateSpaceDMN, sessions: list[dict], dt: float) -> float:
    """Exact Gaussian log-likelihood of all sessions.

    Observation noise is independent across the three channels (y_A, y_Phi,
    rating), so each scalar observation is assimilated sequentially; the 2-D
    state lets the whole filter run on Python floats (fast enough for MLE).
    Missing values (NaN) are skipped.
    """
    F, c, Q = model.discretize(dt)
    f00, f01, f10, f11 = F[0, 0], F[0, 1], F[1, 0], F[1, 1]
    c0, c1 = c
    q00, q01, q11 = Q[0, 0], Q[0, 1], Q[1, 1]
    hq0, hq1 = model.readout()
    g0 = model.gamma0
    obs_rows = ((1.0, 0.0, 0.0, model.r_A ** 2), (0.0, 1.0, 0.0, model.r_Phi ** 2),
                (hq0, hq1, g0, model.r_Q ** 2))
    m_init = model.stationary_mean()
    P_init = model.stationary_cov()
    log2pi = LOG2PI
    ll = 0.0
    for s in sessions:
        y = np.asarray(s["y"], float)
        cols = (y[:, 0].tolist(), y[:, 1].tolist(), np.asarray(s["rating"], float).tolist())
        m0, m1 = float(m_init[0]), float(m_init[1])
        p00, p01, p11 = float(P_init[0, 0]), float(P_init[0, 1]), float(P_init[1, 1])
        for t in range(len(cols[0])):
            if t > 0:
                m0, m1 = f00 * m0 + f01 * m1 + c0, f10 * m0 + f11 * m1 + c1
                a00 = f00 * p00 + f01 * p01
                a01 = f00 * p01 + f01 * p11
                a10 = f10 * p00 + f11 * p01
                a11 = f10 * p01 + f11 * p11
                p00 = a00 * f00 + a01 * f01 + q00
                p01 = a00 * f10 + a01 * f11 + q01
                p11 = a10 * f10 + a11 * f11 + q11
            for (h0, h1, off, r), col in zip(obs_rows, cols):
                v = col[t]
                if v != v:  # NaN
                    continue
                ph0 = p00 * h0 + p01 * h1
                ph1 = p01 * h0 + p11 * h1
                S = h0 * ph0 + h1 * ph1 + r
                if S <= 0:
                    return -np.inf
                innov = v - (h0 * m0 + h1 * m1 + off)
                ll -= 0.5 * (log2pi + np.log(S) + innov * innov / S)
                k0, k1 = ph0 / S, ph1 / S
                m0 += k0 * innov
                m1 += k1 * innov
                p00 -= k0 * ph0
                p01 -= k0 * ph1
                p11 -= k1 * ph1
    return float(ll)


def kalman_loglik_dense(model: StateSpaceDMN, sessions: list[dict], dt: float) -> float:
    """Reference implementation with joint (matrix) updates; used in tests."""
    F, c, Q = model.discretize(dt)
    h_q = model.readout()
    R = [model.r_A ** 2, model.r_Phi ** 2, model.r_Q ** 2]
    m0, P0 = model.stationary_mean(), model.stationary_cov()
    ll = 0.0
    for s in sessions:
        y, rating = np.asarray(s["y"], float), np.asarray(s["rating"], float)
        m, P = m0.copy(), P0.copy()
        for t in range(len(y)):
            if t > 0:
                m, P = F @ m + c, F @ P @ F.T + Q
            rows, obs, noise, offs = [], [], [], []
            for j, (h, val, off) in enumerate(((np.eye(2)[0], y[t, 0], 0.0), (np.eye(2)[1], y[t, 1], 0.0),
                                               (h_q, rating[t], model.gamma0))):
                if np.isfinite(val):
                    rows.append(h); obs.append(val); noise.append(R[j]); offs.append(off)
            if not rows:
                continue
            H = np.array(rows)
            v = np.array(obs) - (H @ m + np.array(offs))
            S = H @ P @ H.T + np.diag(noise)
            ll -= 0.5 * (len(v) * LOG2PI + np.linalg.slogdet(S)[1] + v @ np.linalg.solve(S, v))
            K = np.linalg.solve(S, H @ P).T
            m, P = m + K @ v, P - K @ H @ P
    return float(ll)


def fit(sessions: list[dict], dt: float, init: StateSpaceDMN | None = None, names=None,
        fixed: dict | None = None, n_starts: int = 4, rng=None) -> dict:
    """MLE over `names` (default: all free parameters minus `fixed`), L-BFGS-B
    from the initial guess plus jittered restarts."""
    rng = np.random.default_rng(rng)
    base = (init or StateSpaceDMN()).with_params(**(fixed or {}))
    names = [k for k in (names or StateSpaceDMN.FREE) if k not in (fixed or {})]

    def nll(theta):
        try:
            v = -kalman_loglik(_unpack(theta, names, base), sessions, dt)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return 1e12
        return v if np.isfinite(v) else 1e12

    theta0 = _pack(base, names)
    best = None
    for k in range(n_starts):
        start = theta0 if k == 0 else theta0 + rng.normal(0, 0.3, len(theta0))
        res = minimize(nll, start, method="L-BFGS-B", options={"maxiter": 500})
        if best is None or res.fun < best.fun:
            best = res
    model = _unpack(best.x, names, base)
    return {"model": model, "params": {k: getattr(model, k) for k in names}, "loglik": -float(best.fun),
            "success": bool(best.success), "names": names, "theta": best.x}


def profile_likelihood(sessions: list[dict], dt: float, fit_result: dict, param: str, grid: np.ndarray,
                       n_starts: int = 1) -> np.ndarray:
    """Profile log-likelihood of `param` over `grid` (other free parameters refit).
    A flat profile (drop < 1.92 = chi2(1)/2 at 95% across the grid) means the
    parameter is not identifiable from these data."""
    model = fit_result["model"]
    others = [k for k in fit_result["names"] if k != param]
    out = []
    for g in grid:
        r = fit(sessions, dt, init=model, names=others, fixed={param: float(g)}, n_starts=n_starts, rng=0)
        out.append(r["loglik"])
    return np.array(out)
