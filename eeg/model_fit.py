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
    pos = type(model).POSITIVE
    return np.array([np.log(getattr(model, k)) if k in pos else getattr(model, k) for k in names])


def _unpack(theta: np.ndarray, names, base: StateSpaceDMN) -> StateSpaceDMN:
    pos = type(base).POSITIVE
    kw = {k: float(np.exp(v)) if k in pos else float(v) for k, v in zip(names, theta)}
    return base.with_params(**kw)


# =============================================================================
# Irregular sessions (real data): arbitrary observation times, missing epochs,
# optionally correlated EEG noise. Session dict: t (minutes, EEG epoch end
# times, sorted), y (T x 2, NaN = missing), t_probe (minutes), rating.
# =============================================================================
def _events(session: dict):
    t = np.asarray(session["t"], float)
    y = np.asarray(session["y"], float)
    tp = np.asarray(session.get("t_probe", []), float)
    r = np.asarray(session.get("rating", []), float)
    ev = [(float(t[i]), 0, float(y[i, 0]), float(y[i, 1])) for i in range(len(t))]
    ev += [(float(tp[k]), 1, float(r[k]), float("nan")) for k in range(len(tp))]
    ev.sort(key=lambda e: (e[0], e[1]))
    return ev


def kalman_filter_irregular(model: StateSpaceDMN, session: dict, return_innovations: bool = False,
                            use_rating: bool = True, use_eeg: bool = True, until: float | None = None):
    """Exact Kalman filter on one irregular session. Returns the log-likelihood
    and, if requested, standardised innovations per channel and the predictive
    (mean, var) of every rating observation (one-step-ahead, given everything
    before it in time)."""
    M, _ = model.drift()
    mu0 = model.stationary_mean()
    P0 = model.stationary_cov()
    h0, h1 = model.readout()
    g0 = model.gamma0
    R = model.obs_cov() if hasattr(model, "obs_cov") else np.diag([model.r_A ** 2, model.r_Phi ** 2])
    rA2, rP2, rAP = R[0, 0], R[1, 1], R[0, 1]
    rQ2 = model.r_Q ** 2
    cache: dict = {}

    def step_mats(dtm: float):
        key = round(dtm, 7)
        if key not in cache:
            F, c, Q = model.discretize(dtm)
            cache[key] = (F[0, 0], F[0, 1], F[1, 0], F[1, 1], c[0], c[1], Q[0, 0], Q[0, 1], Q[1, 1])
        return cache[key]

    m0, m1 = float(mu0[0]), float(mu0[1])
    p00, p01, p11 = float(P0[0, 0]), float(P0[0, 1]), float(P0[1, 1])
    ll = 0.0
    t_prev = None
    innov = {"A": [], "Phi": [], "Q": []}
    pred_rating = []
    for (t, kind, v0, v1) in _events(session):
        if until is not None and t > until:
            break
        if t_prev is not None and t > t_prev:
            f00, f01, f10, f11, c0, c1, q00, q01, q11 = step_mats(t - t_prev)
            m0, m1 = f00 * m0 + f01 * m1 + c0, f10 * m0 + f11 * m1 + c1
            a00, a01 = f00 * p00 + f01 * p01, f00 * p01 + f01 * p11
            a10, a11 = f10 * p00 + f11 * p01, f10 * p01 + f11 * p11
            p00, p01, p11 = a00 * f00 + a01 * f01 + q00, a00 * f10 + a01 * f11 + q01, a10 * f10 + a11 * f11 + q11
        t_prev = t
        if kind == 0:
            if not use_eeg:
                continue
            hasA, hasP = v0 == v0, v1 == v1
            if hasA and hasP and rAP != 0.0:
                # joint 2-D update (correlated EEG noise)
                s00, s01, s11 = p00 + rA2, p01 + rAP, p11 + rP2
                det = s00 * s11 - s01 * s01
                if det <= 0:
                    return (-np.inf, None) if return_innovations else -np.inf
                i00, i01, i11 = s11 / det, -s01 / det, s00 / det
                e0, e1 = v0 - m0, v1 - m1
                quad = e0 * (i00 * e0 + i01 * e1) + e1 * (i01 * e0 + i11 * e1)
                ll -= 0.5 * (2 * LOG2PI + np.log(det) + quad)
                if return_innovations:
                    innov["A"].append(e0 / np.sqrt(s00)); innov["Phi"].append(e1 / np.sqrt(s11))
                k00, k01 = p00 * i00 + p01 * i01, p00 * i01 + p01 * i11
                k10, k11 = p01 * i00 + p11 * i01, p01 * i01 + p11 * i11
                m0 += k00 * e0 + k01 * e1
                m1 += k10 * e0 + k11 * e1
                n00 = p00 - (k00 * p00 + k01 * p01)
                n01 = p01 - (k00 * p01 + k01 * p11)
                n11 = p11 - (k10 * p01 + k11 * p11)
                p00, p01, p11 = n00, n01, n11
            else:
                for (hh0, hh1, r2, v, name) in ((1.0, 0.0, rA2, v0, "A"), (0.0, 1.0, rP2, v1, "Phi")):
                    if v != v:
                        continue
                    ph0, ph1 = p00 * hh0 + p01 * hh1, p01 * hh0 + p11 * hh1
                    S = hh0 * ph0 + hh1 * ph1 + r2
                    if S <= 0:
                        return (-np.inf, None) if return_innovations else -np.inf
                    e = v - (hh0 * m0 + hh1 * m1)
                    ll -= 0.5 * (LOG2PI + np.log(S) + e * e / S)
                    if return_innovations:
                        innov[name].append(e / np.sqrt(S))
                    k0, k1 = ph0 / S, ph1 / S
                    m0 += k0 * e; m1 += k1 * e
                    p00 -= k0 * ph0; p01 -= k0 * ph1; p11 -= k1 * ph1
        else:
            ph0, ph1 = p00 * h0 + p01 * h1, p01 * h0 + p11 * h1
            S = h0 * ph0 + h1 * ph1 + rQ2
            pred_mean = h0 * m0 + h1 * m1 + g0
            if return_innovations:
                pred_rating.append((t, pred_mean, S, v0))
            if not use_rating or v0 != v0:
                continue
            if S <= 0:
                return (-np.inf, None) if return_innovations else -np.inf
            e = v0 - pred_mean
            ll -= 0.5 * (LOG2PI + np.log(S) + e * e / S)
            if return_innovations:
                innov["Q"].append(e / np.sqrt(S))
            k0, k1 = ph0 / S, ph1 / S
            m0 += k0 * e; m1 += k1 * e
            p00 -= k0 * ph0; p01 -= k0 * ph1; p11 -= k1 * ph1
    if return_innovations:
        return float(ll), {"innov": {k: np.array(v) for k, v in innov.items()},
                           "pred_rating": np.array(pred_rating, dtype=float).reshape(-1, 4),
                           "state": (m0, m1), "cov": (p00, p01, p11)}
    return float(ll)


def kalman_loglik_irregular(model: StateSpaceDMN, sessions: list[dict], dt: float | None = None) -> float:
    """Sum of exact log-likelihoods over irregular sessions (`dt` ignored; kept
    for signature compatibility with `kalman_loglik`)."""
    ll = 0.0
    for s in sessions:
        v = kalman_filter_irregular(model, s)
        if not np.isfinite(v):
            return -np.inf
        ll += v
    return float(ll)


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
        fixed: dict | None = None, n_starts: int = 4, rng=None, loglik=None, starts=None) -> dict:
    """MLE over `names` (default: all free parameters minus `fixed`), L-BFGS-B
    from the initial guess plus jittered restarts.

    `loglik` defaults to the fixed-step `kalman_loglik`; pass
    `kalman_loglik_irregular` for real sessions. `starts` (optional list of
    parameter dicts) replaces the jittered restarts by explicit starting points.
    """
    rng = np.random.default_rng(rng)
    base = (init or StateSpaceDMN()).with_params(**(fixed or {}))
    names = [k for k in (names or type(base).FREE) if k not in (fixed or {})]
    loglik = loglik or kalman_loglik

    def nll(theta):
        try:
            v = -loglik(_unpack(theta, names, base), sessions, dt)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return 1e12
        return v if np.isfinite(v) else 1e12

    theta0 = _pack(base, names)
    if starts is not None:
        start_list = [_pack(base.with_params(**{k: v for k, v in s.items() if k in names}), names) for s in starts]
    else:
        start_list = [theta0 if k == 0 else theta0 + rng.normal(0, 0.3, len(theta0)) for k in range(n_starts)]
    best = None
    for start in start_list:
        res = minimize(nll, start, method="L-BFGS-B", options={"maxiter": 500})
        if best is None or res.fun < best.fun:
            best = res
    model = _unpack(best.x, names, base)
    return {"model": model, "params": {k: getattr(model, k) for k in names}, "loglik": -float(best.fun),
            "success": bool(best.success), "names": names, "theta": best.x, "loglik_fn": loglik}


def profile_likelihood(sessions: list[dict], dt: float, fit_result: dict, param: str, grid: np.ndarray,
                       n_starts: int = 1) -> np.ndarray:
    """Profile log-likelihood of `param` over `grid` (other free parameters refit).
    A flat profile (drop < 1.92 = chi2(1)/2 at 95% across the grid) means the
    parameter is not identifiable from these data."""
    model = fit_result["model"]
    others = [k for k in fit_result["names"] if k != param]
    out = []
    for g in grid:
        r = fit(sessions, dt, init=model, names=others, fixed={param: float(g)}, n_starts=n_starts, rng=0,
                loglik=fit_result.get("loglik_fn"))
        out.append(r["loglik"])
    return np.array(out)
