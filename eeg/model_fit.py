"""
Maximum-likelihood fit of the corrected DMN-Phi-QoC state-space model
(`bqi.dmn_ode.StateSpaceDMN`, Q1_pipeline.md Sec 7.2) with a Kalman filter,
plus profile likelihoods for identifiability (gate G3).

A session is a dict with `y` (T x 2 EEG-derived observations of [A, Phi],
NaN allowed) and `rating` (length T, NaN except at probe epochs). Sessions
share parameters; they are independent given the parameters.
"""
from __future__ import annotations

import math
import time
from operator import itemgetter
from types import SimpleNamespace

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
    """Observation events of one session sorted by (time, kind): kind 0 = EEG
    epoch (v0, v1 = y_A, y_Phi), kind 1 = probe (v0 = rating)."""
    t = np.asarray(session["t"], float)
    y = np.asarray(session["y"], float)
    tp = np.asarray(session.get("t_probe", []), float)
    r = np.asarray(session.get("rating", []), float)
    nan = float("nan")
    ev = list(zip(t.tolist(), [0] * len(t), y[:, 0].tolist(), y[:, 1].tolist()))
    ev += list(zip(tp.tolist(), [1] * len(tp), r.tolist(), [nan] * len(tp)))
    ev.sort(key=itemgetter(0, 1))
    return ev


def _model_consts(model: StateSpaceDMN) -> tuple:
    """Initial state, read-out and noise variances as Python floats (shared by
    every session of one likelihood evaluation)."""
    mu0 = model.stationary_mean()
    P0 = model.stationary_cov()
    h0, h1 = model.readout()
    R = model.obs_cov() if hasattr(model, "obs_cov") else np.diag([model.r_A ** 2, model.r_Phi ** 2])
    return (float(mu0[0]), float(mu0[1]), float(P0[0, 0]), float(P0[0, 1]), float(P0[1, 1]),
            float(h0), float(h1), float(model.gamma0), float(R[0, 0]), float(R[1, 1]), float(R[0, 1]),
            float(model.r_Q ** 2))


def _step_table(model: StateSpaceDMN, event_lists) -> dict:
    """Discretisation (F, c, Q as 9 floats) of every distinct step length in the
    event lists, keyed by the step rounded to 1e-7 min. Observation times are
    rounded to RD.time_resolution_s, so ~100 distinct steps cover all sessions;
    they are computed in one stacked call instead of one expm pair per session
    and step."""
    first: dict = {}
    for ev in event_lists:
        t_prev = None
        for e in ev:
            t = e[0]
            if t_prev is not None and t > t_prev:
                d = t - t_prev
                k = round(d, 7)
                if k not in first:
                    first[k] = d
            t_prev = t
    if not first:
        return {}
    keys = list(first)
    F, c, Q = model.discretize_many([first[k] for k in keys])
    return {k: (float(F[i, 0, 0]), float(F[i, 0, 1]), float(F[i, 1, 0]), float(F[i, 1, 1]), float(c[i, 0]),
                float(c[i, 1]), float(Q[i, 0, 0]), float(Q[i, 0, 1]), float(Q[i, 1, 1])) for i, k in enumerate(keys)}


def kalman_filter_irregular(model: StateSpaceDMN, session: dict, return_innovations: bool = False,
                            use_rating: bool = True, use_eeg: bool = True, until: float | None = None,
                            _ev=None, _steps=None, _consts=None):
    """Exact Kalman filter on one irregular session. Returns the log-likelihood
    and, if requested, standardised innovations per channel and the predictive
    (mean, var) of every rating observation (one-step-ahead, given everything
    before it in time). `_ev`, `_steps`, `_consts` are precomputed by
    `kalman_loglik_irregular` to share work across sessions."""
    ev = _ev if _ev is not None else _events(session)
    steps = _steps if _steps is not None else _step_table(model, [ev])
    m0, m1, p00, p01, p11, h0, h1, g0, rA2, rP2, rAP, rQ2 = _consts if _consts is not None else _model_consts(model)
    log, sqrt = math.log, math.sqrt
    ll = 0.0
    t_prev = None
    innov = {"A": [], "Phi": [], "Q": []}
    pred_rating = []
    # filtered state after each event (t, kind, m0, m1, p00, p01, p11), recorded only with
    # return_innovations: a record is opened at its event and closed at the next one
    # (the loop body has several `continue`s), the last one after the loop
    filtered = []
    for (t, kind, v0, v1) in ev:
        if return_innovations and filtered:
            filtered[-1] = filtered[-1][:2] + (m0, m1, p00, p01, p11)
        if until is not None and t > until:
            break
        if return_innovations:
            filtered.append((t, kind))
        if t_prev is not None and t > t_prev:
            st = steps.get(round(t - t_prev, 7))
            if st is None:  # a step not in the shared table (cannot happen for tables built from `ev`)
                F, c, Q = model.discretize(t - t_prev)
                st = (float(F[0, 0]), float(F[0, 1]), float(F[1, 0]), float(F[1, 1]), float(c[0]), float(c[1]),
                      float(Q[0, 0]), float(Q[0, 1]), float(Q[1, 1]))
            f00, f01, f10, f11, c0, c1, q00, q01, q11 = st
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
                ll -= 0.5 * (2 * LOG2PI + log(det) + quad)
                if return_innovations:
                    innov["A"].append(e0 / sqrt(s00)); innov["Phi"].append(e1 / sqrt(s11))
                k00, k01 = p00 * i00 + p01 * i01, p00 * i01 + p01 * i11
                k10, k11 = p01 * i00 + p11 * i01, p01 * i01 + p11 * i11
                m0 += k00 * e0 + k01 * e1
                m1 += k10 * e0 + k11 * e1
                n00 = p00 - (k00 * p00 + k01 * p01)
                n01 = p01 - (k00 * p01 + k01 * p11)
                n11 = p11 - (k10 * p01 + k11 * p11)
                p00, p01, p11 = n00, n01, n11
            else:
                # sequential scalar updates, y_A (read-out [1, 0]) then y_Phi ([0, 1])
                if hasA:
                    ph0, ph1 = p00, p01
                    S = ph0 + rA2
                    if S <= 0:
                        return (-np.inf, None) if return_innovations else -np.inf
                    e = v0 - m0
                    ll -= 0.5 * (LOG2PI + log(S) + e * e / S)
                    if return_innovations:
                        innov["A"].append(e / sqrt(S))
                    k0, k1 = ph0 / S, ph1 / S
                    m0 += k0 * e; m1 += k1 * e
                    p00 -= k0 * ph0; p01 -= k0 * ph1; p11 -= k1 * ph1
                if hasP:
                    ph0, ph1 = p01, p11
                    S = ph1 + rP2
                    if S <= 0:
                        return (-np.inf, None) if return_innovations else -np.inf
                    e = v1 - m1
                    ll -= 0.5 * (LOG2PI + log(S) + e * e / S)
                    if return_innovations:
                        innov["Phi"].append(e / sqrt(S))
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
            ll -= 0.5 * (LOG2PI + log(S) + e * e / S)
            if return_innovations:
                innov["Q"].append(e / sqrt(S))
            k0, k1 = ph0 / S, ph1 / S
            m0 += k0 * e; m1 += k1 * e
            p00 -= k0 * ph0; p01 -= k0 * ph1; p11 -= k1 * ph1
    if return_innovations:
        if filtered and len(filtered[-1]) == 2:
            filtered[-1] = filtered[-1] + (m0, m1, p00, p01, p11)
        return float(ll), {"innov": {k: np.array(v) for k, v in innov.items()},
                           "pred_rating": np.array(pred_rating, dtype=float).reshape(-1, 4),
                           "filtered": np.array(filtered, dtype=float).reshape(-1, 7),
                           "state": (m0, m1), "cov": (p00, p01, p11)}
    return float(ll)


def kalman_loglik_irregular(model: StateSpaceDMN, sessions: list[dict], dt: float | None = None,
                            use_eeg: bool = True, use_rating: bool = True) -> float:
    """Sum of exact log-likelihoods over irregular sessions (`dt` ignored; kept
    for signature compatibility with `kalman_loglik`). The step discretisations
    and the model constants are computed once for all sessions."""
    evs = [_events(s) for s in sessions]
    steps = _step_table(model, evs)
    consts = _model_consts(model)
    ll = 0.0
    for s, ev in zip(sessions, evs):
        v = kalman_filter_irregular(model, s, use_rating=use_rating, use_eeg=use_eeg, _ev=ev, _steps=steps,
                                    _consts=consts)
        if not math.isfinite(v):
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


def _minimize_start(nll, x0, label: str | None, tag: str, key: str | None, ckpt, report_every_s: float):
    """One L-BFGS-B run with periodic progress lines and partial checkpoints
    (best point so far, saved every `report_every_s`)."""
    t0 = last = time.time()
    st = {"n_eval": 0, "best_f": np.inf, "best_x": np.asarray(x0, float), "nit": 0}

    def f(theta):
        v = nll(theta)
        st["n_eval"] += 1
        if v < st["best_f"]:
            st["best_f"], st["best_x"] = float(v), np.array(theta, float)
        return v

    def cb(xk):
        nonlocal last
        st["nit"] += 1
        now = time.time()
        if now - last >= report_every_s:
            last = now
            if label:
                print(f"      {tag} vong {st['nit']}, -loglik tot nhat {st['best_f']:.2f}, "
                      f"{(now - t0) / 60:.1f} phut", flush=True)
            if ckpt is not None and key:
                ckpt.put(key, {"final": False, "x": st["best_x"].tolist(), "fun": st["best_f"]})

    res = minimize(f, x0, method="L-BFGS-B", options={"maxiter": 500}, callback=cb)
    if label:
        print(f"    {tag} xong: -loglik {float(res.fun):.2f}, {int(res.nit)} vong, "
              f"{(time.time() - t0) / 60:.1f} phut", flush=True)
    return res


def run_starts(nll, start_list, label: str | None = None, ckpt=None, report_every_s: float = 60.0):
    """L-BFGS-B from each start (unconstrained vectors), best result returned.
    Progress lines and checkpoints as described in `fit`."""
    best = None
    n = len(start_list)
    for i, start in enumerate(start_list):
        tag = f"[{label} {i + 1}/{n}]" if label else ""
        key = f"{label}/start{i}" if (label and ckpt is not None) else None
        saved = ckpt.get(key) if key else None
        if saved and saved.get("final"):
            res = SimpleNamespace(x=np.asarray(saved["x"], float), fun=float(saved["fun"]),
                                  success=bool(saved["success"]), nit=int(saved.get("nit", 0)))
            print(f"    {tag} lay tu checkpoint: -loglik {res.fun:.2f}", flush=True)
        else:
            x0 = np.asarray(saved["x"], float) if saved else np.asarray(start, float)
            if label:
                print(f"    {tag} bat dau" + (" (tiep tu checkpoint)" if saved else ""), flush=True)
            res = _minimize_start(nll, x0, label, tag, key, ckpt, report_every_s)
            if key:
                ckpt.put(key, {"final": True, "x": np.asarray(res.x, float).tolist(), "fun": float(res.fun),
                               "success": bool(res.success), "nit": int(res.nit), "resumed": bool(saved)})
        if best is None or res.fun < best.fun:
            best = res
    return best


def fit(sessions: list[dict], dt: float, init: StateSpaceDMN | None = None, names=None,
        fixed: dict | None = None, n_starts: int = 4, rng=None, loglik=None, starts=None,
        label: str | None = None, ckpt=None, report_every_s: float = 60.0) -> dict:
    """MLE over `names` (default: all free parameters minus `fixed`), L-BFGS-B
    from the initial guess plus jittered restarts.

    `loglik` defaults to the fixed-step `kalman_loglik`; pass
    `kalman_loglik_irregular` for real sessions. `starts` (optional list of
    parameter dicts) replaces the jittered restarts by explicit starting points.

    `label` turns on progress lines (one per start, plus one every
    `report_every_s` seconds). With `ckpt` (model.checkpoint.Checkpoint) and a
    `label`, each finished start is stored under "<label>/start<i>" and
    skipped on a rerun; an interrupted start resumes from its best point so
    far (same optimum, not bit-identical iterates).
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
    best = run_starts(nll, start_list, label=label, ckpt=ckpt, report_every_s=report_every_s)
    model = _unpack(best.x, names, base)
    return {"model": model, "params": {k: getattr(model, k) for k in names}, "loglik": -float(best.fun),
            "success": bool(best.success), "names": names, "theta": best.x, "loglik_fn": loglik}


def profile_likelihood(sessions: list[dict], dt: float, fit_result: dict, param: str, grid: np.ndarray,
                       n_starts: int = 1, label: str | None = None, ckpt=None) -> np.ndarray:
    """Profile log-likelihood of `param` over `grid` (other free parameters refit).
    A flat profile (drop < 1.92 = chi2(1)/2 at 95% across the grid) means the
    parameter is not identifiable from these data. `label`/`ckpt` as in `fit`,
    one item per grid point."""
    model = fit_result["model"]
    others = [k for k in fit_result["names"] if k != param]
    out = []
    for j, g in enumerate(grid):
        sub = f"{label} {j + 1}/{len(grid)}" if label else None
        r = fit(sessions, dt, init=model, names=others, fixed={param: float(g)}, n_starts=n_starts, rng=0,
                loglik=fit_result.get("loglik_fn"), label=sub, ckpt=ckpt)
        out.append(r["loglik"])
    return np.array(out)
