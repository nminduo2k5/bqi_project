"""
Goi ket qua 4 -- LQG vs bang-bang closed-loop control + gate M2.

M1-u: StateSpaceDMN with a control input u(t) in the A equation:
    dA = -k_inh (A - A0) dt - b_control * u dt + sigma_A dW_A
Exact discretisation of the input: B_d = M^-1 (F - I) b, b = (-b_control, 0).

Controller (certainty-equivalence LQG, separation principle):
    * set-point: the unique equilibrium (x_ss, u_ss) with h^T x_ss + gamma0 = target
    * LQR gain K on the deviation, cost Q_cost (QoC - target)^2 + R_cost u^2
    * Kalman filter on the EEG observations y_t = x_t + eta (both channels, every epoch)
    * u_t = u_ss - K (x_hat_t - x_ss), clipped to |u| <= u_max
Bang-bang: u = +/- u_max according to the sign of (target - QoC_hat), using the
SAME state estimate (fair comparison). No-control: u = 0.

`lqg_stationary_cost` gives the closed-loop stationary cost in closed form
(augmented Lyapunov equation, no clipping) and is used by model.sensitivity.

Gate M2: LQG with true parameters beats bang-bang and no-control on the
quadratic cost, and the analytic stationary cost agrees with simulation.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bqi.dmn_ode import StateSpaceDMN
from eeg import gates
from model import config as CFG

C = CFG.CONTROL
DT_DEFAULT = CFG.DT        # epoch length in minutes


# =============================================================================
# System
# =============================================================================
def state_space_with_control(model: StateSpaceDMN, b_control: float = C.b_control, dt: float = DT_DEFAULT) -> dict:
    """Discrete-time matrices of M1-u.

    State x = [A, Phi]; input u (scalar); EEG observation y = x + eta,
    eta ~ N(0, R_obs); read-out QoC = h^T x + gamma0.
    """
    F, c, Q = model.discretize(dt)
    M, _ = model.drift()
    b = np.array([-b_control, 0.0])
    B = np.linalg.solve(M, (F - np.eye(2)) @ b)      # exact zero-order-hold input matrix
    return {"F": F, "c": c, "Q": Q, "B": B, "h": model.readout(), "gamma0": model.gamma0,
            "R_obs": np.diag([model.r_A ** 2, model.r_Phi ** 2]), "r_Q": model.r_Q, "dt": dt}


def equilibrium(sys_m: dict, target_qoc: float) -> dict:
    """Constant input u_ss and state x_ss with F x_ss + c + B u_ss = x_ss and
    h^T x_ss + gamma0 = target."""
    F, c, B, h = sys_m["F"], sys_m["c"], sys_m["B"], sys_m["h"]
    G = np.linalg.inv(np.eye(2) - F)
    gain = h @ G @ B
    if abs(gain) < 1e-12:
        raise ValueError("QoC target unreachable: zero DC gain from u to QoC")
    u_ss = (target_qoc - sys_m["gamma0"] - h @ G @ c) / gain
    x_ss = G @ (c + B * u_ss)
    return {"u_ss": float(u_ss), "x_ss": x_ss}


def solve_lqg(sys_matrices: dict, Q_cost: float = C.Q_cost, R_cost: float = C.R_cost,
              target_qoc: float = C.target_qoc) -> dict:
    """LQR gain + Kalman gain + set-point for the certainty-equivalence LQG controller."""
    F, B, h = sys_matrices["F"], sys_matrices["B"].reshape(-1, 1), sys_matrices["h"]
    Q_state = Q_cost * np.outer(h, h)
    R = np.array([[R_cost]])
    P_lqr = solve_discrete_are(F, B, Q_state, R)
    K = np.linalg.solve(R + B.T @ P_lqr @ B, B.T @ P_lqr @ F).ravel()
    # Kalman filter for y = x + eta (C = I): P = prediction covariance
    P_kf = solve_discrete_are(F.T, np.eye(2), sys_matrices["Q"], sys_matrices["R_obs"])
    L = P_kf @ np.linalg.inv(P_kf + sys_matrices["R_obs"])
    eq = equilibrium(sys_matrices, target_qoc)
    return {"K": K, "L": L, "P_lqr": P_lqr, "P_kf": P_kf, "x_ss": eq["x_ss"], "u_ss": eq["u_ss"],
            "target_qoc": target_qoc, "Q_cost": Q_cost, "R_cost": R_cost}


# =============================================================================
# Simulation of one policy
# =============================================================================
def simulate_policy(model: StateSpaceDMN, controller: dict, policy: str = "lqg", n_epochs: int = C.n_epochs,
                    b_control: float = C.b_control, u_max: float = C.u_max, seed: int = 0,
                    sys_true: dict | None = None) -> dict:
    """Closed loop on the TRUE model (`sys_true` defaults to the model's own
    matrices) with a controller possibly designed on an estimated model.
    Common random numbers: the same seed gives the same noise for every policy.
    """
    rng = np.random.default_rng(seed)
    st = sys_true or state_space_with_control(model, b_control)
    F, c, Q, B, h, g0, R = st["F"], st["c"], st["Q"], st["B"], st["h"], st["gamma0"], st["R_obs"]
    Fc, cc, Bc = controller.get("F", F), controller.get("c", c), controller.get("B", B)
    K, L, x_ss, u_ss = controller["K"], controller["L"], controller["x_ss"], controller["u_ss"]
    target, Qc, Rc = controller["target_qoc"], controller["Q_cost"], controller["R_cost"]
    hc, g0c = controller.get("h", h), controller.get("gamma0", g0)
    Lw = np.linalg.cholesky(Q + 1e-15 * np.eye(2))
    Lr = np.sqrt(np.diag(R))
    x = model.stationary_mean() + np.linalg.cholesky(model.stationary_cov() + 1e-15 * np.eye(2)) @ rng.normal(size=2)
    x_hat = controller.get("x0_hat", model.stationary_mean()).copy()
    xs, us, qocs = np.empty((n_epochs, 2)), np.empty(n_epochs), np.empty(n_epochs)
    for t in range(n_epochs):
        if policy == "lqg":
            u = u_ss - K @ (x_hat - x_ss)
        elif policy == "bangbang":
            u = u_max if (hc @ x_hat + g0c) < target else -u_max
        else:
            u = 0.0
        u = float(np.clip(u, -u_max, u_max))
        x = F @ x + c + B * u + Lw @ rng.normal(size=2)
        y = x + Lr * rng.normal(size=2)
        x_pred = Fc @ x_hat + cc + Bc * u
        x_hat = x_pred + L @ (y - x_pred)
        xs[t], us[t], qocs[t] = x, u, h @ x + g0
    cost = float(np.mean(Qc * (qocs - target) ** 2 + Rc * us ** 2))
    return {"t": np.arange(n_epochs) * st["dt"], "x": xs, "QoC": qocs, "u": us, "cost": cost,
            "tracking_mse": float(np.mean((qocs - target) ** 2)), "u_rms": float(np.sqrt(np.mean(us ** 2)))}


def simulate_lqg(model, controller, n_epochs=C.n_epochs, b_control=C.b_control, u_max=C.u_max, seed=0):
    return simulate_policy(model, controller, "lqg", n_epochs, b_control, u_max, seed)


def simulate_bangbang(model, target_qoc=C.target_qoc, u_max=C.u_max, n_epochs=C.n_epochs, b_control=C.b_control, seed=0,
                      controller=None):
    ctrl = controller or solve_lqg(state_space_with_control(model, b_control), target_qoc=target_qoc)
    return simulate_policy(model, ctrl, "bangbang", n_epochs, b_control, u_max, seed)


def simulate_no_control(model, target_qoc=C.target_qoc, n_epochs=C.n_epochs, seed=0, controller=None):
    ctrl = controller or solve_lqg(state_space_with_control(model), target_qoc=target_qoc)
    return simulate_policy(model, ctrl, "none", n_epochs, 1.0, 0.0, seed)


# =============================================================================
# Closed-form stationary cost of the (unclipped) LQG loop
# =============================================================================
def lqg_stationary_cost(model: StateSpaceDMN, controller: dict, b_control: float = C.b_control,
                        sys_true: dict | None = None) -> dict:
    """Stationary mean cost of the LQG loop via the augmented state z = [x, x_hat].

        x'     = F x - B K x_hat + d + w
        x_hat' = L F x + ((I - L) F - B K) x_hat + d + L w + L v
        d      = c + B u_ss + B K x_ss
    Cov([w, Lw + Lv]) = [[Q, Q L^T], [L Q, L Q L^T + L R L^T]].
    """
    st = sys_true or state_space_with_control(model, b_control)
    F, c, Q, B, h, g0, R = st["F"], st["c"], st["Q"], st["B"], st["h"], st["gamma0"], st["R_obs"]
    K, L, x_ss, u_ss = controller["K"], controller["L"], controller["x_ss"], controller["u_ss"]
    target, Qc, Rc = controller["target_qoc"], controller["Q_cost"], controller["R_cost"]
    I2 = np.eye(2)
    BK = np.outer(B, K)
    A = np.block([[F, -BK], [L @ F, (I2 - L) @ F - BK]])
    d = c + B * u_ss + BK @ x_ss
    bz = np.concatenate([d, d])
    W = np.block([[Q, Q @ L.T], [L @ Q, L @ Q @ L.T + L @ R @ L.T]])
    m = np.linalg.solve(np.eye(4) - A, bz)
    S = solve_discrete_lyapunov(A, W)
    mx, mh = m[:2], m[2:]
    Sxx, Shh = S[:2, :2], S[2:, 2:]
    qoc_mean = h @ mx + g0
    track = (qoc_mean - target) ** 2 + h @ Sxx @ h
    u_mean = u_ss - K @ (mh - x_ss)
    u_sq = u_mean ** 2 + K @ Shh @ K
    return {"cost": float(Qc * track + Rc * u_sq), "tracking_mse": float(track), "u_rms": float(np.sqrt(u_sq)),
            "qoc_mean": float(qoc_mean), "stable": bool(np.max(np.abs(np.linalg.eigvals(A))) < 1)}


def lqg_simulated_stationary_cost(model: StateSpaceDMN, controller: dict, b_control: float = C.b_control,
                                  n_epochs: int = C.stationary_n_epochs, burn_in: int = C.stationary_burn_in,
                                  n_trials: int = 4,
                                  u_max: float = 1e9, seed: int = 0, sys_true: dict | None = None) -> float:
    """Simulated stationary cost of the LQG loop: long sessions, first `burn_in`
    epochs discarded, no input clipping -- the quantity `lqg_stationary_cost`
    computes in closed form."""
    st = sys_true or state_space_with_control(model, b_control)
    vals = []
    for k in range(n_trials):
        r = simulate_policy(model, controller, "lqg", n_epochs, b_control, u_max, seed + k, st)
        q, u = r["QoC"][burn_in:], r["u"][burn_in:]
        vals.append(np.mean(controller["Q_cost"] * (q - controller["target_qoc"]) ** 2
                            + controller["R_cost"] * u ** 2))
    return float(np.mean(vals))


# =============================================================================
# Comparisons
# =============================================================================
def compare_policies(model: StateSpaceDMN, target_qoc: float = C.target_qoc, u_max: float = C.u_max,
                     n_epochs: int = C.n_epochs, b_control: float = C.b_control, n_trials: int = C.n_trials,
                     Q_cost: float = C.Q_cost, R_cost: float = C.R_cost, seed: int = CFG.SEED) -> dict:
    """LQG vs bang-bang vs no control on the true model, common random numbers."""
    st = state_space_with_control(model, b_control)
    ctrl = solve_lqg(st, Q_cost=Q_cost, R_cost=R_cost, target_qoc=target_qoc)
    costs = {"lqg": [], "bangbang": [], "no_control": []}
    for k, pol in (("lqg", "lqg"), ("bangbang", "bangbang"), ("no_control", "none")):
        for trial in range(n_trials):
            costs[k].append(simulate_policy(model, ctrl, pol, n_epochs, b_control, u_max, seed + trial, st)["cost"])
    mean = {k: float(np.mean(v)) for k, v in costs.items()}
    analytic = lqg_stationary_cost(model, ctrl, b_control, st)
    sim_ss = lqg_simulated_stationary_cost(model, ctrl, b_control, seed=seed, sys_true=st)
    return {**mean, "lqg_analytic": analytic["cost"], "lqg_sim_stationary": sim_ss,
            "lqg_analytic_stable": analytic["stable"],
            "u_ss": ctrl["u_ss"], "x_ss": ctrl["x_ss"].tolist(),
            "lqg_beats_bangbang": mean["lqg"] < mean["bangbang"],
            "lqg_beats_nocontrol": mean["lqg"] < mean["no_control"],
            "lqg_wins": mean["lqg"] < mean["bangbang"] and mean["lqg"] < mean["no_control"],
            "costs_all": costs}


def control_efficiency_vs_param_error(model: StateSpaceDMN, mle_df, target_qoc: float = C.target_qoc,
                                      u_max: float = C.u_max, n_epochs: int = C.n_epochs,
                                      b_control: float = C.b_control, n_trials: int = C.n_trials_robustness,
                                      seed: int = CFG.SEED) -> list[dict]:
    """Robustness: for every design cell, design the LQG controller (gains AND
    filter) on each *actually fitted* parameter set stored in mle_df['fits']
    and run it on the true model of THAT cell. Efficiency loss =
    (cost_est - cost_true) / cost_true, in per cent.

    The true model of a cell is the nominal model with the cell's EEG noise
    level (r_A = r_Phi = r_obs): the fits were obtained from data generated at
    that level, so comparing against a fixed-r truth would attribute the
    filter/plant noise mismatch to estimation error (a 15 % "loss" appeared
    for every r = 0.05 cell regardless of N before this was fixed).
    """
    seeds = [seed + i for i in range(n_trials)]
    out = []
    cache: dict[float, tuple] = {}
    for _, row in mle_df.iterrows():
        r = float(row["r_obs"])
        if r not in cache:
            m_true = model.with_params(r_A=r, r_Phi=r)
            st = state_space_with_control(m_true, b_control)
            ctrl_true = solve_lqg(st, target_qoc=target_qoc)
            c_true = float(np.mean([simulate_policy(m_true, ctrl_true, "lqg", n_epochs, b_control, u_max, s, st)["cost"]
                                    for s in seeds]))
            cache[r] = (m_true, st, c_true)
        m_true, st, cost_true = cache[r]
        fits = row["fits"]
        fits = json.loads(fits) if isinstance(fits, str) else fits
        losses = []
        for p in fits:
            try:
                m_est = model.with_params(**{k: float(v) for k, v in p.items()})
                st_est = state_space_with_control(m_est, b_control)
                ctrl = solve_lqg(st_est, target_qoc=target_qoc)
                ctrl.update({"F": st_est["F"], "c": st_est["c"], "B": st_est["B"], "h": st_est["h"],
                             "gamma0": st_est["gamma0"], "x0_hat": m_est.stationary_mean()})
                cost = float(np.mean([simulate_policy(m_true, ctrl, "lqg", n_epochs, b_control, u_max, s, st)["cost"]
                                      for s in seeds]))
                losses.append((cost - cost_true) / cost_true * 100)
            except (ValueError, np.linalg.LinAlgError):
                losses.append(float("nan"))
        avg_err = float(np.nanmean([row.get(f"mle_{k}", np.nan) for k in CFG.KEY_PARAMS]))
        out.append({"N": int(row["N"]), "session_min": float(row["session_min"]),
                    "probe_every_min": float(row["probe_every_min"]), "r_obs": float(row["r_obs"]),
                    "avg_mle_rmse": avg_err, "n_fits": len(fits),
                    "efficiency_loss_pct_median": float(np.nanmedian(losses)) if losses else float("nan"),
                    "efficiency_loss_pct_q90": float(np.nanpercentile(losses, 90)) if losses else float("nan"),
                    "cost_true": cost_true})
    return out


def gate_M2(comparison_result: dict, verbose: bool = True, analytic_tol: float = CFG.GATES.m2_analytic_tol) -> dict:
    """Gate M2: LQG (true parameters) beats bang-bang and no-control, and the
    closed-form stationary cost matches the simulated LQG cost within `analytic_tol`."""
    r = comparison_result
    analytic_ok = (abs(r["lqg_analytic"] - r["lqg_sim_stationary"]) <= analytic_tol * r["lqg_sim_stationary"]
                   and r["lqg_analytic_stable"])
    passed = bool(r["lqg_wins"] and analytic_ok)
    details = {k: r[k] for k in ["lqg", "bangbang", "no_control", "lqg_analytic", "lqg_sim_stationary",
                                 "lqg_beats_bangbang", "lqg_beats_nocontrol", "u_ss"]}
    details["analytic_matches_simulation"] = bool(analytic_ok)
    path = gates.record("M2", passed, details)
    if verbose:
        print(f"M2 {'PASS' if passed else 'FAIL'}: session cost LQG={r['lqg']:.4f}, bang-bang={r['bangbang']:.4f}, "
              f"no-control={r['no_control']:.4f}; stationary LQG analytic {r['lqg_analytic']:.4f} vs "
              f"simulated {r['lqg_sim_stationary']:.4f} -> {path}")
    return {"passed": passed, **details}


def run(seed: int = CFG.SEED, fast: bool = False, verbose: bool = True, output_dir: str | None = None,
        mle_csv: str | None = None) -> dict:
    """Entry point for `python main.py model control`."""
    model = StateSpaceDMN()
    n_trials = 5 if fast else C.n_trials
    if verbose:
        print("Comparing LQG vs bang-bang vs no-control (true parameters)...", flush=True)
    comparison = compare_policies(model, n_trials=n_trials, seed=seed)
    gate_result = gate_M2(comparison, verbose=verbose)
    result = {"comparison": {k: v for k, v in comparison.items() if k != "costs_all"}, "gate_M2": gate_result}
    if mle_csv and os.path.isfile(mle_csv):
        import pandas as pd
        rob = control_efficiency_vs_param_error(model, pd.read_csv(mle_csv),
                                                n_trials=3 if fast else C.n_trials_robustness, seed=seed)
        result["robustness"] = rob
        if verbose:
            for r in rob:
                print(f"  N={r['N']:2d} {r['session_min']:4.0f}min probe/{r['probe_every_min']}min r={r['r_obs']}: "
                      f"loss median {r['efficiency_loss_pct_median']:.1f}%  q90 {r['efficiency_loss_pct_q90']:.1f}%")
    CFG.snapshot(output_dir)
    CFG.snapshot(output_dir)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "control_results.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
    return result
