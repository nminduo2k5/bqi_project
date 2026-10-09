"""
Goi ket qua 1 -- Propositions P1-P5: correcting the original DMN model.

P1: Correct steady states of M0 (A*=0, Phi*=k_int/k_d, QoC*=0).
P2: Transfer function QoC/input is high-pass H(s)=s/(s+gamma) -> DC gain = 0.
P3: Paper eq. 55-57 do not satisfy the ODE at steady state (nonzero residuals).
P4: M1 (StateSpaceDMN) has correct stationary moments (Lyapunov equation).
P5: Stationary distribution of QoC in M1 is Gaussian N(QoC*, h^T P h).

Gate M0 passes iff all five propositions are verified numerically.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bqi.dmn_ode import (ODE_PARAMS_REFERENCE, StateSpaceDMN, _rhs,
                          steady_states, steady_states_paper)
from eeg import gates
from model import config as CFG
from model import config as CFG
from model import config as CFG
from model import config as CFG

# =============================================================================
# P1 — correct steady states
# =============================================================================
def correct_steady_states(params: dict) -> dict:
    """Return (A*, Phi*, QoC*) for M0 with zero-mean xi and verify residuals."""
    ss = steady_states(params)
    zero_xi = lambda t: 0.0
    one_mult = lambda t: 1.0
    residual = _rhs(0.0, [ss["A_DMN_star"], ss["Phi_star"], ss["QoC_star"]], params, zero_xi, one_mult)
    return {"A_star": ss["A_DMN_star"], "Phi_star": ss["Phi_star"], "QoC_star": ss["QoC_star"],
            "residual": residual, "residual_norm": float(np.linalg.norm(residual))}


# =============================================================================
# P2 — transfer function QoC: H(s) = s / (s + gamma) from d[A,Phi]/dt to QoC
# =============================================================================
def transfer_function_qoc(omega: float, gamma: float) -> complex:
    """H(j*omega) = j*omega / (j*omega + gamma).

    This is the transfer function from the derivative inputs alpha*dPhi/dt
    and -beta*dA/dt to QoC in the M0 model. At DC (omega=0): H=0.
    At high freq (omega>>gamma): |H|->1.
    """
    s = 1j * omega
    return s / (s + gamma)


def bode_plot_data(gamma: float, freqs: np.ndarray) -> dict:
    """Magnitude (dB) and phase (deg) of H(j*2*pi*f) for each frequency in freqs."""
    H = np.array([transfer_function_qoc(2 * np.pi * f, gamma) for f in freqs])
    return {"freqs": freqs, "magnitude_db": 20 * np.log10(np.abs(H) + 1e-30),
            "phase_deg": np.angle(H, deg=True), "magnitude": np.abs(H)}


def dc_gain_is_zero(gamma: float, tol: float = 1e-10) -> bool:
    """|H(j*0)| = 0 for any gamma > 0 (analytic statement)."""
    return abs(transfer_function_qoc(0.0, gamma)) < tol


def verify_transfer_on_ode(params: dict, omegas=CFG.P2_OMEGAS, xi_step: float = CFG.P2_STEP_XI,
                           t_end: float = CFG.P2_T_END_MIN) -> dict:
    """Check P2 on the actual M0 ODE (bqi.dmn_ode._rhs), not on the formula.

    (i) Step input: a constant drive xi = xi_step moves A and Phi to non-zero
        levels, yet QoC must decay to 0 (zero DC gain).
    (ii) Sinusoidal drive xi = sin(omega t): after transients, the amplitude
         ratio QoC / (alpha Phi - beta A) must equal |H(j omega)| = omega / sqrt(omega^2 + gamma^2).
    """
    from scipy.integrate import solve_ivp
    one = lambda t: 1.0
    out = {}
    sol = solve_ivp(_rhs, (0, t_end), [0.0, 0.0, 0.0], args=(params, lambda t: xi_step, one),
                    method="LSODA", rtol=1e-9, atol=1e-12, t_eval=[t_end])
    A, Phi, Q = sol.y[:, -1]
    out["step"] = {"A_end": float(A), "Phi_end": float(Phi), "QoC_end": float(Q),
                   "drive_level": float(params["alpha"] * Phi - params["beta"] * A)}
    gamma = params["gamma"]
    ratios = []
    for w in omegas:
        T = 2 * np.pi / w
        t_eval = np.linspace(t_end - 3 * T, t_end, 3000)
        s = solve_ivp(_rhs, (0, t_end), [0.0, 0.0, 0.0], args=(params, lambda t, w=w: np.sin(w * t), one),
                      method="LSODA", rtol=1e-9, atol=1e-12, t_eval=t_eval)
        drive = params["alpha"] * s.y[1] - params["beta"] * s.y[0]
        q = s.y[2]
        meas = (q.max() - q.min()) / (drive.max() - drive.min())
        theory = w / np.sqrt(w ** 2 + gamma ** 2)
        ratios.append({"omega": w, "measured": float(meas), "theory": float(theory),
                       "rel_err": float(abs(meas - theory) / theory)})
    out["sinusoid"] = ratios
    # Zero DC gain: QoC must be attenuated by >= 1e4 relative to the drive level
    # alpha*Phi - beta*A that a low-pass read-out would hold (the residual is the
    # slowest transient, exp(-k_d t), not a steady offset).
    out["step"]["attenuation"] = abs(out["step"]["drive_level"]) / max(abs(out["step"]["QoC_end"]), 1e-300)
    out["ok"] = (out["step"]["attenuation"] > CFG.GATES.m0_p2_min_attenuation
                 and abs(out["step"]["drive_level"]) > 1e-3
                 and all(r["rel_err"] < CFG.GATES.m0_p2_max_amp_err for r in ratios))
    return out


# =============================================================================
# P3 — paper residuals
# =============================================================================
def paper_residual(params: dict) -> dict:
    """Substitute eq. 55-57 (steady_states_paper) into the ODE RHS.
    Returns the three residual components; they should be nonzero."""
    sp = steady_states_paper(params)
    zero_xi = lambda t: 0.0
    one_mult = lambda t: 1.0
    r = _rhs(0.0, [sp["A_DMN_star"], sp["Phi_star"], sp["QoC_star"]], params, zero_xi, one_mult)
    return {"dA_residual": float(r[0]), "dPhi_residual": float(r[1]), "dQoC_residual": float(r[2]),
            "residual_norm": float(np.linalg.norm(r)),
            "paper_values": {"A_star": sp["A_DMN_star"], "Phi_star": sp["Phi_star"],
                             "QoC_star": sp["QoC_star"]}}


# =============================================================================
# P4 — M1 stationary moments via Lyapunov equation
# =============================================================================
def m1_stationary_moments(model: StateSpaceDMN) -> dict:
    """Verify Lyapunov equation: M P + P M^T + G G^T = 0 at stationary cov P."""
    M, u = model.drift()
    mu = model.stationary_mean()
    P = model.stationary_cov()
    G = np.diag([model.sigma_A, model.sigma_Phi])
    lyap_residual = M @ P + P @ M.T + G @ G.T
    lyap_norm = float(np.linalg.norm(lyap_residual))
    ss = model.steady_states()
    h = model.readout()
    qoc_var = float(h @ P @ h)
    return {"mean": mu.tolist(), "QoC_star": ss["QoC_star"], "cov": P.tolist(),
            "lyapunov_residual_norm": lyap_norm, "lyapunov_ok": lyap_norm < 1e-10,
            "qoc_stationary_variance": qoc_var}


# =============================================================================
# P5 — stationary distribution of QoC is Gaussian
# =============================================================================
def qoc_stationary_distribution(model: StateSpaceDMN, n_samples: int = CFG.P5_N_SAMPLES,
                                 dt: float = CFG.DT, seed: int = 0) -> dict:
    """Compare the QoC distribution with the theoretical N(QoC*, h^T P h).

    Consecutive epochs are strongly autocorrelated (time constant 1/min(k_inh, k_d),
    ~30 min = ~180 epochs here), which invalidates a KS test on a raw chain.
    The chain is therefore thinned to one sample every 5 slowest time constants,
    giving ~independent draws.
    """
    tau = 1.0 / min(model.k_inh, model.k_d)
    stride = int(np.ceil(CFG.GATES.m0_p5_thin_time_constants * tau / dt))
    sim = model.simulate(n_samples * stride, dt=dt, probe_every=None, seed=seed)
    qoc_sim = sim["QoC"][::stride]
    ss = model.steady_states()
    h = model.readout()
    P = model.stationary_cov()
    qoc_mean_theory = ss["QoC_star"]
    qoc_std_theory = float(np.sqrt(h @ P @ h))
    ks_stat, ks_p = sps.kstest(qoc_sim, "norm",
                                args=(qoc_mean_theory, qoc_std_theory))
    emp_mean = float(qoc_sim.mean())
    emp_std = float(qoc_sim.std())
    n = len(qoc_sim)
    mean_z = (emp_mean - qoc_mean_theory) / (qoc_std_theory / np.sqrt(n))
    return {"ks_stat": float(ks_stat), "ks_p": float(ks_p), "n_independent": n, "stride_epochs": stride,
            "theoretical_mean": qoc_mean_theory, "theoretical_std": qoc_std_theory,
            "empirical_mean": emp_mean, "empirical_std": emp_std,
            "mean_error": abs(emp_mean - qoc_mean_theory), "mean_z": float(mean_z),
            "std_error": abs(emp_std - qoc_std_theory),
            "gaussian_ok": bool(ks_p > CFG.GATES.m0_p5_min_ks_p and abs(mean_z) < 3)}

# =============================================================================
# Gate M0
# =============================================================================
def gate_M0(cfg: dict | None = None, verbose: bool = True, force: bool = False) -> dict:
    """Run P1-P5 and record gate M0."""
    model = StateSpaceDMN()
    params = ODE_PARAMS_REFERENCE
    checks = {}

    def log(name: str, ok: bool, info: dict):
        checks[name] = {"passed": bool(ok), **info}
        if verbose:
            sym = "PASS" if ok else "FAIL"
            print(f"  [{sym}] {name}: {info}", flush=True)

    # P1
    p1 = correct_steady_states(params)
    CFG.check_nominal_matches_model()
    log("P1_correct_steady_states", p1["residual_norm"] < CFG.GATES.m0_residual_tol,
        {"residual_norm": p1["residual_norm"], "A_star": p1["A_star"], "QoC_star": p1["QoC_star"]})

    # P2
    p2 = verify_transfer_on_ode(params)
    log("P2_highpass_on_ode", p2["ok"] and dc_gain_is_zero(params["gamma"]),
        {"step_QoC_end": p2["step"]["QoC_end"], "step_drive_level": round(p2["step"]["drive_level"], 4),
         "sinusoid_rel_err": [round(r["rel_err"], 5) for r in p2["sinusoid"]]})

    # P3
    p3 = paper_residual(params)
    log("P3_paper_residual_nonzero", p3["residual_norm"] > 1e-3,
        {"residual_norm": round(p3["residual_norm"], 6),
         "dA": round(p3["dA_residual"], 6), "dPhi": round(p3["dPhi_residual"], 6)})

    # P4
    p4 = m1_stationary_moments(model)
    log("P4_lyapunov_residual", p4["lyapunov_residual_norm"] < CFG.GATES.m0_residual_tol,
        {"lyapunov_residual_norm": p4["lyapunov_residual_norm"],
         "QoC_star": round(p4["QoC_star"], 4), "qoc_var": round(p4["qoc_stationary_variance"], 4)})

    # P5
    p5 = qoc_stationary_distribution(model)
    log("P5_qoc_distribution_gaussian", p5["gaussian_ok"],
        {"ks_stat": round(p5["ks_stat"], 4), "ks_p": round(p5["ks_p"], 4),
         "mean_error": round(p5["mean_error"], 4), "std_error": round(p5["std_error"], 4)})

    passed = all(c["passed"] for c in checks.values())
    path = gates.record("M0", passed, checks)
    if verbose:
        print(f"M0 {'PASS' if passed else 'FAIL'} -> {path}")
    return {"passed": passed, "checks": checks}


def run(seed: int = 42, verbose: bool = True, output_dir: str | None = None) -> dict:
    """Entry point for main.py model theory subcommand."""
    import json
    result = gate_M0(verbose=verbose)
    CFG.snapshot(output_dir)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "theory_results.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
    return result