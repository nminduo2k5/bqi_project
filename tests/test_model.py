"""Tests for the model/ package (Q1_pipeline_model.md)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bqi.dmn_ode import ODE_PARAMS_REFERENCE, StateSpaceDMN
from eeg.model_fit import kalman_loglik
from model.fisher import (Design, crb, fisher_exact, gaussian_loglik, identifiability_spectrum,
                          session_moments, stack_observations)


# ---------------------------------------------------------------- fisher.py
def _masked(sim, config):
    y = sim["y"].copy()
    if config in ("a", "c"):
        y[:, 0] = np.nan
    if config in ("a", "b"):
        y[:, 1] = np.nan
    return {"y": y, "rating": sim["rating"]}


@pytest.mark.parametrize("config", ["a", "b", "c", "d"])
def test_exact_gaussian_likelihood_matches_kalman(config):
    m = StateSpaceDMN()
    d = Design.regular(15, 2.0, config=config)
    sims = [m.simulate(d.n_epochs, d.dt, probe_idx=d.probe_idx, seed=s) for s in range(2)]
    exact = gaussian_loglik(m, d, [stack_observations(s, d) for s in sims])
    kal = kalman_loglik(m, [_masked(s, config) for s in sims], d.dt)
    assert abs(exact - kal) < 1e-8


def test_session_moments_covariance_is_psd_and_stationary():
    m = StateSpaceDMN()
    d = Design.regular(10, 1.0)
    mu, S = session_moments(m, d)
    assert np.all(np.linalg.eigvalsh(S) > 0)
    # stationary start: mean of y_A is A0, of y_Phi is Phi*, of rating is QoC*, at every epoch
    labels = []
    for t in range(d.n_epochs):
        labels += ["A", "Phi"] + (["Q"] if t in d.probe_idx else [])
    labels = np.array(labels)
    ss = m.steady_states()
    assert np.allclose(mu[labels == "A"], m.A0)
    assert np.allclose(mu[labels == "Phi"], ss["Phi_star"])
    assert np.allclose(mu[labels == "Q"], ss["QoC_star"])


def test_fisher_scales_with_subjects_and_crb_is_finite_for_config_d():
    m = StateSpaceDMN()
    d = Design.regular(20, 2.0)
    I1 = fisher_exact(m, d, n_subjects=1)
    I4 = fisher_exact(m, d, n_subjects=4)
    assert np.allclose(I4, 4 * I1)
    b = crb(I4, StateSpaceDMN.FREE, m)
    assert all(np.isfinite(b[k]["rel_se"]) for k in StateSpaceDMN.FREE if k != "gamma0")
    assert b["A0"]["rel_se"] < b["k_inh"]["rel_se"]  # the level is easier than the rate


def test_identifiability_full_rank_only_when_both_eeg_channels_observed():
    m = StateSpaceDMN().with_params(gamma0=0.3)
    names = list(StateSpaceDMN.FREE)
    ranks = {}
    for cfg in "abcd":
        I = fisher_exact(m, Design.regular(60, 1.0, config=cfg), names)
        ranks[cfg] = identifiability_spectrum(I, names, tol=1e-7)
    assert ranks["d"]["null_directions"] == [] and ranks["d"]["absent"] == []
    assert ranks["b"]["absent"] == ["r_Phi"] and len(ranks["b"]["null_directions"]) == 1
    assert ranks["c"]["absent"] == ["r_A"] and len(ranks["c"]["null_directions"]) == 1
    assert set(ranks["b"]["null_directions"][0]["direction"]) >= {"alpha", "k_int", "sigma_Phi"}


def test_phi_scaling_symmetry_when_phi_unobserved():
    """Config b: Phi -> c Phi with k_int, sigma_Phi -> c*, alpha -> alpha/c leaves (y_A, rating) invariant."""
    m = StateSpaceDMN().with_params(gamma0=0.3)
    c = 1.7
    m2 = m.with_params(k_int=c * m.k_int, sigma_Phi=c * m.sigma_Phi, alpha=m.alpha / c)
    d = Design.regular(20, 1.0, config="b")
    mu1, S1 = session_moments(m, d)
    mu2, S2 = session_moments(m2, d)
    assert np.allclose(mu1, mu2) and np.allclose(S1, S2)


# ---------------------------------------------------------------- theory.py
def test_p2_highpass_verified_on_the_ode():
    from model.theory import verify_transfer_on_ode
    r = verify_transfer_on_ode(ODE_PARAMS_REFERENCE, omegas=(0.1, 1.0), t_end=400.0)
    assert r["ok"]
    assert abs(r["step"]["drive_level"]) > 0.1 and r["step"]["attenuation"] > 1e4


def test_p5_thinned_chain_is_gaussian():
    from model.theory import qoc_stationary_distribution
    r = qoc_stationary_distribution(StateSpaceDMN(), n_samples=300, seed=3)
    assert r["gaussian_ok"] and r["stride_epochs"] > 50


# ---------------------------------------------------------------- control.py
def test_equilibrium_is_a_fixed_point_hitting_the_target():
    from model.control import equilibrium, state_space_with_control
    m = StateSpaceDMN()
    st = state_space_with_control(m)
    eq = equilibrium(st, target_qoc=2.0)
    x = eq["x_ss"]
    assert np.allclose(st["F"] @ x + st["c"] + st["B"] * eq["u_ss"], x)
    assert abs(st["h"] @ x + st["gamma0"] - 2.0) < 1e-10


def test_lqg_beats_alternatives_and_matches_analytic_cost():
    from model.control import compare_policies
    r = compare_policies(StateSpaceDMN(), n_trials=4, n_epochs=400, u_max=50.0, seed=1)
    assert r["lqg"] < r["bangbang"] and r["lqg"] < r["no_control"]
    # closed-form stationary cost vs long simulation with burn-in (transients excluded)
    assert abs(r["lqg_analytic"] - r["lqg_sim_stationary"]) < 0.15 * r["lqg_sim_stationary"]
    assert r["lqg_analytic_stable"]


def test_lqg_with_wrong_parameters_costs_more():
    from model.control import simulate_policy, solve_lqg, state_space_with_control
    m = StateSpaceDMN()
    st = state_space_with_control(m)
    ctrl_true = solve_lqg(st, target_qoc=2.0)
    m_bad = m.with_params(k_inh=0.6, beta=1.0)
    st_bad = state_space_with_control(m_bad)
    ctrl_bad = solve_lqg(st_bad, target_qoc=2.0)
    ctrl_bad.update({"F": st_bad["F"], "c": st_bad["c"], "B": st_bad["B"]})
    c_true = np.mean([simulate_policy(m, ctrl_true, "lqg", 300, seed=s, sys_true=st)["cost"] for s in range(4)])
    c_bad = np.mean([simulate_policy(m, ctrl_bad, "lqg", 300, seed=s, sys_true=st)["cost"] for s in range(4)])
    assert c_bad > c_true


# ---------------------------------------------------------------- design.py
def test_crb_cell_uses_all_parameters_and_shrinks_with_n():
    from model.design import _crb_for_cell
    m = StateSpaceDMN()
    c6 = _crb_for_cell(m, {"N": 6, "session_min": 20, "probe_every_min": 2.0, "r_obs": 0.1})
    c24 = _crb_for_cell(m, {"N": 24, "session_min": 20, "probe_every_min": 2.0, "r_obs": 0.1})
    for k in ("k_inh", "A0", "alpha", "beta"):
        assert np.isfinite(c6[k]) and abs(c24[k] - c6[k] / 2) < 1e-9


def test_probe_schedule_optimiser_returns_valid_schedules():
    from model.design import doptimal_probe_schedule
    r = doptimal_probe_schedule(StateSpaceDMN(), session_min=20, n_probes=8)
    assert len(r["optimal_schedule"]) == 8 and r["efficiency_vs_uniform"] >= 1.0 - 1e-9
    assert np.isfinite(r["uniform_logdet_crb"])


# ---------------------------------------------------------------- sensitivity.py
def test_sobol_estimator_recovers_additive_linear_model():
    from model.sensitivity import saltelli_sample, sobol_indices
    s = saltelli_sample(n_base=2048, seed=0)
    w = np.array([3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Y = 3 x1 + x2 -> S1 = 0.9, 0.1
    f = lambda X: X @ w
    idx = sobol_indices(s, "lin", f(s["A"]), f(s["B"]), [f(M) for M in s["ABi"]])
    assert abs(idx["S1"][0] - 0.9) < 0.05 and abs(idx["S1"][1] - 0.1) < 0.05
    assert max(idx["ST"][2:]) < 0.05


def test_sensitivity_outputs_are_deterministic():
    from model.sensitivity import _evaluate_model
    f = np.zeros(8)
    a, b = _evaluate_model(f), _evaluate_model(f)
    assert a == b and all(np.isfinite(v) for v in a.values())


# ---------------------------------------------------------------- run_model_paper.py
def test_paper_figures_script_runs_and_skips_missing_inputs(tmp_path):
    from experiments.run_model_paper import run
    man = run(out=str(tmp_path), profiles=False, seed=1)
    made = set(man["made"])
    # closed-form parts never depend on pipeline outputs
    assert {"table1_steady_states", "fig1_highpass_qoc", "fig2_m1_stationary", "fig6_control_trajectories"} <= made
    for name in made:
        assert (tmp_path / f"{name}.png").exists() or (tmp_path / f"{name}.csv").exists()
    assert all(isinstance(why, str) and why for _, why in man["skipped"])


# ---------------------------------------------------------------- config.py (shared settings)
def test_shared_config_is_consistent_across_packages():
    from model import config as CFG
    from model import control, design, sensitivity
    CFG.check_nominal_matches_model()
    # reference design is one cell of the grid and of the representative cells
    R = CFG.REFERENCE
    assert R.n_subjects in CFG.GRID.n_subjects and R.session_min in CFG.GRID.session_min
    assert R.probe_every_min in CFG.GRID.probe_every_min and R.r_obs in CFG.GRID.r_obs
    assert {"N": R.n_subjects, "session_min": R.session_min, "probe_every_min": R.probe_every_min,
            "r_obs": R.r_obs} in CFG.GRID.cells()
    assert design.DS001787_DESIGN["N"] == R.n_subjects and design.REPRESENTATIVE_CELLS == CFG.GRID.cells()
    # control horizon = reference session; sensitivity CRB output uses the reference design
    assert CFG.CONTROL.n_epochs == R.n_epochs == 270
    assert sensitivity.CRB_DESIGN["n_subjects"] == R.n_subjects
    assert sensitivity.CRB_DESIGN["session_min"] == R.session_min
    assert control.DT_DEFAULT == CFG.DT
    snap = CFG.as_dict()
    assert snap["seed"] == 42 and snap["control"]["target_qoc"] == 2.0


def test_sobol_bootstrap_ci_and_log_scale_and_resave_roundtrip(tmp_path):
    import pandas as pd
    from model import config as CFG
    from model.sensitivity import (SENSITIVITY_PARAMS, evaluations_table, indices_from_evaluations,
                                   saltelli_sample, sobol_indices)
    s = saltelli_sample(n_base=512, seed=1)                      # power of 2 -> balanced Sobol sequence
    w = np.array([3.0, 1.0, 0, 0, 0, 0, 0, 0])
    f = lambda X: X @ w
    idx = sobol_indices(s, "lin", f(s["A"]), f(s["B"]), [f(M) for M in s["ABi"]], n_boot=300, seed=0)
    assert idx["scale"] == "raw" and idx["ST_ci"][0][0] <= 0.9 <= idx["ST_ci"][0][1]
    assert idx["max_abs_dST_half"] < 0.1
    # multiplicative output: log scale makes the index of an additive-in-log model clean
    g = lambda X: np.exp(2.0 * X[:, 2] + 0.5 * X[:, 3])
    idx_log = sobol_indices(s, "crb_k_inh", g(s["A"]), g(s["B"]), [g(M) for M in s["ABi"]], n_boot=100, seed=0)
    assert idx_log["scale"] == "log" and abs(idx_log["S1"][2] - 16 / 16.25) < 0.08
    # saved-evaluations round trip reproduces the indices exactly
    evals = [{"lin": float(v), "crb_k_inh": float(u)} for v, u in zip(
        np.concatenate([f(s["A"]), f(s["B"])] + [f(M) for M in s["ABi"]]),
        np.concatenate([g(s["A"]), g(s["B"])] + [g(M) for M in s["ABi"]]))]
    table = evaluations_table(s, evals, ["lin", "crb_k_inh"])
    table.to_csv(tmp_path / "e.csv", index=False)
    back = indices_from_evaluations(pd.read_csv(tmp_path / "e.csv"), n_boot=10, seed=0, verbose=False)
    assert np.allclose(back["results"]["lin"]["ST"], idx["ST"]) and back["n_base"] == 512
    assert set(table.columns) >= {"block", "row", *[f"f_{p}" for p in SENSITIVITY_PARAMS], "lin"}


def test_robustness_uses_each_cells_noise_level_for_the_truth():
    """A perfectly estimated parameter set must give ~0 % loss, whatever the cell's r_obs."""
    import json
    import pandas as pd
    from model.control import control_efficiency_vs_param_error
    m = StateSpaceDMN()
    rows = []
    for r in (0.05, 0.2):
        exact = {k: getattr(m, k) for k in StateSpaceDMN.FREE}
        exact["r_A"] = exact["r_Phi"] = r
        rows.append({"N": 12, "session_min": 45.0, "probe_every_min": 2.0, "r_obs": r,
                     "mle_k_inh": 0.0, "mle_A0": 0.0, "mle_alpha": 0.0, "mle_beta": 0.0, "fits": json.dumps([exact])})
    out = control_efficiency_vs_param_error(m, pd.DataFrame(rows), n_trials=3, n_epochs=200, seed=0)
    assert all(abs(o["efficiency_loss_pct_median"]) < 1e-9 for o in out)
    assert out[0]["cost_true"] != out[1]["cost_true"]
