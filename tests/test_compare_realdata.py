"""experiments/compare_realdata.py: the per-dataset column, the replication
checks R1-R3 and the joint fit, on synthetic summaries and sessions."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bqi.dmn_ode import DeviationDMN
from experiments import compare_realdata as C
from model import realdata as R
from model.fisher import IrregularDesign


def _summary(tau_inh, ci, alpha=10.0, h5=-1.0, obs_ratio=2.0):
    k = 1 / tau_inh
    return {"data": {"n_kept": 4, "n_sessions_total": 5, "n_subjects_kept": 4, "n_rated_probes_kept": 100,
                     "kept": {"expert": 2, "novice": 2}},
            "theta_hat": {"k_inh": k, "k_int": 0.0, "k_d": 0.2, "alpha": alpha, "beta": 0.1, "r_Q": 0.5},
            "time_constants_min": {"1/k_inh": tau_inh, "1/k_d": 5.0},
            "profiles": {"k_inh": {"time_constant_ci_min": ci}, "k_d": {"time_constant_ci_min": [3.0, 8.0]}},
            "errors": {"k_inh": {"se_fisher": 0.1, "se_observed": 0.2, "rmse_boot": 0.1, "crb_pred_E1": 0.09, "ratio_rmse_se": 1.0}},
            "E_H1": {"se_profile_log": 0.2, "supported_fisher": True},
            "E_H2": {"supported": False, "delta_lpd": {n: {"mean": 0.0} for n in ("a_M1_vs_N2", "b_M1_vs_N1", "b_M1_vs_N3")}},
            "E_H3": {"ci95_fisher": [1.0, 20.0], "supported": True}, "E_H4": {"ci95_fisher": [-0.1, 0.3], "sign_determined": False},
            "E_H5": {"E_H5": {"diff": h5, "ci95": [h5 - 0.3, h5 + 0.3], "supported": h5 - 0.3 > 0,
                              "opposite_direction": h5 + 0.3 < 0, "mde_log": 0.4}},
            "observed_vs_expected": {"k_inh": {"real": obs_ratio, "simulated": 1.05}, "sigma_A": {"real": obs_ratio},
                                     "r_A": {"real": obs_ratio}},
            "snr": {"fitted": {"A": 1.0, "Phi": 0.1}},
            "residuals": {"ljung_box_p_A_median": 0.3, "ljung_box_p_Phi_median": 0.4},
            "sensitivity": {"rule": {"alpha_sign_kept": 6, "beta_sign_kept": 4, "tau_inh_in_ci": 5, "n_locked_variants": 7}},
            "subjects": {"meta_time_constant_min": tau_inh, "meta_time_constant_ci_min": ci, "n_in_meta": 4, "meta": {"tau": 0.5}},
            "gates": {"F3": {"passed": True}}, "lock": {"locked_utc": "2026-11-03T09:21:17+00:00"}}


def test_column_and_replication_checks():
    c1 = C.column(_summary(0.76, [0.49, 1.17]), sd_r=1.1)
    c2 = C.column(_summary(0.9, [0.6, 1.4], alpha=20.0, h5=-0.8), sd_r=5.5)
    assert abs(c1["alpha_per_sd"] - 10 / 1.1) < 1e-9 and abs(c2["alpha_per_sd"] - 20 / 5.5) < 1e-9
    assert c1["H5_opposite_direction"] and c1["sensitivity_tau_inh_in_ci"] == "5/7"
    rep = C.replication_checks(c1, c2)
    assert rep["R1_tau_inh"]["replicated"] and abs(rep["R1_tau_inh"]["ratio_point_estimates_osf_over_ds001787"] - 0.9 / 0.76) < 1e-9
    assert rep["R2_misspecification"]["replicated"] and rep["H5_direction"]["same_sign"]
    c3 = C.column(_summary(3.0, [2.0, 4.5], obs_ratio=1.1), sd_r=5.5)
    rep2 = C.replication_checks(c1, c3)
    assert not rep2["R1_tau_inh"]["replicated"] and not rep2["R2_misspecification"]["replicated"]


def test_joint_fit_two_datasets():
    truth = DeviationDMN(**R.RD.nominal())
    sess = []
    for k in range(4):
        n = 80
        t = (np.arange(n) + 1) / 6
        probes = tuple(np.round(np.arange(1.2, t[-1] - 0.3, 1.6), 1))
        d = IrregularDesign(tuple(t), (True,) * n, (True,) * n, probes)
        sess.append(R.simulate_session(truth, d, seed=10 + k) | {"sub": f"s{k}", "ses": "x", "group": "expert"})
    s1, _ = R.normalise_sessions(sess[:2])
    s2, _ = R.normalise_sessions([{**s, "rating": 3 * np.asarray(s["rating"])} for s in sess[2:]])   # a wider scale
    out = C.joint_fit(s1, s2, truth, seed=0, verbose=False)
    assert abs(out["rating_sd"]["osf3uszv"] / out["rating_sd"]["ds001787"] - 3) < 0.5
    assert set(out["k_inh_per_dataset"]) == {"ds001787", "osf3uszv"} and all(v > 0 for v in out["k_inh_per_dataset"].values())
    assert out["loglik"] >= out["common"]["loglik"] - 1e-6 and 0 <= out["lr_k_inh_differs"]["p"] <= 1
