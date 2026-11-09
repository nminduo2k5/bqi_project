"""
The two real datasets side by side (Q1_pipeline.md Sec 16): ds001787 and OSF
3uszv, each fitted on its own by model.realdata with its own preregistration.

    python main.py osf3uszv compare [--joint]
    python experiments/compare_realdata.py [--joint] [--out outputs/paper/comparison]

Inputs: outputs/paper/{realdata,osf3uszv}/summary.json (from the `paper`
step of each dataset) and the feature sessions (for the rating SD). Outputs:
comparison.csv / comparison.json with
  - one column per dataset: sessions, probes, 1/k_inh and 1/k_d with profile
    intervals, alpha and beta also in rating-SD units (the two rating scales
    differ: -3..3 and -7..7), SNR, the error numbers, the hypothesis outcomes;
  - the replication checks R1-R3 of predictions_osf3uszv.md Sec 2;
  - with --joint: the exploratory joint fit of both datasets with a
    dataset-specific k_inh and everything else shared, ratings in SD units.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bqi.dmn_ode import DeviationDMN
from model import config as CFG
from model import realdata as R

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEFAULT_OUT = os.path.join(ROOT, "outputs", "paper", "comparison")
PAPER = {"ds001787": os.path.join(ROOT, "outputs", "paper", "realdata", "summary.json"),
         "osf3uszv": os.path.join(ROOT, "outputs", "paper", "osf3uszv", "summary.json")}


def _json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def feature_sessions(dataset: str) -> list[dict]:
    if dataset == "ds001787":
        return R.load_feature_sessions(R.FEATURES, artifact_rule="locked")[0]
    from model import osf_pipeline as OP
    return R.load_feature_sessions(OP.FEATURES, artifact_rule=CFG.OSF.artifact_rule)[0]


def rating_sd(sessions: list[dict]) -> float:
    """Pooled within-session SD of the ratings (they are already session-centred)."""
    r = np.concatenate([np.asarray(s["rating"], float) for s in sessions])
    return float(np.sqrt(np.nanmean(r ** 2)))


def _ci(x):
    return None if x is None or any(v is None for v in x) else [float(x[0]), float(x[1])]


def column(S: dict, sd_r: float | None) -> dict:
    """The numbers of one dataset in a flat dict (same keys for both)."""
    th, er = S["theta_hat"], S["errors"]
    pk, pd_ = S["profiles"]["k_inh"], S["profiles"]["k_d"]
    h5 = S["E_H5"].get("E_H5", {})
    d = S["E_H2"]["delta_lpd"]
    sens = S["sensitivity"]["rule"]
    out = {
        "n_sessions_kept": S["data"]["n_kept"], "n_sessions_total": S["data"]["n_sessions_total"],
        "n_subjects_kept": S["data"]["n_subjects_kept"], "n_rated_probes": S["data"]["n_rated_probes_kept"],
        "kept_expert": S["data"]["kept"].get("expert"), "kept_novice": S["data"]["kept"].get("novice"),
        "tau_inh_min": S["time_constants_min"]["1/k_inh"], "tau_inh_ci95_min": _ci(pk.get("time_constant_ci_min")),
        "tau_d_min": S["time_constants_min"]["1/k_d"], "tau_d_ci95_min": _ci(pd_.get("time_constant_ci_min")),
        "k_int": th["k_int"], "alpha": th["alpha"], "alpha_ci95": _ci(S["E_H3"]["ci95_fisher"]),
        "beta": th["beta"], "beta_ci95": _ci(S["E_H4"]["ci95_fisher"]),
        "rating_sd_within": sd_r,
        "alpha_per_sd": th["alpha"] / sd_r if sd_r else None, "beta_per_sd": th["beta"] / sd_r if sd_r else None,
        "r_Q_per_sd": th["r_Q"] / sd_r if sd_r else None,
        "snr_A_fitted": S["snr"]["fitted"]["A"], "snr_Phi_fitted": S["snr"]["fitted"]["Phi"],
        "k_inh_rel_se_fisher": er["k_inh"]["se_fisher"], "k_inh_rel_se_observed": er["k_inh"]["se_observed"],
        "k_inh_rel_se_profile": S["E_H1"].get("se_profile_log"), "k_inh_rmse_boot": er["k_inh"]["rmse_boot"],
        "k_inh_crb_pred": er["k_inh"]["crb_pred_E1"], "k_inh_ratio_rmse_se": er["k_inh"]["ratio_rmse_se"],
        "obs_over_exp_k_inh_real": S["observed_vs_expected"]["k_inh"]["real"],
        "obs_over_exp_k_inh_sim": S["observed_vs_expected"]["k_inh"]["simulated"],
        "obs_over_exp_sigma_A_real": S["observed_vs_expected"]["sigma_A"]["real"],
        "obs_over_exp_r_A_real": S["observed_vs_expected"]["r_A"]["real"],
        "H1_supported_fisher": S["E_H1"]["supported_fisher"], "H2_supported": S["E_H2"].get("supported"),
        "H2_delta_a_M1_vs_N2": d["a_M1_vs_N2"]["mean"], "H2_delta_b_M1_vs_N1": d["b_M1_vs_N1"]["mean"],
        "H2_delta_b_M1_vs_N3": d["b_M1_vs_N3"]["mean"],
        "H3_supported": S["E_H3"]["supported"], "H4_sign_determined": S["E_H4"]["sign_determined"],
        "H5_log_k_inh_expert_minus_novice": h5.get("diff"), "H5_ci95": _ci(h5.get("ci95")),
        "H5_supported_theory": h5.get("supported"), "H5_opposite_direction": h5.get("opposite_direction"),
        "H5_mde_log": h5.get("mde_log"),
        "ljung_box_p_A_median": S["residuals"]["ljung_box_p_A_median"],
        "ljung_box_p_Phi_median": S["residuals"]["ljung_box_p_Phi_median"],
        "sensitivity_alpha_sign_kept": f"{sens['alpha_sign_kept']}/{sens['n_locked_variants']}",
        "sensitivity_beta_sign_kept": f"{sens['beta_sign_kept']}/{sens['n_locked_variants']}",
        "sensitivity_tau_inh_in_ci": f"{sens['tau_inh_in_ci']}/{sens['n_locked_variants']}",
        "meta_tau_inh_min": S["subjects"].get("meta_time_constant_min"),
        "meta_tau_inh_ci95_min": _ci(S["subjects"].get("meta_time_constant_ci_min")),
        "meta_n": S["subjects"].get("n_in_meta"), "meta_tau": (S["subjects"].get("meta") or {}).get("tau"),
        "gates": {g: v.get("passed") for g, v in S["gates"].items()}, "locked_utc": S["lock"]["locked_utc"],
    }
    return out


def replication_checks(c1: dict, c2: dict) -> dict:
    """R1-R3 of predictions_osf3uszv.md: dataset 2 (c2) against dataset 1 (c1)."""
    ci1, ci2 = c1["tau_inh_ci95_min"], c2["tau_inh_ci95_min"]
    overlap = bool(ci1 and ci2 and ci2[0] <= ci1[1] and ci1[0] <= ci2[1])
    r2_real = all(c2[k] is not None and c2[k] > 1.5 for k in ("obs_over_exp_k_inh_real", "obs_over_exp_sigma_A_real",
                                                                "obs_over_exp_r_A_real"))
    r2_sim = c2["obs_over_exp_k_inh_sim"] is not None and 0.8 <= c2["obs_over_exp_k_inh_sim"] <= 1.25
    return {"R1_tau_inh": {"ds001787_ci95": ci1, "osf_ci95": ci2, "profile_intervals_overlap": overlap,
                           "ratio_point_estimates_osf_over_ds001787": c2["tau_inh_min"] / c1["tau_inh_min"],
                           "replicated": overlap},
            "R2_misspecification": {"osf_obs_over_exp": {k: c2[k] for k in ("obs_over_exp_k_inh_real", "obs_over_exp_sigma_A_real",
                                                                           "obs_over_exp_r_A_real", "obs_over_exp_k_inh_sim")},
                                    "replicated": bool(r2_real and r2_sim)},
            "R3_tau_d": {"ds001787_min": c1["tau_d_min"], "ds001787_ci95": c1["tau_d_ci95_min"],
                         "osf_min": c2["tau_d_min"], "osf_ci95": c2["tau_d_ci95_min"]},
            "H5_direction": {"ds001787": c1["H5_log_k_inh_expert_minus_novice"], "osf": c2["H5_log_k_inh_expert_minus_novice"],
                             "same_sign": bool(c1["H5_log_k_inh_expert_minus_novice"] is not None
                                               and c2["H5_log_k_inh_expert_minus_novice"] is not None
                                               and np.sign(c1["H5_log_k_inh_expert_minus_novice"])
                                               == np.sign(c2["H5_log_k_inh_expert_minus_novice"]))}}


def joint_fit(sess1: list[dict], sess2: list[dict], start: DeviationDMN, seed: int = CFG.SEED,
              verbose: bool = True) -> dict:
    """Exploratory: both datasets together, ratings rescaled to SD units, one
    k_inh per dataset and every other parameter shared (reuses
    realdata.fit_group_k_inh with the dataset as the group)."""
    sd1, sd2 = rating_sd(sess1), rating_sd(sess2)
    tagged = ([{**s, "group": "ds001787", "rating": np.asarray(s["rating"], float) / sd1} for s in sess1]
              + [{**s, "group": "osf3uszv", "rating": np.asarray(s["rating"], float) / sd2} for s in sess2])
    # a shared start: the given model with alpha, beta, r_Q expressed in SD units of its own dataset
    sd0 = sd2 if start is not None else 1.0
    base = start.with_params(alpha=start.alpha / sd0, beta=start.beta / sd0, r_Q=start.r_Q / sd0)
    if verbose:
        print(f"  fit chung {len(sess1)} + {len(sess2)} phien (k_inh rieng tung bo, rating theo don vi SD)...", flush=True)
    pooled = R.fit_sessions(tagged, base=base, n_starts=2, seed=seed, starts=[{k: getattr(base, k) for k in base.FREE}],
                            label="chung" if verbose else None)
    m = pooled["model"]
    jk = R.fit_group_k_inh(tagged, m, {"ds001787": m.k_inh, "osf3uszv": m.k_inh}, label="k_inh rieng" if verbose else None)
    from scipy.stats import chi2
    lr = 2 * (jk["loglik"] - pooled["loglik"])
    return {"rating_sd": {"ds001787": sd1, "osf3uszv": sd2}, "common": {"params": pooled["params"], "loglik": pooled["loglik"]},
            "k_inh_per_dataset": jk["k_inh"], "tau_inh_min_per_dataset": {k: 1 / v for k, v in jk["k_inh"].items()},
            "shared": jk["shared"], "loglik": jk["loglik"],
            "lr_k_inh_differs": {"stat": float(lr), "df": 1, "p": float(chi2.sf(lr, 1))}}


def run(out: str = DEFAULT_OUT, joint: bool = False, seed: int = CFG.SEED, verbose: bool = True) -> dict:
    os.makedirs(out, exist_ok=True)
    cols, sessions = {}, {}
    for name, path in PAPER.items():
        if not os.path.isfile(path):
            if verbose:
                print(f"  {name}: chua co {os.path.relpath(path, ROOT)} (chay buoc paper cua bo nay truoc)")
            continue
        sessions[name] = feature_sessions(name)
        cols[name] = column(_json(path), rating_sd(sessions[name]))
    res = {"datasets": cols}
    if len(cols) == 2:
        res["replication"] = replication_checks(cols["ds001787"], cols["osf3uszv"])
        if joint:
            start = DeviationDMN(**_json(PAPER["osf3uszv"])["theta_hat"])
            res["joint_fit"] = joint_fit(sessions["ds001787"], sessions["osf3uszv"], start, seed, verbose)
    df = pd.DataFrame({k: {kk: (json.dumps(vv) if isinstance(vv, (list, dict)) else vv) for kk, vv in v.items()}
                       for k, v in cols.items()})
    df.to_csv(os.path.join(out, "comparison.csv"))
    with open(os.path.join(out, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    if verbose:
        with pd.option_context("display.max_rows", 200, "display.width", 160):
            print(df)
        if "replication" in res:
            print(json.dumps(res["replication"], indent=1, default=float))
        print(f"-> {out}")
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description="ds001787 vs OSF 3uszv: comparison table, replication checks, joint fit.")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--joint", action="store_true", help="also the exploratory joint fit (slow)")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    run(a.out, a.joint, a.seed)


if __name__ == "__main__":
    main()
