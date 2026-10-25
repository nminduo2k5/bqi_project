"""
Figures, tables and numbers for the real-data section of the model paper
(Q1_pipeline.md Sec 11: Sec. 4.7, Fig. 9-10, Tables 10-11, Appendix I).

    python main.py realdata paper           # after fit, groups, boot and check
    python experiments/run_realdata.py [--out outputs/paper/realdata] [--n-ppc 200]

Inputs: outputs/results/realdata/{fit,boot,check}_results.json, e1_design_crb.json,
the feature files and outputs/gates/E*.json. Nothing is refitted. The script adds
only quantities that the preregistration asks for and that need no refit:
  - the CRB recomputed at the fitted SNR of the DMN channel (E-H1),
  - the SE from the observed information (numerical Hessian) beside the Fisher SE,
  - profile-likelihood and bootstrap-percentile intervals,
  - the posterior predictive check of the ordinal ratings (prereg Sec 3 item 8).
Every number quoted in the paper is written to summary.json; figures as PNG + PDF,
tables as CSV (the LaTeX tables are typeset by hand from summary.json).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bqi.dmn_ode import DeviationDMN
from eeg import gates
from eeg.model_fit import kalman_filter_irregular
from experiments.run_model_paper import BLUE, GRAY, MAGENTA, ORANGE, Writer
from model import config as CFG
from model import realdata as R

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEFAULT_OUT = os.path.join(ROOT, "outputs", "paper", "realdata")
KEY = list(R.RD.key_params)                       # k_inh, k_d, alpha, beta
TEX = {"k_inh": r"$k_{\mathrm{inh}}$", "k_int": r"$\kappa$", "k_d": r"$k_d$", "alpha": r"$\alpha$",
       "beta": r"$\beta$", "sigma_A": r"$\sigma_A$", "sigma_Phi": r"$\sigma_\Phi$", "r_A": r"$r_A$",
       "r_Phi": r"$r_\Phi$", "r_Q": r"$r_Q$"}


def _json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _f(x):
    """Plain floats for JSON."""
    if isinstance(x, dict):
        return {k: _f(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_f(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


# =============================================================================
# numbers
# =============================================================================
def profile_interval(prof: dict, level_drop: float = 1.92) -> dict:
    """95 % profile-likelihood interval (drop of chi2_1(0.95)/2 from the maximum),
    interpolated on the log grid; open ends are reported as None."""
    g = np.asarray(prof["grid"], float)
    ll = np.asarray(prof["loglik"], float)
    drop = ll.max() - ll
    i0 = int(np.argmin(drop))
    x = np.log(g)
    lo = float(np.exp(np.interp(level_drop, drop[:i0 + 1][::-1], x[:i0 + 1][::-1]))) if drop[0] >= level_drop else None
    hi = float(np.exp(np.interp(level_drop, drop[i0:], x[i0:]))) if drop[-1] >= level_drop else None
    se_log = (np.log(hi) - np.log(lo)) / (2 * 1.96) if lo and hi else None
    return {"mle_grid": float(g[i0]), "ci95": [lo, hi], "se_log": se_log, "max_drop": float(drop.max()),
            "unimodal": bool(prof.get("unimodal"))}


def prep_summary() -> dict:
    logs = [_json(p) for p in sorted(glob.glob(os.path.join(R.FEATURES, "*_prep_log.json")))]
    nb = [len(l.get("bad_channels") or []) for l in logs]
    nr = [len(l.get("ica_rejected") or []) for l in logs]
    return {"n_sessions": len(logs), "bad_channels": {"median": float(np.median(nb)), "max": int(max(nb))},
            "ica_rejected": {"median": float(np.median(nr)), "min": int(min(nr)), "max": int(max(nr))},
            "ica_methods": sorted({str(l.get("ica_method")) for l in logs}),
            "missing_frac_locked": {"median": float(np.median([l["missing_frac"] for l in logs]))},
            "missing_frac_adaptive": {"median": float(np.median([l["missing_frac_adaptive"] for l in logs]))}}


def summarise(fr, br, cr, e1, sessions, info, model, n_ppc: int, seed: int, verbose: bool = True) -> dict:
    S: dict = {}
    pooled = fr["pooled"]
    # ---- data, stop criteria, gates, lock -------------------------------------
    kept = {g: sum(s["group"] == g for s in sessions) for g in ("expert", "novice")}
    dropped = {g: sum(d["group"] == g for d in info["dropped"]) for g in ("expert", "novice")}
    S["data"] = {"n_sessions_total": e1["n_sessions"], "n_probes_total": e1["n_probes"],
                 "subject_minutes": e1["subject_minutes"], "kept": kept, "dropped": dropped,
                 "n_kept": len(sessions), "n_rated_probes_kept": int(sum(np.isfinite(s["rating"]).sum() for s in sessions)),
                 "n_epochs_kept": int(sum(len(s["t"]) for s in sessions)),
                 "missing_frac_kept_median": float(np.median([s["missing_frac"] for s in sessions])),
                 "n_subjects_kept": len({s["sub"] for s in sessions}),
                 "dropped_list": [{"session": f"{d['sub']} {d['ses']}", "group": d["group"],
                                   "missing_frac": d["missing_frac"], "n_rated": d["n_probes_rated"]}
                                  for d in info["dropped"]]}
    S["prep"] = prep_summary()
    S["stop_checks"] = {k: v for k, v in fr["stop_checks"].items() if k != "per_session_lag1"}
    S["gates"] = {}
    for g in ("E0", "E1", "E2", "E3"):
        p = os.path.join(gates.GATE_DIR, f"{g}.json")
        if os.path.isfile(p):
            j = _json(p)
            S["gates"][g] = {"passed": j.get("passed"), "time_utc": j.get("time_utc")}
    lock = gates.read_lock(gates.LOCK_REALDATA)
    man = pd.read_csv(os.path.join(ROOT, R.RD.data_dir, "manifest.csv"))
    S["lock"] = {"locked_utc": lock["locked_utc"], "first_bdf_download_utc": str(man["downloaded_utc"].min()),
                 "sha256_matches": lock["sha256"] == gates._sha256(gates.PREDICTIONS_REALDATA)}
    # ---- pooled fit -------------------------------------------------------------
    S["theta_hat"] = pooled["params"]
    S["loglik"] = pooled["loglik"]
    S["time_constants_min"] = {"1/k_inh": 1 / model.k_inh, "1/k_d": 1 / model.k_d}
    S["snr"] = {"fitted": R.snr(model), "nominal": R.snr(R.nominal_model())}
    S["profiles"] = {p: profile_interval(v) for p, v in fr["profiles"].items()}
    for p in ("k_inh", "k_d"):
        ci = S["profiles"][p]["ci95"]
        S["profiles"][p]["time_constant_ci_min"] = [1 / ci[1], 1 / ci[0]] if all(ci) else None
    # ---- three error numbers (+ CRB at fitted SNR, observed information) --------
    if verbose:
        print("  CRB at nominal and at the fitted SNR on the kept designs...", flush=True)
    crb_nom_kept = R.crb_on_sessions(R.nominal_model(), sessions)
    snr_fit = S["snr"]["fitted"]["A"]
    crb_snr_kept = R.crb_on_sessions(R.nominal_at_snr(snr_fit), sessions)
    if verbose:
        print("  observed information (numerical Hessian)...", flush=True)
    obs = R.observed_information_se(model, sessions)
    boots = br["boot_params"]
    rows = {}
    for k in model.FREE:
        v = np.array([b[k] for b in boots], float)
        rows[k] = {"estimate": pooled["params"][k], "crb_pred_E1": e1["crb"]["pooled"][k],
                   "crb_nominal_kept": crb_nom_kept[k], "crb_fitted_snr_kept": crb_snr_kept[k],
                   "se_fisher": pooled["rel_se"][k], "se_observed": obs["params"][k]["rel_se"],
                   "rmse_boot": br["rmse_bootstrap"][k], "ratio_rmse_se": br["ratio_rmse_over_se"][k],
                   "boot_p2.5": float(np.percentile(v, 2.5)), "boot_p50": float(np.median(v)),
                   "boot_p97.5": float(np.percentile(v, 97.5)),
                   "abs_se_fisher": pooled["se"][k], "abs_se_observed": obs["params"][k]["se"]}
    S["errors"] = rows
    S["observed_information_pd"] = obs["positive_definite"]
    # control: one data set simulated from theta_hat on the same real designs; under a correct
    # model the observed information (at the true parameter) matches the expected one
    if verbose:
        print("  control: observed vs expected information on data simulated from theta_hat...", flush=True)
    sims = R.simulate_sessions(model, [R.design_of(s) for s in sessions], seed + 12345)
    obs_sim = R.observed_information_se(model, sims)
    S["observed_vs_expected"] = {k: {"real": rows[k]["se_observed"] / rows[k]["se_fisher"],
                                     "simulated": obs_sim["params"][k]["rel_se"] / rows[k]["se_fisher"]}
                                 for k in model.FREE}
    # ---- hypotheses ------------------------------------------------------------
    a, sa = pooled["params"]["alpha"], pooled["se"]["alpha"]
    b, sb = pooled["params"]["beta"], pooled["se"]["beta"]
    S["E_H1"] = {"rel_se_fisher": pooled["rel_se"]["k_inh"], "ratio_rmse_se": br["ratio_rmse_over_se"]["k_inh"],
                 "snr_fitted": snr_fit, "snr_nominal": S["snr"]["nominal"]["A"],
                 "crb_pred_E1": e1["crb"]["pooled"]["k_inh"], "crb_fitted_snr_kept": crb_snr_kept["k_inh"],
                 "se_profile_log": S["profiles"]["k_inh"]["se_log"], "se_observed": obs["params"]["k_inh"]["rel_se"],
                 "supported_fisher": bool(pooled["rel_se"]["k_inh"] <= 0.20
                                          and R.RD.e2_ratio_bounds[0] <= br["ratio_rmse_over_se"]["k_inh"] <= R.RD.e2_ratio_bounds[1])}
    S["E_H3"] = {"alpha": a, "ci95_fisher": [a - 1.96 * sa, a + 1.96 * sa],
                 "boot_interval": [rows["alpha"]["boot_p2.5"], rows["alpha"]["boot_p97.5"]],
                 "boot_n_nonpositive": int(sum(bb["alpha"] <= 0 for bb in boots)),
                 "corr_log_sigmaPhi_alpha_boot": float(np.corrcoef(np.log([bb["sigma_Phi"] for bb in boots]),
                                                                   [bb["alpha"] for bb in boots])[0, 1]),
                 "supported": bool(a - 1.96 * sa > 0)}
    S["E_H4"] = {"beta": b, "ci95_fisher": [b - 1.96 * sb, b + 1.96 * sb],
                 "sign_determined": bool((b - 1.96 * sb) * (b + 1.96 * sb) > 0)}
    g = fr["groups"]
    S["E_H5"] = {k: g[k] for k in ("E_H5", "log_k_inh_difference", "lr_all_params") if k in g}
    if "lr_k_inh_1df" in g:
        S["E_H5"]["lr_k_inh_1df"] = {k: v for k, v in g["lr_k_inh_1df"].items() if k != "joint_fit"}
        S["E_H5"]["joint_fit_k_inh"] = g["lr_k_inh_1df"]["joint_fit"]["k_inh"]
    S["E_H5"]["groups"] = {k: {"k_inh": v["params"]["k_inh"], "se_log_k_inh": v["se_log_k_inh"],
                               "r_A": v["params"]["r_A"], "sigma_A": v["params"]["sigma_A"],
                               "snr_A": R.snr(DeviationDMN(**v["params"]))["A"], "n_sessions": v["n_sessions"]}
                           for k, v in g["groups"].items()}
    subj = fr["subjects"]
    per = subj["per_subject"]
    S["subjects"] = {"n_subjects": len(per), "n_in_meta": subj["meta_log_k_inh"].get("n"),
                     "n_sigmaPhi_boundary": sum("sigma_Phi" in (v.get("at_boundary") or []) for v in per.values()),
                     "meta": subj["meta_log_k_inh"],
                     "k_inh_range_in_meta": [min(v["params"]["k_inh"] for v in per.values()
                                                 if np.isfinite(v["se_log_k_inh"]) and v["se_log_k_inh"] < 5),
                                             max(v["params"]["k_inh"] for v in per.values()
                                                 if np.isfinite(v["se_log_k_inh"]) and v["se_log_k_inh"] < 5)]}
    m = subj["meta_log_k_inh"]
    if m:
        lo, hi = m["mu_log_k_inh"] - 1.96 * m["se_mu"], m["mu_log_k_inh"] + 1.96 * m["se_mu"]
        S["subjects"]["meta_time_constant_ci_min"] = [float(np.exp(-hi)), float(np.exp(-lo))]
        S["subjects"]["meta_time_constant_min"] = float(np.exp(-m["mu_log_k_inh"]))
    S["E_H2"] = cr["loso"]
    d = cr["loso"]["delta_lpd"]
    S["E_H2"]["supported"] = bool(d["a_M1_vs_N2"]["wins"] and d["b_M1_vs_N1"]["wins"] and d["b_M1_vs_N3"]["wins"])
    # ---- model checks ----------------------------------------------------------
    resid = R.residual_checks(model, sessions)
    thr = R.RD.e3_min_lb_p
    S["residuals"] = {**cr["residuals"],
                      "n_sessions_p_below_thr_A": int(np.sum(np.array(resid["per_session"]["lb_A"]) < thr)),
                      "n_sessions_p_below_thr_Phi": int(np.sum(np.array(resid["per_session"]["lb_Phi"]) < thr)),
                      "threshold": thr}
    if verbose:
        print(f"  posterior predictive check of the ratings ({n_ppc} replicates)...", flush=True)
    S["ppc"] = R.posterior_predictive_ratings(model, sessions, n_rep=n_ppc, seed=seed)
    # ---- sensitivity -----------------------------------------------------------
    loaders = R.default_sensitivity_loaders(R.FEATURES)
    ci_k = S["profiles"]["k_inh"]["time_constant_ci_min"]
    sens = {}
    for name, v in cr["sensitivity"].items():
        e = {"n_sessions": len(loaders[name]()) if name in loaders else None, **v}
        if "params" in v:
            e["sign_alpha_same"] = bool(np.sign(v["params"]["alpha"]) == np.sign(a))
            e["sign_beta_same"] = bool(np.sign(v["params"]["beta"]) == np.sign(b))
            e["1/k_d"] = 1 / v["params"]["k_d"]
            e["tau_inh_in_main_ci"] = bool(ci_k[0] <= v["1/k_inh"] <= ci_k[1]) if ci_k else None
        sens[name] = e
    locked = {k: v for k, v in sens.items() if not k.startswith("postreg_") and "params" in v}
    S["sensitivity"] = {"variants": sens, "rule": {
        "n_locked_variants": len(locked),
        "alpha_sign_kept": sum(v["sign_alpha_same"] for v in locked.values()),
        "beta_sign_kept": sum(v["sign_beta_same"] for v in locked.values()),
        "tau_inh_in_ci": sum(bool(v["tau_inh_in_main_ci"]) for v in locked.values()),
        "threshold_fraction": 0.8}}
    return _f(S)


# =============================================================================
# figures and tables
# =============================================================================
def example_session(sessions: list[dict]) -> dict:
    """The kept session with the most rated probes (fixed rule, no choice by eye)."""
    return sorted(sessions, key=lambda s: (-int(np.isfinite(s["rating"]).sum()), s["sub"], s["ses"]))[0]


def fig9_session(w: Writer, model: DeviationDMN, sessions: list[dict]) -> dict:
    s = example_session(sessions)
    _, inf_b = kalman_filter_irregular(model, s, return_innovations=True)                     # EEG + ratings
    _, inf_a = kalman_filter_irregular(model, s, return_innovations=True, use_rating=False)   # EEG only
    F = inf_b["filtered"]
    t, mA, mP = F[:, 0], F[:, 2], F[:, 3]
    sA, sP = np.sqrt(np.clip(F[:, 4], 0, None)), np.sqrt(np.clip(F[:, 6], 0, None))
    te, y = np.asarray(s["t"]), np.asarray(s["y"])
    pb, pa = inf_b["pred_rating"], inf_a["pred_rating"]
    fig, ax = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=True)
    n_clip = 0
    for k, (m, sd, col, lab, ylab) in enumerate(((mA, sA, BLUE, r"$A$", r"$y_A$ (alpha, SD units)"),
                                                 (mP, sP, MAGENTA, r"$\Phi$", r"$y_\Phi$ (LZc, SD units)"))):
        ax[k].plot(te, y[:, k], ".", ms=2.2, color=GRAY, label="observed (gaps: missing epochs)")
        ax[k].fill_between(t, m - 2 * sd, m + 2 * sd, color=col, alpha=0.18, lw=0)
        ax[k].plot(t, m, color=col, lw=1.0, label=f"filtered {lab} $\\pm$ 2 SD")
        ax[k].set_ylabel(ylab)
        ax[k].legend(frameon=False, loc="upper left", ncol=2, fontsize=7.5)
    lo_a, hi_a = np.nanmin(y[:, 0]), np.nanmax(y[:, 0])
    ax[0].set_ylim(lo_a - 0.3, hi_a + 1.4)                      # head-room for the legend
    lim_p = (-4.0, 4.0)                                        # a few LZc outliers would flatten the panel
    yp = y[:, 1]
    n_clip = int(np.sum((yp < lim_p[0]) | (yp > lim_p[1])))
    ax[1].set_ylim(*lim_p)
    if n_clip:
        ax[1].text(0.995, 0.03, f"{n_clip} epochs outside the axis range", transform=ax[1].transAxes,
                   ha="right", va="bottom", fontsize=7, color=GRAY)
    ax[2].errorbar(pb[:, 0], pb[:, 1], yerr=2 * np.sqrt(pb[:, 2]), fmt="o", ms=3, color=BLUE, lw=0.8, capsize=0,
                   label=r"predicted, EEG + past ratings ($\pm$ 2 SD)")
    ax[2].plot(pa[:, 0], pa[:, 1], "s", ms=3, mfc="none", color=ORANGE, label="predicted, EEG only")
    ax[2].plot(pb[:, 0], pb[:, 3], "k.", ms=6, label="observed")
    ax[2].axhline(0, color=GRAY, lw=0.5)
    ax[2].set(ylabel="rating $-$ session mean", xlabel="time in session (min)")
    top = max(np.nanmax(pb[:, 3]), np.nanmax(pb[:, 1] + 2 * np.sqrt(pb[:, 2])))
    ax[2].set_ylim(top=top + 1.6)
    ax[2].legend(frameon=False, loc="upper left", ncol=3, fontsize=7)
    ax[0].set_title(f"{s['sub']} {s['ses']} ({s['group']}): {int(np.isfinite(s['rating']).sum())} rated probes, "
                    f"{100 * s['missing_frac']:.0f} % missing epochs", fontsize=9)
    fig.tight_layout()
    w.fig(fig, "fig9_realdata_session")
    return {"session": f"{s['sub']} {s['ses']}", "group": s["group"], "n_rated": int(np.isfinite(s["rating"]).sum()),
            "missing_frac": s["missing_frac"], "duration_min": float(te[-1]), "n_yPhi_outside_axis": n_clip,
            "yPhi_axis": list(lim_p)}


def fig10_profiles(w: Writer, fr: dict) -> None:
    from matplotlib.ticker import NullFormatter
    pooled = fr["pooled"]
    params = list(R.RD.profile_params)
    fig, axes = plt.subplots(1, len(params), figsize=(7.2, 2.5))
    for ax, p in zip(axes, params):
        pr = fr["profiles"][p]
        v0 = pooled["params"][p]
        x = np.asarray(pr["grid"], float) / v0
        ll = np.asarray(pr["loglik"], float)
        xx = np.exp(np.linspace(np.log(x.min()), np.log(x.max()), 200))
        se = pooled["se"][p]
        q = -0.5 * (np.log(xx) / se) ** 2 if p in DeviationDMN.POSITIVE else -0.5 * ((xx - 1) * v0 / se) ** 2
        ax.plot(xx, q, "--", color=GRAY, lw=0.8, label="Fisher quadratic")
        ax.plot(x, ll - ll.max(), "o-", color=BLUE, ms=3.5, label="profile")
        ax.axhline(-1.92, color=ORANGE, ls="--", lw=0.9, label=r"$\chi^2_1$ 95 %")
        ax.axvline(1.0, color=GRAY, ls=":", lw=0.8)
        ax.set_xscale("log")
        ticks = [0.5, 0.7, 1.0, 1.4, 2.0] if x.min() > 0.4 else [0.25, 0.5, 1.0, 1.8]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(-10, 0.6)
        ax.set(xlabel=f"{TEX[p]} / MLE", title=f"profile of {TEX[p]} (data)")
    axes[0].set_ylabel(r"$\ell_p - \ell_{\max}$")
    axes[0].legend(frameon=False, loc="lower center", fontsize=7)
    fig.tight_layout()
    w.fig(fig, "fig10_realdata_profiles")


def _csv(w: Writer, df: pd.DataFrame, name: str) -> None:
    df.to_csv(os.path.join(w.out, f"{name}.csv"))
    w.made.append(name)
    print(f"  {name}.csv")


def tables(w: Writer, S: dict) -> None:
    _csv(w, pd.DataFrame(S["errors"]).T, "table10_errors")
    L = S["E_H2"]
    rows = []
    for name, (a, b) in {"a_M1_vs_N2": ("lpd_M1_a", "lpd_N2"), "b_M1_vs_N1": ("lpd_M1_b", "lpd_N1"),
                         "b_M1_vs_N3": ("lpd_M1_b", "lpd_N3")}.items():
        dl = L["delta_lpd"][name]
        rows.append({"comparison": name, "lpd_M1": L["mean_lpd"][a], "lpd_null": L["mean_lpd"][b],
                     "delta": dl["mean"], "ci_lo": dl["ci95"][0], "ci_hi": dl["ci95"][1], "wins": dl["wins"],
                     "mae_M1": L["mae"][a.replace("lpd", "ae")], "mae_null": L["mae"][b.replace("lpd", "ae")]})
    _csv(w, pd.DataFrame(rows).set_index("comparison"), "table11_loso")
    sens = pd.DataFrame({k: {"n_sessions": v.get("n_sessions"), **(v.get("params") or {}),
                             "1/k_inh": v.get("1/k_inh"), "1/k_d": v.get("1/k_d")}
                         for k, v in S["sensitivity"]["variants"].items()}).T
    _csv(w, sens, "tableI1_sensitivity")
    p = S["ppc"]["level_prop"]
    _csv(w, pd.DataFrame({"level": S["ppc"]["levels"], "observed": p["observed"], "sim_mean": p["sim_mean"],
                          "sim_lo": p["sim_lo"], "sim_hi": p["sim_hi"]}).set_index("level"), "tableI2_ppc_levels")


# =============================================================================
def run(out: str = DEFAULT_OUT, n_ppc: int = 200, seed: int = CFG.SEED, verbose: bool = True) -> dict:
    gates.require("E3")
    d = R.RESULTS
    fr, br, cr = (_json(os.path.join(d, f"{n}_results.json")) for n in ("fit", "boot", "check"))
    e1 = _json(os.path.join(d, "e1_design_crb.json"))
    sessions, info = R.load_feature_sessions()
    model = DeviationDMN(**fr["pooled"]["params"])
    w = Writer(out)
    print(f"Hinh/bang phan du lieu that -> {out}", flush=True)
    S = summarise(fr, br, cr, e1, sessions, info, model, n_ppc, seed, verbose)
    S["fig9_session"] = fig9_session(w, model, sessions)
    fig10_profiles(w, fr)
    tables(w, S)
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(S, f, indent=2, ensure_ascii=False)
    manifest = {"made": w.made + ["summary"], "skipped": w.skipped, "seed": seed, "n_ppc": n_ppc,
                "gates": {g: v.get("passed") for g, v in S["gates"].items()}}
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Xong: {len(w.made)} tep + summary.json", flush=True)
    return S


def main(argv=None):
    p = argparse.ArgumentParser(description="Figures/tables for the real-data section of the model paper.")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--n-ppc", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    run(a.out, a.n_ppc, a.seed)


if __name__ == "__main__":
    main()
