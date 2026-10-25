"""
Real-data section of the model paper (Q1_pipeline.md v2.0): M1 in deviation
form fitted to ds001787 features, with the three error numbers (predicted CRB,
SE at the MLE, bootstrap RMSE), group / subject fits, residual checks and
leave-one-session-out prediction against null models. Gates E1-E3.

Steps (python main.py realdata ...):
    design-crb   E1: Rothenberg rank + CRB on the real probe schedules (no EEG needed)
    fit          pooled / group / subject fits, SE at MLE, profiles
    boot         E2: parametric bootstrap on the real designs
    check        E3: residual whiteness, LOSO vs N1-N3, sensitivity variants

Feature sessions are npz files written by eeg.realdata_features (one per
session) with: t (min), one array per feature, mask, t_probe (min), rating,
q1, q2, q3 and the meta fields.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bqi.dmn_ode import DeviationDMN
from eeg import gates
from eeg.loaders import ds001787 as DS
from eeg.model_fit import fit, kalman_filter_irregular, kalman_loglik_irregular, profile_likelihood, run_starts
from model import config as CFG
from model.checkpoint import Checkpoint, fingerprint, sessions_key
from model.fisher import IrregularDesign, crb, fisher_exact, identifiability_spectrum

RD = CFG.REALDATA
CKPT_VERSION = 1   # bump when a change to the fitting code should invalidate saved checkpoints
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "outputs", "results", "realdata")
FEATURES = os.path.join(ROOT, RD.data_dir, "derivatives", "features")


def nominal_model(rho: float = 0.0) -> DeviationDMN:
    return DeviationDMN(**RD.nominal(), rho_obs=rho)


def _save(name: str, obj: dict, output_dir: str | None = None) -> str:
    d = output_dir or RESULTS
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


# =============================================================================
# E1: structural identifiability and predicted CRB on the real designs
# =============================================================================
def fisher_sum(model: DeviationDMN, designs: list[IrregularDesign], names) -> np.ndarray:
    I = None
    for d in designs:
        Ii = fisher_exact(model, d, names)
        I = Ii if I is None else I + Ii
    return I


def design_crb(output_dir: str | None = None, verbose: bool = True, max_sessions: int | None = None) -> dict:
    """Gate E1 and the numbers for predictions_realdata.md."""
    gates.require("E0")
    trip = DS.all_designs()
    if max_sessions:
        trip = trip[:max_sessions]
    designs = [d for _, d, _ in trip]
    infos = [i for _, _, i in trip]
    groups = {g: [d for (s, d, _) in trip if s["group"] == g] for g in ("expert", "novice")}
    names10 = list(DeviationDMN.FREE)
    names11 = list(DeviationDMN.FREE_RHO)
    m0 = nominal_model(0.0)
    m_rho = nominal_model(RD.rho_generic)
    if verbose:
        print(f"E1: Fisher on {len(designs)} real sessions ({sum(i['n_probes'] for i in infos)} probes)...", flush=True)
    I10 = fisher_sum(m0, designs, names10)
    I11 = fisher_sum(m_rho, designs, names11)
    spec10 = identifiability_spectrum(I10, names10, tol=RD.e1_eig_tol)
    spec11 = identifiability_spectrum(I11, names11, tol=RD.e1_eig_tol)
    out = {"n_sessions": len(designs), "n_probes": int(sum(i["n_probes"] for i in infos)),
           "subject_minutes": float(sum(i["duration_min"] for i in infos)),
           "missing_frac_mean": float(np.mean([i["missing_frac"] for i in infos])),
           "rank_10": {k: spec10[k] for k in ("rank", "n_identifiable_candidates", "min_rel_eigenvalue", "condition_number", "null_directions")},
           "rank_11_with_rho": {k: spec11[k] for k in ("rank", "n_identifiable_candidates", "min_rel_eigenvalue", "condition_number", "null_directions")},
           "crb": {"pooled": {k: v["rel_se"] for k, v in crb(I10, names10, m0).items()}}}
    for g, ds_g in groups.items():
        if ds_g:
            Ig = fisher_sum(m0, ds_g, names10)
            out["crb"][g] = {k: v["rel_se"] for k, v in crb(Ig, names10, m0).items()}
            out["crb"][g]["_n_sessions"] = len(ds_g)
    if "expert" in out["crb"] and "novice" in out["crb"]:
        se_e, se_n = out["crb"]["expert"]["k_inh"], out["crb"]["novice"]["k_inh"]
        se_diff = float(np.hypot(se_e, se_n))
        out["group_difference_log_k_inh"] = {"se_diff": se_diff, "mde_80pct_alpha05": 2.80 * se_diff,
                                             "mde_factor": float(np.exp(2.80 * se_diff))}
    out["nominal"] = RD.nominal()
    out["sessions"] = infos
    full10 = spec10["rank"] == len(names10)
    full11 = spec11["rank"] == len(names11)
    finite = all(np.isfinite(v) for v in out["crb"]["pooled"].values())
    passed = bool(full10 and full11 and finite and spec10["min_rel_eigenvalue"] > RD.e1_eig_tol)
    details = {"rank_10": spec10["rank"], "rank_11_with_rho": spec11["rank"],
               "min_rel_eigenvalue_10": spec10["min_rel_eigenvalue"], "min_rel_eigenvalue_11": spec11["min_rel_eigenvalue"],
               "crb_pooled": out["crb"]["pooled"], "n_sessions": len(designs), "n_probes": out["n_probes"],
               "predictions_locked": os.path.isfile(gates.LOCK_REALDATA)}
    gates.record("E1", passed, details)
    out["gate_E1"] = {"passed": passed, **details}
    _save("e1_design_crb.json", out, output_dir)
    if verbose:
        p = out["crb"]["pooled"]
        print(f"  rank {spec10['rank']}/{len(names10)} (min rel eig {spec10['min_rel_eigenvalue']:.1e}); "
              f"with rho_obs {spec11['rank']}/{len(names11)}")
        print("  predicted relative CRB (pooled): " + ", ".join(f"{k} {v:.3f}" for k, v in p.items()))
        for g in ("expert", "novice"):
            if g in out["crb"]:
                print(f"  {g:7s}: k_inh {out['crb'][g]['k_inh']:.3f}  k_d {out['crb'][g]['k_d']:.3f}  "
                      f"alpha {out['crb'][g]['alpha']:.3f}  beta {out['crb'][g]['beta']:.3f}")
        if "group_difference_log_k_inh" in out:
            gd = out["group_difference_log_k_inh"]
            print(f"  MDE for log k_inh expert-novice (80% power): {gd['mde_80pct_alpha05']:.2f} log units "
                  f"(factor {gd['mde_factor']:.2f})")
        print(f"E1 {'PASS' if passed else 'FAIL'}")
    return out


# =============================================================================
# Sessions: real features or simulated on the real designs
# =============================================================================
def simulate_session(model: DeviationDMN, design: IrregularDesign, seed: int) -> dict:
    """Exact simulation of M1-dev at the design's observation times."""
    rng = np.random.default_rng(seed)
    t_eeg = np.asarray(design.t_eeg)
    oa, op = np.asarray(design.obs_A, bool), np.asarray(design.obs_Phi, bool)
    t_probe = np.asarray(design.t_probe)
    events = sorted([(t, 0, i) for i, t in enumerate(t_eeg)] + [(t, 1, k) for k, t in enumerate(t_probe)])
    P0 = model.stationary_cov()
    x = np.linalg.cholesky(P0 + 1e-15 * np.eye(2)) @ rng.normal(size=2)
    Lr = np.linalg.cholesky(model.obs_cov() + 1e-18 * np.eye(2))
    h = model.readout()
    y = np.full((len(t_eeg), 2), np.nan)
    rating = np.full(len(t_probe), np.nan)
    cache = {}
    t_prev = None
    for t, kind, idx in events:
        if t_prev is not None and t > t_prev:
            key = round(t - t_prev, 7)
            if key not in cache:
                F, c, Q = model.discretize(t - t_prev)
                cache[key] = (F, c, np.linalg.cholesky(Q + 1e-18 * np.eye(2)))
            F, c, Lq = cache[key]
            x = F @ x + c + Lq @ rng.normal(size=2)
        t_prev = t
        if kind == 0:
            obs = x + Lr @ rng.normal(size=2)
            y[idx, 0] = obs[0] if oa[idx] else np.nan
            y[idx, 1] = obs[1] if op[idx] else np.nan
        else:
            rating[idx] = h @ x + model.gamma0 + rng.normal(0, model.r_Q)
    return {"t": t_eeg, "y": y, "t_probe": t_probe, "rating": rating}


def simulate_sessions(model: DeviationDMN, designs: list[IrregularDesign], seed: int, metas=None) -> list[dict]:
    out = []
    for k, d in enumerate(designs):
        s = simulate_session(model, d, seed + 1000 * k)
        if metas:
            s.update(metas[k])
        out.append(s)
    return out


def design_of(session: dict) -> IrregularDesign:
    y = np.asarray(session["y"], float)
    return IrregularDesign(tuple(np.asarray(session["t"], float)), tuple(np.isfinite(y[:, 0])),
                           tuple(np.isfinite(y[:, 1])), tuple(np.asarray(session["t_probe"], float)))


def normalise_sessions(raw: list[dict], zscore_per_session: bool = False) -> tuple[list[dict], dict]:
    """Deviation form (Sec 5): subtract the session mean of y_A, y_Phi and
    rating; divide y_A, y_Phi by one pooled within-session SD per feature
    (or by the session SD when `zscore_per_session`)."""
    out = []
    dev = []
    for s in raw:
        y = np.asarray(s["y"], float).copy()
        mu = np.nanmean(y, axis=0)
        y = y - mu
        dev.append(y)
        r = np.asarray(s["rating"], float).copy()
        offset = float(np.nanmean(r)) if np.isfinite(r).any() else 0.0
        r = r - offset
        out.append({**s, "y": y, "rating": r, "rating_offset": offset})
    allv = np.vstack(dev)
    pooled_sd = np.sqrt(np.nanmean(allv ** 2, axis=0))
    for s in out:
        if zscore_per_session:
            sd = np.sqrt(np.nanmean(s["y"] ** 2, axis=0))
            s["y"] = s["y"] / np.where(sd > 0, sd, 1.0)
        else:
            s["y"] = s["y"] / np.where(pooled_sd > 0, pooled_sd, 1.0)
    return out, {"pooled_sd": pooled_sd.tolist(), "zscore_per_session": zscore_per_session}


def load_feature_sessions(feature_dir: str = FEATURES, y_A: str | None = None, y_Phi: str | None = None,
                          zscore_per_session: bool = False, min_probes: int | None = None,
                          max_missing: float | None = None, artifact_rule: str = "locked") -> tuple[list[dict], dict]:
    """Load the npz feature files into sessions; apply the keep criteria of Sec 4.

    artifact_rule="locked": preregistered mask (probe windows + 150 uV on any of
    the 64 channels). "adaptive": post-registration sensitivity rule (probe
    windows + session-adaptive threshold on the feature channels), available in
    feature files of format_version >= 2."""
    y_A = y_A or RD.feature_y_A
    y_Phi = y_Phi or RD.feature_y_Phi
    min_probes = RD.min_probes_per_session if min_probes is None else min_probes
    max_missing = RD.max_missing_frac if max_missing is None else max_missing
    raw, dropped = [], []
    for name in sorted(os.listdir(feature_dir)) if os.path.isdir(feature_dir) else []:
        if not name.endswith(".npz"):
            continue
        z = np.load(os.path.join(feature_dir, name), allow_pickle=False)
        if artifact_rule == "adaptive":
            if "mask_adaptive" not in z.files:
                raise RuntimeError(f"{name}: thieu mask_adaptive (file dac trung dinh dang cu; chay lai realdata preprocess)")
            mask = z["mask_adaptive"].astype(bool)
        else:
            mask = z["mask"].astype(bool)
        y = np.column_stack([z[y_A], z[y_Phi]]).astype(float)
        y[~mask] = np.nan
        rating = z["rating"].astype(float)
        keep_p = np.isfinite(rating)
        info = {"sub": str(z["sub"]), "ses": str(z["ses"]), "group": str(z["group"]),
                "missing_frac": float((~mask).mean()), "n_probes_rated": int(keep_p.sum())}
        if info["missing_frac"] > max_missing or info["n_probes_rated"] < min_probes:
            dropped.append(info)
            continue
        raw.append({"t": z["t"].astype(float), "y": y, "t_probe": z["t_probe"].astype(float)[keep_p],
                    "rating": rating[keep_p], **info})
    sessions, norm = normalise_sessions(raw, zscore_per_session)
    return sessions, {"n_loaded": len(sessions), "dropped": dropped, "features": (y_A, y_Phi),
                      "artifact_rule": artifact_rule, **norm}


# =============================================================================
# Fitting
# =============================================================================
def moment_starts(sessions: list[dict], base: DeviationDMN) -> dict:
    """Method-of-moments starting point: k from the lag-1 autocorrelation of
    each EEG feature (assuming 10-s steps), alpha/beta from a regression of the
    rating on the features at the probe epochs."""
    dt = RD.epoch_s / 60.0
    ac = []
    for j in range(2):
        num = den = 0.0
        for s in sessions:
            v = np.asarray(s["y"])[:, j]
            ok = np.isfinite(v[1:]) & np.isfinite(v[:-1])
            num += np.sum(v[1:][ok] * v[:-1][ok]); den += np.sum(v[:-1][ok] ** 2)
        ac.append(min(max(num / max(den, 1e-12), 0.05), 0.99))
    k_A, k_P = -np.log(ac[0]) / dt, -np.log(ac[1]) / dt
    X, r = [], []
    for s in sessions:
        t = np.asarray(s["t"]); y = np.asarray(s["y"])
        for tp, rv in zip(s["t_probe"], s["rating"]):
            if not np.isfinite(rv):
                continue
            i = np.searchsorted(t, tp - 1e-9) - 1
            i = min(max(i, 0), len(t) - 1)
            if np.all(np.isfinite(y[i])):
                X.append(y[i]); r.append(rv)
    coef = np.linalg.lstsq(np.array(X), np.array(r), rcond=None)[0] if len(X) > 5 else np.array([-base.beta, base.alpha])
    # pooled over all epochs; sessions differ in length, so concatenate (not vstack)
    var_A = float(np.nanmean(np.concatenate([np.asarray(s["y"])[:, 0] for s in sessions]) ** 2))
    var_P = float(np.nanmean(np.concatenate([np.asarray(s["y"])[:, 1] for s in sessions]) ** 2))
    return {"k_inh": float(k_A), "k_d": float(k_P), "alpha": float(coef[1]), "beta": float(-coef[0]),
            "sigma_A": float(np.sqrt(max(2 * k_A * var_A * 0.7, 1e-4))),
            "sigma_Phi": float(np.sqrt(max(2 * k_P * var_P * 0.7, 1e-4)))}


def starting_points(sessions: list[dict], base: DeviationDMN, n_starts: int, seed: int) -> list[dict]:
    """Sec 6: nominal, moment estimates, then jittered copies of both."""
    rng = np.random.default_rng(seed)
    nominal = {k: getattr(base, k) for k in base.FREE}
    mom = {**nominal, **moment_starts(sessions, base)}
    pts = [nominal, mom]
    while len(pts) < n_starts:
        src = pts[len(pts) % 2]
        pts.append({k: (v * float(np.exp(rng.normal(0, 0.3))) if k in base.POSITIVE else v + 0.3 * abs(v) * rng.normal() + 1e-3 * rng.normal())
                    for k, v in src.items()})
    return pts[:n_starts]


def fit_sessions(sessions: list[dict], base: DeviationDMN | None = None, names=None, n_starts: int = RD.n_starts,
                 seed: int = CFG.SEED, starts: list[dict] | None = None, label: str | None = None, ckpt=None) -> dict:
    base = base or nominal_model()
    names = list(names or base.FREE)
    starts = starts if starts is not None else starting_points(sessions, base, n_starts, seed)
    res = fit(sessions, RD.epoch_s / 60.0, init=base, names=names, n_starts=n_starts, rng=seed,
              loglik=kalman_loglik_irregular, starts=starts, label=label, ckpt=ckpt)
    return res


def _checkpoint(step: str, output_dir: str | None, *parts, verbose: bool = True) -> Checkpoint:
    """Checkpoint of one step; the fingerprint covers the data and settings."""
    path = os.path.join(output_dir or RESULTS, "checkpoints", f"{step}.json")
    return Checkpoint(path, fingerprint(step, CKPT_VERSION, repr(RD), *parts), verbose=verbose)


def se_at_mle(model: DeviationDMN, sessions: list[dict], names=None) -> dict:
    names = list(names or model.FREE)
    I = fisher_sum(model, [design_of(s) for s in sessions], names)
    return {"crb": crb(I, names, model), "fisher": I}


def observed_information_se(model: DeviationDMN, sessions: list[dict], names=None, h: float = 1e-3) -> dict:
    """SE from the observed information: central-difference Hessian of the
    log-likelihood at `model` on the fitting scale (log for positive
    parameters, linear for kappa, alpha, beta). Set beside the expected-Fisher
    SE; under a correct model both agree, a large gap signals misspecification
    (Q1_pipeline.md Sec 6, "SE at MLE ... cross-checked with a numerical Hessian")."""
    from eeg.model_fit import _pack, _unpack
    names = list(names or model.FREE)
    th0 = _pack(model, names)
    step = h * np.maximum(1.0, np.abs(th0))
    n = len(names)

    def ll(th):
        return kalman_loglik_irregular(_unpack(th, names, model), sessions)

    f0 = ll(th0)
    H = np.zeros((n, n))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = step[i]
        H[i, i] = (ll(th0 + ei) - 2 * f0 + ll(th0 - ei)) / step[i] ** 2
        for j in range(i):
            ej = np.zeros(n)
            ej[j] = step[j]
            H[i, j] = H[j, i] = (ll(th0 + ei + ej) - ll(th0 + ei - ej) - ll(th0 - ei + ej)
                                 + ll(th0 - ei - ej)) / (4 * step[i] * step[j])
    J = -H
    eig = np.linalg.eigvalsh(J)
    try:
        se = np.sqrt(np.clip(np.diag(np.linalg.inv(J)), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(n, np.inf)
    out = {}
    for k, s in zip(names, se):
        v = getattr(model, k)
        out[k] = {"se": float(s), "rel_se": float(s if k in model.POSITIVE else s / abs(v))}
    return {"params": out, "positive_definite": bool(eig.min() > 0), "min_eigenvalue": float(eig.min())}


def snr(model: DeviationDMN) -> dict:
    """Stationary SD of each latent channel over its observation-noise SD."""
    P = model.stationary_cov()
    return {"A": float(np.sqrt(P[0, 0]) / model.r_A), "Phi": float(np.sqrt(P[1, 1]) / model.r_Phi)}


def nominal_at_snr(snr_A: float) -> DeviationDMN:
    """Nominal parameters with r_A set so that sd_inf(A) / r_A = snr_A: the CRB
    'recomputed at the fitted SNR' required by E-H1 when the SNR is below nominal."""
    m = nominal_model()
    return m.with_params(r_A=float(np.sqrt(m.stationary_cov()[0, 0])) / snr_A)


def crb_on_sessions(model: DeviationDMN, sessions: list[dict], names=None) -> dict:
    """Relative CRB at `model` on the observation designs of `sessions`."""
    names = list(names or model.FREE)
    I = fisher_sum(model, [design_of(s) for s in sessions], names)
    return {k: v["rel_se"] for k, v in crb(I, names, model).items()}


def posterior_predictive_ratings(model: DeviationDMN, sessions: list[dict], n_rep: int = 200,
                                 seed: int = CFG.SEED) -> dict:
    """Posterior predictive check of the ordinal ratings (predictions_realdata.md
    Sec 3 item 8; Q1_pipeline.md Sec 7): ratings simulated from theta_hat on the
    real designs, the session mean of the observed ratings added back, rounded
    and clipped to -3..3, compared with the data on (i) the distribution of the
    seven levels, (ii) the median lag-1 autocorrelation between consecutive
    probes, (iii) the expert - novice difference in within-session SD."""
    levels = np.arange(-3, 4)
    designs = [design_of(s) for s in sessions]
    groups = [s.get("group") for s in sessions]
    offsets = [float(s.get("rating_offset", 0.0)) for s in sessions]

    def stats(rs):
        allr = np.concatenate(rs)
        prop = np.array([np.mean(allr == lv) for lv in levels])
        ac = []
        for r in rs:
            if len(r) > 3 and np.std(r[:-1]) > 0 and np.std(r[1:]) > 0:
                ac.append(np.corrcoef(r[:-1], r[1:])[0, 1])
        sd = {g: [np.std(r) for r, gg in zip(rs, groups) if gg == g] for g in ("expert", "novice")}
        d_sd = float(np.mean(sd["expert"]) - np.mean(sd["novice"])) if sd["expert"] and sd["novice"] else float("nan")
        return prop, float(np.median(ac)) if ac else float("nan"), d_sd

    obs = [np.rint(np.asarray(s["rating"], float) + o).astype(int) for s, o in zip(sessions, offsets)]
    p_obs, ac_obs, dsd_obs = stats(obs)
    P, AC, DSD = [], [], []
    for b in range(n_rep):
        sims = simulate_sessions(model, designs, seed + 7919 * b)
        rs = [np.clip(np.rint(np.asarray(x["rating"]) + o), -3, 3).astype(int) for x, o in zip(sims, offsets)]
        p, a, d = stats(rs)
        P.append(p); AC.append(a); DSD.append(d)
    P, AC, DSD = np.array(P), np.array(AC), np.array(DSD)

    def band(v):
        return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]

    return {"n_rep": n_rep, "levels": levels.tolist(),
            "level_prop": {"observed": p_obs.tolist(), "sim_mean": P.mean(axis=0).tolist(),
                           "sim_lo": np.percentile(P, 2.5, axis=0).tolist(),
                           "sim_hi": np.percentile(P, 97.5, axis=0).tolist()},
            "lag1_autocorr_median": {"observed": ac_obs, "sim_mean": float(np.nanmean(AC)), "sim_95": band(AC)},
            "sd_expert_minus_novice": {"observed": dsd_obs, "sim_mean": float(np.nanmean(DSD)), "sim_95": band(DSD)}}


def _boot_job(args):
    theta_dict, designs, names, seed, n_starts = args
    model = DeviationDMN(**theta_dict)
    sims = simulate_sessions(model, designs, seed)
    starts = [{k: getattr(model, k) for k in names}] + starting_points(sims, model, n_starts, seed)[1:]
    r = fit_sessions(sims, base=model, names=names, n_starts=n_starts, seed=seed, starts=starts[:n_starts])
    return {k: r["params"][k] for k in names}


def bootstrap(model: DeviationDMN, sessions: list[dict], n_boot: int, seed: int = CFG.SEED, n_starts: int = 2,
              n_jobs: int | None = None, verbose: bool = True, ckpt=None) -> list[dict]:
    """Parametric bootstrap; with `ckpt`, every finished replicate is stored
    under "boot/<b>" and skipped on a rerun."""
    import time
    designs = [design_of(s) for s in sessions]
    names = list(model.FREE)
    theta = {k: getattr(model, k) for k in names + ["rho_obs"]}
    jobs = {b: (theta, designs, names, seed + 7919 * b, n_starts) for b in range(n_boot)}
    out = {b: ckpt.get(f"boot/{b}") for b in jobs} if ckpt is not None else {}
    out = {b: v for b, v in out.items() if v is not None}
    todo = [b for b in jobs if b not in out]
    if verbose:
        print(f"  bootstrap: {n_boot} lan, da co {len(out)}, con {len(todo)}", flush=True)
    t0, n_new = time.time(), 0

    def done(b, r):
        nonlocal n_new
        out[b] = r
        n_new += 1
        if ckpt is not None:
            ckpt.put(f"boot/{b}", r)
        if verbose:
            el = time.time() - t0
            eta = el / n_new * (len(todo) - n_new)
            print(f"  bootstrap {len(out)}/{n_boot} xong, {el / 60:.0f} phut, con khoang {eta / 60:.0f} phut", flush=True)

    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    if n_jobs == 1:
        for b in todo:
            done(b, _boot_job(jobs[b]))
    elif todo:
        from concurrent.futures import as_completed
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = {ex.submit(_boot_job, jobs[b]): b for b in todo}
            for fu in as_completed(futs):
                done(futs[fu], fu.result())
    return [out[b] for b in range(n_boot)]


def rmse_same_scale(boots: list[dict], model: DeviationDMN, names) -> dict:
    out = {}
    for k in names:
        v = np.array([b[k] for b in boots], float)
        truth = getattr(model, k)
        if k in model.POSITIVE:
            out[k] = float(np.sqrt(np.mean((np.log(np.abs(v)) - np.log(abs(truth))) ** 2)))
        else:
            out[k] = float(np.sqrt(np.mean((v - truth) ** 2)) / max(abs(truth), 1e-12))
    return out


def gate_E2(ratio: dict, profiles_ok: bool, verbose: bool = True) -> dict:
    lo, hi = RD.e2_ratio_bounds
    inside = {k: bool(lo <= r <= hi) for k, r in ratio.items() if k in RD.key_params}
    passed = bool(sum(inside.values()) >= RD.e2_min_params and profiles_ok)
    details = {"ratio_rmse_over_se": ratio, "inside": inside, "profiles_unimodal": profiles_ok}
    gates.record("E2", passed, details)
    if verbose:
        print(f"E2 {'PASS' if passed else 'FAIL'}: RMSE/SE " + ", ".join(f"{k} {v:.2f}" for k, v in ratio.items() if k in RD.key_params))
    return {"passed": passed, **details}


def profile_unimodal(prof: np.ndarray) -> bool:
    """One interior maximum, no second local maximum above the chi2 threshold."""
    p = np.asarray(prof) - np.max(prof)
    peaks = [i for i in range(1, len(p) - 1) if p[i] >= p[i - 1] and p[i] >= p[i + 1]]
    return len(peaks) == 1 and 0 < int(np.argmax(p)) < len(p) - 1


# =============================================================================
# Residuals, LOSO, null models (E3)
# =============================================================================
def ljung_box_p(x: np.ndarray, lags: int = 30) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    m = min(lags, n // 4)
    if m < 1 or n < lags + 5:
        return float("nan")
    import scipy.stats as st
    xd = x - np.mean(x)
    c0 = float(np.dot(xd, xd) / n)
    if c0 <= 1e-15:
        return 1.0
    ks = np.arange(1, m + 1)
    rk = np.array([np.dot(xd[:-k], xd[k:]) / (n * c0) for k in ks])
    q = float(n * (n + 2) * np.sum((rk ** 2) / (n - ks)))
    return float(1.0 - st.chi2.cdf(q, df=m))


def residual_checks(model: DeviationDMN, sessions: list[dict]) -> dict:
    lb_A, lb_P, rho0 = [], [], []
    for s in sessions:
        _, info = kalman_filter_irregular(model, s, return_innovations=True)
        a, p = info["innov"]["A"], info["innov"]["Phi"]
        lb_A.append(ljung_box_p(a)); lb_P.append(ljung_box_p(p))
        n = min(len(a), len(p))
        if n > 10:
            rho0.append(float(np.corrcoef(a[:n], p[:n])[0, 1]))
    return {"ljung_box_p_A_median": float(np.nanmedian(lb_A)), "ljung_box_p_Phi_median": float(np.nanmedian(lb_P)),
            "innovation_corr_A_Phi_median": float(np.median(rho0)) if rho0 else float("nan"),
            "per_session": {"lb_A": lb_A, "lb_Phi": lb_P, "rho0": rho0}}


def _gauss_lpd(y, m, v):
    return -0.5 * (np.log(2 * np.pi * v) + (y - m) ** 2 / v)


def loso(sessions: list[dict], base: DeviationDMN, n_starts: int = 2, seed: int = CFG.SEED, verbose: bool = True,
         ckpt=None) -> dict:
    """Leave-one-session-out prediction of the ratings (Sec 7).

    Mode (a) EEG only: M1 state filtered from EEG (ratings never used) vs N2
    (instantaneous regression on the features at the probe epoch).
    Mode (b) EEG + past ratings: M1 one-step predictive density vs N1 (M1 with
    the rating channel only) and N3 (running session mean).
    Returns per-probe log predictive densities and MAE per model."""
    recs = []
    for k, test in enumerate(sessions):
        fold = f"loso {k + 1}/{len(sessions)}"
        saved = ckpt.get(f"loso/{k}") if ckpt is not None else None
        if saved is not None:
            recs.extend(saved)
            if verbose:
                print(f"  {fold} lay tu checkpoint", flush=True)
            continue
        if verbose:
            print(f"  {fold} ({test.get('sub')} {test.get('ses', '')}) ...", flush=True)
        n_before = len(recs)
        lab = fold if verbose else None
        train = sessions[:k] + sessions[k + 1:]
        fm = fit_sessions(train, base=base, n_starts=n_starts, seed=seed,
                          label=f"{lab} M1" if lab else None, ckpt=ckpt)["model"]
        # N1: rating-only M1 (A, Phi unobserved): same model family, EEG dropped
        fn1 = fit(train, RD.epoch_s / 60.0, init=base, names=list(base.FREE), n_starts=1, rng=seed,
                  loglik=lambda m, ss, dt: kalman_loglik_irregular(m, ss, use_eeg=False),
                  starts=[{kk: getattr(fm, kk) for kk in base.FREE}],
                  label=f"{lab} N1" if lab else None, ckpt=ckpt)["model"]
        # N2: instantaneous regression on train
        X, r = [], []
        for s in train:
            t = np.asarray(s["t"]); y = np.asarray(s["y"])
            for tp, rv in zip(s["t_probe"], s["rating"]):
                i = min(max(np.searchsorted(t, tp - 1e-9) - 1, 0), len(t) - 1)
                if np.isfinite(rv) and np.all(np.isfinite(y[i])):
                    X.append([1.0, y[i, 0], y[i, 1]]); r.append(rv)
        X, r = np.array(X), np.array(r)
        bN2 = np.linalg.lstsq(X, r, rcond=None)[0]
        vN2 = float(np.mean((r - X @ bN2) ** 2))
        vN3 = float(np.var(r))
        # test session
        _, infA = kalman_filter_irregular(fm, test, return_innovations=True, use_rating=False)
        _, infB = kalman_filter_irregular(fm, test, return_innovations=True, use_rating=True)
        _, infN1 = kalman_filter_irregular(fn1, test, return_innovations=True, use_rating=True, use_eeg=False)
        t = np.asarray(test["t"]); y = np.asarray(test["y"])
        past = []
        for j, (tp, m_a, v_a, rv) in enumerate(infA["pred_rating"]):
            if not np.isfinite(rv):
                continue
            i = min(max(np.searchsorted(t, tp - 1e-9) - 1, 0), len(t) - 1)
            yi = y[i] if np.all(np.isfinite(y[i])) else np.array([0.0, 0.0])
            mN2 = float(np.array([1.0, yi[0], yi[1]]) @ bN2)
            mN3 = float(np.mean(past)) if past else 0.0
            m_b, v_b = infB["pred_rating"][j][1], infB["pred_rating"][j][2]
            m_n1, v_n1 = infN1["pred_rating"][j][1], infN1["pred_rating"][j][2]
            recs.append({"session": k, "sub": test.get("sub"), "rating": rv,
                         "lpd_M1_a": _gauss_lpd(rv, m_a, v_a), "lpd_N2": _gauss_lpd(rv, mN2, vN2),
                         "lpd_M1_b": _gauss_lpd(rv, m_b, v_b), "lpd_N1": _gauss_lpd(rv, m_n1, v_n1),
                         "lpd_N3": _gauss_lpd(rv, mN3, vN3),
                         "ae_M1_a": abs(rv - m_a), "ae_N2": abs(rv - mN2), "ae_M1_b": abs(rv - m_b),
                         "ae_N1": abs(rv - m_n1), "ae_N3": abs(rv - mN3)})
            past.append(rv)
        if ckpt is not None:
            ckpt.put(f"loso/{k}", recs[n_before:])
    return summarise_loso(recs, seed)


def summarise_loso(recs: list[dict], seed: int, n_boot: int = RD.n_loso_boot) -> dict:
    import pandas as pd
    df = pd.DataFrame(recs)
    rng = np.random.default_rng(seed)
    sess = df["session"].unique()
    comps = {"a_M1_vs_N2": ("lpd_M1_a", "lpd_N2"), "b_M1_vs_N1": ("lpd_M1_b", "lpd_N1"), "b_M1_vs_N3": ("lpd_M1_b", "lpd_N3")}
    out = {"n_probes": int(len(df)), "n_sessions": int(len(sess)),
           "mean_lpd": {c: float(df[c].mean()) for c in df.columns if c.startswith("lpd_")},
           "mae": {c: float(df[c].mean()) for c in df.columns if c.startswith("ae_")}, "delta_lpd": {}}
    for name, (a, b) in comps.items():
        per = (df[a] - df[b]).groupby(df["session"]).mean().to_numpy()
        boots = [np.mean(per[rng.integers(0, len(per), len(per))]) for _ in range(n_boot)]
        out["delta_lpd"][name] = {"mean": float(per.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                  "wins": bool(np.percentile(boots, 2.5) > 0)}
    return out


def gate_E3(resid: dict, loso_done: bool, n_sensitivity: int, verbose: bool = True) -> dict:
    p_ok = min(resid["ljung_box_p_A_median"], resid["ljung_box_p_Phi_median"]) > RD.e3_min_lb_p
    passed = bool(p_ok and loso_done and n_sensitivity >= 5)
    details = {"ljung_box_p_median": [resid["ljung_box_p_A_median"], resid["ljung_box_p_Phi_median"]],
               "innovation_corr": resid["innovation_corr_A_Phi_median"], "loso_done": loso_done,
               "n_sensitivity_variants": n_sensitivity}
    gates.record("E3", passed, details)
    if verbose:
        print(f"E3 {'PASS' if passed else 'FAIL'}: Ljung-Box median p {details['ljung_box_p_median']}, LOSO {loso_done}, "
              f"sensitivity variants {n_sensitivity}")
    return {"passed": passed, **details}


# =============================================================================
# Group and subject fits
# =============================================================================
EH5_MDE_LOG = 0.51   # preregistered MDE of the expert - novice log k_inh difference (predictions_realdata.md)


def _se_log_k_inh(model: DeviationDMN, sessions: list[dict]) -> tuple[float, str | None]:
    """Fisher SE of log k_inh at `model`; NaN plus the reason when the Fisher
    matrix is undefined (a positive parameter at the boundary)."""
    try:
        return float(se_at_mle(model, sessions)["crb"]["k_inh"]["se"]), None
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as e:
        return float("nan"), f"{type(e).__name__}: {e}"


def fit_group_k_inh(sessions: list[dict], pooled: DeviationDMN, group_k: dict, label: str | None = None,
                    ckpt=None) -> dict:
    """M1 with k_inh specific to each group and every other free parameter
    shared: the alternative of the 1-df likelihood-ratio test of E-H5. Starts:
    the pooled MLE with a common k_inh, and the pooled MLE with the k_inh of
    the separate group fits."""
    groups = sorted(group_k)
    by_g = {g: [s for s in sessions if s["group"] == g] for g in groups}
    pos = type(pooled).POSITIVE
    shared = [k for k in pooled.FREE if k != "k_inh"]
    ns = len(shared)

    def unpack(theta):
        sh = {k: float(np.exp(v)) if k in pos else float(v) for k, v in zip(shared, theta[:ns])}
        return sh, {g: float(np.exp(v)) for g, v in zip(groups, theta[ns:])}

    def nll(theta):
        sh, ks = unpack(theta)
        try:
            ll = sum(kalman_loglik_irregular(pooled.with_params(**sh, k_inh=ks[g]), by_g[g]) for g in groups)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return 1e12
        return -ll if np.isfinite(ll) else 1e12

    sh0 = [np.log(getattr(pooled, k)) if k in pos else getattr(pooled, k) for k in shared]
    starts = [np.r_[sh0, [np.log(pooled.k_inh)] * len(groups)], np.r_[sh0, [np.log(group_k[g]) for g in groups]]]
    best = run_starts(nll, starts, label=label, ckpt=ckpt)
    sh, ks = unpack(best.x)
    return {"shared": sh, "k_inh": ks, "loglik": -float(best.fun), "success": bool(best.success)}


def group_fits(sessions: list[dict], pooled: DeviationDMN, n_starts: int, seed: int,
               verbose: bool = False, ckpt=None, prev: dict | None = None) -> dict:
    """E-H5 (Q1_pipeline.md Sec 7.3 (b)): a separate fit per group; the Wald
    CI of the difference in log k_inh between the two group fits (Fisher SE of
    each, as in the preregistered MDE); the likelihood-ratio test of a
    group-specific k_inh against a common one (1 df, everything else shared);
    and the omnibus test with every parameter group-specific.
    `prev`: per-group results of an earlier run ("groups" of fit_results.json),
    reused instead of refitting."""
    from scipy.stats import chi2
    out = {"pooled_loglik": kalman_loglik_irregular(pooled, sessions)}
    groups = sorted({s["group"] for s in sessions})
    ll_sep, fits = 0.0, {}
    for g in groups:
        sg = [s for s in sessions if s["group"] == g]
        if prev and g in prev:
            m_g = pooled.with_params(**prev[g]["params"])
            params, ll = dict(prev[g]["params"]), float(kalman_loglik_irregular(m_g, sg))
            if verbose:
                print(f"    [nhom {g}] dung fit da luu: -loglik {-ll:.2f}", flush=True)
        else:
            r = fit_sessions(sg, base=pooled, n_starts=n_starts, seed=seed,
                             starts=[{k: getattr(pooled, k) for k in pooled.FREE}],
                             label=f"nhom {g}" if verbose else None, ckpt=ckpt)
            m_g, params, ll = r["model"], r["params"], r["loglik"]
        se_k, se_err = _se_log_k_inh(m_g, sg)
        fits[g] = {"params": params, "loglik": ll, "n_sessions": len(sg), "se_log_k_inh": se_k, "se_error": se_err}
        ll_sep += ll
    out["groups"] = fits
    stat, df = 2 * (ll_sep - out["pooled_loglik"]), len(pooled.FREE) * (len(groups) - 1)
    out["lr_all_params"] = {"stat": float(stat), "df": df, "p": float(chi2.sf(stat, df))}
    if len(groups) == 2:
        a, b = groups   # sorted: expert, novice -> diff > 0 means k_inh expert > novice
        d = float(np.log(fits[a]["params"]["k_inh"]) - np.log(fits[b]["params"]["k_inh"]))
        se_d = float(np.hypot(fits[a]["se_log_k_inh"], fits[b]["se_log_k_inh"]))
        ci = [d - 1.96 * se_d, d + 1.96 * se_d]
        out["log_k_inh_difference"] = {"groups": [a, b], "diff": d, "se": se_d, "ci95": ci}
        jk = fit_group_k_inh(sessions, pooled, {g: fits[g]["params"]["k_inh"] for g in groups},
                             label="nhom k_inh rieng" if verbose else None, ckpt=ckpt)
        lr1 = 2 * (jk["loglik"] - out["pooled_loglik"])
        out["lr_k_inh_1df"] = {"stat": float(lr1), "df": 1, "p": float(chi2.sf(lr1, 1)),
                               "log_k_inh_difference": float(np.log(jk["k_inh"][a]) - np.log(jk["k_inh"][b])),
                               "joint_fit": jk}
        finite = bool(np.isfinite(se_d))
        out["E_H5"] = {"prediction": f"k_inh {a} > {b}", "diff": d, "ci95": ci,
                       "supported": bool(finite and ci[0] > 0),
                       "opposite_direction": bool(finite and ci[1] < 0),
                       "ci_contains_0": bool(finite and ci[0] <= 0 <= ci[1]), "mde_log": EH5_MDE_LOG}
    return out


def print_groups(g: dict) -> None:
    lr = g["lr_all_params"]
    print(f"  nhom: LR moi tham so rieng {lr['stat']:.1f} (df {lr['df']}, p {lr['p']:.2g})", flush=True)
    if "E_H5" in g:
        l1, e = g["lr_k_inh_1df"], g["E_H5"]
        verdict = ("ung ho" if e["supported"] else "nguoc chieu du doan" if e["opposite_direction"]
                   else f"CI chua 0, bao kem MDE {e['mde_log']} log" if e["ci_contains_0"] else "SE khong tinh duoc")
        print(f"  E-H5: hieu log k_inh {e['prediction'].split()[1]} - {e['prediction'].split()[-1]} = {e['diff']:.2f}, "
              f"CI95 [{e['ci95'][0]:.2f}, {e['ci95'][1]:.2f}] -> {verdict}", flush=True)
        print(f"        LR k_inh rieng (1 df) {l1['stat']:.1f}, p {l1['p']:.2g}; "
              f"hieu log k_inh khi cac tham so khac chung {l1['log_k_inh_difference']:.2f}", flush=True)


def subject_fits(sessions: list[dict], pooled: DeviationDMN, n_starts: int, seed: int,
                 verbose: bool = False, ckpt=None) -> dict:
    """Per-subject fits and a random-effects (REML) meta-analysis of log k_inh."""
    subs = sorted({s["sub"] for s in sessions})
    est, se = [], []
    per = {}
    for i, sub in enumerate(subs):
        ss = [s for s in sessions if s["sub"] == sub]
        r = fit_sessions(ss, base=pooled, n_starts=n_starts, seed=seed, starts=[{k: getattr(pooled, k) for k in pooled.FREE}],
                         label=f"nguoi {i + 1}/{len(subs)} {sub}" if verbose else None, ckpt=ckpt)
        m = r["model"]
        # a single subject (1-3 sessions) can push a positive parameter to the
        # boundary (exp underflow to 0): the Fisher matrix is then undefined.
        # Record it and leave the subject out of the meta-analysis, as for any
        # non-finite SE.
        se_k, se_err = _se_log_k_inh(m, ss)
        boundary = [k for k in m.POSITIVE if k in r["params"] and not r["params"][k] > 1e-8]
        per[sub] = {"params": r["params"], "se_log_k_inh": se_k, "n_sessions": len(ss), "group": ss[0]["group"],
                    "se_error": se_err, "at_boundary": boundary}
        if verbose and (se_err or boundary):
            why = f"{', '.join(boundary)} ve bien 0" if boundary else se_err
            print(f"    {sub}: SE khong tinh duoc ({why}), bo khoi meta-analysis", flush=True)
        if np.isfinite(se_k) and se_k < 5:
            est.append(np.log(m.k_inh)); se.append(se_k)
    meta = reml_meta(np.array(est), np.array(se)) if len(est) >= 3 else {}
    return {"per_subject": per, "meta_log_k_inh": meta}


def reml_meta(y: np.ndarray, se: np.ndarray) -> dict:
    """Random-effects meta-analysis of log k_inh: REML estimate of tau^2."""
    from scipy.optimize import minimize_scalar
    v = se ** 2

    def nll(log_tau2):
        w = 1.0 / (v + np.exp(log_tau2))
        mu = np.sum(w * y) / np.sum(w)
        return 0.5 * (np.sum(np.log(v + np.exp(log_tau2))) + np.sum(w * (y - mu) ** 2) + np.log(np.sum(w)))
    res = minimize_scalar(nll, bounds=(-15, 5), method="bounded")
    tau2 = float(np.exp(res.x))
    w = 1.0 / (v + tau2)
    mu = float(np.sum(w * y) / np.sum(w))
    return {"mu_log_k_inh": mu, "se_mu": float(1 / np.sqrt(np.sum(w))), "tau2": tau2, "tau": float(np.sqrt(tau2)),
            "k_inh_pooled": float(np.exp(mu)), "n": int(len(y)),
            "I2": float(max(0.0, 1 - np.sum(1 / v) ** -1 * len(y) / max(tau2 + np.mean(v), 1e-12)))}


def lag1_autocorr(v: np.ndarray) -> float:
    """Lag-1 autocorrelation of a feature over consecutive observed epochs."""
    v = np.asarray(v, float)
    ok = np.isfinite(v[1:]) & np.isfinite(v[:-1])
    if ok.sum() < 10:
        return float("nan")
    a, b = v[:-1][ok], v[1:][ok]
    return float(np.corrcoef(a, b)[0, 1])


def stop_checks(sessions: list[dict], n_total: int, verbose: bool = True) -> dict:
    """Locked stop criteria evaluated on the features before fitting:
    K2 = median lag-1 autocorrelation of y_A < 0.3 (features near white noise);
    K3 = more than 50 % of the sessions dropped by the keep criteria."""
    ac_A = [lag1_autocorr(np.asarray(s["y"])[:, 0]) for s in sessions]
    ac_P = [lag1_autocorr(np.asarray(s["y"])[:, 1]) for s in sessions]
    med_A, med_P = float(np.nanmedian(ac_A)), float(np.nanmedian(ac_P))
    dropped_frac = 1 - len(sessions) / max(n_total, 1)
    out = {"lag1_autocorr_yA_median": med_A, "lag1_autocorr_yPhi_median": med_P,
           "K2_triggered": bool(med_A < 0.3), "dropped_frac": dropped_frac, "K3_triggered": bool(dropped_frac > 0.5),
           "per_session_lag1": {"y_A": ac_A, "y_Phi": ac_P}}
    if verbose:
        print(f"  K2: median lag-1 autocorrelation y_A {med_A:.2f}, y_Phi {med_P:.2f} "
              f"({'TRIGGERED: features near white noise' if out['K2_triggered'] else 'ok'})")
        print(f"  K3: {len(sessions)}/{n_total} sessions kept, dropped {100 * dropped_frac:.0f}% "
              f"({'TRIGGERED' if out['K3_triggered'] else 'ok'})")
    return out


# =============================================================================
# Entry points
# =============================================================================
def run_fit(seed: int = CFG.SEED, fast: bool = False, n_starts: int | None = None, output_dir: str | None = None,
            feature_dir: str = FEATURES, sessions: list[dict] | None = None, verbose: bool = True,
            force: bool = False) -> dict:
    gates.require("E1")
    n_starts = n_starts or (2 if fast else RD.n_starts)
    if sessions is None:
        sessions, load_info = load_feature_sessions(feature_dir)
        n_total = load_info["n_loaded"] + len(load_info["dropped"])
    else:
        load_info = {"n_loaded": len(sessions), "synthetic": True}
        n_total = len(sessions)
    if not sessions:
        raise RuntimeError("Khong co phien dac trung nao (chay realdata preprocess truoc)")
    stops = stop_checks(sessions, n_total, verbose)
    if (stops["K2_triggered"] or stops["K3_triggered"]) and not force:
        _save("fit_results.json", {"load": load_info, "stop_checks": stops, "stopped": True}, output_dir)
        raise RuntimeError("Tieu chi dung K2/K3 kich hoat (xem fit_results.json). Day la ket qua can bao cao; "
                           "chi fit tiep voi --force-gate cho muc dich kham pha.")
    base = nominal_model()
    ckpt = _checkpoint("fit", output_dir, sessions_key(sessions), seed, n_starts, fast, verbose=verbose)
    n_prof = 5 if fast else CFG.PROFILE_N_GRID
    if verbose:
        print(f"Fitting M1-dev on {len(sessions)} sessions, {n_starts} starts...", flush=True)
        print(f"  [1/4] fit gop ({n_starts} diem khoi dau); sau do [2/4] profile {len(RD.profile_params)} tham so "
              f"x {n_prof} diem, [3/4] theo nhom, [4/4] theo nguoi", flush=True)
    r = fit_sessions(sessions, base=base, n_starts=n_starts, seed=seed, label="gop" if verbose else None, ckpt=ckpt)
    m = r["model"]
    se = se_at_mle(m, sessions)["crb"]
    out = {"load": load_info, "stop_checks": stops, "pooled": {"params": r["params"], "loglik": r["loglik"], "success": r["success"],
                                        "se": {k: v["se"] for k, v in se.items()}, "rel_se": {k: v["rel_se"] for k, v in se.items()}},
           "time_constants_min": {"1/k_inh": 1 / m.k_inh, "1/k_d": 1 / m.k_d}}
    if verbose:  # the headline numbers as soon as the pooled fit is done
        print("  theta_hat: " + ", ".join(f"{k} {v:.4g}" for k, v in r["params"].items()), flush=True)
        print("  rel SE   : " + ", ".join(f"{k} {se[k]['rel_se']:.3f}" for k in RD.key_params), flush=True)
        print(f"  1/k_inh = {1 / m.k_inh:.1f} min, 1/k_d = {1 / m.k_d:.1f} min", flush=True)
    # profiles
    prof = {}
    for i, p in enumerate(RD.profile_params):
        if verbose:
            print(f"  [2/4] profile {p} ({i + 1}/{len(RD.profile_params)})", flush=True)
        v0 = getattr(m, p)
        grid = (v0 * np.exp(np.linspace(-CFG.PROFILE_LOG_HALF_RANGE, CFG.PROFILE_LOG_HALF_RANGE, n_prof))
                if p in m.POSITIVE else v0 + np.linspace(-0.8, 0.8, n_prof) * abs(v0))
        ll = profile_likelihood(sessions, RD.epoch_s / 60.0, r, p, grid, n_starts=1,
                                label=f"profile {p}" if verbose else None, ckpt=ckpt)
        prof[p] = {"grid": grid.tolist(), "loglik": ll.tolist(), "unimodal": profile_unimodal(ll)}
    out["profiles"] = prof
    if verbose:
        print("  [3/4] fit theo nhom", flush=True)
    out["groups"] = group_fits(sessions, m, 1 if fast else 2, seed, verbose=verbose, ckpt=ckpt)
    if verbose:
        print_groups(out["groups"])
    if not fast:
        if verbose:
            print("  [4/4] fit theo nguoi", flush=True)
        out["subjects"] = subject_fits(sessions, m, 1, seed, verbose=verbose, ckpt=ckpt)
    path = _save("fit_results.json", out, output_dir)
    ckpt.clear()
    if verbose:
        print(f"  xong, ket qua: {path}", flush=True)
    return out


def run_groups(seed: int = CFG.SEED, fast: bool = False, output_dir: str | None = None, feature_dir: str = FEATURES,
               sessions: list[dict] | None = None, verbose: bool = True) -> dict:
    """Redo the group analysis (E-H5) on an existing fit_results.json and update
    it in place: the saved pooled and per-group fits are reused, only the
    group SEs, the CI and the 1-df likelihood-ratio fit are computed."""
    gates.require("E1")
    d = output_dir or RESULTS
    with open(os.path.join(d, "fit_results.json"), encoding="utf-8") as f:
        fitres = json.load(f)
    pooled = DeviationDMN(**fitres["pooled"]["params"])
    if sessions is None:
        sessions, _ = load_feature_sessions(feature_dir)
    ckpt = _checkpoint("groups", output_dir, sessions_key(sessions), fitres["pooled"]["params"], seed, fast,
                       verbose=verbose)
    if verbose:
        print(f"Phan tich theo nhom (E-H5) tren {len(sessions)} phien", flush=True)
    prev = (fitres.get("groups") or {}).get("groups")
    fitres["groups"] = group_fits(sessions, pooled, 1 if fast else 2, seed, verbose=verbose, ckpt=ckpt, prev=prev)
    path = _save("fit_results.json", fitres, output_dir)
    ckpt.clear()
    if verbose:
        print_groups(fitres["groups"])
        print(f"  da cap nhat: {path}", flush=True)
    return fitres["groups"]


def run_boot(seed: int = CFG.SEED, fast: bool = False, n_boot: int | None = None, output_dir: str | None = None,
             feature_dir: str = FEATURES, sessions: list[dict] | None = None, verbose: bool = True) -> dict:
    gates.require("E1")
    d = output_dir or RESULTS
    with open(os.path.join(d, "fit_results.json"), encoding="utf-8") as f:
        fitres = json.load(f)
    model = DeviationDMN(**fitres["pooled"]["params"])
    if sessions is None:
        sessions, _ = load_feature_sessions(feature_dir)
    n_boot = n_boot or (RD.n_boot_fast if fast else RD.n_boot)
    # n_boot is not in the fingerprint: replicate b is the same whatever the total
    ckpt = _checkpoint("boot", output_dir, sessions_key(sessions), fitres["pooled"]["params"], seed, verbose=verbose)
    boots = bootstrap(model, sessions, n_boot, seed, n_starts=2, n_jobs=1 if fast else None, verbose=verbose, ckpt=ckpt)
    rmse = rmse_same_scale(boots, model, list(model.FREE))
    se = fitres["pooled"]["rel_se"]
    ratio = {k: rmse[k] / se[k] if se[k] and np.isfinite(se[k]) and se[k] > 0 else float("nan") for k in rmse}
    profiles_ok = all(v["unimodal"] for v in fitres["profiles"].values())
    gate = gate_E2(ratio, profiles_ok, verbose)
    out = {"n_boot": n_boot, "rmse_bootstrap": rmse, "se_at_mle": se, "ratio_rmse_over_se": ratio, "gate_E2": gate,
           "boot_params": boots}
    _save("boot_results.json", out, output_dir)
    ckpt.clear()
    return out


def run_check(seed: int = CFG.SEED, fast: bool = False, output_dir: str | None = None, feature_dir: str = FEATURES,
              sessions: list[dict] | None = None, sensitivity_loaders: dict | None = None, verbose: bool = True) -> dict:
    """Residuals, LOSO with null models and the five locked sensitivity variants."""
    gates.require("E2")
    d = output_dir or RESULTS
    with open(os.path.join(d, "fit_results.json"), encoding="utf-8") as f:
        fitres = json.load(f)
    model = DeviationDMN(**fitres["pooled"]["params"])
    if sessions is None:
        sessions, _ = load_feature_sessions(feature_dir)
    ckpt = _checkpoint("check", output_dir, sessions_key(sessions), fitres["pooled"]["params"], seed, fast, verbose=verbose)
    if verbose:
        print("  [1/3] phan du", flush=True)
    resid = residual_checks(model, sessions)
    out = {"residuals": {k: v for k, v in resid.items() if k != "per_session"}}
    if verbose:
        print(f"  [2/3] LOSO tren {len(sessions)} phien", flush=True)
    out["loso"] = loso(sessions, nominal_model(), n_starts=1 if fast else 2, seed=seed, verbose=verbose, ckpt=ckpt)
    # sensitivity variants (Sec 7): each returns a session list
    variants = sensitivity_loaders if sensitivity_loaders is not None else default_sensitivity_loaders(feature_dir)
    sens = {}
    for i, (name, loader) in enumerate(variants.items()):
        saved = ckpt.get(f"sens/{name}")
        if saved is not None:
            sens[name] = saved
            if verbose:
                print(f"  [3/3] phan tich nhay {i + 1}/{len(variants)} {name}: lay tu checkpoint", flush=True)
            continue
        if verbose:
            print(f"  [3/3] phan tich nhay {i + 1}/{len(variants)} {name}", flush=True)
        try:
            ss = loader()
            if not ss:
                sens[name] = {"skipped": "no sessions"}
            else:
                r = fit_sessions(ss, base=model, n_starts=1 if fast else 2, seed=seed,
                                 starts=[{k: getattr(model, k) for k in model.FREE}],
                                 label=f"nhay {name}" if verbose else None, ckpt=ckpt)
                sens[name] = {"params": r["params"], "1/k_inh": 1 / r["params"]["k_inh"],
                              "sign_alpha": float(np.sign(r["params"]["alpha"])), "sign_beta": float(np.sign(r["params"]["beta"]))}
        except Exception as e:  # a variant must never abort the check
            sens[name] = {"error": repr(e)}
        if "error" not in sens[name]:  # errors are retried on a rerun
            ckpt.put(f"sens/{name}", sens[name])
    out["sensitivity"] = sens
    n_ok = sum(1 for k, v in sens.items() if "params" in v and not k.startswith("postreg_"))
    out["gate_E3"] = gate_E3(resid, True, n_ok, verbose)
    _save("check_results.json", out, output_dir)
    ckpt.clear()
    return out


def default_sensitivity_loaders(feature_dir: str) -> dict:
    alt = RD.feature_alternatives
    return {
        "alt_yA_" + alt[0]: lambda: load_feature_sessions(feature_dir, y_A=alt[0])[0],
        "alt_yA_" + alt[1]: lambda: load_feature_sessions(feature_dir, y_A=alt[1])[0],
        "alt_yPhi_" + alt[2]: lambda: load_feature_sessions(feature_dir, y_Phi=alt[2])[0],
        "zscore_per_session": lambda: load_feature_sessions(feature_dir, zscore_per_session=True)[0],
        "missing_le_20pct": lambda: load_feature_sessions(feature_dir, max_missing=0.20)[0],
        "first_20_probes": lambda: [_first_probes(s, 20) for s in load_feature_sessions(feature_dir)[0]],
        # post-registration (Q1_pipeline.md v2.2): reported separately, not part of the 4/5 rule
        "postreg_adaptive_artifact_rule": lambda: load_feature_sessions(feature_dir, artifact_rule="adaptive")[0],
    }


def _first_probes(s: dict, n: int) -> dict:
    return {**s, "t_probe": np.asarray(s["t_probe"])[:n], "rating": np.asarray(s["rating"])[:n]}


def run(step: str = "design-crb", **kw) -> dict:
    return {"design-crb": design_crb, "fit": run_fit, "groups": run_groups, "boot": run_boot,
            "check": run_check}[step](**kw)
