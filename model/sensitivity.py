"""
Goi ket qua 5 -- Sobol/Saltelli global sensitivity analysis.

* Sampling: scrambled Sobol sequence (scipy.stats.qmc) in the 2p-dimensional
  unit cube, split into the Saltelli A / B / A_B(i) matrices; parameters are
  log-uniform over x0.25-x4 of their nominal values (A-noise as cv_A).
* Estimators: Saltelli (2010) for S1, Jansen for ST, on NumPy.
* Scale: outputs listed in config.SENSITIVITY.log_outputs are analysed on the
  log scale (multiplicative outputs: on the raw scale a few extreme samples
  dominate the variance, indices clip at 1 and S1 << ST artificially).
* Uncertainty: bootstrap over sample rows gives a 95 % interval per index;
  convergence is checked by recomputing on the first half of the rows.
* Every raw evaluation is saved (sobol_evaluations.csv) so that any
  re-analysis (`--from-saved`) is a few seconds, not a new simulation.

Outputs analysed: qoc_stationary (raw, can be negative), qoc_variance,
crb_k_inh, lqg_efficiency (log).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bqi.dmn_ode import StateSpaceDMN
from model import config as CFG

# Parameters to vary. The A-noise is parameterised by the stationary coefficient
# of variation cv_A = sd_inf(A) / A0 = sigma_A / (A0 sqrt(2 k_inh)) instead of
# sigma_A: with sigma_A varied independently (x0.25-x4) the corner
# {sigma_A x4, k_inh x0.25} makes A_DMN negative ~30 % of the time, outside the
# model's physical range [0, A_max]. cv_A in [0.05, 0.4] keeps P(A < 0) <= 0.6 %.
SENSITIVITY_PARAMS = list(CFG.SENSITIVITY.params)
# Factor range for ordinary parameters (log-uniform)
LOG_FACTOR_LOW = np.log(CFG.SENSITIVITY.factor_range[0])
LOG_FACTOR_HIGH = np.log(CFG.SENSITIVITY.factor_range[1])
CV_A_RANGE = tuple(CFG.SENSITIVITY.cv_A_range)


def cv_A_of(m: StateSpaceDMN) -> float:
    return float(m.sigma_A / (m.A0 * np.sqrt(2 * m.k_inh)))


def _nominal_values() -> np.ndarray:
    m = StateSpaceDMN()
    return np.array([cv_A_of(m) if k == "cv_A" else getattr(m, k) for k in SENSITIVITY_PARAMS])


def saltelli_sample(n_base: int = 128, seed: int = CFG.SEED, sampler: str = CFG.SENSITIVITY.sampler) -> dict:
    """Generate Saltelli (A, B, A_Bi) sample matrices for Sobol analysis.

    A scrambled Sobol sequence of dimension 2p is split into A (first p columns)
    and B (last p columns) -- the standard quasi-Monte-Carlo construction, which
    converges much faster than independent uniform draws. `sampler="uniform"`
    keeps the plain pseudo-random version. Rows are in log-factor space
    (actual value = nominal * exp(factor)); n_total = n_base * (2 + p).
    """
    n_params = len(SENSITIVITY_PARAMS)
    if sampler == "sobol_qmc":
        from scipy.stats import qmc
        if n_base & (n_base - 1):
            import warnings
            warnings.warn(f"n_base={n_base} is not a power of 2; the Sobol sequence is unbalanced")
        U = qmc.Sobol(d=2 * n_params, scramble=True, seed=seed).random(n_base)
    else:
        U = np.random.default_rng(seed).random((n_base, 2 * n_params))
    A = LOG_FACTOR_LOW + (LOG_FACTOR_HIGH - LOG_FACTOR_LOW) * U[:, :n_params]
    B = LOG_FACTOR_LOW + (LOG_FACTOR_HIGH - LOG_FACTOR_LOW) * U[:, n_params:]
    # A_Bi: A with column i replaced by B column i
    ABi = [A.copy() for _ in range(n_params)]
    for i in range(n_params):
        ABi[i][:, i] = B[:, i]
    return {"A": A, "B": B, "ABi": ABi, "n_base": n_base, "n_params": n_params, "sampler": sampler}


def _factors_to_model(factors: np.ndarray) -> StateSpaceDMN:
    """Convert a log-factor array to a StateSpaceDMN. For cv_A the factor is
    mapped onto CV_A_RANGE (log-uniform) and converted back to sigma_A."""
    nominal = _nominal_values()
    vals = nominal * np.exp(factors)
    kw = {k: float(v) for k, v in zip(SENSITIVITY_PARAMS, vals) if k != "cv_A"}
    i = SENSITIVITY_PARAMS.index("cv_A")
    u = (factors[i] - LOG_FACTOR_LOW) / (LOG_FACTOR_HIGH - LOG_FACTOR_LOW)      # in [0, 1]
    cv = float(np.exp(np.log(CV_A_RANGE[0]) + u * (np.log(CV_A_RANGE[1]) - np.log(CV_A_RANGE[0]))))
    kw["sigma_A"] = cv * kw["A0"] * np.sqrt(2 * kw["k_inh"])
    return StateSpaceDMN(**kw)


# Design used for the CRB output (ds001787-like, config d)
CRB_DESIGN = {"session_min": CFG.REFERENCE.session_min, "probe_every_min": CFG.REFERENCE.probe_every_min,
              "n_subjects": CFG.REFERENCE.n_subjects, "config": CFG.REFERENCE.config}
NAN_OUT = {"qoc_stationary": float("nan"), "qoc_variance": float("nan"),
           "crb_k_inh": float("nan"), "lqg_efficiency": float("nan")}


def _evaluate_model(factors: np.ndarray) -> dict:
    """Evaluate all output quantities for a given parameter factor vector.

    Every output is deterministic (closed form or exact Fisher information):
    Sobol indices must reflect parameter sensitivity, not Monte Carlo noise.
      qoc_stationary  QoC* of M1
      qoc_variance    stationary variance h^T P h
      crb_k_inh       relative CRB of k_inh, all 12 parameters jointly, CRB_DESIGN
      lqg_efficiency  stationary LQG cost / stationary no-control cost, target = QoC* + 1
    """
    model = _factors_to_model(factors)
    try:
        ss = model.steady_states()
        qoc_star = float(ss["QoC_star"])
        P = model.stationary_cov()
        h = model.readout()
        qoc_var = float(h @ P @ h)
    except Exception:
        return dict(NAN_OUT)
    try:
        from model.fisher import Design, fisher_exact, crb
        d = Design.regular(CRB_DESIGN["session_min"], CRB_DESIGN["probe_every_min"], config=CRB_DESIGN["config"])
        names = list(StateSpaceDMN.FREE)
        F = fisher_exact(model, d, names, n_subjects=CRB_DESIGN["n_subjects"])
        crb_k_inh = float(crb(F, names, model)["k_inh"]["rel_se"])
    except Exception:
        crb_k_inh = float("nan")
    try:
        from model.control import state_space_with_control, solve_lqg, lqg_stationary_cost
        target = qoc_star + 1.0
        sys_m = state_space_with_control(model)
        ctrl = solve_lqg(sys_m, target_qoc=target)
        c_lqg = lqg_stationary_cost(model, ctrl, sys_true=sys_m)["cost"]
        c_none = ctrl["Q_cost"] * ((qoc_star - target) ** 2 + qoc_var)
        lqg_eff = float(c_lqg / c_none)
    except Exception:
        lqg_eff = float("nan")
    return {"qoc_stationary": qoc_star, "qoc_variance": qoc_var,
            "crb_k_inh": crb_k_inh, "lqg_efficiency": lqg_eff}


def _estimators(Y_A: np.ndarray, Y_B: np.ndarray, Y_ABi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised Saltelli S1 and Jansen ST. Y_ABi: (n_params, n_rows)."""
    var = np.nanvar(np.concatenate([Y_A, Y_B]))
    if var < 1e-30:
        return np.zeros(Y_ABi.shape[0]), np.zeros(Y_ABi.shape[0])
    s1 = np.nanmean(Y_B[None, :] * (Y_ABi - Y_A[None, :]), axis=1) / var
    st = np.nanmean((Y_A[None, :] - Y_ABi) ** 2, axis=1) / (2 * var)
    return s1, st


def sobol_indices(samples: dict, output_name: str, Y_A: np.ndarray, Y_B: np.ndarray,
                  Y_ABi: list[np.ndarray], n_boot: int = CFG.SENSITIVITY.n_boot, seed: int = CFG.SEED,
                  transform: str | None = None) -> dict:
    """First-order (S1, Saltelli 2010) and total (ST, Jansen) Sobol indices.

    transform: None -> as configured (log for outputs in config.log_outputs), "log", or "raw".
    Returns point estimates (not clipped), 95 % bootstrap intervals over rows,
    and a convergence check (indices from the first half of the rows).
    """
    n_params = samples["n_params"]
    use_log = (transform == "log") if transform else (output_name in CFG.SENSITIVITY.log_outputs)
    f = (lambda y: np.log(np.where(np.asarray(y, float) > 0, y, np.nan))) if use_log else (lambda y: np.asarray(y, float))
    YA, YB = f(Y_A), f(Y_B)
    YAB = np.vstack([f(y) for y in Y_ABi])
    S1, ST = _estimators(YA, YB, YAB)
    rng = np.random.default_rng(seed)
    n = len(YA)
    boot = np.empty((n_boot, 2, n_params))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b, 0], boot[b, 1] = _estimators(YA[idx], YB[idx], YAB[:, idx])
    ci = np.nanpercentile(boot, [2.5, 97.5], axis=0)          # (2, 2, n_params)
    half = n // 2
    S1h, STh = _estimators(YA[:half], YB[:half], YAB[:, :half])
    return {"S1": S1.tolist(), "ST": ST.tolist(),
            "S1_ci": np.stack([ci[0, 0], ci[1, 0]], axis=1).tolist(),
            "ST_ci": np.stack([ci[0, 1], ci[1, 1]], axis=1).tolist(),
            "ST_half": STh.tolist(), "max_abs_dST_half": float(np.nanmax(np.abs(STh - ST))),
            "scale": "log" if use_log else "raw", "Var_Y": float(np.nanvar(np.concatenate([YA, YB]))),
            "n_rows": int(n), "n_boot": int(n_boot), "output": output_name, "params": SENSITIVITY_PARAMS}


def evaluations_table(samples: dict, evals: list[dict], outputs: list[str]) -> pd.DataFrame:
    """Every raw evaluation as one row: block (A, B, AB_<param>), row index,
    log-factors per parameter, outputs. Enough to recompute any index later."""
    n_base, n_params = samples["n_base"], samples["n_params"]
    blocks = ["A"] * n_base + ["B"] * n_base + [f"AB_{p}" for p in SENSITIVITY_PARAMS for _ in range(n_base)]
    rows_idx = list(range(n_base)) * (2 + n_params)
    factors = np.vstack([samples["A"], samples["B"]] + samples["ABi"])
    df = pd.DataFrame(factors, columns=[f"f_{p}" for p in SENSITIVITY_PARAMS])
    df.insert(0, "row", rows_idx)
    df.insert(0, "block", blocks)
    for o in outputs:
        df[o] = [e.get(o, np.nan) for e in evals]
    return df


def indices_from_evaluations(df: pd.DataFrame, n_boot: int = CFG.SENSITIVITY.n_boot, seed: int = CFG.SEED,
                             verbose: bool = True) -> dict:
    """Recompute all indices from a saved evaluations table (no simulation)."""
    outputs = [c for c in df.columns if c not in ("block", "row") and not c.startswith("f_")]
    n_base = int(df["row"].max()) + 1
    samples = {"n_base": n_base, "n_params": len(SENSITIVITY_PARAMS)}
    A = df[df.block == "A"].sort_values("row")
    B = df[df.block == "B"].sort_values("row")
    ABi = [df[df.block == f"AB_{p}"].sort_values("row") for p in SENSITIVITY_PARAMS]
    results = {}
    for o in outputs:
        results[o] = sobol_indices(samples, o, A[o].values, B[o].values, [x[o].values for x in ABi],
                                   n_boot=n_boot, seed=seed)
        if verbose:
            idx = results[o]
            top = sorted(range(len(SENSITIVITY_PARAMS)), key=lambda i: idx["ST"][i], reverse=True)[:3]
            print(f"  {o} [{idx['scale']}]: " + ", ".join(
                f"{SENSITIVITY_PARAMS[i]} ST={idx['ST'][i]:.2f} [{idx['ST_ci'][i][0]:.2f}, {idx['ST_ci'][i][1]:.2f}]"
                for i in top) + f"  | max|dST| half-vs-full = {idx['max_abs_dST_half']:.3f}")
    return {"results": results, "params": SENSITIVITY_PARAMS, "n_base": n_base, "n_boot": n_boot}


def sensitivity_analysis(n_base: int = CFG.SENSITIVITY.n_base_default, seed: int = CFG.SEED,
                          verbose: bool = True, n_jobs: int | None = None) -> dict:
    """Run full Sobol sensitivity analysis for all outputs.

    n_base=64 is fast (OK for exploration); n_base>=256 for publication.
    Total evaluations = n_base * (2 + n_params) = n_base * 10, ~2 s each,
    spread over `n_jobs` processes (default: all cores but one).
    """
    samples = saltelli_sample(n_base=n_base, seed=seed)
    n_params = len(SENSITIVITY_PARAMS)
    outputs = ["qoc_stationary", "qoc_variance", "crb_k_inh", "lqg_efficiency"]
    n_total = n_base * (2 + n_params)

    if verbose:
        print(f"  Evaluating model for {n_total} parameter samples...", flush=True)

    # Flat list of all rows (A, B, then AB_i) -> evaluate in parallel; the
    # result is independent of the scheduling because every row is deterministic.
    rows = [samples["A"][j] for j in range(n_base)] + [samples["B"][j] for j in range(n_base)]
    for i in range(n_params):
        rows += [samples["ABi"][i][j] for j in range(n_base)]
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()
    evals = []
    if n_jobs == 1:
        it = map(_evaluate_model, rows)
    else:
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=n_jobs)
        it = pool.map(_evaluate_model, rows, chunksize=8)
    step = max(1, n_total // 20)
    for k, res in enumerate(it, 1):
        evals.append(res)
        if verbose and (k % step == 0 or k == n_total):
            el = time.time() - t0
            print(f"  {k}/{n_total} evaluations ({100 * k / n_total:.0f}%), {el / 60:.1f} min elapsed, "
                  f"~{el / k * (n_total - k) / 60:.1f} min left", flush=True)
    if n_jobs > 1:
        pool.shutdown()

    table = evaluations_table(samples, evals, outputs)
    out = indices_from_evaluations(table, seed=seed, verbose=verbose)
    out["evaluations"] = table
    out["sampler"] = samples["sampler"]
    return out


def to_dataframe(sensitivity_result: dict) -> pd.DataFrame:
    """Tidy table: one row per (output, param) with point estimates, 95 % CIs,
    the half-sample ST (convergence) and the analysis scale."""
    rows = []
    for output, idx in sensitivity_result["results"].items():
        for i, param in enumerate(sensitivity_result["params"]):
            rows.append({"output": output, "param": param, "scale": idx["scale"],
                         "S1": idx["S1"][i], "S1_lo": idx["S1_ci"][i][0], "S1_hi": idx["S1_ci"][i][1],
                         "ST": idx["ST"][i], "ST_lo": idx["ST_ci"][i][0], "ST_hi": idx["ST_ci"][i][1],
                         "ST_half": idx["ST_half"][i]})
    return pd.DataFrame(rows)


def run(seed: int = CFG.SEED, fast: bool = False, verbose: bool = True,
        output_dir: str | None = None, n_base: int | None = None, from_saved: bool = False) -> dict:
    """Entry point for main.py model sensitivity subcommand.
    n_base: 16 (fast) / 64 (default) / >= 256 for the paper (~N x 10 evaluations, ~2 s each).
    from_saved: recompute indices (scale, bootstrap) from output_dir/sobol_evaluations.csv, no simulation."""
    n_base = n_base or (CFG.SENSITIVITY.n_base_fast if fast else CFG.SENSITIVITY.n_base_default)
    saved = os.path.join(output_dir, "sobol_evaluations.csv") if output_dir else None
    if from_saved:
        if not (saved and os.path.isfile(saved)):
            raise FileNotFoundError("Khong co sobol_evaluations.csv de tinh lai; chay khong co --from-saved truoc.")
        if verbose:
            print(f"Recomputing Sobol indices from {saved} (no simulation)...", flush=True)
        table = pd.read_csv(saved)
        result = indices_from_evaluations(table, seed=seed, verbose=verbose)
        result["evaluations"] = table
        result["sampler"] = "from_saved"
    else:
        if verbose:
            print(f"Sobol sensitivity analysis (n_base={n_base}, sampler={CFG.SENSITIVITY.sampler})...", flush=True)
        result = sensitivity_analysis(n_base=n_base, seed=seed, verbose=verbose)
    df = to_dataframe(result)

    CFG.snapshot(output_dir)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "sobol_indices.csv"), index=False)
        if not from_saved:
            result["evaluations"].to_csv(saved, index=False)
        conv = {o: r["max_abs_dST_half"] for o, r in result["results"].items()}
        with open(os.path.join(output_dir, "sensitivity_results.json"), "w", encoding="utf-8") as f:
            json.dump({"n_base": result["n_base"], "n_boot": result["n_boot"], "params": result["params"],
                       "sampler": result["sampler"], "log_outputs": list(CFG.SENSITIVITY.log_outputs),
                       "cv_A_range": CV_A_RANGE,
                       "factor_range": [float(np.exp(LOG_FACTOR_LOW)), float(np.exp(LOG_FACTOR_HIGH))],
                       "convergence_max_abs_dST_half": conv},
                      f, indent=2, default=str)
        if verbose:
            print(f"  Saved to {output_dir}")

    return {"result": result, "df": df}