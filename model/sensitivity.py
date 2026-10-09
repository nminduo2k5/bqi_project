"""
Goi ket qua 5 -- Sobol/Saltelli global sensitivity analysis.

Saltelli estimator for first-order (S1) and total (ST) Sobol indices using
pure NumPy, no external library. Parameters are varied log-uniformly over
x0.25 to x4 of their nominal values.

Outputs analyzed: qoc_stationary, qoc_variance, crb_k_inh, lqg_efficiency.
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


def saltelli_sample(n_base: int = 128, seed: int = CFG.SEED) -> dict:
    """Generate Saltelli (A, B, A_Bi) sample matrices for Sobol analysis.

    Returns matrices as a dict with keys 'A', 'B', 'ABi' (list of n_params matrices).
    Each row is a sample in log-factor space (actual values = nominal * exp(factor)).
    n_total = n_base * (2 + n_params) rows total.
    """
    rng = np.random.default_rng(seed)
    n_params = len(SENSITIVITY_PARAMS)
    # A and B: (n_base, n_params) uniform log-factor samples
    A = rng.uniform(LOG_FACTOR_LOW, LOG_FACTOR_HIGH, (n_base, n_params))
    B = rng.uniform(LOG_FACTOR_LOW, LOG_FACTOR_HIGH, (n_base, n_params))
    # A_Bi: A with column i replaced by B column i
    ABi = [A.copy() for _ in range(n_params)]
    for i in range(n_params):
        ABi[i][:, i] = B[:, i]
    return {"A": A, "B": B, "ABi": ABi, "n_base": n_base, "n_params": n_params}


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


def sobol_indices(samples: dict, output_name: str,
                   Y_A: np.ndarray, Y_B: np.ndarray,
                   Y_ABi: list[np.ndarray]) -> dict:
    """Compute first-order (S1) and total (ST) Sobol indices from Saltelli estimator.

    Uses Jansen estimator for ST (more robust for small n_base).
    """
    n_base = samples["n_base"]
    n_params = samples["n_params"]
    Var_Y = float(np.nanvar(np.concatenate([Y_A, Y_B])))
    if Var_Y < 1e-30:
        return {"S1": [0.0] * n_params, "ST": [0.0] * n_params,
                "Var_Y": Var_Y, "output": output_name}

    S1 = []
    ST = []
    for i in range(n_params):
        # Saltelli (2010) estimator for S1: cov(Y_B, Y_ABi) / Var
        s1_num = float(np.nanmean(Y_B * (Y_ABi[i] - Y_A)))
        s1 = s1_num / Var_Y

        # Jansen estimator for ST: mean((Y_A - Y_ABi)^2) / (2 * Var)
        st_num = float(np.nanmean((Y_A - Y_ABi[i]) ** 2))
        st = st_num / (2 * Var_Y)

        S1.append(float(np.clip(s1, 0, 1)))
        ST.append(float(np.clip(st, 0, 1)))

    return {"S1": S1, "ST": ST, "Var_Y": float(Var_Y),
            "output": output_name, "params": SENSITIVITY_PARAMS}


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

    Y_A = {o: np.array([evals[j].get(o, np.nan) for j in range(n_base)]) for o in outputs}
    Y_B = {o: np.array([evals[n_base + j].get(o, np.nan) for j in range(n_base)]) for o in outputs}
    Y_ABi = {o: [np.array([evals[(2 + i) * n_base + j].get(o, np.nan) for j in range(n_base)])
                 for i in range(n_params)] for o in outputs}

    # Compute indices
    results = {}
    for o in outputs:
        idx = sobol_indices(samples, o, Y_A[o], Y_B[o], Y_ABi[o])
        results[o] = idx
        if verbose:
            top = sorted(range(n_params), key=lambda i: idx["ST"][i], reverse=True)[:3]
            top_str = ", ".join(f"{SENSITIVITY_PARAMS[i]}(ST={idx['ST'][i]:.2f})" for i in top)
            print(f"  {o}: top params = {top_str}")

    return {"results": results, "params": SENSITIVITY_PARAMS, "n_base": n_base}


def to_dataframe(sensitivity_result: dict) -> pd.DataFrame:
    """Convert sensitivity results to a tidy DataFrame."""
    rows = []
    for output, idx in sensitivity_result["results"].items():
        for i, param in enumerate(sensitivity_result["params"]):
            rows.append({"output": output, "param": param,
                         "S1": idx["S1"][i], "ST": idx["ST"][i]})
    return pd.DataFrame(rows)


def run(seed: int = CFG.SEED, fast: bool = False, verbose: bool = True,
        output_dir: str | None = None, n_base: int | None = None) -> dict:
    """Entry point for main.py model sensitivity subcommand.
    n_base: 16 (fast) / 64 (default) / >= 256 for the paper (~N x 10 evaluations, ~2 s each)."""
    n_base = n_base or (CFG.SENSITIVITY.n_base_fast if fast else CFG.SENSITIVITY.n_base_default)

    if verbose:
        print(f"Sobol sensitivity analysis (n_base={n_base})...", flush=True)
    result = sensitivity_analysis(n_base=n_base, seed=seed, verbose=verbose)
    df = to_dataframe(result)

    CFG.snapshot(output_dir)
    CFG.snapshot(output_dir)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "sobol_indices.csv"), index=False)
        with open(os.path.join(output_dir, "sensitivity_results.json"), "w", encoding="utf-8") as f:
            json.dump({**{k: v for k, v in result.items() if k != "results"}, "cv_A_range": CV_A_RANGE,
                   "factor_range": [float(np.exp(LOG_FACTOR_LOW)), float(np.exp(LOG_FACTOR_HIGH))]},
                  f, indent=2, default=str)
        if verbose:
            print(f"  Saved to {output_dir}")

    return {"result": result, "df": df}