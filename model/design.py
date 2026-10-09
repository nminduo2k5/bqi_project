"""
Goi ket qua 3 -- Experimental design map + gate M1.

CRB-first approach: compute Fisher/CRB for the full design grid (fast, seconds per cell),
then run full MLE recovery on 24 representative cells (slow, parallel).

Gate M1 passes if CRB predictions match MLE recovery errors within factor 2 at >= 80% of
representative cells.

Starting point: G3 FAIL with k_inh median relative error = 23% for ds001787-like design
(N=12, 45 min session, probe every 2 min). This point is highlighted on the design map.
"""
from __future__ import annotations

import json
import os
import sys
import time
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bqi.dmn_ode import StateSpaceDMN
from eeg import gates
from eeg.model_fit import fit
from model.identifiability import cramer_rao_bound
from model import config as CFG
from model import config as CFG

# ds001787-like design (starting point, G3 was FAIL here)
DS001787_DESIGN = {"N": CFG.REFERENCE.n_subjects, "session_min": CFG.REFERENCE.session_min,
                   "probe_every_min": CFG.REFERENCE.probe_every_min, "r_obs": CFG.REFERENCE.r_obs,
                   "config": CFG.REFERENCE.config}

# 24 representative cells chosen to cover the design space uniformly
REPRESENTATIVE_CELLS = CFG.GRID.cells()

KEY_PARAMS = list(CFG.KEY_PARAMS)


def _design_to_session_params(cell: dict, dt_s: float = CFG.EPOCH_S) -> dict:
    """Convert a design cell to simulation parameters."""
    dt = dt_s / 60.0  # epochs in minutes
    n_epochs = int(cell["session_min"] / dt)
    probe_every = max(1, int(cell["probe_every_min"] / dt))
    return {"n_epochs": n_epochs, "dt": dt, "probe_every": probe_every}


_FISHER_CACHE: dict = {}


def _fisher_one_subject(model: StateSpaceDMN, cell: dict, dt_s: float, config: str = CFG.REFERENCE.config) -> np.ndarray:
    """Exact expected Fisher information of ONE session for this design
    (all 12 free parameters, noise level r_obs). Cached: it does not depend on N."""
    from model.fisher import Design, fisher_exact
    key = (cell["session_min"], cell["probe_every_min"], cell["r_obs"], dt_s, config,
           tuple(getattr(model, k) for k in StateSpaceDMN.FREE))
    if key not in _FISHER_CACHE:
        m = model.with_params(r_A=cell["r_obs"], r_Phi=cell["r_obs"])
        d = Design.regular(cell["session_min"], cell["probe_every_min"], epoch_s=dt_s, config=config)
        _FISHER_CACHE[key] = fisher_exact(m, d, list(StateSpaceDMN.FREE))
    return _FISHER_CACHE[key]


def _crb_for_cell(model: StateSpaceDMN, cell: dict, dt_s: float = CFG.EPOCH_S,
                  seed: int = CFG.SEED) -> dict:
    """Relative CRB of KEY_PARAMS for a design cell.

    Exact expected Fisher information over ALL 12 free parameters (the MLE
    estimates all of them jointly, so the bound must too), for one session at
    the cell's noise level, times N independent subjects. `seed` is unused:
    the bound is deterministic.
    """
    try:
        F = cell["N"] * _fisher_one_subject(model, cell, dt_s)
        crb = cramer_rao_bound(F, list(StateSpaceDMN.FREE), model)
        return {k: crb[k]["relative_se"] for k in KEY_PARAMS}
    except Exception:
        return {k: float("nan") for k in KEY_PARAMS}


def design_grid_crb(model: StateSpaceDMN | None = None, seed: int = CFG.SEED,
                    dt_s: float = CFG.EPOCH_S, verbose: bool = True) -> pd.DataFrame:
    """Compute CRB for a grid of design cells. Returns DataFrame with one row per cell."""
    if model is None:
        model = StateSpaceDMN()

    Ns = list(CFG.GRID.n_subjects)
    session_mins = list(CFG.GRID.session_min)
    probe_every_mins = list(CFG.GRID.probe_every_min)
    r_obs_vals = list(CFG.GRID.r_obs)

    rows = []
    total = len(Ns) * len(session_mins) * len(probe_every_mins) * len(r_obs_vals)
    done = 0
    for N, sm, pm, ro in product(Ns, session_mins, probe_every_mins, r_obs_vals):
        cell = {"N": N, "session_min": sm, "probe_every_min": pm, "r_obs": ro}
        crb = _crb_for_cell(model, cell, dt_s=dt_s, seed=seed)
        row = {**cell, **{f"crb_{k}": crb[k] for k in KEY_PARAMS}}
        rows.append(row)
        done += 1
        if verbose and done % 50 == 0:
            print(f"  CRB grid: {done}/{total} cells done", flush=True)
    df = pd.DataFrame(rows)
    if verbose:
        print(f"  CRB grid complete: {len(df)} cells")
    return df


def mle_recovery_for_cell(model: StateSpaceDMN, cell: dict, n_reps: int = CFG.GRID.n_reps,
                           seed: int = CFG.SEED, dt_s: float = CFG.EPOCH_S, verbose: bool = False,
                           n_starts: int = CFG.GRID.n_starts) -> dict:
    """MLE parameter recovery for a design cell.

    Data are simulated at the cell's noise level r_obs. Errors are measured in
    the scale of the CRB: log(estimate/true) for positive parameters and
    (estimate - true)/|true| for linear ones. Returns the RMSE per key
    parameter (directly comparable with the relative CRB), the median
    absolute error, and every fitted parameter set (used by model.control).
    """
    sp = _design_to_session_params(cell, dt_s)
    rng = np.random.default_rng(seed)
    m = model.with_params(r_A=cell["r_obs"], r_Phi=cell["r_obs"])
    errors = {k: [] for k in KEY_PARAMS}
    fits = []
    for rep in range(n_reps):
        sessions = [m.simulate(sp["n_epochs"], sp["dt"], probe_every=sp["probe_every"],
                               seed=int(rng.integers(0, 1000000)))
                    for _ in range(cell["N"])]
        init = m.with_params(**{k: getattr(m, k) * float(np.exp(rng.normal(0, CFG.GRID.init_jitter_log_sd)))
                                for k in StateSpaceDMN.POSITIVE})
        try:
            r = fit(sessions, sp["dt"], init=init, n_starts=n_starts, rng=int(rng.integers(0, 1000000)))
            fits.append(r["params"])
            for k in KEY_PARAMS:
                true_val, est = getattr(m, k), r["params"][k]
                err = np.log(est / true_val) if k in StateSpaceDMN.POSITIVE else (est - true_val) / abs(true_val)
                errors[k].append(float(err))
        except Exception:
            for k in KEY_PARAMS:
                errors[k].append(float("nan"))
        if verbose:
            print(f"    rep {rep + 1}/{n_reps} done", flush=True)
    out = {}
    for k in KEY_PARAMS:
        e = np.array(errors[k], float)
        out[k] = float(np.sqrt(np.nanmean(e ** 2)))
        out[f"medabs_{k}"] = float(np.nanmedian(np.abs(e)))
    out["fits"] = fits
    return out


def design_grid_mle(model: StateSpaceDMN | None = None, cells: list | None = None,
                    n_reps: int = CFG.GRID.n_reps, seed: int = CFG.SEED, verbose: bool = True,
                    n_jobs: int | None = None, n_starts: int = CFG.GRID.n_starts) -> pd.DataFrame:
    """Run MLE recovery for representative cells in parallel. Slow -- run overnight."""
    if model is None:
        model = StateSpaceDMN()
    if cells is None:
        cells = REPRESENTATIVE_CELLS
    from concurrent.futures import ProcessPoolExecutor
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    jobs = [(model, cell, n_reps, seed + i, n_starts) for i, cell in enumerate(cells)]
    rows = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for i, (cell, res, secs) in enumerate(ex.map(_mle_job, jobs)):
            row = {**cell, **{f"mle_{k}": res[k] for k in KEY_PARAMS},
                   **{f"medabs_{k}": res[f"medabs_{k}"] for k in KEY_PARAMS},
                   "seconds": secs, "fits": json.dumps(res["fits"])}
            rows.append(row)
            if verbose:
                print(f"  MLE cell {i + 1}/{len(cells)}: N={cell['N']}, session={cell['session_min']}min, "
                      f"probe={cell['probe_every_min']}min, r={cell['r_obs']}: "
                      + ", ".join(f"{k} {res[k]:.3f}" for k in KEY_PARAMS) + f"  ({secs}s)", flush=True)
    return pd.DataFrame(rows)


def _mle_job(args):
    model, cell, n_reps, seed, n_starts = args
    t0 = time.time()
    res = mle_recovery_for_cell(model, cell, n_reps=n_reps, seed=seed, verbose=False, n_starts=n_starts)
    return cell, res, round(time.time() - t0, 1)


def doptimal_probe_schedule(model: StateSpaceDMN, session_min: float, n_probes: int,
                             dt_s: float = CFG.EPOCH_S, seed: int = CFG.SEED) -> dict:
    """Ds-optimal probe schedule with a fixed number of probes.

    Criterion: minimise log det of the CRB covariance block of KEY_PARAMS
    (all 12 parameters estimated jointly), from the exact Fisher information
    with the actual probe epochs. `seed` is unused (deterministic).

    Search space (cheap, interpretable families instead of free positions):
      * power spacing   t_k = T * ((k + 1/2) / n) ** p,  p in [0.4, 2.5]
        (p < 1: probes concentrated early, p > 1: late)
      * paired probes   n/2 pairs, partner g epochs later, g in {1, 2, 3, 6, 12}
        (short lags inform the fast time constants)
    The uniform schedule is p = 1. Efficiency is reported per parameter:
    exp(-(logdet_opt - logdet_uniform) / n_key) > 1 means smaller confidence volume.
    """
    from model.fisher import Design, fisher_exact
    dt = dt_s / 60.0
    n_epochs = int(round(session_min / dt))
    names = list(StateSpaceDMN.FREE)
    key_idx = [names.index(k) for k in KEY_PARAMS]

    def logdet_crb(probe_idx) -> float:
        idx = tuple(sorted(set(int(i) for i in np.clip(probe_idx, 0, n_epochs - 1))))
        if len(idx) < n_probes:
            return np.inf
        I = fisher_exact(model, Design(n_epochs, dt, idx, "d"), names)
        try:
            cov = np.linalg.inv(I)[np.ix_(key_idx, key_idx)]
            sign, ld = np.linalg.slogdet(cov)
            return ld if sign > 0 else np.inf
        except np.linalg.LinAlgError:
            return np.inf

    def power(p):
        return np.round(n_epochs * ((np.arange(n_probes) + 0.5) / n_probes) ** p - 0.5).astype(int)

    def paired(g):
        n_pairs = n_probes // 2
        first = np.round(np.linspace(0, n_epochs - 1 - g, n_pairs)).astype(int)
        return np.sort(np.concatenate([first, first + g]))

    candidates = [("uniform", 1.0, power(1.0))]
    candidates += [("power", float(p), power(p)) for p in CFG.SCHEDULE.power_exponents]
    candidates += [("paired", float(g), paired(g)) for g in CFG.SCHEDULE.pair_gaps]
    scored = [{"family": f, "param": v, "logdet_crb": logdet_crb(idx), "probe_idx": idx.tolist()}
              for f, v, idx in candidates]
    uni = scored[0]
    best = min(scored, key=lambda r: r["logdet_crb"])
    eff = float(np.exp(-(best["logdet_crb"] - uni["logdet_crb"]) / len(KEY_PARAMS)))
    return {"session_min": session_min, "n_probes": n_probes,
            "uniform_schedule": uni["probe_idx"], "optimal_schedule": best["probe_idx"],
            "optimal_family": best["family"], "optimal_param": best["param"],
            "uniform_logdet_crb": uni["logdet_crb"], "optimal_logdet_crb": best["logdet_crb"],
            "efficiency_vs_uniform": eff,
            "all": [{k: v for k, v in r.items() if k != "probe_idx"} for r in scored]}


def gate_M1(crb_df: pd.DataFrame, mle_df: pd.DataFrame,
             threshold: float = CFG.GATES.m1_ratio_bounds[1], min_fraction: float = CFG.GATES.m1_min_fraction,
             verbose: bool = True) -> dict:
    """Gate M1: CRB predictions match MLE recovery errors within factor `threshold`
    at >= `min_fraction` of representative cells.

    A parameter matches when 1/threshold <= (MLE RMSE / CRB) <= threshold; both
    are in the same scale (log for positive parameters, relative for linear).
    A cell matches when at least half of its key parameters match.
    """
    matches, ratios = [], []
    for _, mle_row in mle_df.iterrows():
        cell_matches = 0
        checked = 0
        # Find matching CRB row (same design AND same noise level)
        crb_match = crb_df[
            (crb_df["N"] == mle_row["N"]) &
            (np.abs(crb_df["session_min"] - mle_row["session_min"]) < 0.1) &
            (np.abs(crb_df["probe_every_min"] - mle_row["probe_every_min"]) < 0.01) &
            (np.abs(crb_df["r_obs"] - mle_row["r_obs"]) < 1e-9)
        ]
        if crb_match.empty:
            continue
        crb_row = crb_match.iloc[0]
        for k in KEY_PARAMS:
            mle_err = mle_row.get(f"mle_{k}", float("nan"))
            crb_pred = crb_row.get(f"crb_{k}", float("nan"))
            if np.isfinite(mle_err) and np.isfinite(crb_pred) and crb_pred > 0:
                ratio = mle_err / crb_pred
                ratios.append({"N": int(mle_row["N"]), "session_min": float(mle_row["session_min"]),
                               "probe_every_min": float(mle_row["probe_every_min"]),
                               "r_obs": float(mle_row["r_obs"]), "param": k, "ratio": float(ratio)})
                cell_matches += int(1 / threshold <= ratio <= threshold)
                checked += 1
        if checked > 0:
            matches.append(cell_matches / checked)

    if not matches:
        passed = False
        fraction = 0.0
    else:
        fraction = float(np.mean(np.array(matches) >= 0.5))
        passed = fraction >= min_fraction

    r = np.array([x["ratio"] for x in ratios]) if ratios else np.array([np.nan])
    details = {"fraction_matching": round(fraction, 3), "threshold": threshold,
               "min_fraction": min_fraction, "n_cells": len(matches),
               "ratio_rmse_over_crb": {"median": float(np.nanmedian(r)),
                                       "q10": float(np.nanpercentile(r, 10)),
                                       "q90": float(np.nanpercentile(r, 90))},
               "per_param": ratios}
    path = gates.record("M1", passed, details)
    if verbose:
        print(f"M1 {'PASS' if passed else 'FAIL'}: {fraction:.1%} cells match "
              f"(need {min_fraction:.0%}) -> {path}")
    return {"passed": passed, **details}


def minimum_design_for_k_inh(crb_df: pd.DataFrame, target_rel_se: float = CFG.TARGET_REL_ERROR) -> pd.DataFrame:
    """Find the cheapest design (N * session_min) achieving CRB(k_inh) <= target."""
    ok = crb_df[crb_df["crb_k_inh"] <= target_rel_se].copy()
    ok["cost"] = ok["N"] * ok["session_min"]
    return ok.sort_values("cost").head(10)[
        ["N", "session_min", "probe_every_min", "r_obs", "crb_k_inh", "cost"]]


def run(seed: int = CFG.SEED, fast: bool = False, verbose: bool = True,
        output_dir: str | None = None, n_reps: int | None = None, n_starts: int = CFG.GRID.n_starts) -> dict:
    """Entry point for main.py model design subcommand.
    n_reps: recoveries per cell (2 fast / 5 default); n_starts: L-BFGS-B restarts per fit
    (raise to 4-6 to rule out local optima if gate M1 fails at small N)."""
    model = StateSpaceDMN()
    n_reps = n_reps or (2 if fast else CFG.GRID.n_reps)

    if verbose:
        print("Computing CRB design grid (all cells)...", flush=True)
    crb_df = design_grid_crb(model, seed=seed, verbose=verbose)

    if verbose:
        print(f"\nRunning MLE recovery for {len(REPRESENTATIVE_CELLS)} representative cells "
              f"({n_reps} reps each)...", flush=True)
    mle_df = design_grid_mle(model, cells=REPRESENTATIVE_CELLS, n_reps=n_reps,
                              seed=seed, verbose=verbose, n_starts=n_starts)

    gate_result = gate_M1(crb_df, mle_df, verbose=verbose)

    if verbose:
        print("\nMinimum design for k_inh relative SE <= 20%:")
        rec = minimum_design_for_k_inh(crb_df)
        print(rec.to_string(index=False))
        # Show ds001787-like point
        R = CFG.REFERENCE
        ds_row = crb_df[(crb_df["N"] == R.n_subjects) & (np.abs(crb_df["session_min"] - R.session_min) < 1) &
                        (np.abs(crb_df["probe_every_min"] - R.probe_every_min) < 0.1) &
                        (np.abs(crb_df["r_obs"] - R.r_obs) < 1e-9)]
        if not ds_row.empty:
            print(f"\nds001787-like design (N=12, 45min, probe/2min): "
                  f"CRB k_inh = {ds_row['crb_k_inh'].values[0]:.3f} "
                  f"(G3 result: 23% actual error)")

    result = {"gate_M1": gate_result, "n_crb_cells": len(crb_df), "n_mle_cells": len(mle_df),
              "n_reps": n_reps, "n_starts": n_starts}

    CFG.snapshot(output_dir)
    CFG.snapshot(output_dir)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        crb_df.to_csv(os.path.join(output_dir, "design_grid_crb.csv"), index=False)
        mle_df.to_csv(os.path.join(output_dir, "design_grid_mle.csv"), index=False)
        with open(os.path.join(output_dir, "design_results.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        if verbose:
            print(f"  Saved design outputs to {output_dir}")

    return {"crb_df": crb_df, "mle_df": mle_df, "gate": gate_result}