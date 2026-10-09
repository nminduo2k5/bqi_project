"""
Goi ket qua 2 -- Structural and practical identifiability of StateSpaceDMN.

Structural: match polynomial coefficients of the transfer function from
process noise to each observable under 4 observation configurations.

Practical: Fisher information matrix (Hessian of kalman_loglik w.r.t.
log-params via second-order finite differences) and Cramer-Rao lower bound.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bqi.dmn_ode import StateSpaceDMN
from eeg.model_fit import kalman_loglik
from model import config as CFG
from model import config as CFG

# The four observation configurations
CONFIGS = {
    "a": {"has_y_A": False, "has_y_Phi": False, "has_rating": True},
    "b": {"has_y_A": True,  "has_y_Phi": False, "has_rating": True},
    "c": {"has_y_A": False, "has_y_Phi": True,  "has_rating": True},
    "d": {"has_y_A": True,  "has_y_Phi": True,  "has_rating": True},
}


def _mask_sessions(sessions: list[dict], config: str) -> list[dict]:
    """Mask out observation channels not in the given config."""
    cfg = CONFIGS[config]
    out = []
    for s in sessions:
        y = np.array(s["y"], float)
        rating = np.array(s["rating"], float)
        if not cfg["has_y_A"]:
            y[:, 0] = np.nan
        if not cfg["has_y_Phi"]:
            y[:, 1] = np.nan
        if not cfg["has_rating"]:
            rating[:] = np.nan
        out.append({"y": y, "rating": rating})
    return out


def structural_identifiability(model: StateSpaceDMN | None = None, tol: float = CFG.STRUCTURAL.eig_tol,
                               details: bool = False) -> dict:
    """Structural identifiability by Rothenberg (1971): with the exact expected
    Fisher information of a long, densely probed session (model.fisher), the
    parameters are locally identifiable iff the information is non-singular.
    Eigenvalues at machine precision mark exact non-identifiable directions
    (scaling symmetries of an unobserved latent state).

    Returns config -> list of parameters identifiable in principle (default),
    or the full spectrum analysis per config when `details=True`.
    gamma0 is set to a generic non-zero value so that no symmetry is hidden by 0.
    """
    from model.fisher import Design, fisher_exact, identifiability_spectrum
    model = (model or StateSpaceDMN()).with_params(gamma0=CFG.STRUCTURAL.gamma0_generic)
    names = list(StateSpaceDMN.FREE)
    result = {}
    for config in CONFIGS:
        I = fisher_exact(model, Design.regular(CFG.STRUCTURAL.session_min, CFG.STRUCTURAL.probe_every_min,
                                               config=config), names)
        spec = identifiability_spectrum(I, names, tol=tol)
        # Normalisation: fix ONE parameter per null direction (e.g. the unit of
        # an unobserved latent state), greedily the largest loading, until the
        # information over the remaining parameters is non-singular.
        free = [k for k in names if k not in spec["absent"]]
        fixed = []
        while True:
            idx = [names.index(k) for k in free]
            sub = identifiability_spectrum(I[np.ix_(idx, idx)], free, tol=tol)
            if not sub["null_directions"]:
                break
            load = sub["null_directions"][0]["direction"]
            k = max(load, key=lambda p: abs(load[p]))
            fixed.append(k)
            free.remove(k)
        spec["normalisation"] = fixed
        spec["estimable"] = free
        result[config] = spec if details else free
    return result


def fisher_information(model: StateSpaceDMN, sessions: list[dict], dt: float,
                        names: list[str] | None = None, h: float = 1e-4) -> np.ndarray:
    """Hessian of -log-likelihood w.r.t. log(theta) for POSITIVE params,
    theta for non-positive params. Uses central second-order finite differences.

    Returns positive semi-definite matrix (Fisher information = -E[Hessian]).
    """
    names = names or list(StateSpaceDMN.FREE)
    n = len(names)
    # pack current parameters
    theta0 = np.array([np.log(getattr(model, k)) if k in StateSpaceDMN.POSITIVE
                       else getattr(model, k) for k in names])

    def ll(theta):
        kw = {k: float(np.exp(v)) if k in StateSpaceDMN.POSITIVE else float(v)
              for k, v in zip(names, theta)}
        try:
            v = kalman_loglik(model.with_params(**kw), sessions, dt)
        except Exception:
            return -np.inf
        return v if np.isfinite(v) else -np.inf

    ll0 = ll(theta0)
    # Diagonal elements: second derivative by central differences
    diag = np.zeros(n)
    for i in range(n):
        ei = np.zeros(n); ei[i] = h
        lp = ll(theta0 + ei)
        lm = ll(theta0 - ei)
        diag[i] = -(lp - 2 * ll0 + lm) / (h ** 2)

    # Off-diagonal elements
    F = np.diag(diag)
    for i in range(n):
        for j in range(i + 1, n):
            ei = np.zeros(n); ei[i] = h
            ej = np.zeros(n); ej[j] = h
            lpp = ll(theta0 + ei + ej)
            lpm = ll(theta0 + ei - ej)
            lmp = ll(theta0 - ei + ej)
            lmm = ll(theta0 - ei - ej)
            cross = -(lpp - lpm - lmp + lmm) / (4 * h ** 2)
            F[i, j] = F[j, i] = cross

    return F


def cramer_rao_bound(F: np.ndarray, names: list[str], model: StateSpaceDMN | None = None) -> dict:
    """Cramer-Rao lower bound on standard errors from Fisher information F.

    `se_logspace` is the SE in the fitted scale (log for positive params,
    linear otherwise). `relative_se` = SE for log-params (relative error) and
    SE / |value| for linear params (A0, alpha, beta, gamma0); NaN if the value
    is 0 or no model is given.
    """
    model = model or StateSpaceDMN()
    try:
        cov = np.linalg.inv(F)
        se = np.sqrt(np.maximum(np.diag(cov), 0))
        if not np.all(np.isfinite(se)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        se = np.full(len(names), np.inf)
    out = {}
    for i, k in enumerate(names):
        if k in StateSpaceDMN.POSITIVE:
            rel = float(se[i])
        else:
            v = abs(getattr(model, k))
            rel = float(se[i] / v) if v > 1e-12 else float("nan")
        out[k] = {"se_logspace": float(se[i]), "relative_se": rel}
    return out


ABSENT_BY_CONFIG: dict = {}


def identifiability_table(model: StateSpaceDMN, n_subjects: int = CFG.REFERENCE.n_subjects,
                           session_min: float = CFG.REFERENCE.session_min,
                           probe_every_min: float = CFG.REFERENCE.probe_every_min,
                           seed: int = 42) -> pd.DataFrame:
    """Build Table 2: relative CRB of every parameter by observation configuration,
    from the exact expected Fisher information (no simulated data; `seed` unused,
    kept for a uniform interface). Parameters that are absent from the likelihood
    are marked 'absent'; one parameter per non-identifiable direction is fixed
    (normalisation, marked 'fixed') and the rest are estimated jointly."""
    from model.fisher import Design, fisher_exact
    full = structural_identifiability(model, details=True)
    struct = {c: v["estimable"] for c, v in full.items()}
    ABSENT_BY_CONFIG.clear()
    ABSENT_BY_CONFIG.update({c: v["absent"] for c, v in full.items()})
    rows = []
    for config in ["a", "b", "c", "d"]:
        names = list(struct[config])
        crb = {}
        if names:
            d = Design.regular(session_min, probe_every_min, config=config)
            F = fisher_exact(model, d, names, n_subjects=n_subjects)
            crb = cramer_rao_bound(F, names, model)
        for k in StateSpaceDMN.FREE:
            if k not in crb:
                val = "fixed" if k not in ABSENT_BY_CONFIG.get(config, []) else "absent"
            else:
                se = crb[k]["relative_se"]
                val = round(se, 4) if np.isfinite(se) else f"se={crb[k]['se_logspace']:.3f}"
            rows.append({"config": config, "param": k, "relative_se": val})
    df = pd.DataFrame(rows)
    return df.pivot(index="param", columns="config", values="relative_se").reindex(list(StateSpaceDMN.FREE))


def run(seed: int = 42, verbose: bool = True, output_dir: str | None = None) -> dict:
    """Entry point for main.py model identifiability subcommand."""
    from model.fisher import Design, fisher_exact
    model = StateSpaceDMN()
    struct = structural_identifiability(model, details=True)
    if verbose:
        print(f"Structural identifiability (Rothenberg, exact Fisher, {CFG.STRUCTURAL.session_min:g} min, "
              f"probe/{CFG.STRUCTURAL.probe_every_min:g} min):")
        for cfg, s in struct.items():
            print(f"  Config {cfg}: rank {s['rank']}/{s['n_identifiable_candidates']}, absent {s['absent']}, "
                  f"normalisation (fixed) {s['normalisation']}")
            for nd in s["null_directions"]:
                print(f"      non-identifiable direction: {nd['direction']}")

    if verbose:
        R = CFG.REFERENCE
        print(f"\nExact Fisher + CRB (config {R.config}, {R.n_subjects} subjects x {R.session_min:g} min, "
              f"probe every {R.probe_every_min:g} min):")
    names = list(StateSpaceDMN.FREE)
    R = CFG.REFERENCE
    F = fisher_exact(model, Design.regular(R.session_min, R.probe_every_min, config=R.config), names,
                     n_subjects=R.n_subjects)
    crb = cramer_rao_bound(F, names, model)
    if verbose:
        for k, v in crb.items():
            print(f"  {k:12s}: rel_se = {v['relative_se']:.3f}")

    if verbose:
        print("\nIdentifiability table (relative SE by config):")
    table = identifiability_table(model, seed=seed)
    if verbose:
        print(table.to_string())

    result = {"structural": struct, "crb": crb, "fisher_diag": np.diag(F).tolist()}

    CFG.snapshot(output_dir)
    CFG.snapshot(output_dir)
    if output_dir:
        import json
        os.makedirs(output_dir, exist_ok=True)
        table.to_csv(os.path.join(output_dir, "table2_identifiability.csv"))
        with open(os.path.join(output_dir, "identifiability_results.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        if verbose:
            print(f"  Saved to {output_dir}")

    return result