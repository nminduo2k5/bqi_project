"""
Statistics (Q1_pipeline.md Sec 6.1): paired effect sizes with subject
bootstrap CIs, linear mixed models, TOST equivalence, FDR, and a sign-flip
cluster permutation test over channels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


def paired_dz(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, rng=None) -> dict:
    """d_z = mean(a - b) / sd(a - b) with a percentile bootstrap CI over subjects."""
    rng = np.random.default_rng(rng)
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    dz = d.mean() / d.std(ddof=1)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        s = d[rng.integers(0, n, n)]
        sd = s.std(ddof=1)
        boot[i] = s.mean() / sd if sd > 0 else 0.0
    t = sps.ttest_1samp(d, 0.0)
    return {"n": n, "mean_diff": float(d.mean()), "dz": float(dz),
            "ci": tuple(np.percentile(boot, [2.5, 97.5])), "t": float(t.statistic), "p": float(t.pvalue)}


def tost_paired(a: np.ndarray, b: np.ndarray, bound_dz: float = 0.2) -> dict:
    """Two one-sided tests for equivalence of paired means within +/- bound_dz
    (in units of the SD of the differences). Equivalent if p_tost < alpha."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    n, sd = len(d), d.std(ddof=1)
    se = sd / np.sqrt(n)
    delta = bound_dz * sd
    p_lower = sps.t.sf((d.mean() + delta) / se, n - 1)   # H0: mean <= -delta
    p_upper = sps.t.cdf((d.mean() - delta) / se, n - 1)  # H0: mean >= +delta
    return {"p_tost": float(max(p_lower, p_upper)), "bound": float(delta)}


def fdr_bh(p: np.ndarray, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg. Returns (reject mask, adjusted p)."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    ranked = p[order] * n / np.arange(1, n + 1)
    adj[order] = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    return adj <= q, adj


def mixed_model(df: pd.DataFrame, metric: str, condition: str = "condition", group: str = "subject",
                reference: str | None = None, vc: dict | None = None) -> dict:
    """metric ~ condition + (1 | subject) [+ variance components, e.g. session]."""
    import statsmodels.formula.api as smf
    d = df[[metric, condition, group] + list((vc or {}).keys())].dropna().copy()
    cond = f"C({condition}, Treatment('{reference}'))" if reference else f"C({condition})"
    vc_formula = {k: f"0 + C({k})" for k in (vc or {})} or None
    model = smf.mixedlm(f"{metric} ~ {cond}", d, groups=d[group], vc_formula=vc_formula)
    res = model.fit(reml=True, method="lbfgs")
    rows = [{"term": k, "coef": float(res.params[k]), "se": float(res.bse[k]), "p": float(res.pvalues[k])}
            for k in res.params.index if k.startswith("C(")]
    return {"terms": rows, "n_obs": int(res.nobs), "n_groups": int(len(res.model.group_labels)),
            "converged": bool(res.converged)}


def cluster_permutation(D: np.ndarray, adjacency: np.ndarray, n_perm: int = 5000, t_thresh: float | None = None,
                        rng=None) -> dict:
    """Sign-flip cluster test on paired differences D: (n_subjects, n_channels).
    Clusters = connected sets of channels (adjacency) with |t| above threshold
    and the same sign; statistic = sum of t; p = fraction of permutations whose
    maximum |cluster mass| reaches the observed one."""
    rng = np.random.default_rng(rng)
    D = np.asarray(D, float)
    n, n_ch = D.shape
    if t_thresh is None:
        t_thresh = sps.t.ppf(0.975, n - 1)
    A = np.asarray(adjacency, bool)

    def tvals(X):
        return X.mean(0) / (X.std(0, ddof=1) / np.sqrt(n))

    def clusters(t):
        out = []
        for sign in (1, -1):
            mask = sign * t > t_thresh
            seen = np.zeros(n_ch, bool)
            for c in range(n_ch):
                if mask[c] and not seen[c]:
                    stack, members = [c], []
                    seen[c] = True
                    while stack:
                        k = stack.pop()
                        members.append(k)
                        for j in np.flatnonzero(A[k] & mask & ~seen):
                            seen[j] = True
                            stack.append(j)
                    out.append((sorted(members), float(t[members].sum())))
        return out

    t_obs = tvals(D)
    obs = clusters(t_obs)
    null_max = np.empty(n_perm)
    for i in range(n_perm):
        cl = clusters(tvals(D * rng.choice([-1.0, 1.0], n)[:, None]))
        null_max[i] = max((abs(m) for _, m in cl), default=0.0)
    return {"t": t_obs, "clusters": [{"channels": ch, "mass": m,
                                      "p": float((np.sum(null_max >= abs(m)) + 1) / (n_perm + 1))}
                                     for ch, m in obs]}
