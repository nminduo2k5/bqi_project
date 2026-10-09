"""
Random-effects meta-analysis of meditation-induced DMN BOLD suppression,
paper Sec. 7.3 (Table meta, Fig forest). Implements DerSimonian-Laird
random-effects pooling and the I^2 heterogeneity statistic.

    I^2 = (Q - (k-1)) / Q * 100%
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# The 16 published studies from Table meta (paper Sec 7.3), hardcoded as the
# reference dataset (this is the paper's ground truth, used to validate the
# pooling implementation against the reported pooled d=1.70, I^2=31%).
STUDIES_REFERENCE = [
    {"study": "Brewer et al. (2011)",            "N": 12, "type": "FA/LKM/CA",              "delta_pct": -42.3, "delta_sd": 6.1,  "d": 1.82, "p": 0.001},
    {"study": "Jang et al. (2011)",               "N": 20, "type": "Brain-wave vibration",   "delta_pct": -28.7, "delta_sd": 8.4,  "d": 1.21, "p": 0.001},
    {"study": "Garrison et al. (2015)",           "N": 23, "type": "Focused attention",      "delta_pct": -51.2, "delta_sd": 9.2,  "d": 2.14, "p": 0.001},
    {"study": "Panda et al. (2016)",               "N": 15, "type": "Mixed (fMRI+EEG)",       "delta_pct": -38.6, "delta_sd": 7.3,  "d": 1.65, "p": 0.001},
    {"study": "Tang et al. (2015)",                "N": 68, "type": "Integrative body-mind",  "delta_pct": -31.4, "delta_sd": 5.9,  "d": 1.34, "p": 0.001},
    {"study": "Berkovich-Ohana et al. (2022)",     "N": 31, "type": "MBSR mindfulness",       "delta_pct": -44.8, "delta_sd": 8.1,  "d": 1.91, "p": 0.001},
    {"study": "Josipovic (2014)",                  "N": 16, "type": "Non-dual awareness",     "delta_pct": -57.3, "delta_sd": 11.4, "d": 2.41, "p": 0.001},
    {"study": "Lutz et al. (2008)",                "N": 32, "type": "Open monitoring",        "delta_pct": -39.1, "delta_sd": 6.7,  "d": 1.68, "p": 0.001},
    {"study": "Hölzel et al. (2011)",              "N": 16, "type": "MBSR (8-week)",          "delta_pct": -33.2, "delta_sd": 7.8,  "d": 1.42, "p": 0.001},
    {"study": "Kilpatrick et al. (2011)",          "N": 40, "type": "FA training",            "delta_pct": -29.8, "delta_sd": 5.4,  "d": 1.27, "p": 0.001},
    {"study": "Taylor et al. (2011)",              "N": 26, "type": "Zen/Vipassana",          "delta_pct": -48.7, "delta_sd": 9.9,  "d": 2.07, "p": 0.001},
    {"study": "Manna et al. (2010)",               "N": 14, "type": "FA + OM",                "delta_pct": -36.4, "delta_sd": 8.2,  "d": 1.55, "p": 0.001},
    {"study": "Pagnoni (2012)",                    "N": 24, "type": "Zen (long-term)",         "delta_pct": -53.9, "delta_sd": 10.3, "d": 2.29, "p": 0.001},
    {"study": "Van Dam et al. (2018)",             "N": 82, "type": "Loving-kindness",        "delta_pct": -27.3, "delta_sd": 4.8,  "d": 1.16, "p": 0.001},
    {"study": "Farb et al. (2010)",                "N": 18, "type": "Mindfulness (8-week)",   "delta_pct": -35.7, "delta_sd": 7.1,  "d": 1.52, "p": 0.001},
    {"study": "Simon & Engström (2015)",           "N": 31, "type": "Mantra meditation",       "delta_pct": -41.6, "delta_sd": 8.5,  "d": 1.77, "p": 0.001},
]


def hedges_g_variance(d: float, n: int) -> float:
    """Approximate sampling variance of a within-group Cohen's d (one-sample /
    paired design proxy): Var(d) ~ 1/n + d^2/(2n)."""
    return 1.0 / n + (d ** 2) / (2 * n)


def random_effects_meta_analysis(studies: list[dict] | None = None) -> dict:
    """DerSimonian-Laird random-effects pooling of Cohen's d across studies.

    Returns pooled d, its CI, Q, I^2, tau^2, and per-study weights -- directly
    reproducing the numbers in Table meta / Fig forest.
    """
    studies = studies or STUDIES_REFERENCE
    d = np.array([s["d"] for s in studies])
    n = np.array([s["N"] for s in studies])
    v = np.array([hedges_g_variance(di, ni) for di, ni in zip(d, n)])
    w_fixed = 1.0 / v

    # Fixed-effect pooled estimate & Q statistic
    d_fixed = np.sum(w_fixed * d) / np.sum(w_fixed)
    Q = np.sum(w_fixed * (d - d_fixed) ** 2)
    k = len(studies)
    df = k - 1
    I2 = max(0.0, (Q - df) / Q * 100) if Q > 0 else 0.0

    # DerSimonian-Laird between-study variance tau^2
    C = np.sum(w_fixed) - np.sum(w_fixed ** 2) / np.sum(w_fixed)
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0

    # Random-effects weights & pooled estimate
    w_re = 1.0 / (v + tau2)
    d_re = np.sum(w_re * d) / np.sum(w_re)
    se_re = np.sqrt(1.0 / np.sum(w_re))
    ci_lower, ci_upper = d_re - 1.96 * se_re, d_re + 1.96 * se_re

    return {
        "pooled_d": float(d_re),
        "se": float(se_re),
        "ci": (float(ci_lower), float(ci_upper)),
        "Q": float(Q),
        "I2_percent": float(I2),
        "tau2": float(tau2),
        "weights_re": w_re,
        "per_study_d": d,
        "k_studies": k,
        "total_N": int(n.sum()),
    }


def pooled_percent_change(studies: list[dict] | None = None) -> dict:
    """Inverse-variance-weighted pooled percentage DMN BOLD change (the
    'Delta%' column of Table meta), with its own random-effects CI."""
    studies = studies or STUDIES_REFERENCE
    delta = np.array([s["delta_pct"] for s in studies])
    sd = np.array([s["delta_sd"] for s in studies])
    n = np.array([s["N"] for s in studies])
    v = (sd ** 2) / n
    w = 1.0 / v
    pooled = np.sum(w * delta) / np.sum(w)
    se = np.sqrt(1.0 / np.sum(w))
    return {"pooled_delta_pct": float(pooled), "ci": (float(pooled - 1.96 * se), float(pooled + 1.96 * se))}


def generate_resampled_meta_dataset(studies: list[dict] | None = None, n_bootstrap: int = 2000,
                                     seed: int = 0) -> pd.DataFrame:
    """A *new* dataset: bootstrap-resampled per-study effect sizes (drawing
    each study's d from N(d_i, se_i)) to build an empirical distribution of
    the pooled estimate -- used to sanity-check the analytic random-effects
    CI against a resampling-based CI."""
    studies = studies or STUDIES_REFERENCE
    rng = np.random.default_rng(seed)
    d = np.array([s["d"] for s in studies])
    n = np.array([s["N"] for s in studies])
    v = np.array([hedges_g_variance(di, ni) for di, ni in zip(d, n)])
    se = np.sqrt(v)

    pooled_samples = []
    for _ in range(n_bootstrap):
        d_sample = rng.normal(d, se)
        w = 1.0 / v
        pooled_samples.append(np.sum(w * d_sample) / np.sum(w))
    return pd.DataFrame({"bootstrap_pooled_d": pooled_samples})
