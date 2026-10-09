"""
Conscious-level axis (Q1_pipeline.md Sec 6.3).

Inputs are per-subject difference vectors over K metrics, already z-scored
within each dataset:  D_sleep = wake - N3,  D_prop = baseline - moderate,
D_med = meditation - mind-wandering. The axis u is the first principal
direction of the pooled reference differences (sign chosen so the mean
reference difference projects positively). Each meditation subject gets a
projection c (in units of the mean reference shift) and an orthogonal
residual; both are bootstrapped over subjects and the residual is compared
with a sign-flip null.
"""
from __future__ import annotations

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_axis(*reference_deltas: np.ndarray) -> dict:
    """reference_deltas: arrays (n_subjects_i, K). Returns the unit axis, the
    agreement between reference contrasts (cosine of their means; gate G2 needs
    > 0.5) and the length of the mean reference shift along the axis."""
    means = [np.asarray(D, float).mean(axis=0) for D in reference_deltas]
    pooled = np.vstack([np.asarray(D, float) for D in reference_deltas])
    # first principal direction of the uncentred differences = direction of
    # the dominant shift (centring would remove exactly the effect we want)
    _, _, Vt = np.linalg.svd(pooled, full_matrices=False)
    u = Vt[0]
    ref_mean = pooled.mean(axis=0)
    if u @ ref_mean < 0:
        u = -u
    agreement = cosine(means[0], means[1]) if len(means) >= 2 else float("nan")
    along = pooled @ u
    return {"u": u, "agreement": agreement, "ref_shift": float(ref_mean @ u),
            "explained": float((along ** 2).sum() / (pooled ** 2).sum())}


def project(D: np.ndarray, axis: dict) -> dict:
    """Per-subject projection c (units of the mean reference shift) and the
    norm of the orthogonal residual."""
    D = np.asarray(D, float)
    u = axis["u"]
    along = D @ u
    resid = D - np.outer(along, u)
    return {"c": along / axis["ref_shift"], "along": along, "residual_norm": np.linalg.norm(resid, axis=1)}


def bootstrap_projection(D: np.ndarray, axis: dict, n_boot: int = 10000, rng=None) -> dict:
    """Bootstrap over subjects for mean c and mean residual norm, plus a
    sign-flip null for the residual (is the orthogonal shift larger than
    expected from noise around zero?)."""
    rng = np.random.default_rng(rng)
    D = np.asarray(D, float)
    n = len(D)
    p = project(D, axis)
    c_mean = float(p["c"].mean())
    resid_vec = (D - np.outer(p["along"], axis["u"])).mean(axis=0)
    resid_obs = float(np.linalg.norm(resid_vec))
    boot_c, boot_r = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        pb = project(D[idx], axis)
        boot_c[b] = pb["c"].mean()
        boot_r[b] = np.linalg.norm((D[idx] - np.outer(pb["along"], axis["u"])).mean(axis=0))
    null_r = np.empty(n_boot)
    R = D - np.outer(p["along"], axis["u"])
    for b in range(n_boot):
        signs = rng.choice([-1.0, 1.0], n)[:, None]
        null_r[b] = np.linalg.norm((R * signs).mean(axis=0))
    return {"c_mean": c_mean, "c_ci": tuple(np.percentile(boot_c, [2.5, 97.5])),
            "residual_mean_norm": resid_obs, "residual_ci": tuple(np.percentile(boot_r, [2.5, 97.5])),
            "residual_p": float((np.sum(null_r >= resid_obs) + 1) / (n_boot + 1))}
