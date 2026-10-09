"""
Per-epoch feature extraction (Q1_pipeline.md Sec 5) and per-subject
aggregation. Epochs are processed in parallel (one process per CPU core).

Primary measures (preregistration Sec 2):
    lzc, lzc_si            multichannel LZc and its spectrum-independent z-score
    exponent, offset       aperiodic fit, 1-40 Hz
    theta_r, theta_meta    Kuramoto order parameter / metastability (CSD data)
    theta_fiedler          normalised-Laplacian lambda_2 of the full theta wPLI matrix (CSD)
Secondary: relative band powers, theta mean wPLI, ACE, SCE, phi_r.
"""
from __future__ import annotations

import os
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .metrics.coalition import ace, sce
from .metrics.lzc import lzc
from .metrics.network import fiedler, wpli
from .metrics.phi_ar import phi_ar
from .metrics.spectral import aperiodic_fit, psd, relative_band_power
from .metrics.sync import analytic, kuramoto_order
from .preprocess import EpochSet, load_config
from .surrogate import spectrally_controlled

PRIMARY = ["lzc", "lzc_si", "exponent", "theta_r", "theta_fiedler"]


def epoch_features(X: np.ndarray, X_csd: np.ndarray | None, sfreq: float, cfg: dict, seed: int,
                   secondary: bool = True) -> dict:
    m = cfg["metrics"]
    rng = np.random.default_rng(seed)
    out = {}
    sc = spectrally_controlled(lambda Y: lzc(Y, norm=m["lzc_norm"]), X, n_surrogates=m["n_surrogates"],
                               multivariate=m["surrogate_multivariate"], rng=rng)
    out["lzc"], out["lzc_si"], out["lzc_surr_mean"] = sc["raw"], sc["z"], sc["surr_mean"]
    f, P = psd(X, sfreq)
    ap = aperiodic_fit(f, P, m["aperiodic"]["fmin"], m["aperiodic"]["fmax"], backend=m["aperiodic"]["backend"])
    out["exponent"], out["offset"] = ap["exponent_mean"], ap["offset_mean"]
    for k, v in relative_band_power(f, P, {k: tuple(b) for k, b in m["bands"].items()}).items():
        out[f"rel_{k}"] = v
    S = X_csd if X_csd is not None else X
    band = tuple(m["bands"][m["sync_band"]])
    z = analytic(S, sfreq, band)
    trim = int(0.5 * sfreq)
    z = z[:, trim:-trim]
    r = kuramoto_order(np.angle(z))
    out["theta_r"], out["theta_meta"] = float(r.mean()), float(r.std())
    W = wpli(z)
    out["theta_fiedler"] = fiedler(W, normalized=True)
    out["theta_wpli_mean"] = float(W[np.triu_indices(len(W), 1)].mean())
    if secondary:
        out["ace"] = ace(X, rng=rng)
        out["sce"] = sce(X, threshold=m["sce_threshold"], rng=rng)
        n_phi = m["phi"]["n_channels"]
        lag = max(1, int(round(m["phi"]["lag_ms"] / 1000 * sfreq)))
        idx = np.linspace(0, X.shape[0] - 1, n_phi).round().astype(int)
        out["phi_r"] = phi_ar(X[idx], lag=lag)["phi_r"]
    return out


def _job(args):
    X, Xc, sfreq, cfg, seed, secondary = args
    return epoch_features(X, Xc, sfreq, cfg, seed, secondary)


def extract(es: EpochSet, meta: dict, cfg: dict | None = None, secondary: bool = True,
            n_jobs: int | None = None) -> pd.DataFrame:
    """Features for every epoch of an EpochSet; `meta` columns (dataset,
    subject, condition, ...) are attached to each row."""
    cfg = cfg or load_config()
    base_seed = int(cfg["seed"])
    jobs = [(es.data[i], None if es.csd is None else es.csd[i], es.sfreq, cfg,
             base_seed + 7919 * i + zlib.crc32(repr(sorted(meta.items())).encode()) % 100003, secondary)
            for i in range(es.n_epochs)]
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    if n_jobs == 1 or len(jobs) < 4:
        rows = [_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            rows = list(ex.map(_job, jobs, chunksize=2))
    df = pd.DataFrame(rows)
    df.insert(0, "epoch", np.arange(len(df)))
    for k, v in reversed(list(meta.items())):
        df.insert(0, k, v)
    return df


def subject_means(df: pd.DataFrame, keys=("dataset", "subject", "condition")) -> pd.DataFrame:
    """Per-subject x condition means of every numeric feature, with epoch counts."""
    num = [c for c in df.columns if c not in keys and c != "epoch" and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby(list(keys))
    out = g[num].mean()
    out["n_epochs"] = g.size()
    return out.reset_index()
