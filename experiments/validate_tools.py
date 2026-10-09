"""
Tool validation on synthetic data with known ground truth (Q1_pipeline.md
Sec 8) -> gate G0; state-space parameter recovery -> gate G3.

    python main.py validate-tools            # G0
    python main.py validate-tools --g3       # G0 + G3 (slow: maximum likelihood fits)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import yaml
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bqi.pci import lempel_ziv_complexity
from bqi.spectral_graph import build_small_world_brain_graph, graph_adjacency, simulate_kuramoto_graph
from eeg import gates
from eeg.axis import build_axis, bootstrap_projection
from eeg.metrics.lzc import lz76_phrase_count, lzc
from eeg.metrics.network import fiedler, wpli_matrix
from eeg.metrics.phi_ar import phi_ar
from eeg.metrics.sync import synchrony_metrics
from eeg.surrogate import phase_randomize, spectrally_controlled

CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "eeg", "config.yaml")
SFREQ = 250.0


def load_cfg() -> dict:
    with open(CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# =============================================================================
# Synthetic generators with known ground truth
# =============================================================================
def mixed_signal(level: float, n_ch: int = 19, n_t: int = 2500, rng=None) -> np.ndarray:
    """Alpha carrier with a slow amplitude envelope shared by all channels (low
    envelope complexity, the quantity LZc measures) mixed with independent
    white noise (high complexity) in proportion `level`, then 1-40 Hz.
    A constant-amplitude sinusoid would NOT be simple for LZc: its envelope is
    flat and binarises to noise."""
    from eeg.metrics.sync import bandpass
    rng = np.random.default_rng(rng)
    t = np.arange(n_t) / SFREQ
    env = 1 + 0.8 * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, 2 * np.pi))
    common = env * np.sin(2 * np.pi * 10.0 * t)
    gains = rng.uniform(0.5, 1.5, (n_ch, 1))
    X = (1 - level) * gains * common[None, :] + level * rng.normal(0, 1, (n_ch, n_t))
    return bandpass(X, SFREQ, 1.0, 40.0)


def power_law_noise(exponent: float, n_ch: int = 19, n_t: int = 2500, rng=None) -> np.ndarray:
    """Gaussian linear noise with PSD ~ 1/f^exponent (random phases)."""
    rng = np.random.default_rng(rng)
    f = np.fft.rfftfreq(n_t, 1 / SFREQ)
    amp = np.zeros_like(f)
    amp[1:] = f[1:] ** (-exponent / 2)
    ph = rng.uniform(0, 2 * np.pi, (n_ch, len(f)))
    return np.fft.irfft(amp[None, :] * np.exp(1j * ph), n=n_t, axis=1)


def kuramoto_eeg(K: float, n_t: int = 2500, rng=0) -> np.ndarray:
    """EEG-like signals from a Kuramoto network on a small-world graph with
    theta-band natural frequencies (6 +/- 0.5 Hz) plus measurement noise."""
    rng = np.random.default_rng(rng)
    W = graph_adjacency(build_small_world_brain_graph(19, 4, 0.2, seed=int(rng.integers(1e6))))
    _, ph = simulate_kuramoto_graph(W, K, T=n_t / SFREQ, dt=1 / SFREQ, natural_freq_mean=2 * np.pi * 6.0,
                                    natural_freq_std=2 * np.pi * 0.5, seed=int(rng.integers(1e6)),
                                    return_phases=True)
    return np.cos(ph.T) + 0.3 * rng.normal(size=(19, ph.shape[0]))


def var_process(coupled: bool, n_t: int = 5000, rng=0) -> np.ndarray:
    """4-channel VAR(1): two independent 2-channel blocks, optionally coupled."""
    rng = np.random.default_rng(rng)
    A = np.zeros((4, 4))
    A[:2, :2] = [[0.5, 0.3], [0.3, 0.5]]
    A[2:, 2:] = [[0.5, 0.3], [0.3, 0.5]]
    if coupled:
        A[0, 2] = A[2, 0] = 0.3
    X = np.zeros((4, n_t))
    for t in range(1, n_t):
        X[:, t] = A @ X[:, t - 1] + rng.normal(size=4)
    return X


# =============================================================================
# G0 checks
# =============================================================================
def run_g0(cfg: dict, verbose: bool = True) -> dict:
    g = cfg["gates"]["G0"]
    rng = np.random.default_rng(cfg["seed"])
    checks = {}

    def log(name, ok, info):
        checks[name] = {"passed": bool(ok), **info}
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {info}", flush=True)

    # 1. fast LZ76 == Kaspar-Schuster reference
    bad = 0
    for _ in range(200):
        n = int(rng.integers(3, 300))
        s = "".join("1" if x else "0" for x in rng.random(n) < rng.uniform(0.05, 0.95))
        bad += lempel_ziv_complexity(s) != lz76_phrase_count(s)
    log("lz76_matches_reference", bad <= g["lz76_reference_mismatches"], {"mismatches": int(bad), "n": 200})

    # 2. LZc increases monotonically with the true complexity level
    levels = np.linspace(0.05, 0.95, 10)
    vals = [np.mean([lzc(mixed_signal(l, rng=rng)) for _ in range(3)]) for l in levels]
    rho = spearmanr(levels, vals).statistic
    log("lzc_monotonic_in_complexity", rho >= g["lzc_monotonic_spearman"],
        {"spearman": round(float(rho), 3), "lzc_range": [round(min(vals), 3), round(max(vals), 3)]})

    # 3. surrogate preserves amplitude spectra and cross-spectra
    X = mixed_signal(0.5, rng=rng)
    S = phase_randomize(X, rng=rng)
    FX, FS = np.fft.rfft(X, axis=1), np.fft.rfft(S, axis=1)
    same_amp = np.allclose(np.abs(FX), np.abs(FS), rtol=1e-6, atol=1e-8)
    cross = np.allclose(FX[0] * np.conj(FX[1]), FS[0] * np.conj(FS[1]), rtol=1e-6, atol=1e-6)
    log("surrogate_preserves_spectra", same_amp and cross, {"amplitude": bool(same_amp), "cross_spectrum": bool(cross)})

    # 4. raw LZc depends on the spectral exponent; LZc_si does not
    exps = [0.5, 1.0, 1.5, 2.0, 2.5]
    raw, zs = [], []
    for e in exps:
        r = spectrally_controlled(lzc, power_law_noise(e, rng=rng), n_surrogates=cfg["metrics"]["n_surrogates"],
                                  rng=rng)
        raw.append(r["raw"])
        zs.append(r["z"])
    log("raw_lzc_is_spectrum_sensitive", (max(raw) - min(raw)) >= g["raw_lzc_spectral_sensitivity_min_range"],
        {"raw_lzc": [round(v, 3) for v in raw]})
    log("lzc_si_is_spectrum_invariant", float(np.max(np.abs(zs))) <= g["lzc_si_spectral_invariance_max_abs_z"],
        {"z": [round(v, 2) for v in zs]})

    # 5. Kuramoto r and wPLI Fiedler recover coupling on a graph
    lo, hi = kuramoto_eeg(2.0, rng=1), kuramoto_eeg(200.0, rng=1)
    band = tuple(cfg["metrics"]["bands"]["theta"])
    r_lo, r_hi = synchrony_metrics(lo, SFREQ, band)["kuramoto_r"], synchrony_metrics(hi, SFREQ, band)["kuramoto_r"]
    log("kuramoto_r_tracks_coupling", r_hi - r_lo >= g["kuramoto_r_increase_min"],
        {"r_low_K": round(r_lo, 3), "r_high_K": round(r_hi, 3)})
    f_lo = fiedler(wpli_matrix(lo, SFREQ, band), normalized=True)
    f_hi = fiedler(wpli_matrix(hi, SFREQ, band), normalized=True)
    log("wpli_fiedler_tracks_coupling", f_hi > f_lo + 0.1,
        {"fiedler_norm_low_K": round(f_lo, 3), "fiedler_norm_high_K": round(f_hi, 3)})

    # 6. Phi_AR ~ 0 for independent blocks, > 0 when coupled
    r0, r1 = phi_ar(var_process(False, rng=2)), phi_ar(var_process(True, rng=2))
    p0, p1 = r0["phi_r"], r1["phi_r"]
    log("phi_r_detects_integration", p0 <= g["phi_ar_disconnected_max"] and p1 > p0 + 0.01,
        {"phi_r_independent": round(p0, 4), "phi_r_coupled": round(p1, 4),
         "phi_wms_coupled": round(r1["phi_wms"], 4)})

    # 7. axis recovers a planted direction and projection
    K = 5
    u_true = np.array([1.0, -0.5, 0.8, 0.6, 0.3]); u_true /= np.linalg.norm(u_true)
    D1 = 1.0 * u_true + rng.normal(0, 0.3, (25, K))
    D2 = 1.2 * u_true + rng.normal(0, 0.3, (20, K))
    ax = build_axis(D1, D2)
    ortho = np.array([0.5, 1.0, 0.0, -0.5, 0.0]); ortho -= (ortho @ u_true) * u_true
    Dm = 0.5 * 1.1 * u_true + 0.8 * ortho / np.linalg.norm(ortho) + rng.normal(0, 0.3, (30, K))
    bp = bootstrap_projection(Dm, ax, n_boot=2000, rng=rng)
    ok = abs(ax["u"] @ u_true) > 0.99 and bp["c_ci"][0] < 0.5 < bp["c_ci"][1] and bp["residual_p"] < 0.05
    log("axis_recovers_planted_structure", ok,
        {"cos_u": round(float(ax["u"] @ u_true), 4), "c_mean": round(bp["c_mean"], 3),
         "c_ci": [round(x, 3) for x in bp["c_ci"]], "residual_p": bp["residual_p"]})

    passed = all(c["passed"] for c in checks.values())
    path = gates.record("G0", passed, checks)
    if verbose:
        print(f"G0 {'PASS' if passed else 'FAIL'} -> {path}")
    return {"passed": passed, "checks": checks}


# =============================================================================
# G3: parameter recovery of the state-space model (ds001787-like design)
# =============================================================================
def run_g3(cfg: dict, n_subjects: int = 12, n_epochs: int = 270, probe_every: int = 12, n_datasets: int = 3,
           verbose: bool = True, force: bool = False) -> dict:
    from bqi.dmn_ode import StateSpaceDMN
    from eeg.model_fit import fit
    gates.require("G0", force=force)
    g = cfg["gates"]["G3"]
    dt = cfg["preprocess"]["epoch_s"] / 60.0  # minutes
    true = StateSpaceDMN()
    rows = []
    for d in range(n_datasets):
        sess = [true.simulate(n_epochs, dt, probe_every=probe_every, seed=1000 * d + s) for s in range(n_subjects)]
        init = true.with_params(**{k: getattr(true, k) * float(np.exp(np.random.default_rng(d).normal(0, 0.3)))
                                   for k in StateSpaceDMN.POSITIVE})
        t0 = time.time()
        r = fit(sess, dt, init=init, n_starts=2, rng=d)
        rel = {k: abs(r["params"][k] - getattr(true, k)) / abs(getattr(true, k)) for k in g["params"]}
        rows.append({"dataset": d, "seconds": round(time.time() - t0, 1), "rel_error": rel,
                     "fit": {k: r["params"][k] for k in r["names"]}})
        if verbose:
            print(f"  dataset {d}: " + ", ".join(f"{k} {v:.1%}" for k, v in rel.items()), flush=True)
    med = {k: float(np.median([row["rel_error"][k] for row in rows])) for k in g["params"]}
    passed = all(v <= g["max_rel_error"] for v in med.values())
    details = {"design": {"n_subjects": n_subjects, "n_epochs": n_epochs, "probe_every": probe_every,
                          "dt_min": dt}, "median_rel_error": med, "runs": rows,
               "true": {k: getattr(true, k) for k in StateSpaceDMN.FREE}}
    path = gates.record("G3", passed, details)
    if verbose:
        print(f"G3 {'PASS' if passed else 'FAIL'} (median rel. error {med}) -> {path}")
    return {"passed": passed, **details}


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Synthetic tool validation (gates G0, G3).")
    p.add_argument("--g3", action="store_true", help="Also run state-space parameter recovery (slow)")
    p.add_argument("--force-gate", action="store_true")
    a = p.parse_args(argv)
    cfg = load_cfg()
    print("=== G0: tool validation ===")
    r0 = run_g0(cfg)
    if a.g3:
        print("=== G3: parameter recovery ===")
        run_g3(cfg, force=a.force_gate)
    return 0 if r0["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
