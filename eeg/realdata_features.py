"""
ds001787: preprocessing and per-epoch features for the real-data section
(Q1_pipeline.md v2.0 Sec 4-5). One npz per session in
data/ds001787/derivatives/features/.

Pipeline per session
  1. read BDF (64 EEG channels, biosemi64 montage), 1-40 Hz FIR + 50 Hz notch
  2. bad channels (robust log-variance z > 4, reuses eeg.preprocess.mark_bad_channels)
     interpolated; average reference
  3. ICA (picard if installed, else fastica); components rejected by ICLabel
     (mne-icalabel, p >= 0.8 eye/muscle) when available, otherwise by a
     documented heuristic (|r| > 0.8 with Fp1/Fp2, or > 60 % power above 20 Hz)
  4. regular 10-s grid from the recording start (epoch end times t_k = 10 k s);
     an epoch is *missing* (not deleted) when it overlaps a probe window
     [onset, last response + 3 s] (onset + 15 s without response) or its
     peak-to-peak exceeds 150 uV on any channel
  5. features per epoch: alpha_post (log 8-12 Hz power, posterior channels),
     theta_front (log 4-7 Hz, frontal midline), alpha_specparam (log periodic
     alpha = alpha minus aperiodic fit at 10 Hz), lzc_si (spectrum-controlled
     LZc, 19 common channels, 20 surrogates), exponent_neg (-aperiodic exponent)
  6. ratings from derivatives/probes.csv (E0) at the probe onsets
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eeg.loaders import ds001787 as DS
from eeg.metrics.lzc import lzc
from eeg.metrics.spectral import aperiodic_fit, psd
from eeg.preprocess import load_config, map_common_channels, mark_bad_channels
from eeg.surrogate import spectrally_controlled
from model import config as CFG

RD = CFG.REALDATA
FEATURES_DIR = os.path.join(DS.data_dir(), "derivatives", "features")
EEG_CHANNELS = 64


# =============================================================================
# Raw -> clean continuous data
# =============================================================================
def channel_labels() -> list[str]:
    """10-20 labels of the 64 EEG channels, in BDF order (A1..B32), from the
    dataset's task-meditation_channels.tsv. The BDF files themselves carry the
    raw Biosemi labels A1..B32."""
    tsv = os.path.join(DS.bids_dir(), "task-meditation_channels.tsv")
    ch = pd.read_csv(tsv, sep="\t")
    labels = ch.loc[ch["type"].str.upper() == "EEG", "name"].tolist()
    if len(labels) != EEG_CHANNELS:
        raise ValueError(f"expected {EEG_CHANNELS} EEG labels in {tsv}, got {len(labels)}")
    return labels


def load_raw(bdf_path: str):
    import mne
    raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose="error")
    raw_names = raw.ch_names[:EEG_CHANNELS]
    raw.pick(raw_names)
    labels = channel_labels() if raw_names[0] == "A1" else raw_names
    raw.rename_channels(dict(zip(raw_names, labels)))
    raw.set_channel_types({ch: "eeg" for ch in labels})
    raw.set_montage("biosemi64", on_missing="raise", verbose="error")
    return raw


def _ica_reject(raw, ica, log: dict) -> list[int]:
    """Component indices to reject: ICLabel if available, else heuristics."""
    try:
        from mne_icalabel import label_components
        lab = label_components(raw, ica, method="iclabel")
        probs, labels = np.asarray(lab["y_pred_proba"]), lab["labels"]
        bad = [i for i, (p, l) in enumerate(zip(probs, labels)) if l in ("eye blink", "muscle artifact") and p >= 0.8]
        log["ica_method"] = "iclabel"
        return bad
    except Exception as e:  # not installed or failed
        log["ica_method"] = f"heuristic ({type(e).__name__})"
    src = ica.get_sources(raw).get_data()
    fp = raw.get_data(picks=[c for c in ("Fp1", "Fp2") if c in raw.ch_names])
    bad = []
    sf = raw.info["sfreq"]
    for i, s in enumerate(src):
        r = max(abs(np.corrcoef(s, f)[0, 1]) for f in fp) if len(fp) else 0.0
        f, P = psd(s[None, :], sf, nperseg_s=2.0)
        hi = float(np.trapezoid(P[0, f >= 20], f[f >= 20]) / max(np.trapezoid(P[0, f >= 1], f[f >= 1]), 1e-30))
        if r > 0.8 or hi > 0.6:
            bad.append(i)
    return bad


def preprocess_raw(raw, log: dict, do_ica: bool = True):
    import mne
    raw = raw.copy()
    raw.filter(RD.l_freq, RD.h_freq, fir_design="firwin", verbose="error")
    raw.notch_filter(RD.notch_hz, verbose="error")
    bads = mark_bad_channels(raw, 4.0)
    log["bad_channels"] = bads
    if bads:
        raw.info["bads"] = bads
        raw.interpolate_bads(reset_bads=True, verbose="error")
    raw.set_eeg_reference("average", projection=False, verbose="error")
    if do_ica:
        try:
            import picard  # noqa: F401
            method = "picard"
        except ImportError:
            method = "fastica"
        n_comp = min(30, len(raw.ch_names) - len(bads) - 1)
        ica = mne.preprocessing.ICA(n_components=n_comp, method=method, random_state=CFG.SEED, max_iter="auto",
                                    verbose="error")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ica.fit(raw, verbose="error")
        bad_ics = _ica_reject(raw, ica, log)
        ica.exclude = bad_ics
        raw = ica.apply(raw, verbose="error")
        log["ica_components"] = n_comp
        log["ica_rejected"] = bad_ics
    return raw


# =============================================================================
# Epoch grid and features
# =============================================================================
def epoch_grid(raw, duration_s: float, probes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Regular 10-s epochs (end times), data (n_ep, n_ch, n_times), mask."""
    sf = raw.info["sfreq"]
    ep = RD.epoch_s
    n_ep = int(np.floor(min(duration_s, raw.times[-1]) / ep))
    n_t = int(round(ep * sf))
    X = raw.get_data()
    data = np.stack([X[:, k * n_t:(k + 1) * n_t] for k in range(n_ep)]) if n_ep else np.zeros((0, X.shape[0], n_t))
    t_end = np.arange(1, n_ep + 1) * ep
    mask = np.ones(n_ep, bool)
    reasons = {"probe_window": 0, "amplitude": 0}
    for _, p in probes.iterrows():
        end = p["resp_end_s"] + RD.probe_pad_after_s if pd.notna(p["resp_end_s"]) else p["onset_s"] + RD.probe_no_response_s
        hit = (t_end > p["onset_s"]) & (t_end - ep < end)
        reasons["probe_window"] += int((mask & hit).sum())
        mask &= ~hit
    if n_ep:
        ptp = data.max(axis=2) - data.min(axis=2)
        amp_bad = np.any(ptp > RD.reject_uv * 1e-6, axis=1)
        reasons["amplitude"] = int((mask & amp_bad).sum())
        mask &= ~amp_bad
    return t_end, data, mask, reasons


def _log_band(f, P, lo, hi) -> float:
    sel = (f >= lo) & (f < hi)
    return float(np.log(np.mean(np.trapezoid(P[:, sel], f[sel], axis=1)) + 1e-30))


def epoch_features(X: np.ndarray, ch_names: list, sfreq: float, common_idx: list, seed: int) -> dict:
    """All primary and alternative features of one clean epoch (n_ch, n_times)."""
    names = {c.upper(): i for i, c in enumerate(ch_names)}
    post = [names[c.upper()] for c in RD.posterior_channels if c.upper() in names]
    front = [names[c.upper()] for c in RD.frontal_channels if c.upper() in names]
    f, P = psd(X, sfreq, nperseg_s=2.0)
    out = {"alpha_post": _log_band(f, P[post], *RD.alpha_band), "theta_front": _log_band(f, P[front], *RD.theta_band)}
    ap = aperiodic_fit(f, P[post], 1.0, 40.0)
    # periodic alpha: log alpha power minus the aperiodic prediction at 10 Hz (specparam-like, robust backend)
    ap_at10 = float(np.mean(ap["offset"] - ap["exponent"] * np.log10(10.0)))
    out["alpha_specparam"] = out["alpha_post"] - ap_at10 * np.log(10)
    ap_all = aperiodic_fit(f, P[common_idx], 1.0, 40.0)
    out["exponent_neg"] = -float(ap_all["exponent_mean"])
    sc = spectrally_controlled(lambda Y: lzc(Y, norm="entropy"), X[common_idx], n_surrogates=RD.n_surrogates,
                               multivariate=True, rng=np.random.default_rng(seed))
    out["lzc"], out["lzc_si"] = sc["raw"], sc["z"]
    return out


def _feat_job(args):
    X, ch_names, sf, common_idx, seed = args
    return epoch_features(X, ch_names, sf, common_idx, seed)


def session_features(raw, duration_s: float, probes: pd.DataFrame, seed: int, n_jobs: int | None = None) -> dict:
    t_end, data, mask, reasons = epoch_grid(raw, duration_s, probes)
    cfg = load_config()
    mapping = map_common_channels(raw, cfg["channels_common"])
    common_idx = [raw.ch_names.index(mapping[c]) for c in cfg["channels_common"]]
    jobs = [(data[k], raw.ch_names, raw.info["sfreq"], common_idx, seed + 7919 * k) for k in range(len(t_end)) if mask[k]]
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    if n_jobs == 1 or len(jobs) < 4:
        rows = [_feat_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            rows = list(ex.map(_feat_job, jobs, chunksize=4))
    feats = {k: np.full(len(t_end), np.nan) for k in (rows[0].keys() if rows else ["alpha_post", "lzc_si"])}
    for (k_ep, r) in zip(np.flatnonzero(mask), rows):
        for name, v in r.items():
            feats[name][k_ep] = v
    return {"t_end_s": t_end, "mask": mask, "reasons": reasons, "features": feats}


def process_session(sess: dict, probes: pd.DataFrame, out_dir: str = FEATURES_DIR, seed: int = CFG.SEED,
                    n_jobs: int | None = None, do_ica: bool = True, raw=None) -> str:
    """Full pipeline for one session; returns the npz path."""
    os.makedirs(out_dir, exist_ok=True)
    log = {"sub": sess["sub"], "ses": sess["ses"]}
    raw = raw if raw is not None else load_raw(sess["bdf_local"])
    raw = preprocess_raw(raw, log, do_ica=do_ica)
    pr = probes[(probes["sub"] == sess["sub"]) & (probes["ses"] == sess["ses"])].sort_values("onset_s")
    sf = session_features(raw, sess["duration_s"], pr, seed, n_jobs)
    res = RD.time_resolution_s
    t_min = np.round(sf["t_end_s"] / res) * res / 60.0
    t_probe = np.round(pr["onset_s"].to_numpy(float) / res) * res / 60.0
    path = os.path.join(out_dir, f"{sess['sub']}_{sess['ses']}.npz")
    np.savez_compressed(path, t=t_min, mask=sf["mask"], t_probe=t_probe,
                        rating=pr["rating"].to_numpy(float), q1=pr["q1"].to_numpy(float), q2=pr["q2"].to_numpy(float),
                        q3=pr["q3"].to_numpy(float), sub=sess["sub"], ses=sess["ses"], group=sess["group"],
                        **{k: v for k, v in sf["features"].items()})
    log.update({"n_epochs": int(len(sf["mask"])), "n_missing": int((~sf["mask"]).sum()),
                "missing_frac": float((~sf["mask"]).mean()), **sf["reasons"], "n_probes": int(len(pr)),
                "n_rated": int(pr["rating"].notna().sum())})
    with open(path.replace(".npz", "_prep_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, default=str)
    return path


def run(subjects=None, seed: int = CFG.SEED, n_jobs: int | None = None, do_ica: bool = True, verbose: bool = True) -> list[str]:
    from eeg import gates
    gates.require("E1")
    probes = DS.load_probes()
    done = []
    for s in DS.sessions():
        if subjects and s["sub"] not in subjects and s["sub"].replace("sub-", "") not in subjects:
            continue
        if not os.path.isfile(s["bdf_local"]):
            if verbose:
                print(f"  {s['sub']} {s['ses']}: BDF chua tai, bo qua")
            continue
        out = os.path.join(FEATURES_DIR, f"{s['sub']}_{s['ses']}.npz")
        if os.path.isfile(out):
            done.append(out)
            continue
        if verbose:
            print(f"  {s['sub']} {s['ses']} ...", flush=True)
        done.append(process_session(s, probes, seed=seed, n_jobs=n_jobs, do_ica=do_ica))
    return done
