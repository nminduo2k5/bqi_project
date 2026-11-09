"""
OSF 3uszv: EEG of the task recordings -> the same per-epoch feature files as
ds001787 (eeg.realdata_features), in data/osf3uszv/derivatives/features/.

Differences from ds001787, all fixed before the lock (predictions_osf3uszv.md):
  - 21 channels (19 of the 10-20 system + A1, A2) read from the text archive,
    every 4th row (512 Hz), montage standard_1020, resampled to 256 Hz so that
    the LZc feature is computed at the sampling rate of ds001787; the features
    use the channels present: alpha_post on P3, Pz, P4, O1, O2; theta_front on
    Fz, Cz; LZc on the 19 common channels (all present).
  - primary artifact rule = session-adaptive threshold on the feature channels
    (Q1_pipeline.md v2.2); the fixed 150 uV rule is a sensitivity variant.
Filters, bad channels, ICA + ICLabel, the 10-s grid and the features are
eeg.realdata_features.preprocess_raw / session_features, unchanged.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eeg import realdata_features as RF
from eeg.loaders import osf3uszv as DS
from model import config as CFG

FEATURES_DIR = os.path.join(DS.data_dir(), "derivatives", "features")


def repeat_phase(block: np.ndarray, repeat: int = DS.REPEAT) -> int:
    """Row index (mod `repeat`) at which each group of identical rows starts."""
    same = np.all(block[1:] == block[:-1], axis=1)
    starts = np.flatnonzero(~same) + 1
    if len(starts) == 0:
        return 0
    return int(np.bincount(starts % repeat, minlength=repeat).argmax())


def read_eeg(sess: dict, verbose: bool = False) -> tuple[np.ndarray, dict]:
    """EEG of one recording as (21, n_samples) float32 in microvolts at 512 Hz:
    the 21 EEG columns, every REPEAT-th row from the detected phase, trailing
    NaN rows dropped. Chunked, so the 1.7 GB text never sits in memory at once."""
    f, _ = DS.open_text(sess["file"])
    cols = list(DS.FILE_COLUMNS[:21])
    parts, phase, n_rows, t0 = [], None, 0, time.time()
    with f:
        for chunk in pd.read_csv(f, usecols=cols, dtype=np.float32, chunksize=DS.ROWS_PER_S * 300,
                                 encoding="latin-1"):
            X = chunk[cols].to_numpy()                # chunk rows are consecutive; sizes are multiples of REPEAT
            if phase is None:
                phase = repeat_phase(X[:DS.ROWS_PER_S * 10])
            parts.append(X[phase::DS.REPEAT])
            n_rows += len(X)
            if verbose and time.time() - t0 > 20:
                t0 = time.time()
                print(f"    doc {n_rows / DS.ROWS_PER_S / 60:.0f} phut...", flush=True)
    X = np.concatenate(parts).T
    bad_tail = np.flatnonzero(~np.all(np.isfinite(X), axis=0))
    n_nan = int(len(bad_tail))
    if n_nan:
        X = X[:, :bad_tail.min()] if bad_tail.max() == X.shape[1] - 1 else X
    return np.ascontiguousarray(X), {"n_rows": n_rows, "phase": phase, "n_nan_samples": n_nan,
                                     "duration_s": X.shape[1] / DS.EEG_SFREQ}


def load_raw(sess: dict, verbose: bool = False):
    """MNE Raw of one recording: 21 EEG channels in volts, standard_1020
    positions, resampled to the target rate (256 Hz)."""
    import mne
    X, info = read_eeg(sess, verbose)
    names = list(DS.EEG_CHANNELS)
    raw = mne.io.RawArray(X.astype(np.float64) * 1e-6, mne.create_info(names, DS.EEG_SFREQ, "eeg"), verbose="error")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")               # montage name deprecation in MNE 1.13
        raw.set_montage("standard_1020", on_missing="raise", verbose="error")
    target = CFG.OSF.target_sfreq
    if target and target < DS.EEG_SFREQ:
        raw.resample(target, verbose="error")
    raw.info["description"] = f"osf3uszv {sess['stem']}: phase {info['phase']}, {info['n_nan_samples']} NaN samples dropped"
    return raw, info


def process_session(sess: dict, probes: pd.DataFrame, out_dir: str = FEATURES_DIR, seed: int = CFG.SEED,
                    n_jobs: int | None = None, do_ica: bool = True, verbose: bool = False) -> str:
    raw, info = load_raw(sess, verbose)
    sess = {**sess, "duration_s": float(info["duration_s"])}
    return RF.process_session(sess, probes, out_dir=out_dir, seed=seed, n_jobs=n_jobs, do_ica=do_ica, raw=raw)


def run(subjects=None, seed: int = CFG.SEED, n_jobs: int | None = None, do_ica: bool = True, verbose: bool = True) -> list[str]:
    from eeg import gates
    gates.require("F1")
    gates.require_preregistration(gates.PREDICTIONS_OSF, gates.LOCK_OSF)
    probes = DS.load_probes()
    sess_list = [s for s in DS.sessions("task") if not subjects or s["sub"] in subjects or s["stem"] in subjects]
    done = []
    for i, s in enumerate(sess_list, 1):
        tag = f"[{i}/{len(sess_list)}] {s['stem']}"
        out = os.path.join(FEATURES_DIR, f"{s['sub']}_{s['ses']}.npz")
        if os.path.isfile(out):
            stale = RF._stale_reason(out, do_ica)
            if stale is None:
                done.append(out)
                continue
            if verbose:
                print(f"  {tag}: {stale}, tinh lai", flush=True)
        if verbose:
            print(f"  {tag} ... ({time.strftime('%H:%M')})", flush=True)
        t0 = time.time()
        done.append(process_session(s, probes, seed=seed, n_jobs=n_jobs, do_ica=do_ica, verbose=verbose))
        if verbose:
            print(f"  {tag}: xong, {(time.time() - t0) / 60:.1f} phut", flush=True)
    return done
