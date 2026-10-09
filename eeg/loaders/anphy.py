"""
ANPHY-Sleep loader (OSF r26fh): 29 overnight high-density (83-ch) PSGs.

Each subject is a separate ~3 GB zip. `prepare_subject` downloads one zip,
reads only the scored 30-s windows it needs (lazy EDF access, never the whole
night at 1000 Hz), writes the preprocessed 19-channel epochs to
data/derivatives/anphy/<subject>_<stage>.npz and deletes the raw files, so
peak disk use stays around one subject.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import urllib.request
import zipfile

import numpy as np

from ..preprocess import epochs_from_mne_epochs, load_config, save_epochset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "data", "raw", "anphy")
DERIV = os.path.join(ROOT, "data", "derivatives", "anphy")
OSF_API = "https://api.osf.io/v2/nodes/r26fh/files/osfstorage/"

STAGES = ("W", "N2", "N3", "R")
STAGE_ALIASES = {"W": "W", "WAKE": "W", "N2": "N2", "N3": "N3", "R": "R", "REM": "R", "N1": "N1"}


def list_subjects() -> list[dict]:
    """[{name, size, url}] for every per-subject zip on OSF (paginated API)."""
    out, url = [], OSF_API
    while url:
        with urllib.request.urlopen(url, timeout=60) as r:
            d = json.load(r)
        for it in d["data"]:
            a = it["attributes"]
            if a["kind"] == "file" and re.match(r"EPCTL\d+\.zip$", a["name"]):
                out.append({"name": a["name"][:-4], "size": a["size"], "url": it["links"]["download"]})
        url = d["links"].get("next")
    return sorted(out, key=lambda x: x["name"])


def _download(url: str, dest: str) -> None:
    import subprocess
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    subprocess.run(["curl", "-L", "-C", "-", "--retry", "5", "--retry-delay", "10", "-s", "-o", dest, url],
                   check=True)


def parse_stages(path: str) -> list[tuple[str, float, float]]:
    """Stage annotation txt -> [(stage, onset_s, duration_s)]. Tolerant to
    tab/comma/space separators and a header line."""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = [p for p in re.split(r"[\t,;]+|\s{2,}|\s", line.strip()) if p]
            if len(parts) < 3:
                continue
            label = parts[0].upper()
            if label not in STAGE_ALIASES:
                continue
            try:
                onset, dur = float(parts[1]), float(parts[2])
            except ValueError:
                continue
            rows.append((STAGE_ALIASES[label], onset, dur))
    return rows


def _find(folder: str, patterns) -> str | None:
    for pat in patterns:
        hits = glob.glob(os.path.join(folder, "**", pat), recursive=True)
        if hits:
            return hits[0]
    return None


def stage_epochs(edf_path: str, stages_path: str, stage: str, cfg: dict, max_windows: int = 10,
                 seed: int = 42):
    """Up to `max_windows` scored 30-s windows of `stage`, each read lazily with
    2-s padding, filtered/resampled, then split into 10-s epochs."""
    import mne
    pp = cfg["preprocess"]
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="error")
    eeg = [c for c in raw.ch_names if not re.search(r"EOG|EMG|ECG|EKG|CHIN|LEG", c, re.I)]
    rows = [r for r in parse_stages(stages_path) if r[0] == stage and r[2] >= 30]
    rng = np.random.default_rng(seed)
    if len(rows) > max_windows:
        rows = [rows[i] for i in sorted(rng.choice(len(rows), max_windows, replace=False))]
    pad, chunks = 2.0, []
    for _, onset, _ in rows:
        t0, t1 = onset - pad, onset + 30.0 + pad
        if t0 < 0 or t1 > raw.times[-1]:
            continue
        seg = raw.copy().pick(eeg).crop(t0, t1).load_data(verbose="error")
        seg.filter(pp["l_freq"], pp["h_freq"], fir_design="firwin", verbose="error")
        seg.resample(pp["sfreq"], verbose="error")
        x = seg.get_data()
        i0 = int(pad * pp["sfreq"])
        n = int(pp["epoch_s"] * pp["sfreq"])
        chunks += [x[:, i0 + k * n: i0 + (k + 1) * n] for k in range(int(30 // pp["epoch_s"]))]
    if not chunks:
        return None
    info = mne.create_info(eeg, pp["sfreq"], ch_types="eeg")
    ep = mne.EpochsArray(np.stack(chunks), info, verbose="error")
    from ..preprocess import _standard_1005
    ep.set_montage(_standard_1005(), match_case=False, on_missing="ignore")
    with ep.info._unlock():
        ep.info["highpass"], ep.info["lowpass"] = pp["l_freq"], pp["h_freq"]
    return ep


def prepare_subject(subject: dict, cfg: dict | None = None, keep_raw: bool = False) -> dict:
    """Download, extract, preprocess the four stages, save .npz, delete raw."""
    cfg = cfg or load_config()
    os.makedirs(DERIV, exist_ok=True)
    done = {s: os.path.join(DERIV, f"{subject['name']}_{s}.npz") for s in STAGES}
    if all(os.path.isfile(p) for p in done.values()):
        return {"subject": subject["name"], "status": "cached"}
    zpath = os.path.join(RAW, subject["name"] + ".zip")
    folder = os.path.join(RAW, subject["name"])
    if not os.path.isdir(folder):
        _download(subject["url"], zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(folder)
        os.remove(zpath)
    edf = _find(folder, ["*.edf", "*.EDF"])
    txt = _find(folder, ["*[Ss]tag*.txt", "*[Hh]ypno*.txt", "*[Ss]cor*.txt", "*.txt"])
    if edf is None or txt is None:
        raise FileNotFoundError(f"{subject['name']}: EDF or stage file not found in {folder}")
    report = {"subject": subject["name"], "edf": os.path.basename(edf), "stages_file": os.path.basename(txt)}
    for stage in STAGES:
        ep = stage_epochs(edf, txt, stage, cfg, max_windows=cfg["datasets"]["anphy"]["windows_per_stage"],
                          seed=cfg["seed"])
        if ep is None:
            report[stage] = 0
            continue
        es = epochs_from_mne_epochs(ep, cfg)
        save_epochset(done[stage], es)
        report[stage] = es.n_epochs
    if not keep_raw:
        shutil.rmtree(folder, ignore_errors=True)
    return report
