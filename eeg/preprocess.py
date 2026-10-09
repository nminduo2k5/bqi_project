"""
Unified preprocessing (Q1_pipeline.md Sec 4): one function for every dataset,
parameters from eeg/config.yaml.

    epochs_from_raw(raw, cfg)              -> EpochSet for continuous data
    epochs_from_mne_epochs(epochs, cfg)    -> EpochSet for already-epoched data (Chennu)

An EpochSet holds the 19 common 10-20 channels (mapped by position from any
montage, e.g. EGI HydroCel) as float arrays (n_epochs, n_channels, n_times)
in volts, plus the CSD-transformed copy used for synchrony/network metrics.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import yaml

CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

# Standard 10-20 equivalents on the EGI GSN-HydroCel-129 net (Luu & Ferree 2005;
# EGI technical note). Used before any geometric matching.
EGI129_1020 = {"Fp1": "E22", "Fp2": "E9", "F7": "E33", "F3": "E24", "Fz": "E11", "F4": "E124", "F8": "E122",
               "T7": "E45", "C3": "E36", "Cz": "Cz", "C4": "E104", "T8": "E108", "P7": "E58", "P3": "E52",
               "Pz": "E62", "P4": "E92", "P8": "E96", "O1": "E70", "O2": "E83"}


def _standard_1005():
    import mne
    for name in ("colin27_1005", "standard_1005"):
        try:
            return mne.channels.make_standard_montage(name)
        except ValueError:
            continue
    raise ValueError("no 10-05 template montage available")


def load_config(path: str = CFG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class EpochSet:
    data: np.ndarray                 # (n_epochs, n_channels, n_times), volts
    sfreq: float
    ch_names: list
    csd: np.ndarray | None = None    # same shape, surface Laplacian
    source_channels: dict = field(default_factory=dict)   # common name -> original channel
    log: dict = field(default_factory=dict)

    @property
    def n_epochs(self) -> int:
        return int(self.data.shape[0])


# =============================================================================
# Channel mapping
# =============================================================================
def _positions(inst) -> dict:
    """Channel name -> 3-D position (head frame) from an MNE object's montage."""
    pos = inst.get_montage().get_positions()["ch_pos"] if inst.get_montage() else {}
    return {k: np.asarray(v) for k, v in pos.items() if v is not None and np.all(np.isfinite(v))}


def map_common_channels(inst, targets: list[str]) -> dict:
    """Map each 10-20 target to a channel of `inst`: exact (case-insensitive,
    with T3/T4/T5/T6 aliases) if present, otherwise the nearest electrode by
    position after projecting both montages onto the unit sphere."""
    import mne
    names = {n.upper(): n for n in inst.ch_names}
    for old, new in ALIASES.items():
        if old.upper() in names and new.upper() not in names:
            names[new.upper()] = names[old.upper()]
    mapping, need = {}, []
    is_egi = sum(bool(re.fullmatch(r"E\d+", n)) for n in inst.ch_names) > 0.5 * len(inst.ch_names)
    for t in targets:
        if t.upper() in names:
            mapping[t] = names[t.upper()]
        elif is_egi and EGI129_1020.get(t) in inst.ch_names:
            mapping[t] = EGI129_1020[t]
        else:
            need.append(t)
    if need:
        src = _positions(inst)
        if not src:
            raise ValueError("Missing channels and no montage to map by position: " + ", ".join(need))
        ref = _standard_1005().get_positions()["ch_pos"]
        unit = lambda v: (v - centre) / np.linalg.norm(v - centre)
        centre = np.mean(np.array(list(src.values())), axis=0)
        src_names = [n for n in src if n not in mapping.values()]
        src_xyz = np.array([unit(src[n]) for n in src_names])
        ref_centre = np.mean(np.array([ref[t] for t in targets]), axis=0)
        for t in need:
            v = ref[t] - ref_centre
            v = v / np.linalg.norm(v)
            j = int(np.argmax(src_xyz @ v))
            mapping[t] = src_names[j]
    return mapping


# =============================================================================
# Core steps
# =============================================================================
def _reject(data: np.ndarray, reject_uv: float) -> np.ndarray:
    ptp = data.max(axis=2) - data.min(axis=2)
    return np.all(ptp < reject_uv * 1e-6, axis=1)


def _finish(epochs, cfg: dict, mapping: dict, log: dict) -> EpochSet:
    """Average reference over the full montage, keep the common channels,
    compute CSD, reject by amplitude."""
    import mne
    pp = cfg["preprocess"]
    targets = list(mapping)
    epochs = epochs.copy().set_eeg_reference("average", projection=False, verbose="error")
    csd_data = None
    if pp.get("csd_for_sync", True) and epochs.get_montage() is not None:
        try:
            csd = mne.preprocessing.compute_current_source_density(epochs, verbose="error")
            csd_data = csd.get_data(picks=[mapping[t] for t in targets])
        except Exception as e:  # montage without enough positions etc.
            log["csd_error"] = repr(e)
    data = epochs.get_data(picks=[mapping[t] for t in targets])
    keep = _reject(data, pp["reject_uv"])
    log.update({"n_epochs_in": int(len(keep)), "n_epochs_kept": int(keep.sum()),
                "rejected_amplitude": int((~keep).sum())})
    return EpochSet(data=data[keep], sfreq=float(epochs.info["sfreq"]), ch_names=targets,
                    csd=None if csd_data is None else csd_data[keep], source_channels=mapping, log=log)


def _filter_resample(inst, pp: dict):
    inst = inst.copy()
    if inst.info["lowpass"] > pp["h_freq"] + 1 or inst.info["highpass"] < pp["l_freq"] - 0.01:
        inst.filter(pp["l_freq"], pp["h_freq"], fir_design="firwin", verbose="error")
    if abs(inst.info["sfreq"] - pp["sfreq"]) > 1e-6:
        inst.resample(pp["sfreq"], verbose="error")
    return inst


def mark_bad_channels(inst, z: float) -> list[str]:
    """Flag channels whose log-variance is an outlier (robust z-score)."""
    X = inst.get_data(picks="eeg")
    if X.ndim == 3:
        X = X.transpose(1, 0, 2).reshape(X.shape[1], -1)
    lv = np.log(X.var(axis=1) + 1e-30)
    med = np.median(lv)
    mad = np.median(np.abs(lv - med)) * 1.4826 + 1e-12
    names = [inst.ch_names[i] for i in np.flatnonzero(np.abs(lv - med) / mad > z)]
    return names


def epochs_from_mne_epochs(epochs, cfg: dict | None = None) -> EpochSet:
    """Already epoched/cleaned data (Chennu): filter, resample, bad-channel
    interpolation, average reference, common channels, CSD, rejection."""
    cfg = cfg or load_config()
    pp = cfg["preprocess"]
    epochs = _filter_resample(epochs, pp)
    bads = mark_bad_channels(epochs, pp["bad_channel_z"])
    log = {"bad_channels": bads, "n_channels": len(epochs.ch_names)}
    if len(bads) > pp["max_bad_channel_frac"] * len(epochs.ch_names):
        log["excluded"] = "too many bad channels"
    if bads and epochs.get_montage() is not None:
        epochs.info["bads"] = bads
        epochs = epochs.interpolate_bads(reset_bads=True, verbose="error")
    mapping = map_common_channels(epochs, cfg["channels_common"])
    return _finish(epochs, cfg, mapping, log)


def epochs_from_raw(raw, cfg: dict | None = None, start_s: float = 0.0, stop_s: float | None = None) -> EpochSet:
    """Continuous recording segment -> fixed-length epochs."""
    import mne
    cfg = cfg or load_config()
    pp = cfg["preprocess"]
    raw = raw.copy().crop(start_s, stop_s).pick("eeg")
    raw = _filter_resample(raw, pp)
    bads = mark_bad_channels(raw, pp["bad_channel_z"])
    log = {"bad_channels": bads, "n_channels": len(raw.ch_names)}
    if len(bads) > pp["max_bad_channel_frac"] * len(raw.ch_names):
        log["excluded"] = "too many bad channels"
    if bads and raw.get_montage() is not None:
        raw.info["bads"] = bads
        raw = raw.interpolate_bads(reset_bads=True, verbose="error")
    epochs = mne.make_fixed_length_epochs(raw, duration=pp["epoch_s"], preload=True, verbose="error")
    mapping = map_common_channels(epochs, cfg["channels_common"])
    return _finish(epochs, cfg, mapping, log)


def equalize(a: EpochSet, b: EpochSet, seed: int) -> tuple[EpochSet, EpochSet]:
    """Equalise epoch counts between two conditions of one subject by random
    subsampling (preregistration Sec 2)."""
    rng = np.random.default_rng(seed)
    n = min(a.n_epochs, b.n_epochs)

    def take(e: EpochSet) -> EpochSet:
        idx = np.sort(rng.choice(e.n_epochs, n, replace=False))
        return EpochSet(e.data[idx], e.sfreq, e.ch_names, None if e.csd is None else e.csd[idx],
                        e.source_channels, {**e.log, "equalized_to": int(n)})
    return take(a), take(b)


# =============================================================================
# Storage
# =============================================================================
def save_epochset(path: str, es: EpochSet) -> None:
    np.savez_compressed(path, data=es.data.astype(np.float32),
                        csd=np.zeros(0) if es.csd is None else es.csd.astype(np.float32),
                        sfreq=es.sfreq, ch_names=np.array(es.ch_names),
                        log=json.dumps({**es.log, "source_channels": es.source_channels}, default=str))


def load_epochset(path: str) -> EpochSet:
    z = np.load(path, allow_pickle=False)
    csd = z["csd"] if z["csd"].size else None
    return EpochSet(data=z["data"].astype(float), sfreq=float(z["sfreq"]), ch_names=list(z["ch_names"]),
                    csd=None if csd is None else csd.astype(float), log=json.loads(str(z["log"])))
