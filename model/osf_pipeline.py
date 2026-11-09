"""
Second real dataset (Q1_pipeline.md v2.4): OSF 3uszv fitted with the same code
as ds001787 (model.realdata), with its own feature directory, results
directory, gates F0-F3, preregistration lock and rating range.

    python main.py osf3uszv ratings | design-crb | preregister | preprocess | fit | groups | boot | check
"""
from __future__ import annotations

import json
import os

from eeg import gates
from eeg.loaders import osf3uszv as DS
from model import config as CFG
from model import realdata as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, CFG.OSF.results_dir)
FEATURES = os.path.join(ROOT, CFG.OSF.data_dir, "derivatives", "features")
GATES = {"E0": "F0", "E1": "F1", "E2": "F2", "E3": "F3"}


def mde_log() -> float:
    """MDE of the expert - novice log k_inh difference from this dataset's F1
    (predictions_osf3uszv.md), NaN before design-crb has run."""
    path = os.path.join(RESULTS, "e1_design_crb.json")
    if not os.path.isfile(path):
        return float("nan")
    with open(path, encoding="utf-8") as f:
        return float(json.load(f).get("group_difference_log_k_inh", {}).get("mde_80pct_alpha05", float("nan")))


def _common(output_dir: str | None) -> dict:
    return {"feature_dir": FEATURES, "output_dir": output_dir or RESULTS, "artifact_rule": CFG.OSF.artifact_rule}


def design_crb(output_dir: str | None = None, verbose: bool = True) -> dict:
    return R.design_crb(output_dir=output_dir or RESULTS, verbose=verbose, designs_info=DS.all_designs(),
                        gate=("F0", "F1"), lock=gates.LOCK_OSF)


def run_fit(output_dir: str | None = None, **kw) -> dict:
    return R.run_fit(**_common(output_dir), gate="F1", mde_log=mde_log(), **kw)


def run_groups(output_dir: str | None = None, **kw) -> dict:
    return R.run_groups(**_common(output_dir), gate="F1", mde_log=mde_log(), **kw)


def run_boot(output_dir: str | None = None, **kw) -> dict:
    return R.run_boot(**_common(output_dir), gate="F1", gate_record="F2", **kw)


def after_first_probe(s: dict, lead_min: float = 1.0) -> dict:
    """Post-registration variant: the EEG before (first probe - lead) is marked
    missing. The sessions begin with instructions and a practice probe (first
    bell after 6.7 min on average); this asks whether that segment drives the
    slow alpha time constant."""
    import numpy as np
    t_probe = np.asarray(s["t_probe"], float)
    if len(t_probe) == 0:
        return s
    y = np.asarray(s["y"], float).copy()
    y[np.asarray(s["t"], float) < t_probe.min() - lead_min] = np.nan
    return {**s, "y": y}


def sensitivity_loaders() -> dict:
    """The locked variants of model.realdata plus the post-registration one
    (reported separately, not part of the 80 % rule)."""
    out = R.default_sensitivity_loaders(FEATURES, CFG.OSF.artifact_rule)
    out["postreg_after_first_probe"] = lambda: [
        after_first_probe(s) for s in R.load_feature_sessions(FEATURES, artifact_rule=CFG.OSF.artifact_rule)[0]]
    return out


def run_check(output_dir: str | None = None, **kw) -> dict:
    kw.setdefault("sensitivity_loaders", sensitivity_loaders())
    return R.run_check(**_common(output_dir), gate="F2", gate_record="F3", **kw)


def run(step: str, **kw) -> dict:
    return {"design-crb": design_crb, "fit": run_fit, "groups": run_groups, "boot": run_boot, "check": run_check}[step](**kw)
