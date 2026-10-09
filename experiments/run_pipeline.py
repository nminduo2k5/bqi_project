"""
Q1 pipeline orchestrator (Q1_pipeline.md Sec 6, 9, 11). Every step is
resumable (finished units are cached on disk) and gated.

    python main.py eeg chennu        # preprocess + features, propofol reference
    python main.py eeg anphy         # per-subject download -> stage epochs -> features (long)
    python main.py eeg g1            # gate G1 on the reference contrasts
    python main.py eeg axis          # gate G2 + conscious-level axis
    python main.py eeg status        # what is done

Features land in data/derivatives/features/<dataset>_epochs.csv and
<dataset>_subjects.csv; gate results in outputs/gates/.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eeg import gates
from eeg.preprocess import load_config

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FEAT = os.path.join(ROOT, "data", "derivatives", "features")
MANIFEST = os.path.join(ROOT, "data", "manifest.json")


# =============================================================================
# helpers
# =============================================================================
def _manifest() -> dict:
    try:
        with open(MANIFEST, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _update_manifest(dataset: str, key: str, value) -> None:
    m = _manifest()
    m.setdefault(dataset, {})[key] = value
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False, default=str)


def _append_features(dataset: str, df: pd.DataFrame) -> None:
    os.makedirs(FEAT, exist_ok=True)
    path = os.path.join(FEAT, f"{dataset}_epochs.csv")
    df.to_csv(path, mode="a", header=not os.path.isfile(path), index=False)


def _done_units(dataset: str) -> set:
    path = os.path.join(FEAT, f"{dataset}_epochs.csv")
    if not os.path.isfile(path):
        return set()
    d = pd.read_csv(path, usecols=["subject", "condition"])
    return set(map(tuple, d.drop_duplicates().values.tolist()))


def load_features(dataset: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(FEAT, f"{dataset}_epochs.csv"))


def subject_table(dataset: str, primary_only: bool = False, cfg: dict | None = None) -> pd.DataFrame:
    """Per-subject x condition means. With primary_only, ANPHY is restricted to
    the preregistered first N subjects in ascending ID order."""
    from eeg.features import subject_means
    df = subject_means(load_features(dataset))
    df.to_csv(os.path.join(FEAT, f"{dataset}_subjects.csv"), index=False)
    if primary_only and dataset == "anphy":
        n = (cfg or load_config())["datasets"]["anphy"]["primary_n"]
        keep = sorted(df["subject"].unique())[:n]
        df = df[df["subject"].isin(keep)]
    return df


def _features_for(dataset: str, subject: str, condition: str, es, cfg: dict, extra: dict | None = None,
                  secondary: bool = True) -> int:
    from eeg.features import extract
    if es.n_epochs == 0:
        return 0
    meta = {"dataset": dataset, "subject": subject, "condition": condition, **(extra or {})}
    t0 = time.time()
    df = extract(es, meta, cfg, secondary=secondary)
    _append_features(dataset, df)
    print(f"  {subject:>10s} {condition:<10s} {es.n_epochs:3d} epochs  {time.time() - t0:6.1f}s", flush=True)
    return len(df)


# =============================================================================
# steps
# =============================================================================
def step_chennu(cfg: dict, force_gate: bool = False) -> None:
    from eeg.loaders import chennu
    gates.require("G0", force=force_gate)
    done = _done_units("chennu")
    items = chennu.list_recordings()
    _update_manifest("chennu", "recordings", len(items))
    for it in items:
        if (it["subject"], it["condition"]) in done:
            continue
        es = chennu.load_recording(it, cfg)
        _features_for("chennu", it["subject"], it["condition"], es, cfg,
                      extra={"propofol_ugml": it.get("concentration", np.nan)})
    subject_table("chennu")


def step_anphy(cfg: dict, force_gate: bool = False, max_subjects: int | None = None) -> None:
    from eeg.loaders import anphy
    from eeg.preprocess import load_epochset
    gates.require("G0", force=force_gate)
    subs = anphy.list_subjects()
    _update_manifest("anphy", "osf_subjects", [s["name"] for s in subs])
    _update_manifest("anphy", "osf_total_gb", round(sum(s["size"] for s in subs) / 1e9, 1))
    done = _done_units("anphy")
    n = max_subjects or cfg["datasets"]["anphy"]["primary_n"]
    for s in subs[:n]:
        if all((s["name"], st) in done for st in anphy.STAGES):
            continue
        rep = anphy.prepare_subject(s, cfg)
        _update_manifest("anphy", s["name"], rep)
        for st in anphy.STAGES:
            p = os.path.join(anphy.DERIV, f"{s['name']}_{st}.npz")
            if (s["name"], st) not in done and os.path.isfile(p):
                _features_for("anphy", s["name"], st, load_epochset(p), cfg)
    subject_table("anphy")


def _paired(sub: pd.DataFrame, metric: str, a: str, b: str):
    from eeg.stats import paired_dz
    w = sub.pivot(index="subject", columns="condition", values=metric)[[a, b]].dropna()
    return paired_dz(w[a].values, w[b].values, rng=0), len(w)


def step_g1(cfg: dict, force_gate: bool = False) -> dict:
    """LZc must reproduce wake > N3 (ANPHY) and baseline > moderate (Chennu)."""
    gates.require("G0", force=force_gate)
    details, passed = {}, True
    for ds, a, b in (("chennu", "baseline", "moderate"), ("anphy", "W", "N3")):
        path = os.path.join(FEAT, f"{ds}_epochs.csv")
        if not os.path.isfile(path):
            details[f"{ds}: {a} vs {b}"] = {"status": "pending", "note": "chưa có dữ liệu (hoãn)"}
            passed = False
            print(f"  {ds}: {a} vs {b} -> HOÃN (chưa có dữ liệu)")
            continue
        sub = subject_table(ds, primary_only=True, cfg=cfg)
        res = {}
        for metric in ("lzc", "lzc_si", "exponent"):
            r, n = _paired(sub, metric, a, b)
            res[metric] = {"n": n, "dz": r["dz"], "ci": r["ci"], "mean_diff": r["mean_diff"]}
        ok = res["lzc"]["ci"][0] > 0
        res["passed"] = bool(ok)
        details[f"{ds}: {a} vs {b}"] = res
        passed &= ok
        print(f"  {ds}: LZc {a} - {b}: d_z = {res['lzc']['dz']:.2f} CI {np.round(res['lzc']['ci'], 2)} "
              f"-> {'PASS' if ok else 'FAIL'}  (LZc_si d_z = {res['lzc_si']['dz']:.2f}, "
              f"exponent d_z = {res['exponent']['dz']:.2f})")
    pending = [k for k, v in details.items() if isinstance(v, dict) and v.get("status") == "pending"]
    path = gates.record("G1", passed, {**details, "pending": pending})
    label = "PASS" if passed else ("CHƯA ĐỦ (còn: " + ", ".join(pending) + ")" if pending else "FAIL")
    print(f"G1 {label} -> {path}")
    return details


def reference_deltas(cfg: dict) -> dict:
    """Per-subject difference vectors over the preregistered axis measures,
    z-scored within each dataset."""
    measures = cfg["axis"]["measures"]
    out = {}
    for ds, a, b in (("chennu", "baseline", "moderate"), ("anphy", "W", "N3")):
        sub = subject_table(ds, primary_only=True, cfg=cfg)
        z = sub.copy()
        for m in measures:
            z[m] = (sub[m] - sub[m].mean()) / sub[m].std(ddof=1)
        wa = z[z.condition == a].set_index("subject")[measures]
        wb = z[z.condition == b].set_index("subject")[measures]
        common = wa.index.intersection(wb.index)
        out[ds] = (wa.loc[common] - wb.loc[common]).dropna()
    return out


def step_axis(cfg: dict, force_gate: bool = False) -> dict:
    from eeg.axis import build_axis, cosine
    gates.require("G1", force=force_gate)
    D = reference_deltas(cfg)
    ax = build_axis(D["chennu"].values, D["anphy"].values)
    ok = ax["agreement"] > cfg["axis"]["min_agreement"]
    details = {"measures": cfg["axis"]["measures"], "agreement_cosine": ax["agreement"],
               "u": dict(zip(cfg["axis"]["measures"], np.round(ax["u"], 4).tolist())),
               "explained": ax["explained"], "ref_shift": ax["ref_shift"],
               "n": {k: len(v) for k, v in D.items()},
               "mean_delta": {k: dict(zip(cfg["axis"]["measures"], np.round(v.mean().values, 3).tolist()))
                              for k, v in D.items()}}
    path = gates.record("G2", ok, details)
    print(f"  cosine(chennu, anphy) = {ax['agreement']:.3f}; axis = {details['u']}")
    print(f"G2 {'PASS' if ok else 'FAIL'} -> {path}")
    np.save(os.path.join(FEAT, "axis_u.npy"), ax["u"])
    return details


def step_status() -> None:
    for g in gates.GATES:
        st = gates.status(g)
        print(f"{g}: {'—' if st is None else ('PASS' if st['passed'] else 'FAIL')}")
    for ds in ("chennu", "anphy", "ds003969", "ds001787"):
        u = _done_units(ds)
        print(f"{ds:9s}: {len({s for s, _ in u})} subjects, {len(u)} subject x condition units")


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Q1 EEG pipeline steps")
    p.add_argument("step", choices=["chennu", "anphy", "g1", "axis", "status"])
    p.add_argument("--max-subjects", type=int, default=None, help="anphy: process at most N subjects")
    p.add_argument("--force-gate", action="store_true")
    a = p.parse_args(argv)
    cfg = load_config()
    try:
        if a.step == "chennu":
            step_chennu(cfg, a.force_gate)
        elif a.step == "anphy":
            step_anphy(cfg, a.force_gate, a.max_subjects)
        elif a.step == "g1":
            step_g1(cfg, a.force_gate)
        elif a.step == "axis":
            step_axis(cfg, a.force_gate)
        else:
            step_status()
    except gates.GateError as e:
        print(f"[DỪNG] {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
