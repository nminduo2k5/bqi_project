"""
ds001787 (Brandmeyer & Delorme 2018, OpenNeuro, CC0): metadata, probe decoding
(gate E0) and the real observation design of every session (for E1 and the fit).

Only the git clone of the OpenNeuro repository is needed here (events.tsv,
*_eeg.json, participants.tsv, git-annex pointers of the BDF files) plus the
behaviour files `code/MW_Current_TextFileBIDS.zip` (360 KB). No EEG is read.

Probe structure (run_mw_experiment6.m): marker 128 = first question
("rate meditation depth"), then the participant answers Q1 (depth, status 1),
Q2 (mind wandering, status 2), Q3 (tiredness, status 3) with keys 0-3. Key k
pulses parallel-port line k+1, so key 1/2/3 -> trigger 2/4/8 and key 0 ->
trigger 1, which is absent from events.tsv (hypothesis "key 0 leaves no
trigger", tested in E0 against the behaviour files, which log every key).
"""
from __future__ import annotations

import json
import os
import re
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CFG = None


def cfg():
    global _CFG
    if _CFG is None:
        from model import config as C
        _CFG = C.REALDATA
    return _CFG


def bids_dir() -> str:
    return os.path.join(ROOT, cfg().bids_dir)


def data_dir() -> str:
    return os.path.join(ROOT, cfg().data_dir)


def derivatives_dir() -> str:
    d = os.path.join(data_dir(), "derivatives")
    os.makedirs(d, exist_ok=True)
    return d


# =============================================================================
# Metadata
# =============================================================================
def participants() -> pd.DataFrame:
    return pd.read_csv(os.path.join(bids_dir(), "participants.tsv"), sep="\t")


def _annex_key(pointer_path: str):
    """Size and MD5 of an annexed file from its git-annex pointer (symlink text)."""
    try:
        with open(pointer_path, encoding="utf-8", errors="ignore") as f:
            txt = f.read(2000)
    except OSError:
        return None, None
    m = re.search(r"MD5E-s(\d+)--([0-9a-f]{32})", txt)
    return (int(m.group(1)), m.group(2)) if m else (None, None)


def sessions() -> list[dict]:
    """All (subject, session) pairs with paths, group, duration and BDF annex key."""
    groups = dict(zip(participants()["participant_id"], participants()["group"]))
    out = []
    for sub in sorted(d for d in os.listdir(bids_dir()) if re.fullmatch(r"sub-\d+", d)):
        for ses in sorted(d for d in os.listdir(os.path.join(bids_dir(), sub)) if d.startswith("ses-")):
            eeg = os.path.join(bids_dir(), sub, ses, "eeg")
            stem = f"{sub}_{ses}_task-meditation"
            js = os.path.join(eeg, stem + "_eeg.json")
            if not os.path.isfile(js):
                continue
            with open(js, encoding="utf-8") as f:
                meta = json.load(f)
            size, md5 = _annex_key(os.path.join(eeg, stem + "_eeg.bdf"))
            rel = f"{sub}/{ses}/eeg/{stem}_eeg.bdf"
            out.append({"sub": sub, "ses": ses, "group": groups.get(sub, "n/a"),
                        "events": os.path.join(eeg, stem + "_events.tsv"), "json": js,
                        "bdf_rel": rel, "bdf_local": os.path.join(data_dir(), *rel.split("/")),
                        "bdf_size": size, "bdf_md5": md5,
                        "duration_s": float(meta.get("RecordingDuration", np.nan)),
                        "sfreq": float(meta.get("SamplingFrequency", 256))})
    return out


def read_events(path: str) -> pd.DataFrame:
    e = pd.read_csv(path, sep="\t")
    e["value"] = pd.to_numeric(e["value"], errors="coerce")
    return e


def probes_from_events(ev: pd.DataFrame) -> list[dict]:
    """Group the events into probes: one marker 128 followed by its responses
    (codes 2/4/8; 16 = involuntary press flagged, other codes ignored)."""
    ev = ev.sort_values("onset").reset_index(drop=True)
    probes = []
    for _, row in ev.iterrows():
        v = row["value"]
        if v == 128:
            probes.append({"onset": float(row["onset"]), "responses": [], "involuntary": 0})
        elif probes and v in (2, 4, 8):
            probes[-1]["responses"].append((float(row["onset"]), int(v)))
        elif probes and v == 16:
            probes[-1]["involuntary"] += 1
    for p in probes:
        p["resp_end"] = p["responses"][-1][0] if p["responses"] else None
    return probes


# =============================================================================
# Behaviour files
# =============================================================================
_BEH_RE = re.compile(r"^sub(\d+)_*(?:(\d+)_)?info(_old)?\.txt$")


def behaviour_candidates() -> dict:
    """sub-XXX -> list of behaviour log paths of that subject (all sessions)."""
    code = os.path.join(data_dir(), "code")
    folder = os.path.join(code, "MW_Current_TextFileBIDS")
    zpath = os.path.join(code, "MW_Current_TextFileBIDS.zip")
    if not os.path.isdir(folder) and os.path.isfile(zpath):
        with zipfile.ZipFile(zpath) as z:
            z.extractall(code, members=[m for m in z.namelist() if not m.startswith("__MACOSX")])
    out: dict = {}
    if not os.path.isdir(folder):
        return out
    for name in sorted(os.listdir(folder)):
        m = _BEH_RE.match(name)
        if not m or m.group(3):          # skip *_old
            continue
        out.setdefault(f"sub-{int(m.group(1)):03d}", []).append(os.path.join(folder, name))
    return out


def match_behaviour_files(sess_list: list[dict] | None = None, tol: float | None = None) -> dict:
    """(sub, ses) -> behaviour path, choosing for every subject the assignment
    of files to EEG sessions that maximises the number of aligned probes
    (the file suffix _2/_3 is not a reliable session index: e.g. sub14_info.txt
    does not correspond to the only BIDS session of sub-014)."""
    tol = tol or cfg().e0_max_offset_s
    sess_list = sess_list or sessions()
    cands = behaviour_candidates()
    out = {}
    for sub in sorted({s["sub"] for s in sess_list}):
        files = cands.get(sub, [])
        if not files:
            continue
        ss = [s for s in sess_list if s["sub"] == sub]
        beh_probes = {f: [b["t_q"] for b in parse_behaviour(f)] for f in files}
        scores = []
        for s in ss:
            onsets = [p["onset"] for p in probes_from_events(read_events(s["events"]))]
            for f in files:
                _, m = align_times(onsets, beh_probes[f], tol)
                scores.append((len(m), s["ses"], f))
        used_s, used_f = set(), set()
        for n, ses, f in sorted(scores, key=lambda x: -x[0]):
            if n == 0 or ses in used_s or f in used_f:
                continue
            out[(sub, ses)] = f
            used_s.add(ses); used_f.add(f)
    return out


def behaviour_files() -> dict:
    """(sub-XXX, ses-YY) -> path of the behaviour log, by file-name index
    (sub01_info -> ses-01, sub01_2_info -> ses-02). Prefer
    `match_behaviour_files`, which assigns files by probe alignment."""
    code = os.path.join(data_dir(), "code")
    folder = os.path.join(code, "MW_Current_TextFileBIDS")
    zpath = os.path.join(code, "MW_Current_TextFileBIDS.zip")
    if not os.path.isdir(folder) and os.path.isfile(zpath):
        with zipfile.ZipFile(zpath) as z:
            z.extractall(code, members=[m for m in z.namelist() if not m.startswith("__MACOSX")])
    out = {}
    if not os.path.isdir(folder):
        return out
    for name in sorted(os.listdir(folder)):
        m = _BEH_RE.match(name)
        if not m or m.group(3):          # skip *_old
            continue
        sub = f"sub-{int(m.group(1)):03d}"
        ses = f"ses-{int(m.group(2) or 1):02d}"
        out[(sub, ses)] = os.path.join(folder, name)
    return out


def parse_behaviour(path: str) -> list[dict]:
    """Probes of one behaviour log: question time and the (time, key, status) presses."""
    probes = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.match(r"MW question asked at time\s+([\d.]+)\s+second", line)
            if m:
                probes.append({"t_q": float(m.group(1)), "presses": []})
                continue
            # some files have no space before "(code": "Key 2(code 2) pressed at time 125.3 seconds, status 1"
            m = re.match(r"Key (\S+?)\s*\(code (-?\d+)\) pressed at time\s+([\d.]+)\s+seconds, status (\d)", line)
            if m and probes:
                key = m.group(1)
                probes[-1]["presses"].append((float(m.group(3)), key, int(m.group(4))))
    for p in probes:
        ans = {}
        for t, key, st in p["presses"]:
            if st in (1, 2, 3) and key in ("0", "1", "2", "3") and st not in ans:
                ans[st] = (int(key), t)
        p["answers"] = ans
    return probes


def align_times(onsets_eeg: np.ndarray, times_beh: np.ndarray, tol: float) -> tuple[float, list]:
    """Constant clock offset between the EEG (marker 128) and the behaviour log
    (toc): choose the offset that matches the most probes within `tol`, then
    refine it as the median of the matched differences. Returns (offset,
    list of (i_eeg, j_beh))."""
    onsets_eeg, times_beh = np.asarray(onsets_eeg, float), np.asarray(times_beh, float)
    if len(onsets_eeg) == 0 or len(times_beh) == 0:
        return float("nan"), []
    cands = []
    for i, o in enumerate(onsets_eeg):
        for j in range(max(0, i - 4), min(len(times_beh), i + 5)):
            cands.append(o - times_beh[j])
    best, best_n = None, -1
    for c in cands:
        d = np.abs(onsets_eeg[:, None] - (times_beh[None, :] + c))
        n = int((d.min(axis=1) < tol).sum())
        if n > best_n:
            best, best_n = c, n
    for _ in range(2):   # refine
        d = onsets_eeg[:, None] - (times_beh[None, :] + best)
        j = np.abs(d).argmin(axis=1)
        ok = np.abs(d[np.arange(len(onsets_eeg)), j]) < tol
        if ok.any():
            best = best + float(np.median(d[np.arange(len(onsets_eeg)), j][ok]))
    d = onsets_eeg[:, None] - (times_beh[None, :] + best)
    j = np.abs(d).argmin(axis=1)
    used, matches = set(), []
    for i in range(len(onsets_eeg)):
        if abs(d[i, j[i]]) < tol and j[i] not in used:
            matches.append((i, int(j[i])))
            used.add(int(j[i]))
    return float(best), matches


# =============================================================================
# Decoding (E0)
# =============================================================================
def decode_session(sess: dict, beh_path: str | None) -> tuple[pd.DataFrame, dict]:
    """Probe table of one session with Q1-Q3 and the E0 diagnostics."""
    c = cfg()
    ev = read_events(sess["events"])
    probes = probes_from_events(ev)
    rows, stats = [], {"n_probe_eeg": len(probes), "n_matched": 0, "n_beh": 0, "offset_s": None,
                       "key0_with_trigger": 0, "key0_total": 0, "nonzero_without_trigger": 0, "nonzero_total": 0,
                       "code_agree": 0, "code_total": 0, "source": "events"}
    beh = parse_behaviour(beh_path) if beh_path and os.path.isfile(beh_path) else []
    matches, offset = [], float("nan")
    if beh:
        offset, matches = align_times([p["onset"] for p in probes], [b["t_q"] for b in beh], c.e0_max_offset_s)
        stats.update(n_beh=len(beh), n_matched=len(matches), offset_s=offset, source="behaviour")
    mdict = dict(matches)
    for i, p in enumerate(probes):
        q = {1: None, 2: None, 3: None}
        src = "none"
        if i in mdict:
            b = beh[mdict[i]]
            for st, (key, _) in b["answers"].items():
                q[st] = key
            src = "behaviour"
            # hypothesis check: key 0 -> no trigger; key 1-3 -> trigger 2^key in order
            keys = [b["answers"][s][0] for s in (1, 2, 3) if s in b["answers"]]
            codes = [code for _, code in p["responses"]]
            nz = [2 ** k for k in keys if k != 0]
            stats["key0_total"] += sum(1 for k in keys if k == 0)
            stats["nonzero_total"] += len(nz)
            stats["code_total"] += len(nz)
            stats["code_agree"] += sum(1 for a, b2 in zip(nz, codes) if a == b2)
            if len(codes) > len(nz):
                stats["key0_with_trigger"] += len(codes) - len(nz)
            elif len(codes) < len(nz):
                stats["nonzero_without_trigger"] += len(nz) - len(codes)
        elif len(p["responses"]) == 3:
            # all three answered with keys 1-3: unambiguous from the events alone
            for st, (_, code) in zip((1, 2, 3), p["responses"]):
                q[st] = int(np.log2(code))
            src = "events"
        rows.append({"sub": sess["sub"], "ses": sess["ses"], "group": sess["group"], "probe": i,
                     "onset_s": p["onset"], "resp_end_s": p["resp_end"], "n_resp_events": len(p["responses"]),
                     "involuntary": p["involuntary"], "q1": q[1], "q2": q[2], "q3": q[3], "source": src})
    df = pd.DataFrame(rows)
    if len(df):
        df["rating"] = df["q1"].astype(float) - df["q2"].astype(float)
        stats["n_known_q1q2"] = int((df["q1"].notna() & df["q2"].notna()).sum())
    else:
        stats["n_known_q1q2"] = 0
    return df, stats


def decode_all(verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Decode every session; write derivatives/probes.csv and e0_summary.json."""
    c = cfg()
    sess_list = sessions()
    beh = match_behaviour_files(sess_list)
    frames, per = [], []
    for s in sess_list:
        df, st = decode_session(s, beh.get((s["sub"], s["ses"])))
        st.update(sub=s["sub"], ses=s["ses"], group=s["group"], duration_s=s["duration_s"],
                  behaviour_file=os.path.basename(beh[(s["sub"], s["ses"])]) if (s["sub"], s["ses"]) in beh else None)
        frames.append(df)
        per.append(st)
        if verbose:
            print(f"  {s['sub']} {s['ses']} {s['group']:6s} probes {st['n_probe_eeg']:3d} "
                  f"matched {st['n_matched']:3d} ({st['source']}) Q1Q2 known {st['n_known_q1q2']:3d}"
                  + (f" offset {st['offset_s']:.2f}s" if st['offset_s'] is not None else ""))
    probes = pd.concat([f for f in frames if len(f)], ignore_index=True)
    tot = {k: int(sum(p[k] for p in per)) for k in ("n_probe_eeg", "n_matched", "n_known_q1q2", "key0_with_trigger",
                                                      "key0_total", "nonzero_without_trigger", "nonzero_total",
                                                      "code_agree", "code_total")}
    summary = {**tot, "n_sessions": len(per), "n_sessions_with_behaviour": sum(p["source"] == "behaviour" for p in per),
               "match_rate": tot["n_matched"] / max(tot["n_probe_eeg"], 1),
               "known_rate": tot["n_known_q1q2"] / max(tot["n_probe_eeg"], 1),
               "code_agreement": tot["code_agree"] / max(tot["code_total"], 1),
               "key0_trigger_rate": tot["key0_with_trigger"] / max(tot["key0_total"], 1),
               "sessions": per}
    # derivatives (next to the data, ignored by git) and a tracked copy under outputs/results/realdata
    for d in (derivatives_dir(), os.path.join(ROOT, "outputs", "results", "realdata")):
        os.makedirs(d, exist_ok=True)
        probes.to_csv(os.path.join(d, "probes.csv"), index=False)
        with open(os.path.join(d, "e0_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
    return probes, summary


def gate_E0(summary: dict, verbose: bool = True) -> dict:
    from eeg import gates
    c = cfg()
    hypothesis_ok = summary["key0_trigger_rate"] < 0.05 and summary["code_agreement"] >= c.e0_min_match
    passed = bool(summary["match_rate"] >= c.e0_min_match and summary["known_rate"] >= c.e0_min_known and hypothesis_ok)
    details = {k: summary[k] for k in ("n_sessions", "n_sessions_with_behaviour", "n_probe_eeg", "n_matched", "n_known_q1q2",
                                       "match_rate", "known_rate", "code_agreement", "key0_trigger_rate",
                                       "key0_total", "nonzero_total")}
    details["hypothesis_key0_no_trigger"] = hypothesis_ok
    path = gates.record("E0", passed, details)
    if verbose:
        print(f"E0 {'PASS' if passed else 'FAIL'}: match {summary['match_rate']:.1%}, Q1/Q2 known {summary['known_rate']:.1%}, "
              f"code agreement {summary['code_agreement']:.1%}, key-0 presses with a trigger "
              f"{summary['key0_with_trigger']}/{summary['key0_total']} -> {path}")
    return {"passed": passed, **details}


# =============================================================================
# Real observation design of a session (E1 and the fit)
# =============================================================================
def load_probes() -> pd.DataFrame:
    return pd.read_csv(os.path.join(derivatives_dir(), "probes.csv"))


def session_design(sess: dict, probes: pd.DataFrame, require_rating: bool = False):
    """IrregularDesign of one session from its events: regular 10-s EEG grid
    from the recording start (epoch end times), epochs overlapping a probe
    window masked, ratings at the probe onsets.

    Returns (design, info). Times in minutes, rounded to the configured
    resolution so that the Kalman step cache stays small."""
    from model.fisher import IrregularDesign
    c = cfg()
    dur = sess["duration_s"]
    ep = c.epoch_s
    n_ep = int(np.floor(dur / ep))
    t_end = np.arange(1, n_ep + 1) * ep
    pr = probes[(probes["sub"] == sess["sub"]) & (probes["ses"] == sess["ses"])].sort_values("onset_s")
    if require_rating:
        pr = pr[pr["rating"].notna()]
    windows = []
    for _, p in pr.iterrows():
        end = p["resp_end_s"] + c.probe_pad_after_s if pd.notna(p["resp_end_s"]) else p["onset_s"] + c.probe_no_response_s
        windows.append((p["onset_s"], end))
    mask = np.ones(n_ep, bool)
    for a, b in windows:
        mask &= ~((t_end > a) & (t_end - ep < b))
    res = c.time_resolution_s
    t_min = np.round(t_end / res) * res / 60.0
    t_probe = np.round(pr["onset_s"].to_numpy(float) / res) * res / 60.0
    design = IrregularDesign(tuple(t_min), tuple(mask), tuple(mask), tuple(t_probe))
    info = {"sub": sess["sub"], "ses": sess["ses"], "group": sess["group"], "n_epochs": n_ep,
            "n_masked": int((~mask).sum()), "missing_frac": float((~mask).mean()), "n_probes": len(pr),
            "duration_min": dur / 60.0}
    return design, info


def all_designs(require_rating: bool = False) -> list[tuple[dict, object, dict]]:
    probes = load_probes()
    return [(s, *session_design(s, probes, require_rating)) for s in sessions()]
