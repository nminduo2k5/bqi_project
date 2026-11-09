"""
OSF 3uszv / v8msp (Rodriguez-Larios, Bracho Montes de Oca & Alaerts 2021,
NeuroImage, CC0): 29 experienced meditators (m_) and 29 novices (c_), about
60 min of breath-focus meditation interrupted by experience-sampling probes
(the "task" files), 21-channel Nexus-32 EEG.

File format. `<name>.txt.cpgz` is gzip(cpio odc archive) holding one CSV text
file with the header FP1,...,O2,A1,A2,SensorCEOG,SensorDEOG,Events. The text
has 2048 rows per second with every EEG sample repeated 4 times (BioTrace+
export at the highest rate of the session); the EEG itself is 512 Hz. Checked
on one 2-min segment of m_task_2 before anything else was read: runs of
identical rows are exactly 4 long and the occipital alpha peak sits at 10 Hz
only when the rows are taken at 2048/s.

Probe structure (preprocessing.m of the authors): trigger 100 = bell, then the
answer (1 breath focus, 2 thinking about something else, 3 distracted by
something else), confidence, sleepiness, then focus (answer 1) or engagement,
visual, auditory (answer 2) or sensation (answer 3). Trigger 0 is a known
glitch; 'Light Trigger' and 'TTL Trigger' are technical and ignored. Codes
before the first bell belong to the practice probe.

Rating (predictions_osf3uszv.md): r = +confidence for breath focus,
-confidence for answers 2 and 3, so -7..7 like Q1 - Q2 of ds001787 but wider.

Only the Events column is read for the probe table (gate F0) and the probe
schedule (gate F1); the EEG samples are read by eeg/osf_features.py after the
preregistration lock.
"""
from __future__ import annotations

import gzip
import io
import json
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROWS_PER_S = 2048                 # rows per second in the text files
REPEAT = 4                        # each EEG sample is written REPEAT times
EEG_SFREQ = ROWS_PER_S / REPEAT   # 512 Hz
FILE_COLUMNS = ("FP1", "FP2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "PZ", "P4",
                "T6", "O1", "O2", "A1", "A2", "SensorCEOG", "SensorDEOG", "Events")
CHANNEL_RENAME = {"FP1": "Fp1", "FP2": "Fp2", "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8", "PZ": "Pz"}
EEG_CHANNELS = tuple(CHANNEL_RENAME.get(c, c) for c in FILE_COLUMNS[:21])   # 19 of the 10-20 system + A1, A2
_NAME = re.compile(r"^([mc])_(task|med|rest)_(n?\d+)\.txt\.cpgz$")
_RS232 = re.compile(r"^RS232 Trigger: (\d+)")
BELL = 100
_CFG = None


def cfg():
    global _CFG
    if _CFG is None:
        from model import config as C
        _CFG = C.OSF
    return _CFG


def data_dir() -> str:
    return os.path.join(ROOT, cfg().data_dir)


def raw_dir() -> str:
    return os.path.join(data_dir(), "raw")


def derivatives_dir() -> str:
    d = os.path.join(data_dir(), "derivatives")
    os.makedirs(d, exist_ok=True)
    return d


def results_dir() -> str:
    return os.path.join(ROOT, cfg().results_dir)


# =============================================================================
# Sessions and the text inside the archives
# =============================================================================
def sessions(condition: str = "task") -> list[dict]:
    """One entry per downloaded recording of `condition`, sorted by group and
    number; duration from the cached Events pass when available."""
    groups = dict(cfg().group_names)
    out = []
    for name in sorted(os.listdir(raw_dir())) if os.path.isdir(raw_dir()) else []:
        m = _NAME.match(name)
        if not m or m.group(2) != condition:
            continue
        g, cond, num = m.groups()
        s = {"sub": f"{g}_{num}", "ses": cond, "group": groups[g], "stem": name[:-len(".txt.cpgz")],
             "file": os.path.join(raw_dir(), name), "sfreq": EEG_SFREQ, "duration_s": float("nan")}
        meta = events_cache_path(s) + ".json"
        if os.path.isfile(meta):
            with open(meta, encoding="utf-8") as f:
                s["duration_s"] = float(json.load(f)["duration_s"])
        out.append(s)
    out.sort(key=lambda s: (s["group"], s["sub"].lstrip("mc_").startswith("n"), int(s["sub"].lstrip("mc_n"))))
    return out


class _Limited(io.RawIOBase):
    """The first `n` bytes of a stream: the archives hold a second entry after
    the text (a macOS '._<name>' resource file) that must not reach the parser."""

    def __init__(self, f, n: int):
        self.f, self.left = f, n

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        if self.left <= 0:
            return 0
        data = self.f.read(min(len(b), self.left))
        n = len(data)
        b[:n] = data
        self.left -= n
        return n

    def close(self) -> None:
        self.f.close()
        super().close()


def open_text(path: str):
    """Binary stream of the CSV text inside the .cpgz (first archive entry only); also its size in bytes."""
    f = gzip.open(path, "rb")
    h = f.read(76)                                   # cpio odc header: magic 070707, ..., namesize(6), filesize(11)
    if h[:6] != b"070707":
        f.close()
        raise ValueError(f"{path}: not a cpio odc archive inside gzip")
    namesize, filesize = int(h[59:65], 8), int(h[65:76], 8)
    f.read(namesize)
    return io.BufferedReader(_Limited(f, filesize), 1 << 20), filesize


def events_cache_path(sess: dict) -> str:
    d = os.path.join(derivatives_dir(), "events")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, sess["stem"] + "_events.csv")


def extract_events(sess: dict, cache: bool = True) -> tuple[pd.DataFrame, int]:
    """Non-empty Events cells (row index at 2048/s, text) and the number of rows.
    Reads only the Events column, in chunks; cached next to the data."""
    path = events_cache_path(sess)
    if cache and os.path.isfile(path) and os.path.isfile(path + ".json"):
        with open(path + ".json", encoding="utf-8") as f:
            n_rows = json.load(f)["n_rows"]
        return pd.read_csv(path, dtype={"row": int, "text": str}, keep_default_na=False), n_rows
    f, _ = open_text(sess["file"])
    rows, texts, n_rows = [], [], 0
    with f:
        # latin-1: the trigger strings carry raw control bytes and non-UTF-8 characters
        for chunk in pd.read_csv(f, usecols=["Events"], dtype=str, keep_default_na=False, chunksize=ROWS_PER_S * 300,
                                 encoding="latin-1"):
            e = chunk["Events"]
            nz = e.index[e != ""]
            rows.extend(nz.tolist())
            texts.extend(e[nz].tolist())
            n_rows = chunk.index[-1] + 1 if len(chunk) else n_rows
    ev = pd.DataFrame({"row": rows, "text": texts})
    if cache:
        ev.to_csv(path, index=False)
        with open(path + ".json", "w", encoding="utf-8") as fj:
            json.dump({"n_rows": int(n_rows), "duration_s": n_rows / ROWS_PER_S, "stem": sess["stem"]}, fj)
    return ev, int(n_rows)


def extract_all(condition: str = "task", verbose: bool = True) -> list[dict]:
    """Cache the Events column of every downloaded recording (about 20 s per task file)."""
    import time
    out = []
    sess_list = sessions(condition)
    for i, s in enumerate(sess_list, 1):
        t0 = time.time()
        cached = os.path.isfile(events_cache_path(s))
        ev, n = extract_events(s)
        out.append({**s, "n_rows": n, "n_events": len(ev)})
        if verbose:
            print(f"  [{i}/{len(sess_list)}] {s['stem']:12s} {n / ROWS_PER_S / 60:6.1f} min, {len(ev):4d} events"
                  + ("" if cached else f"  ({time.time() - t0:.0f} s)"), flush=True)
    return out


# =============================================================================
# Probe decoding (F0)
# =============================================================================
def trigger_code(text: str) -> int | None:
    m = _RS232.match(text)
    return int(m.group(1)) if m else None


def probes_from_events(ev: pd.DataFrame, window_s: float | None = None) -> list[dict]:
    """Group the RS232 codes into probes: one bell (100) and the codes that
    follow it up to the next bell. Code 0 (glitch) is dropped. Codes later than
    `window_s` after the bell are counted as late and not used."""
    window_s = cfg().probe_window_s if window_s is None else window_s
    codes = [(row / ROWS_PER_S, c) for row, text in zip(ev["row"], ev["text"])
             if (c := trigger_code(str(text))) is not None and c != 0]
    probes, practice = [], []
    for t, c in codes:
        if c == BELL:
            probes.append({"onset_s": t, "codes": [], "late": []})
        elif not probes:
            practice.append(c)
        elif t - probes[-1]["onset_s"] <= window_s:
            probes[-1]["codes"].append((t, c))
        else:
            probes[-1]["late"].append((t, c))
    for p in probes:
        p["resp_end_s"] = p["codes"][-1][0] if p["codes"] else None
    if probes:
        probes[0]["practice_codes"] = practice
    return probes


def decode_probe(p: dict) -> dict:
    """Answer, confidence, sleepiness and the answer-specific items of one probe;
    rating = signed confidence, NaN unless answer and confidence are valid."""
    c = cfg()
    vals = [v for _, v in p["codes"]]
    get = lambda i: vals[i] if i < len(vals) else None
    a, conf, sleep = get(0), get(1), get(2)
    out = {"answer": a, "confidence": conf, "sleepiness": sleep, "focus": None, "engagement": None,
           "visual": None, "auditory": None, "sensation": None, "n_codes": len(vals), "n_late": len(p["late"])}
    if a == 1:
        out["focus"] = get(3)
    elif a == 2:
        out["engagement"], out["visual"], out["auditory"] = get(3), get(4), get(5)
    elif a == 3:
        out["sensation"] = get(3)
    lo, hi = c.confidence_range
    valid = a in (c.answers_focus + c.answers_away) and conf is not None and lo <= conf <= hi
    out["valid"] = bool(valid)
    out["rating"] = float(conf if a in c.answers_focus else -conf) if valid else float("nan")
    return out


def decode_session(sess: dict) -> tuple[pd.DataFrame, dict]:
    """Probe table of one session (columns as ds001787's probes.csv: onset_s,
    resp_end_s, rating, q1 = answer, q2 = confidence, q3 = sleepiness) and the
    F0 diagnostics."""
    ev, n_rows = extract_events(sess)
    probes = probes_from_events(ev)
    rows = []
    for i, p in enumerate(probes):
        d = decode_probe(p)
        rows.append({"sub": sess["sub"], "ses": sess["ses"], "group": sess["group"], "probe": i,
                     "onset_s": p["onset_s"], "resp_end_s": p["resp_end_s"],
                     "q1": d["answer"], "q2": d["confidence"], "q3": d["sleepiness"], "rating": d["rating"],
                     **{k: d[k] for k in ("focus", "engagement", "visual", "auditory", "sensation",
                                          "n_codes", "n_late", "valid")}})
    df = pd.DataFrame(rows)
    lat = [p["codes"][-1][0] - p["onset_s"] for p in probes if p["codes"]]
    gaps = np.diff([p["onset_s"] for p in probes]) if len(probes) > 1 else np.array([])
    stats = {"sub": sess["sub"], "group": sess["group"], "duration_s": n_rows / ROWS_PER_S, "n_bells": len(probes),
             "n_valid": int(df["valid"].sum()) if len(df) else 0,
             "n_with_3_codes": int((df["n_codes"] >= 3).sum()) if len(df) else 0,
             "n_late_codes": int(df["n_late"].sum()) if len(df) else 0,
             "n_practice_codes": len(probes[0].get("practice_codes", [])) if probes else 0,
             "answer_counts": {int(k): int(v) for k, v in df["q1"].value_counts().items()} if len(df) else {},
             "response_latency_s_median": float(np.median(lat)) if lat else None,
             "bell_gap_s": {"min": float(gaps.min()), "median": float(np.median(gaps)), "max": float(gaps.max())}
             if len(gaps) else None,
             "first_bell_s": probes[0]["onset_s"] if probes else None}
    return df, stats


def decode_all(verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Decode every task session; write derivatives/probes.csv and f0_summary.json
    (and a tracked copy under outputs/results/osf3uszv)."""
    frames, per = [], []
    sess_list = sessions("task")
    for s in sess_list:
        df, st = decode_session(s)
        frames.append(df)
        per.append(st)
        if verbose:
            print(f"  {s['sub']:6s} {s['group']:6s} {st['duration_s'] / 60:5.1f} min  bells {st['n_bells']:3d}  "
                  f"valid {st['n_valid']:3d}  answers {st['answer_counts']}  "
                  f"gap median {st['bell_gap_s']['median'] if st['bell_gap_s'] else float('nan'):.0f} s", flush=True)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)   # all-NaN columns (e.g. sensation) in some sessions
        probes = pd.concat([f for f in frames if len(f)], ignore_index=True)
    tot = {k: int(sum(p[k] for p in per)) for k in ("n_bells", "n_valid", "n_with_3_codes", "n_late_codes")}
    ranges = {k: [int(probes[k].min()), int(probes[k].max())] for k in ("q2", "q3", "focus", "engagement")
              if probes[k].notna().any()}
    summary = {**tot, "n_sessions": len(per), "n_sessions_per_group": {g: sum(p["group"] == g for p in per)
                                                                     for g in ("expert", "novice")},
               "decoded_rate": tot["n_valid"] / max(tot["n_bells"], 1),
               "three_codes_rate": tot["n_with_3_codes"] / max(tot["n_bells"], 1),
               "answer_counts": {int(k): int(v) for k, v in probes["q1"].value_counts().items()},
               "observed_ranges": ranges, "subject_minutes": float(sum(p["duration_s"] for p in per) / 60.0),
               "bell_gap_s_median": float(np.median([p["bell_gap_s"]["median"] for p in per if p["bell_gap_s"]])),
               "sessions": per}
    for d in (derivatives_dir(), results_dir()):
        os.makedirs(d, exist_ok=True)
        probes.to_csv(os.path.join(d, "probes.csv"), index=False)
        with open(os.path.join(d, "f0_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
    return probes, summary


def gate_F0(summary: dict, verbose: bool = True) -> dict:
    from eeg import gates
    c = cfg()
    passed = bool(summary["decoded_rate"] >= c.f0_min_decoded and summary["n_sessions"] >= c.f0_min_sessions
                  and all(p["n_bells"] >= 1 for p in summary["sessions"]))
    details = {k: summary[k] for k in ("n_sessions", "n_sessions_per_group", "n_bells", "n_valid", "decoded_rate",
                                       "three_codes_rate", "answer_counts", "observed_ranges", "bell_gap_s_median")}
    path = gates.record("F0", passed, details)
    if verbose:
        print(f"F0 {'PASS' if passed else 'FAIL'}: {summary['n_sessions']} sessions, {summary['n_bells']} bells, "
              f"decoded {summary['decoded_rate']:.1%}, ranges {summary['observed_ranges']} -> {path}")
    return {"passed": passed, **details}


# =============================================================================
# Real observation design of a session (F1 and the fit)
# =============================================================================
def load_probes() -> pd.DataFrame:
    return pd.read_csv(os.path.join(derivatives_dir(), "probes.csv"))


def session_design(sess: dict, probes: pd.DataFrame, require_rating: bool = False):
    """Same grid, probe windows and time resolution as ds001787 (REALDATA)."""
    from eeg.loaders import ds001787 as DS1
    return DS1.session_design(sess, probes, require_rating)


def all_designs(require_rating: bool = False) -> list[tuple[dict, object, dict]]:
    probes = load_probes()
    return [(s, *session_design(s, probes, require_rating)) for s in sessions("task") if np.isfinite(s["duration_s"])]
