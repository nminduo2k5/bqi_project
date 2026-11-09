"""
Second real dataset (OSF 3uszv): the archive reader, the probe decoding and
the rating of the trigger column, the EEG reader (repeat phase, NaN tail), the
designs, and the dataset-parametrised gates of model.realdata. Everything runs
on a synthetic .txt.cpgz written here; the real data are never needed.
"""
from __future__ import annotations

import gzip
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from eeg.loaders import osf3uszv as DS
from model import config as CFG

ROWS = DS.ROWS_PER_S


def _cpgz(path: str, name: str, n_s: float, events: dict, seed: int = 0, nan_tail: int = 0) -> np.ndarray:
    """Write a synthetic archive: `n_s` seconds, EEG at 512 Hz repeated 4 times
    per row with the first group cut to 3 rows (phase 3), events {row: text}."""
    rng = np.random.default_rng(seed)
    n_samp = int(n_s * DS.EEG_SFREQ)
    X = rng.normal(size=(n_samp, 23)).astype(np.float32).round(3)
    rows = np.repeat(X, DS.REPEAT, axis=0)[1:]             # first group 3 rows long: full groups start at row 3
    lines = [",".join(DS.FILE_COLUMNS)]
    for i, r in enumerate(rows):
        vals = ["nan"] * 23 if i >= len(rows) - nan_tail else [f"{v:.3f}" for v in r]
        lines.append(",".join(vals) + "," + events.get(i, ""))
    text = ("\n".join(lines) + "\n").encode("latin-1")
    # magic, dev ino mode uid gid nlink rdev (7 x 6), mtime (11), namesize (6), filesize (11)
    def hdr(nm: str, size: int) -> bytes:
        h = b"070707" + b"0" * 42 + b"0" * 11 + f"{len(nm) + 1:06o}".encode() + f"{size:011o}".encode()
        assert len(h) == 76
        return h + nm.encode() + b"\0"
    junk = b"\x00\x05\x16\x07resource fork, not text\n1,2,3\n"   # the real archives carry a '._<name>' entry after the text
    with gzip.open(path, "wb") as f:
        f.write(hdr(name, len(text)) + text + hdr("._" + name, len(junk)) + junk + hdr("TRAILER!!!", 0))
    return X


def _events() -> dict:
    t = lambda s: int(s * ROWS)
    ev = {t(0.5): "Light Trigger", t(1.0): "RS232 Trigger: 2('\x02')", t(1.2): "RS232 Trigger: 5('\x05')",   # practice
          t(3.0): "RS232 Trigger: 100('d')", t(3.5): "RS232 Trigger: 1('\x01')", t(4.0): "RS232 Trigger: 6('\x06')",
          t(4.5): "RS232 Trigger: 2('\x02')", t(5.0): "RS232 Trigger: 4('\x04')", t(5.1): "RS232 Trigger: 0('\xe2",
          t(9.0): "RS232 Trigger: 100('d')", t(9.4): "RS232 Trigger: 2('\x02')", t(9.9): "RS232 Trigger: 3('\x03')",
          t(10.3): "RS232 Trigger: 2('\x02')", t(10.8): "RS232 Trigger: 6('\x06')", t(11.0): "RS232 Trigger: 3('\x03')",
          t(11.2): "RS232 Trigger: 3('\x03')",
          t(15.0): "RS232 Trigger: 100('d')", t(15.6): "RS232 Trigger: 3('\x03')", t(16.0): "RS232 Trigger: 7('\x07')",
          t(16.4): "RS232 Trigger: 1('\x01')", t(16.8): "RS232 Trigger: 5('\x05')",
          t(20.0): "RS232 Trigger: 100('d')", t(20.5): "RS232 Trigger: 1('\x01')"}            # no confidence: invalid
    return ev


@pytest.fixture
def fake_data(tmp_path, monkeypatch):
    root = tmp_path / "osf"
    raw = root / "raw"
    raw.mkdir(parents=True)
    monkeypatch.setattr(DS, "data_dir", lambda: str(root))
    X = _cpgz(str(raw / "m_task_2.txt.cpgz"), "m_task_2.txt", 24.0, _events(), seed=1, nan_tail=6)
    _cpgz(str(raw / "c_task_n1.txt.cpgz"), "c_task_n1.txt", 24.0, _events(), seed=2)
    return root, X


def test_sessions_and_events(fake_data):
    ss = DS.sessions("task")
    assert [s["sub"] for s in ss] == ["m_2", "c_n1"] and [s["group"] for s in ss] == ["expert", "novice"]
    ev, n_rows = DS.extract_events(ss[0], cache=False)
    assert n_rows == 24 * ROWS - 1 and len(ev) == len(_events())
    assert "Light" in ev["text"].iloc[0]
    ev2, n2 = DS.extract_events(ss[0])                       # cached pass gives the same table and the duration
    assert n2 == n_rows and ev2["row"].tolist() == ev["row"].tolist()
    assert abs(DS.sessions("task")[0]["duration_s"] - 24.0) < 1e-3


def test_probe_decoding_and_rating(fake_data):
    ss = DS.sessions("task")
    df, st = DS.decode_session(ss[0])
    assert st["n_bells"] == 4 and st["n_practice_codes"] == 2 and st["n_valid"] == 3
    assert df["q1"].tolist()[:3] == [1, 2, 3] and df["q2"].tolist()[:3] == [6, 3, 7]
    assert df["rating"].tolist()[:3] == [6.0, -3.0, -7.0] and np.isnan(df["rating"].iloc[3])
    assert df["q3"].tolist()[:3] == [2, 2, 1]
    r0, r1, r2 = df.iloc[0], df.iloc[1], df.iloc[2]
    assert r0["focus"] == 4 and pd.isna(r0["engagement"])     # the glitch code 0 is dropped
    assert (r1["engagement"], r1["visual"], r1["auditory"]) == (6, 3, 3)
    assert r2["sensation"] == 5
    assert abs(r0["onset_s"] - 3.0) < 1e-6 and abs(r0["resp_end_s"] - 5.0) < 1e-6
    assert st["bell_gap_s"]["min"] == 5.0 and st["response_latency_s_median"] > 0


def test_decode_all_gate_and_designs(fake_data, monkeypatch, tmp_path):
    from eeg import gates
    monkeypatch.setattr(gates, "GATE_DIR", str(tmp_path / "gates"))
    monkeypatch.setattr(DS, "results_dir", lambda: str(tmp_path / "res"))
    probes, summary = DS.decode_all(verbose=False)
    assert summary["n_sessions"] == 2 and summary["n_bells"] == 8 and summary["decoded_rate"] == 0.75
    assert summary["observed_ranges"]["q2"] == [3, 7]
    f0 = DS.gate_F0(summary, verbose=False)
    assert not f0["passed"]                                 # 2 sessions < 40, decoded 75 % < 90 %
    monkeypatch.setattr(CFG.OSF, "f0_min_sessions", 2, raising=True) if False else None
    trip = DS.all_designs()
    assert len(trip) == 2
    s, d, info = trip[0]
    assert info["n_probes"] == 4 and info["n_epochs"] == 2 and len(d.t_probe) == 4
    trip_r = DS.all_designs(require_rating=True)
    assert trip_r[0][2]["n_probes"] == 3


def test_read_eeg_phase_and_nan_tail(fake_data):
    from eeg import osf_features as OF
    root, X = fake_data
    s = DS.sessions("task")[0]
    Y, info = OF.read_eeg(s)
    # the first group has 3 rows, so full groups start at row 3 (mod 4) and the first sample is X[1]
    assert info["phase"] == 3 and info["n_nan_samples"] >= 1
    n = Y.shape[1]
    assert Y.shape[0] == 21 and n < X.shape[0] and np.allclose(Y, X[1:n + 1, :21].T, atol=2e-3)
    assert np.isfinite(Y).all()


def test_load_raw_resampled(fake_data):
    pytest.importorskip("mne")
    from eeg import osf_features as OF
    raw, info = OF.load_raw(DS.sessions("task")[1])
    assert raw.info["sfreq"] == CFG.OSF.target_sfreq and len(raw.ch_names) == 21
    assert raw.ch_names[:3] == ["Fp1", "Fp2", "F7"] and "T7" in raw.ch_names and "A1" in raw.ch_names
    assert all(np.isfinite(raw.info["chs"][i]["loc"][:3]).all() for i in range(21))
    assert abs(raw.times[-1] - 24.0) < 0.1


def test_empty_variant_and_signed_profile():
    """A sensitivity variant that keeps no session gives an empty list (on OSF
    every session has > 20 % missing epochs from the probe windows alone), and
    the profile interval of a negative kappa is taken on the linear grid."""
    from model import realdata as R
    from experiments.run_realdata import profile_interval
    ss, info = R.normalise_sessions([])
    assert ss == [] and len(info["pooled_sd"]) == 2
    g = -0.47 + np.linspace(-0.8, 0.8, 9) * 0.47                  # decreasing order is not assumed
    ll = -0.5 * ((g + 0.47) / 0.1) ** 2
    pi = profile_interval({"grid": g[::-1].tolist(), "loglik": ll[::-1].tolist(), "unimodal": True})
    assert pi["scale"] == "linear" and pi["se_log"] is None
    assert abs(pi["ci95"][0] - (-0.47 - 1.96 * 0.1)) < 0.02 and abs(pi["ci95"][1] - (-0.47 + 1.96 * 0.1)) < 0.02
    gp = 0.2 * np.exp(np.linspace(-0.8, 0.8, 9))
    pp = profile_interval({"grid": gp.tolist(), "loglik": (-0.5 * (np.log(gp / 0.2) / 0.2) ** 2).tolist()})
    assert pp["scale"] == "log" and abs(pp["se_log"] - 0.2) < 0.02


def test_after_first_probe_variant():
    from model import osf_pipeline as OP
    s = {"t": np.arange(1, 61) / 6.0, "y": np.ones((60, 2)), "t_probe": np.array([3.5, 5.0]), "rating": np.array([1.0, -2.0])}
    v = OP.after_first_probe(s, lead_min=1.0)
    t = np.asarray(s["t"])
    assert np.isnan(v["y"][t < 2.5]).all() and np.isfinite(v["y"][t >= 2.5]).all()
    assert np.isfinite(s["y"]).all()                       # the input is not modified
    assert "postreg_after_first_probe" in OP.sensitivity_loaders()


def test_parametrised_gates_of_realdata(tmp_path, monkeypatch):
    """design_crb and the fit entry points record the gates of the second
    dataset (F*) and the wider rating range in the posterior predictive check."""
    from eeg import gates
    from bqi.dmn_ode import DeviationDMN
    from model import realdata as R
    from model.fisher import IrregularDesign
    monkeypatch.setattr(gates, "GATE_DIR", str(tmp_path / "gates"))
    gates.record("F0", True, {})
    designs, metas = [], []
    for k in range(2):
        n = 90
        t = (np.arange(n) + 1) / 6
        probes = tuple(np.round(np.arange(1.2, t[-1] - 0.3, 1.6), 1))
        designs.append(IrregularDesign(tuple(t), (True,) * n, (True,) * n, probes))
        metas.append({"sub": f"m_{k}", "ses": "task", "group": "expert" if k == 0 else "novice"})
    infos = [{"sub": m["sub"], "ses": "task", "group": m["group"], "n_probes": len(d.t_probe), "duration_min": 15.0,
              "missing_frac": 0.0, "n_epochs": 90} for m, d in zip(metas, designs)]
    out = R.design_crb(output_dir=str(tmp_path / "res"), verbose=False,
                       designs_info=list(zip(metas, designs, infos)), gate=("F0", "F1"), lock=str(tmp_path / "none.lock"))
    assert gates.status("F1") is not None and gates.status("E1") is None
    assert out["gate_E1"]["gate"] == "F1" and not out["gate_E1"]["predictions_locked"]
    truth = DeviationDMN(**R.RD.nominal())
    raw = R.simulate_sessions(truth, designs, seed=3, metas=metas)
    sessions, _ = R.normalise_sessions(raw)
    fr = R.run_fit(seed=0, fast=True, output_dir=str(tmp_path / "res"), sessions=sessions, verbose=False,
                   gate="F1", mde_log=0.7)
    assert fr["groups"]["E_H5"]["mde_log"] == 0.7
    ppc = R.posterior_predictive_ratings(DeviationDMN(**fr["pooled"]["params"]), sessions, n_rep=3, seed=0, rating_range=7)
    assert ppc["levels"] == list(range(-7, 8)) and len(ppc["level_prop"]["observed"]) == 15
    loaders = R.default_sensitivity_loaders(str(tmp_path / "nofeat"), artifact_rule="adaptive")
    assert "alt_artifact_rule_locked" in loaders and "postreg_adaptive_artifact_rule" not in loaders
