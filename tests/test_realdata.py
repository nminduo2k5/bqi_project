"""Tests for the real-data section (Q1_pipeline.md v2.0): deviation-form model,
irregular Fisher information and Kalman filter, ds001787 decoding and designs,
parameter recovery on the real probe schedules."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bqi.dmn_ode import DeviationDMN
from eeg.loaders import ds001787 as DS
from eeg.model_fit import kalman_filter_irregular, kalman_loglik, kalman_loglik_irregular
from model import config as CFG
from model import realdata as RDM
from model.fisher import Design, IrregularDesign, fisher_exact, gaussian_loglik, session_moments, session_moments_irregular

HAS_DS = os.path.isdir(os.path.join(DS.bids_dir(), "sub-001"))


def _dev(rho=0.0):
    return DeviationDMN(**CFG.REALDATA.nominal(), rho_obs=rho)


def test_deviation_model_has_zero_mean_and_keeps_type():
    m = _dev()
    assert np.allclose(m.stationary_mean(), 0) and m.gamma0 == 0 and m.A0 == 0
    m2 = m.with_params(k_inh=0.3, rho_obs=0.1)
    assert isinstance(m2, DeviationDMN) and m2.k_inh == 0.3 and m2.rho_obs == 0.1
    assert "k_int" not in DeviationDMN.POSITIVE and len(DeviationDMN.FREE) == 10


def test_irregular_moments_match_regular_grid():
    m = _dev()
    d = Design.regular(10.0, 2.0)
    t = tuple((np.arange(d.n_epochs) + 1) * d.dt)
    di = IrregularDesign(t, (True,) * d.n_epochs, (True,) * d.n_epochs, tuple(t[i] for i in d.probe_idx))
    mu1, S1 = session_moments(m, d)
    mu2, S2 = session_moments_irregular(m, di)
    # same observations, different ordering: compare via permutation-free summaries
    assert np.allclose(mu1, 0) and np.allclose(mu2, 0)
    assert abs(np.linalg.slogdet(S1)[1] - np.linalg.slogdet(S2)[1]) < 1e-8
    assert abs(np.trace(S1) - np.trace(S2)) < 1e-10


def test_irregular_kalman_matches_fixed_step_and_gaussian():
    m = _dev()
    d = Design.regular(15.0, 2.0)
    sim = m.simulate(d.n_epochs, d.dt, probe_every=12, seed=3)
    s_reg = {"y": sim["y"], "rating": sim["rating"]}
    t = (np.arange(d.n_epochs) + 1) * d.dt
    pidx = np.flatnonzero(np.isfinite(sim["rating"]))
    s_irr = {"t": t, "y": sim["y"], "t_probe": t[pidx], "rating": sim["rating"][pidx]}
    assert abs(kalman_loglik(m, [s_reg], d.dt) - kalman_loglik_irregular(m, [s_irr])) < 1e-8
    # with missing epochs and correlated noise: compare to the stacked Gaussian density
    mr = _dev(rho=0.4)
    sim2 = mr.simulate(40, d.dt, probe_every=7, seed=5)
    y = sim2["y"].copy(); y[5:9, 0] = np.nan; y[20, 1] = np.nan
    t2 = (np.arange(40) + 1) * d.dt
    p2 = np.flatnonzero(np.isfinite(sim2["rating"]))
    s2 = {"t": t2, "y": y, "t_probe": t2[p2], "rating": sim2["rating"][p2]}
    di = RDM.design_of(s2)
    mu, S = session_moments_irregular(mr, di)
    obs = []
    for i in range(40):
        if np.isfinite(y[i, 0]): obs.append(y[i, 0])
        if np.isfinite(y[i, 1]): obs.append(y[i, 1])
    obs += list(sim2["rating"][p2])
    L = np.linalg.cholesky(S)
    a = np.linalg.solve(L, np.array(obs) - mu)
    ll_gauss = -0.5 * (len(obs) * np.log(2 * np.pi) + 2 * np.log(np.diag(L)).sum() + a @ a)
    assert abs(kalman_filter_irregular(mr, s2) - ll_gauss) < 1e-7


def test_simulated_session_statistics():
    m = _dev()
    d = IrregularDesign(tuple((np.arange(3000) + 1) / 6), (True,) * 3000, (True,) * 3000, tuple(np.arange(10, 500, 1.7)))
    s = RDM.simulate_session(m, d, seed=1)
    P = m.stationary_cov()
    assert abs(np.var(s["y"][:, 0]) - (P[0, 0] + m.r_A ** 2)) < 0.15 * (P[0, 0] + m.r_A ** 2)
    assert np.isfinite(s["rating"]).all() and len(s["rating"]) == len(d.t_probe)


def test_behaviour_alignment_and_decoding(tmp_path):
    # synthetic events (offset 12.3 s): probe 1 keys 1,2,0 ; probe 2 keys 0,3,1 ; probe 3 keys 2,2,2
    beh = tmp_path / "sub01_info.txt"
    beh.write_text("MW question asked at time  50.0 second\n"
                   "Key 1 (code 1) pressed at time  53.0 seconds, status 1\n"
                   "Key 2 (code 2) pressed at time  55.0 seconds, status 2\n"
                   "Key 0 (code 0) pressed at time  57.0 seconds, status 3\n"
                   "MW question asked at time 150.0 second\n"
                   "Key 0 (code 0) pressed at time 153.0 seconds, status 1\n"
                   "Key 3 (code 3) pressed at time 155.0 seconds, status 2\n"
                   "Key 1 (code 1) pressed at time 157.0 seconds, status 3\n"
                   "MW question asked at time 250.0 second\n"
                   "Key 2 (code 2) pressed at time 253.0 seconds, status 1\n"
                   "Key 2 (code 2) pressed at time 255.0 seconds, status 2\n"
                   "Key 2 (code 2) pressed at time 257.0 seconds, status 3\n")
    ev = pd.DataFrame({"onset": [62.3, 65.3, 67.3, 162.3, 167.3, 169.3, 262.3, 265.3, 267.3, 269.3, 300.0],
                       "trial_type": ["stimulus", "response", "response", "stimulus", "response", "response",
                                      "stimulus", "response", "response", "response", "STATUS"],
                       "value": [128, 2, 4, 128, 8, 2, 128, 4, 4, 4, 254]})
    evp = tmp_path / "ev.tsv"
    ev.to_csv(evp, sep="\t", index=False)
    sess = {"sub": "sub-001", "ses": "ses-01", "group": "expert", "events": str(evp), "duration_s": 400.0}
    df, st = DS.decode_session(sess, str(beh))
    assert st["n_matched"] == 3 and abs(st["offset_s"] - 12.3) < 0.05
    assert st["key0_with_trigger"] == 0 and st["key0_total"] == 2 and st["code_agree"] == st["code_total"] == 7
    assert df["q1"].tolist() == [1, 0, 2] and df["q2"].tolist() == [2, 3, 2] and df["q3"].tolist() == [0, 1, 2]
    assert df["rating"].tolist() == [-1.0, -3.0, 0.0]
    # without behaviour: only the 3-response probe is decodable
    df2, st2 = DS.decode_session(sess, None)
    assert df2["q1"].tolist()[2] == 2 and pd.isna(df2["q1"].iloc[0]) and st2["source"] == "events"


def test_session_design_masks_probe_windows():
    probes = pd.DataFrame({"sub": ["sub-001"] * 2, "ses": ["ses-01"] * 2, "onset_s": [71.4, 125.6],
                           "resp_end_s": [79.8, np.nan], "rating": [1.0, np.nan]})
    sess = {"sub": "sub-001", "ses": "ses-01", "group": "expert", "duration_s": 200.0}
    d, info = DS.session_design(sess, probes)
    t = np.array(d.t_eeg) * 60
    mask = np.array(d.obs_A)
    assert info["n_epochs"] == 20 and len(d.t_probe) == 2
    # epochs ending at 80 and 90 s overlap [71.4, 82.8]; 130, 140, 150 overlap [125.6, 140.6]
    assert not mask[7] and not mask[8] and mask[6] and mask[9]
    assert not mask[12] and not mask[13] and not mask[14] and mask[15] and mask[11]
    d2, _ = DS.session_design(sess, probes, require_rating=True)
    assert len(d2.t_probe) == 1


@pytest.mark.skipif(not HAS_DS, reason="ds001787 metadata clone not present")
def test_real_metadata_sessions_and_decoding():
    ss = DS.sessions()
    assert len(ss) == 40 and sum(s["group"] == "expert" for s in ss) == 22
    assert all(s["bdf_md5"] and s["bdf_size"] for s in ss)
    beh = DS.behaviour_files()
    if beh:
        s = ss[0]
        df, st = DS.decode_session(s, beh.get((s["sub"], s["ses"])))
        assert st["n_matched"] >= 0.9 * st["n_probe_eeg"]


def test_recovery_on_real_like_designs():
    """Parameter recovery of M1-dev on irregular designs with gaps (fixed-step
    inference would be wrong here); loose tolerance, 3 short sessions."""
    rng = np.random.default_rng(0)
    designs = []
    for k in range(3):
        n = 150
        t = (np.arange(n) + 1) / 6
        mask = np.ones(n, bool)
        probes = []
        tp = 1.5
        while tp < t[-1] - 0.5:
            probes.append(round(tp, 1))
            mask &= ~((t > tp) & (t - 1 / 6 < tp + 0.15))
            tp += rng.uniform(1.0, 2.2)
        designs.append(IrregularDesign(tuple(t), tuple(mask), tuple(mask), tuple(probes)))
    truth = _dev()
    sims = RDM.simulate_sessions(truth, designs, seed=11)
    r = RDM.fit_sessions(sims, base=truth, n_starts=1, seed=0, starts=[{k: getattr(truth, k) for k in truth.FREE}])
    ll_true = kalman_loglik_irregular(truth, sims)
    assert r["loglik"] >= ll_true - 1e-6
    assert 0.4 < r["params"]["k_inh"] / truth.k_inh < 2.5
    se = RDM.se_at_mle(r["model"], sims)["crb"]
    assert all(np.isfinite(se[k]["rel_se"]) for k in truth.FREE)


def test_fit_boot_check_pipeline_on_synthetic(tmp_path, monkeypatch):
    """End-to-end on synthetic sessions (no EEG): fit -> boot -> check with fast settings."""
    from eeg import gates
    monkeypatch.setattr(gates, "GATE_DIR", str(tmp_path / "gates"))
    gates.record("E0", True, {}); gates.record("E1", True, {})
    truth = _dev()
    rng = np.random.default_rng(1)
    designs, metas = [], []
    for k in range(4):
        n = 120
        t = (np.arange(n) + 1) / 6
        probes = tuple(np.round(np.arange(1.2, t[-1] - 0.3, 1.6) + rng.uniform(-0.2, 0.2, len(np.arange(1.2, t[-1] - 0.3, 1.6))), 1))
        designs.append(IrregularDesign(tuple(t), (True,) * n, (True,) * n, probes))
        metas.append({"sub": f"sub-{k // 2:03d}", "ses": f"ses-{k % 2 + 1:02d}", "group": "expert" if k < 2 else "novice"})
    raw = RDM.simulate_sessions(truth, designs, seed=5, metas=metas)
    sessions, _ = RDM.normalise_sessions(raw)
    out = str(tmp_path / "res")
    fr = RDM.run_fit(seed=0, fast=True, output_dir=out, sessions=sessions, verbose=False)
    assert set(fr["pooled"]["params"]) == set(truth.FREE) and "profiles" in fr and "groups" in fr
    br = RDM.run_boot(seed=0, fast=True, n_boot=3, output_dir=out, sessions=sessions, verbose=False)
    assert len(br["boot_params"]) == 3 and set(br["ratio_rmse_over_se"]) == set(truth.FREE)
    gates.record("E2", True, {})
    cr = RDM.run_check(seed=0, fast=True, output_dir=out, sessions=sessions, verbose=False,
                       sensitivity_loaders={"v1": lambda: sessions[:3], "v2": lambda: sessions[1:]})
    assert "loso" in cr and cr["loso"]["n_probes"] > 0 and "a_M1_vs_N2" in cr["loso"]["delta_lpd"]
    assert cr["residuals"]["ljung_box_p_A_median"] > 0.01
