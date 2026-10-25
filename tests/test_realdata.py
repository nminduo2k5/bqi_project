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


def test_starting_points_with_unequal_session_lengths():
    # real sessions have 120-361 epochs; moment_starts once used vstack and crashed
    m = _dev()
    sessions = []
    for i, n in enumerate((120, 272, 361)):
        d = IrregularDesign(tuple((np.arange(n) + 1) / 6), (True,) * n, (True,) * n, tuple(np.arange(2, n / 6 - 1, 2.5)))
        sessions.append(RDM.simulate_session(m, d, seed=i))
    pts = RDM.starting_points(sessions, m, 4, 0)
    assert len(pts) == 4 and all(np.isfinite(list(p.values())).all() for p in pts)


def _sim_sessions(m, lengths=(240, 240, 240, 240), seed=0):
    out = []
    for i, n in enumerate(lengths):
        d = IrregularDesign(tuple((np.arange(n) + 1) / 6), (True,) * n, (True,) * n, tuple(np.arange(1.3, n / 6 - 1, 1.7)))
        s = RDM.simulate_session(m, d, seed=seed + i)
        out.append({**s, "sub": f"sub-{i:03d}", "group": "expert" if i % 2 == 0 else "novice", "rating_offset": 0.5})
    return out


def test_filtered_states_and_observed_information():
    m = _dev()
    sessions = _sim_sessions(m)
    ll, info = kalman_filter_irregular(m, sessions[0], return_innovations=True)
    F = info["filtered"]
    n_ev = len(sessions[0]["t"]) + len(sessions[0]["t_probe"])
    assert F.shape == (n_ev, 7) and np.all(np.diff(F[:, 0]) >= 0)
    assert np.allclose(F[-1, 2:4], info["state"]) and np.allclose(F[-1, [4, 5, 6]], info["cov"])
    assert abs(ll - kalman_filter_irregular(m, sessions[0])) < 1e-9
    # correctly specified data: observed and expected information agree (sampling noise only)
    obs = RDM.observed_information_se(m, sessions)
    fis = RDM.se_at_mle(m, sessions)["crb"]
    assert obs["positive_definite"]
    for k in ("k_inh", "k_d", "r_A", "r_Phi"):
        assert 0.5 < obs["params"][k]["rel_se"] / fis[k]["rel_se"] < 2.0, k


def test_snr_and_posterior_predictive_ratings():
    assert abs(RDM.snr(RDM.nominal_model())["A"] - 1.581) < 1e-3
    assert abs(RDM.snr(RDM.nominal_at_snr(1.11))["A"] - 1.11) < 1e-9
    m = _dev()
    sessions = _sim_sessions(m, lengths=(150, 150, 150, 150), seed=3)
    ppc = RDM.posterior_predictive_ratings(m, sessions, n_rep=5, seed=0)
    lp = ppc["level_prop"]
    assert ppc["levels"] == list(range(-3, 4))
    assert abs(sum(lp["observed"]) - 1) < 1e-9 and abs(sum(lp["sim_mean"]) - 1) < 1e-9
    assert all(lo <= hi for lo, hi in zip(lp["sim_lo"], lp["sim_hi"]))
    assert np.isfinite(ppc["lag1_autocorr_median"]["sim_mean"]) and np.isfinite(ppc["sd_expert_minus_novice"]["observed"])


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


def test_epoch_grid_flags_and_adaptive_rule():
    """Locked mask = probe windows + 150 uV on any channel; adaptive rule only
    rejects epochs that are outliers within the session (feature channels)."""
    import mne
    from eeg import realdata_features as RF
    sf, n_ep = 256.0, 30
    names = mne.channels.make_standard_montage("biosemi64").ch_names
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, int(n_ep * 10 * sf))) * 40e-6        # high baseline: ptp ~ 250-300 uV everywhere
    X[names.index("Pz"), int(125 * sf):int(125.2 * sf)] += 2e-3         # one short huge artifact in epoch 12 (120-130 s)
    raw = mne.io.RawArray(X, mne.create_info(names, sf, "eeg"), verbose="error")
    probes = pd.DataFrame({"onset_s": [101.0], "resp_end_s": [105.0]})
    common = [names.index(c) for c in ("Fz", "Cz", "Pz", "O1", "O2")]
    t_end, data, mask, rs = RF.epoch_grid(raw, n_ep * 10.0, probes, common)
    assert rs["amp_locked"].all()                           # fixed 150 uV rejects every epoch of this session
    assert rs["amp_adaptive"][12] and rs["amp_adaptive"].sum() == 1
    assert not rs["probe_mask"][10] and rs["probe_mask"].sum() == len(t_end) - 1
    assert not mask.any()


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
    g = fr["groups"]
    assert {"E_H5", "lr_k_inh_1df", "log_k_inh_difference"} <= set(g) and g["lr_k_inh_1df"]["stat"] >= -1e-6
    assert 0 <= g["lr_k_inh_1df"]["p"] <= 1 and 0 <= g["lr_all_params"]["p"] <= 1
    # standalone group step reuses the saved group fits and gives the same difference
    g2 = RDM.run_groups(seed=0, fast=True, output_dir=out, sessions=sessions, verbose=False)
    assert abs(g2["E_H5"]["diff"] - g["E_H5"]["diff"]) < 1e-9
    br = RDM.run_boot(seed=0, fast=True, n_boot=3, output_dir=out, sessions=sessions, verbose=False)
    assert len(br["boot_params"]) == 3 and set(br["ratio_rmse_over_se"]) == set(truth.FREE)
    gates.record("E2", True, {})
    cr = RDM.run_check(seed=0, fast=True, output_dir=out, sessions=sessions, verbose=False,
                       sensitivity_loaders={"v1": lambda: sessions[:3], "v2": lambda: sessions[1:]})
    assert "loso" in cr and cr["loso"]["n_probes"] > 0 and "a_M1_vs_N2" in cr["loso"]["delta_lpd"]
    assert cr["residuals"]["ljung_box_p_A_median"] > 0.01
