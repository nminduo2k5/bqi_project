import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bqi.pci import lempel_ziv_complexity
from eeg.axis import build_axis, project
from eeg.metrics.coalition import ace
from eeg.metrics.lzc import longest_previous_factor, lz76_phrase_count, lzc, normalised_lz
from eeg.metrics.network import fiedler, proportional_threshold, wpli
from eeg.metrics.phi_ar import phi_ar
from eeg.metrics.spectral import aperiodic_fit, psd, relative_band_power
from eeg.stats import fdr_bh, paired_dz, tost_paired
from eeg.surrogate import phase_randomize


def _brute_lpf(a):
    out = []
    for i in range(len(a)):
        best = 0
        for j in range(i):
            k = 0
            while i + k < len(a) and a[i + k] == a[j + k]:
                k += 1
            best = max(best, k)
        out.append(best)
    return out


def test_lpf_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(100):
        a = rng.integers(0, int(rng.integers(1, 4)), int(rng.integers(1, 50)))
        assert list(longest_previous_factor(a)) == _brute_lpf(a)


def test_fast_lz76_matches_kaspar_schuster():
    rng = np.random.default_rng(1)
    for _ in range(100):
        n = int(rng.integers(3, 250))
        s = "".join("1" if x else "0" for x in rng.random(n) < rng.uniform(0.1, 0.9))
        assert lz76_phrase_count(s) == lempel_ziv_complexity(s)


def test_normalised_lz_is_about_one_for_random_and_low_for_periodic():
    rng = np.random.default_rng(2)
    assert 0.9 < normalised_lz(rng.integers(0, 2, 20000)) < 1.1
    assert normalised_lz(np.tile([0, 1, 1, 0], 5000)) < 0.05


def test_lzc_higher_for_noise_than_shared_slow_envelope():
    """Schartner LZc measures amplitude-envelope diversity: a carrier whose
    envelope is slowly modulated and shared across channels is simple; a pure
    constant-amplitude sinusoid is NOT (its envelope binarises to noise)."""
    rng = np.random.default_rng(3)
    t = np.arange(2500) / 250
    env = 1 + 0.8 * np.sin(2 * np.pi * 0.4 * t)
    am = (env * np.sin(2 * np.pi * 10 * t))[None, :] * rng.uniform(0.5, 1.5, (8, 1))
    noise = rng.normal(size=(8, 2500))
    assert lzc(noise) > lzc(am + 0.05 * noise) + 0.5


def test_phase_randomize_preserves_power_spectrum():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(4, 1000)).cumsum(axis=1)
    S = phase_randomize(X, rng=5)
    assert np.allclose(np.abs(np.fft.rfft(X, axis=1)), np.abs(np.fft.rfft(S, axis=1)))
    assert not np.allclose(X, S)


def test_aperiodic_exponent_recovered():
    rng = np.random.default_rng(6)
    n, fs = 25000, 250.0
    f = np.fft.rfftfreq(n, 1 / fs)
    amp = np.zeros_like(f)
    amp[1:] = f[1:] ** -1.0  # PSD ~ 1/f^2
    X = np.fft.irfft(amp * np.exp(1j * rng.uniform(0, 2 * np.pi, (3, len(f)))), n=n, axis=1)
    fr, P = psd(X, fs)
    assert abs(aperiodic_fit(fr, P)["exponent_mean"] - 2.0) < 0.15
    rb = relative_band_power(fr, P)
    assert rb["delta"] > rb["alpha"] > rb["beta"] * 0.1


def test_wpli_zero_for_zero_lag_and_high_for_lagged_coupling():
    t = np.arange(5000) / 250
    base = np.exp(1j * 2 * np.pi * 6 * t)
    rng = np.random.default_rng(7)
    noise = lambda: 0.3 * (rng.normal(size=t.size) + 1j * rng.normal(size=t.size))
    Z0 = np.vstack([base + noise(), base + noise()])                   # zero lag
    Z1 = np.vstack([base + noise(), base * np.exp(1j * 0.8) + noise()])  # consistent lag
    assert wpli(Z0)[0, 1] < 0.3
    assert wpli(Z1)[0, 1] > 0.9


def test_fiedler_zero_when_disconnected():
    A = np.zeros((4, 4))
    A[0, 1] = A[1, 0] = A[2, 3] = A[3, 2] = 1
    assert fiedler(A) == pytest.approx(0.0, abs=1e-12)
    W = np.ones((5, 5)) - np.eye(5)
    assert fiedler(proportional_threshold(W, 1.0)) == pytest.approx(5.0)


def test_phi_ar_zero_for_independent_channels():
    rng = np.random.default_rng(8)
    X = np.zeros((4, 4000))
    for t in range(1, 4000):
        X[:, t] = 0.6 * X[:, t - 1] + rng.normal(size=4)
    assert phi_ar(X)["phi_ar"] < 0.01


def test_ace_lower_for_synchronised_amplitude():
    rng = np.random.default_rng(9)
    env = np.abs(np.sin(2 * np.pi * 0.5 * np.arange(2500) / 250))
    sync = env[None, :] * rng.normal(size=(6, 2500)) * 0.1 + env[None, :]
    indep = rng.normal(size=(6, 2500))
    assert ace(sync, rng=0) < ace(indep, rng=0)


def test_axis_projection_units():
    u = np.array([1.0, 0.0, 0.0])
    ref = np.tile(2 * u, (10, 1))
    ax = build_axis(ref, ref)
    p = project(np.array([[1.0, 0.0, 0.0], [0.0, 3.0, 0.0]]), ax)
    assert np.allclose(p["c"], [0.5, 0.0])
    assert np.allclose(p["residual_norm"], [0.0, 3.0])


def test_stats_helpers():
    rng = np.random.default_rng(10)
    a = rng.normal(1.0, 1.0, 30)
    b = a - 0.8 + rng.normal(0, 0.3, 30)
    r = paired_dz(a, b, n_boot=2000, rng=0)
    assert r["ci"][0] > 0
    # differences symmetric around zero -> equivalent within |d_z| < 0.5
    x = rng.normal(0, 1, 100)
    diffs = np.concatenate([np.abs(rng.normal(0, 1, 50)), -np.abs(rng.normal(0, 1, 50))])
    diffs -= diffs.mean()
    assert tost_paired(x + diffs, x, bound_dz=0.5)["p_tost"] < 0.05
    assert tost_paired(x + diffs + 2.0, x, bound_dz=0.5)["p_tost"] > 0.5
    rej, adj = fdr_bh(np.array([0.001, 0.02, 0.04, 0.5]))
    assert list(rej) == [True, True, False, False]
    assert np.all(adj >= np.array([0.001, 0.02, 0.04, 0.5]))


def test_common_channel_mapping_on_egi_net():
    mne = pytest.importorskip("mne")
    from eeg.preprocess import EGI129_1020, load_config, map_common_channels
    targets = load_config()["channels_common"]
    mont = mne.channels.make_standard_montage("GSN-HydroCel-129")
    keep = [c for c in mont.ch_names if c not in ("E45", "E108")]
    ep = mne.EpochsArray(np.zeros((1, len(keep), 50)), mne.create_info(keep, 250.0, "eeg"), verbose="error")
    ep.set_montage(mont, on_missing="ignore")
    m = map_common_channels(ep, targets)
    assert set(m) == set(targets) and len(set(m.values())) == len(targets)
    assert all(m[t] == EGI129_1020[t] for t in targets if t not in ("T7", "T8"))
    assert m["T7"] not in ("E45",) and m["T8"] not in ("E108",)


def test_common_channel_mapping_handles_old_1020_names():
    mne = pytest.importorskip("mne")
    from eeg.preprocess import load_config, map_common_channels
    names = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ", "C4", "T4", "T5", "P3", "PZ", "P4",
             "T6", "O1", "O2"]
    ep = mne.EpochsArray(np.zeros((1, 19, 50)), mne.create_info(names, 250.0, "eeg"), verbose="error")
    m = map_common_channels(ep, load_config()["channels_common"])
    assert m["T7"] == "T3" and m["P8"] == "T6" and m["Fp1"] == "FP1"
