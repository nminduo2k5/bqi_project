import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bqi.meta_analysis import random_effects_meta_analysis, pooled_percent_change, STUDIES_REFERENCE


def test_pooled_d_in_plausible_range():
    res = random_effects_meta_analysis()
    assert 1.0 < res["pooled_d"] < 2.5
    assert res["k_studies"] == 16


def test_i_squared_between_0_and_100():
    res = random_effects_meta_analysis()
    assert 0.0 <= res["I2_percent"] <= 100.0


def test_pooled_percent_change_is_negative_reduction():
    res = pooled_percent_change()
    assert res["pooled_delta_pct"] < 0  # DMN activity decreases with meditation


def test_ci_contains_pooled_estimate():
    res = random_effects_meta_analysis()
    lo, hi = res["ci"]
    assert lo <= res["pooled_d"] <= hi
