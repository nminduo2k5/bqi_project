import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bqi.bqi_algorithm import BQIModel, BQIConfig


def test_bqi_algorithm_runs_and_reduces_free_energy():
    cfg = BQIConfig(n_levels=3, dims=(6, 6, 4, 3), n_snis=32, d_snis=3, d_k=3, T=60, seed=0)
    model = BQIModel(cfg)
    o = np.random.default_rng(1).normal(0, 1, 6)
    result = model.run(o)
    assert result["free_energy_trace"][-1] <= result["free_energy_trace"][0]
    assert np.isfinite(result["F_final"])


def test_bqi_algorithm_attention_weights_normalised():
    cfg = BQIConfig(n_levels=2, dims=(4, 4, 3), n_snis=16, d_snis=3, d_k=3, T=20, seed=2)
    model = BQIModel(cfg)
    o = np.random.default_rng(3).normal(0, 1, 4)
    result = model.run(o)
    assert np.allclose(result["attention_weights"].sum(axis=-1), 1.0)


def test_bqi_algorithm_output_shapes():
    cfg = BQIConfig(n_levels=3, dims=(8, 8, 6, 4), n_snis=64, d_snis=4, d_k=4, T=40, seed=4)
    model = BQIModel(cfg)
    o = np.random.default_rng(5).normal(0, 1, 8)
    result = model.run(o)
    assert result["mu_star"].shape == (4,)
    assert result["perception"].shape == (4,)
