import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bqi.hopfield import (store_patterns_classic, retrieve_classic, retrieve_modern,
                           random_patterns, corrupt, theoretical_capacity_classic,
                           theoretical_capacity_modern, retrieval_error_experiment)


def test_classic_hopfield_perfect_recall_below_capacity():
    n = 100
    P = 3  # well below 0.14*100=14
    patterns = random_patterns(P, n, seed=0)
    W = store_patterns_classic(patterns)
    probe = corrupt(patterns[0], flip_fraction=0.05, seed=1)
    recon = retrieve_classic(W, probe)
    assert np.mean(recon != patterns[0]) < 0.05


def test_modern_hopfield_outperforms_classic_at_high_load():
    n = 40
    P = 200  # far beyond classic capacity, well within modern capacity
    patterns = random_patterns(P, n, seed=0)
    probe = corrupt(patterns[5], flip_fraction=0.1, seed=1)

    W = store_patterns_classic(patterns)
    recon_classic = retrieve_classic(W, probe)
    err_classic = np.mean(recon_classic != patterns[5])

    recon_modern = retrieve_modern(patterns, probe, beta=8.0)
    err_modern = np.mean(recon_modern != patterns[5])

    assert err_modern <= err_classic


def test_theoretical_capacities_ordering():
    n = 50
    assert theoretical_capacity_modern(n) > theoretical_capacity_classic(n)


def test_retrieval_error_experiment_returns_dataframe():
    df = retrieval_error_experiment(n=20, pattern_counts=[1, 5, 20, 100], trials_per_count=3, seed=0)
    assert set(["n_patterns", "classic_error_rate", "modern_error_rate"]).issubset(df.columns)
    assert len(df) == 4
