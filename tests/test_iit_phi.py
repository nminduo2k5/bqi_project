import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bqi.iit_phi import integrated_information_Phi, np_hardness_partition_count


def test_np_hardness_partition_count():
    assert np_hardness_partition_count(5) == 30
    assert np_hardness_partition_count(3) == 6


def test_phi_nonnegative():
    rng = np.random.default_rng(0)
    n = 5
    W = rng.normal(0, 1.5, (n, n))
    np.fill_diagonal(W, 0)
    state = rng.integers(0, 2, n)
    res = integrated_information_Phi(W, state)
    assert res["Phi"] >= 0
    for phi in res["mechanism_phis"].values():
        assert phi >= -1e-9


def test_disconnected_system_has_near_zero_phi():
    """A fully disconnected network (W=0) should have ~zero integrated
    information: cutting any partition changes nothing."""
    n = 4
    W = np.zeros((n, n))
    state = np.array([1, 0, 1, 0])
    res = integrated_information_Phi(W, state)
    assert res["Phi"] < 1e-6
