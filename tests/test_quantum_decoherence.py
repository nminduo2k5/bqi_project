import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bqi.quantum_decoherence import (von_neumann_entropy, random_density_matrix,
                                      simulate_entropy_dynamics, orch_or_collapse_time)


def test_pure_state_has_zero_entropy():
    psi = np.array([1, 0], dtype=complex)
    rho = np.outer(psi, psi.conj())
    assert abs(von_neumann_entropy(rho)) < 1e-10


def test_maximally_mixed_state_has_max_entropy():
    dim = 4
    rho = np.eye(dim) / dim
    S = von_neumann_entropy(rho)
    assert abs(S - np.log(dim)) < 1e-10


def test_entropy_increases_under_decoherence():
    res = simulate_entropy_dynamics(dim=4, gamma=2.0, T=3.0, dt=0.01, purity0=0.98, seed=0)
    assert res["S"][-1] > res["S"][0]
    assert res["S"][-1] <= res["S_max"] + 1e-6


def test_faster_decoherence_reaches_higher_entropy_sooner():
    res_slow = simulate_entropy_dynamics(dim=4, gamma=0.5, T=1.0, dt=0.01, seed=0)
    res_fast = simulate_entropy_dynamics(dim=4, gamma=10.0, T=1.0, dt=0.01, seed=0)
    assert res_fast["S"][-1] > res_slow["S"][-1]


def test_orch_or_collapse_time_positive():
    tau = orch_or_collapse_time(mass_kg=1e-23, delta_x_m=1e-9)
    assert tau > 0
