import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bqi.predictive_coding import (free_energy, free_energy_gradient,
                                    gradient_descent_free_energy, HierarchicalBPC)


def _quadratic_problem(seed=0, n=5):
    rng = np.random.default_rng(seed)
    Pi_s = np.diag(rng.uniform(1, 2, n))
    Pi_o = np.diag(rng.uniform(1, 2, n))
    g = np.eye(n)
    prior_mean = np.zeros(n)
    o = rng.normal(0, 1, n)
    return Pi_s, Pi_o, g, prior_mean, o


def test_free_energy_nonnegative_at_prior():
    Pi_s, Pi_o, g, prior_mean, o = _quadratic_problem()
    F = free_energy(prior_mean, o, prior_mean, Pi_s, Pi_o, g)
    assert F >= 0


def test_gradient_matches_finite_difference():
    Pi_s, Pi_o, g, prior_mean, o = _quadratic_problem()
    mu = np.array([0.3, -0.2, 0.1, 0.4, -0.1])
    analytic = free_energy_gradient(mu, o, prior_mean, Pi_s, Pi_o, g)
    h = 1e-6
    numeric = np.zeros_like(mu)
    for i in range(len(mu)):
        mu_p, mu_m = mu.copy(), mu.copy()
        mu_p[i] += h
        mu_m[i] -= h
        numeric[i] = (free_energy(mu_p, o, prior_mean, Pi_s, Pi_o, g)
                      - free_energy(mu_m, o, prior_mean, Pi_s, Pi_o, g)) / (2 * h)
    assert np.allclose(analytic, numeric, atol=1e-4)


def test_gradient_descent_decreases_free_energy_monotonically():
    Pi_s, Pi_o, g, prior_mean, o = _quadratic_problem()
    mu0 = np.random.default_rng(1).normal(0, 1, 5)
    _, F_trace = gradient_descent_free_energy(mu0, o, prior_mean, Pi_s, Pi_o, g,
                                               eta=0.05, n_iter=100)
    assert np.all(np.diff(F_trace) <= 1e-8)


def test_hierarchical_bpc_energy_converges_toward_zero():
    bpc = HierarchicalBPC(n_levels=3, dims=[4, 4, 3, 2], seed=1)
    o = np.random.default_rng(2).normal(0, 1, 4)
    energies = bpc.run(o, n_steps=300, eta=0.05)
    assert energies[-1] < energies[0]
    assert energies[-1] < 1e-4
