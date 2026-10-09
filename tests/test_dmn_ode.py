import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bqi.dmn_ode import (ODE_PARAMS_REFERENCE, steady_states, steady_states_paper, simulate,
                          generate_ode_param_dataset, StateSpaceDMN)


def test_steady_states_satisfy_the_ode():
    """Corrected closed form: plugging (A*, Phi*, QoC*) into eq. (25)-(27) with
    xi = 0 gives zero derivatives; the paper's eq. (55)-(57) do not."""
    P = ODE_PARAMS_REFERENCE
    from bqi.dmn_ode import _rhs
    zero = lambda t: 0.0
    one = lambda t: 1.0
    ss = steady_states(P)
    d = _rhs(0, [ss["A_DMN_star"], ss["Phi_star"], ss["QoC_star"]], P, zero, one)
    assert np.allclose(d, 0, atol=1e-12)
    sp = steady_states_paper(P)
    d_paper = _rhs(0, [sp["A_DMN_star"], sp["Phi_star"], sp["QoC_star"]], P, zero, one)
    assert not np.allclose(d_paper, 0, atol=1e-3)


def test_long_simulation_qoc_decays_to_zero():
    sol = simulate(ODE_PARAMS_REFERENCE, t_span=(0, 400), n_points=4000, seed=1)
    assert abs(np.mean(sol["QoC"][-500:])) < 0.1


def test_state_space_discretisation_matches_stationary_moments():
    m = StateSpaceDMN(k_inh=0.5, k_d=0.3, k_int=0.3)
    F, c, Q = m.discretize(dt=0.5)
    mu = m.stationary_mean()
    assert np.allclose(F @ mu + c, mu)
    P = m.stationary_cov()
    assert np.allclose(F @ P @ F.T + Q, P, atol=1e-10)


def test_state_space_qoc_has_nonzero_steady_state():
    m = StateSpaceDMN()
    ss = m.steady_states()
    assert abs(ss["QoC_star"]) > 0.1
    sim = m.simulate(6000, dt=1 / 6, probe_every=None, seed=0)
    assert abs(sim["QoC"][1000:].mean() - ss["QoC_star"]) < 0.15


def test_meditator_suppresses_dmn_more_than_control():
    sol_ctrl = simulate(ODE_PARAMS_REFERENCE, meditator=False, seed=0)
    sol_med = simulate(ODE_PARAMS_REFERENCE, meditator=True, seed=0)
    assert sol_med["A_DMN"][-1] < sol_ctrl["A_DMN"][-1]


def test_meditator_has_higher_phi_than_control():
    sol_ctrl = simulate(ODE_PARAMS_REFERENCE, meditator=False, seed=0)
    sol_med = simulate(ODE_PARAMS_REFERENCE, meditator=True, seed=0)
    assert sol_med["Phi"][-1] > sol_ctrl["Phi"][-1]


def test_parameter_dataset_recovers_population_mean():
    df = generate_ode_param_dataset(n_subjects=2000, seed=0)
    assert abs(df["k_inh"].mean() - ODE_PARAMS_REFERENCE["k_inh"]) < 0.01
    assert abs(df["alpha"].mean() - ODE_PARAMS_REFERENCE["alpha"]) < 0.2
