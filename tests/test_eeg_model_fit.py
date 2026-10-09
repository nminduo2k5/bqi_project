import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bqi.dmn_ode import StateSpaceDMN
from eeg.model_fit import kalman_loglik, kalman_loglik_dense


def test_sequential_kalman_matches_joint_update():
    m = StateSpaceDMN()
    sess = [m.simulate(120, 1 / 6, probe_every=10, seed=s) for s in range(3)]
    sess[0]["y"][7, 1] = np.nan
    sess[1]["y"][3, :] = np.nan
    assert abs(kalman_loglik(m, sess, 1 / 6) - kalman_loglik_dense(m, sess, 1 / 6)) < 1e-8


def test_true_parameters_beat_wrong_ones():
    m = StateSpaceDMN()
    sess = [m.simulate(270, 1 / 6, probe_every=12, seed=s) for s in range(6)]
    ll_true = kalman_loglik(m, sess, 1 / 6)
    for wrong in (dict(alpha=0.5), dict(beta=4.0), dict(A0=0.1), dict(k_inh=2.0)):
        assert kalman_loglik(m.with_params(**wrong), sess, 1 / 6) < ll_true
