import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bqi.com_algorithm import COMConfig, run_com_algorithm


def test_com_algorithm_suppresses_dmn():
    cfg = COMConfig(T=90, d_mu=6, n_snis=30, d_snis=6, d_k=6, seed=1)
    res = run_com_algorithm(cfg)
    A_dmn = res["trace"]["A_dmn"]
    assert A_dmn[-1] < A_dmn[0]  # Phase 1 suppresses DMN


def test_com_algorithm_raises_phi():
    cfg = COMConfig(T=90, d_mu=6, n_snis=30, d_snis=6, d_k=6, seed=1)
    res = run_com_algorithm(cfg)
    Phi = res["trace"]["Phi"]
    assert Phi[-1] > Phi[0]  # Phase 2 raises Phi as DMN falls


def test_com_algorithm_qoc_finite_and_delta_s_positive():
    cfg = COMConfig(T=120, d_mu=8, n_snis=40, d_snis=8, d_k=8, seed=2)
    res = run_com_algorithm(cfg)
    assert np.isfinite(res["QoC_final"])
    assert res["delta_S"] > 0
