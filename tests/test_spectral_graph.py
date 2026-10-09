import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bqi.spectral_graph import (build_small_world_brain_graph, fiedler_value, small_world_index,
                                 global_efficiency, meditation_graph_transform, simulate_kuramoto,
                                 simulate_kuramoto_graph, graph_adjacency)


def test_fiedler_value_positive_for_connected_graph():
    G = build_small_world_brain_graph(n_nodes=50, k=6, p_rewire=0.1, seed=0)
    assert fiedler_value(G) > 0


def test_small_world_index_greater_than_one():
    G = build_small_world_brain_graph(n_nodes=60, k=6, p_rewire=0.1, seed=0)
    sigma = small_world_index(G, n_random=3)
    assert sigma > 1.0  # small-world networks have sigma >> 1


def test_meditation_transform_increases_global_efficiency():
    G = build_small_world_brain_graph(n_nodes=60, k=6, p_rewire=0.1, seed=0)
    G2 = meditation_graph_transform(G, boost_fraction=0.1, seed=1)
    assert global_efficiency(G2) >= global_efficiency(G)


def test_kuramoto_order_parameter_bounded():
    r = simulate_kuramoto(n_oscillators=30, coupling_K=2.0, T=5.0, dt=0.02, seed=0)
    assert (r >= 0).all() and (r <= 1.0001).all()


def test_kuramoto_strong_coupling_synchronises_more_than_weak():
    r_weak = simulate_kuramoto(n_oscillators=30, coupling_K=0.1, T=8.0, dt=0.02, seed=0)
    r_strong = simulate_kuramoto(n_oscillators=30, coupling_K=5.0, T=8.0, dt=0.02, seed=0)
    assert r_strong[-50:].mean() > r_weak[-50:].mean()


def test_graph_kuramoto_reduces_to_mean_field_for_complete_coupling():
    r_graph = simulate_kuramoto_graph(np.ones((25, 25)), 2.0, T=2.0, dt=0.02, seed=3)
    r_mf = simulate_kuramoto(25, 2.0, T=2.0, dt=0.02, seed=3)
    assert np.allclose(r_graph, r_mf, atol=1e-8)


def test_graph_kuramoto_synchrony_increases_with_coupling():
    W = graph_adjacency(build_small_world_brain_graph(60, 6, 0.1, seed=0))
    r = [simulate_kuramoto_graph(W, K, T=15.0, dt=0.02, seed=1)[-100:].mean() for K in (1.0, 60.0)]
    assert r[1] > r[0] + 0.5
