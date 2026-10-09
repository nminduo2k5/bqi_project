import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bqi.attention import bqi_attention, softmax, MultiHeadBQIAttention


def test_softmax_rows_sum_to_one():
    x = np.random.default_rng(0).normal(0, 1, (5, 7))
    p = softmax(x, axis=-1)
    assert np.allclose(p.sum(axis=-1), 1.0)
    assert np.all(p >= 0)


def test_attention_output_shape_and_weight_normalisation():
    rng = np.random.default_rng(0)
    Q = rng.normal(0, 1, (3, 8))
    K = rng.normal(0, 1, (10, 8))
    V = rng.normal(0, 1, (10, 6))
    A, weights = bqi_attention(Q, K, V)
    assert A.shape == (3, 6)
    assert weights.shape == (3, 10)
    assert np.allclose(weights.sum(axis=-1), 1.0)


def test_multihead_attention_output_shape():
    mha = MultiHeadBQIAttention(d_model=16, n_heads=4, d_snis=12, seed=0)
    phi_e = np.random.default_rng(0).normal(0, 1, (5, 16))
    S = np.random.default_rng(1).normal(0, 1, (20, 12))
    out, head_weights = mha.forward(phi_e, S)
    assert out.shape == (5, 16)
    assert len(head_weights) == 4
    for w in head_weights:
        assert np.allclose(w.sum(axis=-1), 1.0)
