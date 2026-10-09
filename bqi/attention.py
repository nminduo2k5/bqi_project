"""
Attention-based SNIS search, paper Sec. 3.3 (Definition 3.4, eq. 9-11).

    Sigma(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

and multi-head attention over H parallel "cortical processing streams".
"""
from __future__ import annotations
import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def bqi_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eq. (9): Sigma(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V.

    Q: (n_q, d_k), K: (n_s, d_k), V: (n_s, d_v)
    Returns (A, weights) where A is (n_q, d_v) and weights is (n_q, n_s).
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    A = weights @ V
    return A, weights


def project_qkv(phi_e: np.ndarray, S: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray,
                 W_V: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eq. (10)-(12): Q = W_Q phi_E(B,C), K = W_K S, V = W_V S."""
    Q = phi_e @ W_Q
    K = S @ W_K
    V = S @ W_V
    return Q, K, V


class MultiHeadBQIAttention:
    """Multi-head SNIS search, eq. (13): MH(Q,K,V) = Concat(head_1..head_H) W^O.

    Each head represents a distinct cortical processing stream (visual,
    auditory, semantic, proprioceptive, ...).
    """

    def __init__(self, d_model: int, n_heads: int, d_snis: int, seed: int = 0):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(d_model)
        self.W_Q = [rng.normal(0, scale, (d_model, self.d_head)) for _ in range(n_heads)]
        self.W_K = [rng.normal(0, scale, (d_snis, self.d_head)) for _ in range(n_heads)]
        self.W_V = [rng.normal(0, scale, (d_snis, self.d_head)) for _ in range(n_heads)]
        self.W_O = rng.normal(0, scale, (d_model, d_model))

    def forward(self, phi_e: np.ndarray, S: np.ndarray):
        """phi_e: (n_q, d_model) query encoding. S: (n_s, d_snis) SNIS proxy."""
        head_outputs = []
        head_weights = []
        for h in range(self.n_heads):
            Q = phi_e @ self.W_Q[h]
            K = S @ self.W_K[h]
            V = S @ self.W_V[h]
            A, w = bqi_attention(Q, K, V)
            head_outputs.append(A)
            head_weights.append(w)
        concat = np.concatenate(head_outputs, axis=-1)
        out = concat @ self.W_O
        return out, head_weights
