"""
Integrated Information Theory and the Phi measure, paper Sec. 5.1 (eq. 22-23).

    phi(M, P) = min( EMD(p^P(X_past|M), p(X_past|M)), EMD(p^P(X_future|M), p(X_future|M)) )
    phi(M)    = min_P phi(M, P)
    Phi(X)    = sum_{M subset X, phi(M)>0} phi(M) * 1[M in MIP-complex]

Exact Phi requires evaluating 2^n - 2 partitions (Proposition, NP-hardness),
so this module only tackles small systems (n <= ~8-10 binary elements),
exactly as intended for a demonstration / validation of the theory rather
than a production-scale consciousness meter.

We build a discrete Markov-chain "mechanism" over n binary nodes from a
weighted directed connectivity matrix, then brute-force all bipartitions to
find the Minimum Information Partition (MIP) and the resulting phi.
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.stats import wasserstein_distance


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class TPMSystem:
    """A small binary system with a transition probability matrix (TPM)
    induced by a weighted connectivity matrix W (n x n), used to instantiate
    the "five-element neuronal mechanism network" example (Fig phi decomposition,
    Phi = 2.31 nats)."""

    def __init__(self, W: np.ndarray, bias: np.ndarray | None = None):
        self.W = W
        self.n = W.shape[0]
        self.bias = np.zeros(self.n) if bias is None else bias

    def next_state_probs(self, state: np.ndarray) -> np.ndarray:
        """P(each node ON at t+1 | state at t) via independent sigmoid units."""
        s = 2 * state - 1  # {0,1} -> {-1,1}
        activ = self.W @ s + self.bias
        return sigmoid(activ)

    def full_tpm(self) -> np.ndarray:
        """Return the full 2^n x 2^n transition probability matrix (node-independent)."""
        states = list(itertools.product([0, 1], repeat=self.n))
        tpm = np.zeros((2 ** self.n, 2 ** self.n))
        for i, s in enumerate(states):
            p_on = self.next_state_probs(np.array(s))
            for j, s_next in enumerate(states):
                p = 1.0
                for k in range(self.n):
                    p *= p_on[k] if s_next[k] == 1 else (1 - p_on[k])
                tpm[i, j] = p
        return tpm


def marginal_future_distribution(tpm: np.ndarray, current_index: int) -> np.ndarray:
    """p(X_future | current state) = row of the TPM."""
    return tpm[current_index]


def _cut_tpm_row(sys: "TPMSystem", current_state: np.ndarray, part_a: tuple, part_b: tuple) -> np.ndarray:
    """Causal 'cut': sever directed influence *between* part_a and part_b
    (zero the corresponding off-diagonal blocks of W) and recompute the
    resulting joint future distribution p^P(X_future | current_state) over
    all 2^n outcomes. This directly implements "severing the connections
    specified by partition P", eq. (22)'s partitioned distribution, rather
    than a naive independence assumption -- so cutting an informative
    connection actually changes the induced distribution."""
    n = sys.n
    W_cut = sys.W.copy()
    for i in part_a:
        for j in part_b:
            W_cut[i, j] = 0.0
            W_cut[j, i] = 0.0
    sys_cut = TPMSystem(W_cut, sys.bias)
    p_on = sys_cut.next_state_probs(current_state)
    states = list(itertools.product([0, 1], repeat=n))
    row = np.zeros(len(states))
    for j, s_next in enumerate(states):
        p = 1.0
        for k in range(n):
            p *= p_on[k] if s_next[k] == 1 else (1 - p_on[k])
        row[j] = p
    return row


def phi_for_mechanism(sys: "TPMSystem", tpm: np.ndarray, n: int, mechanism: tuple,
                       current_index: int, current_state: np.ndarray) -> float:
    """Eq. (22): phi(M) = min over bipartitions P of the EMD between the
    partitioned (causally cut) and unpartitioned future distributions."""
    unpartitioned = marginal_future_distribution(tpm, current_index)
    support = np.arange(len(unpartitioned))

    best_phi = np.inf
    m = list(mechanism)
    if len(m) < 2:
        return 0.0
    for r in range(1, len(m)):
        for part_a in itertools.combinations(m, r):
            part_b = tuple(x for x in m if x not in part_a)
            partitioned = _cut_tpm_row(sys, current_state, part_a, part_b)
            emd = wasserstein_distance(support, support, unpartitioned, partitioned)
            best_phi = min(best_phi, emd)
    return float(best_phi)


def integrated_information_Phi(W: np.ndarray, current_state: np.ndarray, scale: float = 1.0) -> dict:
    """Eq. (23): Phi(X) = sum over mechanisms M with phi(M) > 0 that belong to
    the Minimum-Information-Partition complex.

    Brute-forces all non-empty subsets (mechanisms) of the n nodes -- feasible
    for n <= ~8 -- and reports the whole-system phi as our Phi estimate
    (the standard IIT convention that Phi(X) itself is phi computed for the
    maximal mechanism M=X at its MIP), plus every constituent mechanism's phi
    for inspection / the "phi decomposition" figure.
    """
    n = W.shape[0]
    sys = TPMSystem(W)
    tpm = sys.full_tpm()
    current_index = int("".join(str(b) for b in current_state), 2)

    mechanism_phis = {}
    for r in range(2, n + 1):
        for M in itertools.combinations(range(n), r):
            mechanism_phis[M] = phi_for_mechanism(sys, tpm, n, M, current_index, current_state)

    whole_system = tuple(range(n))
    Phi_value = mechanism_phis[whole_system] * scale
    return {
        "Phi": Phi_value,
        "mechanism_phis": mechanism_phis,
        "tpm": tpm,
        "current_index": current_index,
    }


def np_hardness_partition_count(n: int) -> int:
    """Proposition: exact Phi requires evaluating 2^n - 2 partitions."""
    return 2 ** n - 2
