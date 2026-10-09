"""
Lempel-Ziv complexity for EEG (Q1_pipeline.md Sec 5.1).

LZ76 phrase counting via the Longest-Previous-Factor (LPF) array: phrase k
starting at position p has length LPF[p] + 1, where LPF[p] is the length of
the longest prefix of s[p:] that also starts at some earlier position (the
copy may overlap p, as in Kaspar & Schuster 1987). LPF is built from a suffix
array (prefix doubling in NumPy) and Kasai's LCP array, giving the same count
as `bqi.pci.lempel_ziv_complexity` in O(n log^2 n) instead of O(n^2).

Multivariate LZc follows Schartner et al. (2015): each channel is binarised
at the mean of its Hilbert amplitude envelope, channels are interleaved time
point by time point, and the phrase count is normalised by its value for a
random sequence with the same length and symbol entropy.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


# =============================================================================
# Suffix array, LCP, LPF
# =============================================================================
def suffix_array(s: np.ndarray) -> np.ndarray:
    """Suffix array of an integer sequence by prefix doubling (stable lexsort)."""
    n = len(s)
    rank = np.unique(s, return_inverse=True)[1].astype(np.int64)
    idx = np.arange(n)
    k = 1
    while True:
        second = np.full(n, -1, dtype=np.int64)
        second[: n - k] = rank[k:] if k < n else second[:0]
        sa = np.lexsort((second, rank))
        r, s2 = rank[sa], second[sa]
        diff = np.empty(n, dtype=bool)
        diff[0] = True
        diff[1:] = (r[1:] != r[:-1]) | (s2[1:] != s2[:-1])
        new_rank = np.empty(n, dtype=np.int64)
        new_rank[sa] = np.cumsum(diff) - 1
        rank = new_rank
        if rank.max() == n - 1:
            return sa
        k *= 2
        if k >= n:
            return np.argsort(rank, kind="stable")
    return idx  # pragma: no cover


def lcp_array(s: np.ndarray, sa: np.ndarray) -> np.ndarray:
    """Kasai et al. (2001): lcp[r] = LCP(suffix sa[r-1], suffix sa[r]), lcp[0] = 0."""
    n = len(s)
    rank = np.empty(n, dtype=np.int64)
    rank[sa] = np.arange(n)
    lcp = np.zeros(n, dtype=np.int64)
    sl = s.tolist()
    sal = sa.tolist()
    rl = rank.tolist()
    h = 0
    for i in range(n):
        r = rl[i]
        if r > 0:
            j = sal[r - 1]
            while i + h < n and j + h < n and sl[i + h] == sl[j + h]:
                h += 1
            lcp[r] = h
            if h > 0:
                h -= 1
        else:
            h = 0
    return lcp


def _nearest_smaller_pass(sal: list, link_lcp, order, lpf: list) -> None:
    """One sweep over suffix-array ranks in `order`. For each rank r, find the
    nearest already-visited rank whose text position is smaller and the minimum
    LCP across the ranks in between (monotone stack, O(n) amortised). Each stack
    entry is [position, min LCP between it and the entry above it]."""
    stack: list[list[int]] = []
    for r in order:
        p = sal[r]
        carry = link_lcp(r)
        while stack and stack[-1][0] > p:
            stack.pop()
            if stack:
                carry = min(carry, stack[-1][1])
        if stack:
            if carry > lpf[p]:
                lpf[p] = carry
            stack[-1][1] = carry
        stack.append([p, 0])


def longest_previous_factor(s: np.ndarray) -> np.ndarray:
    """LPF[i] = max_{j<i} LCP(s[i:], s[j:]) (Crochemore & Ilie 2008): the best
    earlier occurrence is one of the two nearest suffix-array neighbours with a
    smaller text position, and the LCP to it is a range minimum of the LCP array."""
    s = np.asarray(s)
    n = len(s)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    sa = suffix_array(s)
    lcp = lcp_array(s, sa).tolist()
    sal = sa.tolist()
    lpf = [0] * n
    big = n + 1
    # left sweep: link between rank r-1 and r is lcp[r]
    _nearest_smaller_pass(sal, lambda r: lcp[r] if r > 0 else big, range(n), lpf)
    # right sweep: link between rank r and r+1 is lcp[r+1]
    _nearest_smaller_pass(sal, lambda r: lcp[r + 1] if r + 1 < n else big, range(n - 1, -1, -1), lpf)
    return np.asarray(lpf, dtype=np.int64)


def lz76_phrase_count(s) -> int:
    """LZ76 complexity c(n): number of phrases in the exhaustive-history parsing."""
    if isinstance(s, str):
        s = np.frombuffer(s.encode(), dtype=np.uint8)
    s = np.asarray(s)
    n = len(s)
    if n == 0:
        return 0
    lpf = longest_previous_factor(s)
    c, p = 0, 0
    while p < n:
        p += int(lpf[p]) + 1
        c += 1
    return c


# =============================================================================
# EEG complexity (Schartner et al. 2015)
# =============================================================================
def binarize_envelope(X: np.ndarray) -> np.ndarray:
    """X: (n_channels, n_times). Binarise each channel at the mean of its
    Hilbert amplitude envelope (after removing the channel mean)."""
    X = np.asarray(X, dtype=float)
    env = np.abs(hilbert(X - X.mean(axis=1, keepdims=True), axis=1))
    return (env > env.mean(axis=1, keepdims=True)).astype(np.uint8)


def normalised_lz(bits: np.ndarray, norm: str = "entropy", rng=None) -> float:
    """Normalised LZ76 complexity of a binary sequence.

    norm="entropy": c * log2(n) / (n * H), H = binary source entropy
                    (asymptotic value for a random source; Casali 2013).
    norm="shuffle": c / c(shuffled copy) (Schartner 2015).
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    n = len(bits)
    if n < 2:
        return 0.0
    c = lz76_phrase_count(bits)
    if norm == "shuffle":
        rng = np.random.default_rng(rng)
        return float(c / lz76_phrase_count(rng.permutation(bits)))
    p1 = bits.mean()
    if p1 in (0.0, 1.0):
        return 0.0
    H = -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)
    return float(c * np.log2(n) / (n * H))


def lzc(X: np.ndarray, norm: str = "entropy", rng=None) -> float:
    """Multivariate LZc: binarised channels interleaved time point by time
    point (column-major over the channel x time matrix), then normalised."""
    B = binarize_envelope(X)
    return normalised_lz(B.T.ravel(), norm=norm, rng=rng)


def lzs(X: np.ndarray, norm: str = "entropy", rng=None) -> float:
    """Single-channel LZ averaged over channels (Schartner's LZs)."""
    B = binarize_envelope(X)
    return float(np.mean([normalised_lz(b, norm=norm, rng=rng) for b in B]))
