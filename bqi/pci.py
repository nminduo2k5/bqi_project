"""
Perturbational Complexity Index, paper Sec. 5.2 (eq. 24, Table PCI).

    PCI = K(s) / L(s),   K(s) ~ LZC(s) = |C(s)| log2|s| / |s|

s is a binarised TMS-evoked EEG spatiotemporal matrix; C(s) is the number of
distinct substrings found by the Lempel-Ziv (LZ76) parsing algorithm.
The paper leaves L(s) undefined; `pci_from_binary_matrix` uses the source
entropy H(L) of Casali et al. (2013) so PCI lies on the [0, 1] scale.
"""
from __future__ import annotations
import numpy as np


def lempel_ziv_complexity(binary_sequence: str) -> int:
    """LZ76 complexity: count of distinct substrings under the classic
    incremental parsing rule (Lempel & Ziv, 1976)."""
    s = binary_sequence
    n = len(s)
    i, l = 0, 1
    k = 1
    k_max = 1
    stop = False
    c = 1
    while not stop:
        if s[i + k - 1] != s[l + k - 1]:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    stop = True
                else:
                    i = 0
                    k = 1
                    k_max = 1
            else:
                k = 1
        else:
            k += 1
            if l + k > n:
                c += 1
                stop = True
        if not stop and l >= n:
            stop = True
    return c


def matrix_to_binary_string(binary_matrix: np.ndarray) -> str:
    """Flatten a binary (0/1) spatiotemporal (channels x time) matrix into a
    single bit string, source-coded channel-major (as in Casali et al. 2013)."""
    return "".join(str(int(b)) for b in binary_matrix.flatten())


def source_entropy(binary_matrix: np.ndarray) -> float:
    """Shannon entropy (bits) of the binary source: H = -p log2 p - (1-p) log2 (1-p),
    p = fraction of ones (Casali et al. 2013, eq. for H(L))."""
    p1 = float(np.mean(binary_matrix))
    if p1 <= 0.0 or p1 >= 1.0:
        return 0.0
    return float(-p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1))


def pci_from_binary_matrix(binary_matrix: np.ndarray) -> float:
    """Normalised Lempel-Ziv complexity of a binarised spatiotemporal matrix,
    following Casali et al. (2013):

        PCI = c(L) * log2(L) / (L * H(L))

    with L = n_channels * n_timepoints, c(L) the LZ76 phrase count and H(L)
    the source entropy. PCI -> ~1 for a maximally random source and -> 0 for
    a stereotyped one, the [0, 1] scale used by the paper's Table PCI.

    Note: eq. (49) of the BQI paper leaves L(s) undefined; its literal reading
    is kept as `pci_bqi_literal` for comparison only.
    """
    s = matrix_to_binary_string(binary_matrix)
    n = len(s)
    if n < 2:
        return 0.0
    H = source_entropy(binary_matrix)
    if H == 0.0:
        return 0.0
    C = lempel_ziv_complexity(s)
    return float(C * np.log2(n) / (n * H))


def pci_bqi_literal(binary_matrix: np.ndarray) -> float:
    """Literal reading of BQI eq. (49) with L(s) = 1/log2|s|: K/L = C log2^2|s| / |s|.
    Not normalised (~log2|s| for random data); kept only to document the
    discrepancy with the paper's [0, 1] Table PCI scale."""
    s = matrix_to_binary_string(binary_matrix)
    n = len(s)
    if n < 2:
        return 0.0
    C = lempel_ziv_complexity(s)
    return float(C * np.log2(n) ** 2 / n)


def simulate_tms_eeg(n_channels: int, n_timepoints: int, complexity_level: float,
                      seed: int | None = None) -> np.ndarray:
    """Generate a synthetic binarised TMS-evoked EEG response matrix whose
    LZ-complexity/PCI scales with `complexity_level` in [0,1]:

      * complexity_level -> 0: near-constant, highly stereotyped response
        (deep anaesthesia / coma -- low PCI)
      * complexity_level -> 1: spatiotemporally rich, differentiated response
        (wakefulness / meditation -- high PCI)

    Implemented as a mixture of a slowly-varying, spatially-correlated
    "evoked" component (low complexity) and spatially/temporally
    decorrelated noise (high complexity), binarised by median threshold.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, (n_channels, 1))
    time_kernel = np.sin(np.linspace(0, 3 * np.pi, n_timepoints)) * np.exp(
        -np.linspace(0, 1, n_timepoints) * 2)
    evoked = base @ time_kernel[None, :]
    noise = rng.normal(0, 1, (n_channels, n_timepoints))
    mix = (1 - complexity_level) * evoked + complexity_level * noise
    thresh = np.median(mix)
    return (mix > thresh).astype(int)


# Complexity level per consciousness state calibrated so that the resulting
# PCI values approximately reproduce Table PCI (paper Sec 5.2) in expectation.
STATE_COMPLEXITY_LEVEL = {
    "Coma": 0.06,
    "Propofol anaesthesia (deep)": 0.10,
    "NREM sleep (stage N3)": 0.20,
    "Propofol sedation (light)": 0.34,
    "REM sleep": 0.46,
    "Mindfulness (novice)": 0.52,
    "Alert wakefulness": 0.55,
    "Deep meditation (expert)": 0.63,
    "Near-death experience (NDE)": 0.70,
}

# Reference (published, meta-analytic) values from Table PCI for validation.
PCI_TABLE_REFERENCE = {
    "Alert wakefulness":            {"mean": 0.567, "sd": 0.089, "N": 412},
    "REM sleep":                    {"mean": 0.498, "sd": 0.102, "N": 338},
    "NREM sleep (stage N3)":        {"mean": 0.213, "sd": 0.067, "N": 318},
    "Propofol sedation (light)":    {"mean": 0.384, "sd": 0.091, "N": 276},
    "Propofol anaesthesia (deep)":  {"mean": 0.148, "sd": 0.041, "N": 201},
    "Coma":                         {"mean": 0.106, "sd": 0.032, "N": 189},
    "Mindfulness (novice)":         {"mean": 0.543, "sd": 0.094, "N": 287},
    "Deep meditation (expert)":     {"mean": 0.623, "sd": 0.087, "N": 421},
    "Near-death experience (NDE)":  {"mean": 0.681, "sd": 0.118, "N": 405},
}


def generate_pci_dataset(n_channels: int = 32, n_timepoints: int = 64,
                          n_subjects_per_state: int = 30, seed: int = 0):
    """Simulate a per-subject synthetic PCI dataset across all 9 states
    (a *new* dataset in the spirit of Table PCI, generated from first
    principles via the LZ-complexity algorithm rather than copied from the
    paper) -- used to test `pci_from_binary_matrix` and to compare its
    state ordering against the published meta-analytic ordering."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for state, level in STATE_COMPLEXITY_LEVEL.items():
        for subj in range(n_subjects_per_state):
            jitter = rng.normal(0, 0.03)
            lvl = float(np.clip(level + jitter, 0.01, 0.99))
            mat = simulate_tms_eeg(n_channels, n_timepoints, lvl,
                                    seed=int(rng.integers(0, 1_000_000)))
            pci = pci_from_binary_matrix(mat)
            rows.append({"state": state, "subject": subj, "pci": pci})
    return pd.DataFrame(rows)
