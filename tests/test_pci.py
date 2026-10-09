import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bqi.pci import (lempel_ziv_complexity, pci_from_binary_matrix, pci_bqi_literal, simulate_tms_eeg,
                     generate_pci_dataset)


def test_lz_complexity_constant_sequence_is_low():
    seq = "0" * 200
    c = lempel_ziv_complexity(seq)
    assert c <= 3


def test_lz_complexity_random_sequence_is_higher_than_constant():
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(["0", "1"], size=200))
    c_random = lempel_ziv_complexity(seq)
    c_const = lempel_ziv_complexity("0" * 200)
    assert c_random > c_const


def test_pci_increases_with_complexity_level():
    mat_low = simulate_tms_eeg(16, 48, complexity_level=0.05, seed=0)
    mat_high = simulate_tms_eeg(16, 48, complexity_level=0.9, seed=0)
    pci_low = pci_from_binary_matrix(mat_low)
    pci_high = pci_from_binary_matrix(mat_high)
    assert pci_high > pci_low


def test_pci_state_ordering_qualitatively_matches_paper():
    """Coma should have lower simulated PCI than alert wakefulness, matching
    the direction (not necessarily magnitude) of Table PCI in the paper."""
    df = generate_pci_dataset(n_channels=16, n_timepoints=48, n_subjects_per_state=10, seed=1)
    means = df.groupby("state")["pci"].mean()
    assert means["Coma"] < means["Alert wakefulness"]
    assert means["Propofol anaesthesia (deep)"] < means["Deep meditation (expert)"]


def test_pci_is_normalised_to_unit_scale():
    """Casali (2013) normalisation: random binary source -> PCI ~ 1, independent of size."""
    rng = np.random.default_rng(0)
    for shape in [(16, 48), (32, 64), (64, 128)]:
        assert 0.9 < pci_from_binary_matrix(rng.integers(0, 2, shape)) < 1.15


def test_bqi_literal_reading_scales_with_log_length():
    """The paper's literal eq. (49) grows like log2|s|, so it cannot match Table PCI's [0, 1] scale."""
    rng = np.random.default_rng(1)
    small = pci_bqi_literal(rng.integers(0, 2, (16, 32)))
    large = pci_bqi_literal(rng.integers(0, 2, (64, 128)))
    assert large > small > 2
