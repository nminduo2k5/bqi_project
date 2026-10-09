"""
Generate all synthetic datasets used to test the BQI paper's algorithms.

None of these are copied from the paper's tables -- each is generated from
first principles by the corresponding algorithm in `bqi/`, calibrated so its
summary statistics land in the same qualitative regime as the paper's
(purely illustrative, meta-analytic) tables. Reference values from the paper
tables are also written out separately (as `*_paper_reference.csv`) for
side-by-side comparison, clearly marked as such.

Standalone:  python datasets/generate_datasets.py [--seed N] [--only pci,ode,...]
Output: datasets/generated/*.csv

Also importable as a library:
    from datasets.generate_datasets import generate_all
    paths = generate_all(seed=42, out_dir=..., only=["pci", "ode"])
"""
import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bqi.pci import generate_pci_dataset, PCI_TABLE_REFERENCE
from bqi.dmn_ode import generate_ode_param_dataset, ODE_PARAMS_REFERENCE
from bqi.meta_analysis import STUDIES_REFERENCE, generate_resampled_meta_dataset
from bqi.spectral_graph import generate_kuramoto_dataset
from bqi.quantum_decoherence import generate_decoherence_dataset

DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "generated")

ALL_DATASETS = ["pci", "ode", "meta", "kuramoto", "quantum", "reference_tables"]


def _save(df: pd.DataFrame, name: str, out_dir: str, verbose: bool) -> str:
    path = os.path.join(out_dir, name)
    df.to_csv(path, index=False)
    if verbose:
        print(f"  wrote {path}  ({len(df)} rows)")
    return path


def generate_all(seed: int = 42, out_dir: str = None, only=None,
                  n_subjects_per_state: int = 40, n_ode_subjects: int = 500,
                  n_bootstrap: int = 5000, n_kuramoto_trials: int = 15,
                  n_quantum_trials: int = 10, verbose: bool = True) -> dict:
    """Generate the requested subset of datasets (default: all). Returns a
    dict mapping dataset name -> list of CSV paths written."""
    out_dir = out_dir or DEFAULT_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    which = set(only) if only else set(ALL_DATASETS)
    written = {}

    def log(msg):
        if verbose:
            print(msg)

    log("Generating synthetic datasets for the BQI paper algorithms...\n")

    if "pci" in which:
        log("[PCI] Sec 5.2, Table PCI -- via Lempel-Ziv complexity ...")
        df_pci = generate_pci_dataset(n_channels=32, n_timepoints=64,
                                       n_subjects_per_state=n_subjects_per_state, seed=seed)
        p1 = _save(df_pci, "pci_synthetic_subjects.csv", out_dir, verbose)
        p2 = _save(pd.DataFrame(PCI_TABLE_REFERENCE).T.reset_index().rename(columns={"index": "state"}),
                    "pci_paper_reference.csv", out_dir, verbose)
        written["pci"] = [p1, p2]

    if "ode" in which:
        log("[ODE] Sec 7.2, Table ode_params -- DMN/Phi/QoC parameter dataset ...")
        df_ode = generate_ode_param_dataset(n_subjects=n_ode_subjects, seed=seed)
        p1 = _save(df_ode, "ode_params_synthetic_subjects.csv", out_dir, verbose)
        p2 = _save(pd.DataFrame([ODE_PARAMS_REFERENCE]), "ode_params_paper_reference.csv", out_dir, verbose)
        written["ode"] = [p1, p2]

    if "meta" in which:
        log("[META] Sec 7.3, Table meta -- DMN meta-analysis bootstrap dataset ...")
        p1 = _save(pd.DataFrame(STUDIES_REFERENCE), "dmn_meta_analysis_paper_reference.csv", out_dir, verbose)
        df_boot = generate_resampled_meta_dataset(n_bootstrap=n_bootstrap, seed=seed)
        p2 = _save(df_boot, "dmn_meta_analysis_bootstrap.csv", out_dir, verbose)
        written["meta"] = [p1, p2]

    if "kuramoto" in which:
        log("[KURAMOTO] Sec 9.4, Table kuramoto -- synchrony dataset ...")
        df_kuramoto = generate_kuramoto_dataset(n_oscillators=50, n_trials=n_kuramoto_trials, seed=seed)
        p1 = _save(df_kuramoto, "kuramoto_synthetic_trials.csv", out_dir, verbose)
        written["kuramoto"] = [p1]

    if "quantum" in which:
        log("[QUANTUM] Sec 6 -- decoherence entropy dataset ...")
        df_decoh = generate_decoherence_dataset(dims=(2, 4, 8, 16), gammas=(0.25, 1.0, 4.0, 16.0),
                                                  n_trials=n_quantum_trials, seed=seed)
        p1 = _save(df_decoh, "quantum_decoherence_synthetic.csv", out_dir, verbose)
        written["quantum"] = [p1]

    if "reference_tables" in which:
        log("[REFERENCE] Reasoning-taxonomy / psychiatric-disorders tables ...")
        reasoning_modes = pd.DataFrame([
            {"mode": "Deductive", "logic": "Classical/first-order", "model": "Rule-based inference (modus ponens)", "substrate": "Left prefrontal cortex, LIFG"},
            {"mode": "Inductive", "logic": "Probabilistic", "model": "Bayesian generalisation", "substrate": "Hippocampus, medial PFC"},
            {"mode": "Abductive", "logic": "Non-monotonic", "model": "MAP inference over explanations", "substrate": "DLPFC, ACC"},
            {"mode": "Analogical", "logic": "Structure-mapping", "model": "Relational embedding similarity", "substrate": "Rostrolateral PFC"},
            {"mode": "Counterfactual", "logic": "Modal", "model": "Structural causal model intervention", "substrate": "vmPFC, hippocampus"},
        ])
        p1 = _save(reasoning_modes, "reasoning_taxonomy_reference.csv", out_dir, verbose)

        psychiatric = pd.DataFrame([
            {"disorder": "Major depression", "bqi_parameter": "Prior precision Pi_s (elevated)", "effect": "Over-weighted negative priors resist updating"},
            {"disorder": "Schizophrenia", "bqi_parameter": "Sensory precision Pi_o (reduced)", "effect": "Prediction errors under-weighted -> aberrant salience"},
            {"disorder": "Anxiety disorders", "bqi_parameter": "Expected free energy G(pi)", "effect": "Inflated uncertainty estimates over future states"},
            {"disorder": "Autism spectrum", "bqi_parameter": "Hierarchical precision imbalance", "effect": "Reduced top-down, elevated low-level precision"},
            {"disorder": "Addiction", "bqi_parameter": "SNIS query bias (habitual policy)", "effect": "Narrowed accessible subspace H_B around drug-related cues"},
        ])
        p2 = _save(psychiatric, "psychiatric_disorders_reference.csv", out_dir, verbose)
        written["reference_tables"] = [p1, p2]

    log(f"\nAll requested datasets generated in: {out_dir}")
    return written


def _parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic datasets for the BQI paper algorithms.")
    p.add_argument("--seed", type=int, default=42, help="Global random seed (default: 42)")
    p.add_argument("--only", type=str, default=None,
                   help=f"Comma-separated subset of datasets to generate: {','.join(ALL_DATASETS)}")
    p.add_argument("--out-dir", type=str, default=None, help="Override output directory")
    p.add_argument("--n-subjects-per-state", type=int, default=40, help="PCI: subjects per consciousness state")
    p.add_argument("--n-ode-subjects", type=int, default=500, help="ODE: number of synthetic subjects")
    p.add_argument("--n-bootstrap", type=int, default=5000, help="Meta-analysis: number of bootstrap resamples")
    p.add_argument("--n-kuramoto-trials", type=int, default=15, help="Kuramoto: trials per state x band")
    p.add_argument("--n-quantum-trials", type=int, default=10, help="Quantum: trials per dim x gamma")
    p.add_argument("--quiet", action="store_true", help="Suppress progress logging")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    only = args.only.split(",") if args.only else None
    generate_all(seed=args.seed, out_dir=args.out_dir, only=only,
                 n_subjects_per_state=args.n_subjects_per_state,
                 n_ode_subjects=args.n_ode_subjects,
                 n_bootstrap=args.n_bootstrap,
                 n_kuramoto_trials=args.n_kuramoto_trials,
                 n_quantum_trials=args.n_quantum_trials,
                 verbose=not args.quiet)
