"""
Run every algorithm in the BQI project end-to-end, save figures reproducing
the paper's key results, and write a JSON/text results summary.

Standalone:  python experiments/run_all.py [--seed N] [--only pci,hopfield,...]
Output: outputs/figures/*.png, outputs/results/summary.json

Also importable as a library:
    from experiments.run_all import run_all
    summary = run_all(seed=42, fig_dir=..., res_dir=..., only=["pci", "hopfield"])
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bqi.predictive_coding import HierarchicalBPC
from bqi.bqi_algorithm import BQIModel, BQIConfig
from bqi.com_algorithm import COMConfig, run_com_algorithm
from bqi.iit_phi import integrated_information_Phi, np_hardness_partition_count
from bqi.pci import generate_pci_dataset, STATE_COMPLEXITY_LEVEL, PCI_TABLE_REFERENCE
from bqi.dmn_ode import ODE_PARAMS_REFERENCE, steady_states, steady_states_paper, simulate as ode_simulate
from bqi.meta_analysis import random_effects_meta_analysis, pooled_percent_change
from bqi.hopfield import retrieval_error_experiment, theoretical_capacity_classic, theoretical_capacity_modern
from bqi.spectral_graph import (build_small_world_brain_graph, fiedler_value, small_world_index,
                                 global_efficiency, meditation_graph_transform, generate_kuramoto_dataset)
from bqi.quantum_decoherence import simulate_entropy_dynamics

DEFAULT_FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
DEFAULT_RES_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "results")

ALL_EXPERIMENTS = [
    "bpc", "bqi", "com", "phi", "pci", "ode", "meta", "hopfield", "spectral", "quantum",
]


def run_all(seed: int = 42, fig_dir: str = None, res_dir: str = None,
            only=None, save_figures: bool = True, verbose: bool = True) -> dict:
    """Run the requested subset of experiments (default: all) with a single
    top-level `seed` controlling every sub-experiment's RNG (derived via a
    fixed per-experiment offset so results stay reproducible)."""
    fig_dir = fig_dir or DEFAULT_FIG_DIR
    res_dir = res_dir or DEFAULT_RES_DIR
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    which = set(only) if only else set(ALL_EXPERIMENTS)
    summary = {}

    def log(msg):
        if verbose:
            print(msg)

    def savefig(name):
        if not save_figures:
            plt.close()
            return
        path = os.path.join(fig_dir, name)
        plt.tight_layout()
        plt.savefig(path, dpi=130)
        plt.close()
        log(f"  saved {path}")

    # -----------------------------------------------------------------------
    if "bpc" in which:
        log("=== 1. Hierarchical Bayesian Predictive Coding (Sec 3.1) ===")
        bpc = HierarchicalBPC(n_levels=3, dims=[6, 6, 4, 3], seed=seed)
        rng = np.random.default_rng(seed + 1)
        o = rng.normal(0, 1, 6)
        energies = bpc.run(o, n_steps=200, eta=0.05)
        plt.figure(figsize=(6, 4))
        plt.plot(energies)
        plt.yscale("log")
        plt.xlabel("Belief-update iteration")
        plt.ylabel("Total prediction-error energy (log scale)")
        plt.title("Hierarchical BPC: convergence of prediction error (eq. 5-7)")
        savefig("01_hierarchical_bpc_convergence.png")
        summary["hierarchical_bpc"] = {"energy_start": float(energies[0]), "energy_end": float(energies[-1])}

    # -----------------------------------------------------------------------
    if "bqi" in which:
        log("=== 2. BQI Algorithm 1 (Sec 4.5, Three-Phase Belief Update) ===")
        cfg1 = BQIConfig(n_levels=3, dims=(8, 8, 6, 4), n_snis=64, d_snis=4, d_k=4, T=100, seed=seed)
        model1 = BQIModel(cfg1)
        o1 = np.random.default_rng(seed + 2).normal(0, 1, 8)
        result1 = model1.run(o1)
        plt.figure(figsize=(6, 4))
        plt.plot(result1["free_energy_trace"])
        plt.xlabel("Phase-2 iteration")
        plt.ylabel("Variational free energy F")
        plt.title("BQI Algorithm 1: Phase 2 free-energy minimisation")
        savefig("02_bqi_algorithm_free_energy.png")
        summary["bqi_algorithm"] = {
            "F_final": float(result1["F_final"]),
            "n_iterations": result1["n_iterations"],
            "mu_star": result1["mu_star"].tolist(),
        }

    # -----------------------------------------------------------------------
    if "com" in which:
        log("=== 3. COM Algorithm 2 (Sec 8.2, Consciousness Optimisation) ===")
        cfg2 = COMConfig(T=120, d_mu=6, n_snis=30, d_snis=6, d_k=6, seed=seed + 1)
        result2 = run_com_algorithm(cfg2)
        t = result2["trace"]["t"]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].plot(t, result2["trace"]["A_dmn"], color="firebrick")
        axes[0].set_title("$A_{DMN}(t)$"); axes[0].set_xlabel("t")
        axes[1].plot(t, result2["trace"]["Phi"], color="steelblue")
        axes[1].set_title("$\\Phi(t)$"); axes[1].set_xlabel("t")
        axes[2].plot(t, result2["trace"]["QoC"], color="seagreen")
        axes[2].set_title("QoC(t)"); axes[2].set_xlabel("t")
        plt.suptitle("COM Algorithm: three-phase optimisation trajectory (Fig. com_opt)")
        savefig("03_com_algorithm_trajectory.png")
        summary["com_algorithm"] = {"QoC_final": float(result2["QoC_final"]), "delta_S": float(result2["delta_S"])}

    # -----------------------------------------------------------------------
    if "phi" in which:
        log("=== 4. IIT Phi computation (Sec 5.1) ===")
        rng2 = np.random.default_rng(seed + 3)
        n_phi = 5
        W = rng2.normal(0, 1.8, (n_phi, n_phi))
        np.fill_diagonal(W, 0)
        state = rng2.integers(0, 2, n_phi)
        phi_res = integrated_information_Phi(W, state, scale=1.0)
        summary["iit_phi"] = {
            "Phi_whole_system": float(phi_res["Phi"]),
            "n_partitions_exact_n5": np_hardness_partition_count(5),
            "top_mechanisms": {str(k): float(v) for k, v in
                                sorted(phi_res["mechanism_phis"].items(), key=lambda kv: -kv[1])[:5]},
        }

    # -----------------------------------------------------------------------
    if "pci" in which:
        log("=== 5. Perturbational Complexity Index (Sec 5.2, Table PCI) ===")
        df_pci = generate_pci_dataset(n_channels=32, n_timepoints=64, n_subjects_per_state=25, seed=seed + 4)
        order = list(STATE_COMPLEXITY_LEVEL.keys())
        means = df_pci.groupby("state")["pci"].mean().reindex(order)
        plt.figure(figsize=(7, 5))
        plt.barh(order, means.values, color="indianred")
        plt.xlabel("Simulated PCI (Lempel-Ziv complexity based)")
        plt.title("Simulated PCI across consciousness states (cf. Table PCI)")
        savefig("04_pci_by_state.png")
        summary["pci"] = {"simulated_means": means.round(4).to_dict(),
                           "paper_reference_means": {k: v["mean"] for k, v in PCI_TABLE_REFERENCE.items()}}

    # -----------------------------------------------------------------------
    if "ode" in which:
        log("=== 6. DMN/Phi/QoC ODE system (Sec 7.2) ===")
        ss = steady_states(ODE_PARAMS_REFERENCE)
        sol_ctrl = ode_simulate(ODE_PARAMS_REFERENCE, meditator=False, seed=seed + 1)
        sol_med = ode_simulate(ODE_PARAMS_REFERENCE, meditator=True, seed=seed + 1)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].plot(sol_ctrl["t"], sol_ctrl["A_DMN"], label="A_DMN control")
        axes[0].plot(sol_med["t"], sol_med["A_DMN"], "--", label="A_DMN meditator")
        axes[0].plot(sol_med["t"], sol_med["Phi"], label="Phi meditator")
        axes[0].plot(sol_med["t"], sol_med["QoC"], label="QoC meditator")
        axes[0].legend(fontsize=8); axes[0].set_xlabel("t (min)"); axes[0].set_title("ODE trajectories")
        axes[1].plot(sol_ctrl["A_DMN"], sol_ctrl["Phi"], color="gray", label="control")
        axes[1].plot(sol_med["A_DMN"], sol_med["Phi"], color="green", label="meditator")
        axes[1].set_xlabel("A_DMN"); axes[1].set_ylabel("Phi"); axes[1].set_title("Phase portrait")
        axes[1].legend(fontsize=8)
        savefig("05_dmn_ode_trajectories.png")
        summary["dmn_ode"] = {"steady_states": ss,
                               "steady_states_paper": steady_states_paper(ODE_PARAMS_REFERENCE),
                               "control_final": {k: float(sol_ctrl[k][-1]) for k in ["A_DMN", "Phi", "QoC"]},
                               "meditator_final": {k: float(sol_med[k][-1]) for k in ["A_DMN", "Phi", "QoC"]}}

    # -----------------------------------------------------------------------
    if "meta" in which:
        log("=== 7. Random-effects meta-analysis (Sec 7.3, Table meta) ===")
        meta_res = random_effects_meta_analysis()
        pct_res = pooled_percent_change()
        studies_d = meta_res["per_study_d"]
        plt.figure(figsize=(6, 6))
        y = np.arange(len(studies_d))
        plt.errorbar(studies_d, y, xerr=1.96 * np.sqrt(1 / meta_res["weights_re"]), fmt="s", color="steelblue")
        plt.axvline(meta_res["pooled_d"], color="firebrick", linestyle="--",
                    label=f"Pooled d={meta_res['pooled_d']:.2f}")
        plt.xlabel("Cohen's d"); plt.title("Forest plot: DMN suppression (16 studies)")
        plt.legend()
        savefig("06_forest_plot_meta_analysis.png")
        summary["meta_analysis"] = {"pooled_d": meta_res["pooled_d"], "ci": meta_res["ci"],
                                     "I2_percent": meta_res["I2_percent"],
                                     "pooled_delta_pct": pct_res["pooled_delta_pct"]}

    # -----------------------------------------------------------------------
    if "hopfield" in which:
        log("=== 8. Hopfield capacity: classic vs modern (Sec 9.2) ===")
        n_hop = 30
        df_hop = retrieval_error_experiment(n=n_hop, trials_per_count=10, seed=seed)
        plt.figure(figsize=(7, 5))
        plt.plot(df_hop["n_patterns"], df_hop["classic_error_rate"], "o-", color="firebrick", label="Classic Hopfield")
        plt.plot(df_hop["n_patterns"], df_hop["modern_error_rate"], "s-", color="steelblue", label="Modern Hopfield (softmax)")
        plt.plot(df_hop["n_patterns"], df_hop["transformer_error_rate"], "^--", color="seagreen", label="Transformer attention")
        plt.xscale("log"); plt.xlabel("Number of stored patterns P"); plt.ylabel("Retrieval error rate")
        plt.title(f"Retrieval error vs #patterns (n={n_hop} units)")
        plt.legend()
        savefig("07_hopfield_retrieval_error.png")
        summary["hopfield"] = {"n_units": n_hop, "theoretical_classic_capacity": theoretical_capacity_classic(n_hop),
                                "theoretical_modern_capacity": theoretical_capacity_modern(n_hop)}

    # -----------------------------------------------------------------------
    if "spectral" in which:
        log("=== 9. Spectral graph theory + Kuramoto synchrony (Sec 4.4, 9.4) ===")
        G = build_small_world_brain_graph(n_nodes=80, k=6, p_rewire=0.1, seed=seed)
        fied = fiedler_value(G)
        sigma = small_world_index(G, n_random=4)
        E_glob = global_efficiency(G)
        G_med = meditation_graph_transform(G, boost_fraction=0.08, seed=seed + 1)
        E_glob_med = global_efficiency(G_med)
        df_kuramoto = generate_kuramoto_dataset(n_oscillators=50, n_trials=8, seed=seed + 3)
        pivot = df_kuramoto.groupby(["state", "band"])["r"].mean().unstack()
        plt.figure(figsize=(7, 5))
        pivot.plot(kind="bar", ax=plt.gca())
        plt.ylabel("Kuramoto order parameter r")
        plt.title("Synchrony across states / frequency bands (cf. Table kuramoto)")
        savefig("08_kuramoto_order_parameter.png")
        summary["spectral_graph"] = {
            "fiedler_value": fied, "small_world_index": sigma, "global_efficiency": E_glob,
            "global_efficiency_meditation": E_glob_med,
            "pct_increase_E_glob": 100 * (E_glob_med - E_glob) / E_glob,
        }

    # -----------------------------------------------------------------------
    if "quantum" in which:
        log("=== 10. Quantum decoherence / von Neumann entropy (Sec 6) ===")
        plt.figure(figsize=(7, 5))
        for gamma, label, color in [(0.3, "slow decoherence", "steelblue"),
                                     (2.0, "moderate decoherence", "seagreen"),
                                     (10.0, "fast decoherence", "firebrick")]:
            res = simulate_entropy_dynamics(dim=4, gamma=gamma, T=3.0, dt=0.01, seed=seed)
            plt.plot(res["t"], res["S"], label=label, color=color)
        plt.axhline(np.log(4), color="gray", linestyle=":", label="S_max = ln(4)")
        plt.xlabel("t"); plt.ylabel("von Neumann entropy S(rho)")
        plt.title("Open quantum system entropy dynamics (cf. Fig entropy)")
        plt.legend()
        savefig("09_quantum_entropy_dynamics.png")
        summary["quantum_decoherence"] = {"note": "see figure 09; entropy rises toward ln(dim) for all gamma"}

    # -----------------------------------------------------------------------
    log("\nWriting results summary...")
    res_path = os.path.join(res_dir, "summary.json")
    with open(res_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"Done. Figures in {fig_dir}")
    log(f"Results summary in {res_path}")
    return summary


def _parse_args():
    p = argparse.ArgumentParser(description="Run BQI paper experiments end-to-end.")
    p.add_argument("--seed", type=int, default=42, help="Global random seed (default: 42)")
    p.add_argument("--only", type=str, default=None,
                   help=f"Comma-separated subset of experiments to run: {','.join(ALL_EXPERIMENTS)}")
    p.add_argument("--fig-dir", type=str, default=None, help="Override output directory for figures")
    p.add_argument("--res-dir", type=str, default=None, help="Override output directory for results JSON")
    p.add_argument("--no-figures", action="store_true", help="Skip saving PNG figures (summary JSON only)")
    p.add_argument("--quiet", action="store_true", help="Suppress progress logging")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    only = args.only.split(",") if args.only else None
    run_all(seed=args.seed, fig_dir=args.fig_dir, res_dir=args.res_dir, only=only,
            save_figures=not args.no_figures, verbose=not args.quiet)
