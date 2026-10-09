"""
Publication figures and tables for the model paper (Q1_pipeline_model.md Sec 11-13).

    python main.py model figures            # after `model all` (or any subset)
    python experiments/run_model_paper.py [--out outputs/paper/model] [--profiles]

Reads outputs/results/model/*.{json,csv} and outputs/gates/*.json; recomputes only
cheap closed-form quantities. Every figure is written as PNG (300 dpi) + PDF,
every table as CSV + LaTeX. Missing inputs (e.g. gate M1 not run yet) skip the
corresponding figure with a message instead of failing.

Figures / tables (numbering follows the paper outline):
  T1  steady states: paper eq. 55-57 vs corrected M0 vs M1
  F1  P2 - QoC is a high-pass read-out: step response of M0 + Bode plot
  F2  M1 stationary distribution of QoC (closed form vs simulation)
  T2  identifiability by observation configuration (relative CRB)
  F3  profile likelihoods (--profiles, slow)              [optional]
  F4  CRB vs MLE RMSE at the representative cells (gate M1) [needs design_grid_mle.csv]
  F5  design map: relative CRB over (N, session length) + CRB(k_inh) vs N   [main figure]
  T3  cheapest designs reaching CRB(k_inh) <= 20 %
  F6  closed-loop trajectories: LQG vs bang-bang vs no control
  F7  efficiency loss of LQG designed on estimated parameters [needs control robustness]
  F8  Sobol total-order indices
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bqi.dmn_ode import ODE_PARAMS_REFERENCE, StateSpaceDMN, steady_states, steady_states_paper
from model import config as CFG

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "outputs", "results", "model")
GATES = os.path.join(ROOT, "outputs", "gates")
DEFAULT_OUT = os.path.join(ROOT, "outputs", "paper", "model")

# Colour-blind-safe palette (matches the dashboard)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED, GRAY = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948", "#8a8985")
KEY = list(CFG.KEY_PARAMS)
KEY_TEX = {"k_inh": r"$k_{\mathrm{inh}}$", "A0": r"$A_0$", "alpha": r"$\alpha$", "beta": r"$\beta$",
           "k_int": r"$k_{\mathrm{int}}$", "k_d": r"$k_d$", "sigma_A": r"$\sigma_A$", "sigma_Phi": r"$\sigma_\Phi$",
           "gamma0": r"$\gamma_0$", "r_A": r"$r_A$", "r_Phi": r"$r_\Phi$", "r_Q": r"$r_Q$",
           "cv_A": r"$\mathrm{CV}_\infty(A)$"}

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                     "figure.dpi": 110, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})


# =============================================================================
# helpers
# =============================================================================
def _json(name):
    try:
        with open(os.path.join(RES, name), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _csv(name, **kw):
    p = os.path.join(RES, name)
    return pd.read_csv(p, **kw) if os.path.isfile(p) else None


def _gate(name):
    try:
        with open(os.path.join(GATES, f"{name}.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


class Writer:
    def __init__(self, out):
        self.out = out
        os.makedirs(out, exist_ok=True)
        self.made, self.skipped = [], []

    def fig(self, fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(self.out, f"{name}.{ext}"), bbox_inches="tight")
        plt.close(fig)
        self.made.append(name)
        print(f"  {name}.png/.pdf")

    def table(self, df: pd.DataFrame, name, caption="", index=True, float_fmt="%.3f"):
        df.to_csv(os.path.join(self.out, f"{name}.csv"), index=index)
        tex = df.to_latex(index=index, float_format=float_fmt, escape=False, caption=caption or None,
                          label=f"tab:{name}")
        with open(os.path.join(self.out, f"{name}.tex"), "w", encoding="utf-8") as f:
            f.write(tex)
        self.made.append(name)
        print(f"  {name}.csv/.tex")

    def skip(self, name, why):
        self.skipped.append((name, why))
        print(f"  [bỏ qua] {name}: {why}")


# =============================================================================
# T1 + F1 + F2 — theory
# =============================================================================
def table1_steady_states(w: Writer):
    P = ODE_PARAMS_REFERENCE
    sp, s0, m1 = steady_states_paper(P), steady_states(P), StateSpaceDMN().steady_states()
    df = pd.DataFrame({"Paper eq. 55-57": [sp["A_DMN_star"], sp["Phi_star"], sp["QoC_star"]],
                       "M0, corrected": [s0["A_DMN_star"], s0["Phi_star"], s0["QoC_star"]],
                       "M1 (proposed)": [m1["A_DMN_star"], m1["Phi_star"], m1["QoC_star"]]},
                      index=[r"$A^*$", r"$\Phi^*$", r"$\mathrm{QoC}^*$"])
    w.table(df, "table1_steady_states", "Steady states of the DMN-Phi-QoC model.", float_fmt="%.4f")


def fig1_highpass(w: Writer):
    from scipy.integrate import solve_ivp
    from bqi.dmn_ode import _rhs
    P = ODE_PARAMS_REFERENCE
    one = lambda t: 1.0
    t_end = 300.0
    sol = solve_ivp(_rhs, (0, t_end), [0.0, 0.0, 0.0], args=(P, lambda t: CFG.P2_STEP_XI if t >= 20 else 0.0, one),
                    method="LSODA", rtol=1e-9, atol=1e-12, t_eval=np.linspace(0, t_end, 3000))
    A, Phi, Q = sol.y
    drive = P["alpha"] * Phi - P["beta"] * A
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ax[0].plot(sol.t, A, color=BLUE, label=r"$A_{\mathrm{DMN}}$")
    ax[0].plot(sol.t, Phi, color=AQUA, label=r"$\Phi$")
    ax[0].plot(sol.t, drive, color=GRAY, ls="--", label=r"$\alpha\Phi-\beta A$ (drive)")
    ax[0].plot(sol.t, Q, color=ORANGE, lw=2, label="QoC (M0)")
    ax[0].axvline(20, color="k", lw=0.6, ls=":")
    ax[0].set(xlabel="t (min)", ylabel="level", title="(a) Step in $\\xi$ at t = 20 min: QoC returns to 0")
    ax[0].legend(loc="upper right", frameon=False)
    f = np.logspace(-3, 1, 400)
    wv = 2 * np.pi * f
    H = 1j * wv / (1j * wv + P["gamma"])
    ax[1].semilogx(f, 20 * np.log10(np.abs(H)), color=BLUE)
    ax[1].axvline(P["gamma"] / (2 * np.pi), color=ORANGE, ls="--")
    ax[1].text(P["gamma"] / (2 * np.pi) * 1.15, -35, r"$f_c=\gamma/2\pi$", color=ORANGE)
    ax[1].set(xlabel="frequency (1/min)", ylabel="|H| (dB)", title="(b) $H(s)=s/(s+\\gamma)$: DC gain = 0",
              ylim=(-45, 3))
    fig.tight_layout()
    w.fig(fig, "fig1_highpass_qoc")


def fig2_stationary(w: Writer):
    m = StateSpaceDMN()
    ss, h, P = m.steady_states(), m.readout(), m.stationary_cov()
    sd = float(np.sqrt(h @ P @ h))
    n_long = 60000  # 10,000 min: ~70 independent samples at 5 slowest time constants
    sim = m.simulate(n_long, CFG.DT, probe_every=None, seed=CFG.SEED)
    tau = 1 / min(m.k_inh, m.k_d)
    stride = int(np.ceil(CFG.GATES.m0_p5_thin_time_constants * tau / CFG.DT))
    thin = sim["QoC"][::stride]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [3, 2]})
    ax[0].plot(sim["t"][:1800], sim["QoC"][:1800], color=BLUE, lw=0.8)
    ax[0].axhline(ss["QoC_star"], color=ORANGE, ls="--", label=r"$\mathrm{QoC}^*=\alpha\Phi^*-\beta A^*+\gamma_0$")
    ax[0].fill_between(sim["t"][:1800], ss["QoC_star"] - 2 * sd, ss["QoC_star"] + 2 * sd, color=ORANGE, alpha=0.12,
                       label=r"$\pm 2\sqrt{h^\top P h}$")
    ax[0].set(xlabel="t (min)", ylabel="QoC", title="(a) M1: QoC fluctuates around a non-zero level")
    ax[0].legend(frameon=False, loc="upper right")
    x = np.linspace(ss["QoC_star"] - 4 * sd, ss["QoC_star"] + 4 * sd, 300)
    ax[1].hist(sim["QoC"], bins=60, density=True, color=BLUE, alpha=0.35,
               label=f"simulation ({n_long / 6 / 60:.0f} h)")
    ax[1].plot(x, np.exp(-0.5 * ((x - ss["QoC_star"]) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)), color=ORANGE, lw=2,
               label=r"$\mathcal{N}(\mathrm{QoC}^*,\,h^\top P h)$")
    ax[1].set(xlabel="QoC", ylabel="density",
              title=f"(b) stationary law (KS on {len(thin)} thinned samples)")
    ax[1].legend(frameon=False)
    fig.tight_layout()
    w.fig(fig, "fig2_m1_stationary")


# =============================================================================
# T2 + F3 + F4 — identifiability and recovery
# =============================================================================
def table2_identifiability(w: Writer):
    df = _csv("table2_identifiability.csv", index_col=0)
    if df is None:
        print("  (tính Bảng 2 trực tiếp, ~1 phút)")
        from model.identifiability import identifiability_table
        df = identifiability_table(StateSpaceDMN())
    def _fmt(v):
        try:
            return f"{float(v):.3f}"
        except (TypeError, ValueError):
            return str(v).replace("se=", "SE = ")
    df = df.rename(index=KEY_TEX).map(_fmt)
    R = CFG.REFERENCE
    df.columns = [f"({c}) " + {"a": "rating", "b": "rating + $y_A$", "c": "rating + $y_\\Phi$",
                               "d": "rating + $y_A$ + $y_\\Phi$"}[c] for c in df.columns]
    w.table(df, "table2_identifiability",
            f"Relative Cram\\'er--Rao bound by observation configuration (N = {R.n_subjects}, {R.session_min:g} min, "
            f"probe every {R.probe_every_min:g} min). "
            "`fixed': normalisation of a non-identifiable direction; `absent': parameter does not enter the likelihood.",
            float_fmt="%.3f")


def fig3_profiles(w: Writer, do: bool, seed: int):
    cache = os.path.join(RES, "profiles.csv")
    if not do and not os.path.isfile(cache):
        w.skip("fig3_profiles", "chạy với --profiles (chậm, ~30 phút) để tính profile likelihood")
        return
    if os.path.isfile(cache):
        prof = pd.read_csv(cache)
    else:
        from eeg.model_fit import fit, profile_likelihood
        m = StateSpaceDMN()
        R = CFG.REFERENCE
        dt = CFG.DT
        sess = [m.simulate(R.n_epochs, dt, probe_every=R.probe_every, seed=seed + s) for s in range(R.n_subjects)]
        base = fit(sess, dt, init=m, n_starts=1, rng=seed)
        rows = []
        for p in CFG.PROFILE_PARAMS:
            v0 = getattr(m, p)
            grid = v0 * np.exp(np.linspace(-CFG.PROFILE_LOG_HALF_RANGE, CFG.PROFILE_LOG_HALF_RANGE, CFG.PROFILE_N_GRID))
            ll = profile_likelihood(sess, dt, base, p, grid)
            rows += [{"param": p, "value": g, "loglik": l, "mle_loglik": base["loglik"]} for g, l in zip(grid, ll)]
        prof = pd.DataFrame(rows)
        prof.to_csv(cache, index=False)
    from matplotlib.ticker import NullFormatter
    fig, axes = plt.subplots(1, len(CFG.PROFILE_PARAMS), figsize=(7.2, 2.5))
    for ax, p in zip(axes, CFG.PROFILE_PARAMS):
        d = prof[prof.param == p].sort_values("value")
        truth = getattr(StateSpaceDMN(), p)
        ax.plot(d.value / truth, d.loglik - d.loglik.max(), "o-", color=BLUE)
        ax.axhline(-1.92, color=ORANGE, ls="--", label=r"$\chi^2_1$ 95 %")
        ax.axvline(1.0, color=GRAY, ls=":", label="truth")
        ax.set_xscale("log")
        ax.set_xticks([0.5, 0.7, 1.0, 1.4, 2.0], ["0.5", "0.7", "1", "1.4", "2"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set(xlabel=f"{KEY_TEX[p]} / true value", title=f"profile of {KEY_TEX[p]}")
    axes[0].set_ylabel(r"$\ell_p - \ell_{\max}$")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    w.fig(fig, "fig3_profiles")


def fig4_crb_vs_mle(w: Writer):
    crb, mle = _csv("design_grid_crb.csv"), _csv("design_grid_mle.csv")
    if crb is None or mle is None:
        w.skip("fig4_crb_vs_mle", "cần design_grid_mle.csv (cổng M1, `model design`)")
        return
    m = mle.merge(crb, on=["N", "session_min", "probe_every_min", "r_obs"])
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    cols = [BLUE, ORANGE, AQUA, VIOLET]
    for k, c in zip(KEY, cols):
        ax.scatter(m[f"crb_{k}"], m[f"mle_{k}"], s=22 + m["N"] * 0.6, color=c, alpha=0.8, label=KEY_TEX[k],
                   edgecolor="white", lw=0.5)
    lim = float(max(m[[f"crb_{k}" for k in KEY]].max().max(), m[[f"mle_{k}" for k in KEY]].max().max())) * 1.1
    ax.plot([0, lim], [0, lim], color=GRAY, ls="--", lw=1)
    ax.fill_between([0, lim], [0, lim / 2], [0, 2 * lim], color=GRAY, alpha=0.08, label="within ×2")
    ax.set(xlabel="Cramér–Rao bound (relative)", ylabel="MLE RMSE (same scale)", xlim=(0, lim), ylim=(0, lim),
           title="Gate M1: does the bound predict recovery error?")
    ax.legend(frameon=False, loc="upper left")
    g = _gate("M1")
    if g:
        d = g["details"]
        ax.text(0.98, 0.03, f"{'PASS' if g['passed'] else 'FAIL'}: {d['fraction_matching']:.0%} cells within ×2\n"
                f"median RMSE/CRB = {d['ratio_rmse_over_crb']['median']:.2f}", transform=ax.transAxes, ha="right",
                fontsize=7.5)
    fig.tight_layout()
    w.fig(fig, "fig4_crb_vs_mle")


# =============================================================================
# F5 + T3 — design map
# =============================================================================
def fig5_design_map(w: Writer):
    crb = _csv("design_grid_crb.csv")
    if crb is None:
        w.skip("fig5_design_map", "cần design_grid_crb.csv (`model design`, phần CRB chỉ mất ~2 phút)")
        return
    from matplotlib.ticker import NullFormatter
    probe, r = CFG.REFERENCE.probe_every_min, CFG.REFERENCE.r_obs
    sub = crb[(crb.probe_every_min == probe) & (crb.r_obs == r)]
    fig = plt.figure(figsize=(7.4, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1.05])
    heat_axes = [fig.add_subplot(gs[0, j]) for j in range(4)]
    for ax, k, tag in zip(heat_axes, KEY, "abcd"):
        pv = sub.pivot(index="N", columns="session_min", values=f"crb_{k}")
        im = ax.imshow(pv.values, cmap="Blues_r", vmin=0, vmax=0.4, aspect="auto", origin="lower")
        ax.set_xticks(range(len(pv.columns)), [f"{c:.0f}" for c in pv.columns])
        ax.set_yticks(range(len(pv.index)), [str(i) for i in pv.index])
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                v = pv.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v < 0.2 else "k")
        ax.set(xlabel="session (min)", title=f"({tag}) CRB of {KEY_TEX[k]}")
        if tag == "a":
            ax.set_ylabel("subjects N")
    cb = fig.colorbar(im, ax=heat_axes, shrink=0.9, pad=0.01, aspect=25)
    cb.set_label("relative CRB")
    # (e) CRB(k_inh) vs N for each session length, probe 2 min
    ax = fig.add_subplot(gs[1, :2])
    for sm, c in zip(sorted(sub.session_min.unique()), [BLUE, ORANGE, AQUA]):
        d = sub[sub.session_min == sm].sort_values("N")
        ax.plot(d.N, d.crb_k_inh, "o-", color=c, label=f"{sm:.0f} min")
    ax.axhline(0.2, color=GRAY, ls="--")
    ax.set_xscale("log")
    ax.set_xticks(list(CFG.GRID.n_subjects), [str(n) for n in CFG.GRID.n_subjects])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set(xlabel="subjects N", ylabel=r"CRB($k_{\mathrm{inh}}$)", title="(e) sample size vs bound (probe every 2 min)")
    ax.legend(frameon=False, title="session")
    # (f) effect of probe interval and EEG noise at N = 24, 45 min
    ax = fig.add_subplot(gs[1, 2:])
    d = crb[(crb.N == 24) & (crb.session_min == CFG.REFERENCE.session_min)]
    for ro, c in zip(sorted(d.r_obs.unique()), [BLUE, ORANGE, AQUA]):
        dd = d[d.r_obs == ro].sort_values("probe_every_min")
        ax.plot(dd.probe_every_min, dd.crb_k_inh, "o-", color=c, label=f"r = {ro:g}")
    ax.set_xscale("log")
    ax.set_xticks(list(CFG.GRID.probe_every_min), [f"{p:g}" for p in CFG.GRID.probe_every_min])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set(xlabel="probe interval (min)", ylabel=r"CRB($k_{\mathrm{inh}}$)",
           title=f"(f) N = 24, {CFG.REFERENCE.session_min:g} min")
    ax.legend(frameon=False, title="EEG noise")
    fig.suptitle(f"Design map (observation config d; panels a–d: probe every {probe:g} min, EEG noise r = {r:g})")
    w.fig(fig, "fig5_design_map")


def table3_min_designs(w: Writer):
    crb = _csv("design_grid_crb.csv")
    if crb is None:
        w.skip("table3_min_designs", "cần design_grid_crb.csv")
        return
    rows = []
    for k in KEY:
        ok = crb[crb[f"crb_{k}"] <= CFG.TARGET_REL_ERROR].copy()
        if ok.empty:
            rows.append({"parameter": KEY_TEX[k], "N": "—", "session (min)": "—", "probe (min)": "—",
                         "EEG noise r": "—", "CRB": "> 0.2 everywhere"})
            continue
        ok["cost"] = ok.N * ok.session_min
        b = ok.sort_values(["cost", f"crb_{k}"]).iloc[0]
        rows.append({"parameter": KEY_TEX[k], "N": int(b.N), "session (min)": f"{b.session_min:.0f}",
                     "probe (min)": f"{b.probe_every_min:g}", "EEG noise r": f"{b.r_obs:g}",
                     "CRB": f"{b[f'crb_{k}']:.3f}"})
    w.table(pd.DataFrame(rows), "table3_min_designs",
            f"Cheapest design (N $\\times$ session length) with relative CRB $\\le {100 * CFG.TARGET_REL_ERROR:.0f}\\%$ "
            "for each key parameter.",
            index=False)


# =============================================================================
# F6 + F7 — control
# =============================================================================
def fig6_control(w: Writer, seed: int):
    from model.control import solve_lqg, simulate_policy, state_space_with_control
    m = StateSpaceDMN()
    st = state_space_with_control(m)
    C = CFG.CONTROL
    ctrl = solve_lqg(st, target_qoc=C.target_qoc)
    runs = {p: simulate_policy(m, ctrl, p, C.n_epochs, C.b_control, C.u_max, seed, st) for p in ("none", "bangbang", "lqg")}
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for p, lab, c in [("none", "no control", GRAY), ("bangbang", "bang-bang", ORANGE), ("lqg", "LQG", BLUE)]:
        ax[0].plot(runs[p]["t"], runs[p]["QoC"], color=c, lw=1.4, label=f"{lab} (cost {runs[p]['cost']:.3f})")
    ax[0].axhline(C.target_qoc, color=RED, ls="--", lw=1, label="target")
    ax[0].set(ylabel="QoC", title="Closed loop on the true model, common random numbers")
    ax[0].legend(frameon=False, ncol=4, loc="lower right")
    ax[1].plot(runs["bangbang"]["t"], runs["bangbang"]["u"], color=ORANGE, lw=1)
    ax[1].plot(runs["lqg"]["t"], runs["lqg"]["u"], color=BLUE, lw=1.4)
    ax[1].set(xlabel="t (min)", ylabel="u")
    fig.tight_layout()
    w.fig(fig, "fig6_control_trajectories")


def fig7_robustness(w: Writer):
    cr = _json("control_results.json")
    if not cr or not cr.get("robustness"):
        w.skip("fig7_robustness", "cần phần độ bền trong control_results.json (chạy `model control` sau khi có M1)")
        return
    rb = pd.DataFrame(cr["robustness"])
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    sc = ax.scatter(rb.avg_mle_rmse, rb.efficiency_loss_pct_median, c=rb.N, cmap="viridis", s=40, edgecolor="white")
    ax.errorbar(rb.avg_mle_rmse, rb.efficiency_loss_pct_median,
                yerr=[np.zeros(len(rb)), (rb.efficiency_loss_pct_q90 - rb.efficiency_loss_pct_median).clip(lower=0)],
                fmt="none", ecolor=GRAY, lw=0.7)
    fig.colorbar(sc, ax=ax, label="subjects N")
    ax.set(xlabel="mean MLE RMSE of ($k_{\\mathrm{inh}}, A_0, \\alpha, \\beta$)", ylabel="efficiency loss (%)",
           title="LQG designed on estimated parameters")
    fig.tight_layout()
    w.fig(fig, "fig7_robustness")


# =============================================================================
# F8 — sensitivity
# =============================================================================
def fig8_sobol(w: Writer):
    sob = _csv("sobol_indices.csv")
    meta = _json("sensitivity_results.json") or {}
    if sob is None:
        w.skip("fig8_sobol", "cần sobol_indices.csv (`model sensitivity`)")
        return
    labels = {"qoc_stationary": r"$\mathrm{QoC}^*$", "qoc_variance": r"Var$_\infty$(QoC)",
              "crb_k_inh": r"CRB($k_{\mathrm{inh}}$)", "lqg_efficiency": "LQG cost ratio"}
    order = [p for p in ["k_inh", "A0", "cv_A", "sigma_A", "k_int", "k_d", "sigma_Phi", "alpha", "beta"]
             if p in set(sob["param"])]
    pv = sob.pivot(index="param", columns="output", values="ST").reindex(order)
    pv = pv[[c for c in labels if c in pv.columns]]
    has_ci = {"ST_lo", "ST_hi"} <= set(sob.columns)
    lo = sob.pivot(index="param", columns="output", values="ST_lo").reindex(order)[pv.columns] if has_ci else None
    hi = sob.pivot(index="param", columns="output", values="ST_hi").reindex(order)[pv.columns] if has_ci else None
    scale = sob.drop_duplicates("output").set_index("output")["scale"].to_dict() if "scale" in sob.columns else {}
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    im = ax.imshow(np.clip(pv.values, 0, 1), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pv.columns)),
                  [labels[c] + ("\n(log scale)" if scale.get(c) == "log" else "\n(raw)") for c in pv.columns],
                  fontsize=8)
    ax.set_yticks(range(len(pv.index)), [KEY_TEX[i] for i in pv.index])
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            txt = f"{v:.2f}"
            if has_ci:
                txt += f"\n[{lo.values[i, j]:.2f}, {hi.values[i, j]:.2f}]"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.3, color="white" if v > 0.6 else "k")
    fig.colorbar(im, ax=ax, label="total-order Sobol index $S_T$")
    nb = meta.get("n_base", "?")
    nboot = meta.get("n_boot")
    ax.set_title(f"Total-order Sobol indices $S_T$ (×0.25–×4 log-uniform, CV$_\\infty$(A) ∈ [0.05, 0.4]; "
                 f"n_base = {nb}" + (f", 95 % bootstrap CI, B = {nboot}" if nboot else "") + ")", fontsize=7.5)
    fig.tight_layout()
    w.fig(fig, "fig8_sobol")


# =============================================================================
def run(out: str = DEFAULT_OUT, profiles: bool = False, seed: int = CFG.SEED) -> dict:
    w = Writer(out)
    CFG.snapshot(out)
    print(f"Xuất hình/bảng vào {out}")
    table1_steady_states(w)
    fig1_highpass(w)
    fig2_stationary(w)
    table2_identifiability(w)
    fig3_profiles(w, profiles, seed)
    fig4_crb_vs_mle(w)
    fig5_design_map(w)
    table3_min_designs(w)
    fig6_control(w, seed)
    fig7_robustness(w)
    fig8_sobol(w)
    gates = {g: (_gate(g) or {}).get("passed") for g in ("M0", "M1", "M2")}
    manifest = {"made": w.made, "skipped": w.skipped, "gates": gates, "seed": seed}
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Xong: {len(w.made)} tệp, {len(w.skipped)} bỏ qua. Cổng: {gates}")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description="Publication figures/tables for the model paper.")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--profiles", action="store_true", help="Also compute profile likelihoods (slow)")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    run(a.out, a.profiles, a.seed)


if __name__ == "__main__":
    main()
