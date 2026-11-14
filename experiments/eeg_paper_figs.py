"""Figures for the two-dataset EEG paper (IEEE Access draft, template_latex is
paper 1; this is paper 2).

fig1_pred_vs_achieved : relative errors of the scale-free parameters on the two
                        datasets: CRB computed before the lock, Fisher SE at the
                        MLE, parametric-bootstrap RMSE, observed-information SE.
fig2_ou_timescale     : simulation. A feature made of a slow and a fast
                        Ornstein-Uhlenbeck component plus white noise is fitted
                        with ONE OU process plus white noise (the M1 DMN channel).
                        The recovered time constant moves from the fast to the
                        slow value as the slow component's share of the variance
                        grows: an OU fit returns the dominant time scale of the
                        feature it is given.

Run:  python experiments/eeg_paper_figs.py   (about two minutes; seed fixed)
Outputs go to outputs/paper/eeg/ (PDF + PNG + eeg_sim.json).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "paper" / "eeg"
DT = 1.0 / 6.0  # epoch length in minutes (10 s)

PARAMS = ["k_inh", "k_d", "sigma_A", "sigma_Phi", "r_A", "r_Phi", "r_Q"]
LABELS = {
    "k_inh": r"$k_{\mathrm{inh}}$", "k_d": r"$k_d$", "sigma_A": r"$\sigma_A$",
    "sigma_Phi": r"$\sigma_\Phi$", "r_A": r"$r_A$", "r_Phi": r"$r_\Phi$", "r_Q": r"$r_Q$",
}
SERIES = [
    ("crb_pred_E1", "CRB before the lock", "o", "none", -0.24),
    ("se_fisher", "Fisher SE at the MLE", "o", "full", -0.08),
    ("rmse_boot", "bootstrap RMSE", "s", "full", 0.08),
    ("se_observed", "observed-information SE", "^", "full", 0.24),
]


def _style():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 8, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42,
    })


def _save(fig, name):
    """Default saver: PDF + PNG in outputs/paper/eeg/. experiments/figs_vi.py
    passes its own saver that translates the labels first."""
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def fig_pred_vs_achieved(save=_save) -> dict:
    ds = {
        "ds001787": json.load(open(ROOT / "outputs/paper/realdata/summary.json", encoding="utf-8")),
        "OSF 3uszv": json.load(open(ROOT / "outputs/paper/osf3uszv/summary.json", encoding="utf-8")),
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=True)
    x = np.arange(len(PARAMS))
    summary = {}
    for ax, (name, d) in zip(axes, ds.items()):
        e = d["errors"]
        for key, lab, m, fs, off in SERIES:
            y = [100.0 * e[p][key] for p in PARAMS]
            ax.plot(x + off, y, m, label=lab, fillstyle=fs, ms=4.5, ls="none",
                    color={"o": "#1f4e9c", "s": "#d1495b", "^": "#2a9d8f"}[m] if fs == "full" else "#1f4e9c")
        ax.axhline(20, color="grey", lw=0.6, ls=":")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[p] for p in PARAMS])
        ax.set_title(f"{name}: {d['data']['n_kept']} sessions, {d['data']['n_rated_probes_kept']:,} probes")
        ax.grid(axis="y", lw=0.3, alpha=0.5)
        summary[name] = {p: {k: round(100.0 * e[p][k], 2) for k, *_ in SERIES} for p in PARAMS}
    axes[0].set_ylabel("relative error (%)")
    for ax in axes:
        ax.text(6.45, 21.5, "20 %", color="grey", fontsize=6.5, ha="right")
    axes[1].legend(frameon=False, loc="lower left", ncol=1)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig1_pred_vs_achieved")
    return summary


# ---------------------------------------------------------------- simulation
def _ou(rng, n_sess, n_ep, tau, var):
    phi = np.exp(-DT / tau)
    x = np.empty((n_sess, n_ep))
    x[:, 0] = rng.normal(0.0, np.sqrt(var), n_sess)
    q = var * (1.0 - phi ** 2)
    noise = rng.normal(0.0, np.sqrt(q), (n_sess, n_ep))
    for t in range(1, n_ep):
        x[:, t] = phi * x[:, t - 1] + noise[:, t]
    return x


def simulate(rng, n_sess, n_ep, tau_s, tau_f, share, r):
    """Feature = slow OU (variance `share`) + fast OU (variance 1-share) + white
    noise of sd r. Probe windows: two of every eight epochs missing (a 16-s
    response every 77 s, as on OSF 3uszv) plus 10 % missing at random."""
    y = rng.normal(0.0, r, (n_sess, n_ep))
    if share > 0:
        y += _ou(rng, n_sess, n_ep, tau_s, share)
    if share < 1:
        y += _ou(rng, n_sess, n_ep, tau_f, 1.0 - share)
    mask = np.ones_like(y, dtype=bool)
    mask[:, 0::8] = False
    mask[:, 1::8] = False
    mask &= rng.random(y.shape) > 0.10
    return y, mask


def negloglik(theta, y, mask):
    """Exact Kalman log-likelihood of one OU process plus white noise, all
    sessions in parallel (same grid), missing epochs skipped."""
    k, sig, r = np.exp(theta)
    phi = np.exp(-k * DT)
    V = sig ** 2 / (2.0 * k)
    q = V * (1.0 - phi ** 2)
    R = r ** 2
    n_sess, n_ep = y.shape
    m = np.zeros(n_sess)
    P = np.full(n_sess, V)
    ll = 0.0
    for t in range(n_ep):
        if t > 0:
            m = phi * m
            P = phi ** 2 * P + q
        obs = mask[:, t]
        S = P + R
        v = y[:, t] - m
        ll += -0.5 * np.sum(obs * (np.log(2.0 * np.pi * S) + v ** 2 / S))
        K = np.where(obs, P / S, 0.0)
        m = m + K * v
        P = P - K * P
    return -ll


def fit_ou(y, mask):
    best = None
    for tau0 in (1.0, 10.0):
        x0 = np.log([1.0 / tau0, 1.0, 0.5])
        res = minimize(negloglik, x0, args=(y, mask), method="L-BFGS-B",
                       bounds=[(np.log(1e-3), np.log(60.0)), (np.log(1e-3), np.log(1e2)), (np.log(1e-3), np.log(1e2))])
        if best is None or res.fun < best.fun:
            best = res
    k, sig, r = np.exp(best.x)
    return {"tau_min": 1.0 / k, "sigma": sig, "r": r, "nll": float(best.fun)}


def run_ou_simulation(seed=42, n_sess=48, n_ep=342, tau_f=0.5, r=0.6,
                      taus_slow=(3.0, 7.0, 15.0), shares=None, verbose=True) -> dict:
    shares = np.linspace(0.0, 1.0, 11) if shares is None else np.asarray(shares)
    rng = np.random.default_rng(seed)
    results = {}
    for tau_s in taus_slow:
        rec = []
        for share in shares:
            y, mask = simulate(rng, n_sess, n_ep, tau_s, tau_f, float(share), r)
            f = fit_ou(y, mask)
            rec.append(f["tau_min"])
            if verbose:
                print(f"  tau_slow {tau_s:4.1f} min, slow share {share:.1f}: fitted 1/k = {f['tau_min']:6.2f} min "
                      f"(sigma {f['sigma']:.2f}, r {f['r']:.2f})", flush=True)
        results[str(tau_s)] = rec
    return {"seed": seed, "n_sessions": n_sess, "n_epochs": n_ep, "epoch_min": DT, "tau_fast_min": tau_f,
            "obs_noise_sd": r, "shares": [float(s) for s in shares],
            "fitted_tau_min": {k: [float(v) for v in vals] for k, vals in results.items()},
            "missing_fraction_design": "2 of 8 epochs (probe windows) + 10 % at random"}


def plot_ou_timescale(sim: dict, save=_save) -> None:
    shares, tau_f = np.asarray(sim["shares"]), sim["tau_fast_min"]
    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    colors = ["#2a9d8f", "#1f4e9c", "#d1495b"]
    for (tau_s, rec), c in zip(sim["fitted_tau_min"].items(), colors):
        ax.plot(shares, rec, "o-", ms=3.5, lw=1.1, color=c, label=rf"slow component $\tau_s$ = {float(tau_s):g} min")
        ax.axhline(float(tau_s), color=c, lw=0.6, ls=":")
    ax.axhline(tau_f, color="grey", lw=0.6, ls="--")
    ax.text(0.99, tau_f * 1.12, rf"fast component $\tau_f$ = {tau_f:g} min", color="grey", fontsize=6.5, ha="right")
    ax.set_yscale("log")
    ax.set_xlabel("share of the feature variance in the slow component")
    ax.set_ylabel(r"fitted $1/\hat k$ (min), one OU + white noise")
    ax.set_xlim(-0.03, 1.03)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig2_ou_timescale")


def fig_ou_timescale(save=_save, **kw) -> dict:
    sim = run_ou_simulation(**kw)
    plot_ou_timescale(sim, save)
    return sim


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    fast = "--fast" in argv
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    print("fig1: predicted vs achieved errors")
    err = fig_pred_vs_achieved()
    if "--plot-only" in argv:                      # re-plot fig2 from the saved simulation
        sim = json.load(open(OUT / "eeg_sim.json", encoding="utf-8"))["simulation"]
        plot_ou_timescale(sim)
    else:
        print("fig2: OU time-scale simulation")
        sim = fig_ou_timescale(shares=[0.0, 0.5, 1.0] if fast else None,
                               taus_slow=(7.0,) if fast else (3.0, 7.0, 15.0), n_sess=8 if fast else 48)
    json.dump({"errors_percent": err, "simulation": sim}, open(OUT / "eeg_sim.json", "w", encoding="utf-8"), indent=2)
    print("written to", OUT)


if __name__ == "__main__":
    main()
