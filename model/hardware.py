"""
Goi ket qua 6 -- "Neural design": the M1 closed loop (Kalman filter + LQG)
as a fixed-point neural circuit, bit-exact with the VHDL in hw/vhdl/.

The loop of model/control.py is linear, so every operation is a *linear
neuron*: weighted sum + bias, followed by rounding and saturation (the only
non-linearity, like a hard-limiter activation). The circuit has three layers
per epoch (10 s), see hw/README.md:

    layer 1  predict   x_pred = F x_hat + B u + c           (2 neurons, 3 inputs)
    layer 2  correct   x_hat  = (I - L) x_pred + L y         (2 neurons, 4 inputs)
    layer 3  control   u      = clip(-K x_hat + (u_ss + K x_ss), +-u_max)   (1 neuron)
             read-out  QoC^   = h x_hat + gamma0             (1 neuron, monitoring)

Fixed-point format: signed two's complement, W = FRAC + INT_BITS + 1 bits,
FRAC fraction bits.  A neuron computes, in integers,

    acc = (bias << FRAC) + sum_i w_i * x_i        (exact, 2W+3-bit accumulator)
    y   = saturate_W( (acc + 2^(FRAC-1)) >> FRAC ) (round half up, arithmetic shift)

and the Python `neuron()` below is the reference the VHDL function
`neuron_pkg.neuron` must reproduce bit for bit.  The closed-loop plant stays
in floating point (it is the "brain"); only the controller is quantised.

Outputs (python main.py model hardware):
    outputs/results/model/hardware_results.json   quantisation sweep, chosen format
    outputs/results/model/hw_quantisation.png     cost ratio and |du| vs FRAC
    hw/vhdl/gen/lqg_params_pkg.vhd                constants for the chosen format
    hw/vhdl/gen/vectors.txt                       per-epoch test vectors (bit-exact)
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bqi.dmn_ode import StateSpaceDMN
from model import config as CFG
from model.control import solve_lqg, state_space_with_control

C = CFG.CONTROL
INT_BITS = 4                      # magnitude bits left of the binary point (range +-16)
FRAC_SWEEP = (6, 8, 10, 12, 14, 16, 18)
MAX_COST_EXCESS = 0.01            # chosen format: fixed-point cost <= (1 + 1 %) x float cost
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HW_GEN_DIR = os.path.join(ROOT, "hw", "vhdl", "gen")


# =============================================================================
# Fixed-point format and the neuron primitive (reference for the VHDL)
# =============================================================================
@dataclass(frozen=True)
class FixedFormat:
    frac: int

    @property
    def width(self) -> int:
        return self.frac + INT_BITS + 1

    @property
    def scale(self) -> int:
        return 1 << self.frac

    @property
    def qmax(self) -> int:
        return (1 << (self.width - 1)) - 1

    @property
    def qmin(self) -> int:
        return -(1 << (self.width - 1))

    def sat(self, q: int) -> int:
        return max(self.qmin, min(self.qmax, int(q)))

    def quant(self, v: float) -> int:
        """Nearest integer (ties to even, Python round); saturated to W bits."""
        return self.sat(round(float(v) * self.scale))

    def to_float(self, q: int) -> float:
        return q / self.scale


def neuron(fmt: FixedFormat, inputs, weights, bias: int) -> int:
    """Linear neuron in integer arithmetic: exact MAC, round half up, saturate."""
    acc = int(bias) << fmt.frac
    for x, w in zip(inputs, weights):
        acc += int(x) * int(w)
    acc += 1 << (fmt.frac - 1)
    return fmt.sat(acc >> fmt.frac)            # >> on Python ints is an arithmetic (floor) shift


def clip(fmt: FixedFormat, q: int, lim: int) -> int:
    return max(-lim, min(lim, int(q)))


# =============================================================================
# Controller constants (float) and their quantised circuit
# =============================================================================
def controller_constants(model: StateSpaceDMN | None = None) -> dict:
    model = model or StateSpaceDMN()
    st = state_space_with_control(model)
    ctrl = solve_lqg(st)
    F, B, c, L, K = st["F"], st["B"], st["c"], ctrl["L"], ctrl["K"]
    IL = np.eye(2) - L
    return {
        "F": F, "B": B, "c": c, "L": L, "K": K, "h": st["h"], "gamma0": float(st["gamma0"]),
        "x_ss": ctrl["x_ss"], "u_ss": float(ctrl["u_ss"]), "mu": model.stationary_mean(),
        "u_max": float(C.u_max), "target": float(ctrl["target_qoc"]),
        # layer weights / biases
        "W1": [[F[0, 0], F[0, 1], B[0]], [F[1, 0], F[1, 1], B[1]]], "b1": [c[0], c[1]],
        "W2": [[IL[0, 0], IL[0, 1], L[0, 0], L[0, 1]], [IL[1, 0], IL[1, 1], L[1, 0], L[1, 1]]], "b2": [0.0, 0.0],
        "W3": [-K[0], -K[1]], "b3": float(ctrl["u_ss"] + K @ ctrl["x_ss"]),
        "W4": [st["h"][0], st["h"][1]], "b4": float(st["gamma0"]),
    }


class NeuralCircuit:
    """Bit-exact software model of hw/vhdl/dmn_lqg_core.vhd."""

    def __init__(self, fmt: FixedFormat, consts: dict | None = None):
        self.fmt = fmt
        k = consts or controller_constants()
        q = fmt.quant
        self.W1 = [[q(v) for v in row] for row in k["W1"]]
        self.b1 = [q(v) for v in k["b1"]]
        self.W2 = [[q(v) for v in row] for row in k["W2"]]
        self.b2 = [q(v) for v in k["b2"]]
        self.W3 = [q(v) for v in k["W3"]]
        self.b3 = q(k["b3"])
        self.W4 = [q(v) for v in k["W4"]]
        self.b4 = q(k["b4"])
        self.mu_q = [q(v) for v in k["mu"]]
        self.umax_q = q(k["u_max"])
        self.reset()

    def reset(self) -> None:
        self.xhat = list(self.mu_q)
        self.u = clip(self.fmt, neuron(self.fmt, self.xhat, self.W3, self.b3), self.umax_q)
        self.qoc = neuron(self.fmt, self.xhat, self.W4, self.b4)

    def step(self, y_a_q: int, y_phi_q: int) -> tuple[int, int, int, int]:
        f = self.fmt
        xp = [neuron(f, [self.xhat[0], self.xhat[1], self.u], self.W1[i], self.b1[i]) for i in range(2)]
        xh = [neuron(f, [xp[0], xp[1], y_a_q, y_phi_q], self.W2[i], self.b2[i]) for i in range(2)]
        self.xhat = xh
        self.u = clip(f, neuron(f, xh, self.W3, self.b3), self.umax_q)
        self.qoc = neuron(f, xh, self.W4, self.b4)
        return self.xhat[0], self.xhat[1], self.u, self.qoc

    def constants_int(self) -> dict:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2, "W3": self.W3, "b3": self.b3,
                "W4": self.W4, "b4": self.b4, "mu_q": self.mu_q, "umax_q": self.umax_q}


# =============================================================================
# Closed loop: float plant, float or fixed-point controller
# (same random-number order as model.control.simulate_policy)
# =============================================================================
def simulate_closed_loop(model: StateSpaceDMN, fmt: FixedFormat | None, n_epochs: int = C.n_epochs,
                         seed: int = 0, consts: dict | None = None, trace: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    k = consts or controller_constants(model)
    st = state_space_with_control(model)
    F, c, Q, B, h, g0, R = st["F"], st["c"], st["Q"], st["B"], st["h"], st["gamma0"], st["R_obs"]
    K, L, x_ss, u_ss, target, u_max = k["K"], k["L"], k["x_ss"], k["u_ss"], k["target"], k["u_max"]
    Lw = np.linalg.cholesky(Q + 1e-15 * np.eye(2))
    Lr = np.sqrt(np.diag(R))
    x = model.stationary_mean() + np.linalg.cholesky(model.stationary_cov() + 1e-15 * np.eye(2)) @ rng.normal(size=2)
    circuit = NeuralCircuit(fmt, k) if fmt is not None else None
    x_hat = model.stationary_mean().copy()
    xs, us, qocs = np.empty((n_epochs, 2)), np.empty(n_epochs), np.empty(n_epochs)
    rows = []
    for t in range(n_epochs):
        if circuit is None:
            u = float(np.clip(u_ss - K @ (x_hat - x_ss), -u_max, u_max))
        else:
            u = fmt.to_float(circuit.u)
        x = F @ x + c + B * u + Lw @ rng.normal(size=2)
        y = x + Lr * rng.normal(size=2)
        if circuit is None:
            x_pred = F @ x_hat + c + B * u
            x_hat = x_pred + L @ (y - x_pred)
        else:
            ya_q, yp_q = fmt.quant(y[0]), fmt.quant(y[1])
            out = circuit.step(ya_q, yp_q)
            if trace:
                rows.append((ya_q, yp_q) + out)
        xs[t], us[t], qocs[t] = x, u, h @ x + g0
    cost = float(np.mean(C.Q_cost * (qocs - target) ** 2 + C.R_cost * us ** 2))
    return {"cost": cost, "tracking_mse": float(np.mean((qocs - target) ** 2)),
            "u_rms": float(np.sqrt(np.mean(us ** 2))), "u": us, "QoC": qocs, "x": xs, "rows": rows}


def quantisation_sweep(model: StateSpaceDMN | None = None, fracs=FRAC_SWEEP, n_trials: int = C.n_trials,
                       n_epochs: int = C.n_epochs, seed: int = CFG.SEED, verbose: bool = True) -> list[dict]:
    model = model or StateSpaceDMN()
    k = controller_constants(model)
    ref = [simulate_closed_loop(model, None, n_epochs, seed + i, k) for i in range(n_trials)]
    cost_ref = float(np.mean([r["cost"] for r in ref]))
    out = []
    for fr in fracs:
        fmt = FixedFormat(fr)
        runs = [simulate_closed_loop(model, fmt, n_epochs, seed + i, k) for i in range(n_trials)]
        cost = float(np.mean([r["cost"] for r in runs]))
        du = float(max(np.max(np.abs(a["u"] - b["u"])) for a, b in zip(runs, ref)))
        row = {"frac": fr, "width": fmt.width, "cost_fixed": cost, "cost_float": cost_ref,
               "cost_ratio": cost / cost_ref, "max_abs_du": du,
               "tracking_mse": float(np.mean([r["tracking_mse"] for r in runs])),
               "u_rms": float(np.mean([r["u_rms"] for r in runs])),
               "lsb": 1.0 / fmt.scale}
        out.append(row)
        if verbose:
            print(f"  FRAC={fr:2d} W={fmt.width:2d}  cost {cost:.5f} (float {cost_ref:.5f}, ratio {row['cost_ratio']:.4f})"
                  f"  max|du| {du:.2e}", flush=True)
    return out


def choose_format(sweep: list[dict], max_excess: float = MAX_COST_EXCESS) -> int:
    ok = [r["frac"] for r in sweep if r["cost_ratio"] <= 1 + max_excess]
    return min(ok) if ok else max(r["frac"] for r in sweep)


# =============================================================================
# VHDL generation: constants package and bit-exact test vectors
# =============================================================================
def _vhdl_int_vec(name: str, vals) -> str:
    return f"  constant {name} : int_vec(0 to {len(vals) - 1}) := ({', '.join(str(int(v)) for v in vals)});"


def _one_line(a) -> str:
    """Array as a single line (VHDL comments end at the newline)."""
    return np.array2string(np.asarray(a), precision=6, separator=", ", max_line_width=10_000).replace("\n", " ")


def write_vhdl_package(circuit: NeuralCircuit, path: str, consts: dict) -> str:
    f = circuit.fmt
    ci = circuit.constants_int()
    lines = [
        "-- Generated by model/hardware.py -- DO NOT EDIT (python main.py model hardware)",
        f"-- Fixed-point format Q{INT_BITS}.{f.frac} signed, W = {f.width} bits; nominal M1 parameters, seed {CFG.SEED}.",
        "library ieee;",
        "use ieee.std_logic_1164.all;",
        "use ieee.numeric_std.all;",
        "",
        "package lqg_params_pkg is",
        f"  constant W    : natural := {f.width};",
        f"  constant FRAC : natural := {f.frac};",
        "  type int_vec is array (natural range <>) of integer;",
        "  -- layer 1: predict   x_pred_i = W1_i . (xhat_A, xhat_Phi, u) + b1_i",
        _vhdl_int_vec("W1_A", ci["W1"][0]), _vhdl_int_vec("W1_PHI", ci["W1"][1]),
        f"  constant B1_A   : integer := {ci['b1'][0]};", f"  constant B1_PHI : integer := {ci['b1'][1]};",
        "  -- layer 2: correct   xhat_i = W2_i . (x_pred_A, x_pred_Phi, y_A, y_Phi) + b2_i",
        _vhdl_int_vec("W2_A", ci["W2"][0]), _vhdl_int_vec("W2_PHI", ci["W2"][1]),
        f"  constant B2_A   : integer := {ci['b2'][0]};", f"  constant B2_PHI : integer := {ci['b2'][1]};",
        "  -- layer 3: control   u = clip(W3 . xhat + B3, +-UMAX_Q);  read-out QoC^ = W4 . xhat + B4",
        _vhdl_int_vec("W3", ci["W3"]), f"  constant B3     : integer := {ci['b3']};",
        _vhdl_int_vec("W4", ci["W4"]), f"  constant B4     : integer := {ci['b4']};",
        f"  constant UMAX_Q : integer := {ci['umax_q']};",
        "  -- reset state: stationary mean of M1",
        _vhdl_int_vec("MU_Q", ci["mu_q"]),
        "  -- float values for reference",
        f"  -- F = {_one_line(consts['F'])}",
        f"  -- B = {_one_line(consts['B'])}, c = {_one_line(consts['c'])}",
        f"  -- L = {_one_line(consts['L'])}",
        f"  -- K = {_one_line(consts['K'])}, x_ss = {_one_line(consts['x_ss'])}, u_ss = {consts['u_ss']:.6f}",
        "end package lqg_params_pkg;",
        "",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))
    return path


def write_vectors(rows, path: str, header: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# {header}\n# columns: y_A y_Phi | expected xhat_A xhat_Phi u qoc  (all integers in Q format)\n")
        fh.write(f"{len(rows)}\n")
        for r in rows:
            fh.write(" ".join(str(int(v)) for v in r) + "\n")
    return path


def _figure(sweep: list[dict], chosen: int, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fr = [r["frac"] for r in sweep]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax[0].plot(fr, [r["cost_ratio"] for r in sweep], "o-", color="#1f77b4")
    ax[0].axhline(1 + MAX_COST_EXCESS, color="#d62728", ls="--", lw=1, label=f"+{100 * MAX_COST_EXCESS:.0f} %")
    ax[0].axvline(chosen, color="gray", ls=":", label=f"chosen FRAC = {chosen}")
    ax[0].set(xlabel="fraction bits FRAC", ylabel="cost(fixed) / cost(float)", title="(a) closed-loop cost")
    ax[0].legend(frameon=False)
    ax[1].semilogy(fr, [r["max_abs_du"] for r in sweep], "o-", color="#ff7f0e", label=r"max $|u_{\rm fixed}-u_{\rm float}|$")
    ax[1].semilogy(fr, [r["lsb"] for r in sweep], "s--", color="gray", label="1 LSB")
    ax[1].set(xlabel="fraction bits FRAC", title="(b) control-signal error")
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# =============================================================================
def run(seed: int = CFG.SEED, fast: bool = False, verbose: bool = True, output_dir: str | None = None,
        gen_dir: str = HW_GEN_DIR) -> dict:
    """Entry point for `python main.py model hardware`."""
    model = StateSpaceDMN()
    n_trials = 5 if fast else C.n_trials
    if verbose:
        print(f"Quantisation sweep: FRAC in {FRAC_SWEEP}, {n_trials} trials x {C.n_epochs} epochs, "
              f"float plant + fixed-point neural circuit", flush=True)
    sweep = quantisation_sweep(model, n_trials=n_trials, seed=seed, verbose=verbose)
    chosen = choose_format(sweep)
    fmt = FixedFormat(chosen)
    consts = controller_constants(model)
    circuit = NeuralCircuit(fmt, consts)
    # bit-exact test vectors from one closed-loop session
    tr = simulate_closed_loop(model, fmt, C.n_epochs, seed, consts, trace=True)
    pkg = write_vhdl_package(circuit, os.path.join(gen_dir, "lqg_params_pkg.vhd"), consts)
    vec = write_vectors(tr["rows"], os.path.join(gen_dir, "vectors.txt"),
                        f"generated by model/hardware.py, seed {seed}, FRAC {chosen}, W {fmt.width}, "
                        f"{C.n_epochs} epochs, closed-loop cost {tr['cost']:.6f}")
    result = {"int_bits": INT_BITS, "sweep": sweep, "chosen_frac": chosen, "chosen_width": fmt.width,
              "chosen": next(r for r in sweep if r["frac"] == chosen),
              "constants_float": {k: (np.asarray(v).tolist() if not isinstance(v, (float, int)) else v)
                                  for k, v in consts.items()},
              "constants_int": circuit.constants_int(),
              "vector_session": {"seed": seed, "n_epochs": C.n_epochs, "cost": tr["cost"],
                                 "tracking_mse": tr["tracking_mse"], "u_rms": tr["u_rms"]},
              "circuit": {"neurons": 6, "multipliers_per_epoch": 3 + 3 + 4 + 4 + 2 + 2,
                          "clock_cycles_per_epoch": 3, "epoch_s": CFG.EPOCH_S},
              "files": {"vhdl_package": pkg, "vectors": vec}}
    if verbose:
        print(f"Chosen format: Q{INT_BITS}.{chosen} (W = {fmt.width} bits): cost ratio "
              f"{result['chosen']['cost_ratio']:.4f}, max|du| {result['chosen']['max_abs_du']:.2e}")
        print(f"Wrote {pkg}\n      {vec}")
    CFG.snapshot(output_dir)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "hardware_results.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        _figure(sweep, chosen, os.path.join(output_dir, "hw_quantisation.png"))
    return result


if __name__ == "__main__":
    run(output_dir=os.path.join(ROOT, "outputs", "results", "model"))
