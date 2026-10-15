"""
Assets for hw/report/bao_cao_neural_design.tex (run from the repo root):

    python hw/report/make_assets.py

1. GHDL synthesis of dmn_lqg_core -> hw/vhdl/work/netlist_synth.vhd (gate-level VHDL netlist)
2. Netlist statistics: constant multipliers, adders, registers (flip-flop bits), muxes
3. Post-synthesis simulation: the same testbench run on the synthesized netlist
4. Figures: cycle-level waveform (from the VCD of the RTL simulation) and the
   whole 45-min session, fixed-point circuit vs floating-point controller
Writes hw/report/assets/{synth_stats.json, fig_waveform.pdf, fig_session.pdf}.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
VHDL = os.path.join(ROOT, "hw", "vhdl")
OUT = os.path.join(ROOT, "hw", "report", "assets")


def find_ghdl() -> str:
    g = shutil.which("ghdl") or os.environ.get("GHDL")
    if g:
        return g
    for d in sorted(os.listdir("C:\\")) if os.name == "nt" else []:
        p = os.path.join("C:\\", d, "bin", "ghdl.exe")
        if d.lower().startswith("ghdl") and os.path.isfile(p):
            return p
    raise SystemExit("GHDL not found")


def run(cmd, cwd=VHDL) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


# -----------------------------------------------------------------------------
def synthesize(ghdl: str) -> str:
    r = run([ghdl, "--synth", "--std=08", "gen/lqg_params_pkg.vhd", "neuron_pkg.vhd", "dmn_lqg_core.vhd",
             "-e", "dmn_lqg_core"])
    if r.returncode != 0:
        raise SystemExit("synthesis failed:\n" + r.stderr)
    path = os.path.join(VHDL, "work", "netlist_synth.vhd")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="latin-1", newline="\n") as f:   # GHDL wants latin-1, no BOM
        f.write(r.stdout)
    return path


def netlist_stats(path: str) -> dict:
    txt = open(path, encoding="latin-1").read()
    widths = {}
    for m in re.finditer(r"signal (n\d+) : (std_logic_vector \((\d+) downto 0\)|std_logic)", txt):
        widths[m.group(1)] = int(m.group(3)) + 1 if m.group(3) else 1
    regs = re.findall(r"if rising_edge \(wrap_clk\) then\s+(n\d+) <=", txt)
    mults = re.findall(r"\* signed'\(\"([01]+)\"\)", txt)
    nonzero = [b for b in mults if set(b) != {"0"}]
    return {
        "lines": txt.count("\n"),
        "constant_multipliers": len(mults),
        "constant_multipliers_nonzero": len(nonzero),
        "adders": len(re.findall(r" \+ ", txt)),
        "subtractors": len(re.findall(r" - ", txt)),
        "registers": len(regs),
        "flipflop_bits": int(sum(widths.get(r, 0) for r in regs)),
        "register_widths": sorted(widths.get(r, 0) for r in regs),
        "mux_branches": len(re.findall(r" when ", txt)),
        "comparators": len(re.findall(r" (?:>|<|>=|<=) ", txt)),
    }


def post_synth_sim(ghdl: str, netlist: str) -> str:
    wd = os.path.join(VHDL, "work_ps")
    shutil.rmtree(wd, ignore_errors=True)
    os.makedirs(wd)
    for cmd in ([ghdl, "-a", "--std=08", "--workdir=work_ps", "gen/lqg_params_pkg.vhd", "neuron_pkg.vhd",
                 os.path.relpath(netlist, VHDL), "tb_dmn_lqg_core.vhd"],
                [ghdl, "-e", "--std=08", "--workdir=work_ps", "tb_dmn_lqg_core"],
                [ghdl, "-r", "--std=08", "--workdir=work_ps", "tb_dmn_lqg_core", "-gVEC_FILE=gen/vectors.txt",
                 "--assert-level=error"]):
        r = run(cmd)
        if r.returncode != 0:
            raise SystemExit("post-synthesis step failed:\n" + r.stdout + r.stderr)
    line = [l for l in r.stdout.splitlines() if "PASS" in l or "FAIL" in l]
    return line[-1].split("):")[-1].strip() if line else r.stdout.strip()


def rtl_sim_with_vcd(ghdl: str) -> str:
    os.makedirs(os.path.join(VHDL, "work"), exist_ok=True)
    for cmd in ([ghdl, "-a", "--std=08", "--workdir=work", "gen/lqg_params_pkg.vhd", "neuron_pkg.vhd",
                 "dmn_lqg_core.vhd", "tb_dmn_lqg_core.vhd"],
                [ghdl, "-e", "--std=08", "--workdir=work", "tb_dmn_lqg_core"],
                [ghdl, "-r", "--std=08", "--workdir=work", "tb_dmn_lqg_core", "-gVEC_FILE=gen/vectors.txt",
                 "--assert-level=error", "--vcd=work/tb_dmn_lqg_core.vcd"]):
        r = run(cmd)
        if r.returncode != 0:
            raise SystemExit("RTL simulation failed:\n" + r.stdout + r.stderr)
    line = [l for l in r.stdout.splitlines() if "PASS" in l or "FAIL" in l]
    return line[-1].split("):")[-1].strip()


# -----------------------------------------------------------------------------
def read_vcd(path: str, names: list[str], scope: str = "tb_dmn_lqg_core") -> dict:
    """Minimal VCD reader: value changes of top-level testbench signals (time in ns)."""
    ids, cur, out, t = {}, [], {n: [] for n in names}, 0
    with open(path, encoding="latin-1") as f:
        for line in f:
            s = line.strip()
            if s.startswith("$scope"):
                cur.append(s.split()[2])
            elif s.startswith("$upscope"):
                cur.pop()
            elif s.startswith("$var") and cur and cur[-1] == scope:
                p = s.split()
                nm = p[4].split("[")[0]
                if nm in names:
                    ids[p[3]] = (nm, int(p[2]))
            elif s.startswith("#"):
                t = int(s[1:]) / 1e6          # fs -> ns
            elif s and s[0] == "b":
                val, code = s[1:].split()
                if code in ids:
                    nm, w = ids[code]
                    v = int(val.replace("x", "0").replace("u", "0"), 2) if set(val) <= set("01xu") else 0
                    if len(val) == w and val[0] == "1" and w > 1:
                        v -= 1 << w
                    out[nm].append((t, v))
            elif s and s[0] in "01" and len(s) > 1 and s[1:] in ids:
                nm, _ = ids[s[1:]]
                out[nm].append((t, int(s[0])))
    return out


def step_series(changes, t_end):
    t = [c[0] for c in changes] + [t_end]
    v = [c[1] for c in changes] + [changes[-1][1]]
    return np.array(t), np.array(v)


def fig_waveform(vcd: str, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sig = ["clk", "start", "done", "y_a", "xhat_a", "u_out", "qoc_hat"]
    d = read_vcd(vcd, sig)
    t0 = next(t for t, v in d["start"] if v == 1) - 15      # first epoch
    t1 = t0 + 105                                             # ~2 epochs
    fig, axes = plt.subplots(len(sig), 1, figsize=(7.2, 5.8), sharex=True)
    for ax, s in zip(axes, sig):
        t, v = step_series(d[s], t1 + 50)
        if s in ("clk", "start", "done"):
            ax.step(t, v, where="post", color="#1f77b4", lw=1.1)
            ax.set_ylim(-0.3, 1.3)
        else:
            ax.step(t, v, where="post", color="#ff7f0e", lw=1.3)
            # segments overlapping the window (including the one active at t0) and their integer values
            ends = [c[0] for c in d[s][1:]] + [t1 + 50]
            segs = [(max(ta, t0), min(tb, t1), va) for (ta, va), tb in zip(d[s], ends) if tb > t0 and ta < t1]
            for lo, hi, va in segs:
                if hi - lo > 6:
                    ax.text((lo + hi) / 2, va, f"{va}", ha="center", va="bottom", fontsize=6.5, clip_on=True)
            vals = [va for _, _, va in segs]
            lo_, hi_ = min(vals), max(vals)
            pad = max(12, 0.45 * (hi_ - lo_))
            ax.set_ylim(lo_ - pad, hi_ + 1.6 * pad)
        ax.set_yticks([])
        ax.set_ylabel(s, rotation=0, ha="right", va="center", fontsize=8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    starts = [t for t, v in d["start"] if v == 1 and t0 <= t <= t1]
    for ts in starts:
        for k, lab in enumerate(["IDLE", "PREDICT", "CORRECT", "CONTROL"]):
            axes[0].text(ts + 10 * k + 5, 1.45, lab, ha="center", fontsize=5.8, color="gray")
        for ax in axes:
            ax.axvline(ts, color="gray", lw=0.4, ls=":")
    axes[-1].set_xlim(t0 - 5, t1)
    axes[-1].set_xlabel("time (ns), clock 100 MHz in simulation; values are Q4.8 integers (real = value / 256)")
    fig.tight_layout(h_pad=0.2)
    fig.savefig(path)
    plt.close(fig)


def fig_session(path: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from bqi.dmn_ode import StateSpaceDMN
    from model import config as CFG
    from model import hardware as HW
    m = StateSpaceDMN()
    fmt = HW.FixedFormat(8)
    fx = HW.simulate_closed_loop(m, fmt, CFG.CONTROL.n_epochs, seed=CFG.SEED, trace=True)
    fl = HW.simulate_closed_loop(m, None, CFG.CONTROL.n_epochs, seed=CFG.SEED)
    rows = np.array(fx["rows"], dtype=float)
    t = np.arange(len(rows)) * CFG.DT
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 3.8), sharex=True)
    ax[0].plot(t, fx["QoC"], color="#1f77b4", lw=1, label="QoC (brain model, fixed-point controller)")
    ax[0].plot(t, rows[:, 5] / 256, color="#ff7f0e", lw=1, label=r"$\widehat{\rm QoC}$ estimated by the circuit")
    ax[0].axhline(CFG.CONTROL.target_qoc, color="#d62728", ls="--", lw=0.8, label="target 2.0")
    ax[0].set(ylabel="QoC")
    ax[0].legend(frameon=False, fontsize=7, ncol=3, loc="lower right")
    ax[1].step(t, fx["u"], where="post", color="#ff7f0e", lw=1, label="u, circuit Q4.8")
    ax[1].plot(t, fl["u"], color="k", lw=0.7, ls=":", label="u, floating point")
    ax[1].set(xlabel="time (min)", ylabel="u")
    ax[1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"cost_fixed": fx["cost"], "cost_float": fl["cost"],
            "max_abs_du": float(np.max(np.abs(fx["u"] - fl["u"]))),
            "u_min": float(fx["u"].min()), "u_max": float(fx["u"].max())}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    ghdl = find_ghdl()
    ver = run([ghdl, "--version"]).stdout.splitlines()[0]
    rtl = rtl_sim_with_vcd(ghdl)
    net = synthesize(ghdl)
    stats = netlist_stats(net)
    ps = post_synth_sim(ghdl, net)
    fig_waveform(os.path.join(VHDL, "work", "tb_dmn_lqg_core.vcd"), os.path.join(OUT, "fig_waveform.pdf"))
    sess = fig_session(os.path.join(OUT, "fig_session.pdf"))
    res = {"ghdl": ver, "rtl_simulation": rtl, "post_synthesis_simulation": ps, "netlist": stats, "session": sess}
    with open(os.path.join(OUT, "synth_stats.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
