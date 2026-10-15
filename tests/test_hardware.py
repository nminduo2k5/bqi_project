"""Tests for model/hardware.py: the fixed-point neural circuit of the M1 closed loop."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bqi.dmn_ode import StateSpaceDMN
from model import config as CFG
from model import hardware as HW
from model.control import simulate_policy, solve_lqg, state_space_with_control


def test_neuron_rounding_and_saturation():
    f = HW.FixedFormat(4)                      # W = 9, scale 16, range [-256, 255]
    # 1.5 * 2 + 0.25 = 3.25 -> 52
    assert HW.neuron(f, [24], [32], 4) == 52
    # round half up: 0.5 LSB ties go up, negative ties go towards +inf (floor of acc + half)
    assert HW.neuron(f, [1], [8], 0) == 1      # 8/16 = 0.5 LSB -> 1
    assert HW.neuron(f, [-1], [8], 0) == 0     # -0.5 LSB -> 0
    assert HW.neuron(f, [-3], [8], 0) == -1    # -1.5 LSB -> -1
    # saturation
    assert HW.neuron(f, [255], [255], 0) == 255
    assert HW.neuron(f, [-256], [255], 0) == -256
    assert HW.clip(f, 300, 32) == 32 and HW.clip(f, -300, 32) == -32


def test_float_loop_matches_control_module():
    """The float path of simulate_closed_loop reproduces model.control.simulate_policy exactly."""
    m = StateSpaceDMN()
    st = state_space_with_control(m)
    ctrl = solve_lqg(st)
    for seed in (CFG.SEED, CFG.SEED + 3):
        ref = simulate_policy(m, ctrl, "lqg", CFG.CONTROL.n_epochs, seed=seed, sys_true=st)
        mine = HW.simulate_closed_loop(m, None, CFG.CONTROL.n_epochs, seed=seed)
        assert abs(ref["cost"] - mine["cost"]) < 1e-12
        assert np.allclose(ref["u"], mine["u"], atol=1e-12)


def test_fixed_point_close_to_float_at_high_precision():
    m = StateSpaceDMN()
    fmt = HW.FixedFormat(16)
    ref = HW.simulate_closed_loop(m, None, 270, seed=CFG.SEED)
    fx = HW.simulate_closed_loop(m, fmt, 270, seed=CFG.SEED)
    assert np.max(np.abs(ref["u"] - fx["u"])) < 5e-3
    assert fx["cost"] < 1.01 * ref["cost"] + 1e-3


def test_circuit_is_deterministic_and_trace_consistent():
    m = StateSpaceDMN()
    fmt = HW.FixedFormat(12)
    a = HW.simulate_closed_loop(m, fmt, 50, seed=7, trace=True)
    b = HW.simulate_closed_loop(m, fmt, 50, seed=7, trace=True)
    assert a["rows"] == b["rows"] and len(a["rows"]) == 50
    # replaying the recorded y through a fresh circuit gives the recorded outputs (what the VHDL testbench does)
    circ = HW.NeuralCircuit(fmt)
    for ya, yp, xa, xp, u, q in a["rows"]:
        assert circ.step(ya, yp) == (xa, xp, u, q)


def test_generated_package_and_vectors(tmp_path):
    res = HW.run(seed=CFG.SEED, fast=True, verbose=False, output_dir=str(tmp_path), gen_dir=str(tmp_path / "gen"))
    pkg = open(res["files"]["vhdl_package"], encoding="utf-8").read()
    assert f"constant W    : natural := {res['chosen_width']};" in pkg
    assert f"constant FRAC : natural := {res['chosen_frac']};" in pkg
    for name in ("W1_A", "W1_PHI", "W2_A", "W2_PHI", "W3", "W4", "MU_Q", "UMAX_Q"):
        assert f"constant {name}" in pkg
    # every line is VHDL code, a comment, or blank (a wrapped comment once broke GHDL analysis)
    for line in pkg.splitlines():
        s = line.strip()
        assert s == "" or s.startswith("--") or s[0].isalpha(), line
    lines = [l for l in open(res["files"]["vectors"], encoding="utf-8").read().splitlines() if not l.startswith("#")]
    assert int(lines[0]) == CFG.CONTROL.n_epochs and len(lines) == CFG.CONTROL.n_epochs + 1
    assert all(len(l.split()) == 6 for l in lines[1:])
    assert os.path.isfile(tmp_path / "hardware_results.json")
    assert res["chosen"]["cost_ratio"] <= 1 + HW.MAX_COST_EXCESS
