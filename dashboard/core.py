"""
core.py — Thành phần dùng chung cho BQI Dashboard.

  * Đường dẫn dự án + đăng ký 11 module thuật toán (section bài báo, test, dataset)
  * Bảng màu (đã kiểm định CVD) + hàm định dạng biểu đồ Plotly
  * Chạy tiến trình con (main.py / pytest) và stream log trực tiếp lên giao diện
  * Đọc kết quả pytest (JUnit XML) + coverage JSON, lưu lịch sử các lần chạy test
  * Bảng đối chiếu (scorecard) các khẳng định định tính của bài báo, tính trực tiếp
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DASH_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DASH_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "generated")
FIG_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
RES_DIR = os.path.join(PROJECT_ROOT, "outputs", "results")
TESTS_DIR = os.path.join(PROJECT_ROOT, "tests")
BQI_DIR = os.path.join(PROJECT_ROOT, "bqi")
TEST_HISTORY = os.path.join(RES_DIR, "test_history.json")

# =============================================================================
# Bảng màu — categorical theo thứ tự cố định (không xoay vòng), status dành riêng
# =============================================================================
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
GRAY = "#8a8985"
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"
SEQ_BLUE = [[0.0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1.0, "#0d366b"]]
DIVERGING = [[0.0, "#2a78d6"], [0.5, "#f0efec"], [1.0, "#e34948"]]

STATUS_ICON = {"pass": "✅", "warn": "⚠️", "fail": "❌", "passed": "✅", "failed": "❌",
               "error": "💥", "skipped": "⏭️"}
STATUS_COLOR = {"passed": GOOD, "failed": CRITICAL, "error": CRITICAL, "skipped": GRAY}

# =============================================================================
# Đăng ký module: nối mỗi file trong bqi/ với section bài báo, thực nghiệm,
# dataset và file test tương ứng — dùng cho trang Tổng quan và Phòng thí nghiệm.
# =============================================================================
MODULES = [
    dict(key="predictive_coding", sec="3.1", name="Bayesian Predictive Coding", exp="bpc", data=None,
         desc="Năng lượng tự do biến phân F(μ), gradient flow và định lý hội tụ mũ (eq. 2–7, 15–16)."),
    dict(key="attention", sec="3.3", name="SNIS Attention", exp=None, data=None,
         desc="Σ(Q,K,V) = softmax(QKᵀ/√d_k)V và multi-head attention — tầng truy vấn SNIS."),
    dict(key="bqi_algorithm", sec="4.5", name="Algorithm 1 — BQI", exp="bqi", data=None,
         desc="Cập nhật niềm tin 3 pha: sai số dự đoán → cực tiểu F → truy vấn & giải mã SNIS."),
    dict(key="com_algorithm", sec="8.2", name="Algorithm 2 — COM", exp="com", data=None,
         desc="Tối ưu ý thức: ức chế DMN (BCI vòng kín) → tăng Φ → mở rộng truy vấn → QoC."),
    dict(key="iit_phi", sec="5.1", name="IIT — Φ", exp="phi", data=None,
         desc="Φ vét cạn mọi phân hoạch qua TPM bị cắt nhân quả + Earth Mover's Distance (NP-hard)."),
    dict(key="pci", sec="5.2", name="PCI (Lempel–Ziv)", exp="pci", data="pci",
         desc="Độ phức tạp LZ76 trên TMS-EEG nhị phân hoá, PCI = K(s)/L(s) cho 9 trạng thái ý thức."),
    dict(key="dmn_ode", sec="7.2", name="DMN / Φ / QoC ODE", exp="ode", data="ode",
         desc="Hệ ODE 3 biến (solve_ivp), nghiệm dừng dạng đóng, so sánh đối chứng vs thiền giả."),
    dict(key="meta_analysis", sec="7.3", name="Meta-analysis", exp="meta", data="meta",
         desc="Gộp hiệu ứng ngẫu nhiên DerSimonian–Laird trên 16 nghiên cứu, I², bootstrap."),
    dict(key="hopfield", sec="9.2", name="Hopfield: classic vs modern", exp="hopfield", data=None,
         desc="Dung lượng tuyến tính ~0.14n vs mũ ~e^{n/2}; tương đương Transformer attention."),
    dict(key="spectral_graph", sec="4.4 / 9.4", name="Đồ thị phổ + Kuramoto", exp="spectral", data="kuramoto",
         desc="Mạng small-world Watts–Strogatz, giá trị Fiedler, σ_SW, E_glob, đồng bộ Kuramoto."),
    dict(key="quantum_decoherence", sec="6", name="Quantum decoherence", exp="quantum", data="quantum",
         desc="Entropy von Neumann dưới khử kết hợp Lindblad, thời gian sụp đổ Orch-OR τ = ħ/E_G."),
]
MODULE_BY_KEY = {m["key"]: m for m in MODULES}

EXPERIMENT_LABELS = {
    "bpc": "BPC phân tầng", "bqi": "Algorithm 1 (BQI)", "com": "Algorithm 2 (COM)", "phi": "IIT Φ",
    "pci": "PCI", "ode": "ODE DMN/Φ/QoC", "meta": "Meta-analysis", "hopfield": "Hopfield",
    "spectral": "Đồ thị + Kuramoto", "quantum": "Quantum",
}
DATASET_LABELS = {
    "pci": "PCI theo trạng thái", "ode": "Tham số ODE", "meta": "Meta-analysis bootstrap",
    "kuramoto": "Kuramoto trials", "quantum": "Quantum decoherence", "reference_tables": "Bảng tham chiếu",
}


# =============================================================================
# Plotly styling
# =============================================================================
def style(fig: go.Figure, title: str | None = None, height: int = 360,
          xlab: str | None = None, ylab: str | None = None, legend: bool = True) -> go.Figure:
    grid = "rgba(128,128,128,0.18)"
    fig.update_layout(
        title=dict(text=title, font=dict(size=15), x=0, xanchor="left") if title else None,
        height=height,
        margin=dict(l=10, r=10, t=48 if title else 16, b=10),
        hovermode="closest",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title=None),
        hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(title_text=xlab, gridcolor=grid, zeroline=False, showline=False)
    fig.update_yaxes(title_text=ylab, gridcolor=grid, zeroline=False, showline=False)
    return fig


def show(fig: go.Figure, key: str | None = None):
    st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False})


def fmt(x, digits: int = 4) -> str:
    """Định dạng số gọn: dùng ký hiệu khoa học cho giá trị rất lớn/rất nhỏ."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if x == 0:
        return "0"
    if abs(x) >= 1e5 or abs(x) < 1e-3:
        return f"{x:.{max(digits - 2, 1)}e}"
    return f"{x:.{digits}g}"


# =============================================================================
# Giới thiệu mã nguồn (AST) — đếm dòng, hàm, lớp, test; phụ thuộc nội bộ
# =============================================================================
@st.cache_data(show_spinner=False)
def introspect_codebase(_mtime_key: float) -> dict:
    modules = {}
    for m in MODULES:
        path = os.path.join(BQI_DIR, f"{m['key']}.py")
        src = open(path, encoding="utf-8").read()
        tree = ast.parse(src)
        funcs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
        classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
        deps = sorted({n.module for n in ast.walk(tree)
                       if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module})
        modules[m["key"]] = dict(loc=len(src.splitlines()), functions=funcs, classes=classes, deps=deps)

    tests = {}
    for f in sorted(os.listdir(TESTS_DIR)):
        if f.startswith("test_") and f.endswith(".py"):
            tree = ast.parse(open(os.path.join(TESTS_DIR, f), encoding="utf-8").read())
            tests[f] = [dict(name=n.name, doc=(ast.get_docstring(n) or "").strip())
                        for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
    return dict(modules=modules, tests=tests)


def codebase() -> dict:
    files = [os.path.join(BQI_DIR, f) for f in os.listdir(BQI_DIR) if f.endswith(".py")]
    files += [os.path.join(TESTS_DIR, f) for f in os.listdir(TESTS_DIR) if f.endswith(".py")]
    return introspect_codebase(max(os.path.getmtime(f) for f in files))


def test_file_for(module_key: str) -> str:
    return f"test_{module_key}.py"


# =============================================================================
# Tiến trình con — stream stdout trực tiếp lên một placeholder
# =============================================================================
def run_streaming(args: list[str], placeholder, max_lines: int = 400) -> tuple[int, str, float]:
    """Chạy `python <args>` trong thư mục dự án, cập nhật log lên `placeholder`
    theo thời gian thực. Trả về (exit_code, toàn bộ log, thời gian chạy)."""
    env = os.environ.copy()
    env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    t0 = time.time()
    proc = subprocess.Popen([sys.executable, *args], cwd=PROJECT_ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    lines, last = [], 0.0
    for line in proc.stdout:
        lines.append(line.rstrip("\n"))
        if time.time() - last > 0.2:
            placeholder.code("\n".join(lines[-max_lines:]) or "…", language="text")
            last = time.time()
    proc.wait()
    log = "\n".join(lines)
    placeholder.code("\n".join(lines[-max_lines:]) or "(không có output)", language="text")
    return proc.returncode, log, time.time() - t0


# =============================================================================
# Pytest — chạy, đọc JUnit XML + coverage, lưu lịch sử
# =============================================================================
def has_pytest_cov() -> bool:
    import importlib.util
    return importlib.util.find_spec("pytest_cov") is not None


def run_pytest(targets: list[str], placeholder, keyword: str | None = None,
               stop_on_fail: bool = False, coverage: bool = False) -> dict:
    tmp = tempfile.mkdtemp(prefix="bqi_dash_")
    xml_path = os.path.join(tmp, "junit.xml")
    cov_path = os.path.join(tmp, "coverage.json")
    args = ["-m", "pytest", *targets, "-v", "-p", "no:cacheprovider", f"--junitxml={xml_path}",
            "-o", "junit_family=xunit1", "--color=no"]
    if keyword:
        args += ["-k", keyword]
    if stop_on_fail:
        args += ["-x"]
    if coverage:
        args += ["--cov=bqi", f"--cov-report=json:{cov_path}", "--cov-report="]
    code, log, elapsed = run_streaming(args, placeholder)

    df = parse_junit(xml_path) if os.path.isfile(xml_path) else pd.DataFrame(
        columns=["file", "test", "status", "duration", "message", "details"])
    cov = parse_coverage(cov_path) if coverage and os.path.isfile(cov_path) else None
    result = dict(exit_code=code, log=log, elapsed=elapsed, results=df, coverage=cov,
                  timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                  targets=targets, keyword=keyword)
    _append_history(result)
    return result


def parse_junit(xml_path: str) -> pd.DataFrame:
    root = ET.parse(xml_path).getroot()
    rows = []
    for tc in root.iter("testcase"):
        status, message, details = "passed", "", ""
        for tag in ("failure", "error", "skipped"):
            node = tc.find(tag)
            if node is not None:
                status = {"failure": "failed", "error": "error", "skipped": "skipped"}[tag]
                message = node.get("message", "") or ""
                details = node.text or ""
                break
        file_attr = tc.get("file")
        classname = tc.get("classname", "")
        file = os.path.basename(file_attr) if file_attr else classname.split(".")[-1] + ".py"
        rows.append(dict(file=file, test=tc.get("name", ""), status=status,
                         duration=float(tc.get("time", 0) or 0), message=message, details=details))
    return pd.DataFrame(rows)


def parse_coverage(cov_path: str) -> pd.DataFrame:
    data = json.load(open(cov_path, encoding="utf-8"))
    rows = []
    for path, info in data.get("files", {}).items():
        s = info["summary"]
        rows.append(dict(module=os.path.basename(path), statements=s["num_statements"],
                         missing=s["missing_lines"], percent=s["percent_covered"],
                         missing_lines=", ".join(map(str, info.get("missing_lines", [])[:40]))))
    return pd.DataFrame(rows).sort_values("percent")


def _append_history(result: dict):
    df = result["results"]
    entry = dict(timestamp=result["timestamp"], elapsed=round(result["elapsed"], 2),
                 exit_code=result["exit_code"], total=int(len(df)),
                 passed=int((df["status"] == "passed").sum()) if len(df) else 0,
                 failed=int(df["status"].isin(["failed", "error"]).sum()) if len(df) else 0,
                 skipped=int((df["status"] == "skipped").sum()) if len(df) else 0,
                 scope=", ".join(os.path.basename(t) for t in result["targets"]),
                 keyword=result["keyword"] or "")
    hist = load_test_history()
    hist.append(entry)
    os.makedirs(RES_DIR, exist_ok=True)
    with open(TEST_HISTORY, "w", encoding="utf-8") as f:
        json.dump(hist[-200:], f, indent=2, ensure_ascii=False)


def load_test_history() -> list[dict]:
    try:
        with open(TEST_HISTORY, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return []


# =============================================================================
# Dữ liệu & kết quả trên đĩa
# =============================================================================
def list_datasets() -> pd.DataFrame:
    if not os.path.isdir(DATA_DIR):
        return pd.DataFrame(columns=["file", "rows", "cols", "size_kb", "modified"])
    rows = []
    for f in sorted(os.listdir(DATA_DIR)):
        if f.endswith(".csv"):
            p = os.path.join(DATA_DIR, f)
            try:
                head = pd.read_csv(p, nrows=0)
                n_rows = sum(1 for _ in open(p, encoding="utf-8")) - 1
            except Exception:
                head, n_rows = pd.DataFrame(), -1
            rows.append(dict(file=f, rows=n_rows, cols=len(head.columns),
                             size_kb=round(os.path.getsize(p) / 1024, 1),
                             modified=datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M")))
    return pd.DataFrame(rows)


def list_figures() -> list[str]:
    if not os.path.isdir(FIG_DIR):
        return []
    return sorted(os.path.join(FIG_DIR, f) for f in os.listdir(FIG_DIR) if f.lower().endswith(".png"))


def load_summary() -> dict | None:
    p = os.path.join(RES_DIR, "summary.json")
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def flatten(d, prefix: str = "") -> dict:
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(d, list) and all(isinstance(x, (int, float)) for x in d) and len(d) <= 8:
        for i, v in enumerate(d):
            out[f"{prefix}[{i}]"] = v
    else:
        out[prefix] = d
    return out


# =============================================================================
# Lưu kết quả khám phá tương tác (mọi phép tính chạy trên dashboard)
# =============================================================================
# Hai không gian lưu tách biệt: "model" = bài Q1 (pipeline mô hình), "bqi" = tái hiện bài BQI gốc
EXPLORE_SCOPES = {"model": os.path.join(RES_DIR, "model", "explorations"),
                  "bqi": os.path.join(RES_DIR, "bqi", "explorations")}
EXPLORE_DIR = EXPLORE_SCOPES["model"]


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def save_exploration(kind: str, inputs: dict, outputs: dict, arrays: dict | None = None,
                     scope: str = "model", frames: dict | None = None) -> str:
    """Ghi một lần cho mỗi bộ tham số (khoá = hash của `inputs`) vào không gian `scope`:
    <kind>/<key>.json (inputs, outputs, thời điểm), <kind>/<key>.npz (mảng, nếu có),
    <kind>/<key>_<tên>.csv (DataFrame, nếu có), và một dòng trong explorations/index.csv.
    Trả về đường dẫn file JSON."""
    import hashlib
    base = EXPLORE_SCOPES[scope]
    inputs = _jsonable(inputs)
    key = hashlib.sha1(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:10]
    folder = os.path.join(base, kind)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{key}.json")
    if os.path.isfile(path):
        return path
    rec = {"scope": scope, "kind": kind, "key": key, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "inputs": inputs, "outputs": _jsonable(outputs), "arrays": None, "frames": {}}
    if arrays:
        npz = os.path.join(folder, f"{key}.npz")
        np.savez_compressed(npz, **{k: np.asarray(v) for k, v in arrays.items()})
        rec["arrays"] = os.path.relpath(npz, PROJECT_ROOT)
    for name, df in (frames or {}).items():
        csv = os.path.join(folder, f"{key}_{name}.csv")
        df.to_csv(csv, index=False)
        rec["frames"][name] = os.path.relpath(csv, PROJECT_ROOT)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    idx = os.path.join(base, "index.csv")
    row = pd.DataFrame([{"time": rec["time"], "kind": kind, "key": key,
                         "inputs": json.dumps(inputs, ensure_ascii=False),
                         "file": os.path.relpath(path, PROJECT_ROOT)}])
    row.to_csv(idx, mode="a", header=not os.path.isfile(idx), index=False)
    return path


def list_explorations(scope: str = "model") -> pd.DataFrame:
    idx = os.path.join(EXPLORE_SCOPES[scope], "index.csv")
    if not os.path.isfile(idx):
        return pd.DataFrame(columns=["time", "kind", "key", "inputs", "file"])
    return pd.read_csv(idx).iloc[::-1]


def _split_result(obj, prefix: str = "") -> tuple[dict, dict, dict]:
    """Tách kết quả bất kỳ thành (số liệu JSON, mảng numpy, DataFrame) để lưu."""
    scalars, arrays, frames = {}, {}, {}
    name = prefix or "value"
    if isinstance(obj, pd.DataFrame):
        frames[name] = obj
    elif isinstance(obj, np.ndarray):
        if obj.size <= 16:
            scalars[name] = obj.tolist()
        else:
            arrays[name] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            s, a, f = _split_result(v, f"{prefix}.{k}" if prefix else str(k))
            scalars.update(s); arrays.update(a); frames.update(f)
    elif isinstance(obj, (list, tuple)):
        if obj and all(isinstance(v, (int, float, np.floating, np.integer, bool)) for v in obj):
            if len(obj) <= 16:
                scalars[name] = [float(v) for v in obj]
            else:
                arrays[name] = np.asarray(obj, dtype=float)
        elif obj and all(isinstance(v, np.ndarray) for v in obj) and len(obj) <= 32:
            for i, v in enumerate(obj):
                s, a, f = _split_result(v, f"{name}[{i}]")
                scalars.update(s); arrays.update(a); frames.update(f)
        else:
            for i, v in enumerate(obj):
                s, a, f = _split_result(v, f"{name}[{i}]")
                scalars.update(s); arrays.update(a); frames.update(f)
    elif isinstance(obj, (int, float, str, bool, type(None), np.floating, np.integer, np.bool_)):
        scalars[name] = _jsonable(obj)
    else:
        scalars[name] = repr(obj)[:200]
    return scalars, arrays, frames


def saving(kind: str, scope: str = "bqi"):
    """Decorator: lưu đầu vào (tham số theo tên) và đầu ra của hàm tính toán vào `scope`.
    Đặt DƯỚI @st.cache_data để chỉ ghi khi thực sự tính (cache miss)."""
    import functools
    import inspect

    def deco(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                scalars, arrays, frames = _split_result(result)
                save_exploration(kind, dict(bound.arguments), scalars, arrays or None, scope=scope,
                                 frames=frames or None)
            except Exception as e:  # lưu thất bại không được làm hỏng giao diện
                scalars = {"save_error": repr(e)}
            return result
        return wrapper
    return deco


# =============================================================================
# Scorecard — đối chiếu các khẳng định định tính của bài báo (tính trực tiếp)
# =============================================================================
def _check(module, claim, measured, expected, status, note=""):
    return dict(module=module, claim=claim, measured=measured, expected=expected, status=status, note=note)


@st.cache_data(show_spinner="Đang tính scorecard đối chiếu bài báo…")
def paper_scorecard(seed: int = 42) -> list[dict]:
    from scipy.stats import spearmanr
    from bqi.predictive_coding import HierarchicalBPC
    from bqi.bqi_algorithm import BQIModel, BQIConfig
    from bqi.com_algorithm import COMConfig, run_com_algorithm
    from bqi.iit_phi import integrated_information_Phi
    from bqi.pci import generate_pci_dataset, PCI_TABLE_REFERENCE
    from bqi.dmn_ode import ODE_PARAMS_REFERENCE, steady_states, simulate
    from bqi.meta_analysis import random_effects_meta_analysis
    from bqi.hopfield import retrieval_error_experiment
    from bqi.spectral_graph import (build_small_world_brain_graph, small_world_index, global_efficiency,
                                    meditation_graph_transform, generate_kuramoto_dataset)
    from bqi.quantum_decoherence import simulate_entropy_dynamics

    checks = []

    # 1. BPC phân tầng hội tụ
    bpc = HierarchicalBPC(n_levels=3, dims=[6, 6, 4, 3], seed=seed)
    E = bpc.run(np.random.default_rng(seed + 1).normal(0, 1, 6), n_steps=200, eta=0.05)
    ratio = E[-1] / E[0]
    checks.append(_check("predictive_coding", "Năng lượng sai số dự đoán phân tầng → 0",
                         f"{fmt(E[0])} → {fmt(E[-1])} (×{fmt(ratio)})", "giảm ≥ 1000×",
                         "pass" if ratio < 1e-3 else "fail"))

    E_other = HierarchicalBPC(n_levels=3, dims=[6, 6, 4, 3], seed=seed).run(np.full(6, 50.0), n_steps=200,
                                                                           eta=0.05)
    checks.append(_check("predictive_coding", "BPC: niềm tin phụ thuộc vào quan sát o (μ⁽⁰⁾ = o)",
                         f"o ~ N(0,1) vs o = 50: năng lượng {'giống hệt' if np.allclose(E, E_other) else 'khác nhau'}",
                         "khác nhau", "fail" if np.allclose(E, E_other) else "pass",
                         "HierarchicalBPC.step không dùng mu_full[0] = o: ε⁽⁰⁾ = o − f(μ⁽¹⁾) bị bỏ qua, "
                         "nên năng lượng → 0 chỉ vì μ co về 0."))

    # 2. BQI Algorithm 1 — F giảm đơn điệu
    cfg1 = BQIConfig(n_levels=3, dims=(8, 8, 6, 4), n_snis=64, d_snis=4, d_k=4, T=100, seed=seed)
    res = BQIModel(cfg1).run(np.random.default_rng(seed + 2).normal(0, 1, 8))
    F = res["free_energy_trace"]
    mono = bool(np.all(np.diff(F) <= 1e-9))
    checks.append(_check("bqi_algorithm", "Algorithm 1: F giảm đơn điệu về cực tiểu",
                         f"F: {fmt(F[0])} → {fmt(F[-1])}, đơn điệu={mono}", "đơn điệu, F_cuối < F_đầu",
                         "pass" if mono and F[-1] < F[0] else "fail"))
    res_other = BQIModel(cfg1).run(np.full(8, 50.0))
    same = np.allclose(res["free_energy_trace"], res_other["free_energy_trace"]) and np.allclose(
        res["mu_star"], res_other["mu_star"])
    eps_norm = sum(float(np.linalg.norm(e)) for e in res["prediction_errors"])
    checks.append(_check("bqi_algorithm", "Algorithm 1: μ* và F phụ thuộc vào quan sát o",
                         f"o ~ N(0,1) vs o = 50: {'giống hệt' if same else 'khác nhau'}; "
                         f"Σ‖ε Phase 1‖ = {fmt(eps_norm)}", "khác nhau, ε ≠ 0", "fail" if same else "pass",
                         "BQIModel.free_energy không dùng mu_full[0] = o; Phase 1 tính ε = Π(μ − Iμ) ≡ 0."))

    # 3. COM Algorithm 2
    com = run_com_algorithm(COMConfig(T=120, d_mu=6, n_snis=30, d_snis=6, d_k=6, seed=seed + 1))
    tr = com["trace"]
    ok_dir = tr["A_dmn"][-1] < tr["A_dmn"][0] and tr["Phi"][-1] > tr["Phi"][0]
    near = abs(com["QoC_final"] - 1.0) < 0.1
    checks.append(_check("com_algorithm", "COM: DMN ↓, Φ ↑, QoC → mục tiêu (1.0)",
                         f"A_DMN {fmt(tr['A_dmn'][0])}→{fmt(tr['A_dmn'][-1])}, Φ {fmt(tr['Phi'][0])}→"
                         f"{fmt(tr['Phi'][-1])}, QoC={fmt(com['QoC_final'])}", "|QoC − 1| < 0.1",
                         "pass" if ok_dir and near else ("warn" if ok_dir else "fail")))

    # 4. IIT — hệ ngắt kết nối có Φ = 0, hệ kết nối có Φ > 0
    rng = np.random.default_rng(seed + 3)
    W = rng.normal(0, 1.8, (5, 5)); np.fill_diagonal(W, 0)
    st_ = rng.integers(0, 2, 5)
    phi_c = integrated_information_Phi(W, st_)["Phi"]
    phi_0 = integrated_information_Phi(np.zeros((4, 4)), np.array([1, 0, 1, 0]))["Phi"]
    checks.append(_check("iit_phi", "Φ > 0 khi tích hợp, Φ = 0 khi ngắt kết nối",
                         f"Φ(kết nối, n=5)={fmt(phi_c)}, Φ(W=0)={fmt(phi_0)}", "Φ>0 và Φ≈0",
                         "pass" if phi_c > 0 and phi_0 < 1e-6 else "fail"))

    # 5. PCI — thứ tự trạng thái & thang đo
    df = generate_pci_dataset(n_channels=16, n_timepoints=48, n_subjects_per_state=8, seed=seed + 4)
    means = df.groupby("state")["pci"].mean()
    states = list(PCI_TABLE_REFERENCE)
    rho = spearmanr([means[s] for s in states], [PCI_TABLE_REFERENCE[s]["mean"] for s in states]).statistic
    checks.append(_check("pci", "PCI: thứ tự trạng thái khớp Table PCI",
                         f"Spearman ρ = {rho:.3f}", "ρ ≥ 0.8",
                         "pass" if rho >= 0.8 else ("warn" if rho >= 0.5 else "fail"),
                         "Các trạng thái ý thức cao (thức, thiền, NDE) bị bão hoà gần nhau."))
    from bqi.pci import pci_bqi_literal, simulate_tms_eeg
    lit = pci_bqi_literal(simulate_tms_eeg(16, 48, 0.9, seed=seed))
    checks.append(_check("pci", "PCI nằm trong thang [0, 1] như Table PCI",
                         f"Casali 2013: PCI ∈ [{means.min():.2f}, {means.max():.2f}]; "
                         f"eq. 49 nguyên văn: {lit:.1f}", "0 ≤ PCI ≲ 1",
                         "pass" if means.max() <= 1.1 else "warn",
                         "Eq. 49 để L(s) không xác định; đọc nguyên văn cho ≈ log₂|s|. Mã dùng chuẩn hoá theo "
                         "entropy nguồn của Casali 2013 (LZ hữu hạn có thể vượt 1 nhẹ)."))

    # 6. ODE — thiền giả vs đối chứng, và nghiệm dừng dạng đóng
    P = ODE_PARAMS_REFERENCE
    c = simulate(P, meditator=False, seed=seed + 1)
    m = simulate(P, meditator=True, seed=seed + 1)
    checks.append(_check("dmn_ode", "Thiền giả: A_DMN thấp hơn, Φ cao hơn đối chứng",
                         f"A_DMN {fmt(m['A_DMN'][-1])} vs {fmt(c['A_DMN'][-1])}; Φ {fmt(m['Phi'][-1])} vs "
                         f"{fmt(c['Phi'][-1])}", "A_med < A_ctrl, Φ_med > Φ_ctrl",
                         "pass" if m["A_DMN"][-1] < c["A_DMN"][-1] and m["Phi"][-1] > c["Phi"][-1] else "fail"))
    checks.append(_check("dmn_ode", "Thiền giả đạt QoC cao hơn đối chứng (t = 40)",
                         f"QoC {fmt(m['QoC'][-1])} vs {fmt(c['QoC'][-1])}", "QoC_med > QoC_ctrl",
                         "pass" if m["QoC"][-1] > c["QoC"][-1] else "warn",
                         "dQoC/dt chứa −β·dA/dt: khi A_DMN giảm nhanh QoC tăng vọt rồi tắt dần với γ."))
    from bqi.dmn_ode import steady_states_paper
    ss_paper = steady_states_paper(P)
    long = simulate(P, t_span=(0, 400), n_points=4000, meditator=False, seed=seed + 1)
    qoc_long = float(np.mean(long["QoC"][-1000:]))
    checks.append(_check("dmn_ode", "Nghiệm dừng QoC* của bài (eq. 57) khớp mô phỏng dài",
                         f"Bài báo: QoC* = {fmt(ss_paper['QoC_star'])}; mô phỏng t→400 ≈ {fmt(qoc_long)}; "
                         f"đã sửa: QoC* = {fmt(steady_states(P)['QoC_star'])}", "sai lệch < 10%",
                         "pass" if abs(qoc_long - ss_paper["QoC_star"]) < 0.1 * abs(ss_paper["QoC_star"]) else "fail",
                         "Eq. 54 chỉ phụ thuộc đạo hàm của Φ và A nên γ·QoC* = 0 ⇒ QoC* = 0. "
                         "`steady_states()` đã sửa; mô hình có QoC* ≠ 0 là `StateSpaceDMN`."))

    # 7. Meta-analysis
    meta = random_effects_meta_analysis()
    d_ok = 1.5 <= meta["pooled_d"] <= 1.75
    i_ok = 25 <= meta["I2_percent"] <= 45
    checks.append(_check("meta_analysis", "Pooled d ≈ 1.70, I² ≈ 31% (Table meta)",
                         f"d = {meta['pooled_d']:.3f} [{meta['ci'][0]:.2f}, {meta['ci'][1]:.2f}], "
                         f"I² = {meta['I2_percent']:.1f}%", "d ∈ [1.5, 1.75], I² ∈ [25, 45]%",
                         "pass" if d_ok and i_ok else "warn",
                         "Phương sai d xấp xỉ 1/n + d²/2n nên kết quả lệch nhẹ so với bài báo."))

    # 8. Hopfield
    n_hop = 30
    hop = retrieval_error_experiment(n=n_hop, pattern_counts=[2, 21, 100], trials_per_count=10, seed=seed)
    row = hop.set_index("n_patterns").loc[21]
    checks.append(_check("hopfield", "Modern Hopfield vẫn ~0 lỗi khi vượt dung lượng classic (0.14n)",
                         f"P=21 (≈5×0.14n): classic {row.classic_error_rate:.3f}, modern {row.modern_error_rate:.3f}",
                         "modern < 0.05 < classic",
                         "pass" if row.modern_error_rate < 0.05 < row.classic_error_rate else "fail"))

    # 9. Đồ thị & Kuramoto
    G = build_small_world_brain_graph(n_nodes=80, k=6, p_rewire=0.1, seed=seed)
    sigma = small_world_index(G, n_random=4)
    E0 = global_efficiency(G)
    E1 = global_efficiency(meditation_graph_transform(G, boost_fraction=0.08, seed=seed + 1))
    checks.append(_check("spectral_graph", "Mạng small-world (σ > 1); thiền tăng E_glob",
                         f"σ = {sigma:.2f}; E_glob {E0:.3f} → {E1:.3f} (+{100 * (E1 - E0) / E0:.1f}%)",
                         "σ > 1, ΔE_glob > 0", "pass" if sigma > 1 and E1 > E0 else "fail"))
    kd = generate_kuramoto_dataset(n_oscillators=40, n_trials=4, seed=seed + 3)
    rm = kd.groupby("state")["r"].mean()
    checks.append(_check("spectral_graph", "Đồng bộ Kuramoto: thiền sâu > gây mê propofol",
                         f"r̄ thiền sâu = {rm['Deep meditation']:.3f}, propofol = {rm['Propofol anaesthesia']:.3f}",
                         "r_thiền > r_propofol",
                         "pass" if rm["Deep meditation"] > rm["Propofol anaesthesia"] else "fail"))

    # 10. Quantum
    sl = simulate_entropy_dynamics(dim=4, gamma=0.5, T=1.0, dt=0.01, seed=seed)
    fa = simulate_entropy_dynamics(dim=4, gamma=10.0, T=1.0, dt=0.01, seed=seed)
    mono_q = bool(np.all(np.diff(fa["S"]) >= -1e-10) and np.all(np.diff(sl["S"]) >= -1e-10))
    checks.append(_check("quantum_decoherence", "S(ρ) tăng đơn điệu về ln(dim), nhanh hơn với γ lớn",
                         f"S(t=1): γ=0.5 → {sl['S'][-1]:.3f}, γ=10 → {fa['S'][-1]:.3f}; ln4 = {np.log(4):.3f}",
                         "đơn điệu, S ≤ ln(dim), S_nhanh > S_chậm",
                         "pass" if mono_q and fa["S"][-1] > sl["S"][-1] and fa["S"][-1] <= np.log(4) + 1e-9
                         else "fail"))
    return checks
