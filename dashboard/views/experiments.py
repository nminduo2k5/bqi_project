"""Trang Thực nghiệm — chạy experiments/run_all.py qua CLI, xem summary.json và hình đã xuất."""
import json
import os
import shutil
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core
from core import BLUE, ORANGE, GRAY, fmt, show, style

st.title("📈 Thực nghiệm")
st.caption("10 thực nghiệm tái hiện kết quả chính của bài báo qua `python main.py run-experiments` — "
           "xuất hình PNG vào `outputs/figures/` và số liệu vào `outputs/results/summary.json`.")

EXPS = list(core.EXPERIMENT_LABELS)
# Thứ tự hình 01–09 trong run_all.py (IIT Φ không xuất hình)
FIG_EXP = ["bpc", "bqi", "com", "pci", "ode", "meta", "hopfield", "spectral", "quantum"]
SUMMARY_KEY = {"bpc": "hierarchical_bpc", "bqi": "bqi_algorithm", "com": "com_algorithm", "phi": "iit_phi",
               "pci": "pci", "ode": "dmn_ode", "meta": "meta_analysis", "hopfield": "hopfield",
               "spectral": "spectral_graph", "quantum": "quantum_decoherence"}
EXP_MODULE = {m["exp"]: m for m in core.MODULES if m["exp"]}

# =============================================================================
# Chạy
# =============================================================================
with st.form("run"):
    a, b, c = st.columns([3, 1, 1])
    only = a.multiselect("Thực nghiệm", EXPS, format_func=core.EXPERIMENT_LABELS.get, placeholder="Tất cả")
    seed = b.number_input("Seed", value=42, step=1)
    no_fig = c.toggle("Không xuất hình", help="--no-figures: chỉ ghi summary.json")
    go_run = st.form_submit_button("Chạy thực nghiệm", type="primary", icon=":material/play_arrow:")

if go_run:
    # run_all ghi đè summary.json bằng đúng các thực nghiệm được chọn, nên khi chạy một phần
    # ta ghi vào thư mục tạm rồi gộp vào summary hiện có để không mất kết quả cũ.
    tmp_res = tempfile.mkdtemp(prefix="bqi_exp_") if only else None
    args = ["main.py", "run-experiments", "--seed", str(int(seed))]
    if only:
        args += ["--only", ",".join(only), "--res-dir", tmp_res]
    if no_fig:
        args += ["--no-figures"]
    st.code("python " + " ".join(args), language="bash")
    with st.status("Đang chạy thực nghiệm…", expanded=True) as status:
        code, log, elapsed = core.run_streaming(args, st.empty())
        if code == 0 and tmp_res:
            new = json.load(open(os.path.join(tmp_res, "summary.json"), encoding="utf-8"))
            merged = {**(core.load_summary() or {}), **new}
            os.makedirs(core.RES_DIR, exist_ok=True)
            with open(os.path.join(core.RES_DIR, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2, default=str)
        if tmp_res:
            shutil.rmtree(tmp_res, ignore_errors=True)
        status.update(label=f"{'Hoàn tất' if code == 0 else 'Lỗi'} sau {elapsed:.1f}s (exit {code})",
                      state="complete" if code == 0 else "error", expanded=code != 0)
    st.session_state["exp_last"] = dict(code=code, elapsed=elapsed, seed=int(seed), only=only or EXPS,
                                        time=datetime.now().strftime("%H:%M:%S"))

summary = core.load_summary()
figures = core.list_figures()
if not summary and not figures:
    st.info("Chưa có kết quả — bấm **Chạy thực nghiệm** để tạo `summary.json` và các hình.")
    st.stop()

sum_path = os.path.join(core.RES_DIR, "summary.json")
c = st.columns(4)
c[0].metric("Thực nghiệm có kết quả", f"{len(summary or {})}/{len(EXPS)}")
c[1].metric("Hình đã xuất", len(figures))
c[2].metric("summary.json cập nhật",
            datetime.fromtimestamp(os.path.getmtime(sum_path)).strftime("%Y-%m-%d %H:%M")
            if os.path.isfile(sum_path) else "—")
if "exp_last" in st.session_state:
    r = st.session_state["exp_last"]
    c[3].metric("Lần chạy gần nhất", f"{r['elapsed']:.1f}s", help=f"{r['time']} · seed {r['seed']}",
                delta="thành công" if r["code"] == 0 else f"exit {r['code']}",
                delta_color="normal" if r["code"] == 0 else "inverse")

t_res, t_fig, t_raw = st.tabs(["Kết quả & đối chiếu", "Hình", "summary.json"])

# =============================================================================
# Kết quả chính
# =============================================================================
with t_res:
    s = summary or {}

    def card(exp, metrics, note=None):
        m = EXP_MODULE[exp]
        with st.container(border=True):
            st.markdown(f"**{core.EXPERIMENT_LABELS[exp]}** :gray[§{m['sec']}]")
            if SUMMARY_KEY[exp] not in s:
                st.caption("Chưa chạy.")
                return
            cols = st.columns(len(metrics))
            for col, (label, value, hint) in zip(cols, metrics):
                col.metric(label, value, help=hint)
            if note:
                st.caption(note)

    def g(*path, default=None):
        d = s
        for p in path:
            if not isinstance(d, dict) or p not in d:
                return default
            d = d[p]
        return d

    row = st.columns(3)
    with row[0]:
        card("bpc", [("E đầu", fmt(g("hierarchical_bpc", "energy_start")), None),
                     ("E cuối", fmt(g("hierarchical_bpc", "energy_end")), None)])
    with row[1]:
        card("bqi", [("F cuối", fmt(g("bqi_algorithm", "F_final")), None),
                     ("Số vòng", g("bqi_algorithm", "n_iterations"), None)])
    with row[2]:
        card("com", [("QoC cuối", fmt(g("com_algorithm", "QoC_final")), "Mục tiêu 1.0"),
                     ("ΔS", fmt(g("com_algorithm", "delta_S")), None)])
    row = st.columns(3)
    with row[0]:
        card("phi", [("Φ toàn hệ", fmt(g("iit_phi", "Phi_whole_system")), None),
                     ("Phân hoạch (n=5)", g("iit_phi", "n_partitions_exact_n5"), "Số phân hoạch vét cạn")])
    with row[1]:
        ci = g("meta_analysis", "ci", default=[np.nan, np.nan])
        card("meta", [("Pooled d", fmt(g("meta_analysis", "pooled_d"), 3), "Bài báo: 1.70"),
                      ("I²", f"{g('meta_analysis', 'I2_percent', default=np.nan):.1f}%", "Bài báo: 31%")],
             f"KTC 95%: [{ci[0]:.2f}, {ci[1]:.2f}] · ΔDMN gộp {fmt(g('meta_analysis', 'pooled_delta_pct'), 3)}%")
    with row[2]:
        card("spectral", [("Fiedler λ₂", fmt(g("spectral_graph", "fiedler_value"), 3), None),
                          ("σ small-world", fmt(g("spectral_graph", "small_world_index"), 3), "σ > 1"),
                          ("ΔE_glob", f"+{fmt(g('spectral_graph', 'pct_increase_E_glob'), 3)}%", None)])
    row = st.columns(3)
    with row[0]:
        card("hopfield", [("n", g("hopfield", "n_units"), None),
                          ("Classic ~0.14n", fmt(g("hopfield", "theoretical_classic_capacity"), 3), None),
                          ("Modern ~e^{n/2}", fmt(g("hopfield", "theoretical_modern_capacity"), 3), None)])
    with row[1]:
        ode = g("dmn_ode", default={})
        if ode and isinstance(ode.get("control_final"), dict):
            cf, mf = ode["control_final"], ode["meditator_final"]
            card("ode", [(k, fmt(mf.get(k), 3), f"Đối chứng: {fmt(cf.get(k), 3)}")
                         for k in ("A_DMN", "Phi", "QoC") if k in mf], "Giá trị thiền giả tại t cuối (help = đối chứng).")
        else:
            card("ode", [("Kết quả", "xem JSON", None)])
    with row[2]:
        card("quantum", [("Ghi chú", "—", None)], g("quantum_decoherence", "note"))

    # --- PCI: mô phỏng vs bài báo
    pci = g("pci")
    if pci:
        st.markdown("##### PCI: mô phỏng vs Table PCI")
        sim = pd.Series(pci["simulated_means"])
        ref = pd.Series(pci["paper_reference_means"]).reindex(sim.index)
        k = 1.0
        a, b = st.columns([3, 2], gap="large")
        with a:
            fig = go.Figure()
            fig.add_bar(y=sim.index, x=sim / k, orientation="h", name="Mô phỏng", marker_color=BLUE)
            fig.add_bar(y=ref.index, x=ref, orientation="h", name="Bài báo", marker_color=ORANGE)
            fig.update_layout(barmode="group")
            show(style(fig, height=380, xlab="PCI"))
        with b:
            from scipy.stats import spearmanr
            rho = spearmanr(sim, ref).statistic
            st.metric("Spearman ρ (thứ tự trạng thái)", f"{rho:.3f}")
            st.metric("Khoảng PCI", f"{sim.min():.2f} – {sim.max():.2f}", help="Chuẩn hoá Casali 2013")
            st.caption("Các trạng thái ý thức cao bão hoà gần 1. Mức phức tạp mỗi trạng thái được đặt tay "
                       "nên đây là minh hoạ thuật toán, không phải bằng chứng.")

    # --- IIT: top mechanisms
    top = g("iit_phi", "top_mechanisms")
    if top:
        st.markdown("##### IIT: cơ chế có φ lớn nhất")
        d = pd.Series(top).sort_values()
        fig = go.Figure(go.Bar(y=d.index, x=d.values, orientation="h", marker_color=BLUE))
        show(style(fig, height=60 + 32 * len(d), xlab="φ (mechanism)", legend=False))

# =============================================================================
# Hình
# =============================================================================
with t_fig:
    if not figures:
        st.info("Chưa có hình — chạy thực nghiệm không bật **Không xuất hình**.")
    else:
        def fig_exp(path):
            try:
                return FIG_EXP[int(os.path.basename(path)[:2]) - 1]
            except (ValueError, IndexError):
                return None

        avail = [e for e in FIG_EXP if any(fig_exp(p) == e for p in figures)]
        pick = st.pills("Lọc", avail, format_func=core.EXPERIMENT_LABELS.get, selection_mode="multi",
                        label_visibility="collapsed")
        shown = [p for p in figures if not pick or fig_exp(p) in pick]
        cols = st.columns(2, gap="large")
        for i, p in enumerate(shown):
            e = fig_exp(p)
            with cols[i % 2], st.container(border=True):
                st.markdown(f"**{core.EXPERIMENT_LABELS.get(e, '')}** :gray[`{os.path.basename(p)}` · "
                            f"{datetime.fromtimestamp(os.path.getmtime(p)).strftime('%Y-%m-%d %H:%M')}]")
                st.image(p, width="stretch")
                if e in EXP_MODULE:
                    st.page_link(core.PAGES["lab"], label="Chỉnh tham số trong Phòng thí nghiệm",
                                 icon=":material/science:", query_params={"m": EXP_MODULE[e]["key"]})

# =============================================================================
# JSON thô
# =============================================================================
with t_raw:
    if summary:
        flat = pd.DataFrame([{"Khoá": k, "Giá trị": fmt(v) if isinstance(v, (int, float)) else str(v)}
                             for k, v in core.flatten(summary).items()])
        st.dataframe(flat, hide_index=True, width="stretch", height=420)
        st.download_button("Tải summary.json", json.dumps(summary, indent=2, ensure_ascii=False).encode("utf-8"),
                           file_name="summary.json", mime="application/json", icon=":material/download:")
        with st.expander("JSON"):
            st.json(summary, expanded=1)
