"""Trang Bài Q1 — pipeline mô hình (Q1_pipeline_model.md): giải thích, trạng thái cổng,
kết quả 5 gói và các khám phá tương tác chạy trực tiếp mã trong model/ và bqi/dmn_ode.py."""
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core
from core import BLUE, ORANGE, AQUA, GRAY, RED, GOOD, CRITICAL, fmt, show, style
from model import config as CFG

MODEL_DIR = os.path.join(core.RES_DIR, "model")
GATE_DIR = os.path.join(core.PROJECT_ROOT, "outputs", "gates")
PIPELINE_MD = os.path.join(core.PROJECT_ROOT, "Q1_pipeline_model.md")

st.title("📐 Bài Q1 — Mô hình động lực của thiền")
st.caption("Sửa sai, khả năng xác định tham số, thiết kế thí nghiệm tối ưu và điều khiển vòng kín. "
           "Chỉ dùng dữ liệu *in silico* sinh từ mô hình, có seed, tái tạo 100%. Chi tiết: `Q1_pipeline_model.md`.")


# =============================================================================
# Đọc kết quả trên đĩa
# =============================================================================
def _json(name):
    try:
        with open(os.path.join(MODEL_DIR, name), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _csv(name, **kw):
    p = os.path.join(MODEL_DIR, name)
    return pd.read_csv(p, **kw) if os.path.isfile(p) else None


def _gate(name):
    try:
        with open(os.path.join(GATE_DIR, f"{name}.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# =============================================================================
# 1. Pipeline Q1 so với BQI ban đầu
# =============================================================================
with st.expander("Pipeline Q1 khác gì so với bài BQI ban đầu?", icon=":material/compare_arrows:", expanded=True):
    st.markdown("""
Bài BQI gốc đề xuất hệ ODE DMN–Φ–QoC (eq. 52–54) và khẳng định nhiều kết quả định lượng. Dự án này ban đầu
chỉ *tái hiện* bài đó bằng dữ liệu tổng hợp đặt tay cho khớp — tức là không chứng minh được gì (vòng lặp).
Pipeline Q1 đổi câu hỏi: thay vì hỏi *"mô hình có khớp số liệu của bài không?"*, hỏi
**"mô hình có nhất quán không, tham số có đo được không, và điều khiển được không?"** — ba câu hỏi
trả lời được hoàn toàn bằng toán và mô phỏng, không cần số liệu của bài gốc.
""")
    cmp = pd.DataFrame([
        ["Mô hình", "M0: eq. 52–54; QoC là ODE của đạo hàm dΦ/dt, dA/dt",
         "M1: A_DMN là quá trình OU quanh mức nền A0; QoC = αΦ − βA + γ0 (đại số) → trạng thái dừng ≠ 0"],
        ["Nghiệm dừng", "Eq. 55–57: A* > 0, QoC* = 27.8",
         "Chứng minh A* = 0, **QoC* ≡ 0** với mọi tham số (QoC là bộ lọc thông cao của dΦ, dA); eq. 55–57 không thoả ODE"],
        ["Dữ liệu", "Table 4, 5, 6, 7, 11, 14 (không truy vết được nguồn)",
         "Thí nghiệm *in silico* có seed; **không trích bất kỳ bảng số nào của bài gốc**"],
        ["Tham số", "\"Fit trên N = 3.241\" (Table 5)",
         "Giá trị danh nghĩa + phân tích Sobol trên dải ×0.25–×4; không tuyên bố gì về não thật"],
        ["Khả năng xác định", "Không xét",
         "Định lý Rothenberg trên Fisher kỳ vọng chính xác; chỉ ra kênh quan sát nào cần có để tách α, β, k_int"],
        ["Thiết kế thí nghiệm", "Không xét",
         "Bản đồ CRB theo (số người, độ dài phiên, khoảng probe, nhiễu) → cần bao nhiêu dữ liệu để sai số ≤ 20%"],
        ["Điều khiển (COM)", "Algorithm 2 với proxy đặt tay; khẳng định bang-bang",
         "Bài toán LQG (Kalman + LQR, điểm cân bằng giải tích); so với bang-bang; mất mát khi tham số chỉ ước lượng"],
        ["Kiểm soát chất lượng", "Không có",
         "Cổng M0 → M1 → M2 cưỡng chế trong mã; bước sau từ chối chạy nếu cổng trước chưa pass"],
    ], columns=["", "BQI ban đầu", "Pipeline Q1"])
    st.dataframe(cmp, hide_index=True, width="stretch",
                 column_config={"": st.column_config.TextColumn(width="small"),
                                "BQI ban đầu": st.column_config.TextColumn(width="medium"),
                                "Pipeline Q1": st.column_config.TextColumn(width="large")})
    st.markdown("""
**Luồng 5 gói kết quả** — mỗi gói là một mục trong bài, mỗi cổng là một điều kiện dừng:

`theory` (P1–P5, cổng **M0**) → `identifiability` (Rothenberg + CRB, Bảng 2) → `design` (bản đồ thiết kế, cổng **M1**:
CRB phải khớp sai số MLE thật) → `control` (LQG vs bang-bang, cổng **M2**; độ bền theo tham số ước lượng) →
`sensitivity` (Sobol). Lệnh tái tạo toàn bộ: `python main.py model all --seed 42`.
""")

with st.expander("Thông số chung của mọi thực nghiệm (model/config.py)", icon=":material/tune:"):
    st.caption("Một nguồn duy nhất cho 5 gói, script xuất hình, dashboard và test. Mỗi lần chạy ghi kèm "
               "`config_snapshot.json` vào thư mục kết quả.")
    R, Cc, G = CFG.REFERENCE, CFG.CONTROL, CFG.GRID
    st.dataframe(pd.DataFrame([
        ["Seed", str(CFG.SEED)], ["Epoch", f"{CFG.EPOCH_S:g} s"],
        ["Tham số danh nghĩa", ", ".join(f"{k} = {v:g}" for k, v in CFG.NOMINAL.items())],
        ["Thiết kế tham chiếu (ds001787-like)", f"N = {R.n_subjects}, {R.session_min:g} phút, probe/{R.probe_every_min:g} phút, "
                                                 f"r = {R.r_obs:g}, cấu hình {R.config}"],
        ["Phân tích cấu trúc", f"{CFG.STRUCTURAL.session_min:g} phút, probe/{CFG.STRUCTURAL.probe_every_min:g} phút, "
                                f"γ0 = {CFG.STRUCTURAL.gamma0_generic:g}, tol {CFG.STRUCTURAL.eig_tol:g}"],
        ["Lưới thiết kế", f"N {G.n_subjects}, phiên {G.session_min} phút, probe {G.probe_every_min} phút, r {G.r_obs}; "
                          f"{len(G.representative_cells)} ô đại diện × {G.n_reps} lần, {G.n_starts} điểm xuất phát"],
        ["Điều khiển", f"mục tiêu {Cc.target_qoc:g}, |u| ≤ {Cc.u_max:g}, b = {Cc.b_control:g}, Q = {Cc.Q_cost:g}, λ = {Cc.R_cost:g}, "
                       f"{Cc.n_epochs} epoch, {Cc.n_trials} trial"],
        ["Sobol", f"×{CFG.SENSITIVITY.factor_range[0]:g}–×{CFG.SENSITIVITY.factor_range[1]:g}, "
                  f"CV(A) ∈ {CFG.SENSITIVITY.cv_A_range}, n_base {CFG.SENSITIVITY.n_base_default} "
                  f"(bài: {CFG.SENSITIVITY.n_base_paper})"],
        ["Ngưỡng", f"sai số ≤ {100 * CFG.TARGET_REL_ERROR:.0f}%; M1: RMSE/CRB ∈ {CFG.GATES.m1_ratio_bounds} ở ≥ "
                   f"{100 * CFG.GATES.m1_min_fraction:.0f}% ô; M2: dạng đóng khớp ±{100 * CFG.GATES.m2_analytic_tol:.0f}%"],
    ], columns=["", "Giá trị"]), hide_index=True, width="stretch",
        column_config={"Giá trị": st.column_config.TextColumn(width="large")})

# =============================================================================
# 2. Trạng thái cổng + chạy
# =============================================================================
st.subheader("Cổng kiểm tra")
GATES = [("M0", "P1–P5: M0 sai, M1 nhất quán", "theory"),
         ("M1", "CRB khớp sai số MLE ở ≥ 80% ô đại diện", "design"),
         ("M2", "LQG thắng bang-bang và không điều khiển; chi phí dừng khớp dạng đóng", "control")]
cols = st.columns(3)
for col, (g, desc, _) in zip(cols, GATES):
    s = _gate(g)
    with col, st.container(border=True):
        if s is None:
            st.markdown(f"**{g}** — chưa chạy")
        else:
            st.markdown(f"**{g}** {'✅ PASS' if s['passed'] else '❌ FAIL'}  \n:gray[{s['time_utc'][:16]}]")
        st.caption(desc)

with st.expander("Chạy một bước của pipeline", icon=":material/play_circle:"):
    a, b, c = st.columns([2, 1, 1])
    step = a.selectbox("Bước", ["theory", "identifiability", "design", "control", "sensitivity", "figures", "all"])
    seed = b.number_input("Seed", value=42, step=1)
    fast = c.toggle("Nhanh (thử)", value=True, help="Ít lần lặp; không dùng cho bài")
    st.caption({"theory": "vài giây", "identifiability": "~1 phút", "design": "nhiều giờ (nhanh: ~1 giờ)",
                "control": "~1 phút (có độ bền nếu đã có design_grid_mle.csv)", "sensitivity": "~20 phút (nhanh: ~5 phút)",
                "figures": "~30 s — xuất PNG/PDF + CSV/LaTeX vào outputs/paper/model (bỏ qua hình thiếu dữ liệu)",
                "all": "nhiều giờ"}[step])
    if st.button("Chạy", type="primary", icon=":material/play_arrow:"):
        args = ["main.py", "model", step, "--seed", str(int(seed))] + (["--fast"] if fast else [])
        st.code("python " + " ".join(args), language="bash")
        with st.status(f"model.{step}…", expanded=True) as status:
            code, _, elapsed = core.run_streaming(args, st.empty())
            status.update(label=f"{'Hoàn tất' if code == 0 else 'Lỗi'} sau {elapsed:.0f}s (exit {code})",
                          state="complete" if code == 0 else "error", expanded=code != 0)
        st.cache_data.clear()
        st.rerun()

# =============================================================================
# 3. Năm gói kết quả
# =============================================================================
t_theory, t_ident, t_design, t_control, t_sens = st.tabs(
    ["1 · Sửa mô hình", "2 · Xác định tham số", "3 · Thiết kế thí nghiệm", "4 · Điều khiển", "5 · Độ nhạy"])


@st.cache_data(show_spinner=False)
def _nominal():
    from bqi.dmn_ode import StateSpaceDMN, ODE_PARAMS_REFERENCE, steady_states, steady_states_paper
    m = StateSpaceDMN()
    return (dict(M1=m.steady_states(), M0=steady_states(ODE_PARAMS_REFERENCE),
                 paper=steady_states_paper(ODE_PARAMS_REFERENCE)),
            {k: getattr(m, k) for k in m.FREE})


# ---- 1. theory ---------------------------------------------------------------
with t_theory:
    st.latex(r"\text{M0: }\;\dot Q=\alpha\dot\Phi-\beta\dot A-\gamma Q\;\;\Rightarrow\;\;"
             r"H(s)=\frac{s}{s+\gamma},\;H(0)=0\;\Rightarrow\;Q^*=0\qquad"
             r"\text{M1: }\;dA=-k_{inh}(A-A_0)\,dt+\sigma_A dW,\;\;Q=\alpha\Phi-\beta A+\gamma_0")
    ss, nominal = _nominal()
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.markdown("**Nghiệm dừng**")
        st.dataframe(pd.DataFrame({
            "": ["A*", "Φ*", "QoC*"],
            "Bài gốc (eq. 55–57)": [ss["paper"]["A_DMN_star"], ss["paper"]["Phi_star"], ss["paper"]["QoC_star"]],
            "M0 đúng": [ss["M0"]["A_DMN_star"], ss["M0"]["Phi_star"], ss["M0"]["QoC_star"]],
            "M1 (đề xuất)": [ss["M1"]["A_DMN_star"], ss["M1"]["Phi_star"], ss["M1"]["QoC_star"]],
        }), hide_index=True, width="stretch",
            column_config={k: st.column_config.NumberColumn(format="%.4f")
                           for k in ["Bài gốc (eq. 55–57)", "M0 đúng", "M1 (đề xuất)"]})
        th = _json("theory_results.json")
        if th:
            st.markdown("**Mệnh đề P1–P5 (cổng M0)**")
            for name, c in th["checks"].items():
                st.markdown(f"{'✅' if c['passed'] else '❌'} `{name}`")
        else:
            st.caption("Chưa chạy `model theory`.")
    with right:
        gamma = st.slider("γ (suy giảm QoC) cho Bode", 0.02, 1.0, 0.19, 0.01)
        f = np.logspace(-3, 1, 300)
        w = 2 * np.pi * f
        H = 1j * w / (1j * w + gamma)
        core.save_exploration("bode", {"gamma": gamma}, scope="model", outputs=
                              {"f_c_per_min": gamma / (2 * np.pi), "dc_gain": 0.0,
                               "gain_db_at_fc": float(20 * np.log10(abs(1j * gamma / (1j * gamma + gamma))))},
                              arrays={"f": f, "gain_db": 20 * np.log10(np.abs(H))})
        fig = go.Figure(go.Scatter(x=f, y=20 * np.log10(np.abs(H)), line=dict(color=BLUE, width=2)))
        fig.add_vline(x=gamma / (2 * np.pi), line=dict(color=ORANGE, dash="dash"),
                      annotation_text=f"f_c = γ/2π = {gamma / (2 * np.pi):.3f}/phút")
        fig.update_xaxes(type="log", exponentformat="power")
        show(style(fig, "P2 — QoC là bộ lọc thông cao của (αΦ − βA): độ lợi DC = −∞ dB", xlab="tần số (1/phút)",
                   ylab="|H| (dB)", height=320, legend=False))
        st.caption("Mọi đầu vào dừng cho E[QoC] = 0: QoC của M0 chỉ phản ánh *biến thiên*, không phải mức. "
                   "Đây là lý do QoC* in trong bài (27.8) không thể đúng.")

# ---- 2. identifiability -------------------------------------------------------
with t_ident:
    st.markdown("**Rothenberg (1971):** θ xác định được cục bộ ⇔ ma trận thông tin Fisher không suy biến. "
                "Fisher kỳ vọng **chính xác** (toàn bộ quan sát một phiên là vector Gauss dạng đóng), phiên 90 phút, probe/1 phút.")
    ident = _json("identifiability_results.json")
    tab2 = _csv("table2_identifiability.csv", index_col=0)
    if ident is None:
        st.caption("Chưa chạy `model identifiability`.")
    else:
        rows = []
        for cfg, s in ident["structural"].items():
            rows.append({"Cấu hình": cfg, "Quan sát": {"a": "chỉ rating", "b": "rating + y_A", "c": "rating + y_Φ",
                                                       "d": "rating + y_A + y_Φ"}[cfg],
                         "Hạng": f"{s['rank']}/{s['n_identifiable_candidates']}",
                         "Vắng mặt": ", ".join(s["absent"]) or "—",
                         "Phải cố định": ", ".join(s.get("normalisation", [])) or "—",
                         "Hướng không xác định": "; ".join(
                             " ".join(f"{k}:{v:+.2f}" for k, v in nd["direction"].items())
                             for nd in s["null_directions"]) or "—"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"Hướng không xác định": st.column_config.TextColumn(width="large")})
        st.caption("Config b: Φ → cΦ với k_int, σ_Φ → c·, α → α/c giữ nguyên (y_A, rating) — đối xứng co giãn của "
                   "trạng thái ẩn không quan sát; đã kiểm chứng giải tích. Chỉ cấu hình d ước lượng được đủ 12 tham số.")
        if tab2 is not None:
            st.markdown("**Bảng 2 — CRB tương đối theo cấu hình** (N = 12, 45 phút, probe mỗi 2 phút)")
            st.dataframe(tab2, width="stretch")
    with st.expander("CRB cho một thiết kế tuỳ chọn (tính trực tiếp, ~2 s)"):
        c1, c2, c3, c4, c5 = st.columns(5)
        R = CFG.REFERENCE
        n_sub = c1.number_input("Số người", 1, 200, R.n_subjects)
        sess = c2.number_input("Phiên (phút)", 5.0, 240.0, float(R.session_min), 5.0)
        probe = c3.number_input("Probe mỗi (phút)", CFG.DT, 30.0, float(R.probe_every_min), 0.5)
        r_obs = c4.number_input("Nhiễu EEG r", 0.01, 1.0, float(R.r_obs), 0.01)
        cfg = c5.selectbox("Cấu hình", ["d", "b", "c", "a"], index=["d", "b", "c", "a"].index(R.config))

        @st.cache_data(show_spinner="Đang tính Fisher chính xác…")
        def _crb_custom(n_sub, sess, probe, r_obs, cfg):
            from bqi.dmn_ode import StateSpaceDMN
            from model.fisher import Design, fisher_exact, crb
            from model.identifiability import structural_identifiability
            m = StateSpaceDMN(r_A=r_obs, r_Phi=r_obs)
            names = structural_identifiability(m)[cfg]
            I = fisher_exact(m, Design.regular(sess, probe, config=cfg), names, n_subjects=int(n_sub))
            b = crb(I, names, m)
            core.save_exploration("crb_custom",
                                  {"n_subjects": n_sub, "session_min": sess, "probe_every_min": probe,
                                   "r_obs": r_obs, "config": cfg},
                                  {"estimable": names, "rel_crb": {k: b[k]["rel_se"] for k in names},
                                   "fisher_diag": np.diag(I)}, scope="model")
            return pd.DataFrame({"Tham số": names, "CRB tương đối": [b[k]["rel_se"] for k in names]})

        df = _crb_custom(int(n_sub), float(sess), float(probe), float(r_obs), cfg)
        colors = [GOOD if v <= 0.2 else CRITICAL for v in df["CRB tương đối"]]
        fig = go.Figure(go.Bar(x=df["Tham số"], y=df["CRB tương đối"], marker_color=colors))
        fig.add_hline(y=0.2, line=dict(color=GRAY, dash="dash"), annotation_text="ngưỡng 20%")
        show(style(fig, "Sai số tương đối tối thiểu (Cramér–Rao)", ylab="CRB", height=300, legend=False))

# ---- 3. design ---------------------------------------------------------------
with t_design:
    crb_df = _csv("design_grid_crb.csv")
    mle_df = _csv("design_grid_mle.csv")
    if crb_df is None:
        st.caption("Chưa chạy `model design`.")
    else:
        key = {"k_inh": "k_inh", "A0": "A0", "alpha": "α", "beta": "β"}
        a, b, c = st.columns(3)
        param = a.selectbox("Tham số", list(key), format_func=key.get)
        probe_sel = b.selectbox("Probe mỗi (phút)", sorted(crb_df["probe_every_min"].unique()), index=2)
        r_sel = c.selectbox("Nhiễu EEG r", sorted(crb_df["r_obs"].unique()), index=1)
        sub = crb_df[(crb_df["probe_every_min"] == probe_sel) & (crb_df["r_obs"] == r_sel)]
        pv = sub.pivot(index="N", columns="session_min", values=f"crb_{param}")
        fig = go.Figure(go.Heatmap(z=pv.values, x=[f"{c:.0f} phút" for c in pv.columns],
                                   y=[f"N = {i}" for i in pv.index], colorscale=core.SEQ_BLUE, reversescale=True,
                                   zmin=0, zmax=max(0.4, float(np.nanmax(pv.values))),
                                   text=np.round(pv.values, 3), texttemplate="%{text}", colorbar=dict(title="CRB")))
        show(style(fig, f"Bản đồ thiết kế — CRB tương đối của {key[param]} (xanh đậm = tốt)", height=340, legend=False))
        ok = crb_df[crb_df["crb_k_inh"] <= 0.2].copy()
        ok["chi phí = N × phút"] = ok["N"] * ok["session_min"]
        st.markdown("**Thiết kế rẻ nhất đạt CRB(k_inh) ≤ 20 %**")
        st.dataframe(ok.sort_values("chi phí = N × phút").head(8)[
            ["N", "session_min", "probe_every_min", "r_obs", "crb_k_inh", "chi phí = N × phút"]].round(3),
            hide_index=True, width="stretch")
        if mle_df is not None:
            st.markdown("**Cổng M1 — CRB có dự đoán được sai số MLE thật không?** (24 ô đại diện)")
            m = mle_df.merge(crb_df, on=["N", "session_min", "probe_every_min", "r_obs"])
            fig = go.Figure()
            for i, k in enumerate(key):
                fig.add_scatter(x=m[f"crb_{k}"], y=m[f"mle_{k}"], mode="markers", name=key[k],
                                marker=dict(color=core.SERIES[i], size=9),
                                customdata=m[["N", "session_min", "probe_every_min", "r_obs"]].values,
                                hovertemplate="N=%{customdata[0]} %{customdata[1]}min probe/%{customdata[2]} "
                                              "r=%{customdata[3]}<br>CRB %{x:.3f} · MLE %{y:.3f}<extra></extra>")
            lim = float(max(m[[f"crb_{k}" for k in key]].max().max(), m[[f"mle_{k}" for k in key]].max().max()))
            fig.add_scatter(x=[0, lim], y=[0, lim], mode="lines", line=dict(color=GRAY, dash="dash"), name="y = x")
            fig.add_scatter(x=[0, lim], y=[0, 2 * lim], mode="lines", line=dict(color=GRAY, dash="dot"), name="×2")
            show(style(fig, "RMSE của MLE so với CRB (cùng thang log)", xlab="CRB", ylab="RMSE MLE", height=360))
            g1 = _gate("M1")
            if g1:
                st.caption(f"M1 {'PASS' if g1['passed'] else 'FAIL'}: {g1['details'].get('fraction_matching', 0):.0%} ô "
                           f"khớp trong hệ số 2 (trung vị RMSE/CRB = "
                           f"{g1['details'].get('ratio_rmse_over_crb', {}).get('median', float('nan')):.2f}).")
        else:
            st.caption("Phần MLE (cổng M1) chưa chạy xong — chỉ có lưới CRB.")

# ---- 4. control --------------------------------------------------------------
with t_control:
    st.latex(r"J=\mathbb{E}\sum_t\big[(Q_t-Q_{\text{target}})^2+\lambda u_t^2\big],\qquad "
             r"dA=-k_{inh}(A-A_0)\,dt-b\,u\,dt+\sigma_A dW,\qquad u_t=u_{ss}-K(\hat x_t-x_{ss})")
    left, right = st.columns([1, 3], gap="large")
    with left, st.form("ctrl"):
        Cc = CFG.CONTROL
        target = st.slider("QoC mục tiêu", 0.0, 4.0, float(Cc.target_qoc), 0.1)
        u_max = st.slider("|u| tối đa", 0.1, 5.0, float(Cc.u_max), 0.1)
        lam = st.select_slider("λ (giá của điều khiển)", [0.01, 0.03, 0.1, 0.3, 1.0], float(Cc.R_cost))
        n_ep = st.slider(f"Số epoch ({CFG.EPOCH_S:g} s)", 60, 600, int(Cc.n_epochs), 30)
        seed_c = st.number_input("Seed", value=0, step=1)
        st.form_submit_button("Mô phỏng", type="primary", icon=":material/play_arrow:", width="stretch")

    @st.cache_data(show_spinner=False)
    def _policies(target, u_max, lam, n_ep, seed):
        from bqi.dmn_ode import StateSpaceDMN
        from model.control import state_space_with_control, solve_lqg, simulate_policy, lqg_stationary_cost
        m = StateSpaceDMN()
        st_ = state_space_with_control(m)
        ctrl = solve_lqg(st_, R_cost=lam, target_qoc=target)
        out = {p: simulate_policy(m, ctrl, p, n_ep, CFG.CONTROL.b_control, u_max, seed, st_)
               for p in ("lqg", "bangbang", "none")}
        analytic = lqg_stationary_cost(m, ctrl, sys_true=st_)
        core.save_exploration("control",
                              {"target_qoc": target, "u_max": u_max, "lambda": lam, "n_epochs": n_ep, "seed": seed},
                              {"cost": {p: out[p]["cost"] for p in out},
                               "tracking_mse": {p: out[p]["tracking_mse"] for p in out},
                               "u_rms": {p: out[p]["u_rms"] for p in out},
                               "lqg_stationary_analytic": analytic["cost"], "u_ss": ctrl["u_ss"],
                               "x_ss": ctrl["x_ss"], "K": ctrl["K"]},
                              {"t": out["lqg"]["t"], **{f"QoC_{p}": out[p]["QoC"] for p in out},
                               **{f"u_{p}": out[p]["u"] for p in out}}, scope="model")
        return out, analytic, ctrl["u_ss"]

    runs, analytic, u_ss = _policies(float(target), float(u_max), float(lam), int(n_ep), int(seed_c))
    with right:
        k = st.columns(4)
        for col, (p, label) in zip(k, [("lqg", "LQG"), ("bangbang", "Bang-bang"), ("none", "Không điều khiển")]):
            col.metric(f"Chi phí {label}", f"{runs[p]['cost']:.4f}")
        k[3].metric("LQG dừng (dạng đóng)", f"{analytic['cost']:.4f}", help=f"u_ss = {u_ss:.3f}")
        fig = go.Figure()
        for p, label, colr in [("none", "Không điều khiển", GRAY), ("bangbang", "Bang-bang", ORANGE), ("lqg", "LQG", BLUE)]:
            fig.add_scatter(x=runs[p]["t"], y=runs[p]["QoC"], name=label, line=dict(color=colr, width=2))
        fig.add_hline(y=target, line=dict(color=RED, dash="dash"), annotation_text="mục tiêu")
        show(style(fig, "QoC(t) theo chính sách (cùng nhiễu)", xlab="t (phút)", ylab="QoC", height=320))
        fig = go.Figure()
        for p, label, colr in [("bangbang", "Bang-bang", ORANGE), ("lqg", "LQG", BLUE)]:
            fig.add_scatter(x=runs[p]["t"], y=runs[p]["u"], name=label, line=dict(color=colr, width=1.5))
        show(style(fig, "Tín hiệu điều khiển u(t)", xlab="t (phút)", ylab="u", height=240))
    cr = _json("control_results.json")
    if cr and cr.get("robustness"):
        st.markdown("**Độ bền — mất mát hiệu quả khi bộ điều khiển dùng tham số ước lượng từ từng thiết kế thí nghiệm**")
        rb = pd.DataFrame(cr["robustness"])
        fig = go.Figure(go.Scatter(x=rb["avg_mle_rmse"], y=rb["efficiency_loss_pct_median"], mode="markers",
                                   marker=dict(color=AQUA, size=10),
                                   customdata=rb[["N", "session_min", "probe_every_min"]].values,
                                   hovertemplate="N=%{customdata[0]} %{customdata[1]}min probe/%{customdata[2]}"
                                                 "<br>RMSE %{x:.3f} → mất %{y:.1f}%<extra></extra>"))
        show(style(fig, "Mất mát hiệu quả (%) theo chất lượng ước lượng", xlab="RMSE trung bình (k_inh, A0, α, β)",
                   ylab="mất mát (%)", height=300, legend=False))

# ---- 5. sensitivity -----------------------------------------------------------
with t_sens:
    sob = _csv("sobol_indices.csv")
    meta = _json("sensitivity_results.json")
    if sob is None:
        st.caption("Chưa chạy `model sensitivity`.")
    else:
        labels = {"qoc_stationary": "QoC*", "qoc_variance": "Var dừng QoC", "crb_k_inh": "CRB(k_inh)",
                  "lqg_efficiency": "Hiệu quả LQG"}
        idx = st.radio("Chỉ số", ["ST", "S1"], horizontal=True, help="ST = tổng (kể cả tương tác), S1 = bậc 1")
        pv = sob.pivot(index="param", columns="output", values=idx).rename(columns=labels)
        fig = go.Figure(go.Heatmap(z=pv.values, x=list(pv.columns), y=list(pv.index), colorscale=core.SEQ_BLUE,
                                   zmin=0, zmax=1, text=np.round(pv.values, 2), texttemplate="%{text}"))
        show(style(fig, f"Chỉ số Sobol {idx} (tham số quét ×0.25–×4, log-uniform)", height=360, legend=False))
        if meta:
            st.caption(f"n_base = {meta.get('n_base')} → {meta.get('n_base', 0) * 10} lần đánh giá. "
                       + ("Bản nhanh chỉ để kiểm tra pipeline; bài cần n_base ≥ 256." if meta.get("n_base", 0) < 128 else ""))

# =============================================================================
# 4. Hình và bảng cho bài (experiments/run_model_paper.py)
# =============================================================================
PAPER_DIR = os.path.join(core.PROJECT_ROOT, "outputs", "paper", "model")
with st.expander("Hình và bảng cho bài (xuất bản)", icon=":material/image:",
                 expanded=os.path.isfile(os.path.join(PAPER_DIR, "manifest.json"))):
    man = None
    try:
        with open(os.path.join(PAPER_DIR, "manifest.json"), encoding="utf-8") as f:
            man = json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    if man is None:
        st.caption("Chưa xuất. Chọn bước **figures** ở trên hoặc chạy `python main.py model figures`.")
    else:
        st.caption(f"`{os.path.relpath(PAPER_DIR, core.PROJECT_ROOT)}/` — {len(man['made'])} tệp "
                   f"(mỗi hình có PNG 300 dpi + PDF, mỗi bảng có CSV + LaTeX). Seed {man['seed']}.")
        if man["skipped"]:
            st.warning("Bỏ qua vì thiếu dữ liệu: " + "; ".join(f"**{n}** — {why}" for n, why in man["skipped"]),
                       icon=":material/info:")
        pngs = sorted(f for f in os.listdir(PAPER_DIR) if f.endswith(".png"))
        cols = st.columns(2, gap="medium")
        for i, f in enumerate(pngs):
            with cols[i % 2], st.container(border=True):
                st.markdown(f"**{f[:-4]}**")
                st.image(os.path.join(PAPER_DIR, f), width="stretch")
        texs = sorted(f for f in os.listdir(PAPER_DIR) if f.endswith(".tex"))
        if texs:
            pick = st.selectbox("Bảng LaTeX", texs)
            with open(os.path.join(PAPER_DIR, pick), encoding="utf-8") as f:
                st.code(f.read(), language="latex")

# =============================================================================
# 5. Kết quả khám phá đã lưu
# =============================================================================
st.subheader("Kết quả đã lưu")
ex = core.list_explorations("model")
st.caption(f"Mọi phép tính tương tác trên trang này được tự động ghi (một lần cho mỗi bộ tham số) vào "
           f"`{os.path.relpath(core.EXPLORE_SCOPES['model'], core.PROJECT_ROOT)}/<loại>/<khoá>.json` (+ `.npz` cho mảng). "
           f"Kết quả các bước pipeline nằm trong `{os.path.relpath(MODEL_DIR, core.PROJECT_ROOT)}/`, "
           f"cổng trong `outputs/gates/`.")
if ex.empty:
    st.caption("Chưa có bản ghi nào.")
else:
    st.dataframe(ex.rename(columns={"time": "Thời điểm", "kind": "Loại", "key": "Khoá", "inputs": "Tham số",
                                    "file": "File"}),
                 hide_index=True, width="stretch", height=min(300, 40 + 35 * len(ex)),
                 column_config={"Tham số": st.column_config.TextColumn(width="large")})
    by_key = ex.set_index("key")
    a, b = st.columns([1, 3])
    pick = a.selectbox("Xem bản ghi", ex["key"].tolist(), format_func=lambda k: f"{k} · {by_key.loc[k, 'kind']}")
    with open(os.path.join(core.PROJECT_ROOT, by_key.loc[pick, "file"]), encoding="utf-8") as f:
        b.json(json.load(f), expanded=1)
    st.download_button("Tải index.csv", ex.to_csv(index=False).encode("utf-8"), file_name="explorations_index.csv",
                       mime="text/csv", icon=":material/download:")

st.divider()
st.page_link(core.PAGES["lab"], label="Khám phá ODE gốc và mô hình M1 trong Phòng thí nghiệm (§7.2)",
             icon=":material/science:", query_params={"m": "dmn_ode"})
