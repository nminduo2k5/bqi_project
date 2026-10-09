"""Phòng thí nghiệm — chỉnh tham số và chạy trực tiếp từng thuật toán trong bqi/."""
import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import core
from core import BLUE, ORANGE, AQUA, VIOLET, RED, GRAY, fmt, show, style

st.title("🔬 Phòng thí nghiệm thuật toán")
st.caption("Mỗi thay đổi tham số chạy lại đúng mã trong `bqi/` — kết quả được cache theo bộ tham số.")

LABS = {m["key"]: f"§{m['sec']} · {m['name']}" for m in core.MODULES}
default = st.query_params.get("m", "predictive_coding")
choice = st.pills("Module", list(LABS), format_func=LABS.get,
                  default=default if default in LABS else "predictive_coding", label_visibility="collapsed")
choice = choice or "predictive_coding"
st.query_params["m"] = choice
mod = core.MODULE_BY_KEY[choice]
info = core.codebase()


def header(latex: str | None = None):
    meta = info["modules"][mod["key"]]
    n_tests = len(info["tests"].get(core.test_file_for(mod["key"]), []))
    st.subheader(f"§{mod['sec']} — {mod['name']}")
    st.markdown(f"{mod['desc']}  \n"
                f":gray[`bqi/{mod['key']}.py` · {meta['loc']} dòng · {n_tests} test"
                + (f" · dùng `{', '.join(meta['deps'])}`" if meta["deps"] else "") + "]")
    if latex:
        st.latex(latex)


def panel():
    """Trả về (cột tham số, cột kết quả)."""
    return st.columns([1, 3], gap="large")


def run_button():
    return st.form_submit_button("Chạy", type="primary", icon=":material/play_arrow:", width="stretch")


def rng(seed):
    return np.random.default_rng(seed)


# =============================================================================
# §3.1 Bayesian Predictive Coding
# =============================================================================
@st.cache_data(show_spinner=False)
@core.saving("bpc_hier", scope="bqi")
def bpc_hier(levels, n_steps, eta, seed):
    from bqi.predictive_coding import HierarchicalBPC
    dims = {1: [6, 4], 2: [6, 5, 3], 3: [6, 6, 4, 3], 4: [6, 6, 5, 4, 3]}[levels]
    bpc = HierarchicalBPC(n_levels=levels, dims=dims, seed=seed)
    return bpc.run(rng(seed + 1).normal(0, 1, dims[0]), n_steps=n_steps, eta=eta)


@st.cache_data(show_spinner=False)
@core.saving("bpc_single", scope="bqi")
def bpc_single(n, eta, n_iter, pi_max, seed):
    from bqi.predictive_coding import (free_energy, gradient_descent_free_energy,
                                       theoretical_convergence_bound)
    r = rng(seed)
    Pi_s = np.diag(r.uniform(1, pi_max, n))
    Pi_o = np.diag(r.uniform(1, pi_max, n))
    g, prior, o = np.eye(n), np.zeros(n), r.normal(0, 1, n)
    mu0 = r.normal(0, 1, n)
    with np.errstate(over="ignore", invalid="ignore"):
        _, F = gradient_descent_free_energy(mu0, o, prior, Pi_s, Pi_o, g, eta=eta, n_iter=n_iter)
    H = Pi_s + g.T @ Pi_o @ g
    mu_star = np.linalg.solve(H, Pi_s @ prior + g.T @ Pi_o @ o)
    F_star = free_energy(mu_star, o, prior, Pi_s, Pi_o, g)
    lam = np.linalg.eigvalsh(H)
    bound = theoretical_convergence_bound(F[0], F_star, lam.min(), eta * np.arange(len(F)))
    gap = np.where(np.isfinite(F), np.maximum(F - F_star, 1e-300), np.nan)
    return dict(F=F, F_star=F_star, gap=gap, bound=np.maximum(bound, 1e-300),
                lam_min=lam.min(), lam_max=lam.max())


def lab_predictive_coding():
    header(r"F(\mu)=\tfrac12(\mu-\bar\mu)^\top\Pi_s(\mu-\bar\mu)+\tfrac12(o-g\mu)^\top\Pi_o(o-g\mu),\qquad "
           r"F(\mu(t))-F^{*}\le(F(\mu_0)-F^{*})\,e^{-2\lambda_{\min}t}")
    left, right = panel()
    with left, st.form("bpc"):
        st.markdown("**Định lý hội tụ (1 tầng)**")
        n = st.slider("Số chiều μ", 2, 20, 5)
        eta = st.slider("Bước học η", 0.005, 0.6, 0.05, 0.005)
        pi_max = st.slider("Π tối đa (độ lệch điều kiện)", 1.5, 20.0, 2.0, 0.5)
        n_iter = st.slider("Số vòng lặp", 20, 1000, 200, 10)
        st.markdown("**BPC phân tầng**")
        levels = st.select_slider("Số tầng L", [1, 2, 3, 4], 3)
        n_steps = st.slider("Bước cập nhật", 20, 1000, 200, 10)
        eta_h = st.slider("η phân tầng", 0.005, 0.5, 0.05, 0.005)
        seed = st.number_input("Seed", value=42, step=1)
        run_button()
    s = bpc_single(n, eta, n_iter, pi_max, int(seed))
    E = bpc_hier(levels, n_steps, eta_h, int(seed))
    with right:
        safe = eta <= 2 / (s["lam_min"] + s["lam_max"])
        c = st.columns(4)
        c[0].metric("λ_min(H)", f"{s['lam_min']:.3f}")
        c[1].metric("λ_max(H)", f"{s['lam_max']:.3f}")
        c[2].metric("η·λ_max", f"{eta * s['lam_max']:.3f}", help="Gradient descent ổn định khi η·λ_max < 2")
        c[3].metric("F*", fmt(s["F_star"]))
        if not safe:
            st.warning(f"η = {eta} > 2/(λ_min+λ_max) = {2 / (s['lam_min'] + s['lam_max']):.3f}: "
                       "cận lý thuyết không còn đảm bảo cho bản rời rạc"
                       + (" và gradient descent phân kỳ." if eta * s["lam_max"] >= 2 else "."))
        k = np.arange(len(s["F"]))
        fig = go.Figure()
        fig.add_scatter(x=k, y=s["gap"], name="F(μₖ) − F* (thực nghiệm)", line=dict(color=BLUE, width=2))
        fig.add_scatter(x=k, y=s["bound"], name="Cận (F₀−F*)·exp(−2λ_min·ηk)",
                        line=dict(color=ORANGE, width=2, dash="dash"))
        fig.update_yaxes(type="log", exponentformat="power")
        show(style(fig, "Hội tụ mũ của gradient flow (eq. 15)", xlab="Vòng lặp k", ylab="F − F* (log)"))

        fig = go.Figure(go.Scatter(x=np.arange(1, len(E) + 1), y=E, line=dict(color=BLUE, width=2),
                                   name="Năng lượng"))
        fig.update_yaxes(type="log", exponentformat="power")
        show(style(fig, f"BPC phân tầng (L = {levels}): tổng năng lượng sai số Σ‖ε⁽ˡ⁾‖²", xlab="Bước",
                   ylab="Năng lượng (log)", legend=False))
        st.info("**Lưu ý triển khai:** `HierarchicalBPC.step` không dùng quan sát o (μ⁽⁰⁾) — số hạng "
                "ε⁽⁰⁾ = o − f(μ⁽¹⁾) bị bỏ qua, nên mọi μ co về 0 và đường cong trên giống hệt nhau với mọi o.",
                icon=":material/info:")


# =============================================================================
# §3.3 SNIS Attention
# =============================================================================
@st.cache_data(show_spinner=False)
@core.saving("attention_run", scope="bqi")
def attention_run(n_q, n_s, d_model, n_heads, temp, seed):
    from bqi.attention import MultiHeadBQIAttention
    mha = MultiHeadBQIAttention(d_model=d_model, n_heads=n_heads, d_snis=d_model, seed=seed)
    phi_e = rng(seed + 1).normal(0, temp, (n_q, d_model))
    S = rng(seed + 2).normal(0, 1, (n_s, d_model))
    out, weights = mha.forward(phi_e, S)
    return out, list(weights)


def lab_attention():
    header(r"\Sigma(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V,\qquad "
           r"\mathrm{MH}=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_H)W^O")
    left, right = panel()
    with left, st.form("att"):
        n_q = st.slider("Số query (n_q)", 1, 16, 6)
        n_s = st.slider("Kích thước SNIS (n_s)", 4, 64, 24)
        d_model = st.select_slider("d_model", [8, 16, 32, 64], 16)
        n_heads = st.select_slider("Số head H", [1, 2, 4, 8], 4)
        temp = st.slider("Độ lớn query (≈ 1/nhiệt độ)", 0.1, 20.0, 4.0, 0.1,
                         help="Query lớn → softmax sắc → attention tập trung vào ít mục SNIS")
        seed = st.number_input("Seed", value=0, step=1)
        run_button()
    out, weights = attention_run(n_q, n_s, d_model, n_heads, temp, int(seed))
    with right:
        ent = [float(np.mean(-(w * np.log(w + 1e-12)).sum(-1))) for w in weights]
        c = st.columns(3)
        c[0].metric("Entropy attention TB", f"{np.mean(ent):.3f}", help=f"Tối đa ln(n_s) = {np.log(n_s):.3f}")
        c[1].metric("Trọng số max TB", f"{np.mean([w.max(-1).mean() for w in weights]):.3f}")
        c[2].metric("Output", f"{out.shape[0]} × {out.shape[1]}")
        cols = min(n_heads, 4)
        n_rows = int(np.ceil(n_heads / cols))
        fig = make_subplots(rows=n_rows, cols=cols,
                            subplot_titles=[f"Head {h + 1} · H = {ent[h]:.2f}" for h in range(n_heads)],
                            horizontal_spacing=0.04, vertical_spacing=0.12)
        for h, w in enumerate(weights):
            fig.add_trace(go.Heatmap(z=w, coloraxis="coloraxis",
                                     hovertemplate="query %{y} → SNIS %{x}<br>w = %{z:.3f}<extra></extra>"),
                          row=h // cols + 1, col=h % cols + 1)
        fig.update_layout(coloraxis=dict(colorscale=core.SEQ_BLUE, cmin=0, colorbar=dict(title="w")))
        fig.update_yaxes(autorange="reversed")
        show(style(fig, "Trọng số softmax mỗi head (hàng = query, cột = mục SNIS)",
                   height=260 * n_rows + 40, legend=False))
        st.caption("Mỗi hàng cộng lại bằng 1. Mỗi head biểu diễn một luồng xử lý vỏ não riêng "
                   "(thị giác, thính giác, ngữ nghĩa…).")


# =============================================================================
# §4.5 Algorithm 1 — BQI
# =============================================================================
@st.cache_data(show_spinner=False)
@core.saving("bqi_run", scope="bqi")
def bqi_run(levels, T, eta, alpha, n_snis, obs_scale, seed):
    from bqi.bqi_algorithm import BQIModel, BQIConfig
    dims = {2: (8, 6, 4), 3: (8, 8, 6, 4), 4: (8, 8, 6, 5, 4)}[levels]
    cfg = BQIConfig(n_levels=levels, dims=dims, n_snis=n_snis, d_snis=4, d_k=4, eta=eta, alpha=alpha, T=T,
                    seed=seed)
    model = BQIModel(cfg)
    o = rng(seed + 2).normal(0, obs_scale, dims[0])

    def level_energy():
        mu_full = [o] + model.mu + [np.zeros(1)]
        out = []
        for l in range(1, levels + 1):
            pred = model.f[l - 1] @ mu_full[l + 1] if l < levels else np.zeros(dims[l])
            r = mu_full[l] - pred
            out.append(float(0.5 * r @ model.Pi[l - 1] @ r))
        return out

    eps = model.phase1_prediction_errors(o)
    e_before = level_energy()
    F = model.phase2_minimise_free_energy(o)
    e_after = level_energy()
    mu_top = model.mu[-1].copy()
    mu_star, _, w = model.phase3_snis_query_decode()
    return dict(F=F, eps=[float(np.linalg.norm(e)) for e in eps], e_before=e_before, e_after=e_after,
                mu_top=mu_top, mu_star=mu_star, w=w.flatten())


def lab_bqi_algorithm():
    header(r"\textbf{P1: }\varepsilon^{(l)}=\Pi^{(l)}(o^{(l)}-\hat o^{(l)})\;\to\;"
           r"\textbf{P2: }\mu\leftarrow\mu-\eta\nabla_\mu F\;\to\;"
           r"\textbf{P3: }\mu^{*}=\mu+\alpha\,\Sigma(W_Q\mu,\,M,\,M)")
    left, right = panel()
    with left, st.form("bqi"):
        levels = st.select_slider("Số tầng L", [2, 3, 4], 3)
        T = st.slider("T (vòng Phase 2 tối đa)", 10, 400, 100, 10)
        eta = st.slider("η", 0.005, 0.3, 0.05, 0.005)
        alpha = st.slider("α (độ lợi cập nhật SNIS)", 0.0, 2.0, 0.3, 0.05)
        n_snis = st.slider("|SNIS| (số hàng M)", 8, 256, 64, 8)
        obs_scale = st.slider("Độ lớn quan sát ‖o‖", 0.0, 50.0, 1.0, 0.5,
                              help="Thử thay đổi: kết quả không đổi vì F không dùng o (xem lưu ý).")
        seed = st.number_input("Seed", value=42, step=1)
        run_button()
    t0 = time.time()
    r = bqi_run(levels, T, eta, alpha, n_snis, obs_scale, int(seed))
    with right:
        w = r["w"]
        c = st.columns(4)
        c[0].metric("F đầu → cuối", f"{fmt(r['F'][0])} → {fmt(r['F'][-1])}")
        c[1].metric("Vòng lặp Phase 2", len(r["F"]) - 1)
        c[2].metric("Entropy attention", f"{-(w * np.log(w + 1e-12)).sum():.3f}",
                    help=f"Tối đa ln|SNIS| = {np.log(len(w)):.3f}")
        c[3].metric("Thời gian", f"{time.time() - t0:.2f}s")
        a, b = st.columns(2)
        with a:
            fig = go.Figure(go.Scatter(y=r["F"], line=dict(color=BLUE, width=2), name="F"))
            show(style(fig, "Phase 2 — cực tiểu năng lượng tự do", xlab="Vòng lặp", ylab="F", legend=False))
        with b:
            lv = [f"Tầng {l + 1}" for l in range(levels)]
            fig = go.Figure()
            fig.add_bar(x=lv, y=r["e_before"], name="Trước Phase 2", marker_color=GRAY)
            fig.add_bar(x=lv, y=r["e_after"], name="Sau Phase 2", marker_color=BLUE)
            fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
            show(style(fig, "Năng lượng ½εᵀΠε theo tầng", ylab="Năng lượng"))
        a, b = st.columns(2)
        with a:
            top = set(np.argsort(w)[::-1][:5].tolist())
            colors = [BLUE if i in top else "rgba(42,120,214,0.35)" for i in range(len(w))]
            fig = go.Figure(go.Bar(x=np.arange(len(w)), y=w, marker_color=colors,
                                   hovertemplate="SNIS #%{x}<br>w = %{y:.4f}<extra></extra>"))
            fig.add_hline(y=1 / len(w), line=dict(color=GRAY, dash="dot"), annotation_text="đều 1/n",
                          annotation_position="top left")
            show(style(fig, "Phase 3 — trọng số truy vấn SNIS (top-5 đậm)", xlab="Mục SNIS", ylab="w",
                       legend=False))
        with b:
            dims = [f"μ[{i}]" for i in range(len(r["mu_top"]))]
            fig = go.Figure()
            fig.add_bar(x=dims, y=r["mu_top"], name="μ tầng đỉnh (sau P2)", marker_color=GRAY)
            fig.add_bar(x=dims, y=r["mu_star"], name="μ* (sau P3)", marker_color=ORANGE)
            fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
            show(style(fig, "Niềm tin tầng đỉnh trước / sau truy vấn SNIS", ylab="Giá trị"))
        st.warning("**Phát hiện:** Phase 1 trả về ε ≡ 0 "
                   f"(‖ε‖ = {', '.join(fmt(e) for e in r['eps'])}) vì g = I và o⁽ˡ⁾ = μ⁽ˡ⁾; và "
                   "`free_energy()` không dùng quan sát o — kéo thanh ‖o‖ sẽ không làm kết quả thay đổi.",
                   icon=":material/bug_report:")


# =============================================================================
# §8.2 Algorithm 2 — COM
# =============================================================================
@st.cache_data(show_spinner=False)
@core.saving("com_run", scope="bqi")
def com_run(T, d_mu, n_snis, K_gain, eta_dmn, eta_F, eta_S, alpha, beta, target, seed):
    from bqi.com_algorithm import COMConfig, run_com_algorithm
    cfg = COMConfig(T=T, d_mu=d_mu, d_snis=d_mu, n_snis=n_snis, d_k=d_mu, QoC_target=target, K_gain=K_gain,
                    eta_dmn=eta_dmn, eta_F=eta_F, eta_S=eta_S, alpha=alpha, beta=beta, seed=seed)
    return run_com_algorithm(cfg)


def lab_com_algorithm():
    header(r"\textbf{P1: }A\leftarrow A\,(1-\eta\,\nabla F\,(1+u)),\;u=K(\mathrm{QoC}^{*}-\mathrm{QoC})"
           r"\;\to\;\textbf{P2: }\dot\Phi=k_{int}(1-A/A_{max})-k_d\Phi\;\to\;"
           r"\textbf{P3: }\mathrm{QoC}=\alpha\Phi-\beta A+\gamma_0")
    left, right = panel()
    with left, st.form("com"):
        T = st.slider("T (bước, chia 3 pha)", 30, 600, 120, 3)
        K_gain = st.slider("K (độ lợi BCI vòng kín)", 0.0, 3.0, 0.4, 0.05)
        eta_dmn = st.slider("η_DMN", 0.0, 0.3, 0.05, 0.01)
        alpha = st.slider("α (Φ → QoC)", 0.0, 5.0, 2.34, 0.01)
        beta = st.slider("β (DMN → QoC)", 0.0, 5.0, 1.87, 0.01)
        target = st.slider("QoC mục tiêu", 0.0, 3.0, 1.0, 0.05)
        with st.expander("Nâng cao"):
            d_mu = st.select_slider("d_μ", [2, 4, 6, 8, 12, 16], 6)
            n_snis = st.slider("|SNIS|", 5, 200, 30, 5)
            eta_F = st.slider("η_F", 0.0, 0.5, 0.05, 0.01)
            eta_S = st.slider("η_S", 0.0, 1.0, 0.15, 0.01)
        seed = st.number_input("Seed", value=43, step=1)
        run_button()
    r = com_run(T, d_mu, n_snis, K_gain, eta_dmn, eta_F, eta_S, alpha, beta, target, int(seed))
    tr = r["trace"]
    with right:
        c = st.columns(4)
        c[0].metric("QoC cuối", f"{r['QoC_final']:.3f}", delta=f"{r['QoC_final'] - target:+.3f} so với mục tiêu",
                    delta_color="off")
        c[1].metric("A_DMN cuối", f"{tr['A_dmn'][-1]:.4f}", delta=f"{tr['A_dmn'][-1] - tr['A_dmn'][0]:+.3f}",
                    delta_color="inverse")
        c[2].metric("Φ cuối", f"{tr['Phi'][-1]:.3f}", delta=f"{tr['Phi'][-1] - tr['Phi'][0]:+.3f}")
        c[3].metric("ΔS = ‖B*‖/‖B₀‖", f"{r['delta_S']:.3f}")
        third = max(1, T // 3)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            subplot_titles=["A_DMN(t)", "Φ(t)", "QoC(t)"])
        for i, (k, col) in enumerate([("A_dmn", RED), ("Phi", BLUE), ("QoC", AQUA)]):
            fig.add_scatter(x=tr["t"], y=tr[k], line=dict(color=col, width=2), name=k, row=i + 1, col=1)
        fig.add_hline(y=target, line=dict(color=GRAY, dash="dot"), row=3, col=1,
                      annotation_text="mục tiêu", annotation_position="bottom right")
        fig.add_vrect(x0=third, x1=2 * third, fillcolor="rgba(128,128,128,0.07)", line_width=0)
        for x0, x1, lab in [(0, third, "P1 · ức chế DMN"), (third, 2 * third, "P2 · tăng Φ"),
                            (2 * third, T, "P3 · mở rộng truy vấn")]:
            fig.add_annotation(x=(x0 + x1) / 2, y=1.07, xref="x", yref="paper", text=lab, showarrow=False,
                               font=dict(size=11, color=GRAY))
        for x in (third, 2 * third):
            fig.add_vline(x=x, line=dict(color=GRAY, width=1, dash="dot"))
        fig.update_xaxes(title_text="t", row=3, col=1)
        show(style(fig, None, height=580, legend=False))
        st.caption("QoC chỉ được tính ở Phase 3 (giữ 0 ở P1–P2) đúng như pseudocode của Algorithm 2.")


# =============================================================================
# §5.1 IIT — Φ
# =============================================================================
@st.cache_data(show_spinner=False)
@core.saving("phi_run", scope="bqi")
def phi_run(n, w_std, topology, seed):
    from bqi.iit_phi import integrated_information_Phi
    r = rng(seed)
    W = np.zeros((n, n))
    if topology == "Ngẫu nhiên":
        W = r.normal(0, w_std, (n, n))
    elif topology == "Vòng (ring)":
        for i in range(n):
            W[i, (i + 1) % n] = W[(i + 1) % n, i] = w_std
    elif topology == "Hai cụm rời":
        h = n // 2
        W[:h, :h] = r.normal(0, w_std, (h, h))
        W[h:, h:] = r.normal(0, w_std, (n - h, n - h))
    np.fill_diagonal(W, 0)
    state = r.integers(0, 2, n)
    t0 = time.time()
    res = integrated_information_Phi(W, state)
    return dict(W=W, state=state, Phi=res["Phi"], mech={str(k): v for k, v in res["mechanism_phis"].items()},
                elapsed=time.time() - t0)


def lab_iit_phi():
    header(r"\varphi(M)=\min_{P}\,\mathrm{EMD}\big(p^{P}(X_{fut}\mid M),\,p(X_{fut}\mid M)\big),\qquad "
           r"\#\text{partitions}=2^n-2")
    left, right = panel()
    with left, st.form("phi"):
        n = st.slider("Số phần tử n", 3, 7, 5, help="n = 7 mất vài giây — độ phức tạp mũ")
        topology = st.radio("Cấu trúc kết nối", ["Ngẫu nhiên", "Vòng (ring)", "Hai cụm rời", "Ngắt kết nối"])
        w_std = st.slider("Độ mạnh kết nối", 0.1, 4.0, 1.8, 0.1)
        seed = st.number_input("Seed", value=45, step=1)
        run_button()
    with st.spinner("Đang vét cạn mọi phân hoạch…"):
        r = phi_run(n, w_std, topology, int(seed))
    with right:
        c = st.columns(4)
        c[0].metric("Φ toàn hệ", f"{r['Phi']:.4f}")
        c[1].metric("Số cơ chế M", len(r["mech"]))
        c[2].metric("Phân hoạch (toàn hệ)", 2 ** n - 2)
        c[3].metric("Thời gian", f"{r['elapsed']:.2f}s")
        st.caption(f"Trạng thái hiện tại: `{''.join(map(str, r['state']))}`")
        a, b = st.columns([2, 3])
        with a:
            m = float(np.abs(r["W"]).max()) or 1.0
            fig = go.Figure(go.Heatmap(z=r["W"], zmin=-m, zmax=m, colorscale=core.DIVERGING,
                                       x=[str(i) for i in range(n)], y=[str(i) for i in range(n)],
                                       hovertemplate="W[%{y},%{x}] = %{z:.3f}<extra></extra>"))
            fig.update_yaxes(autorange="reversed")
            show(style(fig, "Ma trận kết nối W", xlab="nút đích", ylab="nút nguồn", height=340, legend=False))
        with b:
            s = pd.Series(r["mech"]).sort_values(ascending=False).head(15)
            fig = go.Figure(go.Bar(x=s.values[::-1], y=s.index[::-1], orientation="h", marker_color=BLUE,
                                   hovertemplate="M = %{y}<br>φ = %{x:.4f}<extra></extra>"))
            show(style(fig, "φ(M) của 15 cơ chế lớn nhất", xlab="φ (EMD)", height=340, legend=False))
        ns = np.arange(2, 31)
        fig = go.Figure(go.Scatter(x=ns, y=2.0 ** ns - 2, line=dict(color=RED, width=2), name="2ⁿ − 2"))
        fig.add_vline(x=n, line=dict(color=GRAY, dash="dot"), annotation_text=f"n = {n}")
        fig.update_yaxes(type="log", exponentformat="power")
        show(style(fig, "Mệnh đề NP-hard: số phân hoạch cần đánh giá tăng theo hàm mũ", xlab="n",
                   ylab="Số phân hoạch (log)", height=300, legend=False))


# =============================================================================
# §5.2 PCI
# =============================================================================
@st.cache_data(show_spinner=False)
@core.saving("pci_sample", scope="bqi")
def pci_sample(n_ch, n_t, level, seed):
    from bqi.pci import simulate_tms_eeg, pci_from_binary_matrix
    mat = simulate_tms_eeg(n_ch, n_t, level, seed=seed)
    return mat, pci_from_binary_matrix(mat)


@st.cache_data(show_spinner=False)
@core.saving("pci_dataset", scope="bqi")
def pci_dataset(n_ch, n_t, n_sub, seed):
    from bqi.pci import generate_pci_dataset
    return generate_pci_dataset(n_channels=n_ch, n_timepoints=n_t, n_subjects_per_state=n_sub, seed=seed)


@st.cache_data(show_spinner=False)
@core.saving("pci_sweep", scope="bqi")
def pci_sweep(n_ch, n_t, seed):
    from bqi.pci import simulate_tms_eeg, pci_from_binary_matrix
    levels = np.linspace(0.0, 1.0, 21)
    return levels, [np.mean([pci_from_binary_matrix(simulate_tms_eeg(n_ch, n_t, l, seed=seed + i))
                             for i in range(3)]) for l in levels]


def lab_pci():
    header(r"\mathrm{PCI}=\frac{c(L)\,\log_2 L}{L\,H(L)},\qquad H(L)=-p_1\log_2p_1-(1-p_1)\log_2(1-p_1)"
           r"\quad\text{(Casali 2013)}")
    from scipy.stats import spearmanr
    from bqi.pci import PCI_TABLE_REFERENCE, STATE_COMPLEXITY_LEVEL
    left, right = panel()
    with left, st.form("pci"):
        level = st.slider("Mức phức tạp (mẫu minh hoạ)", 0.0, 1.0, 0.55, 0.01)
        n_ch = st.select_slider("Số kênh EEG", [8, 12, 16, 24, 32], 16)
        n_t = st.select_slider("Số mẫu thời gian", [24, 32, 48, 64], 48)
        n_sub = st.slider("Subject / trạng thái", 3, 40, 8,
                          help="LZ76 chạy thuần Python: 32×64 kênh ≈ 0.2 s/subject")
        seed = st.number_input("Seed", value=46, step=1)
        run_button()
    mat, pci_val = pci_sample(n_ch, n_t, level, int(seed))
    with st.spinner(f"Đang mô phỏng {9 * n_sub} subject TMS-EEG + LZ76…"):
        df = pci_dataset(n_ch, n_t, n_sub, int(seed))
    scale = 1.0
    with right:
        order = sorted(PCI_TABLE_REFERENCE, key=lambda s: PCI_TABLE_REFERENCE[s]["mean"])
        means = df.groupby("state")["pci"].mean() * scale
        rho = spearmanr([means[s] for s in order], [PCI_TABLE_REFERENCE[s]["mean"] for s in order]).statistic
        c = st.columns(4)
        c[0].metric("PCI mẫu", f"{pci_val * scale:.3f}")
        c[1].metric("|s| (bit)", n_ch * n_t)
        c[2].metric("Spearman ρ vs bài báo", f"{rho:.3f}")
        c[3].metric("Thang đo", "[0, ~1]", help="Chuẩn hoá Casali 2013: nguồn ngẫu nhiên ≈ 1")
        a, b = st.columns([2, 3])
        with a:
            fig = go.Figure(go.Heatmap(z=mat, colorscale=[[0, "#f0efec"], [1, BLUE]], showscale=False,
                                       hovertemplate="kênh %{y}, t = %{x}: %{z}<extra></extra>"))
            fig.update_yaxes(autorange="reversed")
            show(style(fig, f"TMS-EEG nhị phân hoá (mức {level:.2f})", xlab="thời gian", ylab="kênh",
                       height=380, legend=False))
        with b:
            fig = go.Figure()
            fig.add_bar(y=order, x=[PCI_TABLE_REFERENCE[s]["mean"] for s in order], orientation="h",
                        name="Bài báo (Table PCI) ± SD", marker_color="rgba(138,137,133,0.45)",
                        error_x=dict(array=[PCI_TABLE_REFERENCE[s]["sd"] for s in order], color=GRAY))
            fig.add_scatter(y=df["state"], x=df["pci"] * scale, mode="markers", name="Từng subject",
                            marker=dict(color=ORANGE, size=5, opacity=0.3), hoverinfo="skip")
            fig.add_scatter(y=order, x=[means[s] for s in order], mode="markers", name="Mô phỏng (TB)",
                            marker=dict(color=ORANGE, size=11, line=dict(color="white", width=2)))
            fig.update_yaxes(categoryorder="array", categoryarray=order)
            show(style(fig, "PCI theo trạng thái ý thức", xlab="PCI", height=380))
        lv, sw = pci_sweep(n_ch, n_t, int(seed))
        fig = go.Figure(go.Scatter(x=lv, y=np.array(sw) * scale, mode="lines+markers",
                                   line=dict(color=BLUE, width=2), marker=dict(size=8)))
        for l in STATE_COMPLEXITY_LEVEL.values():
            fig.add_vline(x=l, line=dict(color="rgba(138,137,133,0.35)", width=1))
        show(style(fig, "PCI theo mức phức tạp (vạch dọc = mức hiệu chuẩn của 9 trạng thái)",
                   xlab="complexity_level", ylab="PCI", height=300, legend=False))
        st.caption("Đường cong bão hoà từ mức ≈ 0.4 nên 5 trạng thái ý thức cao gần như trùng nhau.")
        st.caption("Eq. 49 của bài để L(s) không xác định; đọc nguyên văn (`pci_bqi_literal`) cho ≈ log₂|s| "
                   "với chuỗi ngẫu nhiên nên không thể ở thang [0, 1] của Table PCI. Mức hiệu chuẩn 9 trạng thái "
                   "được đặt tay — đây là minh hoạ, không phải bằng chứng.")


# =============================================================================
# §7.2 DMN / Φ / QoC ODE
# =============================================================================
@st.cache_data(show_spinner=False)
@core.saving("ode_run", scope="bqi")
def ode_run(params_items, t_end, seed):
    from bqi.dmn_ode import simulate, steady_states, steady_states_paper
    P = dict(params_items)
    c = simulate(P, t_span=(0, t_end), n_points=600, meditator=False, seed=seed)
    m = simulate(P, t_span=(0, t_end), n_points=600, meditator=True, seed=seed)
    return c, m, steady_states(P), steady_states_paper(P)


@st.cache_data(show_spinner=False)
@core.saving("ode_population", scope="bqi")
def ode_population(n, t_end, seed):
    from bqi.dmn_ode import simulate, generate_ode_param_dataset
    df = generate_ode_param_dataset(n_subjects=n, seed=seed)
    out = {"control": [], "meditator": []}
    for _, row in df.iterrows():
        P = row.drop("subject").to_dict()
        for k, med in (("control", False), ("meditator", True)):
            out[k].append(simulate(P, t_span=(0, t_end), n_points=200, meditator=med,
                                   seed=seed + int(row["subject"])))
    return out


def lab_dmn_ode():
    header(r"\dot A=-k_{inh}A+k_{spont}\xi(t),\quad \dot\Phi=k_{int}\Big(1-\frac{A}{A_{max}}\Big)-k_d\Phi,\quad "
           r"\dot Q=\alpha\dot\Phi-\beta\dot A-\gamma Q")
    from bqi.dmn_ode import ODE_PARAMS_REFERENCE as REF
    left, right = panel()
    with left, st.form("ode"):
        st.caption("Mặc định = Table ode_params")
        P = dict(REF)
        P["k_inh"] = st.slider("k_inh (ức chế DMN)", 0.005, 0.3, REF["k_inh"], 0.001, format="%.3f")
        P["k_spont"] = st.slider("k_spont (kích hoạt tự phát)", 0.0, 0.2, REF["k_spont"], 0.001, format="%.3f")
        P["k_int"] = st.slider("k_int (tích hợp Φ)", 0.005, 0.3, REF["k_int"], 0.001, format="%.3f")
        P["k_d"] = st.slider("k_d (suy giảm Φ)", 0.005, 0.2, REF["k_d"], 0.001, format="%.3f")
        P["alpha"] = st.slider("α (Φ → QoC)", 0.0, 5.0, REF["alpha"], 0.01)
        P["beta"] = st.slider("β (DMN → QoC)", 0.0, 5.0, REF["beta"], 0.01)
        P["gamma"] = st.slider("γ (suy giảm QoC)", 0.01, 1.0, REF["gamma"], 0.01)
        P["sigma_xi"] = st.slider("σ_ξ (nhiễu)", 0.0, 0.5, REF["sigma_xi"], 0.01)
        t_end = st.slider("Thời gian mô phỏng (phút)", 10, 400, 40, 10)
        seed = st.number_input("Seed", value=43, step=1)
        run_button()
    c, m, ss, ss_paper = ode_run(tuple(sorted(P.items())), t_end, int(seed))
    with right:
        k = st.columns(3)
        for i, var in enumerate(["A_DMN", "Phi", "QoC"]):
            k[i].metric(f"{var} cuối — thiền giả", f"{m[var][-1]:.3f}",
                        delta=f"{m[var][-1] - c[var][-1]:+.3f} so với đối chứng",
                        delta_color="inverse" if var == "A_DMN" else "normal")
        fig = make_subplots(rows=1, cols=3, subplot_titles=["A_DMN(t)", "Φ(t)", "QoC(t)"], horizontal_spacing=0.06)
        for i, var in enumerate(["A_DMN", "Phi", "QoC"]):
            fig.add_scatter(x=c["t"], y=c[var], name="Đối chứng", line=dict(color=GRAY, width=2),
                            legendgroup="c", showlegend=i == 0, row=1, col=i + 1)
            fig.add_scatter(x=m["t"], y=m[var], name="Thiền giả (k_inh ↑ ~10×)", line=dict(color=BLUE, width=2),
                            legendgroup="m", showlegend=i == 0, row=1, col=i + 1)
        for i, var in enumerate(["A_DMN_star", "Phi_star"]):
            fig.add_hline(y=ss[var], line=dict(color=ORANGE, dash="dot"), row=1, col=i + 1)
        fig.update_xaxes(title_text="t (phút)")
        show(style(fig, "Quỹ đạo ODE — vạch cam = nghiệm dừng dạng đóng", height=340))
        a, b = st.columns([3, 2])
        with a:
            fig = go.Figure()
            fig.add_scatter(x=c["A_DMN"], y=c["Phi"], name="Đối chứng", line=dict(color=GRAY, width=2))
            fig.add_scatter(x=m["A_DMN"], y=m["Phi"], name="Thiền giả", line=dict(color=BLUE, width=2))
            fig.add_scatter(x=[ss["A_DMN_star"]], y=[ss["Phi_star"]], mode="markers", name="(A*, Φ*) dạng đóng",
                            marker=dict(color=ORANGE, size=12, symbol="x"))
            show(style(fig, "Chân dung pha A_DMN – Φ", xlab="A_DMN", ylab="Φ", height=320))
        with b:
            st.markdown("**Nghiệm dừng: dạng đóng vs mô phỏng (đối chứng, t cuối)**")
            keys = ["A_DMN_star", "Phi_star", "QoC_star"]
            st.dataframe(pd.DataFrame({
                "Biến": ["A_DMN*", "Φ*", "QoC*"],
                "Đã sửa": [ss[k] for k in keys],
                "Bài báo (eq. 55–57)": [ss_paper[k] for k in keys],
                "Mô phỏng": [c["A_DMN"][-1], c["Phi"][-1], c["QoC"][-1]],
            }), hide_index=True, width="stretch",
                column_config={k: st.column_config.NumberColumn(format="%.4f")
                               for k in ["Đã sửa", "Bài báo (eq. 55–57)", "Mô phỏng"]})
            st.caption("E[ξ] = 0 nên A* = 0; eq. 27 chỉ phụ thuộc đạo hàm nên γ·QoC* = 0 ⇒ **QoC* = 0**. "
                       f"Bài báo cho QoC* = {ss_paper['QoC_star']:.2f}. Kéo t lên 400 để thấy mô phỏng hội tụ về 0. "
                       "Mô hình đã sửa (QoC* ≠ 0) là `StateSpaceDMN`.")
    with st.expander("Mô phỏng quần thể (tham số lấy mẫu từ CI của Table ode_params)"):
        n_pop = st.slider("Số subject", 5, 100, 30, 5, key="ode_pop_n")
        if st.button("Chạy quần thể", icon=":material/groups:"):
            with st.spinner(f"Đang tích phân {2 * n_pop} hệ ODE…"):
                pop = ode_population(n_pop, t_end, int(seed))
            fig = make_subplots(rows=1, cols=3, subplot_titles=["A_DMN", "Φ", "QoC"], horizontal_spacing=0.06)
            for grp, col, fill, label in (("control", GRAY, "rgba(138,137,133,0.18)", "Đối chứng"),
                                          ("meditator", BLUE, "rgba(42,120,214,0.18)", "Thiền giả")):
                t = pop[grp][0]["t"]
                for i, var in enumerate(["A_DMN", "Phi", "QoC"]):
                    Y = np.array([s[var] for s in pop[grp]])
                    lo, mid, hi = np.percentile(Y, [5, 50, 95], axis=0)
                    fig.add_scatter(x=np.r_[t, t[::-1]], y=np.r_[hi, lo[::-1]], fill="toself", fillcolor=fill,
                                    line=dict(width=0), hoverinfo="skip", showlegend=False, row=1, col=i + 1)
                    fig.add_scatter(x=t, y=mid, line=dict(color=col, width=2), legendgroup=grp,
                                    name=f"{label} (trung vị, dải 5–95%)", showlegend=i == 0, row=1, col=i + 1)
            fig.update_xaxes(title_text="t (phút)")
            show(style(fig, f"Quần thể {n_pop} subject", height=320))


# =============================================================================
# §7.3 Meta-analysis
# =============================================================================
@st.cache_data(show_spinner=False)
@core.saving("meta_bootstrap", scope="bqi")
def meta_bootstrap(names, n_boot, seed):
    from bqi.meta_analysis import STUDIES_REFERENCE, generate_resampled_meta_dataset
    studies = [s for s in STUDIES_REFERENCE if s["study"] in names]
    return generate_resampled_meta_dataset(studies, n_bootstrap=n_boot, seed=seed)["bootstrap_pooled_d"].values


def lab_meta_analysis():
    header(r"\hat d_{RE}=\frac{\sum w_i d_i}{\sum w_i},\;w_i=\frac{1}{v_i+\tau^2},\qquad "
           r"I^2=\frac{Q-(k-1)}{Q}\times100\%")
    from bqi.meta_analysis import (STUDIES_REFERENCE, random_effects_meta_analysis, pooled_percent_change,
                                   hedges_g_variance)
    all_names = [s["study"] for s in STUDIES_REFERENCE]
    left, right = panel()
    with left, st.form("meta"):
        excl = st.multiselect("Loại bỏ nghiên cứu (phân tích độ nhạy)", all_names)
        n_boot = st.slider("Số lần bootstrap", 200, 20000, 5000, 200)
        seed = st.number_input("Seed", value=42, step=1)
        run_button()
    studies = [s for s in STUDIES_REFERENCE if s["study"] not in excl]
    if len(studies) < 3:
        st.error("Cần ít nhất 3 nghiên cứu.")
        return
    res = random_effects_meta_analysis(studies)
    pct = pooled_percent_change(studies)
    with right:
        c = st.columns(5)
        c[0].metric("Pooled d", f"{res['pooled_d']:.3f}", delta=f"{res['pooled_d'] - 1.70:+.3f} vs bài báo 1.70",
                    delta_color="off")
        c[1].metric("95% CI", f"[{res['ci'][0]:.2f}, {res['ci'][1]:.2f}]")
        c[2].metric("I²", f"{res['I2_percent']:.1f}%", delta=f"{res['I2_percent'] - 31:+.1f} vs 31%",
                    delta_color="off")
        c[3].metric("τ²", f"{res['tau2']:.4f}")
        c[4].metric("Δ% BOLD gộp", f"{pct['pooled_delta_pct']:.1f}%")

        d = np.array([s["d"] for s in studies])
        v = np.array([hedges_g_variance(s["d"], s["N"]) for s in studies])
        w = res["weights_re"] / res["weights_re"].sum()
        labels = [f"{s['study']} (N={s['N']})" for s in studies]
        lo, hi, pd_ = res["ci"][0], res["ci"][1], res["pooled_d"]
        fig = go.Figure()
        fig.add_scatter(x=d, y=labels, mode="markers", name="Nghiên cứu ± 95% CI",
                        error_x=dict(array=1.96 * np.sqrt(v), color=GRAY, thickness=1.5),
                        marker=dict(symbol="square", color=BLUE, size=6 + 60 * w, line=dict(color="white", width=1)),
                        customdata=np.c_[w * 100, [s["type"] for s in studies]],
                        hovertemplate="%{y}<br>d = %{x:.2f}<br>trọng số RE = %{customdata[0]:.1f}%"
                                      "<br>%{customdata[1]}<extra></extra>")
        fig.add_scatter(x=[pd_], y=["Gộp (RE)"], mode="markers", name=f"Gộp d = {pd_:.2f} [{lo:.2f}, {hi:.2f}]",
                        marker=dict(symbol="diamond-wide", color=ORANGE, size=22, line=dict(color="white", width=2)),
                        error_x=dict(type="data", symmetric=False, array=[hi - pd_], arrayminus=[pd_ - lo],
                                     color=ORANGE, thickness=2.5))
        fig.add_vline(x=pd_, line=dict(color=ORANGE, dash="dash", width=1))
        fig.add_vline(x=1.70, line=dict(color=GRAY, dash="dot", width=1), annotation_text="bài báo 1.70",
                      annotation_position="top")
        fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=labels + ["Gộp (RE)"])
        show(style(fig, "Forest plot — ức chế DMN do thiền (kích thước ô ∝ trọng số RE)", xlab="Cohen's d",
                   height=110 + 26 * (len(studies) + 1)))
        a, b = st.columns(2)
        with a:
            boot = meta_bootstrap(tuple(s["study"] for s in studies), n_boot, int(seed))
            q = np.percentile(boot, [2.5, 97.5])
            fig = go.Figure(go.Histogram(x=boot, nbinsx=60, marker_color=BLUE, name="bootstrap"))
            for x, txt in [(q[0], "2.5%"), (q[1], "97.5%")]:
                fig.add_vline(x=x, line=dict(color=BLUE, dash="dot"), annotation_text=txt)
            for x in (lo, hi):
                fig.add_vline(x=x, line=dict(color=ORANGE, dash="dot"))
            show(style(fig, f"Bootstrap pooled d (n = {n_boot}); cam = CI random-effects", xlab="d",
                       ylab="tần suất", height=320, legend=False))
            st.caption(f"CI bootstrap [{q[0]:.2f}, {q[1]:.2f}] hẹp hơn CI RE vì bootstrap dùng trọng số "
                       "fixed-effect (bỏ qua τ²).")
        with b:
            loo = [random_effects_meta_analysis([s for s in studies if s is not x])["pooled_d"] for x in studies]
            fig = go.Figure(go.Scatter(x=loo, y=[s["study"] for s in studies], mode="markers",
                                       marker=dict(color=BLUE, size=9),
                                       hovertemplate="bỏ %{y}<br>d = %{x:.3f}<extra></extra>"))
            fig.add_vline(x=pd_, line=dict(color=ORANGE, dash="dash"))
            fig.update_yaxes(autorange="reversed")
            show(style(fig, "Leave-one-out: d gộp khi bỏ từng nghiên cứu", xlab="pooled d", height=320,
                       legend=False))
        se = np.sqrt(v)
        fig = go.Figure(go.Scatter(x=d, y=se, mode="markers", marker=dict(color=BLUE, size=9), name="nghiên cứu",
                                   text=[s["study"] for s in studies],
                                   hovertemplate="%{text}<br>d = %{x:.2f}, SE = %{y:.3f}<extra></extra>"))
        ys = np.linspace(0, se.max() * 1.1, 20)
        for sign in (-1, 1):
            fig.add_scatter(x=pd_ + sign * 1.96 * ys, y=ys, mode="lines", line=dict(color=GRAY, dash="dot"),
                            hoverinfo="skip", showlegend=False)
        fig.add_vline(x=pd_, line=dict(color=ORANGE, dash="dash"))
        fig.update_yaxes(autorange="reversed")
        show(style(fig, "Funnel plot (đánh giá thiên lệch xuất bản)", xlab="Cohen's d", ylab="SE", height=320,
                   legend=False))
        with st.expander("Bảng 16 nghiên cứu (Table meta)"):
            st.dataframe(pd.DataFrame(STUDIES_REFERENCE), hide_index=True, width="stretch")


# =============================================================================
# §9.2 Hopfield
# =============================================================================
@st.cache_data(show_spinner=False)
@core.saving("hop_curve", scope="bqi")
def hop_curve(n, flip, beta, trials, max_p, seed):
    from bqi.hopfield import retrieval_error_experiment
    counts = sorted(set([int(x) for x in np.geomspace(1, max_p, 22)] + list(range(1, min(12, max_p) + 1))))
    return retrieval_error_experiment(n=n, pattern_counts=counts, flip_fraction=flip, trials_per_count=trials,
                                      beta=beta, seed=seed)


@st.cache_data(show_spinner=False)
@core.saving("hop_demo", scope="bqi")
def hop_demo(n, P, flip, beta, seed):
    from bqi.hopfield import (random_patterns, corrupt, store_patterns_classic, retrieve_classic,
                              retrieve_modern, transformer_attention_retrieve)
    pats = random_patterns(P, n, seed=seed)
    target = pats[0]
    probe = corrupt(target, flip, seed=seed + 1)
    return dict(target=target, probe=probe, classic=retrieve_classic(store_patterns_classic(pats), probe),
                modern=retrieve_modern(pats, probe, beta=beta), transformer=transformer_attention_retrieve(pats, probe))


def lab_hopfield():
    header(r"\text{Classic: } N_{max}\approx0.14\,n,\qquad \text{Modern: } x^{new}=X^\top\mathrm{softmax}(\beta Xx),"
           r"\;N_{max}\sim e^{n/2}")
    from bqi.hopfield import theoretical_capacity_classic, theoretical_capacity_modern
    left, right = panel()
    with left, st.form("hop"):
        n = st.select_slider("Số neuron n", [16, 25, 36, 49, 64, 100], 36)
        flip = st.slider("Tỉ lệ bit nhiễu của probe", 0.0, 0.45, 0.1, 0.01)
        beta = st.slider("β (modern)", 0.1, 20.0, 8.0, 0.1)
        trials = st.slider("Trial / số pattern", 2, 30, 8)
        max_p = st.select_slider("Số pattern tối đa", [50, 100, 300, 1000, 2000], 300)
        demo_p = st.slider("Số pattern cho minh hoạ", 1, 300, 20)
        seed = st.number_input("Seed", value=42, step=1)
        run_button()
    with st.spinner("Đang đo lỗi truy hồi…"):
        df = hop_curve(n, flip, beta, trials, max_p, int(seed))
    with right:
        cap = theoretical_capacity_classic(n)
        c = st.columns(3)
        c[0].metric("Dung lượng classic 0.14n", f"{cap:.1f}")
        c[1].metric("Dung lượng modern e^{n/2}", fmt(theoretical_capacity_modern(n)))
        above = df[df.n_patterns > 2 * cap]
        c[2].metric("Lỗi modern TB khi P > 2×0.14n", f"{above.modern_error_rate.mean():.3f}" if len(above) else "—")
        fig = go.Figure()
        for col, name, color, dash in [("classic_error_rate", "Classic Hopfield", RED, None),
                                       ("modern_error_rate", "Modern Hopfield (softmax)", BLUE, None),
                                       ("transformer_error_rate", "Transformer attention (β = 1/√n)", AQUA, "dash")]:
            fig.add_scatter(x=df.n_patterns, y=df[col], name=name, mode="lines+markers",
                            line=dict(color=color, width=2, dash=dash), marker=dict(size=8))
        fig.add_vline(x=cap, line=dict(color=GRAY, dash="dot"), annotation_text="0.14n")
        fig.update_xaxes(type="log")
        show(style(fig, f"Tỉ lệ lỗi truy hồi theo số pattern lưu trữ (n = {n})", xlab="Số pattern P (log)",
                   ylab="Tỉ lệ bit sai", height=380))
        r = hop_demo(n, demo_p, flip, beta, int(seed))
        side = int(np.sqrt(n))
        panels = [("Pattern gốc", r["target"]), ("Probe nhiễu", r["probe"]), ("Classic", r["classic"]),
                  ("Modern", r["modern"]), ("Transformer", r["transformer"])]
        titles = [t if i == 0 else f"{t} · {np.mean(x != r['target']) * 100:.0f}% sai"
                  for i, (t, x) in enumerate(panels)]
        fig = make_subplots(rows=1, cols=5, horizontal_spacing=0.03, subplot_titles=titles)
        for i, (_, x) in enumerate(panels):
            fig.add_trace(go.Heatmap(z=x.reshape(side, side), zmin=-1, zmax=1, showscale=False,
                                     colorscale=[[0, "#f0efec"], [1, VIOLET]], hoverinfo="skip"), row=1, col=i + 1)
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, autorange="reversed")
        show(style(fig, f"Minh hoạ truy hồi với P = {demo_p} pattern", height=260, legend=False))


# =============================================================================
# §4.4 / 9.4 Graph + Kuramoto
# =============================================================================
@st.cache_data(show_spinner=False)
@core.saving("graph_run", scope="bqi")
def graph_run(n_nodes, k, p, boost, seed):
    import networkx as nx
    from bqi.spectral_graph import (build_small_world_brain_graph, fiedler_value, small_world_index,
                                    global_efficiency, meditation_graph_transform)
    G = build_small_world_brain_graph(n_nodes=n_nodes, k=k, p_rewire=p, seed=seed)
    G2 = meditation_graph_transform(G, boost_fraction=boost, seed=seed + 1)
    connected = nx.is_connected(G)

    def metrics(g):
        return dict(fiedler=fiedler_value(g), sigma=small_world_index(g, n_random=3) if connected else np.nan,
                    E=global_efficiency(g), C=nx.average_clustering(g))
    added = [e for e in G2.edges() if not G.has_edge(*e)]
    return dict(edges=list(G.edges()), added=added, before=metrics(G), after=metrics(G2), connected=connected)


@st.cache_data(show_spinner=False)
@core.saving("ws_sweep", scope="bqi")
def ws_sweep(n_nodes, k, seed):
    import networkx as nx
    ps = np.geomspace(1e-4, 1, 14)
    G0 = nx.watts_strogatz_graph(n_nodes, k, 0, seed=seed)
    C0, L0 = nx.average_clustering(G0), nx.average_shortest_path_length(G0)
    C, L = [], []
    for p in ps:
        G = nx.connected_watts_strogatz_graph(n_nodes, k, p, seed=seed)
        C.append(nx.average_clustering(G) / C0 if C0 else np.nan)
        L.append(nx.average_shortest_path_length(G) / L0)
    return ps, C, L


@st.cache_data(show_spinner=False)
@core.saving("kuramoto_sweep", scope="bqi")
def kuramoto_sweep(n_osc, seed):
    from bqi.spectral_graph import simulate_kuramoto
    Ks = np.linspace(0, 5, 26)
    r = [np.mean([simulate_kuramoto(n_osc, K, T=12.0, dt=0.02, seed=seed + i)[-100:].mean() for i in range(3)])
         for K in Ks]
    return Ks, r


@st.cache_data(show_spinner=False)
@core.saving("kuramoto_traces", scope="bqi")
def kuramoto_traces(n_osc, Ks, seed):
    from bqi.spectral_graph import simulate_kuramoto
    return {K: simulate_kuramoto(n_osc, K, T=12.0, dt=0.02, seed=seed) for K in Ks}


@st.cache_data(show_spinner=False)
@core.saving("kuramoto_states", scope="bqi")
def kuramoto_states(n_osc, n_trials, seed):
    from bqi.spectral_graph import generate_kuramoto_dataset
    return generate_kuramoto_dataset(n_oscillators=n_osc, n_trials=n_trials, seed=seed)


def lab_spectral_graph():
    header(r"\lambda_2(L)\;(\text{Fiedler}),\quad \sigma_{SW}=\frac{C/C_{rand}}{L/L_{rand}},\quad "
           r"\dot\theta_i=\omega_i+K r\sin(\psi-\theta_i),\; r=\Big|\tfrac1N\sum_j e^{i\theta_j}\Big|")
    left, right = panel()
    with left, st.form("graph"):
        st.markdown("**Mạng não small-world**")
        n_nodes = st.slider("Số vùng não (nút)", 20, 200, 80, 10)
        k = st.select_slider("Bậc k (láng giềng)", [2, 4, 6, 8, 10], 6)
        p = st.slider("Xác suất nối lại p", 0.0, 1.0, 0.1, 0.01)
        boost = st.slider("Thiền: tỉ lệ cạnh tầm xa thêm vào", 0.0, 0.5, 0.08, 0.01)
        st.markdown("**Kuramoto**")
        n_osc = st.slider("Số dao động tử", 10, 200, 50, 10)
        K_show = st.multiselect("Hiển thị r(t) với K =", [0.5, 1.0, 1.5, 2.0, 3.0, 5.0], [0.5, 1.5, 3.0])
        seed = st.number_input("Seed", value=42, step=1)
        run_button()
    with st.spinner("Đang tính chỉ số đồ thị…"):
        g = graph_run(n_nodes, k, p, boost, int(seed))
    with right:
        b, a = g["before"], g["after"]
        c = st.columns(4)
        c[0].metric("Fiedler λ₂", f"{b['fiedler']:.3f}", delta=f"{a['fiedler'] - b['fiedler']:+.3f} khi thiền")
        c[1].metric("σ_SW", f"{b['sigma']:.2f}" if np.isfinite(b["sigma"]) else "—",
                    delta=f"{a['sigma'] - b['sigma']:+.2f} khi thiền" if np.isfinite(b["sigma"]) else None)
        c[2].metric("E_glob", f"{b['E']:.3f}", delta=f"{100 * (a['E'] - b['E']) / b['E']:+.1f}% khi thiền")
        c[3].metric("Clustering C", f"{b['C']:.3f}", delta=f"{a['C'] - b['C']:+.3f} khi thiền")
        if not g["connected"]:
            st.warning("Đồ thị không liên thông — σ_SW không xác định, λ₂ = 0.")
        th = 2 * np.pi * np.arange(n_nodes) / n_nodes
        x, y = np.cos(th), np.sin(th)

        def edge_xy(edges):
            ex, ey = [], []
            for u, v in edges:
                ex += [x[u], x[v], None]
                ey += [y[u], y[v], None]
            return ex, ey
        left2, right2 = st.columns([2, 3])
        with left2:
            fig = go.Figure()
            ex, ey = edge_xy(g["edges"])
            fig.add_scatter(x=ex, y=ey, mode="lines", line=dict(color="rgba(138,137,133,0.35)", width=1),
                            name="Cạnh gốc", hoverinfo="skip")
            ex, ey = edge_xy(g["added"])
            fig.add_scatter(x=ex, y=ey, mode="lines", line=dict(color=ORANGE, width=1.5),
                            name=f"Thêm khi thiền ({len(g['added'])})", hoverinfo="skip")
            fig.add_scatter(x=x, y=y, mode="markers", marker=dict(color=BLUE, size=7, line=dict(color="white", width=1)),
                            name="Vùng não", text=[f"nút {i}" for i in range(n_nodes)], hoverinfo="text")
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False, scaleanchor="x")
            show(style(fig, "Mạng Watts–Strogatz (bố cục vòng)", height=420))
        with right2:
            ps, C, L = ws_sweep(n_nodes, k, int(seed))
            fig = go.Figure()
            fig.add_scatter(x=ps, y=C, name="C(p)/C(0)", mode="lines+markers", line=dict(color=BLUE, width=2))
            fig.add_scatter(x=ps, y=L, name="L(p)/L(0)", mode="lines+markers", line=dict(color=ORANGE, width=2))
            fig.add_vline(x=max(p, 1e-4), line=dict(color=GRAY, dash="dot"), annotation_text=f"p = {p}")
            fig.update_xaxes(type="log")
            show(style(fig, "Vùng small-world: L giảm sớm, C giữ cao", xlab="p (log)", ylab="tỉ lệ", height=420))

        st.markdown("##### Đồng bộ Kuramoto")
        Ks, rK = kuramoto_sweep(n_osc, int(seed))
        Kc = 2 * np.sqrt(2 * np.pi) / np.pi
        left2, right2 = st.columns(2)
        with left2:
            fig = go.Figure(go.Scatter(x=Ks, y=rK, mode="lines+markers", line=dict(color=BLUE, width=2),
                                       marker=dict(size=8), name="r dừng"))
            fig.add_vline(x=Kc, line=dict(color=ORANGE, dash="dash"),
                          annotation_text=f"K_c = 2/(πg(0)) ≈ {Kc:.2f}")
            show(style(fig, "Chuyển pha đồng bộ (ω ~ N(0,1))", xlab="Độ ghép K", ylab="r dừng", height=320,
                       legend=False))
        with right2:
            traces = kuramoto_traces(n_osc, tuple(sorted(K_show)), int(seed))
            fig = go.Figure()
            for i, (K, tr) in enumerate(traces.items()):
                fig.add_scatter(x=np.arange(len(tr)) * 0.02, y=tr, name=f"K = {K}",
                                line=dict(color=core.SERIES[i % 8], width=2))
            show(style(fig, "Tham số trật tự r(t)", xlab="t", ylab="r", height=320))
        kd = kuramoto_states(min(n_osc, 60), 5, int(seed))
        pivot = kd.groupby(["state", "band"])["r"].mean().unstack()[["theta", "alpha", "gamma"]].sort_values("theta")
        fig = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale=core.SEQ_BLUE,
                                   text=np.round(pivot.values, 2), texttemplate="%{text}",
                                   hovertemplate="%{y} · %{x}<br>r̄ = %{z:.3f}<extra></extra>",
                                   colorbar=dict(title="r̄")))
        show(style(fig, "r̄ theo trạng thái × băng tần (cf. Table kuramoto)", height=320, legend=False))


# =============================================================================
# §6 Quantum
# =============================================================================
@st.cache_data(show_spinner=False)
@core.saving("quantum_run", scope="bqi")
def quantum_run(dim, gammas, T, purity, seed):
    from bqi.quantum_decoherence import simulate_entropy_dynamics
    return {g: simulate_entropy_dynamics(dim, g, T=T, dt=0.01, purity0=purity, seed=seed) for g in gammas}


@st.cache_data(show_spinner=False)
@core.saving("quantum_grid", scope="bqi")
def quantum_grid(seed):
    from bqi.quantum_decoherence import generate_decoherence_dataset
    return generate_decoherence_dataset(dims=(2, 4, 8, 16), gammas=(0.25, 1.0, 4.0, 16.0), n_trials=5, seed=seed)


def lab_quantum_decoherence():
    header(r"S(\rho)=-\mathrm{Tr}[\rho\ln\rho],\qquad \rho_{ij}(t+dt)=\rho_{ij}e^{-\gamma dt}\;(i\neq j),\qquad "
           r"\tau=\frac{\hbar}{E_G},\;E_G\approx\frac{Gm^2}{\Delta x}")
    from bqi.quantum_decoherence import orch_or_collapse_time, G_NEWTON
    left, right = panel()
    with left, st.form("q"):
        dim = st.select_slider("Chiều Hilbert", [2, 3, 4, 8, 16, 32], 4)
        gammas = st.multiselect("Tốc độ khử kết hợp γ", [0.1, 0.3, 1.0, 2.0, 5.0, 10.0, 30.0], [0.3, 2.0, 10.0])
        T = st.slider("Thời gian T", 0.5, 10.0, 3.0, 0.5)
        purity = st.slider("Độ thuần ban đầu", 0.5, 1.0, 0.98, 0.01)
        st.markdown("**Orch-OR**")
        log_m = st.slider("log₁₀ khối lượng (kg)", -27.0, -15.0, -23.0, 0.25)
        log_dx = st.slider("log₁₀ Δx (m)", -15.0, -6.0, -9.0, 0.25)
        seed = st.number_input("Seed", value=42, step=1)
        run_button()
    with right:
        if gammas:
            res = quantum_run(dim, tuple(sorted(gammas)), T, purity, int(seed))
            fig = go.Figure()
            for i, (g, r) in enumerate(res.items()):
                fig.add_scatter(x=r["t"], y=r["S"], name=f"γ = {g}", line=dict(color=core.SERIES[i % 8], width=2))
            fig.add_hline(y=np.log(dim), line=dict(color=GRAY, dash="dot"), annotation_text=f"ln({dim})")
            show(style(fig, "Entropy von Neumann dưới khử kết hợp", xlab="t", ylab="S(ρ)", height=360))
            st.caption("Dephasing giữ nguyên đường chéo nên S hội tụ về entropy của phân bố population "
                       "(≤ ln dim), không nhất thiết đạt ln dim.")
        m_kg, dx = 10 ** log_m, 10 ** log_dx
        tau = orch_or_collapse_time(m_kg, dx)
        c = st.columns(3)
        c[0].metric("Khối lượng", f"{m_kg:.2e} kg")
        c[1].metric("E_G", f"{G_NEWTON * m_kg ** 2 / dx:.2e} J")
        c[2].metric("τ = ħ/E_G", f"{tau:.3e} s", help="Penrose–Hameroff: ~25 ms ứng với nhịp gamma 40 Hz")
        a, b = st.columns(2)
        with a:
            ms = np.logspace(-27, -15, 60)
            fig = go.Figure(go.Scatter(x=ms, y=[orch_or_collapse_time(mm, dx) for mm in ms],
                                       line=dict(color=VIOLET, width=2), name="τ(m)"))
            fig.add_hline(y=0.025, line=dict(color=ORANGE, dash="dash"), annotation_text="25 ms (gamma 40 Hz)")
            fig.add_scatter(x=[m_kg], y=[tau], mode="markers", marker=dict(color=VIOLET, size=12), name="lựa chọn")
            fig.update_xaxes(type="log", exponentformat="power")
            fig.update_yaxes(type="log", exponentformat="power")
            show(style(fig, f"Thời gian sụp đổ Orch-OR (Δx = 1e{log_dx:g} m)", xlab="m (kg)", ylab="τ (s)",
                       height=320, legend=False))
        with b:
            df = quantum_grid(int(seed))
            pv = df.assign(ratio=df.S_final / df.S_max).groupby(["dim", "gamma"])["ratio"].mean().unstack()
            fig = go.Figure(go.Heatmap(z=pv.values, x=[f"γ={g}" for g in pv.columns], y=[f"dim={d}" for d in pv.index],
                                       colorscale=core.SEQ_BLUE, zmin=0, zmax=1, text=np.round(pv.values, 2),
                                       texttemplate="%{text}", colorbar=dict(title="S/S_max")))
            show(style(fig, "S_final / ln(dim) sau T = 3", height=320, legend=False))


# =============================================================================
LAB_FUNCS = {
    "predictive_coding": lab_predictive_coding, "attention": lab_attention, "bqi_algorithm": lab_bqi_algorithm,
    "com_algorithm": lab_com_algorithm, "iit_phi": lab_iit_phi, "pci": lab_pci, "dmn_ode": lab_dmn_ode,
    "meta_analysis": lab_meta_analysis, "hopfield": lab_hopfield, "spectral_graph": lab_spectral_graph,
    "quantum_decoherence": lab_quantum_decoherence,
}
LAB_FUNCS[choice]()

with st.expander("Kết quả đã lưu (tái hiện BQI — tách biệt với bài Q1)", icon=":material/save:"):
    ex = core.list_explorations("bqi")
    st.caption(f"Mọi phép tính ở trang này tự động ghi (một lần cho mỗi bộ tham số) vào "
               f"`{os.path.relpath(core.EXPLORE_SCOPES['bqi'], core.PROJECT_ROOT)}/<hàm>/<khoá>.json` "
               "(+ `.npz` cho mảng, `.csv` cho bảng). Kết quả bài Q1 nằm riêng trong `outputs/results/model/`.")
    if ex.empty:
        st.caption("Chưa có bản ghi nào.")
    else:
        mod_kinds = {k for k in ex["kind"].unique()}
        st.dataframe(ex.rename(columns={"time": "Thời điểm", "kind": "Hàm", "key": "Khoá", "inputs": "Tham số",
                                        "file": "File"}),
                     hide_index=True, width="stretch", height=min(300, 40 + 35 * len(ex)),
                     column_config={"Tham số": st.column_config.TextColumn(width="large")})
        st.download_button("Tải index.csv", ex.to_csv(index=False).encode("utf-8"),
                           file_name="bqi_explorations_index.csv", mime="text/csv", icon=":material/download:")

with st.expander(f"Test liên quan — tests/{core.test_file_for(choice)}"):
    for t in info["tests"].get(core.test_file_for(choice), []):
        st.markdown(f"- `{t['name']}`" + (f" — {t['doc']}" if t["doc"] else ""))
    st.page_link(core.PAGES["tests"], label="Chạy ở trang Kiểm thử", icon=":material/checklist:")
