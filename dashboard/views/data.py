"""Trang Datasets — sinh dữ liệu tổng hợp qua CLI, duyệt và trực quan hoá từng file CSV."""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core
from core import BLUE, ORANGE, GRAY, fmt, show, style

st.title("🗂️ Datasets tổng hợp")
st.caption("Dữ liệu sinh *mới* bằng chính các thuật toán trong `bqi/` qua `python main.py generate-data`, "
           "kèm các bảng `*_paper_reference.csv` để đối chiếu với bài báo.")

# =============================================================================
# Sinh dữ liệu
# =============================================================================
with st.expander("Sinh lại dữ liệu", icon=":material/play_circle:", expanded=core.list_datasets().empty):
    with st.form("gen"):
        a, b = st.columns([2, 1])
        only = a.multiselect("Dataset", list(core.DATASET_LABELS), format_func=core.DATASET_LABELS.get,
                             placeholder="Tất cả")
        seed = b.number_input("Seed", value=42, step=1)
        c = st.columns(5)
        n_pci = c[0].number_input("PCI: subject/trạng thái", 2, 500, 40, help="--n-subjects-per-state")
        n_ode = c[1].number_input("ODE: số subject", 10, 10000, 500, help="--n-ode-subjects")
        n_boot = c[2].number_input("Meta: bootstrap", 100, 50000, 5000, step=500, help="--n-bootstrap")
        n_kur = c[3].number_input("Kuramoto: trial", 1, 200, 15, help="--n-kuramoto-trials")
        n_q = c[4].number_input("Quantum: trial", 1, 200, 10, help="--n-quantum-trials")
        go_gen = st.form_submit_button("Sinh dữ liệu", type="primary", icon=":material/play_arrow:")
    if go_gen:
        args = ["main.py", "generate-data", "--seed", str(int(seed)), "--n-subjects-per-state", str(n_pci),
                "--n-ode-subjects", str(n_ode), "--n-bootstrap", str(n_boot), "--n-kuramoto-trials", str(n_kur),
                "--n-quantum-trials", str(n_q)]
        if only:
            args += ["--only", ",".join(only)]
        st.code("python " + " ".join(args), language="bash")
        with st.status("Đang sinh dữ liệu…", expanded=True) as status:
            code, _, elapsed = core.run_streaming(args, st.empty())
            status.update(label=f"{'Hoàn tất' if code == 0 else 'Lỗi'} sau {elapsed:.1f}s (exit {code})",
                          state="complete" if code == 0 else "error", expanded=code != 0)
        st.cache_data.clear()

datasets = core.list_datasets()
if datasets.empty:
    st.info("Chưa có dataset nào trong `datasets/generated/` — dùng mục **Sinh lại dữ liệu** phía trên.")
    st.stop()

c = st.columns(4)
c[0].metric("File CSV", len(datasets))
c[1].metric("Tổng số dòng", f"{int(datasets['rows'].clip(lower=0).sum()):,}")
c[2].metric("Dung lượng", f"{datasets['size_kb'].sum():,.1f} KB")
c[3].metric("Cập nhật gần nhất", datasets["modified"].max())

st.dataframe(datasets.rename(columns={"file": "File", "rows": "Dòng", "cols": "Cột", "size_kb": "KB",
                                      "modified": "Sửa đổi"}),
             hide_index=True, width="stretch")


# =============================================================================
# Trực quan hoá theo từng file
# =============================================================================
@st.cache_data(show_spinner=False)
def load(name: str, _mtime: float) -> pd.DataFrame:
    return pd.read_csv(os.path.join(core.DATA_DIR, name))


def read(name: str) -> pd.DataFrame | None:
    p = os.path.join(core.DATA_DIR, name)
    return load(name, os.path.getmtime(p)) if os.path.isfile(p) else None


def plot_pci(df):
    ref = read("pci_paper_reference.csv")
    norm, k = True, 1.0
    order = df.groupby("state")["pci"].mean().sort_values().index.tolist()
    fig = go.Figure()
    fig.add_box(y=df["state"], x=df["pci"] / k, orientation="h", name="Mô phỏng",
                marker_color=BLUE, boxmean=True)
    if norm and ref is not None:
        r = ref.set_index("state").reindex(order)
        fig.add_scatter(y=r.index, x=r["mean"], error_x=dict(array=r["sd"], color=ORANGE), mode="markers",
                        marker=dict(color=ORANGE, size=10, symbol="diamond"), name="Bài báo (mean ± sd)")
    fig.update_yaxes(categoryorder="array", categoryarray=order)
    show(style(fig, "PCI theo trạng thái ý thức (chuẩn hoá Casali 2013)", height=420, xlab="PCI"))


def plot_pci_ref(df):
    d = df.sort_values("mean")
    fig = go.Figure(go.Bar(y=d["state"], x=d["mean"], orientation="h", marker_color=ORANGE,
                           error_x=dict(array=d["sd"], color=GRAY), customdata=d["N"],
                           hovertemplate="%{y}<br>PCI = %{x:.3f}<br>N = %{customdata:.0f}<extra></extra>"))
    show(style(fig, "Table PCI của bài báo (mean ± sd)", height=380, xlab="PCI", legend=False))


def plot_ode(df):
    ref = read("ode_params_paper_reference.csv")
    params = [c for c in df.columns if c != "subject" and df[c].std() > 0]
    cols = st.columns(4)
    for i, p in enumerate(params):
        fig = go.Figure(go.Histogram(x=df[p], marker_color=BLUE, nbinsx=30, name=p))
        if ref is not None and p in ref:
            fig.add_vline(x=float(ref[p].iloc[0]), line=dict(color=ORANGE, width=2, dash="dash"))
        with cols[i % 4]:
            show(style(fig, p, height=220, legend=False), key=f"ode_{p}")
    st.caption("Đường đứt cam = giá trị trong Table ode_params của bài báo. "
               + ", ".join(c for c in df.columns if c != "subject" and c not in params) + " là hằng số.")


def plot_meta_boot(df):
    x = df.iloc[:, 0]
    lo, hi = np.percentile(x, [2.5, 97.5])
    fig = go.Figure(go.Histogram(x=x, nbinsx=60, marker_color=BLUE, name="bootstrap"))
    for v, txt in [(lo, "2.5%"), (hi, "97.5%")]:
        fig.add_vline(x=v, line=dict(color=GRAY, dash="dot"), annotation_text=txt)
    fig.add_vline(x=1.70, line=dict(color=ORANGE, width=2, dash="dash"), annotation_text="bài báo d = 1.70")
    show(style(fig, "Phân phối bootstrap của pooled Cohen's d", height=360, xlab="pooled d", legend=False))
    c = st.columns(4)
    c[0].metric("Trung bình", f"{x.mean():.3f}")
    c[1].metric("Độ lệch chuẩn", f"{x.std():.3f}")
    c[2].metric("KTC 95% bootstrap", f"[{lo:.2f}, {hi:.2f}]")
    c[3].metric("Số mẫu", f"{len(x):,}")


def plot_meta_ref(df):
    d = df.sort_values("d")
    fig = go.Figure(go.Scatter(x=d["d"], y=d["study"], mode="markers",
                               marker=dict(size=np.sqrt(d["N"]) * 2.2, color=BLUE, line=dict(width=0)),
                               customdata=np.c_[d["N"], d["delta_pct"], d["type"]],
                               hovertemplate="%{y}<br>d = %{x}<br>N = %{customdata[0]}<br>"
                                             "ΔDMN = %{customdata[1]}%<br>%{customdata[2]}<extra></extra>"))
    fig.add_vline(x=1.70, line=dict(color=ORANGE, dash="dash"), annotation_text="pooled d = 1.70")
    show(style(fig, "16 nghiên cứu trong Table meta (kích thước ∝ √N)", height=480, xlab="Cohen's d",
               legend=False))


def plot_kuramoto(df):
    order = df.groupby("state")["r"].mean().sort_values().index.tolist()
    fig = go.Figure()
    for i, band in enumerate(df["band"].unique()):
        d = df[df["band"] == band]
        fig.add_box(x=d["state"], y=d["r"], name=band, marker_color=core.SERIES[i % 8])
    fig.update_layout(boxmode="group")
    fig.update_xaxes(categoryorder="array", categoryarray=order)
    show(style(fig, "Tham số trật tự Kuramoto r theo trạng thái × băng tần", height=420, ylab="r"))


def plot_quantum(df):
    g = df.assign(ratio=df.S_final / df.S_max).groupby(["dim", "gamma"])["ratio"].agg(["mean", "std"]).reset_index()
    fig = go.Figure()
    for i, (dim, d) in enumerate(g.groupby("dim")):
        fig.add_scatter(x=d["gamma"], y=d["mean"], error_y=dict(array=d["std"]), name=f"dim = {dim}",
                        line=dict(color=core.SERIES[i % 8], width=2))
    fig.update_xaxes(type="log")
    show(style(fig, "Entropy cuối / ln(dim) theo tốc độ khử kết hợp γ", height=380, xlab="γ (log)",
               ylab="S_final / S_max"))


PLOTS = {
    "pci_synthetic_subjects.csv": plot_pci, "pci_paper_reference.csv": plot_pci_ref,
    "ode_params_synthetic_subjects.csv": plot_ode, "dmn_meta_analysis_bootstrap.csv": plot_meta_boot,
    "dmn_meta_analysis_paper_reference.csv": plot_meta_ref, "kuramoto_synthetic_trials.csv": plot_kuramoto,
    "quantum_decoherence_synthetic.csv": plot_quantum,
}


def plot_generic(df):
    num = df.select_dtypes("number").columns.tolist()
    if not num:
        st.caption("Bảng văn bản — không có cột số để vẽ.")
        return
    a, b = st.columns(2)
    col = a.selectbox("Cột số", num)
    cats = [c for c in df.columns if c not in num and df[c].nunique() <= 30]
    by = b.selectbox("Nhóm theo", ["—"] + cats)
    fig = go.Figure()
    if by == "—":
        fig.add_histogram(x=df[col], marker_color=BLUE)
    else:
        for i, (k, d) in enumerate(df.groupby(by)):
            fig.add_box(y=d[col], name=str(k), marker_color=core.SERIES[i % 8])
    show(style(fig, col, height=360, legend=False))


st.subheader("Khám phá dataset")
files = datasets["file"].tolist()
default = st.query_params.get("f", files[0])
name = st.selectbox("File", files, index=files.index(default) if default in files else 0)
st.query_params["f"] = name
df = read(name)

t_plot, t_data, t_stats = st.tabs(["Biểu đồ", "Dữ liệu", "Thống kê"])
with t_plot:
    PLOTS.get(name, plot_generic)(df)
with t_data:
    st.dataframe(df, hide_index=True, width="stretch", height=420)
    st.download_button("Tải CSV", df.to_csv(index=False).encode("utf-8"), file_name=name, mime="text/csv",
                       icon=":material/download:")
with t_stats:
    num = df.select_dtypes("number")
    if num.empty:
        st.caption("Không có cột số.")
    else:
        desc = num.describe().T
        st.dataframe(desc.map(fmt), width="stretch")
    cats = df.select_dtypes(exclude="number").columns
    if len(cats) and not num.empty:
        by = st.selectbox("Trung bình theo nhóm", cats)
        st.dataframe(df.groupby(by)[num.columns.tolist()].mean().map(fmt), width="stretch")
