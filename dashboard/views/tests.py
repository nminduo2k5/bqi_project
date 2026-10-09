"""Trang Kiểm thử — chạy pytest, xem kết quả từng test, coverage và lịch sử các lần chạy."""
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core
from core import GOOD, CRITICAL, GRAY, BLUE, WARNING, show, style

st.title("✅ Kiểm thử")
st.caption("Chạy `pytest` trên `tests/` trong tiến trình con, đọc kết quả JUnit XML (và coverage nếu có "
           "`pytest-cov`). Mỗi lần chạy được lưu vào `outputs/results/test_history.json`.")

info = core.codebase()
test_files = sorted(info["tests"])
n_tests = sum(len(v) for v in info["tests"].values())
has_cov = core.has_pytest_cov()

# =============================================================================
# Chạy
# =============================================================================
with st.form("pytest"):
    a, b = st.columns([3, 2])
    files = a.multiselect("File test", test_files, placeholder=f"Tất cả ({len(test_files)} file, {n_tests} test)",
                          default=[f for f in st.query_params.get_all("t") if f in test_files])
    kw = b.text_input("Lọc từ khoá (-k)", placeholder='vd: "modern or capacity"')
    c = st.columns(3)
    stop = c[0].toggle("Dừng ở lỗi đầu tiên (-x)")
    cov = c[1].toggle("Đo coverage", disabled=not has_cov,
                      help=None if has_cov else "Cần cài pytest-cov: pip install pytest-cov")
    go_run = st.form_submit_button("Chạy pytest", type="primary", icon=":material/play_arrow:")

if go_run:
    targets = [os.path.join("tests", f) for f in files] or ["tests"]
    st.query_params["t"] = files
    with st.status("Đang chạy pytest…", expanded=True) as status:
        res = core.run_pytest(targets, st.empty(), keyword=kw.strip() or None, stop_on_fail=stop,
                              coverage=cov and has_cov)
        df = res["results"]
        bad = int(df["status"].isin(["failed", "error"]).sum()) if len(df) else 0
        ok = res["exit_code"] == 0
        status.update(label=f"{len(df)} test · {bad} lỗi · {res['elapsed']:.1f}s (exit {res['exit_code']})",
                      state="complete" if ok else "error", expanded=False)
    st.session_state["pytest_last"] = res

res = st.session_state.get("pytest_last")

# =============================================================================
# Kết quả lần chạy gần nhất (trong phiên)
# =============================================================================
if res is not None:
    df = res["results"]
    counts = df["status"].value_counts() if len(df) else pd.Series(dtype=int)
    n_bad = int(counts.get("failed", 0) + counts.get("error", 0))
    st.subheader(f"Kết quả · {res['timestamp']}")
    c = st.columns(5)
    c[0].metric("Tổng", len(df))
    c[1].metric("✅ Pass", int(counts.get("passed", 0)))
    c[2].metric("❌ Fail / lỗi", n_bad)
    c[3].metric("⏭️ Skip", int(counts.get("skipped", 0)))
    c[4].metric("Thời gian", f"{res['elapsed']:.1f}s", help="Gồm cả khởi động pytest")

    if df.empty:
        st.error("Không thu được kết quả nào (lỗi thu thập test?) — xem log bên dưới.")
    else:
        if n_bad == 0:
            st.success(f"Tất cả {len(df)} test đều pass.", icon=":material/check_circle:")
        t_tab, t_file, t_slow, t_cov = st.tabs(["Từng test", "Theo file", "Chậm nhất", "Coverage"])
        with t_tab:
            only_bad = st.toggle("Chỉ hiện test lỗi", value=n_bad > 0)
            view = df[df["status"].isin(["failed", "error"])] if only_bad else df
            st.dataframe(pd.DataFrame({
                "": view["status"].map(core.STATUS_ICON), "File": view["file"], "Test": view["test"],
                "Thời gian (s)": view["duration"].round(3), "Thông báo": view["message"].str.slice(0, 200),
            }), hide_index=True, width="stretch",
                column_config={"": st.column_config.TextColumn(width=40),
                               "Thông báo": st.column_config.TextColumn(width="large")})
            for _, r in df[df["status"].isin(["failed", "error"])].iterrows():
                with st.expander(f"{core.STATUS_ICON[r['status']]} {r['file']}::{r['test']}"):
                    st.code(r["details"] or r["message"], language="python")
        with t_file:
            g = df.groupby(["file", "status"]).size().unstack(fill_value=0)
            fig = go.Figure()
            for s in ["passed", "failed", "error", "skipped"]:
                if s in g:
                    fig.add_bar(y=g.index, x=g[s], name=s, orientation="h", marker_color=core.STATUS_COLOR[s])
            fig.update_layout(barmode="stack")
            fig.update_yaxes(autorange="reversed")
            show(style(fig, "Số test theo file", height=80 + 28 * len(g), xlab="Số test"))
        with t_slow:
            d = df.nlargest(15, "duration").iloc[::-1]
            fig = go.Figure(go.Bar(y=d["file"].str.removesuffix(".py") + "::" + d["test"], x=d["duration"],
                                   orientation="h", marker_color=BLUE))
            show(style(fig, "15 test chậm nhất", height=80 + 26 * len(d), xlab="giây", legend=False))
        with t_cov:
            cv = res.get("coverage")
            if cv is None or cv.empty:
                st.caption("Chưa đo coverage cho lần chạy này — bật **Đo coverage** khi chạy."
                           if has_cov else "Cài `pytest-cov` để đo coverage: `pip install pytest-cov`.")
            else:
                total = 100 * (1 - cv["missing"].sum() / max(cv["statements"].sum(), 1))
                st.metric("Coverage tổng (bqi/)", f"{total:.1f}%")
                colors = [GOOD if p >= 80 else WARNING if p >= 50 else CRITICAL for p in cv["percent"]]
                fig = go.Figure(go.Bar(y=cv["module"], x=cv["percent"], orientation="h", marker_color=colors,
                                       text=[f"{p:.0f}%" for p in cv["percent"]], textposition="outside"))
                fig.update_xaxes(range=[0, 108])
                show(style(fig, "Coverage theo module", height=80 + 28 * len(cv), xlab="%", legend=False))
                st.dataframe(cv.rename(columns={"module": "Module", "statements": "Câu lệnh", "missing": "Thiếu",
                                                "percent": "%", "missing_lines": "Dòng chưa chạy"}),
                             hide_index=True, width="stretch",
                             column_config={"%": st.column_config.ProgressColumn(min_value=0, max_value=100,
                                                                                 format="%.0f%%")})
    with st.expander("Log pytest"):
        st.code(res["log"] or "(trống)", language="text")

# =============================================================================
# Lịch sử
# =============================================================================
hist = core.load_test_history()
st.subheader("Lịch sử chạy")
if not hist:
    st.caption("Chưa có lần chạy nào từ dashboard.")
else:
    h = pd.DataFrame(hist)
    h["run"] = range(1, len(h) + 1)
    fig = go.Figure()
    fig.add_bar(x=h["run"], y=h["passed"], name="pass", marker_color=GOOD)
    fig.add_bar(x=h["run"], y=h["failed"], name="fail/lỗi", marker_color=CRITICAL)
    fig.add_bar(x=h["run"], y=h["skipped"], name="skip", marker_color=GRAY)
    fig.update_layout(barmode="stack")
    fig.update_traces(customdata=h[["timestamp", "scope", "elapsed"]].values,
                      hovertemplate="#%{x} · %{customdata[0]}<br>%{customdata[1]}<br>%{y} test"
                                    "<br>%{customdata[2]}s<extra>%{fullData.name}</extra>")
    show(style(fig, f"{len(h)} lần chạy gần nhất", height=280, xlab="Lần chạy", ylab="Số test"))
    st.dataframe(h.iloc[::-1][["timestamp", "scope", "keyword", "total", "passed", "failed", "skipped", "elapsed",
                               "exit_code"]].rename(columns={
        "timestamp": "Thời điểm", "scope": "Phạm vi", "keyword": "-k", "total": "Tổng", "passed": "Pass",
        "failed": "Fail", "skipped": "Skip", "elapsed": "Giây", "exit_code": "Exit"}),
        hide_index=True, width="stretch", height=240)

# =============================================================================
# Danh mục test
# =============================================================================
st.subheader("Danh mục test")
st.caption("Đọc bằng AST từ `tests/` — docstring mô tả khẳng định được kiểm tra.")
mod_by_test = {core.test_file_for(m["key"]): m for m in core.MODULES}
for f in test_files:
    m = mod_by_test.get(f)
    title = f"`{f}` · {len(info['tests'][f])} test" + (f" — §{m['sec']} {m['name']}" if m else "")
    with st.expander(title):
        for t in info["tests"][f]:
            st.markdown(f"- `{t['name']}`" + (f" — {t['doc']}" if t["doc"] else ""))
        if m:
            st.page_link(core.PAGES["lab"], label="Mở module trong Phòng thí nghiệm", icon=":material/science:",
                         query_params={"m": m["key"]})
