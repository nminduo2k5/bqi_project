"""Trang Tổng quan — kiến trúc hệ thống, các module, scorecard đối chiếu bài báo."""
import os

import pandas as pd
import streamlit as st

import core

st.title("🧠 BQI — The Brain as a Query Interface")
st.caption("Cài đặt Python các thuật toán, mô hình toán và kết quả định lượng của bài báo "
           "(Vu, Phenikaa University): 11 module · dữ liệu tổng hợp · 10 thực nghiệm · bộ kiểm thử pytest.")

info = core.codebase()
n_tests = sum(len(v) for v in info["tests"].values())
loc = sum(m["loc"] for m in info["modules"].values())
datasets = core.list_datasets()
figures = core.list_figures()
history = core.load_test_history()

c = st.columns(6)
c[0].metric("Module thuật toán", len(core.MODULES), help="Các file trong bqi/")
c[1].metric("Dòng mã (bqi/)", f"{loc:,}")
c[2].metric("Test pytest", n_tests, help="Đếm bằng AST trong tests/")
c[3].metric("Dataset CSV", len(datasets))
c[4].metric("Hình đã xuất", len(figures))
if history:
    h = history[-1]
    c[5].metric("Lần test gần nhất", f"{h['passed']}/{h['total']} pass",
                delta=f"{h['failed']} lỗi" if h["failed"] else "tất cả xanh",
                delta_color="inverse" if h["failed"] else "normal", help=h["timestamp"])
else:
    c[5].metric("Lần test gần nhất", "—", help="Chưa chạy test từ dashboard")

# -----------------------------------------------------------------------------
st.subheader("Kiến trúc hệ thống")
left, right = st.columns([3, 2], gap="large")
with left:
    dot = ["digraph G {", 'rankdir=LR; compound=true; bgcolor="transparent"; nodesep=0.18; ranksep=0.55;',
           'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, '
           'color="#8a8985", fillcolor="#f0efec", fontcolor="#0b0b0b"];',
           'edge [color="#8a8985", arrowsize=0.6];',
           'paper [label="Bài báo\\nThe Brain as a\\nQuery Interface", fillcolor="#cde2fb"];',
           'main [label="main.py\\n(CLI)", fillcolor="#fbd9c9"];',
           'gen [label="datasets/\\ngenerate_datasets.py", fillcolor="#c9eedd"];',
           'exp [label="experiments/\\nrun_all.py", fillcolor="#c9eedd"];',
           'tests [label="tests/\\npytest", fillcolor="#c9eedd"];',
           'csv [label="datasets/generated\\n*.csv", shape=cylinder];',
           'out [label="outputs/\\nfigures + summary.json", shape=cylinder];',
           'dash [label="dashboard/\\n(Streamlit)", fillcolor="#fbd9c9"];',
           'subgraph cluster_bqi { label="bqi/"; fontname="Helvetica"; fontsize=10; color="#8a8985"; style=rounded;']
    for m in core.MODULES:
        dot.append(f'{m["key"]} [label="{m["key"]}\\n§{m["sec"]}"];')
    dot.append("}")
    dot.append(f'paper -> {core.MODULES[0]["key"]} [lhead=cluster_bqi, style=dashed];')
    for k, meta in info["modules"].items():
        for d in meta["deps"]:
            dot.append(f"{d} -> {k} [color=\"#2a78d6\"];")
    for m in core.MODULES:
        if m["data"]:
            dot.append(f"{m['key']} -> gen;")
        if m["exp"]:
            dot.append(f"{m['key']} -> exp;")
    dot += ["gen -> csv;", "exp -> out;", "main -> gen [style=dashed];", "main -> exp [style=dashed];",
            "main -> tests [style=dashed];", "dash -> main [style=dashed];", "csv -> dash;", "out -> dash;",
            "tests -> dash;", "}"]
    st.graphviz_chart("\n".join(dot), width="stretch")
    st.caption("Mũi tên xanh = phụ thuộc nội bộ giữa các module (phân tích AST). "
               "Nét đứt = điều phối qua CLI.")
with right:
    st.markdown("""
**Luồng dữ liệu**

1. **`bqi/`** — mỗi file dịch trực tiếp phương trình / pseudocode của một section bài báo.
2. **`datasets/generate_datasets.py`** — sinh dữ liệu tổng hợp *mới* (không sao chép bài báo)
   bằng chính các thuật toán, kèm file `*_paper_reference.csv` để so sánh.
3. **`experiments/run_all.py`** — chạy 10 thực nghiệm, xuất 9 hình PNG + `summary.json`.
4. **`tests/`** — kiểm tra tính đúng toán học & khẳng định định tính.
5. **`main.py`** — CLI gộp tất cả; dashboard gọi lại đúng CLI này.

**Trong dashboard**

- **Phòng thí nghiệm**: chỉnh tham số từng thuật toán, chạy trực tiếp mã `bqi/`.
- **Datasets / Thực nghiệm / Kiểm thử**: chạy pipeline thật, xem log trực tiếp, xem kết quả.
""")
    p1, p2, p3 = st.columns(3)
    p1.page_link(core.PAGES["lab"], label="Mở phòng thí nghiệm", icon=":material/science:")
    p2.page_link(core.PAGES["tests"], label="Chạy kiểm thử", icon=":material/checklist:")
    p3.page_link(core.PAGES["model"], label="Bài Q1: pipeline mô hình", icon=":material/architecture:")

# -----------------------------------------------------------------------------
st.subheader("Đối chiếu khẳng định của bài báo")
st.caption("Tính trực tiếp từ mã `bqi/` (không đọc từ file). ✅ khớp · ⚠️ lệch / cần lưu ý · ❌ sai.")
seed = st.number_input("Seed", value=42, step=1, key="score_seed", width=160)
checks = core.paper_scorecard(int(seed))
n_pass = sum(c["status"] == "pass" for c in checks)
n_warn = sum(c["status"] == "warn" for c in checks)
n_fail = sum(c["status"] == "fail" for c in checks)
s = st.columns(4)
s[0].metric("Khẳng định đã kiểm", len(checks))
s[1].metric("✅ Khớp", n_pass)
s[2].metric("⚠️ Cần lưu ý", n_warn)
s[3].metric("❌ Sai", n_fail)

score_df = pd.DataFrame([{
    "": core.STATUS_ICON[c["status"]],
    "Module": f"§{core.MODULE_BY_KEY[c['module']]['sec']} {c['module']}",
    "Khẳng định": c["claim"], "Đo được": c["measured"], "Kỳ vọng": c["expected"], "Ghi chú": c["note"],
} for c in checks])
st.dataframe(score_df, hide_index=True, width="stretch",
             column_config={"": st.column_config.TextColumn(width=40),
                            "Khẳng định": st.column_config.TextColumn(width="medium"),
                            "Đo được": st.column_config.TextColumn(width="large"),
                            "Ghi chú": st.column_config.TextColumn(width="large")})

warns = [c for c in checks if c["status"] != "pass"]
if warns:
    with st.expander(f"Giải thích {len(warns)} điểm cần lưu ý", expanded=False):
        for c in warns:
            st.markdown(f"{core.STATUS_ICON[c['status']]} **{c['claim']}** — {c['measured']}  \n{c['note']}")

# -----------------------------------------------------------------------------
st.subheader("Các module")
rows = []
for m in core.MODULES:
    meta = info["modules"][m["key"]]
    tests = info["tests"].get(core.test_file_for(m["key"]), [])
    rows.append({"§": m["sec"], "Module": m["key"], "Nội dung": m["name"], "Mô tả": m["desc"],
                 "Dòng": meta["loc"], "Hàm/lớp": len(meta["functions"]) + len(meta["classes"]),
                 "Test": len(tests), "Thực nghiệm": m["exp"] or "—", "Dataset": m["data"] or "—",
                 "Phụ thuộc": ", ".join(meta["deps"]) or "—"})
mod_df = pd.DataFrame(rows)
st.dataframe(mod_df, hide_index=True, width="stretch",
             column_config={"Mô tả": st.column_config.TextColumn(width="large"),
                            "Dòng": st.column_config.ProgressColumn(min_value=0, max_value=int(mod_df["Dòng"].max()),
                                                                    format="%d"),
                            "Test": st.column_config.NumberColumn(format="%d")})

with st.expander("API công khai của từng module (hàm & lớp)"):
    for m in core.MODULES:
        meta = info["modules"][m["key"]]
        items = [f"`{c}` (class)" for c in meta["classes"]] + [f"`{f}()`" for f in meta["functions"]
                                                                if not f.startswith("_")]
        st.markdown(f"**{m['key']}.py** — " + " · ".join(items))

if not os.path.isdir(core.DATA_DIR) or datasets.empty:
    st.info("Chưa có dataset nào — vào trang **Datasets** để sinh dữ liệu.")
