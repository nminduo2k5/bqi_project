"""
BQI Dashboard — giao diện trực quan cho dự án "The Brain as a Query Interface".

Chạy:  python main.py dashboard
  hoặc streamlit run dashboard/app.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

import core  # noqa: F401  (đưa PROJECT_ROOT vào sys.path trước khi các trang import bqi)

st.set_page_config(page_title="BQI Dashboard", page_icon="🧠", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
  .block-container { padding-top: 2.2rem; padding-bottom: 3rem; }
  [data-testid="stMetricValue"] { font-size: 1.55rem; }
  [data-testid="stMetricLabel"] p { font-size: 0.85rem; }
  .bqi-eq { font-family: ui-monospace, Consolas, monospace; font-size: 0.85rem; opacity: 0.8; }
</style>
""", unsafe_allow_html=True)

VIEWS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "views")
pages = {
    "Dự án": [
        st.Page(os.path.join(VIEWS, "overview.py"), title="Tổng quan", icon=":material/dashboard:", default=True),
        st.Page(os.path.join(VIEWS, "lab.py"), title="Phòng thí nghiệm", icon=":material/science:", url_path="lab"),
    ],
    "Pipeline": [
        st.Page(os.path.join(VIEWS, "data.py"), title="Datasets", icon=":material/table_chart:", url_path="data"),
        st.Page(os.path.join(VIEWS, "experiments.py"), title="Thực nghiệm", icon=":material/insights:",
                url_path="experiments"),
        st.Page(os.path.join(VIEWS, "tests.py"), title="Kiểm thử", icon=":material/checklist:", url_path="tests"),
    ],
    "Bài Q1": [
        st.Page(os.path.join(VIEWS, "model.py"), title="Pipeline mô hình", icon=":material/architecture:",
                url_path="model"),
    ],
}
core.PAGES = {p.url_path or "overview": p for group in pages.values() for p in group}
nav = st.navigation(pages)

with st.sidebar:
    st.caption("**BQI** · The Brain as a Query Interface")
    st.caption(f"Python {sys.version.split()[0]}  \n`{core.PROJECT_ROOT}`")

nav.run()
