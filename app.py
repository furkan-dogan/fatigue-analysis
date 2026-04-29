"""Streamlit dashboard — taekwondo yorgunluk analizi.

Çalıştırmak için:
    .venv/bin/streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Video Analizi",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🥊 Video Analizi")

from ui.video_analysis import render
render()
