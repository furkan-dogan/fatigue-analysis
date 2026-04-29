"""Streamlit dashboard — taekwondo yorgunluk analizi.

Çalıştırmak için:
    .venv/bin/streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Kickboks Yorgunluk Analizi",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🥊 Kickboks Analizi")
page = st.sidebar.radio("Sayfa", ["Video Analizi", "EMG Sync"], label_visibility="collapsed")

if page == "Video Analizi":
    from pages.video_analysis import render
    render()
elif page == "EMG Sync":
    from pages.emg_sync import render
    render()
