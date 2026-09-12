"""Restrained, responsive visual system for the local analysis workspace."""
import streamlit as st


def configure_page() -> None:
    st.set_page_config(page_title="Voleybol | Video analizi", page_icon="🏐",
                       layout="wide", initial_sidebar_state="auto")
    st.markdown("""
    <style>
      :root { --ink:#172b42; --muted:#56687b; --line:#dce4ec; --accent:#087f75; }
      .stApp { background:#f5f7fa; color:var(--ink); }
      [data-testid="stAppDeployButton"] { display:none; }
      .block-container { max-width:1320px; padding:4.5rem 2.25rem 3rem; }
      [data-testid="stSidebar"] { background:#edf2f6; border-right:1px solid var(--line); }
      [data-testid="stSidebar"] .block-container { padding:1.5rem 1rem; }
      h1,h2,h3 { color:var(--ink); letter-spacing:-.035em; }
      h1 { font-size:2.1rem !important; font-weight:700 !important; padding-bottom:.35rem !important; }
      h2 { font-size:1.45rem !important; }
      h3 { font-size:1.12rem !important; }
      [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color:var(--muted) !important; line-height:1.55; }
      [data-testid="stVerticalBlockBorderWrapper"] > div { border-color:var(--line); }
      .st-key-upload_panel, .st-key-video_panel, .st-key-measurement_panel {
        background:white; border-radius:14px; padding:1.15rem;
      }
      .st-key-video_panel video { max-height:520px; background:#111c2a; object-fit:contain; border-radius:8px; }
      .st-key-video_panel [data-testid="stVideo"] { display:flex; justify-content:center; }
      [data-testid="stMetric"] { min-width:0; }
      [data-testid="stMetricLabel"] p { white-space:normal; line-height:1.4; font-size:.85rem; color:var(--muted); }
      [data-testid="stMetricValue"] { font-size:clamp(1.35rem,2vw,2rem); color:var(--ink); white-space:normal; }
      button[kind="primary"] { background:var(--accent); border-color:var(--accent); border-radius:8px; font-weight:600; }
      button:focus-visible, input:focus-visible { outline:3px solid #45b8ad !important; outline-offset:2px; }
      [data-testid="stFileUploaderDropzone"] { background:#f8fafc; border:1px dashed #9eafbf; border-radius:10px; }
      .workspace-brand { font-size:1.1rem; font-weight:750; letter-spacing:.08em; color:#173b4e; margin-bottom:.25rem; }
      .workspace-eyebrow { color:#087f75; font-size:.72rem; font-weight:700; letter-spacing:.13em; text-transform:uppercase; }
      [data-testid="stExpander"] { background:white; border-radius:10px; }
      @media(max-width:900px) {
        .block-container { padding:4.5rem 1rem 2rem; }
        .st-key-video_panel video { max-height:440px; }
      }
      @media(max-width:640px) {
        h1 { font-size:1.7rem !important; }
        [data-testid="stHorizontalBlock"] { flex-wrap:wrap; }
        [data-testid="stColumn"] { min-width:100% !important; }
        .st-key-video_panel video { max-height:380px; }
      }
    </style>
    """, unsafe_allow_html=True)
