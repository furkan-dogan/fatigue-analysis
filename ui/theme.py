"""Shared Streamlit page configuration and responsive styling."""

import streamlit as st


def configure_page() -> None:
    st.set_page_config(
        page_title="Video Analizi",
        page_icon="🥊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
          [data-testid="stSidebar"],
          [data-testid="collapsedControl"] {
            display: none;
          }
          section[data-testid="stSidebar"] {
            width: 0 !important;
          }
          .block-container {
            max-width: 100%;
            padding-left: clamp(0.75rem, 2vw, 2.5rem);
            padding-right: clamp(0.75rem, 2vw, 2.5rem);
          }
          [data-testid="stMetric"] {
            min-width: 0;
            overflow: visible;
          }
          [data-testid="stMetricLabel"] p {
            white-space: normal !important;
            overflow-wrap: anywhere !important;
            line-height: 1.2 !important;
          }
          [data-testid="stMetricValue"] {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            line-height: 1.08 !important;
            font-size: clamp(1.45rem, 2.3vw, 2.7rem) !important;
          }
          [data-testid="stMetricDelta"] {
            white-space: normal !important;
            overflow-wrap: anywhere !important;
          }
          div[data-testid="stDataFrame"] {
            overflow-x: auto;
          }
          @media (max-width: 900px) {
            .block-container {
              padding-top: 1rem;
            }
            [data-testid="column"] {
              min-width: min(100%, 220px) !important;
            }
          }
          @media (max-width: 640px) {
            h1 { font-size: 1.75rem !important; }
            h2 { font-size: 1.45rem !important; }
            h3 { font-size: 1.25rem !important; }
            [data-testid="stMetricValue"] {
              font-size: 1.35rem !important;
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
