"""Shared honest empty state for analysis screens under development."""
import streamlit as st
from ui.components.models import PanelState
from ui.components.state import render_state


def planned_analysis(title: str, description: str) -> None:
    st.title(title)
    st.caption(description)
    render_state(PanelState('empty', 'Bu branşın video analizi henüz kullanıma hazır değil.'))
