"""Consistent empty, loading and error presentation."""

import streamlit as st

from ui.components.models import PanelState


def render_state(state: PanelState) -> bool:
    """Return whether the caller should render its ready content."""
    if state.status == 'ready':
        return True
    defaults = {'empty': 'Gösterilecek veri yok.', 'loading': 'Hazırlanıyor…', 'error': 'İşlem tamamlanamadı.'}
    if state.status not in defaults:
        raise ValueError(f'Bilinmeyen panel durumu: {state.status}')
    renderer = st.error if state.status == 'error' else st.info
    renderer(state.message or defaults[state.status])
    return False
