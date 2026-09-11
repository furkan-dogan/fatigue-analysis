"""Show supplied quality notices without inferring measurement validity."""

from collections.abc import Sequence

import streamlit as st

from ui.components.models import PanelState, QualityNotice
from ui.components.state import render_state


def quality_panel(notices: Sequence[QualityNotice], *, state: PanelState = PanelState()) -> None:
    if not render_state(state):
        return
    if not notices:
        render_state(PanelState('empty', 'Kalite değerlendirmesi henüz yok.'))
        return
    renderers = {'info': st.info, 'warning': st.warning, 'error': st.error, 'success': st.success}
    for notice in notices:
        if notice.level not in renderers:
            raise ValueError(f'Bilinmeyen kalite seviyesi: {notice.level}')
        renderers[notice.level](notice.message)
