"""Video playback through Streamlit's managed media serving."""
from pathlib import Path
import math
import streamlit as st
from ui.components.models import PanelState
from ui.components.state import render_state


def video_player(path: Path | str, start_time: float = 0.0, *, state=PanelState()):
    if not render_state(state):
        return
    file = Path(path)
    if not file.is_file():
        st.info('Video bulunamadı.')
        return
    if not math.isfinite(start_time) or start_time < 0:
        st.error('Video başlangıç zamanı geçersiz.')
        return
    try:
        st.video(str(file), start_time=start_time)
    except OSError:
        st.error('Video okunamadı.')
