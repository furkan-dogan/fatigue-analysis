"""Video input shared by sport pages. Analysis validation belongs to the service."""

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from ui.components.models import PanelState
from ui.components.state import render_state


def video_uploader(label: str, *, key: str, disabled: bool = False,
                   help: str | None = None, state: PanelState = PanelState()) -> UploadedFile | None:
    if not render_state(state):
        return None
    return st.file_uploader(label, type=['mp4', 'avi', 'mov'], key=key, disabled=disabled, help=help)
