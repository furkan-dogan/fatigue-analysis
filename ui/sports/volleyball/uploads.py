"""Single-video intake; automatic interpretation belongs to the analysis pipeline."""
import hashlib
import streamlit as st
from ui.components.upload import video_uploader
from src.sports.volleyball.service import create_review, load_review
from ui.sports.volleyball.discovery import run_discovery


def render_uploads(store):
    upload = video_uploader('Voleybol videosu', key='volleyball_upload')
    if st.button('Videoyu analiz et', disabled=upload is None, key='volleyball_create', type='primary'):
        try:
            with st.spinner('Video hazırlanıyor…'):
                content = upload.getvalue()
                token = (hashlib.sha256(content).hexdigest(), upload.name, str(store.root))
                saved = st.session_state.setdefault('volleyball_intake_saved', {})
                if token in saved:
                    current = load_review(saved[token], store=store)
                else:
                    current = create_review(content, upload.name, store=store)
                    saved[token] = current['session_id']
                st.session_state['volleyball_review'] = current
            if current['result'].get('analysis', {}).get('mode') != 'automatic':
                current = run_discovery(current, store)
                saved[token] = current['session_id']
            st.session_state['volleyball_view'] = 'result'
            st.rerun()
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(f'Video açılamadı: {exc}')
