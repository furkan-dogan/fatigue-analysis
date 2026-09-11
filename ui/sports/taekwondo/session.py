"""Upload controls and transient Streamlit session state."""
import pandas as pd
import streamlit as st
from src.sports.taekwondo.service import analyze_pair
from ui.components.upload import video_uploader


def load_session():
    before, after = st.columns(2)
    with before:
        pre_upload = video_uploader('Antrenman öncesi video', key='dv_pre')
    with after:
        post_upload = video_uploader('Antrenman sonrası video', key='dv_post')
    ready = pre_upload is not None and post_upload is not None
    if st.button('İki videoyu analiz et', disabled=not ready, type='primary') and ready:
        progress = st.progress(0, text='Analiz başlıyor…')
        try:
            pre, post, directory = analyze_pair(
                pre_upload.getvalue(), post_upload.getvalue(),
                lambda value, text: progress.progress(value, text=text),
            )
        except Exception as exc:
            st.error(f'Analiz tamamlanamadı: {exc}')
            return None
        finally:
            progress.empty()
        # Replace the displayed pair only after both analyses succeed.
        st.session_state.update(dv_pre_result=pre, dv_post_result=post,
                                dv_pre_df=pd.DataFrame(pre.frame_rows),
                                dv_post_df=pd.DataFrame(post.frame_rows), dv_tmp=directory)
    pre, post = st.session_state.get('dv_pre_result'), st.session_state.get('dv_post_result')
    if pre is None or post is None:
        st.info('Her iki videoyu yükleyip analiz başlatın.')
        return None
    return pre, post, st.session_state.get('dv_pre_df'), st.session_state.get('dv_post_df')
