"""Upload controls, persistent history and active display state."""
import pandas as pd
import streamlit as st
from src.adapters.analysis_store import AnalysisStore
from src.sports.taekwondo.service import analyze_pair, load_pair, retry_pair
from ui.components.upload import video_uploader
from ui.components.history import analysis_history


def set_analysis(pair):
    pre, post, directory = pair
    st.session_state['taekwondo_analysis'] = {
        'pre': pre, 'post': post, 'pre_df': pd.DataFrame(pre.frame_rows),
        'post_df': pd.DataFrame(post.frame_rows), 'directory': directory,
    }


def load_session():
    store = AnalysisStore()
    with st.expander('Kayıtlı analizler'):
        records = store.sessions('taekwondo')
        for row in records:
            row['display_label'] = f"{row['label']} · {row['created_at'][:19]} · {row['completed']}/2 tamamlandı · {row['id'][:8]}"
        action, selected = analysis_history(records, key='taekwondo_history')
        if any(row['unfinished'] for row in records):
            st.caption('Tamamlanmamış kayıtlar korunur. Kesilen çalışmayı işaretleyip yeni revizyon başlatabilirsiniz.')
            if st.button('Kesilen çalışmaları işaretle', key='taekwondo_recover'):
                store.recover_interrupted()
                st.rerun()
        if action == 'open':
            try:
                set_analysis(load_pair(selected, store=store))
            except (OSError, ValueError) as exc:
                st.error(f'Kayıt açılamadı: {exc}')
    before, after = st.columns(2)
    with before:
        pre_upload = video_uploader('Antrenman öncesi video', key='taekwondo_upload_pre')
    with after:
        post_upload = video_uploader('Antrenman sonrası video', key='taekwondo_upload_post')
    label = st.text_input('Analiz adı', value='Taekwondo analizi', key='taekwondo_label')
    ready = pre_upload is not None and post_upload is not None
    start = st.button('İki videoyu analiz et', disabled=not ready, type='primary', key='taekwondo_analyze')
    if (start and ready) or action == 'retry':
        progress = st.progress(0, text='Analiz başlıyor…')
        update = lambda value, text: progress.progress(value, text=text)
        try:
            pair = retry_pair(selected, update, store=store) if action == 'retry' else analyze_pair(
                pre_upload.getvalue(), post_upload.getvalue(), update, store=store,
                names=(pre_upload.name, post_upload.name), label=label,
            )
            set_analysis(pair)
        except Exception as exc:
            st.error(f'Analiz tamamlanamadı; kayıtlı kaynaklar korunuyor: {exc}')
        else:
            st.rerun()
        finally:
            progress.empty()
    analysis = st.session_state.get('taekwondo_analysis', {})
    pre, post = analysis.get('pre'), analysis.get('post')
    if pre is None or post is None:
        st.info('Kayıtlı bir analiz açın veya iki video yükleyin.')
        return None
    return pre, post, analysis.get('pre_df'), analysis.get('post_df')
