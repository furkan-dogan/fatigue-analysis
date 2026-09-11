"""Shared history selector, independent of sport and storage."""
import streamlit as st


def analysis_history(records, *, key):
    if not records:
        st.caption('Henüz kayıtlı analiz yok.')
        return None, None
    labels = {row['id']: row['display_label'] for row in records}
    selected = st.selectbox('Kayıtlı analiz', list(labels), format_func=labels.get, key=f'{key}_selected')
    left, right = st.columns(2)
    with left:
        if st.button('Kaydı aç', key=f'{key}_open'):
            return 'open', selected
    with right:
        if st.button('Yeni revizyon olarak analiz et', key=f'{key}_retry'):
            return 'retry', selected
    return None, selected
