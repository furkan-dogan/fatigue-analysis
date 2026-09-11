"""Thin composition of the taekwondo analysis components."""
import streamlit as st
from ui.sports.taekwondo.session import load_session
from ui.sports.taekwondo.views import summary, signals, events, movement, statistics
from ui.sports.taekwondo.report import render_report


def render():
    st.title('Taekwondo — Video Analizi')
    st.caption('Antrenman öncesi ve sonrası hareketleri görüntü üzerinden inceleyin.')
    session = load_session()
    if session is None:
        return
    pre, post, pre_df, post_df = session
    tabs = st.tabs(['Özet', 'Açı ve Hız', 'Tekmeler', 'Faz ve Asimetri', 'İstatistik', 'Rapor'])
    with tabs[0]:
        summary(pre, post)
    with tabs[1]:
        signals(pre_df, post_df)
    with tabs[2]:
        events(pre, post)
    with tabs[3]:
        movement(pre, post)
    with tabs[4]:
        statistics(pre, post)
    with tabs[5]:
        render_report(pre, post, pre_df, post_df)
