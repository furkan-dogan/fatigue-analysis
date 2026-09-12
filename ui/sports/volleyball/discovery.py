"""Automatic discovery controls and movement navigation."""
import cv2
import streamlit as st
from src.sports.volleyball.service import analyze_review
from src.adapters.video_review import read_frame
from src.adapters.pose_preview import overlay
from ui.components.movement_list import movement_list


def run_discovery(current, store, choices=None):
    progress = st.progress(0, text='Model hazırlanıyor…')
    try:
        updated = analyze_review(current, lambda i,n: progress.progress(i/n, text=f'Video taranıyor: {i}/{n} kare'),
                                 store=store, automatic=True, athlete_choices=choices)
        st.session_state['volleyball_review'] = updated
        return updated
    finally:
        progress.empty()


def render_discovery(current, store):
    result = current['result']
    analysis = result.get('analysis', {})
    key = current['session_id']
    if analysis.get('mode') != 'automatic':
        if st.button('Hareketleri otomatik bul', key='volleyball_auto_' + key):
            try:
                run_discovery(current, store)
                st.rerun()
            except (OSError, ValueError, RuntimeError) as exc:
                st.error(f'Tarama tamamlanamadı: {exc}')
        return 0.
    st.markdown(f"**{len(analysis['events'])} hareket adayı · {len(analysis['segments'])} çekim bölümü**")
    st.caption('Otomatik görsel tespit; cm ve hız ölçümü değildir. Sınırlar ve hareket türleri kontrol gerektirebilir.')
    for message in analysis['warnings'][1:]:
        st.info(message)
    for request in analysis['selections_needed']:
        frame = request['frame']
        with st.expander(f'Sporcuyu seç — {frame}. kare'):
            picture = overlay(read_frame(current['source_path'], frame), request['people'])
            st.image(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB), width=320)
            selected = st.selectbox('Hangi sporcu?', range(len(request['people'])),
                                    format_func=lambda i: f'Sporcu {i+1}', key=f'{key}_person_{frame}')
            if st.button('Bu sporcuyla tekrar tara', key=f'{key}_choose_{frame}'):
                try:
                    run_discovery(current, store, {**result.get('athlete_choices', {}), str(frame): selected})
                    st.rerun()
                except (OSError, ValueError, RuntimeError) as exc:
                    st.error(f'Tarama tamamlanamadı: {exc}')
    event = movement_list(analysis['events'], key=key+'_movement')
    if event is None:
        return 0.
    times = result['metadata'].get('timestamps', [])
    if 'takeoff_frame' in event:
        st.caption(f"Kalkış adayı: {event['takeoff_frame']} · Tepe: {event['peak_frame']} · İniş adayı: {event['landing_frame']}")
    if len(times) > event['start_frame'] and times[0] is not None and times[event['start_frame']] is not None:
        return max(0., times[event['start_frame']]-times[0])
    st.caption('Kaynak zamanları eksik; hareketler kare numarasıyla saklandı.')
    return 0.
