"""Automatic discovery controls and movement navigation."""
import cv2
import streamlit as st
from src.sports.volleyball.service import analyze_review
from src.adapters.video_review import read_frame
from src.adapters.pose_preview import overlay
from ui.components.movement_list import movement_list
from ui.components.video_player import video_player
from ui.sports.volleyball.measurements import render_event_metrics, edit_measurement_inputs


def run_discovery(current, store, choices=None, measurement_inputs=None):
    progress = st.progress(0, text='Model hazırlanıyor…')
    try:
        updated = analyze_review(current, lambda i,n: progress.progress(i/n, text=f'Video taranıyor: {i}/{n} kare'),
                                 store=store, automatic=True, athlete_choices=choices, measurement_inputs=measurement_inputs)
        old_events=current['result'].get('analysis',{}).get('events',[])
        old_index=st.session_state.get(current['session_id']+'_movement',0)
        if old_events and 0 <= old_index < len(old_events):
            selected=old_events[old_index]
            for i,event in enumerate(updated['result']['analysis']['events']):
                if all(event.get(k)==selected.get(k) for k in ('kind','start_frame','end_frame')):
                    st.session_state[updated['session_id']+'_movement']=i
                    break
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
    st.caption(f"{len(analysis['events'])} hareket adayı bulundu · {len(analysis['segments'])} çekim bölümü")
    if analysis['warnings'][1:]:
        with st.expander('Kontrol gerektiren noktalar'):
            for message in analysis['warnings'][1:]:
                st.write(message)
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
    video_column, results_column = st.columns([1,1.15],gap='large')
    event = None
    with results_column, st.container(border=True,key='measurement_panel'):
        st.subheader('Hareket ve ölçümler')
        st.caption('Deneysel sonuçlar · Fiziksel doğruluk henüz doğrulanmadı')
        event = movement_list(analysis['events'], key=key+'_movement')
        if event is not None:
            render_event_metrics(event)
        else:
            st.info('Bu videoda gösterilebilecek bir hareket adayı bulunamadı.')
    times = result['metadata'].get('timestamps', [])
    start_time=0.
    if event and len(times)>event['start_frame'] and times[0] is not None and times[event['start_frame']] is not None:
        start_time=max(0.,times[event['start_frame']]-times[0])
    with video_column, st.container(border=True,key='video_panel'):
        st.subheader('Video incelemesi')
        st.caption('Listeden bir hareket seçin; video ilgili başlangıca gider.')
        video_player(store.path(result['preview_path']) if result.get('preview_path') else current['source_path'],start_time=start_time)
        st.caption('İşaretli önizleme · Sessiz' if result.get('preview_path') else 'Kaynak video')
    if event is None:
        return start_time
    changes=edit_measurement_inputs(current,event)
    if changes is not None:
        try:
            run_discovery(current,store,measurement_inputs=changes)
            st.rerun()
        except (OSError,ValueError,RuntimeError) as exc:
            st.error(f'Ölçümler güncellenemedi: {exc}')
    if not analysis.get('measurement_version'):
        if st.button('Bu kaydın ölçümlerini hesapla',key=key+'_measure'):
            try:
                run_discovery(current,store)
                st.rerun()
            except (OSError,ValueError,RuntimeError) as exc:
                st.error(f'Ölçümler hesaplanamadı: {exc}')
    return start_time
