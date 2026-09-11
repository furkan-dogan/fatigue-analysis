"""Compose the volleyball upload, review and revision workflow."""
import streamlit as st
from src.adapters.analysis_store import AnalysisStore
from src.sports.volleyball.service import create_review, load_review, save_revision
from src.sports.volleyball.review import quality_messages, frame_time
from ui.components.upload import video_uploader
from ui.components.video_player import video_player
from ui.components.frame_inspector import frame_inspector
from ui.components.quality import quality_panel
from ui.components.models import QualityNotice, TimelineEvent
from ui.components.timeline import event_timeline
from ui.sports.volleyball.editor import edit_review
from ui.sports.volleyball.results import render_results


def render():
    st.title('Voleybol — Video Analizi')
    st.caption('Video yükleyin, çekimi inceleyin ve tekrarları işaretleyin. Deneysel analizler için çekim ve protokol ayarlarını kaydedin.')
    store = AnalysisStore()
    with st.expander('Kayıtlı incelemeler'):
        records = store.sessions('volleyball')
        if records:
            labels = {r['id']: f"{r['label']} · {r['created_at'][:19]} · {'Kayıt' if r['completed'] else 'Tamamlanmamış'} · {r['id'][:8]}" for r in records}
            selected = st.selectbox('İnceleme kaydı', list(labels), format_func=labels.get, key='volleyball_history')
            if st.button('İncelemeyi aç', key='volleyball_open'):
                try:
                    st.session_state['volleyball_review'] = load_review(selected, store=store)
                except (OSError, ValueError) as exc:
                    st.error(f'Kayıt açılamadı: {exc}')
        else:
            st.caption('Henüz kayıtlı inceleme yok.')
    upload = video_uploader('Voleybol videosu', key='volleyball_upload')
    if st.button('Videoyu kaydet ve incele', disabled=upload is None, key='volleyball_create'):
        try:
            with st.spinner('Video okunuyor…'):
                st.session_state['volleyball_review'] = create_review(upload.getvalue(), upload.name, store=store)
            st.rerun()
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(f'Video hazırlanamadı: {exc}')
    current = st.session_state.get('volleyball_review')
    if current is None:
        return
    result = current['result']
    metadata, review = result['metadata'], result['review']
    st.caption(f"{metadata['width']} × {metadata['height']} piksel · {metadata['frame_count']} kare · Nominal FPS: {metadata['nominal_fps'] or 'Bilinmiyor'}")
    messages = quality_messages(review, metadata)
    if result.get('analysis'):
        messages = messages[1:]
    quality_panel([QualityNotice(message, 'info') for message in messages])
    left, right = st.columns(2)
    with left:
        video_player(current['source_path'])
    with right:
        frame_inspector(current['source_path'], metadata['frame_count'], key='volleyball_frame_' + current['session_id'],
                        boxes=[review['athlete_box']], lines=[review['calibration']])
    if review['repetitions']:
        st.markdown('**Kaydedilmiş tekrarlar**')
        labels = {'start_frame': 'Başlangıç karesi', 'end_frame': 'Bitiş karesi',
                  'takeoff_frame': 'İlk havada kare', 'landing_frame': 'İlk temas karesi',
                  'left_landing_frame': 'Sol temas', 'right_landing_frame': 'Sağ temas'}
        st.dataframe([{labels[k]: v for k, v in row.items()} for row in review['repetitions']], hide_index=True)
        if frame_time(0, metadata) is not None:
            events = [TimelineEvent(str(i), f'Tekrar {i+1}', frame_time(r['start_frame'], metadata), frame_time(r['end_frame'], metadata)) for i, r in enumerate(review['repetitions'])]
            selected = event_timeline(events, key='volleyball_timeline_' + current['session_id'])
            st.caption('Zaman çizelgesi video oynatma zamanıdır; gerçek fiziksel süre doğrulanmış değildir.')
            if selected:
                video_player(current['source_path'], start_time=selected.start_seconds)
    render_results(current, store)
    try:
        changes = edit_review(current)
        if changes is not None:
            st.session_state['volleyball_review'] = save_revision(current, changes, store=store)
            st.rerun()
    except (OSError, ValueError) as exc:
        st.error(f'İnceleme kaydedilmedi: {exc}')
