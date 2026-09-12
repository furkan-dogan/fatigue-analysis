"""Compose the volleyball upload, review and revision workflow."""
import streamlit as st
from src.adapters.analysis_store import AnalysisStore
from src.sports.volleyball.service import load_review, save_revision
from src.sports.volleyball.review import quality_messages
from ui.sports.volleyball.uploads import render_uploads
from ui.sports.volleyball.discovery import render_discovery
from ui.components.video_player import video_player
from ui.components.frame_inspector import frame_inspector
from ui.components.quality import quality_panel
from ui.components.models import QualityNotice
from ui.sports.volleyball.editor import edit_review
from ui.sports.volleyball.results import render_results, render_run


def render():
    st.title('Voleybol — Video Analizi')
    st.caption('Videonuzu yükleyin. Kayıtlı videolarınıza ve analizlerinize buradan ulaşın.')
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
    render_uploads(store)
    current = st.session_state.get('volleyball_review')
    if current is None:
        return
    result = current['result']
    metadata, review = result['metadata'], result['review']
    st.caption(f"{metadata['width']} × {metadata['height']} piksel · {metadata['frame_count']} kare · Nominal FPS: {metadata['nominal_fps'] or 'Bilinmiyor'}")
    st.markdown('**Seçili kayıt**')
    st.caption(review['athlete'] or 'Sporcu henüz eşleştirilmedi')
    if review.get('capture_group'):
        st.caption('Çekim grubu: ' + review['capture_group'])
    start_time = render_discovery(current, store)
    preview, _ = st.columns([1, 2])
    with preview:
        video_player(store.path(result['preview_path']) if result.get('preview_path') else current['source_path'], start_time=start_time)
    if result.get('analysis', {}).get('mode') == 'automatic':
        st.caption('İşaretli önizleme sessizdir; kaynak video korunur.' if result.get('preview_path') else 'Kaynak video gösteriliyor.')
    elif result.get('analysis'):
        render_results(current, store, allow_run=False)
    else:
        st.info('Video kaydedildi; otomatik analiz henüz çalıştırılmadı.')
    if not st.toggle('Teknik incelemeyi aç', key='volleyball_technical_' + current['session_id']):
        return
    messages = quality_messages(review, metadata)
    if result.get('analysis'):
        messages = messages[1:]
    quality_panel([QualityNotice(message, 'info') for message in messages])
    detail, _ = st.columns([1, 2])
    with detail:
        frame_inspector(current['source_path'], metadata['frame_count'], key='volleyball_frame_' + current['session_id'],
                        boxes=[review['athlete_box']], lines=[review['calibration']])
    if not result.get('analysis'):
        render_results(current, store)
    else:
        render_run(current, store)
    try:
        changes = edit_review(current)
        if changes is not None:
            updated = save_revision(current, changes, store=store)
            st.session_state['volleyball_review'] = updated
            st.session_state['volleyball_technical_' + updated['session_id']] = True
            st.rerun()
    except (OSError, ValueError) as exc:
        st.error(f'İnceleme kaydedilmedi: {exc}')
