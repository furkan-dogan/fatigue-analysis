"""Volleyball workspace: intake, saved results and opt-in technical inspection."""
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


def sidebar(store):
    with st.sidebar:
        st.markdown('<div class="workspace-brand">VOLEYBOL</div>',unsafe_allow_html=True)
        st.caption('Performans analiz çalışma alanı')
        st.divider()
        if st.button('Yeni analiz',key='volleyball_new',width='stretch',icon=':material/add:'):
            st.session_state['volleyball_view']='upload'
        current=st.session_state.get('volleyball_review')
        if current and st.session_state.get('volleyball_view')!='result' and st.button('Sonuca dön',key='volleyball_result',width='stretch',icon=':material/analytics:'):
            st.session_state['volleyball_view']='result'
        st.divider()
        st.markdown('**Analiz geçmişi**')
        records=store.sessions('volleyball')
        if records:
            labels={r['id']: f"{len(records)-i}. {r['label'][:40]} · {r['created_at'][:16].replace('T',' ')}"
                    + (' · Tamamlanmamış' if not r['completed'] else '') for i,r in enumerate(records)}
            selected=st.selectbox('Kayıt seçin',list(labels),format_func=labels.get,key='volleyball_history')
            if st.button('Kaydı aç',key='volleyball_open',width='stretch'):
                try:
                    st.session_state['volleyball_review']=load_review(selected,store=store)
                    st.session_state['volleyball_view']='result'
                except (OSError,ValueError) as exc:
                    st.error(f'Kayıt açılamadı: {exc}')
        else:
            st.caption('İlk analiziniz burada görünecek.')
        st.divider()
        st.caption('Yerel çalışma alanı · Video ve sonuçlar bu cihazda saklanır.')


def render():
    store=AnalysisStore()
    sidebar(store)
    st.markdown('<div class="workspace-eyebrow">Kondisyoner çalışma alanı</div>',unsafe_allow_html=True)
    st.title('Video analizi')
    current=st.session_state.get('volleyball_review')
    if st.session_state.get('volleyball_view','upload')!='result' or current is None:
        st.caption('Tek video yükleyin. Hareketleri bulun, ölçümleri inceleyin.')
        upload_column, guide_column=st.columns([1.65,1],gap='large')
        with upload_column, st.container(border=True,key='upload_panel'):
            st.subheader('Yeni video')
            st.caption('MP4, MOV veya AVI · Tek video')
            render_uploads(store)
        with guide_column:
            st.subheader('Üç adımda inceleme')
            st.markdown('**01 · Videoyu yükleyin**')
            st.caption('Sporcunun hareketi ve ayakları kadrajda görünsün.')
            st.markdown('**02 · Otomatik analiz**')
            st.caption('Sistem hareketleri ve tekrar adaylarını bulur.')
            st.markdown('**03 · Sonuçları inceleyin**')
            st.caption('Hareketi videoda izleyin. Gerekirse ölçüm bilgisini tamamlayın.')
        st.caption('İlk analizde model hazırlanması zaman alabilir. Ölçümler deneysel; eksik çekim bilgisinde fiziksel değer üretilmez.')
        return
    result=current['result']; metadata=result['metadata']; review=result['review']
    name=review['athlete'] or result['video']['original_name']
    st.caption(name)
    if result.get('analysis',{}).get('mode')=='automatic':
        render_discovery(current,store)
    else:
        render_discovery(current,store)
        left,right=st.columns([1,1.15],gap='large')
        with left,st.container(border=True,key='video_panel'):
            st.subheader('Video incelemesi')
            video_player(current['source_path'])
        with right,st.container(border=True,key='measurement_panel'):
            st.subheader('Analiz durumu')
            if result.get('analysis'):
                render_results(current,store,allow_run=False)
            else:
                st.info('Video kayıtlı. Sonuçları görmek için otomatik analizi başlatın.')
    st.divider()
    if not st.toggle('Teknik incelemeyi aç',key='volleyball_technical_'+current['session_id']):
        return
    st.caption(f"{metadata['width']} × {metadata['height']} · {metadata['frame_count']} kare · Nominal FPS: {metadata['nominal_fps'] or 'Bilinmiyor'}")
    messages=quality_messages(review,metadata)
    if result.get('analysis'):
        messages=messages[1:]
    quality_panel([QualityNotice(message,'info') for message in messages])
    detail,_=st.columns([1,2])
    with detail:
        frame_inspector(current['source_path'],metadata['frame_count'],key='volleyball_frame_'+current['session_id'],
                        boxes=[review['athlete_box']],lines=[review['calibration']])
    render_run(current,store)
    try:
        changes=edit_review(current)
        if changes is not None:
            updated=save_revision(current,changes,store=store)
            st.session_state['volleyball_review']=updated
            st.session_state['volleyball_technical_'+updated['session_id']]=True
            st.rerun()
    except (OSError,ValueError) as exc:
        st.error(f'İnceleme kaydedilmedi: {exc}')
