"""Compose experimental analysis results using shared display components."""
import pandas as pd
import streamlit as st
from ui.components.metrics import metric_cards
from ui.components.models import MetricCard
from src.sports.volleyball.service import analyze_review, save_revision


def render_run(current, store):
    key = 'volleyball_analysis_' + current['session_id']
    st.caption('Deneysel ölçümler: bağımsız videolarla doğrulama henüz yapılmadı. Kaydedilmiş inceleme ayarları kullanılır.')
    if st.button('Kaydedilmiş ayarlarla analiz et', key=key+'_run'):
        progress = st.progress(0, text='Model hazırlanıyor…')
        try:
            updated = analyze_review(current, lambda i, n: progress.progress(i/n, text=f'Kare {i}/{n}'), store=store)
            st.session_state['volleyball_review'] = updated
            st.session_state['volleyball_technical_' + updated['session_id']] = True
            st.rerun()
        except Exception as exc:
            st.error(f'Analiz tamamlanamadı: {exc}')
        finally:
            progress.empty()


def render_results(current, store, *, allow_run=True):
    key = 'volleyball_analysis_' + current['session_id']
    if allow_run:
        render_run(current, store)
    else:
        st.caption('Deneysel sonuçlar; bağımsız ölçüm doğrulaması henüz yapılmadı.')
    analysis = current['result'].get('analysis')
    if not analysis:
        st.info('Bu kayıt henüz analiz edilmedi. Ölçüm için inceleme ve protokol ayarlarını kaydedin.')
        return
    for message in analysis['warnings']:
        st.warning(message)
    groups = [analysis.get('metrics', [])] + [event['metrics'] for event in analysis['events']]
    for index, values in enumerate(groups):
        if not values:
            continue
        if index:
            st.markdown(f'**Tekrar {index}**')
        metric_cards([MetricCard(v['label'], None if v['value'] is None else round(v['value'], 3), v['unit']) for v in values])
        for reason in dict.fromkeys(v['reason'] for v in values if v['reason']):
            st.caption(reason)
    st.caption('Sıçrama alt/üst sınırları yalnızca kare aralığı belirsizliğidir; model hatasını kapsamaz. Görsel asimetri kuvvet farkı değildir.')
    if analysis['speed_series']:
        frame = pd.DataFrame(analysis['speed_series']).rename(columns={'time_seconds':'Fiziksel zaman (s)', 'speed_m_s':'Hız (m/s)'})
        st.line_chart(frame.set_index('Fiziksel zaman (s)')[['Hız (m/s)']])
        st.caption('Pelvisin görüntü düzlemindeki konumundan 0,20 s pencereli hız; kütle merkezi veya tek kare hızı değildir.')
    if analysis['candidates']:
        st.markdown('**Otomatik tekrar adayları — temasları kontrol edin**')
        labels = {'start_frame':'Başlangıç', 'takeoff_frame':'Kalkış adayı', 'landing_frame':'İniş adayı', 'end_frame':'Bitiş'}
        st.dataframe([{labels[k]: v for k, v in r.items()} for r in analysis['candidates']], hide_index=True)
        if st.button('Adayları yeni inceleme revizyonuna aktar', key=key+'_adopt'):
            review = {**current['result']['review'], 'repetitions': analysis['candidates'], 'contacts_confirmed': False, 'posture_confirmed': False}
            try:
                st.session_state['volleyball_review'] = save_revision(current, review, store=store)
                st.rerun()
            except (ValueError, OSError) as exc:
                st.error(str(exc))
