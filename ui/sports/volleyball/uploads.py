"""Batch intake UI; creates reviews, not an analysis queue."""
import hashlib
import pandas as pd
import streamlit as st
from ui.components.upload import video_uploader
from src.sports.volleyball.service import create_review, load_review


def render_uploads(store):
    uploads = video_uploader('Voleybol videoları', key='volleyball_upload', multiple=True,
                             help='Birden fazla video seçebilirsiniz. Sporcu bilgisi isteğe bağlıdır.') or []
    # Identity includes content: equal filenames do not share assignments.
    identities = [hashlib.sha256(u.getvalue()).hexdigest() + ':' + u.name for u in uploads]
    selection = hashlib.sha256(repr(identities).encode()).hexdigest()
    rows = []
    group = notes = ''
    if uploads:
        st.caption(f'{len(uploads)} video seçildi. Sporcu adını biliyorsanız yazın; boş bırakabilirsiniz.')
        rows = st.data_editor(
            pd.DataFrame([{'Dosya': u.name, 'Sporcu': ''} for u in uploads]),
            disabled=['Dosya'], hide_index=True, key='volleyball_assign_' + selection,
            column_config={'Sporcu': st.column_config.TextColumn('Sporcu adı / kodu (isteğe bağlı)')},
        ).fillna('').to_dict('records')
        with st.expander('Ortak çekim bilgisi — isteğe bağlı'):
            group = st.text_input('Çekim grubu adı', key='volleyball_group_' + selection)
            notes = st.text_area('Bu videolar için ortak not', key='volleyball_notes_' + selection)
            st.caption('Grup ve not yalnızca kayıt bilgisidir; kamera, zaman veya mesafe ayarlarını otomatik onaylamaz.')
    st.caption('Bu aşamada videolar kaydedilir. Otomatik analiz kuyruğu sonraki adımda eklenecek.')
    if st.button('Videoları kaydet', disabled=not uploads, key='volleyball_create', type='primary'):
        saved = st.session_state.setdefault('volleyball_intake_saved', {})
        outcomes = []
        progress = st.progress(0, text='Videolar hazırlanıyor…')
        try:
            for i, (upload, identity, row) in enumerate(zip(uploads, identities, rows)):
                # Changed assignment is a new intake; repeated clicks reuse successful saves.
                token = (identity, row['Sporcu'].strip(), group.strip(), notes.strip(), str(store.root))
                try:
                    if token in saved:
                        current = load_review(saved[token], store=store)
                    else:
                        current = create_review(upload.getvalue(), upload.name, store=store,
                                                athlete=row['Sporcu'], capture_group=group, capture_notes=notes)
                        saved[token] = current['session_id']
                    outcomes.append({'Dosya': upload.name, 'Durum': 'Kaydedildi', 'session_id': current['session_id']})
                except (OSError, ValueError, RuntimeError) as exc:
                    outcomes.append({'Dosya': upload.name, 'Durum': f'Kaydedilemedi: {exc}', 'session_id': None})
                progress.progress((i + 1) / len(uploads), text=f'{i + 1}/{len(uploads)} video işlendi')
        finally:
            progress.empty()
        st.session_state['volleyball_intake_results'] = outcomes
    outcomes = st.session_state.get('volleyball_intake_results', [])
    if outcomes:
        st.dataframe([{k: row[k] for k in ('Dosya', 'Durum')} for row in outcomes], hide_index=True)
        successful = {row['session_id']: row['Dosya'] for row in outcomes if row['session_id']}
        if successful:
            selected = st.selectbox('Eklenen video', list(successful), format_func=successful.get, key='volleyball_intake_open')
            if st.button('Kaydı görüntüle', key='volleyball_intake_view'):
                try:
                    st.session_state['volleyball_review'] = load_review(selected, store=store)
                    st.rerun()
                except (OSError, ValueError) as exc:
                    st.error(f'Kayıt açılamadı: {exc}')
