"""Manual review form. Validation and interpretation live in the sport module."""
import pandas as pd
import streamlit as st
from src.sports.volleyball.review import TESTS


def reference_fields(label, value, metadata, key, *, distance=False):
    if not st.checkbox(label, value=value is not None, key=key + '_enabled'):
        return None
    value = value or {}
    frame = st.number_input('Referans karesi', 0, metadata['frame_count'] - 1,
                            value=int(value.get('frame', 0)), key=key + '_frame')
    points = {}
    for col, axis, bound in zip(st.columns(2), ('x', 'y'), (metadata['width'], metadata['height'])):
        with col:
            for suffix, default in (('1', 0), ('2', bound - 1)):
                name = axis + suffix
                points[name] = st.number_input(name.upper() + ' (piksel)', 0, bound - 1,
                                               value=int(value.get(name, default)), key=key + name)
    result = {'frame': int(frame), **points}
    if distance:
        result['distance_m'] = st.number_input('Bilinen mesafe (m)', min_value=0.001,
                                               value=float(value.get('distance_m', 1.0)), key=key + '_distance')
        result['plane'] = st.text_input('Referansın düzlemi / konumu', value=value.get('plane', ''),
                                        placeholder='Örn. parkur zemini, yandaki dikey referans', key=key + '_plane')
    return result


def edit_review(current):
    result = current['result']
    review, metadata = result['review'], result['metadata']
    key = 'volleyball_' + current['session_id']
    # Outside the form so repetition columns follow the selected protocol immediately.
    test = st.selectbox('Test türü', list(TESTS), index=list(TESTS).index(review['test']),
                        format_func=TESTS.get, key=key + '_test')
    with st.container():
        athlete = st.text_input('Sporcu adı veya kodu', value=review['athlete'])
        left, right = st.columns(2)
        with left:
            start = st.number_input('İnceleme başlangıç karesi', 0, metadata['frame_count'] - 1, value=review['start_frame'])
        with right:
            end = st.number_input('İnceleme bitiş karesi', 0, metadata['frame_count'] - 1, value=review['end_frame'])
        camera_options = {'unknown': 'Bilinmiyor', 'fixed': 'Sabit', 'moving': 'Hareketli / pan / zoom'}
        view_options = {'unknown': 'Bilinmiyor', 'side': 'Yandan', 'front': 'Önden', 'back': 'Arkadan', 'oblique': 'Çapraz'}
        camera = st.selectbox('Kamera', list(camera_options), index=list(camera_options).index(review['camera']), format_func=camera_options.get)
        view = st.selectbox('Çekim yönü', list(view_options), index=list(view_options).index(review['view']), format_func=view_options.get)
        feet = st.checkbox('Kalkış ve inişte ayaklar net görünüyor', value=review['feet_visible'])
        physical = st.checkbox('Kayıt hızı ve ağır çekim ilişkisini kontrol ettim', value=review['physical_time_confirmed'])
        options = {}
        st.caption('Fiziksel süre = video oynatma süresi × çarpan. Örn. 4 kat ağır çekim için 0,25; bilinmiyorsa zaman onayını işaretlemeyin.')
        options['time_scale'] = st.number_input('Video → fiziksel süre çarpanı', min_value=0.001,
                                                value=float(review.get('time_scale', 1.0)), key=key+'_time_scale')
        confirmations = {
            'single_take_confirmed': 'Seçili aralık tek kesintisiz çekim; kurgu geçişi yok',
            'cmj_confirmed': 'Yerinde çift ayak CMJ; yaklaşma ve bacak çekme yok',
            'contacts_confirmed': 'Kalkış / ilk temas karelerini kare kare kontrol ettim',
            'posture_confirmed': 'Kalkış ve ilk temasta benzer vücut duruşunu kontrol ettim',
            'camera_level_confirmed': 'Kamera eğik değil; görüntü dikeyini kontrol ettim',
            'motion_plane_confirmed': 'Mesafe referansı pelvisin hareket düzleminde ve aynı yükseklikte',
            'camera_perpendicular': 'Kamera hareket düzlemine dik; derinlik/perspektif değişimi yok',
        }
        active = ['single_take_confirmed'] + ({'cmj': ['cmj_confirmed', 'contacts_confirmed', 'posture_confirmed'],
                  'asymmetry': ['contacts_confirmed', 'camera_level_confirmed'],
                  'sprint': ['motion_plane_confirmed', 'camera_perpendicular']}.get(test, []))
        for name, label in confirmations.items():
            options[name] = st.checkbox(label, value=bool(review.get(name, False)) if test == review['test'] else False,
                                         key=key+'_'+test+'_'+name) if name in active else False
        st.caption('Bunlar kullanıcı beyanıdır; ölçüm doğruluğu garantisi değildir.')
        box = reference_fields('Sporcuyu referans karesinde kutuyla belirt', review['athlete_box'], metadata, key + '_box')
        st.caption('Sporcu kutusunu seçili analiz aralığının ilk karesinde belirtin; analiz bu seçimle takibi başlatır.')
        calibration = reference_fields('Mesafe referansı ekle', review['calibration'], metadata, key + '_cal', distance=True)
        st.caption('İki nokta ve bilinen mesafe kaydedilir; perspektif düzeltmesi henüz uygulanmaz.')
        columns = ['start_frame', 'end_frame'] if test == 'sprint' else ['start_frame', 'takeoff_frame', 'landing_frame', 'end_frame']
        if test == 'asymmetry':
            columns += ['left_landing_frame', 'right_landing_frame']
        labels = {'left_landing_frame': 'Sol ilk temas (isteğe bağlı)', 'right_landing_frame': 'Sağ ilk temas (isteğe bağlı)', 'start_frame': 'Başlangıç karesi', 'end_frame': 'Bitiş karesi',
                  'takeoff_frame': 'İki ayağın da havada olduğu ilk kare', 'landing_frame': 'İlk ayak teması karesi'}
        st.markdown('**Tekrarlar — manuel işaretleme**')
        st.caption('Kare inceleyicide kontrol edip satır ekleyin. Test türü değişirse eski tekrarlar aktarılmaz.')
        rows = review['repetitions'] if test == review['test'] else []
        table = st.data_editor(pd.DataFrame(rows, columns=columns), num_rows='dynamic', hide_index=True,
                               column_config={c: st.column_config.NumberColumn(labels[c], min_value=0, max_value=metadata['frame_count'] - 1, step=1, required=c not in ('left_landing_frame', 'right_landing_frame')) for c in columns},
                               key=key + '_repeats_' + test)
        notes = st.text_area('Çekim / zamanlama notları', value=review['notes'])
        submitted = st.button('İncelemeyi yeni revizyon olarak kaydet', key=key + '_save')
    if not submitted:
        return None
    repetitions = []
    for row in table.to_dict('records'):
        converted = {}
        for name, value in row.items():
            if pd.isna(value) and name in ('left_landing_frame', 'right_landing_frame'):
                continue
            if pd.isna(value) or float(value) != int(value):
                raise ValueError('Tekrarların tüm kare alanlarını tam sayı olarak doldurun.')
            converted[name] = int(value)
        repetitions.append(converted)
    return dict(test=test, athlete=athlete, start_frame=int(start), end_frame=int(end),
                athlete_box=box, calibration=calibration, repetitions=repetitions,
                camera=camera, view=view, feet_visible=feet, physical_time_confirmed=physical, notes=notes, **options)
