"""Selected-event metrics and optional capture facts; no measurement calculations."""
import streamlit as st
from ui.components.metrics import metric_cards
from ui.components.models import MetricCard


def render_event_metrics(event):
    metrics=event.get('metrics',[])
    if not metrics:
        return
    available=[m for m in metrics if m['value'] is not None]
    if available:
        metric_cards([MetricCard(m['label'],round(m['value'],2),m['unit']) for m in available])
    missing=[m for m in metrics if m['value'] is None]
    if missing:
        with st.expander('Diğer ölçümler neden hesaplanmadı?'):
            for m in missing:
                st.caption(f"{m['label']}: {m['reason']}")
    st.caption('Yüzdeler gövde uzunluğuna göre görüntü oranıdır; gelişim yüzdesi değildir.')
    st.caption('Görüntüdeki pelvis hareketi gerçek kütle merkezi/erişim yüksekliği değildir. Yatay sapma tek başına bacak kuvvet farkını göstermez.')
    if event.get('speed_series'):
        import pandas as pd
        st.line_chart(pd.DataFrame(event['speed_series']).set_index('time_seconds')[['speed_m_s']])
        st.caption('Hız: 0,20 saniyelik pencere, m/s. Zaman: seçili hareketin fiziksel saniyesi.')
    bounds=event.get('flight_sampling_bounds_cm')
    if bounds and all(v is not None for v in bounds):
        st.caption(f'Uçuş süresi kare örnekleme sınırı: {bounds[0]:.1f}–{bounds[1]:.1f} cm. Model/duruş hatasını kapsamaz.')


def edit_measurement_inputs(current,event):
    result=current['result']; inputs=result.get('measurement_inputs',{})
    segment=result['analysis']['segments'][event['segment']]
    segment_key=str(segment['start_frame'])
    context=inputs.get('segments',{}).get(segment_key,{})
    key=current['session_id']+'_facts_'+segment_key
    with st.expander('Ölçüm bilgisi ekle / düzelt'):
        st.caption('Bu bilgiler yalnızca seçili çekim bölümüne uygulanır. Boş bilgiler görüntü ölçümlerini engellemez.')
        camera_options={'unknown':'Bilinmiyor','fixed_perpendicular':'Sabit ve hareket düzlemine dik','moving':'Hareketli / zoom var'}
        view_options={'unknown':'Bilinmiyor','side':'Yandan','front':'Önden','back':'Arkadan','oblique':'Çapraz'}
        camera=st.selectbox('Kamera',list(camera_options),index=list(camera_options).index(context.get('camera','unknown')),
                            format_func=camera_options.get,key=key+'_camera')
        view=st.selectbox('Çekim yönü',list(view_options),index=list(view_options).index(context.get('view','unknown')),
                          format_func=view_options.get,key=key+'_view')
        factor=context.get('time_scale')
        mode=st.selectbox('Videonun oynatma hızı',['Bilinmiyor','Gerçek zaman','Hızı değiştirilmiş'],
                          index=0 if factor is None else 1 if factor==1 else 2,key=key+'_time')
        factor=None if mode=='Bilinmiyor' else 1. if mode=='Gerçek zaman' else st.number_input(
            'Oynatma süresi → gerçek süre çarpanı',min_value=.001,value=float(factor or .25),key=key+'_factor',
            help='Örneğin dört kat yavaşlatılmış videoda 0,25. Kaynak FPS tek başına bu bilgiyi vermez.')
        calibration=context.get('calibration')
        with st.expander('Mesafe referansı — yalnızca santimetre / hız için'):
            st.caption('Sıçramada dikey, hızda ilerleme yönünde bilinen bir uzunluk gerekir. Referans pelvisin hareket düzleminde olmalı; zemin çizgisi otomatik olarak uygun sayılmaz.')
            enabled=st.checkbox('Bu bölüm için referans ekle',value=bool(calibration),key=key+'_cal')
            if enabled:
                calibration=calibration or {}
                values={}
                for axis,bound in [('x',result['metadata']['width']),('y',result['metadata']['height'])]:
                    for suffix,default in [('1',0),('2',bound-1)]:
                        name=axis+suffix
                        values[name]=st.number_input(name.upper()+' (kaynak piksel)',min_value=0,max_value=bound-1,
                                                     value=int(calibration.get(name,default)),key=key+name)
                values['distance_m']=st.number_input('Bilinen mesafe (metre)',min_value=.001,
                                                     value=float(calibration.get('distance_m',1)),key=key+'_distance')
                values['plane_confirmed']=st.checkbox('Referans pelvis hareketiyle aynı düzlemde',value=bool(calibration.get('plane_confirmed')),key=key+'_plane')
                calibration=values
            else:
                calibration=None
        if st.button('Ölçümleri güncelle',key=key+'_save'):
            return {**inputs,'segments':{**inputs.get('segments',{}),segment_key:dict(
                camera=camera,view=view,time_scale=factor,calibration=calibration)}}
    return None
