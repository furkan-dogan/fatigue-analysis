"""Per-event experimental metrics with explicit capture evidence and comparison context."""
from datetime import date
import math
import numpy as np
from src.core.pose import point
from src.sports.volleyball.signals import midpoint, torso, sole, physical_times
from src.sports.volleyball.measurements import jump, sprint

VERSION = 'volleyball-auto-metrics-1'
DEFINITIONS = {
    'image_pelvis_rise_pct': ('Görüntüde pelvis yükselmesi', '% gövde', 'image_pelvis_rise'),
    'image_lateral_drift_pct': ('Görüntüde yatay sapma', '% gövde', 'image_pelvis_drift'),
    'image_trunk_lean_deg': ('Görüntüde en büyük gövde eğimi', '°', 'image_trunk_axis'),
    'pelvis_rise_cm': ('Kalibre pelvis yükselmesi', 'cm', 'vertical_image_scale'),
    'jump_height_cm': ('Yerinde sıçrama yüksekliği (tahmin)', 'cm', 'estimated_flight_contacts'),
    'peak_speed_m_s': ('Pencere ortalamalı en yüksek hız', 'm/s', 'local_linear_fit_0.20s'),
}


def value(key, number=None, reason=None):
    label,unit,method = DEFINITIONS[key]
    return dict(key=key,label=label,unit=unit,method=f'{VERSION}:{method}',value=number,
                quality='unvalidated' if number is not None else 'rejected',reason=reason)


def validate_inputs(inputs, metadata, segments):
    if not isinstance(inputs,dict):
        raise ValueError('Ölçüm bilgileri geçersiz.')
    for name in ('athlete_code','setup_label','recorded_on','session_phase'):
        if name in inputs and not isinstance(inputs[name],str):
            raise ValueError('Kayıt bilgisi metin olmalı.')
    if inputs.get('recorded_on'):
        date.fromisoformat(inputs['recorded_on'])
    if inputs.get('session_phase','unknown') not in ('unknown','before','after','during'):
        raise ValueError('Antrenman evresi geçersiz.')
    known = {str(s['start_frame']) for s in segments}
    for key, context in inputs.get('segments',{}).items():
        if key not in known:
            raise ValueError('Çekim bilgisi video bölümüyle eşleşmiyor.')
        if context.get('camera','unknown') not in ('unknown','fixed_perpendicular','moving'):
            raise ValueError('Kamera bilgisi geçersiz.')
        if context.get('view','unknown') not in ('unknown','side','front','back','oblique'):
            raise ValueError('Çekim yönü geçersiz.')
        factor=context.get('time_scale')
        if factor is not None and (isinstance(factor,bool) or not isinstance(factor,(int,float)) or not math.isfinite(factor) or factor<=0):
            raise ValueError('Zaman çarpanı pozitif olmalı.')
        calibration=context.get('calibration')
        if calibration:
            for axis,bound in [('x',metadata['width']),('y',metadata['height'])]:
                for end in ('1','2'):
                    coordinate=calibration[axis+end]
                    if not math.isfinite(coordinate) or not 0<=coordinate<bound:
                        raise ValueError('Referans noktası görüntü dışında.')
            if not math.isfinite(calibration['distance_m']) or calibration['distance_m']<=0:
                raise ValueError('Referans mesafesi pozitif olmalı.')
            if not isinstance(calibration.get('plane_confirmed',False),bool):
                raise ValueError('Referans düzlemi onayı geçersiz.')


def refine_contacts(samples, event, fps):
    """Refine clearance candidates against nearby observed foot baselines; no manual approval implied."""
    frames={s.frame:i for i,s in enumerate(samples)}
    a,b=frames[event['takeoff_frame']],frames[event['landing_frame']]
    radius=max(3,round((fps or 30)*.2))
    left=max(0,a-radius); right=min(len(samples)-1,b+radius)
    scale=torso(samples[a])
    if scale is None or scale<10 or a<=left or b>=right:
        return None
    feet=[]
    for sample in samples[left:right+1]:
        row=[sole(sample,side) for side in ('left','right')]
        if any(p is None for p in row):
            return None
        feet.append(row)
    feet=np.array(feet)
    # Use independently observed before/after references and reject a sloping/moving apparent floor.
    ground_before=np.median(feet[:min(3,a-left)],axis=0)
    ground_after=np.median(feet[-3:],axis=0)
    if np.max(np.abs(ground_before-ground_after)) > .08*scale:
        return None
    clearance=np.min((ground_before+ground_after)/2-feet,axis=1)
    apex=frames[event['peak_frame']]-left
    if not 0<=apex<len(feet) or clearance[apex] <= .1*scale:
        return None
    first=apex
    while first>0 and clearance[first-1]>.025*scale: first-=1
    last=apex
    while last+1<len(feet) and clearance[last+1]>.025*scale: last+=1
    if first==0 or last+1==len(feet):
        return None
    return dict(event,takeoff_frame=samples[left+first].frame,landing_frame=samples[left+last+1].frame)


def measure_event(samples, event, context, metadata):
    is_jump='takeoff_frame' in event
    keys = ['image_pelvis_rise_pct','image_lateral_drift_pct','image_trunk_lean_deg','pelvis_rise_cm','jump_height_cm'] if is_jump else ['peak_speed_m_s']
    base=dict(metrics=[],speed_series=[],measurement_protocol=event['kind'])
    if not samples or any(s.issue for s in samples):
        return dict(base,metrics=[value(k,reason='Hareket aralığında takip boşluğu var.') for k in keys])
    camera=context.get('camera','unknown')
    view=context.get('view','unknown')
    factor=context.get('time_scale')
    times=physical_times(samples,{'physical_time_confirmed':factor is not None,'time_scale':factor or 1})
    valid_times=(times is not None and metadata.get('timestamps_monotonic') and metadata.get('timestamp_count_matches_frames')
                 and len(times)>4 and np.max(np.diff(times))<=.05)
    if not is_jump:
        reason=None
        if camera!='fixed_perpendicular' or view!='side':
            reason='Hız için sabit, hareket düzlemine dik yandan çekim bilgisi gerekli.'
        elif not valid_times:
            reason='Hız için gerçek zaman / ağır çekim bilgisi gerekli.'
        if reason:
            return dict(base,metrics=[value('peak_speed_m_s',reason=reason)])
        calibration=context.get('calibration')
        results,series=sprint(samples,times,{'calibration':calibration,
                             'motion_plane_confirmed':bool(calibration and calibration.get('plane_confirmed')),
                             'camera_perpendicular':camera=='fixed_perpendicular'})
        converted=[value('peak_speed_m_s',m['value'],m['reason']) for m in results]
        return dict(base,metrics=converted,speed_series=series)
    # Use the detected flight window, excluding the approach run.
    window=[s for s in samples if event['takeoff_frame']-1 <= s.frame <= event['landing_frame']]
    hips=[midpoint(s,'hip') for s in window]
    scales=[torso(s) for s in window]
    if not window or any(h is None for h in hips) or any(s is None or s<10 for s in scales):
        return dict(base,metrics=[value(k,reason='Pelvis/gövde noktaları yetersiz.') for k in keys])
    base['measurement_window'] = dict(start_frame=window[0].frame,end_frame=window[-1].frame,
                                      reference_frame=window[0].frame,reference='detected_flight_window_start')
    hips=np.array(hips); scale=float(np.median(scales))
    if len(hips)>1 and np.max(np.linalg.norm(np.diff(hips,axis=0),axis=1))>.5*scale:
        return dict(base,metrics=[value(k,reason='Nokta konumunda ani sıçrama var.') for k in keys])
    rise=float(max(0.,hips[0,1]-np.min(hips[:,1])))
    lean=[]
    for sample in window:
        shoulder=midpoint(sample,'shoulder'); hip=midpoint(sample,'hip')
        if shoulder is None: break
        lean.append(abs(math.degrees(math.atan2(shoulder[0]-hip[0],hip[1]-shoulder[1]))))
    results=[value('image_pelvis_rise_pct',rise/scale*100),
             value('image_lateral_drift_pct',float(np.max(np.abs(hips[:,0]-hips[0,0]))/scale*100)),
             value('image_trunk_lean_deg',max(lean) if len(lean)==len(window) else None,
                   None if len(lean)==len(window) else 'Omuz noktaları eksik.')]
    calibration=context.get('calibration')
    reason='Santimetre için aynı hareket düzleminde dikey mesafe referansı gerekli.'
    cm=None
    if camera=='fixed_perpendicular' and view in ('side','front','back') and calibration and calibration.get('plane_confirmed'):
        dx=calibration['x2']-calibration['x1']; dy=calibration['y2']-calibration['y1']
        if abs(dy)>=20 and abs(dx)<=abs(dy)*.1:
            # Do not extrapolate beyond the calibrated vertical range.
            low,high=sorted([calibration['y1'],calibration['y2']])
            if low<=np.min(hips[:,1]) and np.max(hips[:,1])<=high:
                cm=rise/abs(dy)*calibration['distance_m']*100;reason=None
            else: reason='Pelvis yolu dikey referans aralığının dışında.'
    results.append(value('pelvis_rise_cm',cm,reason))
    if event['kind']!='stationary_jump':
        results.append(value('jump_height_cm',reason='Yaklaşmalı sıçrama CMJ değildir; kalibre pelvis yükselmesi ayrı ölçümdür.'))
    elif camera!='fixed_perpendicular' or view!='side':
        results.append(value('jump_height_cm',reason='Yerinde sıçrama tahmini için sabit ve dik yandan çekim bilgisi gerekli.'))
    elif not valid_times:
        results.append(value('jump_height_cm',reason='Yerinde sıçrama tahmini için gerçek zaman / ağır çekim bilgisi gerekli.'))
    else:
        contacts=refine_contacts(samples,event,metadata.get('nominal_fps'))
        if contacts is None:
            results.append(value('jump_height_cm',reason='Temas çevresindeki ayak referansı yeterli değil.'))
        else:
            flight=jump(samples,times,contacts,{},automatic_contacts=True)
            for m in flight:
                if m['key']=='jump_height_cm':
                    results.append(value('jump_height_cm',m['value'],m['reason']))
            base['estimated_contacts']={k:contacts[k] for k in ('takeoff_frame','landing_frame')}
            base['flight_sampling_bounds_cm']=[m['value'] for m in flight[1:]]
    return dict(base,metrics=results)


def attach_metrics(analysis, samples, inputs, metadata):
    validate_inputs(inputs,metadata,analysis['segments'])
    for event in analysis['events']:
        segment=analysis['segments'][event['segment']]
        context=inputs.get('segments',{}).get(str(segment['start_frame']),{})
        window=[s for s in samples if event['start_frame']<=s.frame<=event['end_frame']]
        event.update(measure_event(window,event,context,metadata))
        event['comparison_context']={**{k:inputs.get(k,'') for k in ('athlete_code','setup_label','recorded_on','session_phase')},
                                     'capture':context,'protocol':event['kind'],'measurement_version':VERSION,
                                     'compatibility':'unverified','discovery_version':analysis['algorithm'],
                                     'capture_source':'user_input' if context else 'unknown'}
    analysis['measurement_version']=VERSION
    analysis['warnings'][0]='Deneysel görüntü ölçümleri; bağımsız doğrulama henüz yapılmadı.'
    return analysis
