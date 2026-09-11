"""Experimental 2D measurements; physical/protocol gates precede every estimate."""
import math
import numpy as np
from src.core.geometry import calculate_angle
from src.core.pose import point
from src.sports.volleyball.signals import midpoint, torso, sole, physical_times, flight_candidates

VERSION = 'volleyball-1'
DEFINITIONS = {
    'jump_height_cm': ('Tahmini sıçrama yüksekliği', 'cm', 'flight_time_midpoint'),
    'jump_lower_cm': ('Kare belirsizliği alt sınırı', 'cm', 'flight_time_bracket'),
    'jump_upper_cm': ('Kare belirsizliği üst sınırı', 'cm', 'flight_time_bracket'),
    'lateral_drift_torso_pct': ('En büyük yana sapma / başlangıç gövdesi', '% gövde', 'pelvis_displacement_2d'),
    'trunk_lean_deg': ('En büyük gövde eğimi', '°', 'shoulder_pelvis_vertical_2d'),
    'pelvis_tilt_deg': ('En büyük pelvis eğimi', '°', 'bilateral_hip_axis_2d'),
    'landing_difference_ms': ('Sağ eksi sol temas zamanı', 'ms', 'manual_contact_difference'),
    'peak_speed_m_s': ('En yüksek pencere ortalamalı hız', 'm/s', 'local_linear_fit_0.20s'),
}
KEYS = {'cmj': list(DEFINITIONS)[:3], 'asymmetry': list(DEFINITIONS)[3:7], 'sprint': ['peak_speed_m_s']}


def metric(key, value=None, reason=None):
    label, unit, method = DEFINITIONS[key]
    return dict(key=key, label=label, value=value, unit=unit, method=f'{VERSION}:{method}',
                quality='unvalidated' if value is not None else 'rejected', reason=reason)


def gate(review, times, samples, metadata):
    if review['test'] not in KEYS:
        return 'Yaklaşmalı sıçrama bu ölçüm protokolünün dışında; CMJ olarak hesaplanmaz.'
    if times is None or not metadata.get('timestamps_monotonic') or not metadata.get('timestamp_count_matches_frames'):
        return 'Kaynak zamanları ve fiziksel zaman ölçeği doğrulanmalı.'
    if len(samples) < 5 or np.max(np.diff(times)) > 0.05:
        return 'Zaman örneklemesi yetersiz veya kareler arasında büyük boşluk var.'
    if review.get('camera') != 'fixed' or not review.get('single_take_confirmed'):
        return 'Sabit kamera ve tek kesintisiz çekim gerekli.'
    if any(s.issue for s in samples):
        return 'Seçilen aralıkta sahne değişimi, takip kaybı veya sporcu belirsizliği var.'
    if review['test'] == 'cmj' and (review.get('view') != 'side' or not review.get('cmj_confirmed')):
        return 'CMJ için yandan çekim ve yerinde çift ayak protokolü onayı gerekli.'
    if review['test'] == 'asymmetry' and (review.get('view') not in ('front', 'back') or not review.get('camera_level_confirmed')):
        return 'Asimetri için düz tutulmuş ön/arka kamera gerekli.'
    if review['test'] == 'sprint' and review.get('view') != 'side':
        return 'Sprint için yandan çekim gerekli.'
    return None


def jump(samples, times, repeat, review):
    keys = KEYS['cmj']
    def reject(reason):
        return [metric(k, reason=reason) for k in keys]
    if not review.get('contacts_confirmed') or not review.get('feet_visible'):
        return reject('Kalkış/iniş kareleri ve ayak görünürlüğü kontrol edilmeli.')
    if not review.get('posture_confirmed'):
        return reject('Kalkış ve inişte benzer vücut duruşu varsayımı doğrulanmalı.')
    mapping = {s.frame: i for i, s in enumerate(samples)}
    a, b = mapping.get(repeat['takeoff_frame']), mapping.get(repeat['landing_frame'])
    if a is None or b is None or a < 1 or b <= a:
        return reject('Temas sınırları için önceki kare dahil olmalı.')
    window = samples[a-1:b+1]
    required = [side + '_' + part for side in ('left', 'right')
                for part in ('hip', 'knee', 'ankle', 'big_toe', 'small_toe', 'heel', 'shoulder')]
    if any(point(s, k) is None for s in window for k in required):
        return reject('Kalkış/iniş ve uçuş boyunca gerekli noktalar görünmüyor.')
    scale = torso(samples[a-1])
    if scale is None or scale < 10:
        return reject('Gövde referansı yetersiz.')
    hips = np.array([midpoint(s, 'hip') for s in window])
    if np.ptp(hips[:, 0]) > 0.2 * scale:
        return reject('CMJ aralığında yatay ilerleme fazla; yerinde sıçrama doğrulanmadı.')
    # Screen observable posture differences; this does not estimate the 3D center of mass.
    for side in ('left', 'right'):
        angles = [calculate_angle(*(point(s, f'{side}_{part}') for part in ('hip', 'knee', 'ankle')))
                  for s in (samples[a-1], samples[b])]
        if not all(math.isfinite(v) for v in angles) or abs(angles[1] - angles[0]) > 15:
            return reject('Kalkış ve iniş diz duruşu farklı; uçuş süresi varsayımı uygun değil.')
        ankle_angles = [calculate_angle(*(point(s, f'{side}_{part}') for part in ('knee', 'ankle', 'big_toe')))
                        for s in (samples[a-1], samples[b])]
        if not all(math.isfinite(v) for v in ankle_angles) or abs(ankle_angles[1]-ankle_angles[0]) > 15:
            return reject('Kalkış ve inişte ayak bileği duruşu farklı.')
    heights = [midpoint(s, 'hip')[1] - max(sole(s, side) for side in ('left', 'right'))
               for s in (samples[a-1], samples[b])]
    if abs(heights[1]-heights[0]) > .08*scale:
        return reject('Kalkış ve inişte pelvis/ayak yüksekliği ilişkisi farklı.')
    low = times[b-1] - times[a]
    high = times[b] - times[a-1]
    if not 0.1 <= low <= high <= 1.2:
        return reject('Uçuş aralığı protokol sınırları dışında.')
    duration = (low + high) / 2
    return [metric('jump_height_cm', float(9.80665 * duration**2 / 8 * 100)),
            metric('jump_lower_cm', float(9.80665 * low**2 / 8 * 100)),
            metric('jump_upper_cm', float(9.80665 * high**2 / 8 * 100))]


def asymmetry(samples, times, repeat, review):
    keys = KEYS['asymmetry']
    if not review.get('feet_visible'):
        return [metric(k, reason='Ayak görünürlüğü doğrulanmalı.') for k in keys]
    hips, shoulders, scales, tilts = [], [], [], []
    for sample in samples:
        hip, shoulder, scale = midpoint(sample, 'hip'), midpoint(sample, 'shoulder'), torso(sample)
        left, right = point(sample, 'left_hip'), point(sample, 'right_hip')
        if hip is None or shoulder is None or scale is None or scale < 10 or abs(right[0]-left[0]) < scale*0.1:
            return [metric(k, reason='Önden/arkadan gövde ve pelvis noktaları yetersiz.') for k in keys]
        hips.append(hip); shoulders.append(shoulder); scales.append(scale)
        angle = math.degrees(math.atan2(right[1]-left[1], right[0]-left[0]))
        tilts.append((angle+90) % 180-90)
    hips, shoulders = np.array(hips), np.array(shoulders)
    lean = np.degrees(np.arctan2(shoulders[:, 0]-hips[:, 0], hips[:, 1]-shoulders[:, 1]))
    results = [metric('lateral_drift_torso_pct', float(np.max(np.abs(hips[:, 0]-hips[0, 0]))/scales[0]*100)),
               metric('trunk_lean_deg', float(np.max(np.abs(lean)))),
               metric('pelvis_tilt_deg', float(np.max(np.abs(tilts))))]
    indices = {s.frame: i for i, s in enumerate(samples)}
    l, r = indices.get(repeat.get('left_landing_frame')), indices.get(repeat.get('right_landing_frame'))
    if l is None or r is None or not review.get('contacts_confirmed'):
        results.append(metric('landing_difference_ms', reason='Sağ ve sol temas kareleri ayrı işaretlenip onaylanmalı.'))
    elif any(sole(samples[i], side) is None for i in range(min(l, r), max(l, r)+1) for side in ('left', 'right')):
        results.append(metric('landing_difference_ms', reason='Temas aralığında ayak noktaları eksik.'))
    else:
        results.append(metric('landing_difference_ms', float((times[r]-times[l])*1000)))
    return results


def sprint(samples, times, review):
    key = 'peak_speed_m_s'
    calibration = review.get('calibration')
    if not calibration or not review.get('motion_plane_confirmed') or not review.get('camera_perpendicular'):
        return [metric(key, reason='Dik kamera ve pelvis hareket düzleminde mesafe referansı gerekli.')], []
    a = np.array([calibration['x1'], calibration['y1']], dtype=float)
    b = np.array([calibration['x2'], calibration['y2']], dtype=float)
    direction, length = b-a, np.linalg.norm(b-a)
    distance = calibration['distance_m']
    if length < 20 or not np.isfinite(distance) or distance <= 0:
        return [metric(key, reason='Mesafe referansı geçersiz veya görüntüde çok kısa.')], []
    hips = [midpoint(s, 'hip') for s in samples]
    if any(h is None for h in hips):
        return [metric(key, reason='Pelvis takibi kesintisiz değil.')], []
    hips = np.array(hips)
    unit = direction/length
    fractions = (hips-a) @ unit / length
    perpendicular = np.abs((hips-a) @ np.array([-unit[1], unit[0]]))
    if np.max(perpendicular) > 0.1*length or np.min(fractions) < 0 or np.max(fractions) > 1:
        return [metric(key, reason='Pelvis yolu kalibrasyon çizgisinin düzlem/aralık koşullarını karşılamıyor.')], []
    position = fractions * distance
    # Centered local least-squares slope on a 0.20s window; no gap filling/extrapolation.
    series = []
    for i, t in enumerate(times):
        mask = np.abs(times-t) <= 0.1 + 1e-9
        x = times[mask]
        speed = None
        if len(x) >= 5 and t-times[0] >= 0.1-1e-9 and times[-1]-t >= 0.1-1e-9:
            centered = x-x.mean()
            slope = float(centered @ (position[mask]-position[mask].mean()) / (centered @ centered))
            residual = position[mask] - (position[mask].mean() + slope*centered)
            if np.max(np.abs(residual)) <= 0.01*distance:
                speed = abs(slope)
        series.append(dict(frame=samples[i].frame, time_seconds=float(t), position_m=float(position[i]), speed_m_s=speed))
    speeds = [s['speed_m_s'] for s in series if s['speed_m_s'] is not None]
    if not speeds:
        return [metric(key, reason='0,20 saniyelik hız penceresi için yeterli örnek yok.')], series
    return [metric(key, float(max(speeds)))], series


def analyze(samples, review, metadata):
    times = physical_times(samples, review)
    reason = gate(review, times, samples, metadata)
    result = dict(algorithm=VERSION, validation='unvalidated', candidates=[], events=[], speed_series=[], warnings=[])
    if reason:
        result['warnings'].append(reason)
        result['metrics'] = [metric(k, reason=reason) for k in KEYS.get(review['test'], [])]
        return result
    candidates = flight_candidates(samples, times) if review['test'] != 'sprint' else []
    result['candidates'] = candidates
    repeats = review['repetitions']
    if review['test'] == 'sprint' and not repeats:
        repeats = [dict(start_frame=samples[0].frame, end_frame=samples[-1].frame)]
    if not repeats:
        result['metrics'] = [metric(k, reason='Tekrar ve temas karelerini inceleyip kaydedin.') for k in KEYS[review['test']]]
        return result
    result['metrics'] = []
    for repeat in repeats:
        indices = [i for i, s in enumerate(samples) if repeat['start_frame'] <= s.frame <= repeat['end_frame']]
        window = [samples[i] for i in indices]
        t = times[indices]
        if not window:
            continue
        if review['test'] == 'cmj':
            values = jump(window, t, repeat, review)
        elif review['test'] == 'asymmetry':
            values = asymmetry(window, t, repeat, review)
        else:
            values, series = sprint(window, t, review)
            result['speed_series'].extend(series)
        result['events'].append(dict(**repeat, metrics=values))
    return result
