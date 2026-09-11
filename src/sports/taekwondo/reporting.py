"""Video-only result preparation; no UI or clinical inference."""
import math
from src.core.statistics import compare_metric

METRICS = {
    'duration_sec': ('Tekme süresi', 's'),
    'active_knee_rom_deg': ('Aktif diz hareket açıklığı', '°'),
    'active_peak_knee_vel_deg_s': ('Tepe diz açısal hızı', '°/s'),
    'peak_kick_height_norm': ('Gövdeye göre tekme yüksekliği', 'oran'),
    'extension_dur_sec': ('Uzatma süresi', 's'),
    'retraction_dur_sec': ('Geri çekim süresi', 's'),
    'extension_peak_vel_deg_s': ('Uzatma tepe açısal hızı', '°/s'),
    'retraction_peak_vel_deg_s': ('Geri çekim tepe açısal hızı', '°/s'),
    'knee_asi': ('Diz açı asimetrisi', '%'),
    'hip_asi': ('Kalça açı asimetrisi', '%'),
}

def values(events, key):
    result = []
    for event in events:
        try:
            value = float(event.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            result.append(value)
    return result

def comparison_rows(pre, post):
    rows = []
    for key, (label, unit) in METRICS.items():
        a, b = values(pre, key), values(post, key)
        before = sum(a) / len(a) if a else None
        after = sum(b) / len(b) if b else None
        delta = after - before if before is not None and after is not None else None
        rows.append({'Metrik': label, 'Birim': unit, 'Önce': before, 'Sonra': after,
                     'Fark': delta, 'Değişim (%)': delta / abs(before) * 100 if delta is not None and before else None,
                     'Önce tekrar': len(a), 'Sonra tekrar': len(b)})
    return rows

def statistics_rows(pre, post):
    rows = []
    for key, (label, unit) in METRICS.items():
        a, b = values(pre, key), values(post, key)
        if len(a) < 2 or len(b) < 2:
            continue
        result = compare_metric(a, b)
        rows.append({'Metrik': label, 'Birim': unit, 'Önce std': result['pre_std'],
                     'Sonra std': result['post_std'], "Cohen d": result['cohens_d'],
                     'Önce tekrar': len(a), 'Sonra tekrar': len(b)})
    return rows

def quality_notices(pre, post):
    notices = [('info', 'Sonuçlar görüntü düzlemindeki tahminlerdir. Gerçek video ile ölçüm doğrulaması henüz yapılmadı.')]
    for label, events in [('Önce', pre), ('Sonra', post)]:
        if not events:
            notices.append(('warning', f'{label}: tekme tespit edilmedi; metrikler eksik olabilir.'))
        elif len(events) < 5:
            notices.append(('warning', f'{label}: yalnızca {len(events)} tekrar var; karşılaştırma sınırlıdır.'))
        confidence = values(events, 'pose_confidence')
        if events and (len(confidence) != len(events) or any(v < 0.6 for v in confidence)):
            notices.append(('warning', f'{label}: takip görünürlüğü düşük veya eksik.'))
    return notices
