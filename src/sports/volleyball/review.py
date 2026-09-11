"""Manual volleyball review contract and validation; no performance metrics."""
from math import isfinite

TESTS = {'cmj': 'CMJ / Dikey sıçrama', 'asymmetry': 'İniş ve yana sapma', 'sprint': 'Sprint'}


def default_review(metadata):
    return dict(test='cmj', athlete='', start_frame=0, end_frame=metadata['frame_count'] - 1,
                athlete_box=None, calibration=None, repetitions=[],
                camera='unknown', view='unknown', feet_visible=False, physical_time_confirmed=False,
                notes='')


def validate_review(review, metadata):
    if review['test'] not in TESTS:
        raise ValueError('Test türü geçersiz.')
    def frame(value):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < metadata['frame_count']:
            raise ValueError('Kare numarası video sınırları dışında.')
    start, end = review['start_frame'], review['end_frame']
    frame(start); frame(end)
    if start > end:
        raise ValueError('Bitiş karesi başlangıçtan önce olamaz.')
    def point(x, y):
        if not (isfinite(x) and isfinite(y) and 0 <= x < metadata['width'] and 0 <= y < metadata['height']):
            raise ValueError('Referans noktaları görüntü sınırları içinde olmalı.')
    box = review.get('athlete_box')
    if box:
        frame(box['frame'])
        point(box['x1'], box['y1']); point(box['x2'], box['y2'])
        if box['x1'] >= box['x2'] or box['y1'] >= box['y2']:
            raise ValueError('Sporcu kutusunun genişliği ve yüksekliği pozitif olmalı.')
    calibration = review.get('calibration')
    if calibration:
        frame(calibration['frame'])
        point(calibration['x1'], calibration['y1']); point(calibration['x2'], calibration['y2'])
        if (calibration['x1'], calibration['y1']) == (calibration['x2'], calibration['y2']):
            raise ValueError('Kalibrasyon için iki farklı nokta gerekli.')
        if not isfinite(calibration['distance_m']) or calibration['distance_m'] <= 0:
            raise ValueError('Referans mesafesi pozitif ve metre cinsinden olmalı.')
        if not calibration['plane'].strip():
            raise ValueError('Referansın bulunduğu düzlemi belirtin.')
    intervals = []
    for repeat in review['repetitions']:
        a, b = repeat['start_frame'], repeat['end_frame']
        frame(a); frame(b)
        if not start <= a < b <= end:
            raise ValueError('Tekrarlar seçili aralıkta ve başlangıç < bitiş olmalı.')
        if review['test'] != 'sprint':
            takeoff, landing = repeat['takeoff_frame'], repeat['landing_frame']
            frame(takeoff); frame(landing)
            if not a <= takeoff < landing <= b:
                raise ValueError('Kalkış ve iniş sırası tekrar aralığıyla uyumlu olmalı.')
        intervals.append((a, b))
    ordered = sorted(intervals)
    if any(b >= c for (_, b), (c, _) in zip(ordered, ordered[1:])):
        raise ValueError('Tekrar aralıkları çakışamaz.')


def quality_messages(review, metadata):
    messages = ['Bu kayıt manuel incelemedir; otomatik tespit ve performans ölçümü henüz çalıştırılmadı.']
    if metadata.get('reported_frame_count') not in (None, metadata['frame_count']):
        messages.append('Bildirilen kare sayısı ile okunan kare sayısı farklı; dosya eksik veya bozuk olabilir.')
    if not metadata.get('timestamps_monotonic') or not metadata['timestamp_count_matches_frames']:
        messages.append('Kaynak kare zamanları eksik veya tutarsız; zaman temelli ölçüme uygunluğu doğrulanmadı.')
    if not review['physical_time_confirmed']:
        messages.append('Kayıt zamanı / ağır çekim ilişkisi doğrulanmadı.')
    if review['camera'] != 'fixed':
        messages.append('Sabit kamera koşulu doğrulanmadı.')
    if review['test'] != 'sprint' and not review['feet_visible']:
        messages.append('Kalkış ve inişte ayakların görünürlüğü doğrulanmadı.')
    if review['test'] == 'asymmetry' and review['view'] not in ('front', 'back'):
        messages.append('Yana sapma için ön veya arka görünüm incelenmeli.')
    if review['test'] == 'sprint' and not review.get('calibration'):
        messages.append('Sprint için mesafe kalibrasyonu girilmedi.')
    if review.get('calibration'):
        messages.append('İki noktalı referans kaydedildi; perspektif veya hareket düzlemi kalibrasyonu doğrulanmış değildir.')
    return messages


def frame_time(index, metadata):
    times = metadata.get('timestamps', [])
    if metadata.get('timestamps_monotonic') and metadata['timestamp_count_matches_frames']:
        return times[index] - times[0]
    return None
