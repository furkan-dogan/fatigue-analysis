"""Legacy event quality rules, independent of their renderer."""

def quality_notices(pre_events: list[dict], post_events: list[dict]) -> list[tuple[str, str]]:
    notices = []
    low_pre = [e for e in pre_events if e.get('confidence_flag') == 'low']
    low_post = [e for e in post_events if e.get('confidence_flag') == 'low']
    if len(pre_events) < 5 or len(post_events) < 5:
        notices.append(('warning', f"Veri gücü uyarısı: Pre {len(pre_events)}, Post {len(post_events)} tekme var. Her oturumda en az 5-8 geçerli tekme olduğunda ortalama, yüzde değişim ve Cohen's d daha güvenilir yorumlanır."))
    if low_pre or low_post:
        notices.append(('warning', f'Pose güven uyarısı: Pre düşük güvenli tekme {len(low_pre)}, Post düşük güvenli tekme {len(low_post)}. Bu tekmelerde eklem açıları ve hızlar kamera açısı, örtüşme veya model takibi nedeniyle daha gürültülü olabilir.'))
    if not low_pre and (not low_post) and (len(pre_events) >= 5) and (len(post_events) >= 5):
        notices.append(('success', 'Veri kalitesi iyi: tekme sayısı ve pose güveni temel yorum için yeterli görünüyor.'))
    return notices
