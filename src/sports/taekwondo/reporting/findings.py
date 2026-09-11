"""Legacy taekwondo report data; rules remain unvalidated."""
from __future__ import annotations


from src.sports.taekwondo.reporting.catalog import PRIMARY_METRICS
from src.sports.taekwondo.reporting.formatting import change_label, fmt_value, metric_info, readable_metric_value


def top_findings(pre_events: list[dict], post_events: list[dict], fatigue_data: dict, fi: float) -> list[str]:
    findings: list[str] = []
    sorted_metrics = sorted(
        (
            (key, m)
            for key, m in fatigue_data.get("metrics", {}).items()
            if key in PRIMARY_METRICS and m.get("fatigue_contribution") is not None
        ),
        key=lambda item: item[1]["fatigue_contribution"],
        reverse=True,
    )
    for key, m in sorted_metrics[:3]:
        pct = m.get("pct")
        if pct is None:
            continue
        findings.append(
            f"{metric_info(key).label}: Pre {fmt_value(m.get('pre'), key)}, Post {fmt_value(m.get('post'), key)}, değişim {pct:+.1f}%. {change_label(key, pct)}"
        )
    if len(pre_events) < 5 or len(post_events) < 5:
        findings.append(
            f"Veri gücü sınırlı: Pre {len(pre_events)}, Post {len(post_events)} tekme var. Daha güvenilir yorum için her oturumda daha fazla geçerli tekme gerekir."
        )
    if fi >= 66:
        findings.insert(0, f"Genel sonuç: Yorgunluk indeksi {fi:.1f}/100 ve yüksek düzeyde yorgunluk gösteriyor.")
    elif fi >= 33:
        findings.insert(0, f"Genel sonuç: Yorgunluk indeksi {fi:.1f}/100 ve orta düzey yorgunluk gösteriyor.")
    else:
        findings.insert(0, f"Genel sonuç: Yorgunluk indeksi {fi:.1f}/100 ve düşük düzey yorgunluk gösteriyor.")
    return findings


def readable_findings(pre_events: list[dict], post_events: list[dict], fatigue_data: dict, fi: float) -> list[dict]:
    rows: list[dict] = []
    if fi >= 66:
        rows.append({
            "Bulgu": "Genel yorgunluk yüksek",
            "Ne Gördük?": f"Yorgunluk indeksi {fi:.1f}/100.",
            "Bu Ne Demek?": "Antrenman sonrası performans düşüşü belirgin. Hız, yükseklik veya hareket genişliği gibi metriklerde kayıp olabilir.",
        })
    elif fi >= 33:
        rows.append({
            "Bulgu": "Genel yorgunluk orta düzeyde",
            "Ne Gördük?": f"Yorgunluk indeksi {fi:.1f}/100.",
            "Bu Ne Demek?": "Vücutta yük birikimi var. Ağır çalışma yerine kontrollü tempo daha uygun olabilir.",
        })
    else:
        rows.append({
            "Bulgu": "Genel yorgunluk düşük",
            "Ne Gördük?": f"Yorgunluk indeksi {fi:.1f}/100.",
            "Bu Ne Demek?": "Ana performans göstergeleri büyük ölçüde korunmuş görünüyor.",
        })

    labels = {
        "peak_kick_height_norm": "Tekme yüksekliği değişti",
        "active_knee_rom_deg": "Diz hareket genişliği değişti",
        "active_peak_knee_vel_deg_s": "Tekme hızı değişti",
        "active_peak_foot_speed_norm": "Ayak hızı değişti",
        "duration_sec": "Tekme süresi değişti",
        "time_to_peak_knee_vel_sec": "Maksimum hıza ulaşma süresi değişti",
    }
    sorted_metrics = sorted(
        (
            (key, m)
            for key, m in fatigue_data.get("metrics", {}).items()
            if key in labels and m.get("fatigue_contribution") is not None
        ),
        key=lambda item: item[1]["fatigue_contribution"],
        reverse=True,
    )
    for key, m in sorted_metrics[:3]:
        rows.append({
            "Bulgu": labels[key],
            "Ne Gördük?": f"Pre: {readable_metric_value(key, m.get('pre'))} | Post: {readable_metric_value(key, m.get('post'))} | Değişim: {m.get('pct'):+.1f}%",
            "Bu Ne Demek?": change_label(key, m.get("pct")),
        })
    if len(pre_events) < 5 or len(post_events) < 5:
        rows.append({
            "Bulgu": "Tekme sayısı sınırlı",
            "Ne Gördük?": f"Pre {len(pre_events)} tekme, Post {len(post_events)} tekme.",
            "Bu Ne Demek?": "Sonuç okunabilir ama daha güvenilir karşılaştırma için her oturumda 5-8 temiz tekme daha iyi olur.",
        })
    return rows


def action_recommendations(pre_events: list[dict], post_events: list[dict], fatigue_data: dict, fi: float) -> list[str]:
    recs: list[str] = []
    metrics = fatigue_data.get("metrics", {})
    vel = metrics.get("active_peak_knee_vel_deg_s", {})
    rom = metrics.get("active_knee_rom_deg", {})
    height = metrics.get("peak_kick_height_norm", {})
    duration = metrics.get("duration_sec", {})

    if fi >= 66:
        recs.append("Bugün yüksek yoğunluklu antrenman veya sert sparring yerine 48-72 saat toparlanma odaklı çalışmak daha uygun.")
    elif fi >= 33:
        recs.append("Yüklenmeyi kontrollü tut; teknik çalışma, hafif tempo ve aktif toparlanma daha uygun.")
    else:
        recs.append("Genel yorgunluk düşük görünüyor; yine de hız ve teknik kalite korunarak kademeli yüklenilebilir.")

    if vel.get("pct") is not None and vel["pct"] < -8:
        recs.append("Tekme hızın düşmüş. Patlayıcı güç için kısa setli hızlı tekme, tam dinlenmeli sprint ve plyometrik çalışma eklenebilir.")
    if rom.get("pct") is not None and rom["pct"] < -8:
        recs.append("Diz hareket açıklığın azalmış. Antrenman öncesi dinamik ısınma, sonrasında rectus femoris, biceps femoris ve kalça esnetme eklenmeli.")
    if height.get("pct") is not None and height["pct"] < -8:
        recs.append("Tekme yüksekliği düşmüş. Kalça mobilitesi ve teknik yükseklik çalışmaları öncelikli olmalı.")
    if duration.get("pct") is not None and duration["pct"] > 8:
        recs.append("Tekme süren uzamış. Yorgunken tekniğin yavaşlamaması için düşük hacimli ama kaliteli tekrarlar yapılmalı.")
    if len(pre_events) < 5 or len(post_events) < 5:
        recs.append("Daha net sonuç için bir sonraki analizde her oturumda en az 5-8 temiz tekme kaydı alınmalı.")
    return recs


def analysis_paragraph(pre_events: list[dict], post_events: list[dict], fi: float, fatigue_data: dict) -> str:
    main = top_findings(pre_events, post_events, fatigue_data, fi)
    return " ".join(main[:4])
