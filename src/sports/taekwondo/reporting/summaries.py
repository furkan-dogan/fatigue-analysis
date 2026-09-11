"""Legacy taekwondo report data; rules remain unvalidated."""
from __future__ import annotations

from typing import Iterable
from src.core.numeric import pct_change
from src.sports.taekwondo.reporting.catalog import EMG_CH1_GROUP, EMG_CH1_MUSCLE, EMG_CH2_GROUP, EMG_CH2_MUSCLE
from src.sports.taekwondo.reporting.formatting import change_label, fmt_value, mean_for, metric_info, readable_metric_value, status_badge


def comparison_rows(pre_events: list[dict], post_events: list[dict], keys: Iterable[str]) -> list[dict]:
    rows: list[dict] = []
    for key in keys:
        info = metric_info(key)
        pre = mean_for(pre_events, key)
        post = mean_for(post_events, key)
        pct = pct_change(pre, post)
        rows.append(
            {
                "Metrik": info.label,
                "Pre Ortalama": readable_metric_value(key, pre),
                "Post Ortalama": readable_metric_value(key, post),
                "Değişim": f"{pct:+.1f}%" if pct is not None else "-",
                "Sonuç": status_badge(key, pct),
                "Bu Ne Anlama Geliyor?": change_label(key, pct),
                "Nasıl Hesaplandı": info.calculation,
            }
        )
    return rows


def session_summary_rows(events: list[dict], label: str, total_frames: int, fps: float) -> list[dict]:
    duration = total_frames / fps if fps else 0.0
    conf_vals = [float(e["pose_confidence"]) for e in events if e.get("pose_confidence") is not None]
    low_conf = [e for e in events if e.get("confidence_flag") == "low"]
    return [
        {"Oturum": label, "Ölçüm": "Video süresi", "Değer": f"{duration:.1f} sn", "Yorum": "Analiz edilen toplam görüntü süresi."},
        {"Oturum": label, "Ölçüm": "Tespit edilen tekme", "Değer": str(len(events)), "Yorum": "Otomatik event detection ile geçerli kabul edilen tekme sayısı."},
        {"Oturum": label, "Ölçüm": "Ortalama diz ROM", "Değer": fmt_value(mean_for(events, "active_knee_rom_deg"), "active_knee_rom_deg"), "Yorum": "Hareket açıklığı ve teknik genişlik göstergesi."},
        {"Oturum": label, "Ölçüm": "Ortalama maksimum diz hızı", "Değer": fmt_value(mean_for(events, "active_peak_knee_vel_deg_s"), "active_peak_knee_vel_deg_s"), "Yorum": "Patlayıcı hız üretimi göstergesi."},
        {"Oturum": label, "Ölçüm": "Ortalama tekme yüksekliği", "Değer": fmt_value(mean_for(events, "peak_kick_height_norm"), "peak_kick_height_norm"), "Yorum": "Gövde uzunluğuna göre normalize edilmiş yükseklik."},
        {"Oturum": label, "Ölçüm": "Ortalama pose güveni", "Değer": fmt_value(sum(conf_vals) / len(conf_vals) if conf_vals else None, "pose_confidence"), "Yorum": f"Düşük güvenli tekme: {len(low_conf)}."},
    ]


def movement_summary_rows(pre_events: list[dict], post_events: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for key in [
        "peak_kick_height_norm",
        "active_min_knee_angle_deg",
        "active_peak_knee_angle_deg",
        "active_knee_rom_deg",
        "active_peak_knee_vel_deg_s",
        "duration_sec",
        "pose_confidence",
    ]:
        pre = mean_for(pre_events, key)
        post = mean_for(post_events, key)
        pct = pct_change(pre, post)
        if key == "active_min_knee_angle_deg":
            label = "Ortalama diz fleksiyonu"
            meaning = "Tekme öncesi dizin ne kadar büküldüğünü gösterir. Daha küçük açı daha fazla bükülme anlamına gelir."
            calc = "Tekme penceresinde aktif diz açısının minimum değeri alınır."
        elif key == "active_peak_knee_angle_deg":
            label = "Ortalama diz ekstansiyonu"
            meaning = "Tekme sırasında dizin ne kadar açıldığını gösterir. 180 dereceye yaklaşması dizin daha çok açıldığını gösterir."
            calc = "Tekme penceresinde aktif diz açısının maksimum değeri alınır."
        else:
            info = metric_info(key)
            label = info.label
            meaning = info.interpretation
            calc = info.calculation
        rows.append(
            {
                "Ölçüm": label,
                "Pre": readable_metric_value(key, pre),
                "Post": readable_metric_value(key, post),
                "Değişim": f"{pct:+.1f}%" if pct is not None else "-",
                "Bu Ne Anlama Geliyor?": meaning,
                "Nasıl Hesaplandı": calc,
            }
        )
    return rows


def emg_device_rows(events: list[dict], emg_rows: list[dict], session_label: str) -> list[dict]:
    if not events or not emg_rows:
        return []

    baseline_window = max(1, min(len(emg_rows), 120))
    baseline_rectus_freq = sum(float(r.get("EMG_CH1_median_freq_Hz", r["EMG_median_freq_Hz"])) for r in emg_rows[:baseline_window]) / baseline_window
    baseline_biceps_freq = sum(float(r.get("EMG_CH2_median_freq_Hz", r["EMG_median_freq_Hz"])) for r in emg_rows[:baseline_window]) / baseline_window
    rows: list[dict] = []

    for ev in events:
        start = max(0, min(int(ev.get("start_frame", 0)), len(emg_rows) - 1))
        end = max(start, min(int(ev.get("end_frame", start)), len(emg_rows) - 1))
        window = emg_rows[start : end + 1]
        if not window:
            continue

        rectus_vals = [float(r["EMG_RMS_mV"]) for r in window]
        biceps_vals = [float(r["EMG_CH2_RMS_mV"]) for r in window]
        rectus_freq_vals = [float(r.get("EMG_CH1_median_freq_Hz", r["EMG_median_freq_Hz"])) for r in window]
        biceps_freq_vals = [float(r.get("EMG_CH2_median_freq_Hz", r["EMG_median_freq_Hz"])) for r in window]
        rectus_peak = max(rectus_vals)
        biceps_peak = max(biceps_vals)
        rectus_mean = sum(rectus_vals) / len(rectus_vals)
        biceps_mean = sum(biceps_vals) / len(biceps_vals)
        rectus_freq = sum(rectus_freq_vals) / len(rectus_freq_vals)
        biceps_freq = sum(biceps_freq_vals) / len(biceps_freq_vals)
        mean_freq = (rectus_freq + biceps_freq) / 2
        freq_drop = ((baseline_rectus_freq - rectus_freq) + (baseline_biceps_freq - biceps_freq)) / 2

        dominant_rms = max(rectus_mean, biceps_mean)
        if dominant_rms >= 0.45:
            activation = "Yüksek aktivasyon"
        elif dominant_rms >= 0.25:
            activation = "Orta aktivasyon"
        else:
            activation = "Düşük aktivasyon"

        balance_ratio = rectus_mean / max(biceps_mean, 1e-6)
        if balance_ratio >= 1.35:
            balance = "Rectus femoris baskın"
        elif balance_ratio <= 0.85:
            balance = "Biceps femoris baskın"
        else:
            balance = "Ön-arka uyluk dengeli"

        if freq_drop >= 10:
            fatigue = "Belirgin yorgunluk"
            interp = "Median frekans belirgin düşmüş; uyluk kaslarında kasılma verimliliği azalmış görünüyor."
        elif freq_drop >= 5:
            fatigue = "Hafif/orta yorgunluk"
            interp = "Median frekansta düşüş var; kas yorgunluğu başlamış olabilir."
        else:
            fatigue = "Yorgunluk sınırlı"
            interp = "Median frekans korunmuş; bu tekmede kas yorgunluğu belirgin değil."

        rows.append(
            {
                "Oturum": session_label,
                "kick_id": f"T{int(ev['kick_id'])}",
                "start_frame": int(ev.get("start_frame", start)),
                "peak_frame": int(ev.get("peak_frame", start)),
                "end_frame": int(ev.get("end_frame", end)),
                "start_time_sec": f"{float(ev.get('start_time_sec', 0)):.2f}",
                "peak_time_sec": f"{float(ev.get('peak_time_sec', 0)):.2f}",
                "end_time_sec": f"{float(ev.get('end_time_sec', 0)):.2f}",
                "CH1 Kas": EMG_CH1_MUSCLE,
                "CH1 Bölge": EMG_CH1_GROUP,
                "rectus_femoris_rms_mv": f"{rectus_mean:.3f}",
                "rectus_femoris_peak_rms_mv": f"{rectus_peak:.3f}",
                "rectus_femoris_median_freq_hz": f"{rectus_freq:.1f}",
                "CH2 Kas": EMG_CH2_MUSCLE,
                "CH2 Bölge": EMG_CH2_GROUP,
                "biceps_femoris_rms_mv": f"{biceps_mean:.3f}",
                "biceps_femoris_peak_rms_mv": f"{biceps_peak:.3f}",
                "biceps_femoris_median_freq_hz": f"{biceps_freq:.1f}",
                "ortalama_emg_median_freq_hz": f"{mean_freq:.1f}",
                "Aktivasyon Yorumu": activation,
                "Kas Dengesi": balance,
                "fatigue_flag": fatigue,
                "Yorum": interp,
            }
        )
    return rows


def emg_summary_text(rows: list[dict]) -> str:
    if not rows:
        return "EMG çıktısı üretmek için yeterli tekme veya EMG verisi yok."
    freqs = [float(r["ortalama_emg_median_freq_hz"]) for r in rows]
    rectus_freqs = [float(r["rectus_femoris_median_freq_hz"]) for r in rows]
    biceps_freqs = [float(r["biceps_femoris_median_freq_hz"]) for r in rows]
    rectus_rms = [float(r["rectus_femoris_rms_mv"]) for r in rows]
    biceps_rms = [float(r["biceps_femoris_rms_mv"]) for r in rows]
    first_freq, last_freq = freqs[0], freqs[-1]
    first_rectus, last_rectus = rectus_rms[0], rectus_rms[-1]
    first_biceps, last_biceps = biceps_rms[0], biceps_rms[-1]
    freq_delta = last_freq - first_freq
    rectus_delta = last_rectus - first_rectus
    biceps_delta = last_biceps - first_biceps
    avg_rectus = sum(rectus_rms) / len(rectus_rms)
    avg_biceps = sum(biceps_rms) / len(biceps_rms)
    ratio = avg_rectus / max(avg_biceps, 1e-6)
    parts = [
        f"EMG çıktısı {EMG_CH1_MUSCLE} ve {EMG_CH2_MUSCLE} için değerlendirildi.",
        f"İlk tekmeden son tekmeye ortalama median frekans {first_freq:.1f} Hz -> {last_freq:.1f} Hz değişti.",
        f"Rectus femoris frekansı {rectus_freqs[0]:.1f} Hz -> {rectus_freqs[-1]:.1f} Hz; biceps femoris frekansı {biceps_freqs[0]:.1f} Hz -> {biceps_freqs[-1]:.1f} Hz değişti.",
        f"Rectus femoris RMS {first_rectus:.3f} mV -> {last_rectus:.3f} mV; biceps femoris RMS {first_biceps:.3f} mV -> {last_biceps:.3f} mV değişti.",
    ]
    if freq_delta <= -8:
        parts.append("Median frekans düşüşü uyluk kaslarında nöromüsküler yorgunluk ile uyumlu.")
    elif freq_delta <= -4:
        parts.append("Median frekansta hafif düşüş var; yorgunluk başlangıcı olabilir.")
    else:
        parts.append("Median frekans büyük ölçüde korunmuş; EMG tarafında belirgin yorgunluk sınırlı.")
    if ratio >= 1.35:
        parts.append("Ortalama aktivasyon rectus femoris tarafında baskın; vuruşta diz açma ve bacağı kaldırma fazı daha belirgin çalışmış.")
    elif ratio <= 0.85:
        parts.append("Ortalama aktivasyon biceps femoris tarafında baskın; geri çekme ve frenleme fazı daha belirgin çalışmış.")
    else:
        parts.append("Rectus femoris ve biceps femoris aktivasyonu birbirine yakın; ön-arka uyluk yük dağılımı dengeli görünüyor.")
    if rectus_delta > 0.05 or biceps_delta > 0.05:
        parts.append("RMS artışı, aynı hareketi sürdürebilmek için kas aktivasyonunun arttığını gösterebilir.")
    return " ".join(parts)


def readable_kick_rows(events: list[dict]) -> list[dict]:
    rows = []
    for ev in events:
        conf = ev.get("pose_confidence")
        conf_flag = ev.get("confidence_flag")
        rom = ev.get("active_knee_rom_deg")
        vel = ev.get("active_peak_knee_vel_deg_s")
        duration = ev.get("duration_sec")
        notes = []
        if conf_flag == "low":
            notes.append("Pose güveni düşük; ihtiyatlı yorumlanmalı.")
        if rom is not None and float(rom) < 20:
            notes.append("Diz ROM düşük.")
        if duration is not None and float(duration) > 2:
            notes.append("Tekme süresi uzun.")
        if not notes:
            notes.append("Geçerli tekme; karşılaştırmada kullanılabilir.")
        rows.append(
            {
                "Tekme": f"T{int(ev['kick_id'])}",
                "Bacak": "Sağ" if ev.get("active_leg") == "R" else ("Sol" if ev.get("active_leg") == "L" else "Belirsiz"),
                "Zaman": f"{float(ev.get('start_time_sec', 0)):.2f}-{float(ev.get('end_time_sec', 0)):.2f} sn",
                "Süre": fmt_value(duration, "duration_sec"),
                "Diz ROM": fmt_value(rom, "active_knee_rom_deg"),
                "Peak Diz Hızı": fmt_value(vel, "active_peak_knee_vel_deg_s"),
                "Yükseklik": fmt_value(ev.get("peak_kick_height_norm"), "peak_kick_height_norm"),
                "Güven": fmt_value(conf, "pose_confidence"),
                "Yorum": " ".join(notes),
            }
        )
    return rows
