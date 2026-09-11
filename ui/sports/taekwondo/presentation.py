"""Readable presentation helpers for athlete-facing analysis output."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import streamlit as st

from src.sports.taekwondo.fatigue import FATIGUE_WEIGHTS
from src.core.numeric import events_mean, pct_change


@dataclass(frozen=True)
class MetricInfo:
    label: str
    unit: str
    plain: str
    calculation: str
    interpretation: str
    fatigue_direction: int
    decimals: int = 1


METRICS: dict[str, MetricInfo] = {
    "active_knee_rom_deg": MetricInfo(
        "Diz Hareket Açıklığı",
        "derece",
        "Tekme sırasında aktif bacağın dizinin ne kadar açılıp kapandığını gösterir.",
        "Tekme penceresinde aktif diz açısının maksimum değeri ile minimum değeri arasındaki fark alınır.",
        "Post değeri belirgin düşerse eklem hareket açıklığı azalmış, kas sertliği veya yorgunluk artmış olabilir.",
        -1,
        1,
    ),
    "active_min_knee_angle_deg": MetricInfo(
        "Diz Fleksiyon Açısı",
        "derece",
        "Tekme öncesinde dizin en fazla büküldüğü açıyı gösterir.",
        "Tekme penceresinde aktif diz açısının minimum değeri alınır.",
        "Açı küçüldükçe diz daha fazla bükülmüş kabul edilir; bu değer tekme hazırlık fazını anlamaya yardım eder.",
        -1,
        1,
    ),
    "active_peak_knee_angle_deg": MetricInfo(
        "Diz Ekstansiyon Açısı",
        "derece",
        "Tekme sırasında dizin en fazla açıldığı açıyı gösterir.",
        "Tekme penceresinde aktif diz açısının maksimum değeri alınır.",
        "180 dereceye yaklaşması dizin daha fazla açıldığını gösterir; tekme uzatma fazını anlamaya yardım eder.",
        -1,
        1,
    ),
    "active_peak_knee_vel_deg_s": MetricInfo(
        "Maksimum Diz Hızı",
        "derece/sn",
        "Tekme sırasında dizin ulaştığı en yüksek açısal hızdır.",
        "Diz açısı zaman serisinden açısal hız hesaplanır; tekme penceresindeki mutlak maksimum hız alınır.",
        "Post değeri düşerse patlayıcı hız üretimi azalmış kabul edilir. Yorgunluk için en güçlü göstergelerden biridir.",
        -1,
        0,
    ),
    "active_mean_knee_vel_deg_s": MetricInfo(
        "Ortalama Diz Hızı",
        "derece/sn",
        "Tekme boyunca diz hareketinin ortalama hızını gösterir.",
        "Tekme penceresindeki mutlak diz açısal hızlarının ortalaması alınır.",
        "Düşüş, tüm tekme boyunca hareket temposunun yavaşladığını gösterir.",
        -1,
        0,
    ),
    "time_to_peak_knee_vel_sec": MetricInfo(
        "Peak Hıza Ulaşma Süresi",
        "sn",
        "Tekmenin başlangıcından maksimum diz hızına ulaşana kadar geçen süredir.",
        "Tekme başlangıcı ile tekme penceresindeki maksimum diz hızı zamanı arasındaki farktır.",
        "Post değeri artarsa sporcu maksimum hıza daha geç ulaşıyor demektir; patlayıcı güçte yorgunluk göstergesidir.",
        +1,
        2,
    ),
    "peak_kick_height_norm": MetricInfo(
        "Tekme Yüksekliği",
        "gövde oranı",
        "Ayağın, sporcunun gövde uzunluğuna göre ne kadar yükseldiğini gösterir.",
        "Ayak yüksekliği, sporcunun gövde uzunluğuna oranlanır. Bu yüzden kamera uzaklığı ve boy farkı daha az etkiler.",
        "Post değeri düşerse yorgunlukla kalça fleksiyonu veya teknik yükseklik korunamamış olabilir.",
        -1,
        2,
    ),
    "active_peak_foot_speed_norm": MetricInfo(
        "Ayak Hızı",
        "gövde/sn",
        "Ayağın, sporcunun gövde uzunluğuna göre ne kadar hızlı hareket ettiğini gösterir.",
        "Frame bazlı ayak konumu değişimi FPS ile hıza çevrilir ve gövde uzunluğuna oranlanır.",
        "Post değeri düşerse tekme uç hızında azalma vardır; performans yorgunluğu ile uyumludur.",
        -1,
        2,
    ),
    "duration_sec": MetricInfo(
        "Tekme Süresi",
        "sn",
        "Tekmenin başlangıçtan bitişe kadar sürdüğü toplam zamandır.",
        "Ayak yüksekliği ve diz hızı sinyallerinden bulunan tekme penceresinin süre uzunluğu hesaplanır.",
        "Post değeri artarsa hareket yavaşlamış olabilir; tek başına değil hız ve ROM ile birlikte yorumlanır.",
        +1,
        2,
    ),
    "extension_dur_sec": MetricInfo(
        "Uzatma Süresi",
        "sn",
        "Dizin en kapalı noktadan maksimum uzamaya gittiği patlayıcı fazın süresidir.",
        "Chamber anı ile dizin uzadığı extension anı arasındaki süre hesaplanır.",
        "Uzatma süresinin uzaması patlayıcı fazın yavaşladığını gösterebilir.",
        +1,
        2,
    ),
    "retraction_dur_sec": MetricInfo(
        "Geri Çekim Süresi",
        "sn",
        "Tekme uzatıldıktan sonra bacağın geri toplandığı fazın süresidir.",
        "Extension anı ile tekme bitişi arasındaki süre hesaplanır.",
        "Yorgunlukta geri çekim fazı uzayabilir; savunmaya dönüş gecikir.",
        +1,
        2,
    ),
    "extension_peak_vel_deg_s": MetricInfo(
        "Uzatma Peak Hızı",
        "derece/sn",
        "Tekmenin uzatma fazındaki en yüksek diz açısal hızıdır.",
        "Chamber-extension aralığındaki mutlak diz hızının maksimumu alınır.",
        "Düşüş, vuruş fazındaki patlayıcı kuvvetin azaldığını gösterir.",
        -1,
        0,
    ),
    "retraction_peak_vel_deg_s": MetricInfo(
        "Geri Çekim Peak Hızı",
        "derece/sn",
        "Bacağın geri toplama fazındaki en yüksek diz açısal hızıdır.",
        "Extension-bitiş aralığındaki mutlak diz hızının maksimumu alınır.",
        "Düşüş, tekmeden sonra savunma pozisyonuna dönüşün yavaşladığını gösterir.",
        -1,
        0,
    ),
    "knee_asi": MetricInfo(
        "Diz Asimetri İndeksi",
        "%",
        "Sağ ve sol diz hareket açıklığı arasındaki farkın yüzde karşılığıdır.",
        "ASI = (Sağ ROM - Sol ROM) / iki bacağın ortalaması x 100. Pozitif değer sağ taraf baskınlığını gösterir.",
        "Mutlak değer 10% üstüne çıkarsa literatürde klinik olarak anlamlı asimetri kabul edilir.",
        +1,
        1,
    ),
    "hip_asi": MetricInfo(
        "Kalça Asimetri İndeksi",
        "%",
        "Sağ ve sol kalça hareket açıklığı arasındaki farkın yüzde karşılığıdır.",
        "ASI = (Sağ kalça ROM - Sol kalça ROM) / iki taraf ortalaması x 100.",
        "Mutlak değer 10% üstüne çıkarsa yük dağılımı ve teknik simetri açısından dikkat gerektirir.",
        +1,
        1,
    ),
    "pose_confidence": MetricInfo(
        "Pose Güven Skoru",
        "0-1",
        "Modelin vücut noktalarını ne kadar güvenilir takip ettiğini gösterir.",
        "Tekme penceresindeki landmark güven skorlarının ortalaması alınır.",
        "0.60 altı tekmelerde açı ve hız hesapları gürültülü olabilir; sonuç yorumunda ihtiyatlı kullanılmalıdır.",
        -1,
        2,
    ),
}


PRIMARY_METRICS = [
    "active_peak_knee_vel_deg_s",
    "active_knee_rom_deg",
    "peak_kick_height_norm",
    "active_peak_foot_speed_norm",
    "time_to_peak_knee_vel_sec",
    "duration_sec",
]

EMG_CH1_MUSCLE = "Rectus femoris"
EMG_CH1_GROUP = "Quadriceps / ön uyluk"
EMG_CH2_MUSCLE = "Biceps femoris"
EMG_CH2_GROUP = "Hamstring / arka uyluk"
EMG_MUSCLE_GROUP = f"{EMG_CH1_GROUP} + {EMG_CH2_GROUP}"
EMG_MUSCLE_NAME = f"CH1: {EMG_CH1_MUSCLE} / CH2: {EMG_CH2_MUSCLE}"
EMG_PLACEMENT = (
    "EMG CH1 rectus femoris üzerinde, CH2 biceps femoris üzerinde değerlendirilir. "
    "Rectus femoris tekmede bacağı kaldırma ve diz açma fazını; biceps femoris diz bükme, geri çekme ve frenleme fazını takip eder."
)


def metric_info(key: str) -> MetricInfo:
    return METRICS.get(
        key,
        MetricInfo(key, "", key, "Bu metrik analiz çıktısından doğrudan alınır.", "Veri bağlamına göre yorumlanır.", -1),
    )


def fmt_value(value: float | int | None, key: str | None = None, suffix: bool = True) -> str:
    if value is None or pd.isna(value):
        return "-"
    info = metric_info(key or "")
    decimals = info.decimals if key else 1
    text = f"{float(value):.{decimals}f}"
    if not suffix or not info.unit:
        return text
    return f"{text} {info.unit}"


def _degree_symbol(value: float, decimals: int = 0) -> str:
    return f"{value:.{decimals}f}°"


def readable_metric_value(key: str, value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    v = float(value)
    if key == "peak_kick_height_norm":
        if v < -0.25:
            label = "Düşük seviye"
        elif v < 0.35:
            label = "Orta seviye"
        else:
            label = "Yüksek seviye"
        return f"{label} (oran: {v:.2f})"
    if key == "active_min_knee_angle_deg":
        if v <= 80:
            label = "Çok iyi bükülme"
        elif v <= 120:
            label = "Yeterli bükülme"
        else:
            label = "Az bükülme"
        return f"{label} ({_degree_symbol(v, 1)})"
    if key == "active_peak_knee_angle_deg":
        if v >= 165:
            label = "Tam açılmaya yakın"
        elif v >= 140:
            label = "Yeterli açılma"
        else:
            label = "Sınırlı açılma"
        return f"{label} ({_degree_symbol(v, 1)})"
    if key == "active_knee_rom_deg":
        if v >= 80:
            label = "Geniş hareket"
        elif v >= 40:
            label = "Orta hareket"
        else:
            label = "Daralmış hareket"
        return f"{label} ({_degree_symbol(v, 1)})"
    if key == "active_peak_knee_vel_deg_s":
        if v >= 700:
            label = "Çok hızlı"
        elif v >= 400:
            label = "Hızlı"
        elif v >= 200:
            label = "Orta hız"
        else:
            label = "Yavaş"
        return f"{label} ({v:.0f}°/sn)"
    if key == "active_peak_foot_speed_norm":
        if v >= 3:
            label = "Çok hızlı ayak"
        elif v >= 1.5:
            label = "Hızlı ayak"
        elif v >= 0.7:
            label = "Orta ayak hızı"
        else:
            label = "Düşük ayak hızı"
        return f"{label} ({v:.2f} gövde/sn)"
    if key == "duration_sec":
        if v <= 0.6:
            label = "Kısa/süratli"
        elif v <= 1.5:
            label = "Normal süre"
        else:
            label = "Uzun/yavaş"
        return f"{label} ({v:.2f} sn)"
    if key == "pose_confidence":
        if v >= 0.8:
            label = "Yüksek güven"
        elif v >= 0.6:
            label = "Yeterli güven"
        else:
            label = "Düşük güven"
        return f"{label} ({v:.2f}/1)"
    return fmt_value(v, key)


def mean_for(events: list[dict], key: str) -> float | None:
    return events_mean(events, key)


def change_label(key: str, pct: float | None) -> str:
    if pct is None:
        return "Yorum için yeterli veri yok."
    info = metric_info(key)
    signal = info.fatigue_direction * pct
    if key in ("knee_asi", "hip_asi"):
        if abs(pct) < 5:
            return "Asimetri değişimi sınırlı."
        return "Asimetri artmış; taraflar arası yük dağılımı izlenmeli." if pct > 0 else "Asimetri azalmış; simetri daha iyi."
    if signal >= 15:
        return "Belirgin yorgunluk göstergesi."
    if signal >= 8:
        return "Orta düzey yorgunluk göstergesi."
    if signal >= 4:
        return "Hafif yorgunluk belirtisi."
    if signal <= -5:
        return "Post performansı korunmuş veya iyileşmiş."
    return "Değişim sınırlı; stabil kabul edilebilir."


def status_badge(key: str, pct: float | None) -> str:
    if pct is None:
        return "Veri yok"
    signal = metric_info(key).fatigue_direction * pct
    if signal >= 15:
        return "Yüksek dikkat"
    if signal >= 8:
        return "Dikkat"
    if signal >= 4:
        return "Hafif değişim"
    if signal <= -5:
        return "İyileşme"
    return "Stabil"


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


def render_metric_help(keys: Iterable[str], title: str = "Metrikler nasıl okunmalı?") -> None:
    with st.expander(title, expanded=False):
        for key in keys:
            info = metric_info(key)
            weight = FATIGUE_WEIGHTS.get(key)
            weight_text = f" Yorgunluk indeksindeki ağırlığı: {weight:.0%}." if weight else ""
            st.markdown(f"**{info.label}**")
            st.markdown(f"- Ne ölçer: {info.plain}")
            st.markdown(f"- Nasıl hesaplandı: {info.calculation}")
            st.markdown(f"- Nasıl yorumlanır: {info.interpretation}{weight_text}")


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


def readable_emg_rms(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    v = float(value)
    if v >= 0.45:
        label = "Yüksek kas aktivasyonu"
    elif v >= 0.25:
        label = "Orta kas aktivasyonu"
    else:
        label = "Düşük kas aktivasyonu"
    return f"{label} ({v:.3f} mV)"


def readable_emg_frequency(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    v = float(value)
    if v >= 80:
        label = "Yorgunluk düşük"
    elif v >= 65:
        label = "Yorgunluk başlıyor"
    else:
        label = "Belirgin yorgunluk"
    return f"{label} ({v:.1f} Hz)"


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


def render_data_quality(pre_events: list[dict], post_events: list[dict]) -> None:
    low_pre = [e for e in pre_events if e.get("confidence_flag") == "low"]
    low_post = [e for e in post_events if e.get("confidence_flag") == "low"]
    if len(pre_events) < 5 or len(post_events) < 5:
        st.warning(
            f"Veri gücü uyarısı: Pre {len(pre_events)}, Post {len(post_events)} tekme var. "
            "Her oturumda en az 5-8 geçerli tekme olduğunda ortalama, yüzde değişim ve Cohen's d daha güvenilir yorumlanır."
        )
    if low_pre or low_post:
        st.warning(
            f"Pose güven uyarısı: Pre düşük güvenli tekme {len(low_pre)}, Post düşük güvenli tekme {len(low_post)}. "
            "Bu tekmelerde eklem açıları ve hızlar kamera açısı, örtüşme veya model takibi nedeniyle daha gürültülü olabilir."
        )
    if not low_pre and not low_post and len(pre_events) >= 5 and len(post_events) >= 5:
        st.success("Veri kalitesi iyi: tekme sayısı ve pose güveni temel yorum için yeterli görünüyor.")


def analysis_paragraph(pre_events: list[dict], post_events: list[dict], fi: float, fatigue_data: dict) -> str:
    main = top_findings(pre_events, post_events, fatigue_data, fi)
    return " ".join(main[:4])
