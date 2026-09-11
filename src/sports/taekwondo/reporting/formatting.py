"""Legacy taekwondo report data; rules remain unvalidated."""
from __future__ import annotations

import pandas as pd
from src.core.numeric import events_mean
from src.sports.taekwondo.reporting.catalog import METRICS, MetricInfo


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
