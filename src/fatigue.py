"""Fatigue index computation from kick event metrics."""
from __future__ import annotations

from src.utils import events_mean

FATIGUE_METRICS: dict[str, tuple[str, str, int]] = {
    "active_knee_rom_deg":         ("Aktif Diz ROM",          "°",   -1),
    "active_peak_knee_vel_deg_s":  ("Peak Diz Hızı",          "°/s", -1),
    "active_mean_knee_vel_deg_s":  ("Ort. Diz Hızı",          "°/s", -1),
    "time_to_peak_knee_vel_sec":   ("Peak Hıza Süre",         "sn",  +1),
    "peak_kick_height_norm":       ("Peak Tekme Yüksekliği",  "",    -1),
    "active_peak_foot_speed_norm": ("Peak Ayak Hızı",         "t/s", -1),
    "duration_sec":                ("Tekme Süresi",           "sn",  +1),
    "R_HIP_rom":                   ("Sağ Kalça ROM",          "°",   -1),
    "L_HIP_rom":                   ("Sol Kalça ROM",          "°",   -1),
    "R_ANKLE_rom":                 ("Sağ Ayak Bileği ROM",    "°",   -1),
}

FATIGUE_WEIGHTS: dict[str, float] = {
    "active_knee_rom_deg":         0.25,
    "active_peak_knee_vel_deg_s":  0.25,
    "time_to_peak_knee_vel_sec":   0.15,
    "peak_kick_height_norm":       0.15,
    "active_peak_foot_speed_norm": 0.10,
    "duration_sec":                0.05,
    "active_mean_knee_vel_deg_s":  0.05,
}


def compute_fatigue(pre_events: list[dict], post_events: list[dict]) -> dict:
    """Return per-metric deltas + composite fatigue index (0–100)."""
    results: dict = {}
    weighted_sum = 0.0
    weight_total = 0.0

    for key, (label, unit, direction) in FATIGUE_METRICS.items():
        pre_v  = events_mean(pre_events,  key)
        post_v = events_mean(post_events, key)
        if pre_v is None or post_v is None or abs(pre_v) < 1e-8:
            results[key] = dict(label=label, unit=unit, direction=direction,
                                pre=pre_v, post=post_v, delta=None, pct=None,
                                fatigue_contribution=None)
            continue

        delta        = post_v - pre_v
        pct          = delta / abs(pre_v) * 100.0
        contribution = float(direction * pct)
        contribution = max(-100.0, min(100.0, contribution))

        w = FATIGUE_WEIGHTS.get(key, 0.0)
        if w > 0:
            weighted_sum += contribution * w
            weight_total += w

        results[key] = dict(label=label, unit=unit, direction=direction,
                            pre=pre_v, post=post_v, delta=delta, pct=pct,
                            fatigue_contribution=contribution)

    composite     = (weighted_sum / weight_total) if weight_total > 0 else 0.0
    fatigue_index = max(0.0, min(100.0, (composite + 100.0) / 2.0))
    return {"metrics": results, "fatigue_index": fatigue_index, "composite_raw": composite}
