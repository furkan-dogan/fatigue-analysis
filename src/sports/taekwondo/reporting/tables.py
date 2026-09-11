from __future__ import annotations

import pandas as pd
from src.sports.taekwondo.reporting.catalog import EMG_CH1_MUSCLE, EMG_CH2_MUSCLE
from src.sports.taekwondo.reporting.formatting import fmt_value, metric_info


def _fatigue_metric_rows(fatigue_data: dict) -> list[dict]:
    rows_ft = []
    for key, m in fatigue_data["metrics"].items():
        if m["pre"] is None or m["post"] is None or m["pct"] is None:
            continue
        info = metric_info(key)
        is_fatigue = (m["direction"] * m["pct"]) > 0
        fatigue_text = "🔴 Evet" if is_fatigue else "🟢 Hayır"
        contribution_text = f"{'🔴' if m['fatigue_contribution'] > 0 else '🟢'} {m['fatigue_contribution']:+.1f}"
        rows_ft.append({
            "Metrik":      info.label,
            "Pre":         fmt_value(m["pre"], key),
            "Post":        fmt_value(m["post"], key),
            "Değişim":     f"{m['pct']:+.1f}%",
            "Yorgunluk Bulgusu": fatigue_text,
            "Katkı Puanı": contribution_text,
            "Yorum":       "Bu değişim yorgunluk yönünde." if is_fatigue else "Bu değişim yorgunluk yönünde değil veya iyileşme gösteriyor.",
        })
    return rows_ft


def _emg_frame_export_rows(
    frame_df: pd.DataFrame,
    emg_rows: list[dict],
    events: list[dict],
    session_label: str,
) -> list[dict]:
    if frame_df.empty or not emg_rows:
        return []

    event_by_frame: dict[int, dict] = {}
    phase_by_frame: dict[int, str] = {}
    for ev in events:
        start = int(ev.get("start_frame", 0))
        peak = int(ev.get("peak_frame", start))
        end = int(ev.get("end_frame", peak))
        chamber = int(ev.get("chamber_frame", start))
        extension = int(ev.get("extension_frame", peak))
        for frame_idx in range(max(0, start), max(start, end) + 1):
            event_by_frame[frame_idx] = ev
            if frame_idx < chamber:
                phase_by_frame[frame_idx] = "hazırlık"
            elif frame_idx < extension:
                phase_by_frame[frame_idx] = "bacağı kaldırma"
            elif frame_idx < peak:
                phase_by_frame[frame_idx] = "vuruşa hızlanma"
            elif frame_idx == peak:
                phase_by_frame[frame_idx] = "vuruş zirvesi"
            else:
                phase_by_frame[frame_idx] = "geri çekme"

    rows: list[dict] = []
    frame_count = min(len(frame_df), len(emg_rows))
    for frame_idx in range(frame_count):
        frame = frame_df.iloc[frame_idx]
        emg = emg_rows[frame_idx]
        ev = event_by_frame.get(frame_idx)
        rows.append(
            {
                "Oturum": session_label,
                "frame_idx": frame_idx,
                "video_time_sec": round(float(frame.get("time_sec", emg.get("time_sec", 0))), 4),
                "emg_time_sec": round(float(emg.get("time_sec", frame.get("time_sec", 0))), 4),
                "kick_id": "" if ev is None else f"T{int(ev['kick_id'])}",
                "kick_phase": "" if ev is None else phase_by_frame.get(frame_idx, "tekme penceresi"),
                "kick_start_time_sec": "" if ev is None else float(ev.get("start_time_sec", 0)),
                "kick_peak_time_sec": "" if ev is None else float(ev.get("peak_time_sec", 0)),
                "kick_end_time_sec": "" if ev is None else float(ev.get("end_time_sec", 0)),
                "CH1_Kas": EMG_CH1_MUSCLE,
                "CH1_RectusFemoris_mV": round(float(emg["EMG_RMS_mV"]), 4),
                "CH1_RectusFemoris_median_freq_Hz": round(float(emg.get("EMG_CH1_median_freq_Hz", emg["EMG_median_freq_Hz"])), 2),
                "CH2_Kas": EMG_CH2_MUSCLE,
                "CH2_BicepsFemoris_mV": round(float(emg["EMG_CH2_RMS_mV"]), 4),
                "CH2_BicepsFemoris_median_freq_Hz": round(float(emg.get("EMG_CH2_median_freq_Hz", emg["EMG_median_freq_Hz"])), 2),
                "EMG_ortalama_median_freq_Hz": round(float(emg["EMG_median_freq_Hz"]), 2),
                "active_knee_angle_deg": frame.get("active_knee_angle_deg", ""),
                "active_knee_vel_deg_s": frame.get("active_knee_vel_deg_s", ""),
                "active_foot_y_norm": frame.get("active_foot_y_norm", ""),
            }
        )
    return rows
