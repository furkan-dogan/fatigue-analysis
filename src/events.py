"""Kick/movement event detection utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np


JOINT_KEYS = [
    "R_SHOULDER",
    "L_SHOULDER",
    "R_ELBOW",
    "L_ELBOW",
    "R_HIP",
    "L_HIP",
    "R_KNEE",
    "L_KNEE",
    "R_ANKLE",
    "L_ANKLE",
]


def _fill_missing(values: Sequence[float | None], default: float = 0.0) -> list[float]:
    if not values:
        return []

    out: list[float] = [default] * len(values)
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return out

    first = valid[0]
    last = first
    for i, v in enumerate(values):
        if v is None:
            out[i] = last
        else:
            last = float(v)
            out[i] = last

    for i, v in enumerate(values):
        if v is None:
            out[i] = first
        else:
            break
    return out


def _moving_average(values: Sequence[float], window: int = 7) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if len(arr) < window or window <= 1:
        return arr
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(arr, kernel, mode="same")


def _local_maxima(values: np.ndarray, threshold: float) -> list[int]:
    peaks: list[int] = []
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] >= values[i + 1] and values[i] >= threshold:
            peaks.append(i)
    return peaks


def _nmax(a: float | None, b: float | None) -> float | None:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _window_stats(series: Sequence[float | None], start: int, end: int) -> tuple[float | None, float | None]:
    vals = [float(v) for v in series[start : end + 1] if v is not None]
    if not vals:
        return None, None
    mean_val = float(np.mean(vals))
    rom_val = float(np.max(vals) - np.min(vals))
    return mean_val, rom_val


def detect_movement_events(
    right_knee_angles: Sequence[float | None],
    left_knee_angles: Sequence[float | None],
    right_kick_heights: Sequence[float | None],
    left_kick_heights: Sequence[float | None],
    joint_series: dict[str, Sequence[float | None]],
    fps: float,
    min_peak_prominence_norm: float = 0.06,
    min_distance_sec: float = 0.4,
    min_duration_sec: float = 0.15,
    max_duration_sec: float = 6.0,
    velocity_series: dict[str, Sequence[float | None]] | None = None,
    foot_speed_series: dict[str, Sequence[float | None]] | None = None,
    min_knee_rom_deg: float = 20.0,
    min_peak_kick_height_norm: float = -0.3,
    confidence_series: Sequence[float | None] | None = None,
) -> list[dict[str, float | int | str | None]]:
    """Detect multi-kick/movement events and return per-event metrics.

    Kick validation criteria (applied after initial peak detection):
    - min_knee_rom_deg: active knee must show at least this much ROM.
      Filters out weight-shifts and stance changes (typically <5°).
    - min_peak_kick_height_norm: peak smoothed height must be above this.
      Filters out events where the foot never actually rises toward hip level.
      Typical kicking height is >0; set to -0.3 to allow low kicks.
    """
    n = max(len(right_knee_angles), len(left_knee_angles), len(right_kick_heights), len(left_kick_heights))
    if n == 0:
        return []

    rk = _fill_missing(right_knee_angles, default=180.0)
    lk = _fill_missing(left_knee_angles, default=180.0)
    rh = list(right_kick_heights)
    lh = list(left_kick_heights)
    active_height_raw = [_nmax(rh[i] if i < len(rh) else None, lh[i] if i < len(lh) else None) for i in range(n)]
    active_height = _fill_missing(active_height_raw, default=0.0)
    smoothed = _moving_average(active_height, window=7)

    baseline = float(np.percentile(smoothed, 40))
    top = float(np.percentile(smoothed, 95))
    if (top - baseline) < min_peak_prominence_norm:
        return []

    peak_threshold = baseline + min_peak_prominence_norm
    candidates = _local_maxima(smoothed, threshold=peak_threshold)
    if not candidates:
        return []

    min_distance = max(1, int(min_distance_sec * fps))
    min_frames = max(1, int(min_duration_sec * fps))
    max_frames = max(min_frames, int(max_duration_sec * fps))

    # Keep strongest peaks first with distance suppression.
    selected: list[int] = []
    for p in sorted(candidates, key=lambda x: smoothed[x], reverse=True):
        if any(abs(p - s) < min_distance for s in selected):
            continue
        selected.append(p)
    selected = sorted(selected)

    events: list[dict[str, float | int | str | None]] = []
    for kick_id, p in enumerate(selected, start=1):
        amplitude = float(smoothed[p] - baseline)
        if amplitude < min_peak_prominence_norm:
            continue

        enter_threshold = baseline + amplitude * 0.25
        start = p
        while start > 0 and smoothed[start] >= enter_threshold:
            start -= 1
        if smoothed[start] < enter_threshold and start < p:
            start += 1

        end = p
        while end < n - 1 and smoothed[end] >= enter_threshold:
            end += 1
        if smoothed[end] < enter_threshold and end > p:
            end -= 1

        duration = end - start + 1
        if duration < min_frames or duration > max_frames:
            continue

        # ── Quick pre-checks before computing full metrics ────────────────
        # 1. Peak height: foot must actually rise toward hip level
        if float(smoothed[p]) < min_peak_kick_height_norm:
            continue

        rh_p = rh[p] if p < len(rh) else None
        lh_p = lh[p] if p < len(lh) else None
        if rh_p is None and lh_p is None:
            active_leg = "UNK"
            active_knee_series = rk
        elif lh_p is None or (rh_p is not None and rh_p >= lh_p):
            active_leg = "R"
            active_knee_series = rk
        else:
            active_leg = "L"
            active_knee_series = lk

        active_slice = active_knee_series[start : end + 1]
        peak_knee = float(np.max(active_slice))
        min_knee = float(np.min(active_slice))
        knee_rom = peak_knee - min_knee

        # 2. Knee ROM: must show real flexion→extension cycle
        if knee_rom < min_knee_rom_deg:
            continue

        row: dict[str, float | int | str | None] = {
            "kick_id": kick_id,
            "active_leg": active_leg,
            "start_frame": int(start),
            "peak_frame": int(p),
            "end_frame": int(end),
            "start_time_sec": round(start / fps, 4),
            "peak_time_sec": round(p / fps, 4),
            "end_time_sec": round(end / fps, 4),
            "duration_sec": round(duration / fps, 4),
            "peak_kick_height_norm": round(float(smoothed[p]), 4),
            "active_peak_knee_angle_deg": round(peak_knee, 4),
            "active_min_knee_angle_deg": round(min_knee, 4),
            "active_knee_rom_deg": round(knee_rom, 4),
        }

        for key in JOINT_KEYS:
            mean_val, rom_val = _window_stats(joint_series.get(key, []), start, end)
            row[f"{key}_mean"] = None if mean_val is None else round(mean_val, 4)
            row[f"{key}_rom"] = None if rom_val is None else round(rom_val, 4)

        # --- Velocity metrics per kick ---
        if velocity_series:
            vel_key = "R_KNEE_vel" if active_leg in ("R", "UNK") else "L_KNEE_vel"
            vel_seq = velocity_series.get(vel_key, [])
            if vel_seq:
                abs_vel_window = [
                    abs(float(v))
                    for v in vel_seq[start : end + 1]
                    if v is not None
                ]
                if abs_vel_window:
                    row["active_peak_knee_vel_deg_s"] = round(max(abs_vel_window), 4)
                    row["active_mean_knee_vel_deg_s"] = round(float(np.mean(abs_vel_window)), 4)
                    # time-to-peak-velocity: explosiveness indicator
                    peak_offset = int(np.argmax(abs_vel_window))
                    row["time_to_peak_knee_vel_sec"] = round(peak_offset / fps, 4)
                else:
                    row["active_peak_knee_vel_deg_s"] = None
                    row["active_mean_knee_vel_deg_s"] = None
                    row["time_to_peak_knee_vel_sec"] = None
            else:
                row["active_peak_knee_vel_deg_s"] = None
                row["active_mean_knee_vel_deg_s"] = None
                row["time_to_peak_knee_vel_sec"] = None

        # --- Foot speed metrics per kick ---
        if foot_speed_series:
            spd_key = "R_FOOT_speed" if active_leg in ("R", "UNK") else "L_FOOT_speed"
            spd_seq = foot_speed_series.get(spd_key, [])
            if spd_seq:
                spd_window = [
                    float(v) for v in spd_seq[start : end + 1] if v is not None
                ]
                if spd_window:
                    row["active_peak_foot_speed_norm"] = round(max(spd_window), 4)
                    row["active_mean_foot_speed_norm"] = round(float(np.mean(spd_window)), 4)
                else:
                    row["active_peak_foot_speed_norm"] = None
                    row["active_mean_foot_speed_norm"] = None
            else:
                row["active_peak_foot_speed_norm"] = None
                row["active_mean_foot_speed_norm"] = None

        # --- Kick phase segmentation ---
        # chamber_frame: max flexion (min knee angle) — end of loading
        # extension_frame: max extension (max knee angle after chamber) — impact point
        chamber_offset = int(np.argmin(active_slice))
        chamber_frame  = start + chamber_offset
        post_chamber   = active_slice[chamber_offset:]
        ext_offset     = int(np.argmax(post_chamber))
        extension_frame = chamber_frame + ext_offset

        loading_dur    = (chamber_frame   - start)          / fps
        extension_dur  = (extension_frame - chamber_frame)  / fps
        retraction_dur = (end             - extension_frame) / fps

        row["chamber_frame"]     = int(chamber_frame)
        row["chamber_time_sec"]  = round(chamber_frame / fps, 4)
        row["extension_frame"]   = int(extension_frame)
        row["extension_time_sec"] = round(extension_frame / fps, 4)
        row["loading_dur_sec"]    = round(loading_dur, 4)
        row["extension_dur_sec"]  = round(extension_dur, 4)
        row["retraction_dur_sec"] = round(retraction_dur, 4)

        # Phase velocities (peak absolute knee velocity within each phase window)
        if velocity_series:
            vel_key = "R_KNEE_vel" if active_leg in ("R", "UNK") else "L_KNEE_vel"
            vel_seq = velocity_series.get(vel_key, [])
            if vel_seq:
                def _phase_peak_vel(f0: int, f1: int) -> float | None:
                    w = [abs(float(v)) for v in vel_seq[f0:f1 + 1] if v is not None]
                    return round(max(w), 4) if w else None

                row["loading_peak_vel_deg_s"]    = _phase_peak_vel(start,          chamber_frame)
                row["extension_peak_vel_deg_s"]  = _phase_peak_vel(chamber_frame,  extension_frame)
                row["retraction_peak_vel_deg_s"] = _phase_peak_vel(extension_frame, end)
            else:
                row["loading_peak_vel_deg_s"] = row["extension_peak_vel_deg_s"] = row["retraction_peak_vel_deg_s"] = None
        else:
            row["loading_peak_vel_deg_s"] = row["extension_peak_vel_deg_s"] = row["retraction_peak_vel_deg_s"] = None

        # --- Bilateral Asymmetry Index (ROM-based) ---
        r_knee_rom = row.get("R_KNEE_rom")
        l_knee_rom = row.get("L_KNEE_rom")
        r_hip_rom  = row.get("R_HIP_rom")
        l_hip_rom  = row.get("L_HIP_rom")

        def _asi(r: float | None, l: float | None) -> float | None:
            if r is None or l is None:
                return None
            denom = (r + l) / 2.0
            return round((r - l) / denom * 100.0, 2) if abs(denom) > 1e-8 else None

        row["knee_asi"] = _asi(r_knee_rom, l_knee_rom)
        row["hip_asi"]  = _asi(r_hip_rom,  l_hip_rom)

        # --- Pose confidence for this kick window ---
        if confidence_series:
            conf_window = [float(v) for v in confidence_series[start:end + 1] if v is not None]
            if conf_window:
                row["pose_confidence"] = round(float(np.mean(conf_window)), 4)
                row["confidence_flag"] = "low" if row["pose_confidence"] < 0.6 else "ok"
            else:
                row["pose_confidence"] = None
                row["confidence_flag"] = "unknown"
        else:
            row["pose_confidence"] = None
            row["confidence_flag"] = "unknown"

        events.append(row)

    # Merge overlapping segments by taking higher peak height.
    if not events:
        return []
    merged: list[dict[str, float | int | str | None]] = []
    for ev in sorted(events, key=lambda x: int(x["start_frame"])):
        if not merged:
            merged.append(ev)
            continue
        last = merged[-1]
        if int(ev["start_frame"]) <= int(last["end_frame"]):
            if float(ev["peak_kick_height_norm"]) > float(last["peak_kick_height_norm"]):
                merged[-1] = ev
        else:
            merged.append(ev)

    # Re-number kick ids after merge.
    for i, ev in enumerate(merged, start=1):
        ev["kick_id"] = i
    return merged
