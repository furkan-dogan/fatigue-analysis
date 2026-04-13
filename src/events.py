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


def _moving_average(values: Sequence[float], window: int = 3) -> np.ndarray:
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


def _vel_peaks(
    vel_seq: Sequence[float | None],
    fps: float,
    min_distance: int,
    min_vel_threshold: float,
) -> list[int]:
    """Find peaks in absolute knee velocity — used as supplementary kick candidates."""
    abs_vel = np.array([abs(float(v)) if v is not None else 0.0 for v in vel_seq])
    smoothed = _moving_average(abs_vel, window=max(3, int(fps * 0.05)))
    peaks = _local_maxima(smoothed, threshold=min_vel_threshold)
    # distance suppression
    selected: list[int] = []
    for p in sorted(peaks, key=lambda x: smoothed[x], reverse=True):
        if any(abs(p - s) < min_distance for s in selected):
            continue
        selected.append(p)
    return sorted(selected)


def detect_movement_events(
    right_knee_angles: Sequence[float | None],
    left_knee_angles: Sequence[float | None],
    right_kick_heights: Sequence[float | None],
    left_kick_heights: Sequence[float | None],
    joint_series: dict[str, Sequence[float | None]],
    fps: float,
    min_peak_prominence_norm: float = 0.06,
    min_distance_sec: float = 0.25,
    min_duration_sec: float = 0.10,
    max_duration_sec: float = 6.0,
    velocity_series: dict[str, Sequence[float | None]] | None = None,
    foot_speed_series: dict[str, Sequence[float | None]] | None = None,
    min_knee_rom_deg: float = 12.0,
    min_peak_kick_height_norm: float = -0.5,
    confidence_series: Sequence[float | None] | None = None,
    vel_assist_threshold: float = 80.0,
) -> list[dict[str, float | int | str | None]]:
    """Detect multi-kick/movement events and return per-event metrics.

    Detection uses two complementary signals:
    1. Normalized foot height peaks (primary) — reliable for high/mid kicks
    2. Knee angular velocity peaks (supplementary) — catches fast kicks where
       foot height peak is too brief to survive smoothing

    Kick validation:
    - min_knee_rom_deg: knee must flex→extend at least this much
    - min_peak_kick_height_norm: foot must rise to at least this height
    - vel_assist_threshold: min peak knee velocity (°/s) for velocity-assisted candidates
    """
    n = max(len(right_knee_angles), len(left_knee_angles), len(right_kick_heights), len(left_kick_heights))
    if n == 0:
        return []

    rk = _fill_missing(right_knee_angles, default=180.0)
    lk = _fill_missing(left_knee_angles, default=180.0)
    rh = list(right_kick_heights)
    lh = list(left_kick_heights)
    active_height_raw = [_nmax(rh[i] if i < len(rh) else None, lh[i] if i < len(lh) else None) for i in range(n)]

    # Gap-fill: when pose is lost for a short burst (spinning, occlusion),
    # hold the last valid height instead of dropping to 0.
    # Max gap = 0.6s — longer gaps are real "foot down" moments.
    max_gap_frames = max(1, int(fps * 0.6))
    active_height_gapfilled: list[float] = []
    last_valid: float = 0.0
    gap_count: int = 0
    for v in active_height_raw:
        if v is not None:
            active_height_gapfilled.append(float(v))
            last_valid = float(v)
            gap_count = 0
        else:
            gap_count += 1
            if gap_count <= max_gap_frames:
                active_height_gapfilled.append(last_valid)
            else:
                active_height_gapfilled.append(0.0)

    active_height = active_height_gapfilled

    # FPS-adaptive smoothing: ~2 frames at any FPS — preserves fast-kick peaks
    smooth_win = max(3, int(fps * 0.07))
    if smooth_win % 2 == 0:
        smooth_win += 1
    smoothed = _moving_average(active_height, window=smooth_win)

    # 15th percentile is a more robust floor for kick-heavy videos
    # (40th was too high when most frames are active kicks).
    baseline = float(np.percentile(smoothed, 15))
    top = float(np.percentile(smoothed, 95))

    peak_threshold = baseline + min_peak_prominence_norm
    candidates: list[int] = []
    if (top - baseline) >= min_peak_prominence_norm:
        candidates = _local_maxima(smoothed, threshold=peak_threshold)

    # Dual-peak: also search on lightly-smoothed (win=2) signal.
    # Fast kicks (4-6 frames) survive minimal smoothing but get flattened
    # by the adaptive window above — this catches what the above misses.
    lightly_smoothed = _moving_average(active_height, window=2)
    fast_candidates = _local_maxima(lightly_smoothed, threshold=peak_threshold * 0.85)
    for fc in fast_candidates:
        if not any(abs(fc - c) < 2 for c in candidates):
            candidates.append(fc)

    min_distance = max(1, int(min_distance_sec * fps))
    min_frames = max(1, int(min_duration_sec * fps))
    max_frames = max(min_frames, int(max_duration_sec * fps))

    # ── Velocity-assisted candidates (co-primary) ────────────────────────────
    # Knee angular velocity is equally reliable for fast kicks — treat it as
    # a co-primary signal. Distance suppression below handles deduplication.
    if velocity_series:
        for vel_key in ("R_KNEE_vel", "L_KNEE_vel"):
            vel_seq = velocity_series.get(vel_key, [])
            if vel_seq:
                vel_cands = _vel_peaks(vel_seq, fps, min_distance, vel_assist_threshold)
                candidates.extend(vel_cands)

    if not candidates:
        return []

    # Keep strongest height peaks first with distance suppression
    selected: list[int] = []
    for p in sorted(candidates, key=lambda x: float(smoothed[x]), reverse=True):
        if any(abs(p - s) < min_distance for s in selected):
            continue
        selected.append(p)
    selected = sorted(selected)

    events: list[dict[str, float | int | str | None]] = []
    for kick_id, p in enumerate(selected, start=1):
        amplitude = float(smoothed[p] - baseline)

        # Low enter_threshold so back-to-back kicks don't bleed into each other.
        # 0.10 × amplitude means an event ends when the signal is still 10% above
        # baseline — leaving a clear gap between consecutive rapid kicks.
        enter_threshold = baseline + max(amplitude * 0.10, min_peak_prominence_norm * 0.2)

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

        # Expand window slightly for very fast kicks (< min_frames)
        duration = end - start + 1
        if duration < min_frames:
            expand = (min_frames - duration) // 2
            start = max(0, start - expand)
            end = min(n - 1, end + expand)
            duration = end - start + 1

        if duration < min_frames or duration > max_frames:
            continue

        # ── Peak height check ─────────────────────────────────────────────────
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

        # For velocity-assisted candidates: try both legs, pick one with higher ROM
        if active_leg == "UNK" and velocity_series:
            r_slice = rk[start : end + 1]
            l_slice = lk[start : end + 1]
            r_rom = float(np.max(r_slice) - np.min(r_slice))
            l_rom = float(np.max(l_slice) - np.min(l_slice))
            if l_rom > r_rom:
                active_leg = "L"
                active_knee_series = lk
            else:
                active_leg = "R"
                active_knee_series = rk

        active_slice = active_knee_series[start : end + 1]
        peak_knee = float(np.max(active_slice))
        min_knee = float(np.min(active_slice))
        knee_rom = peak_knee - min_knee

        # ── Knee ROM check ────────────────────────────────────────────────────
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
        chamber_offset = int(np.argmin(active_slice))
        chamber_frame  = start + chamber_offset
        post_chamber   = active_slice[chamber_offset:]
        ext_offset     = int(np.argmax(post_chamber))
        extension_frame = chamber_frame + ext_offset

        loading_dur    = (chamber_frame   - start)          / fps
        extension_dur  = (extension_frame - chamber_frame)  / fps
        retraction_dur = (end             - extension_frame) / fps

        row["chamber_frame"]      = int(chamber_frame)
        row["chamber_time_sec"]   = round(chamber_frame / fps, 4)
        row["extension_frame"]    = int(extension_frame)
        row["extension_time_sec"] = round(extension_frame / fps, 4)
        row["loading_dur_sec"]    = round(loading_dur, 4)
        row["extension_dur_sec"]  = round(extension_dur, 4)
        row["retraction_dur_sec"] = round(retraction_dur, 4)

        # Phase velocities
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

        # --- Bilateral Asymmetry Index ---
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

        # --- Pose confidence ---
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

    # Merge overlapping segments
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

    for i, ev in enumerate(merged, start=1):
        ev["kick_id"] = i
    return merged
