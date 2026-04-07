"""EMG synchronization utilities.

Workflow:
1. load_emg_csv()          — parse EMG file (Delsys / Noraxon / generic)
2. resample_to_video_times() — interpolate EMG onto video frame timestamps
3. compute_rms_per_kick()  — window RMS per kick event per channel
4. export_synced_frame_csv() — write frame-level merged CSV

Assumptions:
- EMG CSV has one time column (seconds) and N channel columns.
- EMG sample rate is typically much higher than video FPS (1000–2000 Hz vs 60 Hz).
- Synchronisation is done via common time base: both signals start at t=0.
  If a hardware trigger offset is known it can be passed as `emg_time_offset_sec`.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_emg_csv(
    path: str | Path,
    time_col: str | None = None,
    channels: list[str] | None = None,
    delimiter: str = ",",
    skip_rows: int = 0,
) -> dict:
    """Load an EMG CSV file.

    Auto-detects the time column if *time_col* is None (first column whose
    header contains 'time' or 't' case-insensitively, otherwise column 0).

    Args:
        path: Path to the CSV file.
        time_col: Name of the time column (seconds). Auto-detected if None.
        channels: List of channel column names to load. Loads all numeric
                  columns except the time column if None.
        delimiter: CSV delimiter character.
        skip_rows: Number of header/metadata rows to skip before the column
                   header row (common in Delsys / Noraxon exports).

    Returns:
        {
          "time_s": list[float],
          "channels": {channel_name: list[float]},
          "sample_rate_hz": float,
          "path": str,
        }
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"EMG dosyası bulunamadı: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for _ in range(skip_rows):
            f.readline()
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = list(reader)

    if not rows:
        raise ValueError("EMG CSV boş veya okunamadı.")

    headers = list(rows[0].keys())

    # Auto-detect time column
    if time_col is None:
        for h in headers:
            if h.strip().lower() in ("time", "time_s", "time_sec", "t", "timestamp"):
                time_col = h
                break
        if time_col is None:
            time_col = headers[0]

    if time_col not in headers:
        raise ValueError(f"Zaman sütunu bulunamadı: '{time_col}'. Sütunlar: {headers}")

    # Auto-detect channel columns
    if channels is None:
        channels = []
        for h in headers:
            if h == time_col:
                continue
            # Try to parse first row to check if numeric
            raw = rows[0].get(h, "")
            try:
                float(raw.strip())
                channels.append(h)
            except (ValueError, AttributeError):
                pass

    if not channels:
        raise ValueError("Hiç EMG kanalı bulunamadı.")

    def _parse(val: str) -> float:
        try:
            return float(val.strip())
        except (ValueError, AttributeError):
            return float("nan")

    time_s = [_parse(row[time_col]) for row in rows]
    channel_data: dict[str, list[float]] = {ch: [_parse(row.get(ch, "nan")) for row in rows] for ch in channels}

    # Estimate sample rate
    sample_rate = _estimate_sample_rate(time_s)

    return {
        "time_s": time_s,
        "channels": channel_data,
        "sample_rate_hz": sample_rate,
        "path": str(path),
    }


def _estimate_sample_rate(time_s: list[float]) -> float:
    if len(time_s) < 2:
        return 0.0
    diffs = [time_s[i + 1] - time_s[i] for i in range(min(100, len(time_s) - 1)) if not math.isnan(time_s[i + 1] - time_s[i])]
    if not diffs:
        return 0.0
    mean_dt = sum(diffs) / len(diffs)
    return 1.0 / mean_dt if mean_dt > 1e-9 else 0.0


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_to_video_times(
    emg_data: dict,
    video_times: Sequence[float],
    emg_time_offset_sec: float = 0.0,
) -> dict:
    """Interpolate EMG channels onto video frame timestamps.

    Args:
        emg_data: Output of load_emg_csv().
        video_times: List of video frame timestamps in seconds.
        emg_time_offset_sec: Shift EMG time axis by this amount (positive =
            EMG starts later than video). Useful when a sync trigger marks
            the start of recording independently.

    Returns:
        {channel_name: list[float | None]} — one value per video frame.
        None where the video timestamp falls outside the EMG recording range.
    """
    emg_t = [t + emg_time_offset_sec for t in emg_data["time_s"]]
    t_min, t_max = min(emg_t), max(emg_t)

    result: dict[str, list[float | None]] = {}

    for ch, values in emg_data["channels"].items():
        resampled: list[float | None] = []
        for vt in video_times:
            if vt < t_min or vt > t_max:
                resampled.append(None)
            else:
                resampled.append(_lerp(emg_t, values, vt))
        result[ch] = resampled

    return result


def _lerp(xs: list[float], ys: list[float], x_query: float) -> float:
    """Linear interpolation at x_query given sorted xs and corresponding ys."""
    # Binary search for insertion point
    lo, hi = 0, len(xs) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] < x_query:
            lo = mid + 1
        else:
            hi = mid

    if lo == 0:
        return ys[0]
    if lo >= len(xs):
        return ys[-1]

    x0, x1 = xs[lo - 1], xs[lo]
    y0, y1 = ys[lo - 1], ys[lo]
    if abs(x1 - x0) < 1e-12:
        return y0
    t = (x_query - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


# ---------------------------------------------------------------------------
# Per-kick windowing
# ---------------------------------------------------------------------------

def compute_rms_per_kick(
    emg_resampled: dict[str, list[float | None]],
    kick_events: list[dict],
    video_fps: float,
) -> list[dict]:
    """Compute RMS and mean-absolute for each EMG channel within each kick window.

    Args:
        emg_resampled: Output of resample_to_video_times().
        kick_events: List of kick event dicts (must have start_frame / end_frame).
        video_fps: Used only for reference; frame indices come from kick_events.

    Returns:
        List of dicts, one per kick event, containing:
            kick_id, start_frame, end_frame, duration_sec,
            {channel}_rms, {channel}_mean_abs  for each channel.
    """
    results = []
    for ev in kick_events:
        start = int(ev["start_frame"])
        end = int(ev["end_frame"])
        duration = float(ev.get("duration_sec", (end - start + 1) / video_fps))

        row: dict = {
            "kick_id": ev.get("kick_id"),
            "start_frame": start,
            "end_frame": end,
            "duration_sec": round(duration, 4),
        }

        for ch, values in emg_resampled.items():
            window = [v for v in values[start : end + 1] if v is not None and not math.isnan(v)]
            if window:
                rms = math.sqrt(sum(v * v for v in window) / len(window))
                mean_abs = sum(abs(v) for v in window) / len(window)
                row[f"{ch}_rms"] = round(rms, 6)
                row[f"{ch}_mean_abs"] = round(mean_abs, 6)
            else:
                row[f"{ch}_rms"] = None
                row[f"{ch}_mean_abs"] = None

        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_synced_frame_csv(
    path: str | Path,
    frame_rows: list[dict],
    emg_resampled: dict[str, list[float | None]],
) -> None:
    """Write a merged CSV with frame-level video metrics + resampled EMG values.

    Args:
        path: Output CSV file path.
        frame_rows: Frame-level video metrics (from pipeline.run_analysis).
        emg_resampled: Output of resample_to_video_times() — same length as frame_rows.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    merged: list[dict] = []
    for i, row in enumerate(frame_rows):
        merged_row = dict(row)
        for ch, values in emg_resampled.items():
            val = values[i] if i < len(values) else None
            merged_row[f"emg_{ch}"] = None if val is None or math.isnan(val) else round(val, 6)
        merged.append(merged_row)

    if not merged:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(merged[0].keys())
    seen = set(fieldnames)
    for row in merged:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)


def export_kick_emg_csv(
    path: str | Path,
    kick_emg_rows: list[dict],
) -> None:
    """Write per-kick EMG stats CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not kick_emg_rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(kick_emg_rows[0].keys())
    seen = set(fieldnames)
    for row in kick_emg_rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kick_emg_rows)
