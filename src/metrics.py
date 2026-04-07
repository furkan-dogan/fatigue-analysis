"""Metric and geometry helpers for taekwondo kick analysis."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from src.pose_runner import Keypoints2D

# Joints for which velocity/acceleration are computed
VELOCITY_JOINT_KEYS = ["R_KNEE", "L_KNEE", "R_HIP", "L_HIP", "R_ANKLE", "L_ANKLE"]


def calculate_angle(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    """Return angle ABC in degrees."""
    a_np = np.array(a, dtype=float)
    b_np = np.array(b, dtype=float)
    c_np = np.array(c, dtype=float)

    ba = a_np - b_np
    bc = c_np - b_np

    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cosine_angle = float(np.dot(ba, bc) / denom)
    cosine_angle = float(np.clip(cosine_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine_angle)))


def summarize_knee_angles(knee_angles: Iterable[float]) -> dict[str, float | int] | None:
    """Return simple summary metrics for right-knee angle stream."""
    values = list(knee_angles)
    if not values:
        return None

    return {
        "count": len(values),
        "max": max(values),
        "min": min(values),
    }


def calculate_joint_angles(keypoints: Keypoints2D) -> dict[str, float | None]:
    def safe_angle(
        a: tuple[float, float] | None,
        b: tuple[float, float] | None,
        c: tuple[float, float] | None,
    ) -> float | None:
        if a is None or b is None or c is None:
            return None
        return calculate_angle(a, b, c)

    return {
        "R_SHOULDER": safe_angle(keypoints.right_hip, keypoints.right_shoulder, keypoints.right_elbow),
        "L_SHOULDER": safe_angle(keypoints.left_hip, keypoints.left_shoulder, keypoints.left_elbow),
        "R_ELBOW": safe_angle(keypoints.right_shoulder, keypoints.right_elbow, keypoints.right_wrist),
        "L_ELBOW": safe_angle(keypoints.left_shoulder, keypoints.left_elbow, keypoints.left_wrist),
        "R_HIP": safe_angle(keypoints.right_shoulder, keypoints.right_hip, keypoints.right_knee),
        "L_HIP": safe_angle(keypoints.left_shoulder, keypoints.left_hip, keypoints.left_knee),
        "R_KNEE": safe_angle(keypoints.right_hip, keypoints.right_knee, keypoints.right_ankle),
        "L_KNEE": safe_angle(keypoints.left_hip, keypoints.left_knee, keypoints.left_ankle),
        "R_ANKLE": safe_angle(keypoints.right_knee, keypoints.right_ankle, keypoints.right_foot_index),
        "L_ANKLE": safe_angle(keypoints.left_knee, keypoints.left_ankle, keypoints.left_foot_index),
    }


def _fill_none_forward(values: Sequence[float | None], default: float = 0.0) -> list[float]:
    """Forward-fill None values in a series."""
    out: list[float] = []
    last = default
    for v in values:
        if v is not None:
            last = float(v)
        out.append(last)
    return out


def moving_average(values: Sequence[float], window_size: int = 5) -> list[float]:
    if window_size <= 1 or len(values) < window_size:
        return list(values)
    kernel = np.ones(window_size, dtype=float) / window_size
    smoothed = np.convolve(np.array(values, dtype=float), kernel, mode="same")
    return smoothed.tolist()


def detect_peak_frames_from_angles(
    frame_angles: Sequence[float | None],
    fps: float,
    smooth_window: int = 5,
    min_distance_sec: float = 0.45,
    min_prominence_deg: float = 6.0,
) -> list[int]:
    """Detect local maxima on right-knee angle curve and return original frame ids."""
    valid_pairs = [(idx, angle) for idx, angle in enumerate(frame_angles) if angle is not None]
    if len(valid_pairs) < 3:
        return []

    valid_frames = [p[0] for p in valid_pairs]
    valid_angles = [float(p[1]) for p in valid_pairs]
    smoothed = moving_average(valid_angles, window_size=smooth_window)
    min_distance = max(1, int(min_distance_sec * fps))

    candidates: list[tuple[int, float]] = []
    for i in range(1, len(smoothed) - 1):
        prev_v = smoothed[i - 1]
        cur_v = smoothed[i]
        next_v = smoothed[i + 1]
        prominence = cur_v - max(prev_v, next_v)
        if cur_v > prev_v and cur_v >= next_v and prominence >= min_prominence_deg:
            candidates.append((valid_frames[i], cur_v))

    if not candidates:
        best_frame = valid_frames[int(np.argmax(smoothed))]
        return [best_frame]

    selected: list[tuple[int, float]] = []
    for frame_idx, value in sorted(candidates, key=lambda x: x[1], reverse=True):
        if any(abs(frame_idx - sf) < min_distance for sf, _ in selected):
            continue
        selected.append((frame_idx, value))

    selected = sorted(selected, key=lambda x: x[0])
    return [frame_idx for frame_idx, _ in selected]


def compute_normalized_kick_heights(keypoints: Keypoints2D) -> dict[str, float | None]:
    """Return normalized foot heights relative to hip level and torso length."""
    if keypoints.left_shoulder is None or keypoints.right_shoulder is None:
        return {"R_KICK_HEIGHT": None, "L_KICK_HEIGHT": None}
    if keypoints.left_hip is None or keypoints.right_hip is None:
        return {"R_KICK_HEIGHT": None, "L_KICK_HEIGHT": None}

    shoulder_mid = (
        (keypoints.left_shoulder[0] + keypoints.right_shoulder[0]) / 2.0,
        (keypoints.left_shoulder[1] + keypoints.right_shoulder[1]) / 2.0,
    )
    hip_mid = (
        (keypoints.left_hip[0] + keypoints.right_hip[0]) / 2.0,
        (keypoints.left_hip[1] + keypoints.right_hip[1]) / 2.0,
    )
    torso_len = float(np.linalg.norm(np.array(shoulder_mid) - np.array(hip_mid)))
    if torso_len < 1e-6:
        return {"R_KICK_HEIGHT": None, "L_KICK_HEIGHT": None}

    def height(hip: tuple[float, float] | None, ankle: tuple[float, float] | None) -> float | None:
        if hip is None or ankle is None:
            return None
        # In image coordinates, smaller y means higher.
        return float((hip[1] - ankle[1]) / torso_len)

    return {
        "R_KICK_HEIGHT": height(keypoints.right_hip, keypoints.right_ankle),
        "L_KICK_HEIGHT": height(keypoints.left_hip, keypoints.left_ankle),
    }


def compute_torso_length(keypoints: Keypoints2D) -> float | None:
    """Return shoulder-mid to hip-mid distance for scale normalization."""
    if (
        keypoints.left_shoulder is None
        or keypoints.right_shoulder is None
        or keypoints.left_hip is None
        or keypoints.right_hip is None
    ):
        return None
    shoulder_mid = np.array(
        [
            (keypoints.left_shoulder[0] + keypoints.right_shoulder[0]) / 2.0,
            (keypoints.left_shoulder[1] + keypoints.right_shoulder[1]) / 2.0,
        ]
    )
    hip_mid = np.array(
        [
            (keypoints.left_hip[0] + keypoints.right_hip[0]) / 2.0,
            (keypoints.left_hip[1] + keypoints.right_hip[1]) / 2.0,
        ]
    )
    tl = float(np.linalg.norm(shoulder_mid - hip_mid))
    return tl if tl > 1e-6 else None


def compute_angular_velocity(
    angle_series: Sequence[float | None],
    fps: float,
    smooth_window: int = 9,
    polyorder: int = 3,
) -> tuple[list[float | None], list[float | None]]:
    """Compute angular velocity (deg/s) and acceleration (deg/s²).

    Uses Savitzky-Golay filter which fits a polynomial to a sliding window and
    differentiates analytically — no phase lag, better peak timing than MA.
    Falls back to central-difference on MA if series is too short for SG.

    Returns (velocity_series, acceleration_series) — None where input is None.
    """
    n = len(angle_series)
    if n < 3:
        return [None] * n, [None] * n

    filled = _fill_none_forward(angle_series)
    dt = 1.0 / fps

    try:
        from scipy.signal import savgol_filter  # type: ignore[import]

        # window must be odd and > polyorder
        win = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        win = max(win, polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3)

        if n >= win:
            vel_arr = savgol_filter(filled, window_length=win, polyorder=polyorder,
                                    deriv=1, delta=dt)
            acc_arr = savgol_filter(filled, window_length=win, polyorder=polyorder,
                                    deriv=2, delta=dt)
        else:
            # Series too short — degrade to 5-frame MA + finite difference
            smoothed = moving_average(filled, window_size=min(5, n))
            vel_arr = np.gradient(smoothed, dt)
            acc_arr = np.gradient(vel_arr, dt)

    except ImportError:
        # scipy unavailable — central difference on moving average
        smoothed = moving_average(filled, window_size=smooth_window)
        vel_arr = np.gradient(smoothed, dt)
        acc_arr = np.gradient(vel_arr, dt)

    # Re-apply None mask where original input was None
    vel: list[float | None] = [None if v is None else float(vel_arr[i]) for i, v in enumerate(angle_series)]
    acc: list[float | None] = [None if v is None else float(acc_arr[i]) for i, v in enumerate(angle_series)]
    return vel, acc


def compute_bilateral_asi(r_val: float | None, l_val: float | None) -> float | None:
    """Bilateral Asymmetry Index: (R - L) / mean(R, L) × 100.

    Positive → right dominant, negative → left dominant.
    Values >10% are typically considered clinically meaningful.
    """
    if r_val is None or l_val is None:
        return None
    denom = (r_val + l_val) / 2.0
    if abs(denom) < 1e-8:
        return None
    return float((r_val - l_val) / denom * 100.0)


def compute_foot_speed(
    positions: Sequence[tuple[float, float] | None],
    fps: float,
    torso_lengths: Sequence[float | None],
) -> list[float | None]:
    """Compute normalized foot speed (torso_lengths/s) from pixel positions.

    Uses consecutive frame displacement divided by torso length so results
    are scale-invariant across different camera distances.
    """
    n = len(positions)
    if n < 2:
        return [None] * n

    dt = 1.0 / fps
    speeds: list[float | None] = [None] * n

    for i in range(1, n):
        p_prev = positions[i - 1]
        p_cur = positions[i]
        if p_prev is None or p_cur is None:
            continue
        dist = float(np.linalg.norm(np.array(p_cur, dtype=float) - np.array(p_prev, dtype=float)))
        tl = torso_lengths[i] if i < len(torso_lengths) else None
        if tl is not None and tl > 1e-6:
            speeds[i] = (dist / tl) / dt
        else:
            speeds[i] = dist / dt

    # Propagate to frame 0 so callers always get a value at the start
    if n > 1 and speeds[1] is not None:
        speeds[0] = speeds[1]

    return speeds
