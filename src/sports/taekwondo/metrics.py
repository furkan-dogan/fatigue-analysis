"""Legacy kick metrics; normalized foot speed is not sprint speed."""

from __future__ import annotations

from typing import Iterable, Sequence
import numpy as np
from src.core.types import Keypoints2D

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
