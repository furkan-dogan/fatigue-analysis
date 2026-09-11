"""Shared image-plane geometry, independent of pose backends."""

from __future__ import annotations

from typing import Sequence
import numpy as np
from src.core.types import Keypoints2D

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
