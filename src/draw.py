"""Drawing helpers for annotated output video."""

from __future__ import annotations

from typing import Any

import cv2
import mediapipe as mp


_mp_pose = mp.solutions.pose

# BGR colors
_COLORS = {
    "right_arm": (0, 80, 255),
    "left_arm": (255, 170, 0),
    "right_leg": (0, 200, 255),
    "left_leg": (0, 220, 0),
    "torso": (220, 220, 220),
    "head": (180, 100, 255),
}

_SEGMENTS = {
    "right_arm": [
        (_mp_pose.PoseLandmark.RIGHT_SHOULDER, _mp_pose.PoseLandmark.RIGHT_ELBOW),
        (_mp_pose.PoseLandmark.RIGHT_ELBOW, _mp_pose.PoseLandmark.RIGHT_WRIST),
        (_mp_pose.PoseLandmark.RIGHT_WRIST, _mp_pose.PoseLandmark.RIGHT_INDEX),
    ],
    "left_arm": [
        (_mp_pose.PoseLandmark.LEFT_SHOULDER, _mp_pose.PoseLandmark.LEFT_ELBOW),
        (_mp_pose.PoseLandmark.LEFT_ELBOW, _mp_pose.PoseLandmark.LEFT_WRIST),
        (_mp_pose.PoseLandmark.LEFT_WRIST, _mp_pose.PoseLandmark.LEFT_INDEX),
    ],
    "right_leg": [
        (_mp_pose.PoseLandmark.RIGHT_HIP, _mp_pose.PoseLandmark.RIGHT_KNEE),
        (_mp_pose.PoseLandmark.RIGHT_KNEE, _mp_pose.PoseLandmark.RIGHT_ANKLE),
        (_mp_pose.PoseLandmark.RIGHT_ANKLE, _mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
        (_mp_pose.PoseLandmark.RIGHT_ANKLE, _mp_pose.PoseLandmark.RIGHT_HEEL),
    ],
    "left_leg": [
        (_mp_pose.PoseLandmark.LEFT_HIP, _mp_pose.PoseLandmark.LEFT_KNEE),
        (_mp_pose.PoseLandmark.LEFT_KNEE, _mp_pose.PoseLandmark.LEFT_ANKLE),
        (_mp_pose.PoseLandmark.LEFT_ANKLE, _mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
        (_mp_pose.PoseLandmark.LEFT_ANKLE, _mp_pose.PoseLandmark.LEFT_HEEL),
    ],
    "torso": [
        (_mp_pose.PoseLandmark.LEFT_SHOULDER, _mp_pose.PoseLandmark.RIGHT_SHOULDER),
        (_mp_pose.PoseLandmark.LEFT_HIP, _mp_pose.PoseLandmark.RIGHT_HIP),
        (_mp_pose.PoseLandmark.LEFT_SHOULDER, _mp_pose.PoseLandmark.LEFT_HIP),
        (_mp_pose.PoseLandmark.RIGHT_SHOULDER, _mp_pose.PoseLandmark.RIGHT_HIP),
    ],
    "head": [
        (_mp_pose.PoseLandmark.NOSE, _mp_pose.PoseLandmark.LEFT_EYE),
        (_mp_pose.PoseLandmark.NOSE, _mp_pose.PoseLandmark.RIGHT_EYE),
        (_mp_pose.PoseLandmark.LEFT_EYE, _mp_pose.PoseLandmark.LEFT_EAR),
        (_mp_pose.PoseLandmark.RIGHT_EYE, _mp_pose.PoseLandmark.RIGHT_EAR),
    ],
}

_JOINT_GROUP = {}
for group_name, pairs in _SEGMENTS.items():
    for a, b in pairs:
        _JOINT_GROUP[int(a)] = group_name
        _JOINT_GROUP[int(b)] = group_name

_LABEL_JOINTS = {
    int(_mp_pose.PoseLandmark.LEFT_SHOULDER): "L_SH",
    int(_mp_pose.PoseLandmark.RIGHT_SHOULDER): "R_SH",
    int(_mp_pose.PoseLandmark.LEFT_ELBOW): "L_EL",
    int(_mp_pose.PoseLandmark.RIGHT_ELBOW): "R_EL",
    int(_mp_pose.PoseLandmark.LEFT_WRIST): "L_WR",
    int(_mp_pose.PoseLandmark.RIGHT_WRIST): "R_WR",
    int(_mp_pose.PoseLandmark.LEFT_HIP): "L_HIP",
    int(_mp_pose.PoseLandmark.RIGHT_HIP): "R_HIP",
    int(_mp_pose.PoseLandmark.LEFT_KNEE): "L_KNE",
    int(_mp_pose.PoseLandmark.RIGHT_KNEE): "R_KNE",
    int(_mp_pose.PoseLandmark.LEFT_ANKLE): "L_ANK",
    int(_mp_pose.PoseLandmark.RIGHT_ANKLE): "R_ANK",
}


def draw_pose(frame_bgr: Any, pose_landmarks: Any, show_joint_labels: bool = False) -> None:
    if pose_landmarks is None:
        return

    h, w = frame_bgr.shape[:2]
    lms = pose_landmarks.landmark

    def to_px(idx: int) -> tuple[int, int] | None:
        lm = lms[idx]
        if lm.visibility < 0.2:
            return None
        return int(lm.x * w), int(lm.y * h)

    for group_name, pairs in _SEGMENTS.items():
        color = _COLORS[group_name]
        for start_lm, end_lm in pairs:
            p1 = to_px(int(start_lm))
            p2 = to_px(int(end_lm))
            if p1 is None or p2 is None:
                continue
            cv2.line(frame_bgr, p1, p2, color, 3, cv2.LINE_AA)

    for idx, lm in enumerate(lms):
        if lm.visibility < 0.2:
            continue
        center = (int(lm.x * w), int(lm.y * h))
        group_name = _JOINT_GROUP.get(idx, "torso")
        cv2.circle(frame_bgr, center, 4, _COLORS[group_name], -1, cv2.LINE_AA)
        if show_joint_labels and idx in _LABEL_JOINTS:
            cv2.putText(
                frame_bgr,
                _LABEL_JOINTS[idx],
                (center[0] + 6, center[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )


def draw_knee_angle(frame_bgr: Any, angle: float | None) -> None:
    label = "Right Knee Angle: N/A" if angle is None else f"Right Knee Angle: {angle:.1f}"
    cv2.putText(
        frame_bgr,
        label,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def draw_joint_angle_panel(frame_bgr: Any, angle_map: dict[str, float | None]) -> None:
    """Draw two compact panels: Left joints top-left, Right joints top-right."""
    h, w = frame_bgr.shape[:2]

    # Joint display names (short)
    SHORT = {
        "R_SHOULDER": "SHOULDER", "L_SHOULDER": "SHOULDER",
        "R_ELBOW":    "ELBOW",    "L_ELBOW":    "ELBOW",
        "R_HIP":      "HIP",      "L_HIP":      "HIP",
        "R_KNEE":     "KNEE",     "L_KNEE":     "KNEE",
        "R_ANKLE":    "ANKLE",    "L_ANKLE":    "ANKLE",
    }

    left_joints  = ["L_SHOULDER", "L_ELBOW", "L_HIP", "L_KNEE", "L_ANKLE"]
    right_joints = ["R_SHOULDER", "R_ELBOW", "R_HIP", "R_KNEE", "R_ANKLE"]

    # Colors matching skeleton: left=blue, right=orange-red
    COLOR_L = (255, 160, 60)   # BGR — blue tones for left
    COLOR_R = (60, 160, 255)   # BGR — orange-red tones for right
    COLOR_TITLE = (220, 220, 220)
    BG = (0, 0, 0)

    margin   = 8
    pad      = 6
    line_h   = 18
    font     = cv2.FONT_HERSHEY_SIMPLEX
    fscale   = 0.45
    thick    = 1

    n_rows   = len(left_joints)  # same for both sides
    panel_h  = pad * 2 + line_h + n_rows * line_h   # title + rows
    panel_w  = 130

    def _draw_panel(joints: list[str], color: tuple, x0: int) -> None:
        y0 = margin
        # Background
        cv2.rectangle(frame_bgr, (x0, y0), (x0 + panel_w, y0 + panel_h), BG, -1)
        # Subtle border
        cv2.rectangle(frame_bgr, (x0, y0), (x0 + panel_w, y0 + panel_h), color, 1)
        # Title
        side = "LEFT" if "L_" in joints[0] else "RIGHT"
        cv2.putText(frame_bgr, f"{side} (deg)", (x0 + pad, y0 + pad + line_h - 4),
                    font, fscale, COLOR_TITLE, thick, cv2.LINE_AA)
        # Rows
        for i, key in enumerate(joints):
            val = angle_map.get(key)
            label = SHORT.get(key, key)
            text = f"{label}: ---" if val is None else f"{label}: {val:5.1f}"
            y_text = y0 + pad + line_h + (i + 1) * line_h - 4
            cv2.putText(frame_bgr, text, (x0 + pad, y_text),
                        font, fscale, color, thick, cv2.LINE_AA)

    # Left panel: top-left corner
    _draw_panel(left_joints,  COLOR_L, x0=margin)
    # Right panel: top-right corner
    _draw_panel(right_joints, COLOR_R, x0=w - margin - panel_w)
