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
        if lm.visibility < 0.1:
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

    # Skip face landmarks (indices 0–10: nose, eyes, ears, mouth)
    for idx, lm in enumerate(lms):
        if idx <= 10:
            continue
        if lm.visibility < 0.1:
            continue
        center = (int(lm.x * w), int(lm.y * h))
        group_name = _JOINT_GROUP.get(idx, "torso")
        if group_name not in _COLORS:
            continue
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


def draw_pose_from_keypoints(
    frame_bgr: Any,
    keypoints: Any,
    show_joint_labels: bool = False,
) -> None:
    """Draw skeleton from a Keypoints2D object — works with any backend.

    Used by YOLOv8-pose (and any future backend) that does not produce
    MediaPipe-format landmarks.
    """
    from src.pose_runner import Keypoints2D  # avoid circular at module level

    if keypoints is None:
        return

    # (point_a, point_b, group_name)
    connections: list[tuple[Any, Any, str]] = [
        (keypoints.right_shoulder, keypoints.right_elbow, "right_arm"),
        (keypoints.right_elbow,    keypoints.right_wrist,  "right_arm"),
        (keypoints.left_shoulder,  keypoints.left_elbow,   "left_arm"),
        (keypoints.left_elbow,     keypoints.left_wrist,   "left_arm"),
        (keypoints.right_hip,      keypoints.right_knee,   "right_leg"),
        (keypoints.right_knee,     keypoints.right_ankle,  "right_leg"),
        (keypoints.right_ankle,    keypoints.right_foot_index, "right_leg"),
        (keypoints.left_hip,       keypoints.left_knee,    "left_leg"),
        (keypoints.left_knee,      keypoints.left_ankle,   "left_leg"),
        (keypoints.left_ankle,     keypoints.left_foot_index,  "left_leg"),
        (keypoints.left_shoulder,  keypoints.right_shoulder, "torso"),
        (keypoints.left_hip,       keypoints.right_hip,    "torso"),
        (keypoints.left_shoulder,  keypoints.left_hip,     "torso"),
        (keypoints.right_shoulder, keypoints.right_hip,    "torso"),
    ]

    for p1, p2, group in connections:
        if p1 is None or p2 is None:
            continue
        pt1 = (int(p1[0]), int(p1[1]))
        pt2 = (int(p2[0]), int(p2[1]))
        cv2.line(frame_bgr, pt1, pt2, _COLORS[group], 3, cv2.LINE_AA)

    # Joint circles
    all_joints: list[tuple[Any, str, str]] = [
        (keypoints.left_shoulder,  "left_arm",  "L_SH"),
        (keypoints.right_shoulder, "right_arm", "R_SH"),
        (keypoints.left_elbow,     "left_arm",  "L_EL"),
        (keypoints.right_elbow,    "right_arm", "R_EL"),
        (keypoints.left_wrist,     "left_arm",  "L_WR"),
        (keypoints.right_wrist,    "right_arm", "R_WR"),
        (keypoints.left_hip,       "left_leg",  "L_HIP"),
        (keypoints.right_hip,      "right_leg", "R_HIP"),
        (keypoints.left_knee,      "left_leg",  "L_KNE"),
        (keypoints.right_knee,     "right_leg", "R_KNE"),
        (keypoints.left_ankle,     "left_leg",  "L_ANK"),
        (keypoints.right_ankle,    "right_leg", "R_ANK"),
    ]

    for pt, group, label in all_joints:
        if pt is None:
            continue
        center = (int(pt[0]), int(pt[1]))
        cv2.circle(frame_bgr, center, 4, _COLORS[group], -1, cv2.LINE_AA)
        if show_joint_labels:
            cv2.putText(
                frame_bgr, label,
                (center[0] + 6, center[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
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
