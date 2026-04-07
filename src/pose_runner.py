"""Pose extraction backend (MediaPipe for MVP).

Design note:
- This module exposes backend-agnostic frame-level keypoints.
- Later, an OpenPose runner can implement the same output schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp


@dataclass
class Keypoints2D:
    left_shoulder: tuple[float, float] | None
    right_shoulder: tuple[float, float] | None
    left_elbow: tuple[float, float] | None
    right_elbow: tuple[float, float] | None
    left_wrist: tuple[float, float] | None
    right_wrist: tuple[float, float] | None
    left_hip: tuple[float, float] | None
    right_hip: tuple[float, float]
    left_knee: tuple[float, float] | None
    right_knee: tuple[float, float]
    left_ankle: tuple[float, float] | None
    right_ankle: tuple[float, float]
    left_foot_index: tuple[float, float] | None
    right_foot_index: tuple[float, float] | None


class MediaPipePoseRunner:
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        self.pose.close()

    def process_frame(self, frame_bgr: Any) -> tuple[Keypoints2D | None, Any | None]:
        """Run pose inference on a BGR frame and return keypoints + full landmarks."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if not results.pose_landmarks:
            return None, None

        h, w = frame_bgr.shape[:2]
        lm = results.pose_landmarks.landmark

        def pick(landmark: mp.solutions.pose.PoseLandmark) -> tuple[float, float] | None:
            point = lm[landmark]
            if point.visibility < 0.2:
                return None
            return point.x * w, point.y * h

        right_hip = pick(self.mp_pose.PoseLandmark.RIGHT_HIP)
        right_knee = pick(self.mp_pose.PoseLandmark.RIGHT_KNEE)
        right_ankle = pick(self.mp_pose.PoseLandmark.RIGHT_ANKLE)
        if right_hip is None or right_knee is None or right_ankle is None:
            return None, results.pose_landmarks

        keypoints = Keypoints2D(
            left_shoulder=pick(self.mp_pose.PoseLandmark.LEFT_SHOULDER),
            right_shoulder=pick(self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
            left_elbow=pick(self.mp_pose.PoseLandmark.LEFT_ELBOW),
            right_elbow=pick(self.mp_pose.PoseLandmark.RIGHT_ELBOW),
            left_wrist=pick(self.mp_pose.PoseLandmark.LEFT_WRIST),
            right_wrist=pick(self.mp_pose.PoseLandmark.RIGHT_WRIST),
            left_hip=pick(self.mp_pose.PoseLandmark.LEFT_HIP),
            right_hip=right_hip,
            left_knee=pick(self.mp_pose.PoseLandmark.LEFT_KNEE),
            right_knee=right_knee,
            left_ankle=pick(self.mp_pose.PoseLandmark.LEFT_ANKLE),
            right_ankle=right_ankle,
            left_foot_index=pick(self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
            right_foot_index=pick(self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
        )
        return keypoints, results.pose_landmarks
