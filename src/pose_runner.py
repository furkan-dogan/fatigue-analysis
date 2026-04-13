"""Pose extraction backends — MediaPipe (default) and YOLOv8-pose.

Both runners expose the same interface:
  process_frame(frame_bgr) -> (Keypoints2D | None, raw_result | None)
  get_confidence(raw_result) -> float | None
  close() -> None
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp

# MediaPipe lower-body landmark indices used for frame confidence scoring
_MP_KEY_LANDMARK_INDICES = [11, 12, 23, 24, 25, 26, 27, 28, 31, 32]

# COCO 17 lower-body keypoint indices (for YOLO confidence scoring)
_YOLO_KEY_LANDMARK_INDICES = [11, 12, 13, 14, 15, 16]  # hips, knees, ankles


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
            if point.visibility < 0.1:
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

    def get_confidence(self, raw_result: Any) -> float | None:
        """Mean visibility of key lower-body landmarks (0–1)."""
        if raw_result is None:
            return None
        lms = raw_result.landmark
        vals = [lms[i].visibility for i in _MP_KEY_LANDMARK_INDICES if i < len(lms)]
        return float(sum(vals) / len(vals)) if vals else None


class YOLOPoseRunner:
    """YOLOv8-pose (or YOLOv11-pose) backend.

    Uses COCO 17-keypoint format:
      0:nose  1:L_eye  2:R_eye  3:L_ear  4:R_ear
      5:L_sh  6:R_sh   7:L_el   8:R_el   9:L_wr  10:R_wr
      11:L_hip 12:R_hip 13:L_knee 14:R_knee
      15:L_ankle 16:R_ankle

    Note: COCO 17 has no foot-index keypoint — ankle is used as foot-index
    fallback, which is accurate enough for kick-height and foot-speed metrics.

    Person selection: uses ByteTrack to assign persistent track IDs across
    frames. After _lock_after frames the track with the highest accumulated
    lower-body motion is locked in, so a stationary trainer/pad-holder is
    automatically ignored.
    """

    _LOWER_BODY_IDX = [11, 12, 13, 14, 15, 16]  # hips, knees, ankles

    def __init__(
        self,
        model_name: str = "yolo11n-pose.pt",
        conf_threshold: float = 0.35,
        keypoint_conf: float = 0.15,
        device: str = "",
        lock_after_frames: int = 30,
    ) -> None:
        try:
            from ultralytics import YOLO  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "ultralytics paketi bulunamadı. Kurmak için:\n"
                "  pip install ultralytics"
            ) from exc

        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.keypoint_conf = keypoint_conf
        self.device = device
        self._lock_after = lock_after_frames

        # Tracking state (reset per video — runner is created fresh each run_analysis call)
        self._track_motion: dict[int, float] = {}       # track_id → cumulative lower-body motion
        self._prev_xy: dict[int, dict[int, tuple[float, float]]] = {}  # track_id → {kp_idx → (x, y)}
        self._locked_id: int | None = None
        self._frame_count: int = 0

    def close(self) -> None:
        pass  # YOLO models don't need explicit cleanup

    def process_frame(self, frame_bgr: Any) -> tuple[Keypoints2D | None, Any | None]:
        """Run YOLO pose inference with ByteTrack and return keypoints + raw result."""
        import numpy as np

        results = self.model.track(
            frame_bgr,
            persist=True,
            verbose=False,
            conf=self.conf_threshold,
            device=self.device if self.device else None,
            tracker="bytetrack.yaml",
        )

        if not results or results[0].keypoints is None:
            return None, None

        kpts_xy = results[0].keypoints.xy
        kpts_conf = results[0].keypoints.conf

        if kpts_xy is None or len(kpts_xy) == 0:
            return None, None

        # ── Track IDs from ByteTrack ──────────────────────────────────────────
        boxes = results[0].boxes
        track_ids: list[int] | None = None
        if boxes is not None and boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int).tolist()

        # ── Accumulate lower-body motion per track ────────────────────────────
        if track_ids is not None:
            for i, tid in enumerate(track_ids):
                if i >= len(kpts_xy):
                    continue
                xy_i = kpts_xy[i].cpu().numpy()
                conf_i = kpts_conf[i].cpu().numpy() if kpts_conf is not None else np.ones(17)
                motion = 0.0
                for idx in self._LOWER_BODY_IDX:
                    if idx < len(xy_i) and conf_i[idx] > self.keypoint_conf:
                        cur = (float(xy_i[idx, 0]), float(xy_i[idx, 1]))
                        prev_map = self._prev_xy.get(tid, {})
                        prev = prev_map.get(idx)
                        if prev is not None:
                            motion += ((cur[0] - prev[0]) ** 2 + (cur[1] - prev[1]) ** 2) ** 0.5
                        if tid not in self._prev_xy:
                            self._prev_xy[tid] = {}
                        self._prev_xy[tid][idx] = cur
                self._track_motion[tid] = self._track_motion.get(tid, 0.0) + motion

        self._frame_count += 1

        # ── Lock onto most-active track after warm-up ─────────────────────────
        if self._locked_id is None and self._frame_count >= self._lock_after:
            if self._track_motion:
                self._locked_id = max(self._track_motion, key=lambda k: self._track_motion[k])

        # ── Select best detection index ───────────────────────────────────────
        best = 0
        if track_ids is not None and len(kpts_xy) > 1:
            if self._locked_id is not None and self._locked_id in track_ids:
                best = track_ids.index(self._locked_id)
            elif self._track_motion:
                # pre-lock or locked person lost: pick most-active visible track
                scores = [self._track_motion.get(tid, 0.0) for tid in track_ids]
                best = int(np.argmax(scores))
            elif kpts_conf is not None:
                conf_scores = kpts_conf[:, _YOLO_KEY_LANDMARK_INDICES].mean(dim=1)
                best = int(conf_scores.argmax())

        xy = kpts_xy[best].cpu().numpy()     # (17, 2) — pixel coords
        conf = kpts_conf[best].cpu().numpy() if kpts_conf is not None else np.ones(17)

        thr = self.keypoint_conf

        def pick(idx: int) -> tuple[float, float] | None:
            if idx >= len(xy) or conf[idx] < thr:
                return None
            x, y = float(xy[idx, 0]), float(xy[idx, 1])
            if x == 0.0 and y == 0.0:
                return None
            return x, y

        right_hip   = pick(12)
        right_knee  = pick(14)
        right_ankle = pick(16)
        left_hip_pt = pick(11)

        # Need at least one hip to anchor the skeleton
        if right_hip is None and left_hip_pt is None:
            return None, results[0]

        # If right side critical joints lost (kick blur), use raw xy with low-conf fallback
        # so height signal stays alive — angles will be None (pick returns None) which is fine
        if right_hip is None:
            right_hip = (float(xy[12, 0]), float(xy[12, 1])) if xy[12, 0] > 0 else left_hip_pt
        if right_knee is None:
            # Try raw position even if conf is low — better than losing the signal entirely
            raw_knee = (float(xy[14, 0]), float(xy[14, 1])) if xy[14, 0] > 0 else None
            right_knee = raw_knee if raw_knee else right_hip
        if right_ankle is None:
            raw_ankle = (float(xy[16, 0]), float(xy[16, 1])) if xy[16, 0] > 0 else None
            right_ankle = raw_ankle if raw_ankle else right_knee

        # Left leg — same raw-fallback for kick frames
        left_knee_pt  = pick(13) or ((float(xy[13, 0]), float(xy[13, 1])) if xy[13, 0] > 0 else None)
        left_ankle_pt = pick(15) or ((float(xy[15, 0]), float(xy[15, 1])) if xy[15, 0] > 0 else None)

        # Ankle is used as foot_index fallback (COCO 17 has no foot keypoints)
        keypoints = Keypoints2D(
            left_shoulder=pick(5),
            right_shoulder=pick(6),
            left_elbow=pick(7),
            right_elbow=pick(8),
            left_wrist=pick(9),
            right_wrist=pick(10),
            left_hip=left_hip_pt,
            right_hip=right_hip,
            left_knee=left_knee_pt,
            right_knee=right_knee,
            left_ankle=left_ankle_pt,
            right_ankle=right_ankle,
            left_foot_index=left_ankle_pt,
            right_foot_index=right_ankle,
        )
        return keypoints, results[0]

    def get_confidence(self, raw_result: Any) -> float | None:
        """Mean YOLO keypoint confidence for lower-body landmarks (0–1)."""
        if raw_result is None:
            return None
        try:
            kpts_conf = raw_result.keypoints.conf
            if kpts_conf is None or len(kpts_conf) == 0:
                return None
            conf_arr = kpts_conf[0].cpu().numpy()
            vals = [float(conf_arr[i]) for i in _YOLO_KEY_LANDMARK_INDICES if i < len(conf_arr)]
            return float(sum(vals) / len(vals)) if vals else None
        except Exception:
            return None
