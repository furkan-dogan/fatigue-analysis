"""Core analysis pipeline — importable by CLI and Streamlit dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from src.draw import draw_joint_angle_panel, draw_pose, draw_pose_from_keypoints
from src.events import JOINT_KEYS, detect_movement_events
from src.exporter import write_event_metrics_csv, write_frame_metrics_csv
from src.metrics import (
    VELOCITY_JOINT_KEYS,
    calculate_joint_angles,
    compute_angular_velocity,
    compute_foot_speed,
    compute_normalized_kick_heights,
    compute_torso_length,
    summarize_knee_angles,
)
from src.pose_runner import MediaPipePoseRunner, YOLOPoseRunner


@dataclass
class AnalysisResult:
    fps: float
    total_frames: int
    frame_rows: list[dict]
    events: list[dict]
    knee_summary: dict | None
    frame_csv_path: str
    events_csv_path: str
    output_video_path: str


def _empty_angle_map() -> dict[str, float | None]:
    return {k: None for k in JOINT_KEYS}


def run_analysis(
    input_path: str | Path,
    output_path: str | Path,
    frame_csv_path: str | Path,
    events_csv_path: str | Path,
    show_joint_labels: bool = False,
    event_peak_prominence_norm: float = 0.06,
    event_min_distance_sec: float = 0.4,
    event_min_duration_sec: float = 0.15,
    event_max_duration_sec: float = 6.0,
    event_min_knee_rom_deg: float = 20.0,
    event_min_peak_kick_height_norm: float = -0.3,
    progress_callback: Callable[[int, int], None] | None = None,
    backend: str = "mediapipe",
    yolo_model: str = "yolo11n-pose.pt",
    vel_assist_threshold: float = 80.0,
) -> AnalysisResult:
    """Run full pose-analysis pipeline on a single video.

    Args:
        input_path: Source video file.
        output_path: Destination for annotated video.
        frame_csv_path: Destination for frame-level metrics CSV.
        events_csv_path: Destination for kick-event metrics CSV.
        show_joint_labels: Overlay joint label text on video.
        event_*: Kick-event detection parameters.
        progress_callback: Optional callable(current_frame, total_frames) for UI progress bars.
        backend: Pose backend — "mediapipe" (default) or "yolo".
        yolo_model: YOLO model filename, e.g. "yolo11n-pose.pt" or "yolov8n-pose.pt".
                    Downloaded automatically on first use.

    Returns:
        AnalysisResult with all computed data.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    frame_csv_path = Path(frame_csv_path)
    events_csv_path = Path(events_csv_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Video bulunamadı: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_csv_path.parent.mkdir(parents=True, exist_ok=True)
    events_csv_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # avc1 (H.264) → tarayıcı/Streamlit'te doğrudan oynatılabilir
    # mp4v fallback: avc1 bu sistemde desteklenmiyorsa devreye girer
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if backend == "yolo":
        runner: MediaPipePoseRunner | YOLOPoseRunner = YOLOPoseRunner(model_name=yolo_model)
    else:
        runner = MediaPipePoseRunner()
    use_yolo = isinstance(runner, YOLOPoseRunner)
    total_frames = 0

    # Accumulate time-series for post-loop analytics
    knee_angles_r: list[float | None] = []
    knee_angles_l: list[float | None] = []
    kick_heights_r: list[float | None] = []
    kick_heights_l: list[float | None] = []
    joint_series: dict[str, list[float | None]] = {k: [] for k in JOINT_KEYS}
    torso_length_series: list[float | None] = []
    foot_pos_r: list[tuple[float, float] | None] = []
    foot_pos_l: list[tuple[float, float] | None] = []
    confidence_series: list[float | None] = []

    frame_rows: list[dict] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            keypoints, pose_landmarks = runner.process_frame(frame)

            angle_map = _empty_angle_map()
            right_height: float | None = None
            left_height: float | None = None
            active_height: float | None = None
            torso_len: float | None = None
            r_foot: tuple[float, float] | None = None
            l_foot: tuple[float, float] | None = None

            if keypoints is not None:
                angle_map = calculate_joint_angles(keypoints)
                height_map = compute_normalized_kick_heights(keypoints)
                right_height = height_map["R_KICK_HEIGHT"]
                left_height = height_map["L_KICK_HEIGHT"]
                torso_len = compute_torso_length(keypoints)
                r_foot = keypoints.right_foot_index
                l_foot = keypoints.left_foot_index

                if right_height is not None and left_height is not None:
                    active_height = max(right_height, left_height)
                elif right_height is not None:
                    active_height = right_height
                else:
                    active_height = left_height

            conf = runner.get_confidence(pose_landmarks)
            for joint_key in JOINT_KEYS:
                joint_series[joint_key].append(angle_map[joint_key])
            knee_angles_r.append(angle_map["R_KNEE"])
            knee_angles_l.append(angle_map["L_KNEE"])
            kick_heights_r.append(right_height)
            kick_heights_l.append(left_height)
            torso_length_series.append(torso_len)
            foot_pos_r.append(r_foot)
            foot_pos_l.append(l_foot)
            confidence_series.append(conf)

            frame_rows.append(
                {
                    "frame": total_frames,
                    "time_sec": round(total_frames / fps, 4),
                    **angle_map,
                    "R_KICK_HEIGHT": None if right_height is None else round(float(right_height), 6),
                    "L_KICK_HEIGHT": None if left_height is None else round(float(left_height), 6),
                    "KICK_HEIGHT_ACTIVE": None if active_height is None else round(float(active_height), 6),
                    "pose_confidence": None if conf is None else round(conf, 4),
                }
            )

            if use_yolo:
                draw_pose_from_keypoints(frame, keypoints, show_joint_labels=show_joint_labels)
            else:
                draw_pose(frame, pose_landmarks, show_joint_labels=show_joint_labels)
            draw_joint_angle_panel(frame, angle_map)
            writer.write(frame)
            total_frames += 1

            if progress_callback is not None:
                progress_callback(total_frames, max(total_frame_count, total_frames))

    finally:
        cap.release()
        writer.release()
        runner.close()

    # ── Post-loop: velocity / acceleration / foot-speed ──────────────────────
    velocity_series: dict[str, list[float | None]] = {}
    acceleration_series: dict[str, list[float | None]] = {}

    for jk in VELOCITY_JOINT_KEYS:
        vel, acc = compute_angular_velocity(joint_series[jk], fps, smooth_window=5)
        velocity_series[f"{jk}_vel"] = vel
        acceleration_series[f"{jk}_acc"] = acc

    foot_speed_r = compute_foot_speed(foot_pos_r, fps, torso_length_series)
    foot_speed_l = compute_foot_speed(foot_pos_l, fps, torso_length_series)
    foot_speed_series: dict[str, list[float | None]] = {
        "R_FOOT_speed": foot_speed_r,
        "L_FOOT_speed": foot_speed_l,
    }

    # Append velocity/speed columns to existing frame_rows
    for i, row in enumerate(frame_rows):
        for jk in VELOCITY_JOINT_KEYS:
            vel_val = velocity_series[f"{jk}_vel"][i] if i < len(velocity_series[f"{jk}_vel"]) else None
            acc_val = acceleration_series[f"{jk}_acc"][i] if i < len(acceleration_series[f"{jk}_acc"]) else None
            row[f"{jk}_vel_deg_s"] = None if vel_val is None else round(float(vel_val), 4)
            row[f"{jk}_acc_deg_s2"] = None if acc_val is None else round(float(acc_val), 4)
        row["R_FOOT_speed_norm"] = None if foot_speed_r[i] is None else round(float(foot_speed_r[i]), 6)  # type: ignore[index]
        row["L_FOOT_speed_norm"] = None if foot_speed_l[i] is None else round(float(foot_speed_l[i]), 6)  # type: ignore[index]

    # ── Event detection ───────────────────────────────────────────────────────
    knee_summary = summarize_knee_angles([a for a in knee_angles_r if a is not None])

    events = detect_movement_events(
        right_knee_angles=knee_angles_r,
        left_knee_angles=knee_angles_l,
        right_kick_heights=kick_heights_r,
        left_kick_heights=kick_heights_l,
        joint_series=joint_series,
        fps=fps,
        min_peak_prominence_norm=event_peak_prominence_norm,
        min_distance_sec=event_min_distance_sec,
        min_duration_sec=event_min_duration_sec,
        max_duration_sec=event_max_duration_sec,
        velocity_series=velocity_series,
        foot_speed_series=foot_speed_series,
        min_knee_rom_deg=event_min_knee_rom_deg,
        min_peak_kick_height_norm=event_min_peak_kick_height_norm,
        confidence_series=confidence_series,
        vel_assist_threshold=vel_assist_threshold,
    )

    # ── Export ────────────────────────────────────────────────────────────────
    write_frame_metrics_csv(frame_csv_path, frame_rows)
    write_event_metrics_csv(events_csv_path, events)

    return AnalysisResult(
        fps=fps,
        total_frames=total_frames,
        frame_rows=frame_rows,
        events=events,
        knee_summary=knee_summary,
        frame_csv_path=str(frame_csv_path),
        events_csv_path=str(events_csv_path),
        output_video_path=str(output_path),
    )
