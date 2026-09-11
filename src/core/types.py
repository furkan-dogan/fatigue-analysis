"""Backend-independent 2D pose data and joint identifiers."""

from dataclasses import dataclass

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

VELOCITY_JOINT_KEYS = ["R_KNEE", "L_KNEE", "R_HIP", "L_HIP", "R_ANKLE", "L_ANKLE"]
