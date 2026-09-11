"""Deterministic legacy input captured before the package migration."""

import math

from src.sports.taekwondo.events import JOINT_KEYS, detect_movement_events
from src.core.signals import compute_angular_velocity
from src.sports.taekwondo.metrics import compute_foot_speed


def legacy_snapshot() -> dict:
    fps = 30.0
    pulse = [sum(math.exp(-((i - center) / 5) ** 2) for center in (30, 80)) for i in range(120)]
    heights = [-1 + 2 * p for p in pulse]
    knees = [170 - 90 * p for p in pulse]
    velocity, acceleration = compute_angular_velocity(knees, fps, smooth_window=5)
    foot_speed = compute_foot_speed([(0, h * 100) for h in heights], fps, [100.0] * 120)
    events = detect_movement_events(
        right_knee_angles=knees,
        left_knee_angles=[170.0] * 120,
        right_kick_heights=heights,
        left_kick_heights=[-1.0] * 120,
        joint_series={key: knees if key.startswith('R_') else [170.0] * 120 for key in JOINT_KEYS},
        fps=fps,
        velocity_series={'R_KNEE_vel': velocity, 'L_KNEE_vel': [0.0] * 120},
        foot_speed_series={'R_FOOT_speed': foot_speed, 'L_FOOT_speed': [0.0] * 120},
        confidence_series=[0.9] * 120,
    )
    assert events, 'Regression input must contain detected kicks'
    # Rounded values allow harmless floating-point variation across platforms.
    return {
        'events': events,
        'velocity': [round(v, 6) for v in velocity],
        'acceleration': [round(v, 6) for v in acceleration],
        'foot_speed': [round(v, 6) for v in foot_speed],
    }
