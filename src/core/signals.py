"""Shared time-series differentiation."""

from __future__ import annotations

from typing import Sequence
import numpy as np
from src.core.numeric import fill_none_forward, moving_average

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

    filled = fill_none_forward(angle_series)
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
            smoothed = moving_average(filled, window=min(5, n))
            vel_arr = np.gradient(smoothed, dt)
            acc_arr = np.gradient(vel_arr, dt)

    except ImportError:
        # scipy unavailable — central difference on moving average
        smoothed = moving_average(filled, window=smooth_window)
        vel_arr = np.gradient(smoothed, dt)
        acc_arr = np.gradient(vel_arr, dt)

    # Re-apply None mask where original input was None
    vel: list[float | None] = [None if v is None else float(vel_arr[i]) for i, v in enumerate(angle_series)]
    acc: list[float | None] = [None if v is None else float(acc_arr[i]) for i, v in enumerate(angle_series)]
    return vel, acc
