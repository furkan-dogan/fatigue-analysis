"""Shared numeric utilities used across src modules."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def fill_none_forward(values: Sequence[float | None], default: float = 0.0) -> list[float]:
    """Forward-fill None values; leading Nones become `default`."""
    out: list[float] = []
    last = default
    for v in values:
        if v is not None:
            last = float(v)
        out.append(last)
    return out


def fill_missing(values: Sequence[float | None], default: float = 0.0) -> list[float]:
    """Forward-fill Nones, then backward-fill any remaining leading Nones."""
    if not values:
        return []
    out = fill_none_forward(values, default)
    first_real = next((float(v) for v in values if v is not None), default)
    for i, v in enumerate(values):
        if v is None:
            out[i] = first_real
        else:
            break
    return out


def moving_average(values: Sequence[float], window: int = 5) -> list[float]:
    arr = np.array(values, dtype=float)
    if window <= 1 or len(arr) < window:
        return arr.tolist()
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(arr, kernel, mode="same").tolist()


def events_mean(events: list[dict], col: str) -> float | None:
    vals = [float(e[col]) for e in events if e.get(col) is not None]
    return sum(vals) / len(vals) if vals else None


def pct_change(a: float | None, b: float | None) -> float | None:
    if a and b and abs(a) > 1e-6:
        return (b - a) / abs(a) * 100
    return None


def change_status(pct_val: float | None, direction: int = -1) -> str:
    if pct_val is None:
        return "—"
    signal = direction * pct_val
    if signal > 10:
        return "⚠️ Düşüş"
    if signal > 5:
        return "⚡ Hafif Düşüş"
    if signal < -5:
        return "✅ İyileşme"
    return "➡️ Stabil"
