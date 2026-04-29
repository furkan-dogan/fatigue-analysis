"""Taekwondo fatigue analysis — core package.

Importlar lazy tutuldu: mediapipe kurulu değilse bile import-time hata vermez.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.pipeline import AnalysisResult, run_analysis
    from src.pose_runner import Keypoints2D, MediaPipePoseRunner

__all__ = ["AnalysisResult", "run_analysis", "Keypoints2D", "MediaPipePoseRunner"]
