"""Decode source frames for manual review, without inference or resampling."""
import math
import cv2
from src.adapters.video_metadata import probe_video


def inspect_video(path):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError('Video açılamadı.')
        fps = capture.get(cv2.CAP_PROP_FPS)
        reported_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        count, width, height = 0, 0, 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            count += 1
        if count == 0:
            raise ValueError('Videoda okunabilir kare yok.')
        metadata = probe_video(path)
        metadata.update(frame_count=count, width=width, height=height,
                        reported_frame_count=int(reported_count) if math.isfinite(reported_count) and reported_count > 0 else None,
                        nominal_fps=fps if math.isfinite(fps) and fps > 0 else None,
                        timestamp_count_matches_frames=len(metadata['timestamps']) == count)
        return metadata
    finally:
        capture.release()


def read_frame(path, index):
    # Sequential decoding avoids inaccurate keyframe seeking during contact review.
    if not isinstance(index, int) or index < 0:
        raise ValueError('Kare numarası geçersiz.')
    capture = cv2.VideoCapture(str(path))
    try:
        for _ in range(index + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError('İstenen kare okunamadı.')
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()
