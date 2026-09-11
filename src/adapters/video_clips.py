"""Legacy OpenCV clip extraction used by the local dashboard."""

from pathlib import Path

def trim_clip(src: Path, start: float, end: float, out: Path) -> bool:
    """Cut [start, end] seconds from src using OpenCV. Returns True on success."""
    try:
        import cv2 as _cv2
        cap = _cv2.VideoCapture(str(src))
        fps_v = cap.get(_cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = _cv2.VideoWriter_fourcc(*"avc1")
        writer = _cv2.VideoWriter(str(out), fourcc, fps_v, (w, h))
        if not writer.isOpened():
            fourcc = _cv2.VideoWriter_fourcc(*"mp4v")
            writer = _cv2.VideoWriter(str(out), fourcc, fps_v, (w, h))
        pad = 0.4
        f_start = max(0, int((start - pad) * fps_v))
        f_end   = int((end + pad) * fps_v)
        cap.set(_cv2.CAP_PROP_POS_FRAMES, f_start)
        for _ in range(f_end - f_start + 1):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
        cap.release()
        writer.release()
        return out.exists() and out.stat().st_size > 1000
    except Exception:
        return False
