"""Read source presentation timestamps without confusing nominal FPS with time."""
import json
from math import isfinite
import shutil
import subprocess


def probe_video(path):
    if not shutil.which('ffprobe'):
        return {'timestamp_source': 'unavailable', 'reason': 'ffprobe bulunamadı', 'timestamps': []}
    try:
        output = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
             '-show_streams', '-show_entries',
             'frame=best_effort_timestamp_time:stream=width,height,avg_frame_rate,r_frame_rate,time_base,duration',
             '-of', 'json', str(path)], capture_output=True, text=True, check=True, timeout=60,
        )
        data = json.loads(output.stdout)
        times = []
        for frame in data.get('frames', []):
            raw = frame.get('best_effort_timestamp_time')
            value = float(raw) if raw is not None else None
            times.append(value if value is not None and isfinite(value) else None)
        valid = bool(times) and all(t is not None for t in times) and all(b > a for a, b in zip(times, times[1:]))
        return {'timestamp_source': 'ffprobe_best_effort_pts', 'timestamps': times,
                'timestamps_monotonic': valid, 'stream': next(iter(data.get('streams', [])), {})}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {'timestamp_source': 'unavailable', 'reason': str(exc), 'timestamps': []}
