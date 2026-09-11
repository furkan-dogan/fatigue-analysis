"""Offline pair analysis; presentation and session state stay in the UI."""
from pathlib import Path
import shutil
import tempfile
from src.sports.taekwondo.pipeline import run_analysis


def analyze_pair(pre_video: bytes, post_video: bytes, progress):
    directory = Path(tempfile.mkdtemp(prefix='video-analysis-'))
    results = []
    try:
        for index, (name, content) in enumerate([('pre', pre_video), ('post', post_video)]):
            source = directory / f'{name}_input.mp4'
            source.write_bytes(content)

            def update(current, total):
                progress(index * 50 + int(min(current / max(total, 1), 1) * 50),
                         f'{name.upper()} — Kare {current}/{total}')

            results.append(run_analysis(
                source, directory / f'{name}_annotated.mp4',
                directory / f'{name}_frames.csv', directory / f'{name}_events.csv',
                show_joint_labels=False, event_peak_prominence_norm=0.06,
                event_min_distance_sec=0.25, event_min_duration_sec=0.10,
                event_max_duration_sec=6.0, event_min_knee_rom_deg=12.0,
                event_min_peak_kick_height_norm=-0.5, vel_assist_threshold=100.0,
                progress_callback=update,
            ))
        return results[0], results[1], directory
    except Exception:
        shutil.rmtree(directory)
        raise
