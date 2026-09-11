"""Repeatable populated dashboard inputs without real athlete video."""

from copy import deepcopy
import json
from pathlib import Path

import pandas as pd

from src.sports.taekwondo.pipeline import AnalysisResult


def session_pair(*, empty=False):
    fixture = json.loads((Path(__file__).parents[1] / 'regression/legacy_snapshot.json').read_text())
    events = [] if empty else fixture['events']
    rows = [
        {'time_sec': i / 30, 'R_KNEE': 150.0, 'L_KNEE': 155.0,
         'R_KNEE_vel_deg_s': v, 'L_KNEE_vel_deg_s': v * 0.8}
        for i, v in enumerate(fixture['velocity'])
    ]
    pre = AnalysisResult(
        fps=30, total_frames=len(rows), frame_rows=rows, events=events,
        knee_summary=None, frame_csv_path='', events_csv_path='',
        output_video_path='/nonexistent/test_pre.mp4',
    )
    post = deepcopy(pre)
    post.output_video_path = '/nonexistent/test_post.mp4'
    for event in post.events:
        for key in ('active_peak_knee_vel_deg_s', 'active_knee_rom_deg', 'peak_kick_height_norm'):
            if event.get(key) is not None:
                event[key] *= 0.8
    return pre, post, pd.DataFrame(rows), pd.DataFrame(rows)
