"""Repeatable populated dashboard inputs without real athlete video."""

from copy import deepcopy
import json
from pathlib import Path

import pandas as pd

from src.sports.taekwondo.pipeline import AnalysisResult


def session_pair(*, sensors=True, empty=False):
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
        synthetic_emg_rows=fixture['synthetic_emg'] if sensors else [],
        synthetic_nirs_rows=fixture['synthetic_nirs'] if sensors else [],
    )
    post = deepcopy(pre)
    post.output_video_path = '/nonexistent/test_post.mp4'
    for event in post.events:
        for key in ('active_peak_knee_vel_deg_s', 'active_knee_rom_deg', 'peak_kick_height_norm'):
            if event.get(key) is not None:
                event[key] *= 0.8
    return pre, post, pd.DataFrame(rows), pd.DataFrame(rows)


def render_report_case(sensors=True):
    from tests.fixtures.dashboard import session_pair
    from src.sports.taekwondo.fatigue import compute_fatigue
    from src.sports.taekwondo.sensor_summary import sensor_stats
    from ui.sports.taekwondo.report import render_athlete_report

    pre, post, _, _ = session_pair(sensors=sensors)
    ps = sensor_stats(pre.synthetic_emg_rows, pre.synthetic_nirs_rows, 30) if sensors else {}
    pos = sensor_stats(post.synthetic_emg_rows, post.synthetic_nirs_rows, 30) if sensors else {}
    fi = compute_fatigue(pre.events, post.events)['fatigue_index']
    render_athlete_report(pre.events, post.events, pre, post, fi, ps, pos,
                          3, 15, 2, 10, sensors)
