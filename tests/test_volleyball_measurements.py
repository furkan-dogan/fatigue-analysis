"""Known analytical inputs verify math and rejection gates, not real-world accuracy."""
from copy import deepcopy
from dataclasses import replace
import math
import unittest
import numpy as np
from src.core.pose import PoseFrame
from src.adapters.rtmpose_pose import map_person
from src.sports.volleyball.review import PROTOCOL_DEFAULTS
from src.sports.volleyball.measurements import analyze
from src.sports.volleyball.signals import AthleteTracker


def person(x=100., lift=0.):
    points = {}
    for side, dx in [('left', -10), ('right', 10)]:
        for name, y in [('shoulder', 50), ('hip', 100), ('knee', 150), ('ankle', 190),
                        ('big_toe', 200), ('small_toe', 200), ('heel', 200)]:
            points[f'{side}_{name}'] = (x+dx, y-lift, 0.9)
    return points


def case(test='cmj'):
    samples = [PoseFrame(i, i/100, person(lift=40*math.sin(math.pi*(i-29.5)/50) if 30 <= i < 80 else 0)) for i in range(101)]
    review = dict(test=test, start_frame=0, end_frame=100, camera='fixed', view='side', feet_visible=True,
                  physical_time_confirmed=True, repetitions=[dict(start_frame=0, end_frame=100, takeoff_frame=30, landing_frame=80)],
                  calibration=None, **{k: True if isinstance(v, bool) else v for k,v in PROTOCOL_DEFAULTS.items()})
    metadata = dict(timestamps_monotonic=True, timestamp_count_matches_frames=True)
    return samples, review, metadata


def values(result):
    return {m['key']:m for m in result['metrics'] + [v for e in result['events'] for v in e['metrics']]}


class VolleyballMeasurementsTest(unittest.TestCase):
    def test_cmj_known_flight_time_and_frame_bounds(self):
        samples, review, metadata = case()
        output = analyze(samples, review, metadata)
        metrics = values(output)
        self.assertAlmostEqual(metrics['jump_height_cm']['value'], 9.80665*.5**2/8*100)
        self.assertLess(metrics['jump_lower_cm']['value'], metrics['jump_height_cm']['value'])
        self.assertGreater(metrics['jump_upper_cm']['value'], metrics['jump_height_cm']['value'])
        self.assertEqual(len(output['candidates']), 1)
        self.assertEqual(metrics['jump_height_cm']['quality'], 'unvalidated')

    def test_slow_motion_time_scale_changes_height_by_square(self):
        samples, review, metadata = case()
        original = values(analyze(samples, review, metadata))['jump_height_cm']['value']
        review['time_scale'] = .5
        adjusted = values(analyze(samples, review, metadata))['jump_height_cm']['value']
        self.assertAlmostEqual(adjusted, original/4)

    def test_unconfirmed_contacts_never_publish_automatic_height(self):
        samples, review, metadata = case()
        review['contacts_confirmed'] = False
        output = analyze(samples, review, metadata)
        self.assertTrue(output['candidates'])
        self.assertIsNone(values(output)['jump_height_cm']['value'])

    def test_missing_points_and_different_landing_posture_reject_height(self):
        for missing in (True, False):
            samples, review, metadata = case()
            points = dict(samples[80].points)
            if missing:
                points.pop('right_heel')
            else:
                points['right_knee'] = (150, 135, .9)
            samples[80] = replace(samples[80], points=points)
            self.assertIsNone(values(analyze(samples, review, metadata))['jump_height_cm']['value'])

    def test_source_gates_and_scene_cut(self):
        for mutation in ('time', 'camera', 'cut', 'gap', 'approach'):
            samples, review, metadata = case()
            if mutation == 'time': review['physical_time_confirmed'] = False
            if mutation == 'camera': review['camera'] = 'moving'
            if mutation == 'cut': samples[50] = replace(samples[50], issue='cut')
            if mutation == 'gap': samples[50] = replace(samples[50], time_seconds=None)
            if mutation == 'approach': review['test'] = 'approach'
            result = analyze(samples, review, metadata)
            self.assertTrue(result['warnings'])
            self.assertFalse(result['events'])
            self.assertTrue(all(v['value'] is None for v in result['metrics']))

    def test_front_view_drift_and_separate_contact_times(self):
        samples, review, metadata = case('asymmetry')
        samples = [PoseFrame(i, i/100, person(x=100+i*.2)) for i in range(101)]
        review['view'] = 'front'
        review['repetitions'][0].update(left_landing_frame=80, right_landing_frame=82)
        result = values(analyze(samples, review, metadata))
        self.assertAlmostEqual(result['lateral_drift_torso_pct']['value'], 40.)
        self.assertAlmostEqual(result['trunk_lean_deg']['value'], 0.)
        self.assertAlmostEqual(result['pelvis_tilt_deg']['value'], 0.)
        self.assertAlmostEqual(result['landing_difference_ms']['value'], 20.)

    def test_sprint_linear_speed_and_no_extrapolation_at_edges(self):
        samples, review, metadata = case('sprint')
        samples = [PoseFrame(i, i/100, person(x=50+i)) for i in range(101)]
        review['repetitions'] = []
        review['calibration'] = dict(x1=0, y1=100, x2=200, y2=100, distance_m=20.)
        result = analyze(samples, review, metadata)
        self.assertAlmostEqual(values(result)['peak_speed_m_s']['value'], 10.)
        self.assertIsNone(result['speed_series'][0]['speed_m_s'])
        self.assertIsNone(result['speed_series'][-1]['speed_m_s'])
        review['calibration']['y1'] = review['calibration']['y2'] = 200
        self.assertIsNone(values(analyze(samples, review, metadata))['peak_speed_m_s']['value'])

    def test_variable_frame_intervals_preserve_linear_speed(self):
        samples, review, metadata = case('sprint')
        t = np.cumsum([0]+[.008, .012]*50)
        samples = [PoseFrame(i,float(time),person(x=50+100*time)) for i,time in enumerate(t)]
        review['repetitions'] = []
        review['calibration'] = dict(x1=0, y1=100, x2=200, y2=100, distance_m=20.)
        self.assertAlmostEqual(values(analyze(samples, review, metadata))['peak_speed_m_s']['value'], 10.)

    def test_ambiguous_tracking_and_explicit_point_mapping(self):
        tracker = AthleteTracker()
        self.assertIsNotNone(tracker.select([person(50), person(150)])[1])
        tracker = AthleteTracker(dict(x1=20,y1=50,x2=80,y2=150))
        selected, issue = tracker.select([person(50), person(150)])
        self.assertIsNone(issue)
        self.assertEqual(selected['left_hip'][0], 40)
        coords = [[10+i, 20+i] for i in range(23)]
        mapped = map_person(coords, [.9]*23, 100,100)
        self.assertEqual(mapped['left_heel'][:2], (29.,39.))
        self.assertEqual(mapped['right_big_toe'][:2], (30.,40.))
        self.assertNotIn('right_heel', map_person(coords[:17], [.9]*17, 100,100))
