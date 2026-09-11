import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from src.adapters.video_metadata import probe_video
from src.adapters.mediapipe_pose import MediaPipePoseRunner


class VideoMetadataTest(unittest.TestCase):
    def test_variable_timestamps_are_preserved(self):
        raw = {'frames': [{'best_effort_timestamp_time': t} for t in ('0', '0.02', '0.07')]}
        with patch('src.adapters.video_metadata.shutil.which', return_value='/ffprobe'), patch('src.adapters.video_metadata.subprocess.run', return_value=SimpleNamespace(stdout=json.dumps(raw))):
            metadata = probe_video('video.mp4')
        self.assertEqual(metadata['timestamps'], [0, 0.02, 0.07])
        self.assertTrue(metadata['timestamps_monotonic'])

    def test_unavailable_timestamps_are_not_invented(self):
        with patch('src.adapters.video_metadata.shutil.which', return_value=None):
            result = probe_video('video.mp4')
        self.assertEqual(result['timestamp_source'], 'unavailable')
        self.assertEqual(result['timestamps'], [])

    def test_raw_low_visibility_landmarks_are_kept(self):
        raw = SimpleNamespace(landmark=[SimpleNamespace(x=0.1, y=0.2, z=-0.3, visibility=0.01)])
        self.assertEqual(MediaPipePoseRunner.landmark_record(raw),
                         [{'index': 0, 'x': 0.1, 'y': 0.2, 'z': -0.3, 'visibility': 0.01}])
        self.assertEqual(MediaPipePoseRunner.landmark_record(None), [])
