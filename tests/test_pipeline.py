"""Check video/CSV integration without downloading or evaluating pose models."""

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.sports.taekwondo.pipeline import run_analysis


class NoPoseRunner:
    def process_frame(self, frame):
        return None, None

    def get_confidence(self, raw):
        return None

    def close(self):
        pass


class PipelineIntegrationTest(unittest.TestCase):
    def test_missing_pose_still_exports_frames_and_playable_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.mp4'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 30, (64, 64))
            self.assertTrue(writer.isOpened(), 'Test environment needs an mp4v encoder')
            try:
                for _ in range(30):
                    writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
            finally:
                writer.release()
            with patch('src.sports.taekwondo.pipeline.MediaPipePoseRunner', NoPoseRunner):
                result = run_analysis(source, root / 'annotated.mp4', root / 'frames.csv', root / 'events.csv')
            self.assertEqual(result.total_frames, 30)
            self.assertEqual(result.events, [])
            with Path(result.frame_csv_path).open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 30)
            self.assertEqual(rows[0]['R_KNEE'], '')
            capture = cv2.VideoCapture(result.output_video_path)
            try:
                self.assertTrue(capture.read()[0])
            finally:
                capture.release()
