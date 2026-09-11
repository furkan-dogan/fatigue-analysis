"""Check video/CSV integration without downloading or evaluating pose models."""

import csv
import json
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

    @staticmethod
    def landmark_record(raw):
        return []

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
            pose = [json.loads(line) for line in Path(result.pose_path).read_text().splitlines()]
            self.assertEqual(len(pose), 30)
            self.assertEqual(pose[0]['landmarks'], [])
            self.assertIn('source_timestamp_sec', pose[0])
            self.assertEqual(result.metadata['calculation_time_basis'], 'frame_index/nominal_fps')
            with Path(result.frame_csv_path).open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 30)
            self.assertEqual(rows[0]['R_KNEE'], '')
            capture = cv2.VideoCapture(result.output_video_path)
            try:
                self.assertTrue(capture.read()[0])
            finally:
                capture.release()

    def test_real_video_pipeline_persists_and_reopens_without_pose(self):
        from src.adapters.analysis_store import AnalysisStore
        from src.sports.taekwondo.service import analyze_pair, load_pair
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.mp4'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 30, (64, 64))
            self.assertTrue(writer.isOpened())
            for _ in range(5):
                writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
            writer.release()
            store = AnalysisStore(Path(directory) / 'data')
            with patch('src.sports.taekwondo.pipeline.MediaPipePoseRunner', NoPoseRunner):
                pair = analyze_pair(source.read_bytes(), source.read_bytes(), lambda *args: None, store=store)
            restored = load_pair(pair[2].name, store=AnalysisStore(store.root))
            self.assertEqual(restored[0], pair[0])
            self.assertEqual(len(Path(restored[0].pose_path).read_text().splitlines()), 5)
