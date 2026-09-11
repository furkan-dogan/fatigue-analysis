from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from streamlit.testing.v1 import AppTest
from src.adapters.analysis_store import AnalysisStore
from src.adapters.video_review import read_frame
from src.sports.volleyball.service import create_review, load_review, save_revision
from src.sports.volleyball.review import validate_review, quality_messages


class VolleyballReviewTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.store = AnalysisStore(self.root / 'data')
        patcher = patch('src.adapters.analysis_store.DEFAULT_ROOT', self.store.root)
        patcher.start(); self.addCleanup(patcher.stop)
        video = self.root / 'video.mp4'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (64, 64))
        self.assertTrue(writer.isOpened())
        for index in range(8):
            writer.write(np.full((64, 64, 3), index * 25, dtype=np.uint8))
        writer.release()
        self.content = video.read_bytes()
        self.current = create_review(self.content, 'jump.mp4', store=self.store)

    def test_revision_and_reopen_preserve_source_and_manual_marks(self):
        review = deepcopy(self.current['result']['review'])
        review.update(athlete='Sporcu 1', repetitions=[dict(start_frame=0, takeoff_frame=2, landing_frame=5, end_frame=7)],
                      calibration=dict(frame=0, x1=1, y1=2, x2=20, y2=2, distance_m=1.0, plane='zemin'))
        revised = save_revision(self.current, review, store=self.store)
        restored = load_review(revised['session_id'], store=AnalysisStore(self.store.root))
        self.assertEqual(restored['result']['review'], review)
        self.assertEqual(restored['result']['metrics'], [])
        self.assertEqual(Path(restored['source_path']).read_bytes(), self.content)
        self.assertEqual(load_review(self.current['session_id'], store=self.store)['result']['review']['repetitions'], [])
        self.assertEqual(self.store.sessions('volleyball')[0]['revision_of'], self.current['session_id'])
        with self.store.connection() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM metrics').fetchone()[0], 0)

    def test_invalid_intervals_and_calibration_do_not_create_revisions(self):
        original = self.current['result']['review']
        bad = [dict(start_frame=7, end_frame=0),
               dict(repetitions=[dict(start_frame=0, end_frame=7, takeoff_frame=6, landing_frame=2)]),
               dict(calibration=dict(frame=0, x1=1,y1=1,x2=1,y2=1,distance_m=1,plane='zemin')),
               dict(athlete_box=dict(frame=0, x1=0,y1=0,x2=90,y2=10))]
        for change in bad:
            with self.subTest(change=change), self.assertRaises(ValueError):
                save_revision(self.current, {**original, **change}, store=self.store)
        self.assertEqual(len(self.store.sessions('volleyball')), 1)

    def test_source_frames_and_invalid_video(self):
        first = read_frame(self.current['source_path'], 0)
        last = read_frame(self.current['source_path'], 7)
        self.assertLess(first.mean(), last.mean())
        with self.assertRaises(ValueError):
            read_frame(self.current['source_path'], 8)
        with self.assertRaises(ValueError):
            create_review(b'broken', 'broken.mp4', store=self.store)
        broken = self.store.sessions('volleyball')[0]
        self.assertEqual(self.store.runs(broken['id'])[0]['status'], 'failed')

    def test_missing_timing_and_sprint_have_no_fake_metrics(self):
        review = {**self.current['result']['review'], 'test': 'sprint'}
        metadata = {**self.current['result']['metadata'], 'timestamps_monotonic': False}
        validate_review(review, metadata)
        messages = ' '.join(quality_messages(review, metadata))
        self.assertIn('kalibrasyonu girilmedi', messages)
        self.assertIn('zamanları eksik', messages)

    def test_fresh_ui_review_and_revision_save(self):
        app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py')).run()
        app.button(key='volleyball_open').click().run()
        self.assertFalse(list(app.exception))
        self.assertEqual(len(app.get('file_uploader')), 1)
        self.assertGreater(len(app.get('imgs')), 0)
        app.text_input[0].set_value('Sporcu UI').run()
        save = next(button for button in app.button if 'revizyon' in button.label)
        save.click().run()
        self.assertFalse(list(app.exception))
        self.assertEqual(len(self.store.sessions('volleyball')), 2)
        self.assertEqual(app.session_state['volleyball_review']['result']['review']['athlete'], 'Sporcu UI')
        app.radio[0].set_value('taekwondo').run()
        self.assertNotIn('Sporcu UI', ' '.join(t.value for t in app.caption))
        app.radio[0].set_value('volleyball').run()
        self.assertFalse(list(app.exception))
        self.assertEqual(app.session_state['volleyball_review']['result']['review']['athlete'], 'Sporcu UI')
