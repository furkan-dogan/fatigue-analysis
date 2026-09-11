"""Persistence and UI behavior of the experimental measurement path."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from streamlit.testing.v1 import AppTest
from src.adapters.analysis_store import AnalysisStore
from src.sports.volleyball.service import create_review, save_revision, analyze_review, load_review
from tests.test_volleyball_measurements import person


class FakeRunner:
    provenance = {'model': 'test-fixture', 'validation': 'test-only'}
    def process(self, frame):
        return [person()]
    def close(self):
        pass


class VolleyballAnalysisFlowTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.store = AnalysisStore(Path(temp.name)/'data')
        source = Path(temp.name)/'source.mp4'
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 100, (320, 240))
        self.assertTrue(writer.isOpened())
        for _ in range(30): writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
        writer.release()
        self.current = create_review(source.read_bytes(), source.name, store=self.store)

    def test_analysis_and_review_revisions_never_reuse_old_results(self):
        result = analyze_review(self.current, lambda *args: None, store=self.store, runner_factory=FakeRunner)
        restored = load_review(result['session_id'], store=AnalysisStore(self.store.root))
        self.assertEqual(restored['result']['kind'], 'volleyball_analysis')
        self.assertTrue(restored['result']['analysis']['warnings'])
        self.assertTrue(all(m['value'] is None for m in restored['result']['analysis']['metrics']))
        records = self.store.path(restored['result']['pose_path']).read_text().splitlines()
        self.assertEqual(len(records), 30)
        revised = save_revision(restored, restored['result']['review'], store=self.store)
        self.assertNotIn('analysis', revised['result'])
        self.assertNotIn('pose_path', revised['result'])
        self.assertEqual(load_review(result['session_id'], store=self.store)['result']['kind'], 'volleyball_analysis')

    def test_failure_records_failed_status_and_keeps_input(self):
        def broken(): raise RuntimeError('model failed')
        with self.assertRaisesRegex(RuntimeError, 'model failed'):
            analyze_review(self.current, lambda *args: None, store=self.store, runner_factory=broken)
        session = self.store.sessions('volleyball')[0]
        row = self.store.runs(session['id'])[0]
        self.assertEqual(row['status'], 'failed')
        self.assertTrue(self.store.source_bytes(json.loads(row['video'])))

    def test_saved_rejected_analysis_opens_in_fresh_ui(self):
        result = analyze_review(self.current, lambda *args: None, store=self.store, runner_factory=FakeRunner)
        with patch('src.adapters.analysis_store.DEFAULT_ROOT', self.store.root):
            app = AppTest.from_file(str(Path(__file__).parents[1]/'app.py')).run()
            app.selectbox(key='volleyball_history').set_value(result['session_id']).run()
            app.button(key='volleyball_open').click().run()
            self.assertFalse(list(app.exception))
            self.assertTrue(any('zaman' in warning.value for warning in app.warning))
            self.assertTrue(all(m.value == '—' for m in app.metric))

    def test_valid_jump_metrics_persist_with_event_and_model_source(self):
        import math
        from src.sports.volleyball.review import PROTOCOL_DEFAULTS
        class JumpRunner(FakeRunner):
            index = 0
            def process(self, frame):
                i = self.index
                self.index += 1
                lift = 35*math.sin(math.pi*(i-4.5)/20) if 5 <= i < 25 else 0
                return [person(lift=lift)]
        review = {**self.current['result']['review'],
                  **{k: True if isinstance(v, bool) else v for k,v in PROTOCOL_DEFAULTS.items()},
                  'camera': 'fixed', 'view': 'side', 'physical_time_confirmed': True, 'feet_visible': True,
                  'repetitions': [dict(start_frame=0,end_frame=29,takeoff_frame=5,landing_frame=25)]}
        current = save_revision(self.current, review, store=self.store)
        result = analyze_review(current, lambda *args: None, store=self.store, runner_factory=JumpRunner)
        metrics = result['result']['analysis']['events'][0]['metrics']
        self.assertAlmostEqual(metrics[0]['value'],9.80665*.2**2/8*100)
        with self.store.connection() as db:
            row = db.execute('SELECT payload FROM metrics WHERE key=?', ('volleyball.jump_height_cm',)).fetchone()
            self.assertAlmostEqual(json.loads(row['payload'])['value'],metrics[0]['value'])
            count = db.execute('SELECT count(*) FROM events').fetchone()[0]
        self.assertEqual(count,1)
        restored = load_review(result['session_id'],store=self.store)
        self.assertEqual(restored['result']['model']['model'],'test-fixture')
        with patch('src.adapters.analysis_store.DEFAULT_ROOT', self.store.root):
            app = AppTest.from_file(str(Path(__file__).parents[1]/'app.py')).run()
            app.selectbox(key='volleyball_history').set_value(result['session_id']).run()
            app.button(key='volleyball_open').click().run()
            self.assertFalse(list(app.exception))
            self.assertTrue(any(m.value != '—' for m in app.metric))
