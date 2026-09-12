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

    def test_automatic_scans_full_source_and_persists_without_manual_flags(self):
        review = {**self.current['result']['review'], 'start_frame': 5, 'end_frame': 8, 'test': 'approach'}
        current = save_revision(self.current, review, store=self.store)
        result = analyze_review(current, lambda *args: None, store=self.store,
                                runner_factory=FakeRunner, automatic=True)
        restored = load_review(result['session_id'], store=AnalysisStore(self.store.root))
        analysis = restored['result']['analysis']
        self.assertEqual(analysis['mode'], 'automatic')
        self.assertEqual(analysis['segments'][0]['end_frame'], 29)
        self.assertEqual(len(self.store.path(restored['result']['pose_path']).read_text().splitlines()),30)
        self.assertEqual(analysis['metrics'],[])
        self.assertFalse(restored['result']['review']['physical_time_confirmed'])

    def test_ambiguous_people_require_selection_and_can_be_resolved(self):
        class TwoPeople(FakeRunner):
            def process(self, frame):
                first = person()
                second = {k:(x+120,y,s) for k,(x,y,s) in first.items()}
                return [first, second]
        result = analyze_review(self.current, lambda *args: None, store=self.store,
                                runner_factory=TwoPeople, automatic=True)
        analysis = result['result']['analysis']
        self.assertEqual(len(analysis['selections_needed']),1)
        self.assertEqual(analysis['events'],[])
        resolved = analyze_review(result, lambda *args: None, store=self.store,
                                  runner_factory=TwoPeople, automatic=True, athlete_choices={'0':1})
        self.assertEqual(resolved['result']['analysis']['selections_needed'],[])
        self.assertEqual(resolved['result']['analysis']['segments'][0]['missing_frames'],0)

    def test_automatic_failure_does_not_replace_source_and_preview_hash_checked(self):
        result = analyze_review(self.current, lambda *args: None, store=self.store,
                                runner_factory=FakeRunner, automatic=True)
        if result['result'].get('preview_path'):
            self.store.path(result['result']['preview_path']).write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError,'bütünlük'):
                load_review(result['session_id'],store=self.store)
        self.assertEqual(load_review(self.current['session_id'],store=self.store)['result']['kind'],'manual_video_review')

    def test_cached_full_pose_reused_without_loading_model(self):
        from src.adapters.rtmpose_pose import MODELS, RTMPoseRunner
        class CachedFixture(FakeRunner):
            provenance = {'model': 'test-fixture',
                          'models': {name: {'sha256': digest} for name,(_,digest) in MODELS.items()}}
        first = analyze_review(self.current, lambda *args: None, store=self.store,
                               automatic=True, runner_factory=CachedFixture)
        with patch.object(RTMPoseRunner, '__init__', side_effect=AssertionError('model loaded')):
            second = analyze_review(first, lambda *args: None, store=self.store, automatic=True)
        self.assertEqual(second['result']['analysis']['segments'], first['result']['analysis']['segments'])
        row = self.store.runs(second['session_id'])[0]
        self.assertEqual(json.loads(row['provenance'])['pose_reused_from'], first['session_id'])

    def test_single_upload_shows_automatic_movement_without_review_form(self):
        from types import SimpleNamespace
        from functools import partial
        from tests.test_volleyball_discovery import DiscoveryTest
        samples = DiscoveryTest().samples(True)
        class MovementRunner(FakeRunner):
            index = 0
            def process(self, frame):
                p = samples[self.index].points
                self.index += 1
                return [p]
        source = self.store.root/'test-motion.mp4'
        writer = cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),60,(320,240))
        for _ in samples: writer.write(np.zeros((240,320,3),dtype=np.uint8))
        writer.release()
        upload = SimpleNamespace(name='motion.mp4',getvalue=source.read_bytes)
        with patch('src.adapters.analysis_store.DEFAULT_ROOT',self.store.root), \
             patch('ui.sports.volleyball.uploads.video_uploader',return_value=upload), \
             patch('ui.sports.volleyball.discovery.analyze_review',side_effect=partial(analyze_review,runner_factory=MovementRunner)):
            app=AppTest.from_file(str(Path(__file__).parents[1]/'app.py')).run()
            app.button(key='volleyball_create').click().run(timeout=20)
            self.assertFalse(list(app.exception))
            self.assertTrue(any(s.label=='Bulunan hareket' for s in app.selectbox))
            self.assertFalse(any(s.label=='Test türü' for s in app.selectbox))
            self.assertEqual(len(app.get('video')),1)
            self.assertEqual(len(app.get('file_uploader')),0)
            self.assertFalse(any(r.label=='Branş' for r in app.radio))
            self.assertTrue(any(m.value != '—' for m in app.metric))
            next(s for s in app.selectbox if s.label=='Kamera').set_value('fixed_perpendicular').run()
            next(s for s in app.selectbox if s.label=='Çekim yönü').set_value('side').run()
            next(s for s in app.selectbox if s.label=='Videonun oynatma hızı').set_value('Gerçek zaman').run()
            next(b for b in app.button if b.label=='Ölçümleri güncelle').click().run(timeout=20)
            self.assertFalse(list(app.exception))
            updated=app.session_state['volleyball_review']['result']
            self.assertEqual(updated['measurement_inputs']['segments']['0']['time_scale'],1.)
            self.assertFalse(updated['review']['physical_time_confirmed'])
            self.assertEqual(updated['analysis']['events'][0]['comparison_context']['capture']['view'],'side')
            saved=app.session_state['volleyball_review']['session_id']
        with patch('src.adapters.analysis_store.DEFAULT_ROOT',self.store.root):
            fresh=AppTest.from_file(str(Path(__file__).parents[1]/'app.py')).run()
            fresh.selectbox(key='volleyball_history').set_value(saved).run()
            fresh.button(key='volleyball_open').click().run()
            self.assertFalse(list(fresh.exception))
            self.assertTrue(any(s.label=='Bulunan hareket' for s in fresh.selectbox))
