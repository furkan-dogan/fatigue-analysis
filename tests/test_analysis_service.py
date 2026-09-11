import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.adapters.analysis_store import AnalysisStore
from src.core.records import MetricResult
from src.sports.taekwondo.pipeline import AnalysisResult
from src.sports.taekwondo.service import analyze_pair, load_pair, retry_pair


def fake_analysis(source, output, frames, events, **options):
    pose = frames.with_suffix('.pose.jsonl')
    for path, content in ((output, b'annotated'), (frames, b'time_sec\n0\n'),
                          (events, b'kick_id\n1\n'), (pose, b'{"landmarks": []}\n')):
        path.write_bytes(content)
    options['progress_callback'](10, 10)
    return AnalysisResult(30, 1, [{'time_sec': 0}],
                          [{'start_time_sec': 0, 'end_time_sec': 1, 'duration_sec': 0}],
                          None, str(frames), str(events), str(output), str(pose), {'test': True})


class AnalysisServiceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = AnalysisStore(self.temp.name)
        self.engine = patch('src.sports.taekwondo.service.run_analysis', side_effect=fake_analysis)
        self.mock = self.engine.start()
        self.addCleanup(self.engine.stop)

    def analyze(self):
        return analyze_pair(b'first', b'second', lambda *args: None, store=self.store)

    def test_pair_order_progress_and_reopen(self):
        updates = []
        pair = analyze_pair(b'first', b'second', lambda v, t: updates.append(v), store=self.store)
        self.assertEqual(updates, [50, 100])
        reopened = load_pair(pair[2].name, store=AnalysisStore(self.temp.name))
        self.assertEqual(reopened[0], pair[0])
        rows = self.store.runs(pair[2].name)
        self.assertEqual({r['status'] for r in rows}, {'completed'})
        self.assertTrue(json.loads(rows[0]['provenance'])['model_sha256'])
        with self.store.connection() as db:
            metrics = [json.loads(r[0]) for r in db.execute('SELECT payload FROM metrics')]
            self.assertEqual(db.execute('SELECT count(*) FROM comparisons').fetchone()[0], 1)
        self.assertTrue(any(m['value'] == 0 for m in metrics))
        self.assertTrue(any(m['value'] is None and m['quality'] == 'missing' for m in metrics))

    def test_failed_second_run_preserves_sources_and_retry_is_revision(self):
        calls = 0
        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('failed')
            return fake_analysis(*args, **kwargs)
        self.mock.side_effect = fail_second
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            self.analyze()
        session = self.store.sessions('taekwondo')[0]
        statuses = {r['status'] for r in self.store.runs(session['id'])}
        self.assertEqual(statuses, {'completed', 'failed'})
        with self.assertRaises(ValueError):
            load_pair(session['id'], store=self.store)
        for row in self.store.runs(session['id']):
            self.assertIn(self.store.source_bytes(json.loads(row['video'])), (b'first', b'second'))
        self.mock.side_effect = fake_analysis
        pair = retry_pair(session['id'], lambda *args: None, store=self.store)
        self.assertNotEqual(pair[2].name, session['id'])
        self.assertEqual(self.store.sessions('taekwondo')[0]['revision_of'], session['id'])
        self.assertEqual({r['status'] for r in self.store.runs(session['id'])}, statuses)

    def test_corrupt_or_missing_artifact_is_not_silently_loaded(self):
        pair = self.analyze()
        Path(pair[0].output_video_path).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'bozulmuş'):
            load_pair(pair[2].name, store=self.store)
        Path(pair[0].output_video_path).unlink()
        with self.assertRaises(OSError):
            load_pair(pair[2].name, store=self.store)

    def test_source_integrity_and_result_integrity(self):
        pair = self.analyze()
        row = self.store.runs(pair[2].name)[0]
        self.store.path(row['result_path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'bütünlük'):
            self.store.load_result(row)
        video = json.loads(row['video'])
        self.store.path(video['path']).write_bytes(b'changed')
        with self.assertRaises(ValueError):
            self.store.source_bytes(video)

    def test_recovery_is_explicit_and_completed_runs_are_immutable(self):
        pair = self.analyze()
        session = self.store.create_session('taekwondo', 'interrupted')
        video = self.store.add_video(session, 'pre', 'source.mp4', b'video')
        run = self.store.start_run(video, {'test': True})
        other = AnalysisStore(self.temp.name)
        self.assertEqual(other.runs(session.id)[0]['status'], 'running')
        self.assertEqual(other.recover_interrupted(), 1)
        self.assertEqual(other.runs(session.id)[0]['status'], 'interrupted')
        self.assertEqual(len(load_pair(pair[2].name, store=other)[0].events), 1)
        with self.assertRaises(ValueError):
            other.complete_run(run, {}, [], [])

    def test_invalid_records_and_path_escape(self):
        with self.assertRaises(ValueError):
            MetricResult('id', 'key', float('nan'), 's', 'method', 'unvalidated')
        with self.assertRaises(ValueError):
            MetricResult('id', 'key', None, 's', 'method', 'missing')
        with self.assertRaises(ValueError):
            self.store.path('../outside')
        session = self.store.create_session('basketball', 'other')
        with self.assertRaises(ValueError):
            self.store.create_session('taekwondo', 'wrong', session.id)
        with self.assertRaises(ValueError):
            load_pair(session.id, store=self.store)

    def test_conflicting_unit_rolls_back_run_completion(self):
        from src.core.records import MovementEvent
        self.analyze()
        session = self.store.create_session('taekwondo', 'unit conflict')
        video = self.store.add_video(session, 'pre', 'video.mp4', b'video')
        run = self.store.start_run(video, {})
        event = MovementEvent('unit-test-event', run.id, video.id, 'kick', 0, 1)
        metric = MetricResult(event.id, 'taekwondo.duration_sec', 1, 'ms', 'test', 'unvalidated')
        with self.assertRaisesRegex(ValueError, 'birim'):
            self.store.complete_run(run, {}, [event], [metric])
        with self.store.connection() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM events WHERE run_id=?', (run.id,)).fetchone()[0], 0)
        self.assertEqual(self.store.runs(session.id)[0]['status'], 'running')

    def test_future_schema_is_rejected_without_overwrite(self):
        with self.store.connection() as db:
            db.execute('PRAGMA user_version=99')
        with self.assertRaisesRegex(ValueError, 'şeması'):
            AnalysisStore(self.temp.name)
        with self.store.connection() as db:
            self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0], 99)
