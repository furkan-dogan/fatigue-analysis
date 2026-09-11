from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src.sports.taekwondo.service import analyze_pair


class AnalysisServiceTest(unittest.TestCase):
    def test_pair_order_and_progress(self):
        updates = []
        def analyze(source, output, frames, events, **options):
            self.assertEqual(source.read_bytes(), b'first' if source.name.startswith('pre') else b'second')
            options['progress_callback'](10, 10)
            return source.name
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / 'run'
            directory.mkdir()
            with patch('src.sports.taekwondo.service.tempfile.mkdtemp', return_value=str(directory)), patch('src.sports.taekwondo.service.run_analysis', side_effect=analyze):
                pre, post, saved = analyze_pair(b'first', b'second', lambda value, text: updates.append(value))
            self.assertEqual((pre, post), ('pre_input.mp4', 'post_input.mp4'))
            self.assertEqual(updates, [50, 100])
            self.assertEqual(saved, directory)

    def test_failed_pair_cleans_partial_files(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / 'run'
            directory.mkdir()
            with patch('src.sports.taekwondo.service.tempfile.mkdtemp', return_value=str(directory)), patch('src.sports.taekwondo.service.run_analysis', side_effect=[object(), RuntimeError('failed')]):
                with self.assertRaisesRegex(RuntimeError, 'failed'):
                    analyze_pair(b'first', b'second', lambda *args: None)
            self.assertFalse(directory.exists())
