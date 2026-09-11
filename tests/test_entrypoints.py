"""Exercise user entry points and resources after moving their modules."""

from pathlib import Path
import subprocess
import sys
import unittest

from streamlit.testing.v1 import AppTest

from ui.paths import SAMPLE_DATA_DIR


ROOT = Path(__file__).resolve().parents[1]


class EntryPointTest(unittest.TestCase):
    def test_dashboard_opens(self):
        app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
        self.assertFalse(list(app.exception))
        self.assertEqual(app.title[0].value, 'Video Analizi — Yorgunluk Değerlendirmesi')

    def test_cli_help(self):
        for command in (['main.py'], ['-m', 'cli.analyze'], ['cli/compare.py'], ['-m', 'cli.compare']):
            with self.subTest(command=command):
                result = subprocess.run(
                    [sys.executable, *command, '--help'], cwd=ROOT,
                    capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)

    def test_sample_files_resolve_from_nested_pages(self):
        for name in ('emg_sample.csv', 'nirs_moxy_sample.csv'):
            self.assertTrue((SAMPLE_DATA_DIR / name).is_file())
