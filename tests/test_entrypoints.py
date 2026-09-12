"""Exercise user entry points and resources after moving their modules."""

from pathlib import Path
import subprocess
import sys
import unittest
import tempfile
from unittest.mock import patch

from streamlit.testing.v1 import AppTest




ROOT = Path(__file__).resolve().parents[1]


class EntryPointTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        store_root = patch('src.adapters.analysis_store.DEFAULT_ROOT', Path(temp.name))
        store_root.start()
        self.addCleanup(store_root.stop)

    def test_dashboard_opens(self):
        app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
        self.assertFalse(list(app.exception))
        self.assertEqual(app.title[0].value, 'Video analizi')

    def test_cli_help(self):
        for command in (['main.py'], ['-m', 'cli.analyze'], ['cli/compare.py'], ['-m', 'cli.compare']):
            with self.subTest(command=command):
                result = subprocess.run(
                    [sys.executable, *command, '--help'], cwd=ROOT,
                    capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)
