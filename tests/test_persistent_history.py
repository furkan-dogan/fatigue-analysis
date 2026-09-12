"""Reopen a saved analysis in a completely new Streamlit session."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from src.adapters.analysis_store import AnalysisStore
from src.sports.taekwondo.service import analyze_pair
from tests.test_analysis_service import fake_analysis


class PersistentHistoryTest(unittest.TestCase):
    def test_fresh_ui_opens_saved_pair_and_keeps_sport_isolation(self):
        with tempfile.TemporaryDirectory() as directory, patch('src.adapters.analysis_store.DEFAULT_ROOT', Path(directory)):
            store = AnalysisStore()
            with patch('src.sports.taekwondo.service.run_analysis', side_effect=fake_analysis):
                pair = analyze_pair(b'first', b'second', lambda *args: None, store=store, label='Kayıt testi')
            app = AppTest.from_string('from ui.sports.taekwondo.page import render\nrender()').run()
            self.assertEqual(len(app.tabs), 0)
            self.assertEqual(app.selectbox(key='taekwondo_history_selected').value, pair[2].name)
            app.button(key='taekwondo_history_open').click().run()
            self.assertFalse(list(app.exception))
            self.assertEqual(len(app.tabs), 6)
            app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py')).run()
            self.assertEqual(len(app.tabs), 0)
            self.assertEqual(len(app.dataframe), 0)
