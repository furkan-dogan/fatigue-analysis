"""Exercise real sport switching and persistence of completed analysis results."""
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from tests.fixtures.dashboard import session_pair


class NavigationTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        store_root = patch('src.adapters.analysis_store.DEFAULT_ROOT', Path(temp.name))
        store_root.start()
        self.addCleanup(store_root.stop)

    def test_default_and_unavailable_sports(self):
        app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py')).run()
        self.assertEqual(app.radio[0].value, 'volleyball')
        self.assertFalse(list(app.exception))
        self.assertTrue(app.button(key='volleyball_create').disabled)
        self.assertEqual(len(app.get('file_uploader')), 1)
        app.radio[0].set_value('basketball').run()
        self.assertEqual(app.title[0].value, 'Basketbol — Video Analizi')
        self.assertEqual(len(app.button), 0)
        self.assertEqual(len(app.get('file_uploader')), 0)
        self.assertFalse(list(app.exception))

    def test_results_survive_switches_without_leaking(self):
        pre, post, a, b = session_pair()
        app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py'))
        app.session_state['taekwondo_analysis'] = dict(pre=pre, post=post, pre_df=a, post_df=b)
        app.run()
        for sport in ('taekwondo', 'volleyball', 'basketball', 'taekwondo'):
            app.radio[0].set_value(sport).run()
            self.assertFalse(list(app.exception))
            self.assertEqual(len(app.tabs), 6 if sport == 'taekwondo' else 0)
            if sport != 'taekwondo':
                self.assertEqual(len(app.metric), 0)
                self.assertEqual(len(app.dataframe), 0)
        self.assertEqual(len(app.session_state['taekwondo_analysis']['pre'].events), len(pre.events))
