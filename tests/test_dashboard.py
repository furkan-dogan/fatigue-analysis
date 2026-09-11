"""Open every analysis tab with synthetic results and no uploaded athlete video."""

from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from tests.fixtures.dashboard import session_pair


class DashboardTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        store_root = patch('src.adapters.analysis_store.DEFAULT_ROOT', Path(temp.name))
        store_root.start()
        self.addCleanup(store_root.stop)

    def test_all_tabs_with_and_without_events(self):
        for empty in (False, True):
            with self.subTest(empty=empty):
                pre, post, pre_df, post_df = session_pair(empty=empty)
                app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py'))
                app.session_state['active_sport'] = 'taekwondo'
                app.session_state['taekwondo_analysis'] = dict(pre=pre, post=post, pre_df=pre_df, post_df=post_df)
                app.run(timeout=30)
                self.assertFalse(list(app.exception))
                self.assertEqual(len(app.tabs), 6)

    def test_event_selector_reaches_real_detected_events(self):
        pre, post, a, b = session_pair()
        app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py'))
        app.session_state['active_sport'] = 'taekwondo'
        app.session_state['taekwondo_analysis'] = dict(pre=pre, post=post, pre_df=a, post_df=b)
        app.run(timeout=30)
        self.assertFalse(list(app.exception))
        self.assertEqual(len(app.selectbox), 2)
        app.selectbox[0].set_value('1').run()
        self.assertFalse(list(app.exception))
        self.assertEqual(app.selectbox[0].value, '1')
