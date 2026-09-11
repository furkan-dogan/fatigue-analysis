"""Open every analysis tab with synthetic results and no uploaded athlete video."""

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

from tests.fixtures.dashboard import session_pair


class DashboardTest(unittest.TestCase):
    def test_all_tabs_with_and_without_events_or_sensors(self):
        for sensors, empty in ((True, False), (False, False), (False, True)):
            with self.subTest(sensors=sensors, empty=empty):
                pre, post, pre_df, post_df = session_pair(sensors=sensors, empty=empty)
                app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py'))
                for key, value in {'dv_pre_result': pre, 'dv_post_result': post,
                                   'dv_pre_df': pre_df, 'dv_post_df': post_df, 'dv_tmp': None}.items():
                    app.session_state[key] = value
                app.run(timeout=30)
                self.assertFalse(list(app.exception))
                self.assertEqual(len(app.tabs), 12)
