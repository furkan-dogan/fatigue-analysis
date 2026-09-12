"""The MVP always opens volleyball; hidden legacy state cannot change the sport."""
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from tests.fixtures.dashboard import session_pair


class NavigationTest(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        store_root=patch('src.adapters.analysis_store.DEFAULT_ROOT',Path(temp.name))
        store_root.start();self.addCleanup(store_root.stop)

    def test_volleyball_only_entry(self):
        app=AppTest.from_file(str(Path(__file__).parents[1]/'app.py')).run()
        self.assertFalse(list(app.exception))
        self.assertFalse(any(r.label=='Branş' for r in app.radio))
        self.assertEqual(app.title[0].value,'Video analizi')
        self.assertTrue(app.button(key='volleyball_create').disabled)
        self.assertEqual(len(app.get('file_uploader')),1)

    def test_legacy_state_cannot_select_a_hidden_sport(self):
        pre,post,a,b=session_pair()
        app=AppTest.from_file(str(Path(__file__).parents[1]/'app.py'))
        app.session_state['active_sport']='taekwondo'
        app.session_state['taekwondo_analysis']=dict(pre=pre,post=post,pre_df=a,post_df=b)
        app.run()
        self.assertFalse(list(app.exception))
        self.assertEqual(len(app.tabs),0)
        self.assertEqual(len(app.metric),0)
        self.assertEqual(len(app.session_state['taekwondo_analysis']['pre'].events),len(pre.events))
