"""Report regression covers populated data, including legacy unvalidated rules."""

import hashlib
import json
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

from tests.fixtures.dashboard import render_report_case


def report_snapshot(app):
    import datetime
    today = datetime.date.today().strftime('%d.%m.%Y')
    result = {}
    for kind in ('markdown', 'info', 'warning', 'success', 'caption'):
        values = [item.value.replace(today, '<DATE>') for item in app.get(kind)]
        result[kind] = hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest()
    result['tables'] = [
        hashlib.sha256(frame.value.to_json(orient='split', force_ascii=False).encode()).hexdigest()
        for frame in app.dataframe
    ]
    return result


class ReportRenderingTest(unittest.TestCase):
    def test_populated_reports_preserve_legacy_content(self):
        expected = json.loads((Path(__file__).parent / 'fixtures/report_rendering.json').read_text())
        for sensors in (False, True):
            with self.subTest(sensors=sensors):
                app = AppTest.from_function(render_report_case, args=(sensors,)).run(timeout=30)
                self.assertFalse(list(app.exception))
                self.assertEqual(report_snapshot(app), expected[str(sensors)])
